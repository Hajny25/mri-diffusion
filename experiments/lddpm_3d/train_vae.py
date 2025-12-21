"""
Training script for 3D VAE on BRATS dataset.
"""

import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
import mlflow

from vae import get_vae_small, get_vae_base, vae_loss
from dataset import BraTS3DDataset


BASE_DIR = Path(__file__).resolve().parents[2]
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()

DEBUG = os.getenv("DEBUG", "0") == "1" or False


@dataclass
class VAETrainingConfig:
    # Data
    patch_size: tuple = (144, 144, 192)
    
    # Model
    model_size: str = "base"  # tiny, small, base, large
    
    # Training
    batch_size: int = 1 if not DEBUG else 1
    num_epochs: int = 100 if not DEBUG else 10
    learning_rate: float = 1e-4  # Reduced from 2e-4 for stability
    kl_weight: float = 1.0  # Final KL weight after warmup
    kl_warmup_epochs: int = 20  # Gradually increase KL weight over first 5 epochs
    
    # Optimization
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    
    # Validation
    val_split: float = 0.1
    val_every_n_epochs: int = 1
    
    # Checkpointing
    save_every_n_epochs: int = 10
    
    # Logging
    log_every_n_steps: int = 10 if not DEBUG else 1
    
    # System
    num_workers: int = 8 if not DEBUG else 0
    seed: int = 42


config = VAETrainingConfig()


def get_kl_weight(epoch, max_epochs, final_weight, warmup_epochs):
    """Gradually increase KL weight during training (from 0 to final_weight)."""
    if epoch < warmup_epochs:
        # Linear warmup from 0 to final_weight
        return final_weight * ((epoch + 1) / warmup_epochs)
    return final_weight


def train_epoch(model, dataloader, optimizer, accelerator, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    # Get current KL weight
    kl_weight = get_kl_weight(epoch, config.num_epochs, config.kl_weight, config.kl_warmup_epochs)
    
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            x = batch  # (B, C, D, H, W)
            
            # Forward pass
            reconstruction, mean, logvar = model(x, sample_posterior=True)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, x, mean, logvar, kl_weight=kl_weight
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
            # Log periodically
            if step % config.log_every_n_steps == 0 and accelerator.is_main_process:
                accelerator.print(
                    f"Epoch {epoch+1}/{config.num_epochs} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | "
                    f"KL: {kl_loss.item():.4f} | "
                    f"KL weight: {kl_weight:.2e}"
                )
    
    # Average losses
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return avg_loss, avg_recon_loss, avg_kl_loss, kl_weight


@torch.no_grad()
def validate(model, dataloader, accelerator, config, epoch):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    kl_weight = get_kl_weight(epoch, config.num_epochs, config.kl_weight, config.kl_warmup_epochs)
    
    for batch in dataloader:
        x = batch
        
        # Forward pass (use mean, no sampling)
        reconstruction, mean, logvar = model(x, sample_posterior=False)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(
            reconstruction, x, mean, logvar, kl_weight=kl_weight
        )
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
    
    # Gather losses from all processes
    total_loss_tensor = torch.tensor([total_loss], device=accelerator.device)
    total_recon_tensor = torch.tensor([total_recon_loss], device=accelerator.device)
    total_kl_tensor = torch.tensor([total_kl_loss], device=accelerator.device)
    num_batches_tensor = torch.tensor([num_batches], device=accelerator.device)
    
    total_loss_gathered = accelerator.gather_for_metrics(total_loss_tensor).sum().item()
    total_recon_gathered = accelerator.gather_for_metrics(total_recon_tensor).sum().item()
    total_kl_gathered = accelerator.gather_for_metrics(total_kl_tensor).sum().item()
    num_batches_gathered = accelerator.gather_for_metrics(num_batches_tensor).sum().item()
    
    # Average
    avg_loss = total_loss_gathered / num_batches_gathered
    avg_recon_loss = total_recon_gathered / num_batches_gathered
    avg_kl_loss = total_kl_gathered / num_batches_gathered
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def main():
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    accelerator.print("="*70)
    accelerator.print("3D VAE Training for BRATS")
    accelerator.print("="*70)
    accelerator.print(f"Patch size: {config.patch_size}")
    accelerator.print(f"Model size: {config.model_size}")
    accelerator.print(f"Batch size: {config.batch_size}")
    accelerator.print(f"Epochs: {config.num_epochs}")
    accelerator.print(f"KL weight: {config.kl_weight}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print("="*70)
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Create dataset
    accelerator.print(f"Loading dataset from {BRATS_ROOT}...")
    full_dataset = BraTS3DDataset(
        root_dir=BRATS_ROOT,
        patch_size=config.patch_size,
        modalities=("flair",),
        max_cases=10 if DEBUG else None,
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    accelerator.print(f"Train samples: {len(train_dataset)}")
    accelerator.print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    
    # Create model
    accelerator.print("Creating model...")
    if config.model_size == "tiny":
        from vae import get_vae_tiny
        model = get_vae_tiny()
    elif config.model_size == "small":
        model = get_vae_small()
    elif config.model_size == "base":
        model = get_vae_base()
    elif config.model_size == "large":
        from vae import get_vae_large
        model = get_vae_large()
    else:
        raise ValueError(f"Unknown model size: {config.model_size}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # MLflow setup
    if accelerator.is_main_process:
        mlflow.set_experiment("vae_3d")
        mlflow.start_run()
        run_id = mlflow.active_run().info.run_id
        accelerator.print(f"MLflow run ID: {run_id}")
        
        # Log config
        mlflow.log_params({
            "patch_size": str(config.patch_size),
            "model_size": config.model_size,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "kl_weight": config.kl_weight,
            "kl_warmup_epochs": config.kl_warmup_epochs,
            "num_params": num_params,
        })
        
        output_dir = f"output/vae_{run_id}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        run_id = None
        output_dir = None
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = -1
    
    for epoch in range(config.num_epochs):
        accelerator.print(f"\n{'='*70}")
        accelerator.print(f"Epoch {epoch+1}/{config.num_epochs}")
        accelerator.print(f"{'='*70}")
        
        # Train
        train_loss, train_recon, train_kl, kl_weight = train_epoch(
            model, train_loader, optimizer, accelerator, config, epoch
        )
        
        accelerator.print(
            f"Train | Loss: {train_loss:.4f} | "
            f"Recon: {train_recon:.4f} | KL: {train_kl:.4f}"
        )
        
        # Log to MLflow
        if accelerator.is_main_process:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_recon_loss": train_recon,
                "train_kl_loss": train_kl,
                "kl_weight": kl_weight,
            }, step=epoch)
        
        # Validate
        if (epoch + 1) % config.val_every_n_epochs == 0:
            val_loss, val_recon, val_kl = validate(
                model, val_loader, accelerator, config, epoch
            )
            
            accelerator.print(
                f"Val   | Loss: {val_loss:.4f} | "
                f"Recon: {val_recon:.4f} | KL: {val_kl:.4f}"
            )
            
            # Log to MLflow
            if accelerator.is_main_process:
                mlflow.log_metrics({
                    "val_loss": val_loss,
                    "val_recon_loss": val_recon,
                    "val_kl_loss": val_kl,
                }, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        best_model_path
                    )
                    accelerator.print(f"Saved best model (val_loss={val_loss:.4f})")
                    
                    mlflow.log_metrics({
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                    })
        
        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    checkpoint_path
                )
                accelerator.print(f"Saved checkpoint to {checkpoint_path}")
    
    # Finish training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"\n{'='*70}")
        accelerator.print("Training completed!")
        accelerator.print(f"Best epoch: {best_epoch} with val_loss: {best_val_loss:.4f}")
        accelerator.print(f"{'='*70}")
        
        # Log artifacts
        mlflow.log_artifacts(output_dir, artifact_path="vae_output")
        mlflow.end_run()
    
    accelerator.print("Done!")


if __name__ == "__main__":
    main()
