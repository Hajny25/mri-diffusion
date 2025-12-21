"""
Evaluation script for trained 3D VAE.
Computes reconstruction metrics and saves visualizations.
"""

import os
from pathlib import Path
from dataclasses import dataclass
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vae import VAE3D, get_vae_small, get_vae_base, get_vae_large
from dataset import create_dataset


@dataclass
class EvalConfig:
    """Configuration for VAE evaluation."""
    checkpoint_path: str
    data_root: str = "data/brats-2021"
    patch_size: tuple = (128, 160, 160)
    batch_size: int = 2
    num_workers: int = 4
    max_cases: int = None  # Evaluate on all data if None
    save_dir: str = "output/vae_eval"
    save_reconstructions: bool = True
    num_visualizations: int = 5  # Number of cases to save as .nii.gz


def compute_metrics(original, reconstructed):
    """
    Compute reconstruction metrics.
    
    Args:
        original: (B, C, D, H, W) tensor
        reconstructed: (B, C, D, H, W) tensor
    
    Returns:
        dict with MSE, MAE, PSNR per sample
    """
    # Flatten spatial dimensions for per-sample metrics
    B = original.shape[0]
    orig_flat = original.view(B, -1)
    recon_flat = reconstructed.view(B, -1)
    
    # MSE per sample
    mse = F.mse_loss(recon_flat, orig_flat, reduction='none').mean(dim=1)
    
    # MAE per sample
    mae = F.l1_loss(recon_flat, orig_flat, reduction='none').mean(dim=1)
    
    # PSNR (assume data range [-1, 1] -> range = 2)
    # PSNR = 10 * log10(MAX^2 / MSE)
    max_val = 2.0
    psnr = 10 * torch.log10(max_val**2 / (mse + 1e-8))
    
    return {
        'mse': mse,  # (B,)
        'mae': mae,  # (B,)
        'psnr': psnr,  # (B,)
    }


def save_comparison_slices(original, reconstructed, path, num_slices=9):
    """
    Save comparison of original vs reconstructed volume slices.
    
    Args:
        original: (D, H, W) numpy array
        reconstructed: (D, H, W) numpy array
        path: output path
        num_slices: number of slices to show (arranged in grid)
    """
    D, H, W = original.shape
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    # Create figure with 3 rows: original, reconstructed, difference
    fig, axes = plt.subplots(3, num_slices, figsize=(num_slices*2, 6))
    
    for i, slice_idx in enumerate(slice_indices):
        orig_slice = original[slice_idx]
        recon_slice = reconstructed[slice_idx]
        diff = np.abs(orig_slice - recon_slice)
        
        # Original
        axes[0, i].imshow(orig_slice, cmap='gray', vmin=-1, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10, loc='left')
        
        # Reconstructed
        axes[1, i].imshow(recon_slice, cmap='gray', vmin=-1, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10, loc='left')
        
        # Difference
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Abs Diff', fontsize=10, loc='left')
        
        # Add slice number at bottom
        axes[2, i].text(0.5, -0.1, f'#{slice_idx}', 
                       transform=axes[2, i].transAxes,
                       ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate(config: EvalConfig):
    """Run evaluation on validation data."""
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        log_with=None,
    )
    
    # Create output directory
    save_dir = Path(config.save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        if config.save_reconstructions:
            (save_dir / "reconstructions").mkdir(exist_ok=True)
    
    accelerator.print(f"Loading checkpoint from: {config.checkpoint_path}")
    
    # load model
    state_dict = torch.load(config.checkpoint_path, map_location='cpu')
    
    vae = get_vae_base()
    
    vae.load_state_dict(state_dict)
    vae.eval()
    
    # Load validation dataset
    accelerator.print(f"Loading validation data from: {config.data_root}")
    val_dataset = create_dataset(
        root=Path(config.data_root),
        patch_size=config.patch_size,
        max_cases=config.max_cases,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Prepare for distributed evaluation
    vae, val_loader = accelerator.prepare(vae, val_loader)
    
    accelerator.print(f"Evaluating on {len(val_dataset)} samples...")
    
    # Collect metrics
    all_mse = []
    all_mae = []
    all_psnr = []
    
    # For saving visualizations
    saved_count = 0
    
    for batch_idx, volumes in enumerate(tqdm(val_loader, disable=not accelerator.is_local_main_process)):
        # Forward pass
        recon_volumes, mean, logvar = vae(volumes, sample_posterior=False)
        
        # Compute metrics
        metrics = compute_metrics(volumes, recon_volumes)
        
        # Gather metrics across all processes
        mse = accelerator.gather_for_metrics(metrics['mse'])
        mae = accelerator.gather_for_metrics(metrics['mae'])
        psnr = accelerator.gather_for_metrics(metrics['psnr'])
        
        all_mse.append(mse)
        all_mae.append(mae)
        all_psnr.append(psnr)
        
        # Save visualizations (only on main process)
        if config.save_reconstructions and accelerator.is_main_process:
            if saved_count < config.num_visualizations:
                # Get first sample from batch
                orig = volumes[0, 0].cpu().numpy()  # (D, H, W)
                recon = recon_volumes[0, 0].cpu().numpy()  # (D, H, W)
                
                # Save comparison image
                case_idx = batch_idx * config.batch_size * accelerator.num_processes + saved_count
                comparison_path = save_dir / "reconstructions" / f"case_{case_idx:04d}_comparison.png"
                
                save_comparison_slices(orig, recon, comparison_path)
                
                saved_count += 1
    
    # Aggregate metrics
    all_mse = torch.cat(all_mse)
    all_mae = torch.cat(all_mae)
    all_psnr = torch.cat(all_psnr)
    
    if accelerator.is_main_process:
        # Compute statistics
        results = {
            'mse_mean': all_mse.mean().item(),
            'mse_std': all_mse.std().item(),
            'mae_mean': all_mae.mean().item(),
            'mae_std': all_mae.std().item(),
            'psnr_mean': all_psnr.mean().item(),
            'psnr_std': all_psnr.std().item(),
            'num_samples': len(all_mse),
        }
        
        # Print results
        accelerator.print("\n" + "="*50)
        accelerator.print("VAE Evaluation Results")
        accelerator.print("="*50)
        accelerator.print(f"Number of samples: {results['num_samples']}")
        accelerator.print(f"MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
        accelerator.print(f"MAE:  {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
        accelerator.print(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        accelerator.print("="*50 + "\n")
        
        # Save results to file
        results_path = save_dir / "evaluation_results.txt"
        with open(results_path, 'w') as f:
            f.write("VAE Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Checkpoint: {config.checkpoint_path}\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
            f.write(f"MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}\n")
            f.write(f"MAE:  {results['mae_mean']:.6f} ± {results['mae_std']:.6f}\n")
            f.write(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB\n")
        
        accelerator.print(f"Results saved to: {results_path}")
        if config.save_reconstructions:
            accelerator.print(f"Visualizations saved to: {save_dir / 'reconstructions'}")
    
    return results if accelerator.is_main_process else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained 3D VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default="data/brats-2021", help="Path to data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_cases", type=int, default=None, help="Max cases to evaluate")
    parser.add_argument("--save_dir", type=str, default="output/vae_eval", help="Output directory")
    parser.add_argument("--num_vis", type=int, default=5, help="Number of visualizations to save")
    parser.add_argument("--no_save", action="store_true", help="Don't save reconstructions")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_cases=args.max_cases,
        save_dir=args.save_dir,
        save_reconstructions=not args.no_save,
        num_visualizations=args.num_vis,
    )
    
    evaluate(config)


if __name__ == "__main__":
    main()
