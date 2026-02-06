import time
import os
from dataclasses import dataclass, asdict

import accelerate.utils
import mlflow
import perun
from perun.data_model.data import DataNode, MetricType
from perun.processing import processDataNode

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from pathlib import Path

from preprocessed_dataset import PreprocessedBraTSSliceDataset
from mm_dataset import MemmapDataset, get_debug_dataset, get_full_dataset, get_full_masks_dataset
from model import UNetSlicePredictor, UNetSlicePredictorAttention


BASE_DIR = Path(__file__).resolve().parents[2]
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()
PREPROCESSED_ROOT = Path(BASE_DIR / "data" / "preprocessed").expanduser()
PREPROCESSED_MASKS_ROOT = Path(BASE_DIR / "data" / "preprocessed_masks").expanduser()

if not BRATS_ROOT.exists():
    raise FileNotFoundError(f"Expected BRATS2021 data under {BRATS_ROOT}")

DEBUG = False


@dataclass
class TrainingConfig:
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 50 if not DEBUG else 1
    eval_batch_size: int = 50 if not DEBUG else 1
    num_epochs: int = 100 if not DEBUG else 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    log_mlflow_iterations: int = 100
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed: int = 0
    scheduler: str = "DDPMScheduler"
    dataloader_workers: int = 16 if not DEBUG else 0 
    num_train_timesteps: int = 1000 if not DEBUG else 10
    num_inference_steps = 1000 if not DEBUG else 10


config = TrainingConfig()
config_dict = asdict(config)

# -------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

print("=== PyTorch CUDA / Slurm info ===")
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"[GPU {i}] {props.name}, {props.total_memory / (1024 ** 3):.1f} GB")


# Initialize accelerator
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)

accelerator.print("=== Accelerator State ===")
accelerator.print(accelerator.state)


def validate(config, model, noise_scheduler, val_dataloader, run_id, epoch):
    """Compute validation loss on the validation dataset."""
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            context = batch["context"]        # [B, C, H, W]
            target = batch["neighbor"]        # [B, C, H, W]
            direction = batch["direction"]    # [B]
            slice_pos = batch["slice_pos"]    # [B], in [0,1]

            # Sample noise to add to the images
            noise = torch.randn_like(target)
            bs = target.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(target, noise, timesteps)
            input = torch.cat([noisy_images, context], dim=1)
            
            # Predict the noise residual
            noise_pred = model(input, timesteps, direction, slice_pos)
            loss = F.mse_loss(noise_pred, noise)
            
            total_val_loss += loss.item()
            num_batches += 1
    
    # Gather losses from all processes to compute global average
    total_val_loss_tensor = torch.tensor([total_val_loss], device=accelerator.device)
    num_batches_tensor = torch.tensor([num_batches], device=accelerator.device)
    
    # Sum across all processes (gather_for_metrics removes padding from uneven splits)
    total_val_loss_gathered = accelerator.gather_for_metrics(total_val_loss_tensor).sum().item()
    num_batches_gathered = accelerator.gather_for_metrics(num_batches_tensor).sum().item()
    
    avg_val_loss = total_val_loss_gathered / num_batches_gathered if num_batches_gathered > 0 else 0.0
    
    # Log validation loss to MLflow (same value on all processes now)
    if accelerator.is_main_process:
        mlflow.log_metric("validation_loss", avg_val_loss, step=epoch, run_id=run_id)
    accelerator.print(f"Validation Loss: {avg_val_loss:.4f}")
    
    model.train()
    return avg_val_loss


@perun.perun(
    data_out=str(BASE_DIR / "perun_results" / str(accelerator.process_index)),
    format="json",
)
def train_loop(config, model: UNetSlicePredictor, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler, run_id, output_dir):
    # (2) add the data collected by perun to mlflow
    perun.register_callback(log_perun_metrics_to_mlflow)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    best_val_loss = float('inf')
    best_epoch = -1
    
    # (1) log params in mlflow
    if accelerator.is_main_process:
        mlflow.log_params(config_dict, run_id=run_id)

    # Now you train the model
    for epoch in range(config.num_epochs):
        accelerator.print("")
        accelerator.print(f"Epoch {epoch + 1}/{config.num_epochs} ---------------------")

        for step, batch in enumerate(train_dataloader):
            accelerator.print(f"Epoch {epoch + 1}/{config.num_epochs}: Step {step + 1}/{len(train_dataloader)}")
            context = batch["context"]        # [B, C, H, W]
            target = batch["neighbor"]        # [B, C, H, W]
            direction = batch["direction"]    # [B]
            slice_pos = batch["slice_pos"]    # [B], in [0,1]

            # Sample noise to add to the images
            noise = torch.randn_like(target)
            bs = target.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(target, noise, timesteps)
            input = torch.cat([noisy_images, context], dim=1)

            with accelerator.accumulate(model):

                # Predict the noise residual
                noise_pred = model(input, timesteps, direction, slice_pos)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if step % config.log_mlflow_iterations == 0:
                # (6) log the training loss and learning rate
                mlflow.log_metric("training_loss", logs["loss"], step=global_step, run_id=run_id)
                mlflow.log_metric("learning_rate", logs["lr"], step=global_step, run_id=run_id)

            global_step += 1

        # After each epoch, run validation
        accelerator.print(f"Running validation after epoch {epoch + 1}...")
        val_loss = validate(config, accelerator.unwrap_model(model), noise_scheduler, val_dataloader, run_id, epoch)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            accelerator.print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            
            # Save the best model
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                best_model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    best_model_path
                )
                accelerator.print(f"Saved best model to {best_model_path}")
                
                # Log best validation loss to MLflow
                mlflow.log_metric("best_validation_loss", best_val_loss, run_id=run_id)
                mlflow.log_metric("best_epoch", best_epoch, run_id=run_id)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Log best model artifact to MLflow
        print(f"logging best model to mlflow...")
        mlflow.log_artifacts(output_dir, artifact_path="output", run_id=run_id)
        print(f"Best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        print("finished logging to mlflow")

    accelerator.wait_for_everyone()
    print("finished training")

# -------------------------------------------------------------------
# Perun ↔ MLflow bridge
# -------------------------------------------------------------------
@accelerator.on_main_process
def log_perun_metrics_to_mlflow(root: DataNode) -> None:
    print("Logging Perun metrics to MLflow...")
    cfg = getattr(perun, "config", None)
    processed_root = processDataNode(root, cfg, force_process=False) if cfg is not None else root

    def find_first_metric(node: DataNode, metric_type: MetricType):
        metrics = getattr(node, "metrics", None)
        if metrics and metric_type in metrics:
            return float(metrics[metric_type].value)

        for child in getattr(node, "nodes", {}).values():
            val = find_first_metric(child, metric_type)
            if val is not None:
                return val
        return None

    run = mlflow.active_run()
    if run is None:
        print("No active MLflow run found. Skipping logging Perun metrics.")
        return

    total_energy_j = find_first_metric(processed_root, MetricType.ENERGY)
    runtime_s = find_first_metric(processed_root, MetricType.RUNTIME)
    co2_kg = find_first_metric(processed_root, MetricType.CO2)
    money = find_first_metric(processed_root, MetricType.MONEY)

    def log_if_not_none(name: str, value):
        if value is not None:
            mlflow.log_metric(name, float(value))
        else:
            print(f"Perun metric {name} not found; skipping.")

    log_if_not_none("perun_energy_joules", total_energy_j)
    log_if_not_none("perun_runtime_seconds", runtime_s)
    log_if_not_none("perun_co2_kg", co2_kg)
    log_if_not_none("perun_cost", money)

    if total_energy_j is not None:
        energy_kwh = total_energy_j / 3.6e6
        log_if_not_none("perun_energy_kwh", energy_kwh)

    if total_energy_j is not None and runtime_s is not None and runtime_s > 0:
        avg_power_w = total_energy_j / runtime_s
        log_if_not_none("perun_avg_power_watts", avg_power_w)

def start_mlflow_run(experiment: str) -> str:
    mlflow.set_experiment(experiment)  # (2) MLFLOW: set the experiment name
    run = 0
    if accelerator.is_main_process:
        mlflow.start_run()
        run = mlflow.active_run().info.run_id
        print(f"MLflow run id: {run}")

    run_id = accelerate.utils.gather_object([run])[0]
    return run_id

def main():
    resume = False
    if resume:
        checkpoint = BASE_DIR / "output/76539c916ce8437e8aa4def52d3625dc/best_model.pt"
    else:
        checkpoint = None

    model = UNetSlicePredictorAttention()
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        
    model.train()

    # Create full dataset and split into train/validation
    #full_dataset = PreprocessedBraTSSliceDataset(PREPROCESSED_ROOT, config.image_size)
    full_dataset = get_full_masks_dataset()
    
    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    accelerator.print(f"Dataset split: {train_size} train, {val_size} validation")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
        persistent_workers=not DEBUG,
        pin_memory=True,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_workers,
        shuffle=False,
        persistent_workers=not DEBUG,
        pin_memory=True,
    )

    # Create a scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    run_id = start_mlflow_run("ddpm2.5d")
    output_dir = os.path.join("output", "ddpm_25d", run_id)
    os.makedirs(output_dir, exist_ok=True)

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler, run_id, output_dir)


if __name__ == "__main__":
    main()
