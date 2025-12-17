import os
import math
import argparse
import importlib
from dataclasses import dataclass, asdict

import accelerate.utils
import mlflow
import perun
from perun.data_model.data import DataNode, MetricType
from perun.processing import processDataNode

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()

if not BRATS_ROOT.exists():
    raise FileNotFoundError(f"Expected BRATS2021 data under {BRATS_ROOT}")

output_dir = "output" # gets set later to include mlflow run id

DEBUG = False


@dataclass
class TrainingConfig:
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 1
    eval_batch_size: int = 16 if not DEBUG else 1 # how many images to sample during evaluation
    num_epochs: int = 10 if not DEBUG else 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed: int = 0
    scheduler: str = "DDPMScheduler"
    dataloader_workers: int = 8 if not DEBUG else 0
    num_train_timesteps: int = 1000 if not DEBUG else 10
    num_inference_steps = 1000 if not DEBUG else 10


config = TrainingConfig()
config_dict = asdict(config)

assert config.eval_batch_size == (sqrt := math.sqrt(config.eval_batch_size)) * sqrt

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


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device=device).manual_seed(config.seed),
        num_inference_steps=config.num_inference_steps,
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    grid_dims = int(math.sqrt(len(images))) if not DEBUG else 1

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=grid_dims, cols=grid_dims)

    # Save the images
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/epoch_{epoch:04d}.png")


@perun.perun(
    data_out=str(BASE_DIR / "perun_results" / str(accelerator.process_index)),
    format="json",
)
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, run_id):
    # (2) add the data collected by perun to mlflow
    perun.register_callback(log_perun_metrics_to_mlflow)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # (1) log params in mlflow
    if accelerator.is_main_process:
        mlflow.log_params(config_dict, run_id=run_id)

    # Now you train the model
    for epoch in range(config.num_epochs):
        accelerator.print("")
        accelerator.print(f"Epoch {epoch + 1}/{config.num_epochs} ---------------------")

        for step, batch in enumerate(train_dataloader):
            accelerator.print(f"Epoch {epoch + 1}/{config.num_epochs}: Step {step + 1}/{len(train_dataloader)}")
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            #progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            #progress_bar.set_postfix(**logs)
            # (6) log the training loss and learning rate
            mlflow.log_metric("training_loss", logs["loss"], step=global_step, run_id=run_id)
            mlflow.log_metric("learning_rate", logs["lr"], step=global_step, run_id=run_id)

            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                accelerator.print(f"sample demo images in epoch: {epoch + 1}")
                #evaluate(config, epoch, pipeline)


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(output_dir, "unet_weights.pth"))
        accelerator.print(f"Logging model to mlflow...")
        # (7) Log model artifact to MLflow
        mlflow.log_artifacts(output_dir, artifact_path="output", run_id=run_id)
        print("finished logging to mlflow")
        mlflow.pytorch.log_model(accelerator.unwrap_model(model), "unet_weights")
        

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

        global output_dir
        output_dir = os.path.join("output", run)
        os.makedirs(output_dir, exist_ok=True)

    run_id = accelerate.utils.gather_object([run])[0]
    return run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_module = importlib.import_module(f"experiments.{args.model}.model")
    model = getattr(model_module, "create_model")(config.image_size)

    dataset_module = importlib.import_module(f"experiments.{args.model}.dataset")
    dataset = getattr(dataset_module, "create_dataset")(BRATS_ROOT, config.image_size, DEBUG)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
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

    run_id = start_mlflow_run(args.model)

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, run_id)


if __name__ == "__main__":
    main()
