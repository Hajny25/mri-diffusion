from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import save_image

import mlflow
import mlflow.pytorch

import perun
from perun.data_model.data import MetricType, DataNode
from perun.processing import processDataNode

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = (BASE_DIR / "../dataset").resolve()

print("Using DATASET_ROOT:", DATASET_ROOT)

IMAGE_SIZE = 128
BATCH_SIZE = 8
NUM_WORKERS = 0
TIMESTEPS = 1000
LEARNING_RATE = 2e-4

NUM_EPOCHS = 5
PATIENCE = 5
MIN_DELTA = 1e-4

DEBUG_FAST = False
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


# -------------------------------------------------------------------
# Dataset and DataLoaders
# -------------------------------------------------------------------
full_dataset = BraTSSliceDataset(DATASET_ROOT, image_size=IMAGE_SIZE)
if DEBUG_FAST:
    indices = list(range(64))
    full_dataset = Subset(full_dataset, indices)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")


# -------------------------------------------------------------------
# Model, diffusion process, optimizer, scheduler
# -------------------------------------------------------------------
model = UNet(
    img_channels=1,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    time_emb_dim=256,
).to(device)

diffusion = GaussianDiffusion(
    model=model,
    image_size=IMAGE_SIZE,
    channels=1,
    timesteps=TIMESTEPS,
).to(device)

optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# -------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------
def train_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.train()
    running_loss = 0.0
    n_steps = 0

    for step, x in enumerate(train_loader, start=1):
        x = x.to(device)

        t = torch.randint(
            0,
            diffusion.timesteps,
            (x.size(0),),
            device=device,
        ).long()

        loss = diffusion.p_losses(x, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_steps += 1

        if step % 500 == 0:
            avg = running_loss / n_steps
            print(f"[epoch {epoch} | step {step}] avg loss: {avg:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, n_steps)
    print(f"Epoch {epoch} | Train loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.eval()
    running_loss = 0.0
    n_steps = 0

    for step, x in enumerate(val_loader, start=1):
        x = x.to(device)
        t = torch.randint(
            0,
            diffusion.timesteps,
            (x.size(0),),
            device=device,
        ).long()

        loss = diffusion.p_losses(x, t)
        running_loss += loss.item()
        n_steps += 1

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, n_steps)
    print(f"Epoch {epoch} | Val loss:   {avg_loss:.4f}")
    return avg_loss

def sample_and_save(
    diffusion: GaussianDiffusion,
    epoch: int,
    num_samples: int = 16,
    out_dir: str = "samples",
    context_slices=None,
    nrow: int = 4,
) -> None:
    diffusion.model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # samples: (B, C, H, W), C = channels (1 for 2D, >1 for 2.5D)
        samples = diffusion.sample(batch_size=num_samples).cpu()

    # map from [-1, 1] to [0, 1]
    samples = samples.clamp(-1, 1)
    samples = (samples + 1) / 2.0

    save_path = out_dir / f"samples_epoch_{epoch:03d}.png"
    save_image(samples, save_path, nrow=nrow)

    print(f"Saved samples to {save_path}")

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(save_path), artifact_path="samples")


# -------------------------------------------------------------------
# Perun ↔ MLflow bridge
# -------------------------------------------------------------------
def log_perun_metrics_to_mlflow(root: DataNode) -> None:
    """
    Called by Perun after monitoring; `root` is the full DataNode tree.
    Extract total energy/runtime/CO2 and log them to the *current* MLflow run.
    """
    try:
        cfg = getattr(perun, "config", None)
        if cfg is not None:
            processed_root = processDataNode(root, cfg, force_process=False)
        else:
            print("[Perun] No config found, skipping post-processing.")
            processed_root = root
    except Exception as e:
        print(f"[Perun] post-processing failed, skipping Perun metrics: {e}")
        processed_root = root

    def find_first_metric(node: DataNode, metric_type: MetricType):
        if getattr(node, "metrics", None) and metric_type in node.metrics:
            metric = node.metrics[metric_type]
            return float(metric.value)

        if getattr(node, "nodes", None):
            for child in node.nodes.values():
                val = find_first_metric(child, metric_type)
                if val is not None:
                    return val
        return None

    energy_j = find_first_metric(processed_root, MetricType.ENERGY)
    runtime_s = find_first_metric(processed_root, MetricType.RUNTIME)
    co2_kg   = find_first_metric(processed_root, MetricType.CO2)

    run = mlflow.active_run()
    if run is None:
        return

    if energy_j is not None:
        mlflow.log_metric("perun_energy_joules", energy_j)
    if runtime_s is not None:
        mlflow.log_metric("perun_runtime_seconds", runtime_s)
    if co2_kg is not None:
        mlflow.log_metric("perun_co2_kg", co2_kg)

# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def train() -> float:
    print("Starting Training")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(epoch, max_steps=10 if DEBUG_FAST else None)
        val_loss = validate(epoch, max_steps=5 if DEBUG_FAST else None)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            step=epoch,
        )

        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            torch.save(diffusion.state_dict(), str(BASE_DIR / "2d_central_ddpm_flair_best.pt"))
            print(f"✅ New best val loss: {best_val:.4f}")

            mlflow.log_artifact(
                str(BASE_DIR / "2d_central_ddpm_flair_best.pt"),
                artifact_path="checkpoints",
            )
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

        if epoch % 5 == 0:
            sample_and_save(
                diffusion,
                epoch=epoch,
                num_samples=16,
                out_dir="samples",
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"⏹ Early stopping at epoch {epoch} "
                f"(no val improvement for {PATIENCE} epochs).",
            )
            break

    return best_val


@perun.perun(
    data_out=str(BASE_DIR / "perun_results"),
    format="json",
)
def train_with_perun():
    try:
        perun.register_callback(log_perun_metrics_to_mlflow)
    except Exception as e:
        print(f"Perun callback registration failed: {e}")

    return train()


def main() -> None:
    mlflow.set_experiment("brats_ddpm_2d_central_slice")

    with mlflow.start_run(run_name="ddpm_2d_central_flair"):
        mlflow.log_params(
            {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "timesteps": TIMESTEPS,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "min_delta": MIN_DELTA,
                "device": str(device),
                "model": "UNet",
                "dataset": "BraTS_2D_central_slice_flair",
            }
        )

        best_val = train_with_perun()

        mlflow.pytorch.log_model(diffusion.model, artifact_path="final_model")

        if best_val is not None:
            mlflow.log_metric("best_val_loss", best_val)


if __name__ == "__main__":
    main()
