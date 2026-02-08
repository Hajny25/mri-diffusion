import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

from model import UNetSlicePredictorAttention
from preprocessed_dataset import PreprocessedBraTSSliceDataset
from diffusers import DDPMScheduler

BATCH_SIZE = 20

BASE_DIR = Path(__file__).resolve().parents[2]

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = UNetSlicePredictorAttention()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    # Configure to match training
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    return noise_scheduler


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    context: torch.Tensor,    # [B, 1, H, W], in [-1,1]
    direction: torch.Tensor,  # [B], -1 or +1
    slice_pos: torch.Tensor,  # [B], e.g. in [0,1]
    device: torch.device,
) -> torch.Tensor:
    """
    Run reverse diffusion to sample neighbor slices conditioned on context/direction/slice_pos.
    Returns: [B, 1, H, W] predicted neighbor in [-1,1].
    """
    model.eval()

    context = context.to(device)
    direction = direction.to(device)
    slice_pos = slice_pos.to(device)

    B, _, H, W = context.shape
    x = torch.randn(B, 1, H, W, device=device)  # start from pure noise

    for t in noise_scheduler.timesteps:
        x_in = torch.cat([x, context], dim=1)  # [B, 2, H, W]

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model(x_in, t_batch, direction, slice_pos)  # [B,1,H,W]

        step = noise_scheduler.step(noise_pred, t, x)
        x = step.prev_sample

    return x  # predicted neighbor slice


def show_triplets(
    context: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    save_path: str,
    n_show: int = 4
):
    """
    context, predicted, target: [B,1,H,W], in [-1,1]
    Shows up to n_show triplets in a grid: context | predicted | target
    """
    B = context.size(0)
    n = min(B, n_show)

    # Map [-1,1] -> [0,1] for display
    def to_vis(x):
        return ((x.clamp(-1, 1) + 1) / 2.0).cpu()

    c = to_vis(context)
    p = to_vis(predicted)
    t = to_vis(target)

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 3 * n))

    if n == 1:
        axes = [axes]  # make iterable

    for i in range(n):
        axes[i][0].imshow(c[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i][0].set_title("Context (input)")
        axes[i][0].axis("off")

        axes[i][1].imshow(p[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_title("Predicted neighbor")
        axes[i][1].axis("off")

        axes[i][2].imshow(t[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_title("Ground truth neighbor")
        axes[i][2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=}")
    print(f"{__file__=}")

    checkpoint_path = BASE_DIR / "output/76539c916ce8437e8aa4def52d3625dc/best_model.pt"

    # 1) Load model and scheduler
    model = load_model(checkpoint_path, device)
    noise_scheduler = load_scheduler(num_train_timesteps=1000)

    # 2) Build dataset & dataloader
    dataset = PreprocessedBraTSSliceDataset(
        image_size=128,
    )

    # Small batch is enough for visualization
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # everything is in memory
        pin_memory=True,
    )

    # 3) Get one batch from the dataloader
    batch = next(iter(dataloader))
    context = batch["context"]   # [B,1,H,W]
    target = batch["neighbor"]   # [B,1,H,W]
    direction = batch["direction"]  # [B]
    slice_pos = batch["slice_pos"]  # [B]

    # 4) Sample predicted neighbors
    predicted = sample_batch(
        model=model,
        noise_scheduler=noise_scheduler,
        context=context,
        direction=direction,
        slice_pos=slice_pos,
        device=device,
    )

    # 5) Visualize context vs predicted vs target
    show_triplets(context, predicted, target, BASE_DIR / "sample_output_attn.png", n_show=BATCH_SIZE)


if __name__ == "__main__":
    main()