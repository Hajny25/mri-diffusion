import os
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
from safetensors.torch import load_file

from model import UNetSlicePredictor  # neighbor model

BATCH_SIZE = 1

BASE_DIR = Path(__file__).resolve().parents[2]

def load_neighbor_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = UNetSlicePredictor()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_middle_model(ckpt_path: str, device: torch.device) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=128,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # downsample with self-attention
            "AttnDownBlock2D",  # downsample with self-attention
        ),
        up_block_types=(
            "AttnUpBlock2D",  # upsample with self-attention
            "AttnUpBlock2D",  # upsample with self-attention
            "UpBlock2D",  # regular ResNet upsampling block
            "UpBlock2D",  # regular ResNet upsampling block
        ),
        # Default mid block (UNetMidBlock2D) includes self-attention automatically
    )
    sd = load_file(ckpt_path, device="cpu")
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

def load_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    return scheduler

@torch.no_grad()
def sample_neighbor_slice(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    context: torch.Tensor,      # [B,1,H,W], in [-1,1]
    direction: torch.Tensor,    # [B], -1 or +1
    slice_pos: torch.Tensor,    # [B], e.g. in [0,1]
    device: torch.device,
) -> torch.Tensor:
    """
    Runs reverse diffusion to sample one neighbor slice conditioned on context/direction/slice_pos.
    Returns [B,1,H,W] in [-1,1].
    """
    model.eval()
    context = context.to(device)
    direction = direction.to(device)
    slice_pos = slice_pos.to(device)

    B, _, H, W = context.shape
    x = torch.randn(B, 1, H, W, device=device)

    for t in scheduler.timesteps:
        x_in = torch.cat([x, context], dim=1)  # [B,2,H,W]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model(x_in, t_batch, direction, slice_pos)
        step = scheduler.step(noise_pred, t, x)
        x = step.prev_sample

    return x  # [B,1,H,W]

@torch.no_grad()
def sample_middle_slice(model, scheduler, batch_size):
    image = torch.randn(batch_size, 1, 128, 128).to(model.device)

    for t in scheduler.timesteps:
        # 1. predict noise model_output
        model_output = model(image, t).sample
    
        # 2. compute previous image: x_t -> x_t-1
        image = scheduler.step(model_output, t, image).prev_sample

    return image

@torch.no_grad()
def generate_volume_from_middle(
    middle_model: torch.nn.Module,
    neighbor_model: torch.nn.Module,
    scheduler: DDPMScheduler,
    batch_size: int,
    num_slices: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a 3D volume slice-by-slice starting from a middle slice.

    Returns: volume [B, num_slices, 1, H, W] in [-1,1].
    """
    middle_model.eval()
    neighbor_model.eval()

    # 1) Generate middle slice
    # This is a placeholder; adapt to your middle model's forward interface.
    middle_slice = sample_middle_slice(middle_model, scheduler, batch_size, ima)
    print("finished sampling middle slice")

    B, _, H, W = middle_slice.shape
    print(middle_slice.shape)

    volume = torch.zeros(B, num_slices, 1, H, W, device=device)
    mid = num_slices // 2
    volume[:, mid] = middle_slice

    # Helper to compute normalized slice_pos as in training
    def slice_pos_fn(idx: int) -> float:
        return idx / (num_slices - 1) if num_slices > 1 else 0.0

    # 2) Generate slices below (direction = +1)
    for k in range(mid, num_slices - 1):
        print(f"generating slice {k=}")
        context = volume[:, k]  # [B,1,H,W]
        direction = torch.full((B,), +1, device=device, dtype=torch.long)
        slice_pos = torch.full((B,), slice_pos_fn(k + 1), device=device, dtype=torch.float32)

        neighbor = sample_neighbor_slice(
            model=neighbor_model,
            scheduler=scheduler,
            context=context,
            direction=direction,
            slice_pos=slice_pos,
            device=device,
        )  # [B,1,H,W]

        volume[:, k + 1] = neighbor

    # 3) Generate slices above (direction = -1)
    for k in range(mid, 0, -1):
        print(f"generating slice {k=}")
        context = volume[:, k]  # [B,1,H,W]
        direction = torch.full((B,), -1, device=device, dtype=torch.long)
        slice_pos = torch.full((B,), slice_pos_fn(k - 1), device=device, dtype=torch.float32)

        neighbor = sample_neighbor_slice(
            model=neighbor_model,
            scheduler=scheduler,
            context=context,
            direction=direction,
            slice_pos=slice_pos,
            device=device,
        )

        volume[:, k - 1] = neighbor

    return volume

def save_volume_grid(
    volume: torch.Tensor,
    b: int,
    save_path,
    n_cols = None
):
    """
    volume: [B, num_slices, 1, H, W] in [-1,1]
    b: batch index to plot
    save_path: where to save the PNG
    n_cols: number of columns in grid (if None, will use ceil(sqrt(num_slices)))
    """

    vol = volume[b]  # [num_slices, 1, H, W]
    num_slices = vol.shape[0]

    # [-1,1] -> [0,1] for display
    vol_vis = (vol.clamp(-1, 1) + 1) / 2.0  # [num_slices,1,H,W]
    vol_vis = vol_vis.cpu()

    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(num_slices)))
    n_rows = int(math.ceil(num_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.reshape(n_rows, n_cols)

    mid_idx = num_slices // 2
    slice_idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if slice_idx < num_slices:
                img = vol_vis[slice_idx, 0].numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                if slice_idx == mid_idx:
                    ax.set_title(f"{slice_idx} (mid)", fontsize=6, color="red")
                    # Draw a red rectangle around the image
                    h, w = img.shape
                    rect = Rectangle(
                        (0, 0),                      # (x,y)
                        w, h,                        # width, height
                        linewidth=1.5,
                        edgecolor="red",
                        facecolor="none",
                        transform=ax.transData,
                    )
                    ax.add_patch(rect)
                ax.set_title(f"{slice_idx}", fontsize=6)
                slice_idx += 1

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved volume grid to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths – change these
    neighbor_ckpt = BASE_DIR / "output/54ff105b5d53495c9652bfc8fe9a6861/best_model.pt"
    middle_ckpt = BASE_DIR / "output/f89504003d70424492ce6f0c7bec0a08/best_model/diffusion_pytorch_model.safetensors"

    NUM_SLICES = 155

    # 1) Load models and scheduler
    neighbor_model = load_neighbor_model(neighbor_ckpt, device)
    middle_model = load_middle_model(middle_ckpt, device)
    scheduler = load_scheduler(num_train_timesteps=1000)

    # 3) Generate full 3D volume
    volume = generate_volume_from_middle(
        middle_model=middle_model,
        neighbor_model=neighbor_model,
        scheduler=scheduler,
        batch_size=BATCH_SIZE,
        num_slices=NUM_SLICES,
        device=device,
    )  # [B, num_slices, 1, H, W]

    # 4) Save volume to disk for inspection
    out_path = "generated_volume.pt"
    torch.save(volume.cpu(), out_path)
    print(f"Saved generated volume to {out_path}")

    os.makedirs("output/ddpm_25d", exist_ok=True)
    for b in range(BATCH_SIZE):
        save_volume_grid(volume, b, f"output/ddpm_25d/sample3d_{b}.png")

if __name__ == "__main__":
    main()