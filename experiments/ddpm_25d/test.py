import os
import math
import time
from dataset import create_dataset
from preprocessed_dataset import PreprocessedBraTSSliceDataset
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = Path(__file__).resolve().parents[2]
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()
PREPROCESSED_ROOT = Path(BASE_DIR / "data" / "preprocessed").expanduser()
BATCH_SIZE = 1

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



volume = torch.load(BASE_DIR / "generated_volume2.pt", map_location='cpu')

os.makedirs("output/ddpm_25d", exist_ok=True)
for b in range(BATCH_SIZE):
    save_volume_grid(volume, b, f"output/ddpm_25d/sample3d_2_{b}.png")