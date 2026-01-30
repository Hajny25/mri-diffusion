import torchio as tio
import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

def preprocess_volume(volume_path: Path, output_path: Path, image_size: int, modality_suffix: str):
    """
    - Finds all *flair.nii.gz files
    - Uses central 80% slices from each volume
    - Returns normalized 2D slice as tensor in [-1, 1], shape (1, H, W)
    - Also returns normalized slice index in [0, 1]
    """
    print(f"[INFO] Processing volume: {volume_path}")

    # img = tio.ScalarImage(volume_path).data
    # img = (img > 10) * 1
    # crop = tio.CropOrPad((168,224,155))
    # resize = tio.Resize((128, 128, 155))
    # img = resize(crop(img))
    # vol = img.data

    img = tio.ScalarImage(volume_path).data
    normalize = tio.RescaleIntensity((-1, 1))
    crop = tio.CropOrPad((168,224,155))
    resize = tio.Resize((128, 128, 155))
    img = resize(normalize(crop(img)))
    vol = img.data

    slices = vol.permute(3, 0, 1, 2) # [1, H, W, D] -> [D, 1, H, W]
    z_pos = torch.linspace(0, 1, vol.shape[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"slices": slices, "z_pos": z_pos}, output_path)

    print(
        f"[OK] Saved {slices.shape[0]} slices for {volume_path} "
        f"to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of original BraTS data (where *_flair.nii.gz lives).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store preprocessed .pt files.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Target image size (image_size x image_size).",
    )
    parser.add_argument(
        "--modality_suffix",
        type=str,
        default="_flair.nii.gz",
        help="Suffix to match FLAIR volumes.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"[INFO] Scanning for volumes under: {root_dir}")
    t1 = time.time()
    volume_paths = sorted(root_dir.rglob(f"*{args.modality_suffix}"))
    t2 = time.time()
    print(f"Scanning took {t2 - t1} seconds.")

    if not volume_paths:
        raise RuntimeError(f"No volumes matching *{args.modality_suffix} found under {root_dir}")

    print(f"[INFO] Found {len(volume_paths)} volumes.")

    t3 = time.time()
    for i, vol_path in enumerate(volume_paths, start=1):
        # Keep folder structure, just change root and extension:
        # e.g. root/.../xxx_flair.nii.gz -> output/.../xxx_flair.pt
        rel = vol_path.relative_to(root_dir)
        # Remove last suffix (.gz), then replace .nii with .pt
        out_rel = rel.with_suffix("")      # drop .gz
        out_rel = out_rel.with_suffix(".pt")  # replace .nii -> .pt
        out_path = output_dir / out_rel

        print(f"[{i}/{len(volume_paths)}] -> {out_path}")
        preprocess_volume(vol_path, out_path, args.image_size, args.modality_suffix)
        t4 = time.time()
        print(f"Processing took {t4 - t3} seconds.")
        t3 = t4

    print(f"Total: {time.time() - t1} seconds.")



if __name__ == "__main__":
    main()