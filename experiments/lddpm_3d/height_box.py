#!/usr/bin/env python

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np


def find_case_z_bounds(nii_path, eps=0.0):
    """
    Find min and max z indices where the brain (non-zero voxels) is present
    in a 3D or 4D NIfTI volume. Uses non-zero mask.

    Args:
        nii_path (Path or str): Path to NIfTI file.
        eps (float): Optional threshold >0 if you want to ignore tiny values.

    Returns:
        (path_str, z_min, z_max) or (path_str, None, None) if no non-zero voxels.
    """
    nii_path = Path(nii_path)
    img = nib.load(str(nii_path))
    data = img.get_fdata()

    # If 4D (e.g. (H, W, D, 1)), squeeze the last dim
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} for {nii_path}")

    if eps > 0:
        mask = data > eps
    else:
        mask = data != 0

    mask_z = mask.any(axis=(0, 1))  # (D,)

    if not mask_z.any():
        return str(nii_path), None, None

    z_indices = np.where(mask_z)[0]
    z_min = int(z_indices[0])
    z_max = int(z_indices[-1])
    return str(nii_path), z_min, z_max


def main():
    parser = argparse.ArgumentParser(
        description="Find global top/bottom z bounding box of the brain over all cases (parallel)."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory where BraTS cases are stored (recursively searched for *_flair.nii.gz).",
    )
    parser.add_argument(
        "--modality_suffix",
        type=str,
        default="_flair.nii.gz",
        help="Suffix to identify the modality to use for bounding box (default: _flair.nii.gz).",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Optional max number of cases to process.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Optional intensity threshold for mask (default: 0.0 meaning data!=0).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel processes (default: use os.cpu_count()).",
    )

    args = parser.parse_args()
    root = Path(args.root)

    files = sorted(root.rglob(f"*{args.modality_suffix}"))
    if args.max_cases is not None:
        files = files[: args.max_cases]

    if len(files) == 0:
        print(f"No files found with suffix {args.modality_suffix} under {root}")
        return

    print(f"Found {len(files)} cases. Using up to {args.num_workers or 'all available'} workers.")

    global_z_min = None
    global_z_max = None

    # Submit all jobs to the process pool
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(find_case_z_bounds, f, args.eps): f for f in files
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                path_str, z_min, z_max = future.result()
            except Exception as e:
                print(f"[{i:04d}] Error processing {futures[future]}: {e}")
                continue

            if z_min is None:
                print(f"[{i:04d}] {Path(path_str).name}: no non-zero voxels found, skipping.")
                continue

            print(f"[{i:04d}] {Path(path_str).name}: z_min = {z_min}, z_max = {z_max}")

            if global_z_min is None or z_min < global_z_min:
                global_z_min = z_min
            if global_z_max is None or z_max > global_z_max:
                global_z_max = z_max

    if global_z_min is None:
        print("No valid cases with non-zero voxels were found.")
        return

    print("\n========================================")
    print("Global brain z bounding box over dataset")
    print("========================================")
    print(f"Global z_min (top of brain):    {global_z_min}")
    print(f"Global z_max (bottom of brain): {global_z_max}")
    print(f"Total usable z range (inclusive): {global_z_min} .. {global_z_max}")
    print("========================================\n")


if __name__ == "__main__":
    main()