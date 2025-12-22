import os
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset



def _normalize_volume(vol, eps=1e-6, clip_val=5.0):
    """
    vol: (D, H, W) np.float32
    - Z-score over non-zero voxels
    - Clip to [-clip_val, clip_val]
    - Rescale to [-1, 1]
    """
    mask = vol != 0  # ignore pure background
    if mask.any():
        vals = vol[mask]
        mean = vals.mean()
        std = vals.std()
        if std < eps:
            std = 1.0
        vol[mask] = (vol[mask] - mean) / std
    else:
        # fallback: z-score over all voxels
        mean = vol.mean()
        std = vol.std()
        if std < eps:
            std = 1.0
        vol = (vol - mean) / std

    # Clip extremes
    vol = np.clip(vol, -clip_val, clip_val)

    # Map [-clip_val, clip_val] -> [0, 1] -> [-1, 1]
    vol = (vol + clip_val) / (2.0 * clip_val)  # [0, 1]
    vol = vol * 2.0 - 1.0                      # [-1, 1]

    return vol

def _resize_volume(vol, target_shape):
    """
    vol: (C, D, H, W) numpy array
    target_shape: (D, H, W)
    """
    vol_t = torch.from_numpy(vol).unsqueeze(0).float()  # (1, C, D, H, W)
    td, th, tw = target_shape
    vol_t = F.interpolate(
        vol_t,
        size=(td, th, tw),
        mode="trilinear",
        align_corners=False,
    )
    vol = vol_t.squeeze(0).numpy()  # (C, D, H, W)
    return vol


class BraTS3DDataset(Dataset):
    """
    Loads BraTS 3D MRI volumes with configurable modalities.

    Returns:
      volume: torch.float32 tensor of shape (C, D, H, W)
      where C = number of modalities (1 for single modality, 4 for all)
    """

    def __init__(
            self,
            root_dir,
            patch_size=(64, 64, 64),
            modalities=("flair",),
            max_cases=None,
            xy_bbox: tuple[int, int, int, int] = None
    ):
        """
        root_dir: path to BraTS root (recursively searched for *_flair.nii.gz)
        patch_size: tuple (D, H, W) for 3D patches
        modalities: tuple of modality names to load (e.g., ('flair',) or ('flair', 't1'))
        max_cases: maximum number of cases to use (for debugging)
        augment: whether to apply data augmentation (random flips)
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.modalities = modalities
        self.max_cases = max_cases
        self.xy_bbox = xy_bbox

        self.cases = self._find_cases()
        if len(self.cases) == 0:
            raise ValueError(f"No BraTS cases found in {root_dir}")

        print(f"Found {len(self.cases)} BraTS subjects with {len(self.modalities)} modality(ies).")

    def _find_cases(self):
        cases = []
        flair_files = sorted(list(self.root_dir.rglob("*_flair.nii.gz")))
        if self.max_cases is not None:
            flair_files = flair_files[:self.max_cases]
        for flair_path in flair_files:
            flair_path = Path(flair_path)
            base = str(flair_path).replace("_flair.nii.gz", "")
            paths = {
                "flair": flair_path,
                "t1": Path(base + "_t1.nii.gz"),
                "t1ce": Path(base + "_t1ce.nii.gz"),
                "t2": Path(base + "_t2.nii.gz"),
            }

            if all(p.exists() for p in paths.values()):
                cases.append(tuple(paths[m] for m in self.modalities))
        return cases

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, paths):
        """
        paths: tuple of paths (one per modality)
        Returns np.array (C, D, H, W)
        """
        vols = []
        for p in paths:
            img = nib.load(str(p))
            vol = img.get_fdata().astype(np.float32)

            # Expect shape (H, W, D) -> reorder to (D, H, W)
            if vol.ndim == 4:
                # just in case, drop singleton
                vol = vol[..., 0]
            vol = np.transpose(vol, (2, 0, 1))  # (D, H, W)

            # If an in-plane bounding box is given, crop in (H, W)
            if self.xy_bbox is not None:
                y_min, y_max, x_min, x_max = self.xy_bbox
                # vol: (D, H, W) -> keep all D, crop H and W
                vol = vol[:, y_min:y_max, x_min:x_max]

            vol = _normalize_volume(vol)
            vols.append(vol)

        vol = np.stack(vols, axis=0)  # (C=4, D, H, W)

        vol = _resize_volume(vol, (128, 128, 192))

        return vol
    
    def __getitem__(self, idx):
        paths = self.cases[idx]
        vol = self._load_volume(paths)
        
        vol = torch.from_numpy(vol).float()  # (C, D, H, W)
        return vol

def create_dataset(root: Path, patch_size=(64, 64, 64), max_cases=None, modalities=("flair",), xy_bbox=None):
    """Create a BRATS 3D dataset."""
    d, h, w = 155, 160, 224
    xy_bbox = (10, 10 + h, 39, 39 + w) # y_min, y_max, x_min, x_max
    return BraTS3DDataset(
        root_dir=root,
        patch_size=(d, h, w),
        modalities=modalities,
        max_cases=max_cases,
    )