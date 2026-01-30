from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_ROOT = Path(BASE_DIR / "data" / "preprocessed").expanduser()

NUM_SLICES = 155

class PreprocessedBraTSSliceDataset(Dataset):
    """
    Loads preprocessed .pt files created by preprocess_brats.py.

    Each .pt file contains:
      - "slices": tensor (num_slices, 1, H, W) in [-1, 1]
      - "z_pos": tensor (num_slices,) in [0, 1]
    """

    def __init__(self, root_dir=PREPROCESSED_ROOT, image_size=128):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # All preprocessed volume files
        volume_files = sorted(self.root_dir.rglob("*.pt"))
        if not volume_files:
            raise RuntimeError(f"No preprocessed .pt files found under {root_dir}")

        # Build global index: each dataset index -> (volume_idx, slice_idx)
        self.index = []

        print(f"Found {len(volume_files)} preprocessed volumes.")

        # We need to know how many slices in each file
        for file in volume_files:
            for s_idx in range(NUM_SLICES):
                if s_idx > 0:
                    self.index.append((file, s_idx, -1))
                if s_idx < NUM_SLICES - 1:
                    self.index.append((file, s_idx, +1))

        print(f"Total preprocessed slices: {len(self.index)}")

        # Optional small cache of last few loaded volumes
        self._cache = OrderedDict()
        self._cache_size = 4

    def __len__(self):
        return len(self.index)

    def _load_volume(self, path):
        path = str(path)
        if path in self._cache:
            vol = self._cache.pop(path)
            self._cache[path] = vol
            return vol

        vol = torch.load(path, map_location="cpu")  # dict with "slices" and "z_pos"
        self._cache[path] = vol
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def __getitem__(self, idx):
        path, context_idx, direction = self.index[idx]

        # neighbor = context_idx + direction, but clamp to valid [0, num_slices-1]
        neighbor_idx = context_idx + direction
        neighbor_idx = max(0, min(neighbor_idx, NUM_SLICES - 1))

        data = self._load_volume(path)
        slices = data["slices"]     # (num_slices, 1, H, W)

        context_tensor = slices[context_idx]       # (1, H, W), already in [-1, 1]
        neighbor_tensor = slices[neighbor_idx]       # (1, H, W), already in [-1, 1]

        return {
            "context": context_tensor,        # slice k
            "neighbor": neighbor_tensor,    # slice k-1 or k+1 (clamped at boundaries)
            "direction": direction,         # -1 for above, +1 for below
            "slice_pos": context_idx / NUM_SLICES - 1,        
        }