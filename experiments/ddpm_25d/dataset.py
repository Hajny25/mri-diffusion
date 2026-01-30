from dataclasses import dataclass
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

@dataclass
class DataConfig:
    slice_axis: int = 2
    modalities: tuple[str] = ("flair",)

config = DataConfig()

class BratsSlicePairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Path,
        image_size: int,
        slice_axis: int = config.slice_axis,
        modalities: tuple[str, ...] = config.modalities,
        max_cases: int = None,
        debug: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.slice_axis = slice_axis
        self.modalities = modalities
        self.max_cases = max_cases
        self.debug = debug

        self.samples = self._index_cases()
        if not self.samples:
            raise RuntimeError(f"No BRATS2021 samples found under {self.root_dir}")

        # common preprocessing pipeline
        crop_top = 32
        crop_left = 8
        crop_height = 168
        crop_width = 224
        self.preprocess = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: transforms.functional.crop(
                        img, crop_top, crop_left, crop_height, crop_width
                    )
                ),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _index_cases(self) -> list[tuple[Path, int, int]]:
        """
        Returns list of (case_dir, slice_idx, direction),
        where direction ∈ {-1, +1}, meaning 'above' or 'below',
        for *all* slices along slice_axis.
        """
        entries = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if self.max_cases is not None:
            entries = entries[:self.max_cases]

        samples: list[tuple[Path, int, int]] = []
        for case_dir in entries:
            ref_path = self._resolve_modality_path(case_dir, self.modalities[0])
            ref_data = nib.load(ref_path).get_fdata()
            num_slices = ref_data.shape[self.slice_axis]
            # if self.debug:
            #     num_slices = 1

            # cover all slices 0 .. num_slices-1
            for idx in range(num_slices):
                samples.append((case_dir, idx, -1))  # neighbor above
                samples.append((case_dir, idx, +1))  # neighbor below

        return samples

    def _resolve_modality_path(self, case_dir: Path, modality: str) -> Path:
        matches = sorted(case_dir.glob(f"*{modality}.nii*"))
        if not matches:
            raise FileNotFoundError(f"Missing modality '{modality}' in {case_dir}")
        return matches[0]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_slice(self, case_dir: Path, slice_idx: int) -> np.ndarray:
        """Load and average all modalities for a single slice index, returns np.array (H, W)."""
        modality_slices = []
        for modality in self.modalities:
            volume = nib.load(self._resolve_modality_path(case_dir, modality)).get_fdata()
            slice_2d = np.take(volume, indices=slice_idx, axis=self.slice_axis).astype(np.float32)
            modality_slices.append(slice_2d)
        slice_2d = np.mean(modality_slices, axis=0)
        return slice_2d

    def _normalize_and_to_tensor(self, slice_2d: np.ndarray) -> torch.Tensor:
        slice_2d = slice_2d.copy()
        slice_2d -= slice_2d.min()
        max_val = slice_2d.max()
        if max_val > 0:
            slice_2d /= max_val
        pil_img = Image.fromarray((slice_2d * 255).astype(np.uint8))
        return self.preprocess(pil_img)

    def _load_slice_pair(self, case_dir, slice_idx, neighbor_slice_idx):
        modality_volumes = []
        for modality in self.modalities:
            volume = nib.load(self._resolve_modality_path(case_dir, modality)).get_fdata()
            modality_volumes.append(volume)

        volume = np.mean(modality_volumes, axis=0)
        #slices = np.take(volume, indices=[slice_idx, neighbor_slice_idx], axis=self.slice_axis).astype(np.float32)
        slice = volume[:, :, slice_idx]
        neighbor_slice = volume[:, :, neighbor_slice_idx]
        return slice, neighbor_slice



    def __getitem__(self, index: int): #-> dict[str, torch.Tensor | int]:
        case_dir, context_idx, direction = self.samples[index]

        num_slices = 155

        # neighbor = center_idx + direction, but clamp to valid [0, num_slices-1]
        neighbor_idx = context_idx + direction
        neighbor_idx = max(0, min(neighbor_idx, num_slices - 1))

        # Load slices
        context_slaice_np, neighbor_slice_np = self._load_slice_pair(case_dir, context_idx, neighbor_idx)

        # To tensors
        context_tensor = self._normalize_and_to_tensor(context_slaice_np)
        neighbor_tensor = self._normalize_and_to_tensor(neighbor_slice_np)

        return {
            "context": context_tensor,        # slice k
            "neighbor": neighbor_tensor,    # slice k-1 or k+1 (clamped at boundaries)
            "direction": direction,         # -1 for above, +1 for below
            "slice_pos": context_idx / num_slices,        
        }


def create_dataset(brats_root: Path, image_size: int, debug: bool):
    return BratsSlicePairDataset(
        root_dir=brats_root,
        image_size=image_size,
        max_cases=None if not debug else 1,
        debug=debug
    )