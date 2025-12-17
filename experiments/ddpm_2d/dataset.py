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
    slices_per_case: int = 12
    modalities: tuple[str] = ("flair",)


config = DataConfig()


class BratsSliceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
            image_size: int,
            slice_axis: int = 2,
            slices_per_case: int = 12,
            modalities: tuple[str, ...] = ("flair",),
            max_cases: int = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.slice_axis = slice_axis
        self.slices_per_case = slices_per_case
        self.modalities = modalities
        self.max_cases = max_cases
        self.samples = self._index_cases()
        if not self.samples:
            raise RuntimeError(f"No BRATS2021 samples found under {self.root_dir}")

    def _index_cases(self) -> list[tuple[Path, int]]:
        entries = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if self.max_cases is not None:
            entries = entries[: self.max_cases]
        samples: list[tuple[Path, int]] = []
        for case_dir in entries:
            ref_path = self._resolve_modality_path(case_dir, self.modalities[0])
            ref_data = nib.load(ref_path).get_fdata()
            num_slices = ref_data.shape[self.slice_axis]
            center_slice = num_slices // 2
            offset = self.slices_per_case // 2
            usable = range(center_slice - offset, center_slice + offset)
            slice_ids = np.linspace(
                usable.start,
                usable.stop - 1,
                num=min(self.slices_per_case, len(usable)),
                dtype=int,
            )
            samples.extend((case_dir, int(idx)) for idx in slice_ids)
        return samples

    def _resolve_modality_path(self, case_dir: Path, modality: str) -> Path:
        matches = sorted(case_dir.glob(f"*{modality}.nii*"))
        if not matches:
            raise FileNotFoundError(f"Missing modality '{modality}' in {case_dir}")
        return matches[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_dir, slice_idx = self.samples[index]
        modality_slices = []
        for modality in self.modalities:
            volume = nib.load(self._resolve_modality_path(case_dir, modality)).get_fdata()
            slice_2d = np.take(volume, indices=slice_idx, axis=self.slice_axis).astype(np.float32)
            modality_slices.append(slice_2d)
        slice_2d = np.mean(modality_slices, axis=0)
        slice_2d -= slice_2d.min()
        max_val = slice_2d.max()
        if max_val > 0:
            slice_2d /= max_val
        pil_img = Image.fromarray((slice_2d * 255).astype(np.uint8))

        preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        tensor = preprocess(pil_img)
        return tensor


def create_dataset(brats_root: Path, image_size: int, debug: bool):
    return BratsSliceDataset(
        root_dir=brats_root,
        image_size=image_size,
        slice_axis=config.slice_axis,
        slices_per_case=config.slices_per_case,
        modalities=config.modalities,
        max_cases=None if not debug else 1
    )
