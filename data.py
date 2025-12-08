import numpy as np
import nibabel as nib
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path


class BratsSliceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
            transforms: transforms.Compose,
            slice_axis: int = 2,
            slices_per_case: int = 12,
            slice_margin: int = 8,
            modalities: tuple[str, ...] = ("flair",),
            max_cases: int = None,
    ):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.slice_axis = slice_axis
        self.slices_per_case = slices_per_case
        self.slice_margin = slice_margin
        self.modalities = modalities
        self.max_cases = max_cases
        self.samples = self._index_cases()
        if not self.samples:
            raise RuntimeError(f"No BRATS2021 samples found under {self.root_dir}")

    def _index_cases(self) -> list[tuple[Path, int]]:
        print("Indexing BRATS2021 cases...")
        entries = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if self.max_cases is not None:
            entries = entries[: self.max_cases]
        samples: list[tuple[Path, int]] = []
        for case_dir in entries:
            ref_path = self._resolve_modality_path(case_dir, self.modalities[0])
            ref_data = nib.load(ref_path).get_fdata()
            num_slices = ref_data.shape[self.slice_axis]
            usable = range(self.slice_margin, max(self.slice_margin + 1, num_slices - self.slice_margin))
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
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        pil_img = Image.fromarray((slice_rgb * 255).astype(np.uint8))
        tensor = self.transforms(pil_img) if self.transforms else transforms.ToTensor()(pil_img)
        return {"images": tensor}
