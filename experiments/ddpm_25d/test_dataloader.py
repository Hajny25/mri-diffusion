import time
from pathlib import Path
import argparse
import torch

from preprocessed_dataset import PreprocessedBraTSSliceDataset
from mm_dataset import MemmapDataset

BASE_DIR = Path(__file__).resolve().parents[2]
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()
PREPROCESSED_ROOT = Path(BASE_DIR / "data" / "preprocessed").expanduser()

BATCH_SIZE = 128
REPETITIONS = 5
ITERATIONS = 20

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of workers for dataloader",
)
args = parser.parse_args()
workers = args.workers

dataset = PreprocessedBraTSSliceDataset(PREPROCESSED_ROOT, 128)
# dataset = MemmapDataset(BASE_DIR / "data" / "preprocessed_all.npy")

for _ in range(REPETITIONS):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        shuffle=True,
        persistent_workers=False,
        pin_memory=True,
    )
    dataiter = iter(dataloader)
    t0 = time.time()
    batch = next(dataiter)
    t1 = time.time()
    print("First batch load:", t1 - t0, "s")

    for i, batch in enumerate(dataloader):
        if i == ITERATIONS:
            break
    t2 = time.time()
    print("workers:", workers, f", {ITERATIONS} batches:", t2 - t1, "s → per batch ~", (t2 - t1) / ITERATIONS, "s")