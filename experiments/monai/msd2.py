from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

data_dir = "data/brats-2021-msd"

basic_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
])

ds = DecathlonDataset(
    root_dir=data_dir,
    task="Task01_BrainTumour",
    section="training",
    transform=basic_transforms,
    cache_rate=0.0,
    num_workers=0,
    download=False,
    
)

print(len(ds))
example = ds[0]
print(example["image"].shape, example["label"].shape)