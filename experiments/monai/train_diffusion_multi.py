import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path

import monai
import scripts
from monai.networks.nets import DiffusionModelUNet, AutoencoderKL
from monai.networks.schedulers import DDPMScheduler
from monai.inferers import LatentDiffusionInferer
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Lambdad,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Compose,
)
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.handlers import (
    LrScheduleHandler,
    CheckpointSaver,
    StatsHandler,
    TensorBoardStatsHandler,
    from_engine,
)
from monai.utils import set_determinism, first
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
print(f"{local_rank=}")
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')

# =========================================================================
# Utility Functions
# =========================================================================
def compute_scale_factor(
    autoencoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute scale factor for latent space normalization.
    
    The scale factor normalizes the latent representations to have 
    unit standard deviation, which stabilizes diffusion model training.
    """
    print("Computing scale factor...")
    
    autoencoder.eval()
    with torch.no_grad():
        batch = first(dataloader)
        images = batch["image"].to(device)
        z = autoencoder.encode_stage_2_inputs(images)
        scale_factor = 1.0 / z.flatten().std().item()
    
    return scale_factor


# =========================================================================
# Main Training Script
# =========================================================================
def main():
    # =========================================================================
    # Initialize
    # =========================================================================
    set_determinism(seed=0)
    
    # =========================================================================
    # Configuration Variables
    # =========================================================================
    bundle_root = "."
    ckpt_dir = f"{bundle_root}/models/diffusion"
    dataset_dir = "data/brats-2021-msd"
    tf_dir = f"{bundle_root}/logs"
    
    # Create directories
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(tf_dir).mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    train_batch_size = 4
    lr = 4e-05
    train_patch_size = [144, 176, 112]
    
    # Model parameters (from inference config)
    channel = 0
    spacing = [1.1, 1.1, 1.1]
    spatial_dims = 3
    image_channels = 1
    latent_channels = 8
    latent_shape = [latent_channels, 36, 44, 28]
    
    # =========================================================================
    # Build AutoencoderKL Model
    # =========================================================================
    print("Building AutoencoderKL model...")
    autoencoder_def = AutoencoderKL(
        spatial_dims=spatial_dims,
        in_channels=image_channels,
        out_channels=image_channels,
        latent_channels=latent_channels,
        channels=[64, 128, 256],
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        include_fc=False,
    )
    
    # Load autoencoder weights
    load_autoencoder_path = f"{bundle_root}/models/autoencoder/model_epoch=1498.pt"
    print(f"Loading autoencoder weights from: {load_autoencoder_path}")
    autoencoder_def.load_old_state_dict(
        torch.load(load_autoencoder_path, map_location=device)
    )
    autoencoder = autoencoder_def.to(device)
    autoencoder.eval()  # Freeze autoencoder
    
    # =========================================================================
    # Build DiffusionModelUNet
    # =========================================================================
    print("Building DiffusionModelUNet...")
    network_def = DiffusionModelUNet(
        spatial_dims=spatial_dims,
        in_channels=latent_channels,
        out_channels=latent_channels,
        channels=[256, 256, 512],
        attention_levels=[False, True, True],
        num_head_channels=[0, 64, 64],
        num_res_blocks=2,
        include_fc=False,
        use_combined_linear=False,
    )
    diffusion = torch.nn.parallel.DistributedDataParallel(network_def.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # =========================================================================
    # Build Optimizer and Learning Rate Scheduler
    # =========================================================================
    print("Building optimizer and scheduler...")
    optimizer = Adam(params=diffusion.parameters(), lr=lr)
    
    lr_scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=[100, 1000],
        gamma=0.1,
    )
    
    # =========================================================================
    # Build Noise Scheduler
    # =========================================================================
    print("Building noise scheduler...")
    noise_scheduler = DDPMScheduler(
        schedule="scaled_linear_beta",
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
    )
    
    # =========================================================================
    # Build Preprocessing Pipeline
    # =========================================================================
    print("Building preprocessing pipeline...")
    
    preprocessing_transforms = [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
        EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=spacing, mode="bilinear"),
    ]
    
    crop_transforms = [
        CenterSpatialCropd(keys="image", roi_size=train_patch_size),
    ]
    
    final_transforms = [
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0,
            upper=99.5,
            b_min=0,
            b_max=1,
        ),
    ]
    
    preprocessing = Compose(
        preprocessing_transforms + crop_transforms + final_transforms
    )
    
    # =========================================================================
    # Build Dataset and DataLoader
    # =========================================================================
    print("Building training dataset...")
    train_dataset = DecathlonDataset(
        root_dir=dataset_dir,
        task="Task01_BrainTumour",
        section="training",
        cache_rate=0.0,
        num_workers=8,
        download=False,
        transform=preprocessing,
    )
    
    print("Building dataloader...")
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=8,
    )
    
    # =========================================================================
    # Compute Scale Factor
    # =========================================================================
    scale_factor = compute_scale_factor(autoencoder, train_dataloader, device)
    print(f"scale factor: {scale_factor}")
    
    # =========================================================================
    # Build Inferer
    # =========================================================================
    print("Building LatentDiffusionInferer...")
    inferer = LatentDiffusionInferer(
        scheduler=noise_scheduler,
        scale_factor=scale_factor,
    )
    
    # =========================================================================
    # Build Loss Function
    # =========================================================================
    loss_function = nn.MSELoss()
    
    # =========================================================================
    # Build Handlers
    # =========================================================================
    print("Building training handlers...")
    
    handlers = [
        LrScheduleHandler(
            lr_scheduler=lr_scheduler,
            print_lr=True,
        ),
        CheckpointSaver(
            save_dir=ckpt_dir,
            save_dict={"model": diffusion},
            save_interval=10,
            save_final=True,
            epoch_level=True,
            final_filename="model2.pt",
        ),
        StatsHandler(
            tag_name="train_diffusion_loss",
            output_transform=lambda x: from_engine(["loss"], first=True)(x),
        ),
        TensorBoardStatsHandler(
            log_dir=tf_dir,
            tag_name="train_diffusion_loss",
            output_transform=lambda x: from_engine(["loss"], first=True)(x),
        ),
    ][: -2 if dist.get_rank() > 0 else None]
    
    # =========================================================================
    # Build and Run Trainer
    # =========================================================================
    print("Building trainer...")
    trainer = scripts.ldm_trainer.LDMTrainer(
        device=device,
        max_epochs=2500,
        train_data_loader=train_dataloader,
        network=diffusion,
        autoencoder_model=autoencoder,
        optimizer=optimizer,
        loss_function=loss_function,
        latent_shape=latent_shape,
        inferer=inferer,
        key_train_metric=None,
        train_handlers=handlers
    )
    
    # =========================================================================
    # Run Training
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting LDM Training")
    print("=" * 60)
    import logging
    trainer.logger.setLevel(logging.WARNING if dist.get_rank() > 0 else logging.INFO)
    trainer.run()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

if dist.is_initialized():
    dist.destroy_process_group()