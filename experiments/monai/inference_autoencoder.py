import torch
from datetime import datetime
from pathlib import Path

from monai.networks.nets import AutoencoderKL
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
    SaveImage,
)
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.utils import first


def main():
    # =========================================================================
    # Configuration Variables
    # =========================================================================
    bundle_root = "."
    model_dir = f"{bundle_root}/models"
    dataset_dir = "data/brats-2021-msd"
    output_dir = f"{bundle_root}/output/monai/autoencoder"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Output postfixes
    output_orig_postfix = "recon"
    output_recon_postfix = "orig"
    
    # Model parameters
    channel = 0
    spacing = [1.1, 1.1, 1.1]
    spatial_dims = 3
    image_channels = 1
    latent_channels = 8
    infer_patch_size = [144, 176, 112]
    
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
    print(f"Loading weights from: {load_autoencoder_path}")
    autoencoder_def.load_old_state_dict(torch.load(load_autoencoder_path, map_location=device))
    
    # Move to device
    autoencoder = autoencoder_def.to(device)
    autoencoder.eval()
    
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
        CenterSpatialCropd(keys="image", roi_size=infer_patch_size),
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
    
    preprocessing = Compose(preprocessing_transforms + crop_transforms + final_transforms)
    
    # =========================================================================
    # Build Dataset and DataLoader
    # =========================================================================
    print("Building dataset...")
    dataset = DecathlonDataset(
        root_dir=dataset_dir,
        task="Task01_BrainTumour",
        section="validation",
        cache_rate=0.0,
        num_workers=8,
        download=False,
        transform=preprocessing,
    )
    
    print("Building dataloader...")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    
    # =========================================================================
    # Build Image Savers
    # =========================================================================
    saver_orig = SaveImage(
        output_dir=output_dir,
        output_postfix=output_orig_postfix,
        resample=False,
        padding_mode="zeros",
    )
    
    saver_recon = SaveImage(
        output_dir=output_dir,
        output_postfix=output_recon_postfix,
        resample=False,
        padding_mode="zeros",
    )
    
    # =========================================================================
    # Run Inference
    # =========================================================================
    print("Running inference...")
    
    with torch.no_grad():
        # Get first batch
        input_img = first(dataloader)["image"].to(device)
        print(f"Input image shape: {input_img.shape}")
        
        # Run autoencoder reconstruction
        recon_img = autoencoder(input_img)[0][0]
        print(f"Reconstructed image shape: {recon_img.shape}")
        
        # Save original and reconstructed images
        saver_orig(input_img[0][0])
        saver_recon(recon_img)
    
    print(f"Images saved to: {output_dir}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()