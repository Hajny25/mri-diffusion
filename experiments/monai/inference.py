import torch
from datetime import datetime
from pathlib import Path

import monai
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler
from monai.transforms import SaveImage

from scripts.ldm_sampler import LDMSampler


def main():
    # =========================================================================
    # Configuration Variables
    # =========================================================================
    bundle_root = "."
    model_dir = f"{bundle_root}/models"
    output_dir = f"{bundle_root}/output/monai/diffusion"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Output postfix with timestamp
    output_postfix = datetime.now().strftime("sample_%Y%m%d_%H%M%S")
    print(f"Output postfix: {output_postfix}")
    
    # Model parameters
    spatial_dims = 3
    image_channels = 1
    latent_channels = 8
    latent_shape = [8, 36, 44, 28]
    
    # =========================================================================
    # Build and Load AutoencoderKL
    # =========================================================================
    print("\nBuilding AutoencoderKL...")
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
    print(f"Loading autoencoder from: {load_autoencoder_path}")
    autoencoder_def.load_old_state_dict(
        torch.load(load_autoencoder_path, map_location=device)
    )
    autoencoder = autoencoder_def.to(device)
    autoencoder.eval()
    
    # =========================================================================
    # Build and Load DiffusionModelUNet
    # =========================================================================
    print("\nBuilding DiffusionModelUNet...")
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
    
    # Load diffusion model weights
    load_diffusion_path = f"{bundle_root}/models/diffusion/model_epoch=1890.pt"
    print(f"Loading diffusion model from: {load_diffusion_path}")
    network_def.load_old_state_dict(
        torch.load(load_diffusion_path, map_location=device)
    )
    diffusion = network_def.to(device)
    diffusion.eval()
    
    # =========================================================================
    # Setup Noise Scheduler (DDIM)
    # =========================================================================
    print("\nSetting up DDIM scheduler...")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )
    
    # Set inference timesteps (50 steps for faster sampling)
    noise_scheduler.set_timesteps(num_inference_steps=50)
    print(f"Inference timesteps: {len(noise_scheduler.timesteps)} steps")
    
    # =========================================================================
    # Generate Initial Noise
    # =========================================================================
    print("\nGenerating initial noise...")
    noise = torch.randn([1] + latent_shape).to(device)
    print(f"Noise shape: {noise.shape}")
    
    # =========================================================================
    # Setup Sampler and Saver
    # =========================================================================
    inferer = LDMSampler()
    
    saver = SaveImage(
        output_dir=output_dir,
        output_postfix=output_postfix,
        resample=False,
    )
    
    # =========================================================================
    # Run Sampling
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting Latent Diffusion Sampling")
    print("=" * 60 + "\n")
    
    images = inferer.run(
        noise,
        autoencoder,
        diffusion,
        noise_scheduler,
        saver
    )
    
    print(f"\nSamples saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()