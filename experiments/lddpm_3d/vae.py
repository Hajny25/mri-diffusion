"""
3D Variational Autoencoder for BRATS MRI volumes.
Compresses 3D volumes into a lower-dimensional latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual block with group normalization."""
    
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = F.silu(out)
        
        return out + residual


class Encoder3D(nn.Module):
    """3D CNN Encoder with downsampling."""
    
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        latent_channels=4,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Build downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            
            # Residual blocks at this resolution
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock3D(ch, out_ch))
                ch = out_ch
            
            # Downsample (except last level)
            if mult != channel_multipliers[-1]:
                self.down_blocks.append(
                    nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
        
        # Middle block
        self.mid_block = nn.Sequential(
            ResidualBlock3D(ch, ch),
            ResidualBlock3D(ch, ch),
        )
        
        # Output projection to latent space
        self.conv_out = nn.Conv3d(ch, 2 * latent_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial conv
        x = self.conv_in(x)
        
        # Downsample through blocks
        for block in self.down_blocks:
            x = block(x)
        
        # Middle
        x = self.mid_block(x)
        
        # Project to latent
        x = self.conv_out(x)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(x, 2, dim=1)
        
        return mean, logvar


class Decoder3D(nn.Module):
    """3D CNN Decoder with upsampling."""
    
    def __init__(
        self,
        out_channels=1,
        base_channels=32,
        latent_channels=4,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
    ):
        super().__init__()
        
        # Reverse the channel multipliers for upsampling
        channel_multipliers = list(reversed(channel_multipliers))
        
        # Start with highest channel count
        ch = base_channels * channel_multipliers[0]
        
        # Input projection from latent
        self.conv_in = nn.Conv3d(latent_channels, ch, kernel_size=3, padding=1)
        
        # Middle block
        self.mid_block = nn.Sequential(
            ResidualBlock3D(ch, ch),
            ResidualBlock3D(ch, ch),
        )
        
        # Build upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            # Residual blocks at this resolution
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock3D(ch, out_ch))
                ch = out_ch
            
            # Upsample (except last level)
            if i != len(channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1)
                )
        
        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, z):
        # Project from latent
        x = self.conv_in(z)
        
        # Middle
        x = self.mid_block(x)
        
        # Upsample through blocks
        for block in self.up_blocks:
            x = block(x)
        
        # Output
        x = self.conv_out(x)
        
        return x


class VAE3D(nn.Module):
    """3D Variational Autoencoder for MRI volumes."""
    
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        latent_channels=4,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
    ):
        super().__init__()
        
        self.encoder = Encoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )
        
        self.decoder = Decoder3D(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )
        
        self.latent_channels = latent_channels
        
    def reparameterize(self, mean, logvar):
        """Reparameterization trick: z = mean + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        mean, logvar = self.encoder(x)
        # Clamp logvar to prevent KL explosion
        #logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mean, logvar
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x, sample_posterior=True):
        """
        Forward pass through VAE.
        
        Args:
            x: Input volume (B, C, D, H, W)
            sample_posterior: If True, sample from posterior. If False, use mean.
        
        Returns:
            reconstruction: Reconstructed volume
            mean: Latent mean
            logvar: Latent log variance
        """
        mean, logvar = self.encode(x)
        
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        
        reconstruction = self.decode(z)
        
        return reconstruction, mean, logvar
    
    def get_latent(self, x):
        """Encode input to latent representation (using mean, no sampling)."""
        mean, _ = self.encode(x)
        return mean


def vae_loss(reconstruction, target, mean, logvar, kl_weight=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence.
    
    Args:
        reconstruction: Reconstructed volume
        target: Original volume
        mean: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='mean')
    
    # Clamp logvar to prevent numerical issues (safety net)
    #logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=(1, 2, 3, 4)).mean()
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


# Example configurations
def get_vae_tiny():
    """Tiny VAE for testing (fast, low memory)."""
    return VAE3D(
        in_channels=1,
        base_channels=16,
        latent_channels=2,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
    )


def get_vae_small():
    """Small VAE (good balance for 64^3 or 96^3 volumes)."""
    return VAE3D(
        in_channels=1,
        base_channels=32,
        latent_channels=4,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
    )


def get_vae_base():
    """Base VAE (standard config for 128^3 volumes)."""
    return VAE3D(
        in_channels=1,
        base_channels=32,
        latent_channels=4,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
    )


def get_vae_large():
    """Large VAE (high capacity for detailed reconstruction)."""
    return VAE3D(
        in_channels=1,
        base_channels=64,
        latent_channels=8,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=3,
    )


if __name__ == "__main__":
    # Test the VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    vae = get_vae_small().to(device)
    
    # Test input (batch=2, channels=1, depth=64, height=64, width=64)
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    
    # Forward pass
    print("Input shape:", x.shape)
    reconstruction, mean, logvar = vae(x)
    print("Reconstruction shape:", reconstruction.shape)
    print("Latent mean shape:", mean.shape)
    print("Latent logvar shape:", logvar.shape)
    
    # Calculate loss
    loss, recon_loss, kl_loss = vae_loss(reconstruction, x, mean, logvar)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Check latent compression ratio
    latent_size = mean.numel()
    input_size = x.numel()
    compression_ratio = input_size / latent_size
    print(f"\nCompression ratio: {compression_ratio:.2f}x")
    print(f"Latent spatial size: {mean.shape[2:5]}")
