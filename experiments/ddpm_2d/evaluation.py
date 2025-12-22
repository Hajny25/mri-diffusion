import os
import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
from accelerate import Accelerator
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
import matplotlib.pyplot as plt

from dataset import create_dataset


BASE_DIR = Path(__file__).resolve().parents[2]
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()


class DiffusionModelEvaluator:
    def __init__(self, checkpoint_path, accelerator, seed=42, num_train_timesteps=1000, num_inference_steps=1000):
        """
        Initialize the evaluator with a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint directory
            accelerator: Accelerator instance for distributed evaluation
            seed: Random seed for reproducibility
            num_train_timesteps: Number of training timesteps used during training
            num_inference_steps: Number of inference steps for generation
        """
        self.accelerator = accelerator
        self.device = accelerator.device
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        
        accelerator.print(f"Loading model from {checkpoint_path}...")
        model = UNet2DModel.from_pretrained(checkpoint_path)
        model.eval()
        
        # Prepare model with accelerator for multi-GPU support
        self.model = accelerator.prepare(model)
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.pipeline = DDPMPipeline(unet=accelerator.unwrap_model(self.model), scheduler=self.noise_scheduler)
        
        accelerator.print(f"Model loaded successfully on {self.device}")
        accelerator.print(f"Number of processes: {accelerator.num_processes}")
        
        # Initialize inception model for FID calculation
        self.inception_model = None
        
    def generate_samples(self, num_samples, batch_size=16, image_size=128):
        """
        Generate synthetic samples from the trained model.
        For multi-GPU: each process generates independently, main process collects all.
        
        Args:
            num_samples: Total number of samples to generate across all processes
            batch_size: Batch size for generation per process
            image_size: Size of generated images
            
        Returns:
            List of PIL Images (all samples on main process, local samples on other processes)
        """
        # Calculate samples per process (same for all to use gather)
        samples_per_process = math.ceil(num_samples / self.accelerator.num_processes)
        
        self.accelerator.print(f"Process {self.accelerator.process_index}: Generating {samples_per_process} samples...")
        local_images = []
        
        num_batches = math.ceil(samples_per_process / batch_size)
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, samples_per_process - i * batch_size)
            
            # Use different seed for each process to ensure diverse samples
            seed_offset = self.accelerator.process_index * samples_per_process + i * batch_size
            
            with torch.no_grad():
                images = self.pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device=self.device).manual_seed(self.seed + seed_offset),
                    num_inference_steps=self.num_inference_steps,
                ).images
                
            local_images.extend(images)
            
            if (i + 1) % 5 == 0 or i == num_batches - 1:
                self.accelerator.print(f"Process {self.accelerator.process_index}: Generated {len(local_images)}/{samples_per_process} samples")
        
        # Convert PIL images to tensors for gathering
        local_tensors = []
        for img in local_images:
            img_array = np.array(img).astype(np.float32) / 255.0
            local_tensors.append(torch.from_numpy(img_array))
        
        # Stack into a single tensor [num_local_samples, H, W]
        local_tensor = torch.stack(local_tensors).to(self.device)
        
        # Gather all tensors from all processes
        all_tensors = self.accelerator.gather(local_tensor)
        
        # Convert back to PIL images (only matters on main process)
        if self.accelerator.is_main_process:
            all_images = []
            all_tensors_cpu = all_tensors.cpu().numpy()
            for img_array in all_tensors_cpu:
                img_uint8 = (img_array * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                all_images.append(pil_img)
            return all_images[:num_samples]
        else:
            return local_images
    
    def images_to_tensor(self, images):
        """Convert PIL images to normalized tensors."""
        tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                # Convert to numpy array and normalize to [-1, 1]
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = (img_array - 0.5) / 0.5
                tensors.append(torch.from_numpy(img_array).unsqueeze(0))
            else:
                tensors.append(img)
        return torch.stack(tensors)
    
    def get_inception_features(self, images):
        """Extract features from images using InceptionV3."""
        if self.inception_model is None:
            self.accelerator.print("Loading Inception V3 model for FID calculation...")
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = torch.nn.Identity()  # Remove final classification layer
            self.inception_model = self.inception_model.to(self.device)
            self.inception_model.eval()
        
        # Convert images to tensors and resize to 299x299 (Inception input size)
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        features = []
        batch_size = 32
        num_batches = math.ceil(len(images) / batch_size)
        
        self.accelerator.print(f"Extracting Inception features from {len(images)} images ({num_batches} batches)...")
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch]).to(self.device)
            
            with torch.no_grad():
                batch_features = self.inception_model(batch_tensors)
            
            features.append(batch_features.cpu().numpy())
            
            # Log progress every 10 batches or at the end
            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                self.accelerator.print(f"  Processed {batch_idx}/{num_batches} batches ({i + len(batch)}/{len(images)} images)")
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid(self, generated_images, real_images):
        """
        Calculate Fréchet Inception Distance (FID) between generated and real images.
        
        Args:
            generated_images: List of generated PIL Images
            real_images: List of real PIL Images
            
        Returns:
            FID score
        """
        self.accelerator.print("\nCalculating FID...")
        
        # Get features
        self.accelerator.print("  Extracting features from generated images...")
        gen_features = self.get_inception_features(generated_images)
        self.accelerator.print("  Extracting features from real images...")
        real_features = self.get_inception_features(real_images)
        
        # Calculate mean and covariance
        self.accelerator.print("  Computing statistics and FID score...")
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        # Calculate FID
        diff = mu_gen - mu_real
        covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2 * covmean)
        
        self.accelerator.print(f"  FID: {fid:.4f}")
        return fid
    
    def calculate_kid(self, generated_images, real_images, subset_size=100):
        """
        Calculate Kernel Inception Distance (KID) between generated and real images.
        KID is more robust than FID for small sample sizes.
        
        Args:
            generated_images: List of generated PIL Images
            real_images: List of real PIL Images
            subset_size: Size of subsets for computing KID (default: 100)
            
        Returns:
            KID score (multiplied by 1000 for readability)
        """
        self.accelerator.print("\nCalculating KID...")
        
        # Get features
        self.accelerator.print("  Extracting features from generated images...")
        gen_features = self.get_inception_features(generated_images)
        self.accelerator.print("  Extracting features from real images...")
        real_features = self.get_inception_features(real_images)
        
        # Convert to torch tensors
        self.accelerator.print("  Computing kernel matrices...")
        gen_features = torch.from_numpy(gen_features).float()
        real_features = torch.from_numpy(real_features).float()
        
        # Use subset if dataset is large
        n_gen = min(subset_size, len(gen_features))
        n_real = min(subset_size, len(real_features))
        
        gen_features = gen_features[:n_gen]
        real_features = real_features[:n_real]
        
        # Polynomial kernel: (x^T y / d + 1)^3
        def polynomial_kernel(X, Y):
            """Compute polynomial kernel between X and Y."""
            d = X.shape[1]
            dot_products = torch.mm(X, Y.t()) / d
            return (dot_products + 1) ** 3
        
        # Compute kernel matrices
        K_XX = polynomial_kernel(gen_features, gen_features)
        K_YY = polynomial_kernel(real_features, real_features)
        K_XY = polynomial_kernel(gen_features, real_features)
        
        # Remove diagonal elements (self-similarity) for unbiased estimate
        self.accelerator.print("  Computing MMD statistic...")
        m = K_XX.shape[0]
        n = K_YY.shape[0]
        
        # MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
        # Unbiased estimator (exclude diagonal)
        k_xx = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1))
        k_yy = (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1))
        k_xy = K_XY.sum() / (m * n)
        
        kid = k_xx + k_yy - 2 * k_xy
        kid = kid.item()
        
        # Multiply by 1000 for readability (standard practice)
        kid_1000 = kid * 1000
        
        self.accelerator.print(f"  KID: {kid_1000:.4f} (×10^-3)")
        return kid_1000
    
    def calculate_diversity(self, images):
        """
        Calculate diversity metrics for generated images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary with diversity metrics
        """
        self.accelerator.print("\nCalculating diversity metrics...")
        
        # Convert images to tensors
        tensors = []
        for img in images:
            img_array = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(img_array))
        
        tensors = torch.stack(tensors)
        
        # Flatten images
        flat_tensors = tensors.reshape(len(tensors), -1)
        
        # Calculate pairwise distances
        self.accelerator.print("  Computing pairwise distances...")
        distances = []
        num_samples = len(tensors)
        
        for i in range(min(100, num_samples)):  # Sample pairs to avoid O(n^2) computation
            for j in range(i + 1, min(100, num_samples)):
                dist = torch.norm(flat_tensors[i] - flat_tensors[j]).item()
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Calculate variance across the dataset
        self.accelerator.print("  Computing pixel variance...")
        pixel_variance = torch.var(tensors, dim=0).mean().item()
        
        self.accelerator.print(f"  Average pairwise distance: {avg_distance:.4f} ± {std_distance:.4f}")
        self.accelerator.print(f"  Average pixel variance: {pixel_variance:.6f}")
        
        return {
            'avg_pairwise_distance': avg_distance,
            'std_pairwise_distance': std_distance,
            'pixel_variance': pixel_variance
        }
    
    def save_sample_grid(self, images, output_path, grid_size=4):
        """
        Save a grid of sample images.
        
        Args:
            images: List of PIL Images
            output_path: Path to save the grid
            grid_size: Number of images per row/column
        """
        num_images = min(grid_size * grid_size, len(images))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle('Generated MRI Slices', fontsize=16)
        
        for idx in range(num_images):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col]
            
            ax.imshow(images[idx], cmap='gray')
            ax.axis('off')
        
        # Hide any unused subplots
        for idx in range(num_images, grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Sample grid saved to {output_path}")
        plt.close()
    
    def compare_samples(self, generated_images, real_images, output_path, num_pairs=4):
        """
        Create a side-by-side comparison of generated and real images.
        
        Args:
            generated_images: List of generated PIL Images
            real_images: List of real PIL Images
            output_path: Path to save the comparison
            num_pairs: Number of pairs to show
        """
        fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 4 * num_pairs))
        fig.suptitle('Generated vs Real MRI Slices', fontsize=16)
        
        for i in range(num_pairs):
            # Generated image
            axes[i, 0].imshow(generated_images[i], cmap='gray')
            axes[i, 0].set_title('Generated')
            axes[i, 0].axis('off')
            
            # Real image
            axes[i, 1].imshow(real_images[i], cmap='gray')
            axes[i, 1].set_title('Real')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {output_path}")
        plt.close()


def load_real_images(dataset, num_images):
    """
    Load real images from the dataset.
    
    Args:
        dataset: PyTorch dataset
        num_images: Number of images to load
        
    Returns:
        List of PIL Images
    """
    # Note: This should only be called on main process
    print(f"Loading {num_images} real images from dataset...")
    
    images = []
    indices = np.random.RandomState(42).choice(len(dataset), size=min(num_images, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        tensor = dataset[idx]
        
        # Denormalize from [-1, 1] to [0, 255]
        img_array = ((tensor.squeeze().numpy() + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
        
        # Log progress every 20 images
        if (i + 1) % 20 == 0 or (i + 1) == len(indices):
            print(f"  Loaded {i + 1}/{len(indices)} images")
    
    return images


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained diffusion model on BRATS2021 data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory (e.g., output/run_id/best_model)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation per GPU')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (must match training)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_train_timesteps', type=int, default=1000,
                        help='Number of training timesteps (must match training)')
    parser.add_argument('--num_inference_steps', type=int, default=1000,
                        help='Number of inference steps for generation')
    parser.add_argument('--skip_fid', action='store_true',
                        help='Skip FID calculation (requires downloading inception model)')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        help='Mixed precision mode (fp16, bf16, or no)')
    
    args = parser.parse_args()
    
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    accelerator.print("="*60)
    accelerator.print("DIFFUSION MODEL EVALUATION")
    accelerator.print("="*60)
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {args.mixed_precision}")
    
    # Create output directory (only on main process)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Initialize evaluator
    evaluator = DiffusionModelEvaluator(
        checkpoint_path=args.checkpoint,
        accelerator=accelerator,
        seed=args.seed,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps
    )
    
    # Load real dataset for comparison
    accelerator.print("\nLoading validation dataset...")
    full_dataset = create_dataset(BRATS_ROOT, args.image_size, debug=False)
    
    # Use a subset for evaluation (only on main process for metrics calculation)
    if accelerator.is_main_process:
        real_images = load_real_images(full_dataset, args.num_samples)
    else:
        real_images = []
    
    # Generate synthetic samples (distributed across GPUs)
    accelerator.print("\nGenerating synthetic samples...")
    generated_images = evaluator.generate_samples(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    accelerator.wait_for_everyone()
    
    # Only main process handles visualization and metrics
    if not accelerator.is_main_process:
        return
    
    # Save sample grids
    accelerator.print("\nSaving sample visualizations...")
    evaluator.save_sample_grid(
        generated_images,
        output_dir / 'generated_samples.png',
        grid_size=4
    )
    
    evaluator.save_sample_grid(
        real_images,
        output_dir / 'real_samples.png',
        grid_size=4
    )
    
    evaluator.compare_samples(
        generated_images,
        real_images,
        output_dir / 'comparison.png',
        num_pairs=4
    )
    
    # Calculate metrics
    accelerator.print("\n" + "="*60)
    accelerator.print("EVALUATION METRICS")
    accelerator.print("="*60)
    
    results = {}
    
    # FID
    if not args.skip_fid:
        try:
            fid = evaluator.calculate_fid(generated_images, real_images)
            results['fid'] = fid
        except Exception as e:
            print(f"Error calculating FID: {e}")
            print("Skipping FID calculation...")
    
    # KID (better for small sample sizes)
    if not args.skip_fid:  # Use same flag as FID since it uses same features
        try:
            kid = evaluator.calculate_kid(generated_images, real_images)
            results['kid_x1000'] = kid
        except Exception as e:
            print(f"Error calculating KID: {e}")
            print("Skipping KID calculation...")
    
    # Diversity
    diversity_metrics = evaluator.calculate_diversity(generated_images)
    results.update(diversity_metrics)
    
    # Save results to file
    results_file = output_dir / 'evaluation_metrics.txt'
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DIFFUSION MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Image size: {args.image_size}x{args.image_size}\n\n")
        f.write("Metrics:\n")
        f.write("-"*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.6f}\n")
    
    accelerator.print("\n" + "="*60)
    accelerator.print(f"Evaluation complete! Results saved to {output_dir}")
    accelerator.print(f"Metrics saved to {results_file}")
    accelerator.print("="*60)


if __name__ == "__main__":
    main()
