# Evaluation Script for DDPM 2D Model

This evaluation script assesses the quality of generated MRI slices from your trained diffusion model.

## Metrics Calculated

### 1. **SSIM (Structural Similarity Index)**
- Measures structural similarity between generated and real images
- Range: -1 to 1 (higher is better)
- Particularly important for medical images as it captures perceptual quality

### 2. **PSNR (Peak Signal-to-Noise Ratio)**
- Measures reconstruction quality in dB
- Higher values indicate better quality
- Common metric for image generation tasks

### 3. **FID (Fréchet Inception Distance)**
- Measures similarity of feature distributions between generated and real images
- Lower is better
- Industry-standard metric for generative models

### 4. **Diversity Metrics**
- **Average Pairwise Distance**: Measures variability between generated samples
- **Pixel Variance**: Measures overall variance across the dataset
- Higher values indicate more diverse generations

## Usage

### Basic Usage

```bash
# Evaluate the best model from the most recent training run
python experiments/ddpm_2d/evaluation.py \
    --checkpoint output/<run_id>/best_model \
    --num_samples 100
```

### Using SLURM

```bash
# Submit evaluation job
sbatch eval.sh ddpm_2d best_model 100
```

### Command Line Arguments

- `--checkpoint`: Path to model checkpoint directory (required)
  - Example: `output/abc123def456/best_model`
  
- `--num_samples`: Number of samples to generate (default: 100)
  - More samples = more reliable metrics but longer evaluation time
  
- `--batch_size`: Batch size for generation (default: 16)
  - Adjust based on available GPU memory
  
- `--image_size`: Image resolution (default: 128)
  - Must match the training configuration
  
- `--output_dir`: Where to save results (default: evaluation_results)
  
- `--device`: Device to use (default: cuda)
  - Options: cuda, cpu
  
- `--num_train_timesteps`: Training timesteps (default: 1000)
  - Must match training configuration
  
- `--num_inference_steps`: Inference steps for generation (default: 1000)
  - More steps = higher quality but slower
  
- `--skip_fid`: Skip FID calculation
  - Use if you don't want to download Inception model

## Examples

### Evaluate with 200 samples
```bash
python experiments/ddpm_2d/evaluation.py \
    --checkpoint output/abc123/best_model \
    --num_samples 200 \
    --output_dir evaluation_results/experiment_1
```

### Evaluate final model instead of best model
```bash
python experiments/ddpm_2d/evaluation.py \
    --checkpoint output/abc123/final_model \
    --num_samples 100
```

### Quick evaluation without FID
```bash
python experiments/ddpm_2d/evaluation.py \
    --checkpoint output/abc123/best_model \
    --num_samples 50 \
    --skip_fid
```

### CPU-only evaluation
```bash
python experiments/ddpm_2d/evaluation.py \
    --checkpoint output/abc123/best_model \
    --num_samples 20 \
    --device cpu
```

## Output

The evaluation script creates the following outputs in the specified output directory:

1. **evaluation_metrics.txt**: Text file with all metric results
2. **generated_samples.png**: Grid of generated MRI slices
3. **real_samples.png**: Grid of real MRI slices for comparison
4. **comparison.png**: Side-by-side comparison of generated vs real images

## Interpreting Results

### Good Results
- **SSIM**: > 0.7 indicates good structural similarity
- **PSNR**: > 20 dB is reasonable, > 25 dB is good
- **FID**: < 50 is good, < 20 is excellent
- **Diversity**: Higher values indicate the model isn't just memorizing training data

### What to Look For
- Generated images should look realistic
- MRI structures (brain anatomy) should be clearly visible
- Images should show variety (not all identical)
- Visual quality should be comparable to real images

## Notes

- First run will download the Inception V3 model for FID calculation (~100MB)
- Generation time scales with `num_samples` and `num_inference_steps`
- Typical evaluation with 100 samples takes 5-10 minutes on a single GPU
- For research papers, use at least 1000 samples for reliable FID scores

## Troubleshooting

### Out of Memory
Reduce `--batch_size` or `--num_samples`

### Model Loading Error
Verify the checkpoint path exists and contains model files

### Slow Generation
Reduce `--num_inference_steps` (but quality may decrease)

### Import Errors
Install required packages:
```bash
pip install scikit-image matplotlib torchvision scipy
```
