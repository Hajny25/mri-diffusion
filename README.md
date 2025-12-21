# MRI Diffusion Models

This project implements diffusion models for MRI image generation using the BraTS2021 Task 1 dataset.

## Project Structure

```
mri-diffusion/
├── data/                          # BraTS2021 Task 1 dataset (not included)
├── experiments/
│   ├── ddpm_2d/                  # 2D diffusion model experiments
│   ├── ddpm_3d/                  # 3D diffusion model experiments
│   └── lddpm_3d/                 # Latent 3D diffusion model with VAE
├── evaluation_results/            # Generated evaluation metrics
├── logs/                         # Training and GPU usage logs
├── output/                       # Model checkpoints
├── job_submission.sh             # SLURM training job script
├── eval.sh                       # SLURM evaluation job script
└── eval_vae.sh                   # SLURM VAE evaluation job script
```

## Requirements

- CUDA 12.9
- Python virtual environment with required packages
- `.env` file with environment variables
- BraTS2021 Task 1 dataset in `data/` directory

## Setup

1. Place the BraTS2021 Task 1 dataset in the `data/` directory
2. Create and activate virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies
4. Create `.env` file with necessary configurations

## Training

Submit training jobs using SLURM:

```bash
sbatch job_submission.sh <experiment>
```

Available experiments:
- `ddpm_2d` - 2D diffusion model
- `ddpm_3d` - 3D diffusion model
- `lddpm_3d` - Latent 3D diffusion model with VAE

Example:
```bash
sbatch job_submission.sh lddpm_3d
```

## Evaluation

Evaluate trained models using the appropriate script:

### General Evaluation (eval.sh)
```bash
sbatch eval.sh <experiment> <checkpoint> [num_samples]
```

Examples:
```bash
# Auto-detect latest checkpoint
sbatch eval.sh ddpm_2d best_model 100

# Specify full path
sbatch eval.sh ddpm_2d output/abc123/best_model 100
```

### VAE Evaluation (eval_vae.sh)
```bash
sbatch eval_vae.sh <checkpoint_path>
```

Examples:
```bash
# Auto-detect latest checkpoint
sbatch eval_vae.sh best_model.pt

# Specify full path
sbatch eval_vae.sh output/abc123/best_model.pt
```

## Hardware Requirements

- Training: 4x GPU (20GB VRAM each), 8 CPU cores, 48-hour time limit
- Evaluation: 2x GPU, 4 CPU cores, 48-hour time limit

## Output

- Model checkpoints: `output/<run_id>/`
- Evaluation metrics: `evaluation_results/<experiment>/<run_id>/`
- Training logs: `logs/<experiment>/`
- GPU monitoring: `logs/<experiment>/gpu_usage_<job_id>.log`
