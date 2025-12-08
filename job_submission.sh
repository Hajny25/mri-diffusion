#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4g.20gb:2

cd "$SLURM_SUBMIT_DIR"

source "$HOME"/.bashrc
source .env
source venv/bin/activate

nvidia-smi

echo "Starting script..."

python training.py
