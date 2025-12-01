#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:0

cd "$SLURM_SUBMIT_DIR"

source "$HOME"/.bashrc
source .env
source venv/bin/activate

#MODULE=${1:-central_2d_ddpm_base.model}
#shift || true

#echo "Running module: $MODULE"
#echo "CWD: $(pwd)"
#echo "Command: python -m $MODULE $*"

#python -m "$MODULE" "$@"

echo "Starting script..."

python basic_training_diffusion.py
