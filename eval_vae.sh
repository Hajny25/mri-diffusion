#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2     
#SBATCH --cpus-per-task=4

cd "$SLURM_SUBMIT_DIR"
module load devel/cuda/12.9

source "$HOME"/.bashrc
source .env
source "venv/bin/activate"

echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURM_NNODES: $SLURM_NNODES"

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export GPUS_PER_NODE=2
echo "-----------------------------------------------------------"
nvidia-smi -l 3 >> nvidia-smi-${SLURM_JOB_ID}.log &
nvidia-smi --list-gpus

######################

# Usage: sbatch eval.sh <experiment_dir> <checkpoint_subdir_or_path> [num_samples]
# Example 1 (auto-detect latest run): sbatch eval.sh ddpm_2d best_model 100
# Example 2 (manual path): sbatch eval.sh ddpm_2d output/abc123/best_model 100

CHECKPOINT=${1:-"best_model.pt"}

# Check if CHECKPOINT contains a path separator (/) to determine if it's a full path
if [[ "$CHECKPOINT" == */* ]]; then
    # User provided a full path
    CHECKPOINT_PATH="$CHECKPOINT"
    echo "Using manually specified checkpoint: $CHECKPOINT_PATH"
else
    # Auto-detect: Find the most recent run directory
    RUN_DIR=$(ls -td output/*/ 2>/dev/null | head -1)
    
    if [ -z "$RUN_DIR" ]; then
        echo "Error: No run directories found in output/"
        exit 1
    fi
    
    CHECKPOINT_PATH="${RUN_DIR}${CHECKPOINT}"
    echo "Auto-detected checkpoint: $CHECKPOINT_PATH"
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

export TQDM_DISABLE=1

echo "Evaluating checkpoint: $CHECKPOINT_PATH"
echo "Using $GPUS_PER_NODE GPUs per node"

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --mixed_precision fp16 \
    "

export SCRIPT="experiments/lddpm_3d/eval_vae.py"
export SCRIPT_ARGS="--checkpoint $CHECKPOINT_PATH"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun bash -c "$CMD"
 