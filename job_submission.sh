#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4g.20gb:4
#SBATCH --cpus-per-task=8


# Require the experiment name as argument
if [ -z "$1" ]; then
  echo "Error: no target experiment specified." >&2
  echo "Usage: sbatch job_submission.sh <experiment> [args...]" >&2
  echo "Examples:" >&2
  echo "  sbatch job_submission.sh ddpm2d" >&2
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"
module load devel/cuda/12.9

TRAINING_SCRIPT="experiments/$1/training.py"
ls -l "$TRAINING_SCRIPT" || echo "DEBUG: file '$TRAINING_SCRIPT' does not exist"

echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
source "$HOME"/.bashrc
source .env
source "venv/bin/activate"

echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURM_NNODES: $SLURM_NNODES"

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export GPUS_PER_NODE=4

LOG_DIR="logs/$1"
echo "-----------------------------------------------------------"
GPU_LOG="$LOG_DIR/gpu_usage_$SLURM_JOB_ID.log"
nvidia-smi --list-gpus
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,nounits \
  -l 3 >> "$GPU_LOG" &
GPU_MONITOR_PID=$!

######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    "
 

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $TRAINING_SCRIPT" 
echo "script: $TRAINING_SCRIPT"
echo "Running command: $CMD"
srun $CMD
EXIT_CODE=$?

kill "$GPU_MONITOR_PID" 2>/dev/null || true

exit "$EXIT_CODE"
 