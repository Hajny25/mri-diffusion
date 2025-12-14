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

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    "
 
export SCRIPT="training.py --model ddpm_2d"

    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT " 
srun $CMD
 