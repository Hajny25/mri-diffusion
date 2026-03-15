#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=1:00:00
##SBATCH --cpus-per-task=35
##SBATCH --gres=gpu:full:4
#SBATCH --gres=gpu:full:1

source venv/bin/activate

LOG_DIR="logs/monai_autoencoder"
mkdir -p "$LOG_DIR"
GPU_LOG="$LOG_DIR/gpu_usage_$SLURM_JOB_ID.log"
nvidia-smi --list-gpus
nvidia-smi \
  --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,nounits \
  -l 3 >> "$GPU_LOG" &
GPU_MONITOR_PID=$!

# cd brats_mri_generative_diffusion && python -m monai.bundle run --config_file configs/inference.json
# python msd2.py
# cd brats_mri_generative_diffusion && torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/multi_gpu_train_autoencoder.json']" --lr 2e-5 --dataset-dir "../data/brats-2021-msd"
# python experiments/monai/train_autoencoder.py
# python experiments/monai/inference_autoencoder.py
python experiments/monai/inference.py

# torchrun --standalone --nnodes=1 --nproc_per_node=4 experiments/monai/train_autoencoder_multi.py # learning rate
# torchrun --standalone --nnodes=1 --nproc_per_node=4 experiments/monai/train_diffusion_multi.py # learning rate


# python experiments/ddpm_25d/prep_all.py --root_dir data/brats-2021 --output_file data/preprocessed_all_debug.npy --debug
# python experiments/ddpm_25d/prep_all.py --root_dir data/brats-2021 --output_file data/preprocessed_all_masks.npy --masks
#python experiments/ddpm_25d/prep.py --root_dir data/brats-2021 --output_dir data/preprocessed_masks
#python experiments/ddpm_25d/test.py
# python experiments/ddpm_25d/test_dataloader.py --workers 0
# python experiments/ddpm_25d/sample.py
#python experiments/ddpm_25d/sample3d.py

kill "$GPU_MONITOR_PID" 2>/dev/null || true

exit "$EXIT_CODE"