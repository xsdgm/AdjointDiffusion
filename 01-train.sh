#!/bin/bash

# WSL CUDA Issue Fix
CONDA_ENV="${CONDA_ENV:-adjoint_diffusion}"
ENV_BASE="${ENV_BASE:-/home/xsdgm/miniconda3/envs}"
ENV_PREFIX="${ENV_BASE}/${CONDA_ENV}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="${DATA_DIR:-/home/xsdgm/AdjointDiffusion/datasets/21/sigma2/struct}"
LOG_DIR_BASE="/home/xsdgm/AdjointDiffusion/logs"
LOG_SUBDIR="${LOG_SUBDIR:-train_logs_21}"
LOG_DIR="${LOG_DIR_BASE}/${LOG_SUBDIR}"
GPU_ID="0"
IMAGE_SIZE="${IMAGE_SIZE:-21}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"

# Create a dedicated output directory so the old checkpoints stay untouched.
mkdir -p "${LOG_DIR}"

# Set model, diffusion, directory, and training flags
MODEL_FLAGS="--dropout 0.1 --class_cond False --gray_imgs True"
DIFF_FLAGS="--learn_sigma True --diffusion_steps 1000 --noise_schedule cosine"
DIR_FLAGS="--data_dir ${DATA_DIR} --log_dir ${LOG_DIR}"
SIZE_FLAGS="--image_size ${IMAGE_SIZE}"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 64 --gpu_id ${GPU_ID} --log_interval 1 --save_interval ${SAVE_INTERVAL} --lr_anneal_steps ${LR_ANNEAL_STEPS} --print_fom True"

# Run training
mpirun -x LD_LIBRARY_PATH -n 1 "${ENV_PREFIX}/bin/python" -u image_train.py $MODEL_FLAGS $DIFF_FLAGS $DIR_FLAGS $SIZE_FLAGS $TRAIN_FLAGS
