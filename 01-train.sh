#!/bin/bash

DATA_DIR=...
LOG_DIR=...
GPU_ID=...

# Set model, diffusion, directory, and training flags
MODEL_FLAGS="--dropout 0.1 --class_cond True --gray_imgs True"
DIFF_FLAGS="--learn_sigma True --diffusion_steps 1000 --noise_schedule cosine"
DIR_FLAGS="--data_dir ${DATA_DIR} --log_dir ${LOG_DIR}"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 64 --gpu_id ${GPU_ID}"

# Run training
python3 image_train.py $MODEL_FLAGS $DIFF_FLAGS $DIR_FLAGS $TRAIN_FLAGS
