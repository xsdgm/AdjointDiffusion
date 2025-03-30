#!/bin/bash

# Set model, diffusion, directory, and training flags
MODEL_FLAGS="--dropout 0.1 --class_cond True --gray_imgs True"
DIFF_FLAGS="--learn_sigma True --diffusion_steps 1000 --noise_schedule cosine"
DIR_FLAGS="--data_dir /media/usb_media/datasets/physics/wg/64/class-cond --log_dir /media/usb_media/guided-diffusion-phy/wg-64_class-cond"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 64 --gpu_id 7"

# Run training
python3 image_train.py $MODEL_FLAGS $DIFF_FLAGS $DIR_FLAGS $TRAIN_FLAGS
