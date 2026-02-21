gpu_id=0
prop="pbs"
CONDA_ENV=${CONDA_ENV:-pbs_chip_env}
MODEL_PATH=${MODEL_PATH:-/home/kylin/AdjointDiffusion/ema_0.9999_025000.pt}
LOG_PATH=${LOG_PATH:-/home/kylin/AdjointDiffusion/logs}
ENV_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Number of pretraining episodes
NUM_EPISODES=${NUM_EPISODES:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

# Optional: resume from a previous checkpoint
RESUME_PATH=${RESUME_PATH:-""}

echo "===== SAC Pretraining ====="
echo "Episodes: ${NUM_EPISODES}"
echo "Model: ${MODEL_PATH}"
echo "=========================="

conda run -n ${CONDA_ENV} env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    python3 scripts/train_sac.py \
        --dropout 0.1 --class_cond True --gray_imgs True \
        --learn_sigma True --diffusion_steps 1000 --noise_schedule cosine \
        --model_path ${MODEL_PATH} \
        --log_dir ${LOG_PATH}/sac-train/${prop} \
        --batch_size 1 --num_samples 1 \
        --timestep_respacing 100 \
        --num_classes 3 --manual_class_id 0 \
        --gpu_id ${gpu_id} \
        --sim_type ${prop} --prop_dir ${prop} \
        --stoptime 0.0 \
        --num_episodes ${NUM_EPISODES} \
        --save_interval ${SAVE_INTERVAL} \
        --resume_path "${RESUME_PATH}" \
        --sac_lr 3e-4 --sac_alpha_lr 3e-4 \
        --sac_delta 0.1 --sac_patch_size 8 \
        --sac_batch_size 256 --sac_buffer_size 50000
