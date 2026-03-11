gpu_id=0
# ==============================================================
# Simulation Environment Selection
# Uncomment the block of the environment you want to train on.
# ==============================================================

# --- Option 1: Bent Waveguide (Default for testing) ---
sim_type="waveguide"
prop_dir="top"  # 'top' evaluates to bent waveguide performance
# --------------------------------------------------------

# --- Option 2: Polarization Beam Splitter (PBS) ---
# sim_type="pbs"
# prop_dir="pbs"
# --------------------------------------------------------

HF_REPO_ID=""
CONDA_ENV=${CONDA_ENV:-adjoint_diffusion}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MODEL_PATH=${MODEL_PATH:-${SCRIPT_DIR}/ema_0.9999_025000.pt}  # 本地路径
MODEL_PATH=${MODEL_PATH:-hf:xsdgm/diffusioninversedesign/ema_0.9999_025000.pt}  # 从HF加载
LOG_PATH=${LOG_PATH:-${SCRIPT_DIR}/logs}
ENV_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
# 添加 WSL CUDA 库路径以解决 libcuda.so 找不到的问题
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Number of pretraining episodes
NUM_EPISODES=${NUM_EPISODES:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

# Optional: resume from a previous checkpoint
RESUME_PATH=${RESUME_PATH:-""}

# Hugging Face configuration
HF_REPO_ID=${HF_REPO_ID:-""}
HF_ENDPOINT=${HF_ENDPOINT:-"https://huggingface.co"}

echo "===== SAC Pretraining ====="
echo "Environment: ${sim_type} (prop_dir=${prop_dir})"
echo "Episodes: ${NUM_EPISODES}"
echo "Model: ${MODEL_PATH}"
echo "=========================="

conda run -n ${CONDA_ENV} env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    python3 guided_diffusion/train_sac.py \
        --dropout 0.1 --class_cond True --gray_imgs True \
        --learn_sigma True --diffusion_steps 1000 --noise_schedule cosine \
        --model_path ${MODEL_PATH} \
        --log_dir ${LOG_PATH}/sac-train/${sim_type} \
        --batch_size 1 --num_samples 1 \
        --timestep_respacing 100 \
        --num_classes 3 --manual_class_id 0 \
        --gpu_id ${gpu_id} \
        --sim_type ${sim_type} --prop_dir ${prop_dir} \
        --stoptime 0.0 \
        --num_episodes ${NUM_EPISODES} \
        --save_interval ${SAVE_INTERVAL} \
        --resume_path "${RESUME_PATH}" \
        --sac_lr 3e-4 --sac_alpha_lr 3e-4 \
        --sac_delta 0.1 --sac_patch_size 8 \
        --sac_batch_size 64 --sac_buffer_size 50000 \
        --sac_gamma 0.99 --sac_reward_scale 1.0 \
        --sac_start_ratio 0.5 \
        --hf_repo_id "${HF_REPO_ID}" \
        --hf_upload_checkpoints True \
        --hf_upload_best True \
        --hf_upload_final True \
        --hf_endpoint "${HF_ENDPOINT}"
