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

# SAC stability-first defaults (can be overridden via env vars)
SAC_LR=${SAC_LR:-5e-5}
SAC_ACTOR_LR=${SAC_ACTOR_LR:-2e-5}
SAC_ALPHA_LR=${SAC_ALPHA_LR:-5e-4}
SAC_DELTA=${SAC_DELTA:-0.04}
SAC_DELTA_FINAL=${SAC_DELTA_FINAL:-0.08}
SAC_BATCH_SIZE=${SAC_BATCH_SIZE:-64}
SAC_MIN_BUFFER_SIZE=${SAC_MIN_BUFFER_SIZE:-512}
SAC_BUFFER_SIZE=${SAC_BUFFER_SIZE:-50000}
SAC_TARGET_VALUE_CLIP=${SAC_TARGET_VALUE_CLIP:-10.0}
SAC_ACTOR_UPDATE_INTERVAL=${SAC_ACTOR_UPDATE_INTERVAL:-3}
SAC_MAX_GRAD_NORM=${SAC_MAX_GRAD_NORM:-5.0}
SAC_TARGET_ENTROPY_SCALE=${SAC_TARGET_ENTROPY_SCALE:-0.05}
SAC_TARGET_ENTROPY_SCALE_FINAL=${SAC_TARGET_ENTROPY_SCALE_FINAL:-0.10}
SAC_ALPHA_INIT=${SAC_ALPHA_INIT:-0.05}
SAC_START_RATIO=${SAC_START_RATIO:-0.5}
SAC_PHASE_RATIO=${SAC_PHASE_RATIO:-0.35}
SAC_PHASE_RATIO_FINAL=${SAC_PHASE_RATIO_FINAL:-0.65}
SAC_WARMUP_EPISODES=${SAC_WARMUP_EPISODES:-4}
SAC_SCHEDULE_MODE=${SAC_SCHEDULE_MODE:-linear}

# Train-from-scratch controls
# START_FRESH=1: ignore any RESUME_PATH and write to a new run directory.
# START_FRESH=0: allow resume by setting RESUME_PATH externally.
START_FRESH=${START_FRESH:-1}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
if [ "${START_FRESH}" = "1" ]; then
    RESUME_PATH=""
    TRAIN_LOG_DIR="${LOG_PATH}/sac-train/${sim_type}/fresh-${RUN_TAG}"
else
    RESUME_PATH=${RESUME_PATH:-""}
    TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-"${LOG_PATH}/sac-train/${sim_type}"}
fi

# Hugging Face configuration
HF_REPO_ID=${HF_REPO_ID:-""}
HF_ENDPOINT=${HF_ENDPOINT:-"https://huggingface.co"}

echo "===== SAC Pretraining ====="
echo "Environment: ${sim_type} (prop_dir=${prop_dir})"
echo "Episodes: ${NUM_EPISODES}"
echo "Model: ${MODEL_PATH}"
echo "Start Fresh: ${START_FRESH}"
echo "Resume Path: ${RESUME_PATH}"
echo "Log Dir: ${TRAIN_LOG_DIR}"
echo "=========================="

conda run -n ${CONDA_ENV} env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    python3 guided_diffusion/train_sac.py \
        --dropout 0.1 --class_cond True --gray_imgs True \
        --learn_sigma True --diffusion_steps 1000 --noise_schedule cosine \
        --model_path ${MODEL_PATH} \
        --log_dir ${TRAIN_LOG_DIR} \
        --batch_size 1 --num_samples 1 \
        --timestep_respacing 100 \
        --num_classes 3 --manual_class_id 0 \
        --gpu_id ${gpu_id} \
        --sim_type ${sim_type} --prop_dir ${prop_dir} \
        --stoptime 0.0 \
        --num_episodes ${NUM_EPISODES} \
        --save_interval ${SAVE_INTERVAL} \
        --resume_path "${RESUME_PATH}" \
        --sac_lr ${SAC_LR} --sac_actor_lr ${SAC_ACTOR_LR} --sac_alpha_lr ${SAC_ALPHA_LR} \
        --sac_delta ${SAC_DELTA} --sac_patch_size 8 \
        --sac_batch_size ${SAC_BATCH_SIZE} --sac_min_buffer_size ${SAC_MIN_BUFFER_SIZE} --sac_buffer_size ${SAC_BUFFER_SIZE} \
        --sac_gamma 0.99 --sac_reward_scale 1.0 \
        --sac_target_value_clip ${SAC_TARGET_VALUE_CLIP} \
        --sac_actor_update_interval ${SAC_ACTOR_UPDATE_INTERVAL} \
        --sac_max_grad_norm ${SAC_MAX_GRAD_NORM} --sac_target_entropy_scale ${SAC_TARGET_ENTROPY_SCALE} --sac_alpha_init ${SAC_ALPHA_INIT} \
        --sac_target_entropy_scale_final ${SAC_TARGET_ENTROPY_SCALE_FINAL} \
        --sac_start_ratio ${SAC_START_RATIO} \
        --sac_phase_ratio ${SAC_PHASE_RATIO} \
        --sac_phase_ratio_final ${SAC_PHASE_RATIO_FINAL} \
        --sac_delta_final ${SAC_DELTA_FINAL} \
        --sac_warmup_episodes ${SAC_WARMUP_EPISODES} \
        --sac_schedule_mode ${SAC_SCHEDULE_MODE} \
        --hf_repo_id "${HF_REPO_ID}" \
        --hf_upload_checkpoints True \
        --hf_upload_best True \
        --hf_upload_final True \
        --hf_endpoint "${HF_ENDPOINT}"
