gpu_id=0
prop="pbs"
CONDA_ENV=${CONDA_ENV:-adjoint_diffusion}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=${MODEL_PATH:-${SCRIPT_DIR}/ema_0.9999_025000.pt}
LOG_PATH=${LOG_PATH:-${SCRIPT_DIR}/logs}
ENV_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
# 添加 WSL CUDA 库路径
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Path to pretrained SAC model (from 02-train-sac.sh)
SAC_MODEL=${SAC_MODEL:-${LOG_PATH}/sac-train/${prop}/sac_best.pt}

for manual_class_id in 0
do
    for tsr in 100
    do
        MODEL_FLAGS="--dropout 0.1 --class_cond True --gray_imgs True"
        DIFF_FLAGS="--learn_sigma True --diffusion_steps 1000 --noise_schedule cosine"
        DIR_FLAGS="--model_path ${MODEL_PATH} \
                    --log_dir ${LOG_PATH}/sac-guided/${prop}_tsr=${tsr}_class=${manual_class_id}"
        SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ${tsr} --num_classes 3 --manual_class_id ${manual_class_id} --gpu_id ${gpu_id} --save_img False"
        SAC_FLAGS="--sim_guided True --sim_type ${prop} --guidance_type sac \
                    --sac_model_path ${SAC_MODEL} \
                    --sac_lr 3e-4 --sac_delta 0.1 --sac_patch_size 8 \
                    --sac_batch_size 256 --sac_buffer_size 50000 \
                    --sac_training False \
                    --eta 1.0 --prop_dir ${prop} --save_inter True --interval 1 \
                    --use_normed_grad False --use_adjgrad_norm False --stoptime 0.0 --inter_rate 1"
        echo -e "\n\n\n\n############################ SAC Sampling ##############################\n"
        conda run -n ${CONDA_ENV} env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" python3 image_sample.py $MODEL_FLAGS $DIFF_FLAGS $DIR_FLAGS $SAMPLE_FLAGS $SAC_FLAGS
    done
done
