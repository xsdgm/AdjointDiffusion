gpu_id=0
prop="pbs"
CONDA_ENV=${CONDA_ENV:-adjoint_diffusion}
MODEL_PATH=${MODEL_PATH:-/home/xsdgm/AdjointDiffusion/logs/train_logs/ema_0.9999_000000.pt}
LOG_PATH=${LOG_PATH:-/home/xsdgm/AdjointDiffusion/logs}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
ENV_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"
WANDB_MODE=${WANDB_MODE:-offline}
PBS_PLATFORM=${PBS_PLATFORM:-soi}
IMAGE_SIZE=${IMAGE_SIZE:-64}
CLASS_COND=${CLASS_COND:-False}

for manual_class_id in 0
do
    for tsr in 200
    do
        eta_list=(1)
        for eta in "${eta_list[@]}"
        do
            SAMPLE_LOG_DIR="${LOG_PATH}/sim-guided/${prop}_${PBS_PLATFORM}_img=${IMAGE_SIZE}_tsr=${tsr}_class=${manual_class_id}_eta=${eta}_${RUN_STAMP}"
            mkdir -p "${SAMPLE_LOG_DIR}"
            MODEL_FLAGS="--dropout 0.1 --class_cond ${CLASS_COND} --gray_imgs True"
            DIFF_FLAGS="--learn_sigma True --diffusion_steps 1000 --noise_schedule cosine"
            DIR_FLAGS="--model_path ${MODEL_PATH} \
                        --log_dir ${SAMPLE_LOG_DIR}"
            SIZE_FLAGS="--image_size ${IMAGE_SIZE}"
            SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ${tsr} --gpu_id ${gpu_id} --save_img False"
            if [ "${CLASS_COND}" = "True" ]; then
                SAMPLE_FLAGS="${SAMPLE_FLAGS} --num_classes 3 --manual_class_id ${manual_class_id}"
            fi
            SIM_FLAGS="--sim_guided True --sim_type pbs --pbs_platform ${PBS_PLATFORM} --use_normed_grad True --use_adjgrad_norm False --eta ${eta} --prop_dir ${prop} --save_inter True --interval 1"
            echo -e "\n\n\n\n############################ Sampling with eta = ${eta} ##############################\n"
            echo "Saving outputs to: ${SAMPLE_LOG_DIR}"
            conda run -n ${CONDA_ENV} env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" WANDB_MODE="${WANDB_MODE}" python3 image_sample.py $MODEL_FLAGS $DIFF_FLAGS $DIR_FLAGS $SIZE_FLAGS $SAMPLE_FLAGS $SIM_FLAGS
        done
    done
done
