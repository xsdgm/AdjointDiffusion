gpu_id=0
prop="top"
MODEL_PATH=...
LOG_PATH=...

for manual_class_id in 0
do
    for tsr in 100
    do
        eta_list=(1)
        for eta in "${eta_list[@]}"
        do
            MODEL_FLAGS="--dropout 0.1 --class_cond True --gray_imgs True"
            DIFF_FLAGS="--learn_sigma True --diffusion_steps 1000 --noise_schedule cosine"
            DIR_FLAGS="--model_path ${MODEL_PATH} \
                        --log_dir ${LOG_PATH}/sim-guided/${prop}_tsr=${tsr}_class=${manual_class_id}_eta=${eta}"
            SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ${tsr} --num_classes 3 --manual_class_id ${manual_class_id} --gpu_id ${gpu_id} --save_img False"
            SIM_FLAGS="--sim_guided True --sim_type waveguide  --use_normed_grad True --use_adjgrad_norm False  --eta ${eta} --prop_dir ${prop} --save_inter True --interval 1"
            echo -e "\n\n\n\n############################ Sampling with eta = ${eta} ##############################\n"
            python3 image_sample.py $MODEL_FLAGS $DIFF_FLAGS $DIR_FLAGS $SAMPLE_FLAGS $SIM_FLAGS
        done
    done
done
