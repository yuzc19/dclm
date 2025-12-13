gpu_index=0
for s in {0..7}; do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index python mates/modeling/predict_pairwise_data_influence.py --base_dir /project/flame/zichunyu --shard $s 8 \
    > logs/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done