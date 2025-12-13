#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=preempt
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
#SBATCH --gres=gpu:2
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

# CMD="-m mates.probing.probe_data_influence -- \
#   --scale 1b_1x_fast \
#   --data-config exp_data/datasets/tokenized/baseline_01_01_fasttext.json \
#   --logs /project/flame/zichunyu/out/dclm_logs"

CMD="-m mates.probing.probe_data_influence -- \
  --scale 411m_4x \
  --data-config exp_data/datasets/tokenized/baseline_01_01_fasttext.json \
  --logs /project/flame/zichunyu/out/dclm_logs"

SEED=$SLURM_ARRAY_TASK_ID TMPDIR=/tmp torchrun --nproc-per-node 2 --master_port $(expr $RANDOM + 1000) $CMD

# gpu_index=0
# for s in {0..3}; do
#     echo $s
#     CUDA_VISIBLE_DEVICES=$gpu_index,$((gpu_index+1)) SEED=$s TMPDIR=/tmp nohup torchrun --nproc-per-node 2 --master_port $(expr $RANDOM + 1000) $CMD \
#     > logs/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
#     ((gpu_index=(gpu_index+2)%8))
# done