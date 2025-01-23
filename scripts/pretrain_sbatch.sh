#!/bin/bash
#SBATCH --job-name=dclm_pretrain
#SBATCH --output=slurm_logs/dclm_%j.out
#SBATCH --error=slurm_logs/dclm_%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:8
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-37,babel-1-23,babel-1-27,babel-1-31,babel-13-13,babel-13-29,babel-15-32,babel-15-36
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

#SBATCH --mem-per-gpu=64G

# print commands
set -x

export WANDB_DIR="/home/zichunyu/tmp"
mkdir -p $WANDB_DIR

torchrun --nproc-per-node 8 -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/baseline_01_1_fasttext.json \
  --logs /home/zichunyu/out/dclm_logs \
  --multiple-data-passes \
  --report-to-wandb

# DATA_DIR=$1
# DATASET_NAME=$2
# NCCL_P2P_DISABLE=1 torchrun --nproc-per-node 8 -m training.train -- --scale 411m_4x  --data-config $DATA_DIR/${DATASET_NAME}_tokenized/${DATASET_NAME}_tokenized.json  --workers 4  --num-checkpoints 20  --logs /data/user_data/shiyu/dclm_logs/400m_1x_${DATASET_NAME}  --report-to-wandb