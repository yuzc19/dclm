#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# print commands
set -x

cd /project/flame/zichunyu/code/dclm
export WANDB_DIR="/tmp/wandb"
mkdir -p $WANDB_DIR

TORCH_ARGS=(
    --nnodes $SLURM_NNODES
    --node_rank $SLURM_NODEID
    --nproc_per_node $NPROC_PER_NODE
    --rdzv-id $SLURM_JOB_ID
    --rdzv-backend c10d
    # --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv-endpoint orchard-flame-9:29500
)

TMPDIR=$WANDB_DIR torchrun --nproc-per-node 8 -m training.train -- \
  --scale 411m_4x_cooldown \
  --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_3.6B.json \
  --logs /tmp/dclm_logs \
  --num-checkpoints 8 \
  --multiple-data-passes \
  --report-to-wandb

# TMPDIR=$WANDB_DIR torchrun ${TORCH_ARGS[@]} -m training.train -- \
#   --scale 411m_10x \
#   --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_merged.json \
#   --logs /tmp/dclm_logs \
#   --num-checkpoints 11 \
#   --multiple-data-passes \
#   --report-to-wandb

# TMPDIR=$WANDB_DIR torchrun ${TORCH_ARGS[@]} -m training.train -- \
#   --scale 411m_4x \
#   --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_epoch_2-data_influence_model-flan.json \
#   --logs /project/flame/zichunyu/out/dclm_logs \
#   --multiple-data-passes \
#   --report-to-wandb

# TMPDIR=$WANDB_DIR torchrun ${TORCH_ARGS[@]} -m training.train -- \
#   --scale 3b_1x_fast_3e-3_lr_1e-4_zloss \
#   --data-config exp_data/datasets/tokenized/baseline_01_01_fasttext.json \
#   --logs /project/flame/zichunyu/out/dclm_logs \
#   --pretrained /project/flame/zichunyu/out/dclm_logs/baseline_01_01_fasttext-open_lm_3b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=55918643200/checkpoints/epoch_6.pt \
#   --load-pretrained-state \
#   --multiple-data-passes \
#   --report-to-wandb

# DATA_DIR=$1
# DATASET_NAME=$2
# NCCL_P2P_DISABLE=1 torchrun --nproc-per-node 8 -m training.train -- --scale 411m_4x  --data-config $DATA_DIR/${DATASET_NAME}_tokenized/${DATASET_NAME}_tokenized.json  --workers 4  --num-checkpoints 20  --logs /data/user_data/shiyu/dclm_logs/400m_1x_${DATASET_NAME}  --report-to-wandb