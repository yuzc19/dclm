set -x

export WANDB_DIR="/tmp/wandb"
mkdir -p $WANDB_DIR

method="baseline_01_0_fasttext_Qwen3-4B-grpo-1980-7.2B_merged-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=10-seed=124-tokens=164646502400"

TMPDIR=$WANDB_DIR torchrun --nproc-per-node 8 -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_10000-data_influence_model-flan.json \
  --logs /home/zichunyu/out/dclm_logs \
  --pretrained /home/zichunyu/out/dclm_logs/$method/checkpoints/epoch_2.pt \
  --load-pretrained-state \
  --multiple-data-passes \
  --report-to-wandb