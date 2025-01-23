set -x

export WANDB_DIR="/home/zichunyu/tmp"
mkdir -p $WANDB_DIR

method="baseline_01_1_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"

torchrun --nproc-per-node 8 -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_10000-data_influence_model-flan.json \
  --logs /home/zichunyu/out/dclm_logs \
  --pretrained /home/zichunyu/out/dclm_logs/$method/checkpoints/epoch_2.pt \
  --load-pretrained-state \
  --multiple-data-passes \
  --report-to-wandb