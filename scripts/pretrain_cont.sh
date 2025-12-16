set -x

export WANDB_DIR="/tmp/wandb"
mkdir -p $WANDB_DIR

method="baseline_01_0_fasttext_3.6B-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=28795904000"
method="baseline_01_0_fasttext_Qwen3-4B-grpo-1980-3.6B_merged-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=28795904000"

mkdir -p /tmp/dclm_logs
if [ ! -d "/tmp/dclm_logs/$method" ]; then
  gcloud storage cp -r gs://cmu-gpucloud-zichunyu/out/dclm_logs/data-limited-pretraining/$method /tmp/dclm_logs
fi

TMPDIR=$WANDB_DIR torchrun --nproc-per-node 8 -m training.train -- \
  --scale 411m_4x_cooldown \
  --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_merged.json \
  --logs /tmp/dclm_logs \
  --num-checkpoints 8 \
  --pretrained /tmp/dclm_logs/$method/checkpoints/epoch_8.pt \
  --load-pretrained-state \
  --multiple-data-passes \
  --report-to-wandb