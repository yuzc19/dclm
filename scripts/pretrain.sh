export WANDB_DIR="/data/users/zichunyu/tmp"

# torchrun --nproc-per-node 8 -m training.train -- \
#   --scale 411m_1x \
#   --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext.json  \
#   --logs logs \
#   --report-to-wandb

torchrun --nproc-per-node 8 -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/10000-data_influence_model.json \
  --logs /data/users/zichunyu/tmp/logs \
  --report-to-wandb

# python -m training.train \
#   --scale 1b_1x_fast \
#   --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext.json  \
#   --logs logs \
#   --report-to-wandb
