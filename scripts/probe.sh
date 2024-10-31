export WANDB_DIR="/data/users/zichunyu/tmp/wandb"

python -m training.probe_data_influence \
  --scale probe_411m_1x \
  --data-config exp_data/datasets/tokenized/refinedweb_01_0.json  \
  --logs logs \
  --do-eval \
  --downstream-eval
