torchrun --nproc-per-node 2 -m mates.probing.probe_data_influence -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext.json \
  --logs output/logs
