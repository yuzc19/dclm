# with-proxy python3 ray_processing/process.py \
#   --source_ref_paths exp_data/datasets/raw_sources/test.json \
#   --readable_name c4_v4 \
#   --output_dir output/cc_wet_2019_april_baselines/c4_v4/ \
#   --config_path baselines/baselines_configs/c4.yaml \
#   --source_name cc_april_2019 \
#   --overwrite

# with-proxy python3 ray_processing/process.py \
  # --source_ref_paths exp_data/datasets/raw_sources/test.json \
  # --readable_name refinedweb \
  # --output_dir output/cc_wet_2019_april_baselines/refinedweb/ \
  # --config_path baselines/baselines_configs/refinedweb.yaml \
  # --source_name cc \
  # --overwrite

# attempt to make ~5 checkpoints: each on a different portion of the data (instead of 5 epochs on 1/5 of the data).
# 410m - 6h
# global-shard_05/06/07/08_of_10 of dclm-baseline-1.0
# Table 9 results actually subsample from dclm-baseline-1.0, while leaderboard results start with the pools linked above and are processed as described in section 4.
