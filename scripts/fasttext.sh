python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_1.json \
    --readable_name fasttext \
    --output_dir output/refinedweb_01_1/fasttext \
    --config_path baselines/baselines_configs/fasttext_filter.yaml \
    --source_name cc \
    --overwrite

# python ray_processing/process.py \
#     --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
#     --readable_name dim \
#     --output_dir output/refinedweb_01_0/dim \
#     --config_path baselines/baselines_configs/dim_filter.yaml \
#     --source_name cc \
#     --overwrite

# python ray_processing/process.py \
#     --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
#     --readable_name fasttext_dim \
#     --output_dir output/refinedweb_01_0/fasttext_dim \
#     --config_path baselines/baselines_configs/fasttext_dim_filter.yaml \
#     --source_name cc \
#     --overwrite
