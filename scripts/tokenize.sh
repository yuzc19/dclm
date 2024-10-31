# export PYTHONPATH=$(pwd)
# python ray_processing/tokenize_shuffle.py \
#     --source_ref_paths exp_data/datasets/raw_sources/test.json \
#     --output output/cc_wet_2019_april_baselines/refinedweb_tokenized \
#     --ray_spill_location /data/users/zichunyu/tmp/ray \
#     --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
#     --readable_name refinedweb

export PYTHONPATH=$(pwd)
# python ray_processing/tokenize_shuffle.py \
#     --source_ref_paths exp_data/datasets/untokenized/refinedweb.json \
#     --output ../tmp/refinedweb_01_0_tokenized \
#     --ray_spill_location /data/users/zichunyu/tmp/ray \
#     --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
#     --readable_name refinedweb_01_0

python ray_processing/tokenize_shuffle.py \
    --source_ref_paths exp_data/datasets/untokenized/fasttext_dim.json \
    --output ../tmp/output/10000-data_influence_model_tokenized_2 \
    --ray_spill_location /data/users/zichunyu/tmp/ray \
    --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
    --readable_name 10000-data_influence_model_2

# python -m open_lm.utils.make_wds_manifest --data-dir output/cc_wet_2019_april_baselines/refinedweb_tokenized

python -m open_lm.utils.make_wds_manifest --data-dir ../tmp/output/10000-data_influence_model_tokenized_2
