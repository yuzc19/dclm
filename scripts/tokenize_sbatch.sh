#!/bin/bash
#SBATCH --job-name=tokenize_shuffle
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=general
#SBATCH --exclude=babel-5-15,babel-5-23
#SBATCH --time=2-00:00:00

# set -x

# DATA_DIR=$1
# DATASET_NAME=$2
# ADDITIONAL_NAME=$3
# SPILL_LOCATION=/scratch/$(whoami)/tmp/ray

# rm -rf $SPILL_LOCATION

# ray start --head --port=6379

# # tokenize and shuffle
# PYTHONPATH=$(pwd) python ray_processing/tokenize_shuffle.py  --input $DATA_DIR/$DATASET_NAME/$ADDITIONAL_NAME  --output $DATA_DIR/${DATASET_NAME}_tokenized  --content_key Clean-Text  --tokenizer /home/$(whoami)/pretrained_models/gpt-neox-20b  --readable_name $DATASET_NAME  --ray_spill_location $SPILL_LOCATION
# python -m open_lm.utils.make_wds_manifest  --data-dir $DATA_DIR/${DATASET_NAME}_tokenized  --num-workers 64
# python make_correct_json.py $DATA_DIR/${DATASET_NAME}_tokenized

SPILL_LOCATION=/tmp/ray
mkdir -p $SPILL_LOCATION
ray start --head --port=6379 --temp-dir=$SPILL_LOCATION

gcloud storage cp -r gs://cmu-gpucloud-zichunyu/data/dclm28b/textfiles /tmp/dataset/dclm28b

PYTHONPATH=$(pwd) python ray_processing/tokenize_shuffle.py \
    --source_ref_paths exp_data/datasets/untokenized/fasttext_01_0.json \
    --output /tmp/dataset/dclm28b/tokenized \
    --ray_spill_location $SPILL_LOCATION \
    --tokenizer tokenization_configs/pythia-410m \
    --readable_name dclm28b

gcloud storage cp -r /tmp/dataset/dclm28b/tokenized gs://cmu-gpucloud-zichunyu/data/dclm28b

# python ray_processing/tokenize_shuffle.py \
#     --source_ref_paths exp_data/datasets/raw_sources/test.json \
#     --output output/cc_wet_2019_april_baselines/refinedweb_tokenized \
#     --ray_spill_location /data/users/zichunyu/tmp/ray \
#     --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
#     --readable_name refinedweb

# python ray_processing/tokenize_shuffle.py \
#     --source_ref_paths exp_data/datasets/untokenized/refinedweb.json \
#     --output ../tmp/refinedweb_01_0_tokenized \
#     --ray_spill_location /data/users/zichunyu/tmp/ray \
#     --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
#     --readable_name refinedweb_01_0

# python ray_processing/tokenize_shuffle.py \
#     --source_ref_paths exp_data/datasets/untokenized/fasttext_dim.json \
#     --output ../tmp/output/10000-data_influence_model_tokenized_2 \
#     --ray_spill_location /data/users/zichunyu/tmp/ray \
#     --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
#     --readable_name 10000-data_influence_model_2

# python -m open_lm.utils.make_wds_manifest --data-dir output/cc_wet_2019_april_baselines/refinedweb_tokenized

# python -m open_lm.utils.make_wds_manifest --data-dir ../tmp/output/10000-data_influence_model_tokenized_2

ray stop
rm -rf $SPILL_LOCATION