#!/bin/bash
#SBATCH --job-name=fasttext_filter
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --partition=general
#SBATCH --nodelist=babel-4-21
#SBATCH --exclude=babel-5-15,babel-5-23
#SBATCH --time=2-00:00:00

# BASE_DIR=/data/datasets/shared/DCLM
BASE_DIR=/data/datasets/hf_cache
SPILL_LOCATION=/scratch/$(whoami)/tmp/ray
mkdir -p $SPILL_LOCATION
ray start --head --port=6379 --temp-dir=$SPILL_LOCATION

# babel-4-*

TMPDIR=/scratch/$(whoami)/tmp PYTHONPATH=$(pwd) python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
    --readable_name fasttext_new_01_0 \
    --output_dir $BASE_DIR/refinedweb_01_0/fasttext \
    --config_path baselines/baselines_configs/fasttext_filter.yaml \
    --source_name cc \
    --overwrite

# python ray_processing/process.py \
#     --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
#     --readable_name fasttext_dim \
#     --output_dir output/refinedweb_01_0/fasttext_dim \
#     --config_path baselines/baselines_configs/fasttext_dim_filter.yaml \
#     --source_name cc \
#     --overwrite

ray stop
rm -rf $SPILL_LOCATION