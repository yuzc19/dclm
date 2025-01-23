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

# srun --gres=gpu:8 --mem=512G --time 2-00:00:00 -c 128 --job-name "interactive" --pty bash
# srun --partition=general --gres=gpu:0 --mem=512G --time 2-00:00:00 -c 64 --job-name "interactive" --pty bash
# ssh -J zichunyu@babel.lti.cs.cmu.edu babel-2-25
# scp -r -J zichunyu@babel.lti.cs.cmu.edu babel-2-25:/data/datasets/hf_cache/baseline_01_0_fasttext_tokenized.tar.gz .
# scp -J zichunyu@babel.lti.cs.cmu.edu babel-2-25:/data/datasets/hf_cache/baseline_01_1_fasttext_tokenized.tar.gz .
# rsync -azv -e -r 'ssh -A -J zichunyu@babel.lti.cs.cmu.edu' babel-2-25:/data/datasets/hf_cache/baseline_01_0_fasttext_tokenized .
# tar czf baseline_01_0_fasttext_tokenized.tar.gz baseline_01_0_fasttext_tokenized
# tar cf - baseline_01_1_fasttext_tokenized | pigz > baseline_01_1_fasttext_tokenized.tar.gz
# tar -c -I 'xz -9 -T0' -f baseline_01_1_fasttext_tokenized.tar.gz baseline_01_1_fasttext_tokenized
# tar -c -I 'zstd -22 --ultra --long -T0' -f baseline_01_0_fasttext_tokenized.tar.gz baseline_01_0_fasttext_tokenized

# BASE_DIR=/data/datasets/shared/DCLM
BASE_DIR=/home/zichunyu/data
SPILL_LOCATION=/home/$(whoami)/tmp/ray
mkdir -p $SPILL_LOCATION
ray start --head --temp-dir=$SPILL_LOCATION --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/home/zichunyu/tmp/ray\"}}"}'

TMPDIR=/home/$(whoami)/tmp PYTHONPATH=$(pwd) python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
    --readable_name fasttext_01_0 \
    --output_dir $BASE_DIR/refinedweb_01_0/fasttext \
    --config_path baselines/baselines_configs/fasttext_filter.yaml \
    --source_name cc

# python ray_processing/process.py \
#     --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
#     --readable_name fasttext_dim \
#     --output_dir output/refinedweb_01_0/fasttext_dim \
#     --config_path baselines/baselines_configs/fasttext_dim_filter.yaml \
#     --source_name cc \
#     --overwrite

ray stop
rm -rf $SPILL_LOCATION