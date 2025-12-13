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

# srun --mem=512G --time 2-00:00:00 -c 128 --job-name "interactive" --pty bash
# srun --gres=gpu:8 --mem=512G --time 2-00:00:00 -c 128 --job-name "interactive" --pty bash
# srun --partition=general --gres=gpu:0 --mem=512G --time 2-00:00:00 -c 64 --job-name "interactive" --pty bash
# scp -r -J zichunyu@babel.lti.cs.cmu.edu babel-2-25:/data/datasets/hf_cache/baseline_01_0_fasttext_tokenized.tar.gz .
# scp -r -J zichunyu@babel.lti.cs.cmu.edu babel-0-19:/data/datasets/hf_cache/dclm_logs/baseline_01_0_fasttext_10000-data_influence_model-flan-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480 .
# scp -J zichunyu@babel.lti.cs.cmu.edu babel-2-25:/data/datasets/hf_cache/baseline_01_1_fasttext_tokenized.tar.gz .
# rsync -azv -e -r 'ssh -A -J zichunyu@babel.lti.cs.cmu.edu' babel-2-25:/data/datasets/hf_cache/baseline_01_0_fasttext_tokenized .
# tar czf baseline_01_0_fasttext_tokenized.tar.gz baseline_01_0_fasttext_tokenized
# tar cf - baseline_01_1_fasttext_tokenized | pigz > baseline_01_1_fasttext_tokenized.tar.gz
# tar -c -I 'xz -9 -T0' -f baseline_01_1_fasttext_tokenized.tar.gz baseline_01_1_fasttext_tokenized
# tar -c -I 'zstd -22 --ultra --long -T0' -f baseline_01_0_fasttext_tokenized.tar.gz baseline_01_0_fasttext_tokenized
# gcloud compute scp --recurse orchard-login-001:~/out/1b-data_influence_model zichunyu@babel.lti.cs.cmu.edu:~/CODE

# BASE_DIR=/data/datasets/shared/DCLM
# BASE_DIR=/tmp/data
# mkdir -p $BASE_DIR
# gcloud storage cp -r gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/textfiles $BASE_DIR
SPILL_LOCATION=/tmp/ray
mkdir -p $SPILL_LOCATION
ray start --head --temp-dir=$SPILL_LOCATION --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/tmp/ray\"}}"}'

TMPDIR=/tmp PYTHONPATH=$(pwd) python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
    --readable_name fasttext_01_0 \
    --output_dir /tmp/data/fasttext_0.1 \
    --config_path baselines/baselines_configs/fasttext_filter.yaml \
    --source_name cc

rm -rf exp_data/datasets/untokenized/fasttext_01_0.json

ray stop
rm -rf $SPILL_LOCATION