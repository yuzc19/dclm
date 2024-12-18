#!/bin/bash
#SBATCH --job-name=dclm_array
#SBATCH --output=./slurm-out/array_%j.out
#SBATCH --error=./slurm-out/array_%j.err
#SBATCH --partition=array
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-37,babel-1-23,babel-1-27,babel-1-31         
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

# Exit immediately on any error
set -e

RANK=$SLURM_ARRAY_TASK_ID
# CKPT=$(printf "0001%04d\n" $((RANK * 50 + 50)))

echo "Processing RANK $RANK"

# python mates/tokenization/bert_tokenize.py --shard $RANK 8

python mates/modeling/predict_data_influence.py --shard $RANK 8

# PYTHONUNBUFFERED=1 python -m litgpt.probe_comb_data_influence \
#       --model_name pythia-410m \
#       --base_dir $base_dir \
#       --tokenizer_dir checkpoints/EleutherAI/pythia-410m \
#       --data Tulu \
#       --train.resume_steps 10000 \
#       --out_dir $base_dir/out/pythia-410m/fineweb/sample-350BT/random \
#       --seed $RANK \
#       --rank $RANK \
#       --devices 1 || { echo "Error processing RANK $RANK"; exit 1; }
