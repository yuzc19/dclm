#!/bin/bash
#SBATCH --job-name=dclm
#SBATCH --output=slurm_logs/dclm_%j.out
#SBATCH --error=slurm_logs/dclm_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-31,babel-0-37,babel-1-23,babel-1-27,babel-1-31,babel-15-36
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

torchrun --nproc-per-node 2 -m mates.probing.probe_data_influence -- \
  --scale 411m_4x \
  --data-config exp_data/datasets/tokenized/baseline_toy.json \
  --logs /data/datasets/hf_cache/dclm_logs
