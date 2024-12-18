#!/bin/bash
#SBATCH --job-name=select_data
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --partition=preempt
#SBATCH --exclude=babel-5-15,babel-5-23
#SBATCH --time=2-00:00:00

# python mates/tokenization/select_data.py

python mates/tokenization/select_bootstrap_data.py

# cd rust_processing/tokshuf-rs
# bash rust_tokenize.sh