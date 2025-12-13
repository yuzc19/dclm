#!/bin/bash
#SBATCH --job-name=select_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1536G
#SBATCH --cpus-per-task=208
#SBATCH --time=2-00:00:00

# python mates/tokenization/select_data.py

# python mates/tokenization/select_bootstrap_data.py

cd rust_processing/tokshuf-rs
bash rust_tokenize.sh

# cd semdedup/clustering

# export PYTHONPATH=$(pwd)

# cd ..

# SHARD=$SLURM_ARRAY_TASK_ID python clustering.py

# Use srun to launch the 8 tasks concurrently.
# SLURM_PROCID will range from 0 to 7 for the tasks.
# srun bash -c 'SHARD=$((SLURM_PROCID * 2 + 1)) python clustering.py'
# python clustering.py