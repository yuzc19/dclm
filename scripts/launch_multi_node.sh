#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --partition=flame
#SBATCH --qos=flame-t1b_g1_qos
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00

# Setup the environment.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=8000

# Dispatch the training.
srun -W 0 scripts/pretrain_sbatch.sh