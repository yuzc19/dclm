#!/bin/bash
#SBATCH --job-name=dim                # Set the job name
#SBATCH --output=logs/dim-%j.log      # Set the output file
#SBATCH --nodes=1                     # Request 1 compute nodes
#SBATCH --gres=gpu:8                  # Request 8 GPU devices per node
#SBATCH --cpus-per-task=128           # Request 128 CPU cores per task
#SBATCH --mem=512G                    # Request 512 GB of RAM per node
#SBATCH --time=2-00:00:00             # Set the time limit

torchrun --nproc-per-node 8 mates/modeling/train_pairwise_model.py