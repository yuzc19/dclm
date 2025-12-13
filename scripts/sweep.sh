#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --partition=flame
#SBATCH --qos=flame-32gpu_qos
#SBATCH --account=cx
#SBATCH --time=16-00:00:00
#SBATCH --signal=SIGTERM@512

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1536G
#SBATCH --cpus-per-task=208
#SBATCH --gres=gpu:8
#SBATCH --exclude=orchard-flame-4,orchard-flame-27

# trap 'rm -f slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out' EXIT

while true; do
    srun --overlap bash -c 'echo "$(date +"%Y-%m-%d %H:%M:%S") | $(dd if=/dev/urandom bs=1M count=1024 2>/dev/null | md5sum | head -c 32) | Heartbeat from $(hostname)"'
    sleep 5m
done