#!/bin/bash
#SBATCH --job-name=eval_dclm
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:4
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-37,babel-1-23,babel-1-27,babel-1-31,babel-11-17,babel-15-36
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

# less than 1.5 hours
# awscli 1.36.17 requires botocore==1.35.76, but you have botocore 1.29.161 which is incompatible.
# awscli 1.36.17 requires stransfer<0.11.0,>=0.10.0, but you have stransfer 0.6.2 which is incompatible.
# --eval-yaml eval/heavy.yaml \
# --model ../training/open_lm_configs/open_lm_1b_swiglutorch.json \
# --tokenizer ../tokenization_configs/pythia-410m \

export HF_HOME=/data/datasets/hf_cache

# method="baseline_toy-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=2232325120"
# method="baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480"
method="baseline_01_0_fasttext_10000-data_influence_model-flan-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480"

PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 8 --master_port 47762 eval/eval_openlm_ckpt.py \
    --donot-compute-perplexity \
    --checkpoint /data/datasets/hf_cache/dclm_logs/$method/checkpoints/epoch_1.pt \
    --model open_lm_411m_v2.json \
    --config /data/datasets/hf_cache/dclm_logs/$method/params.txt \
    --eval-yaml eval/mmlu_and_lowvar.yaml \
    --output-file results/$method/epoch_1/metrics_mmlu_and_lowvar.json \
    --use-temp-working-dir
