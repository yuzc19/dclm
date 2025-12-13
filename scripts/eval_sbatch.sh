#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=preempt
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --array=5,6
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# less than 1.5 hours
# awscli 1.36.17 requires botocore==1.35.76, but you have botocore 1.29.161 which is incompatible.
# awscli 1.36.17 requires stransfer<0.11.0,>=0.10.0, but you have stransfer 0.6.2 which is incompatible.
# --eval-yaml eval/heavy.yaml \
# --tokenizer ../tokenization_configs/pythia-410m \
# git pull origin main: merge updates from origin main into local current branch

# method="baseline_01_0_fasttext_10000-data_influence_model-flan-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=3-seed=124-tokens=86387712000"
# method="baseline_01_0_fasttext_10000-data_influence_model-flan+edu-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480"
# method="baseline_01_0_fasttext_epoch_2-data_influence_model-lam-0.25-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"
# method="baseline_01_0_fasttext_epoch_2-data_influence_model-flan-group+edu-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480"
# method="baseline_01_0_fasttext_epoch_2-data_influence_model-flan-bs1-group-10-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480"
# method="baseline_01_0_fasttext_epoch_2-data_influence_model-flan-bs1-group-10+edu-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"
# method="baseline_01_0_fasttext_epoch_2-data_influence_model-flan-group+edu-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=3-seed=124-tokens=86387712000"
# method="baseline_01_0_fasttext_epoch_2-data_influence_model-flan-group+edu-open_lm_3b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=55918643200"
# method="baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5/hq"

# for SLURM_ARRAY_TASK_ID in 6; do
#     export c=$SLURM_ARRAY_TASK_ID
#     TMPDIR=/tmp PYTHONUNBUFFERED=1 torchrun --nproc_per_node 8 --master_port 47763 eval/eval_openlm_ckpt.py \
#             --donot-compute-perplexity \
#             --checkpoint /project/flame/zichunyu/out/dclm_logs/baseline_01_0_fasttext_10000-data_influence_model-flan-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5/random_200/epoch_$c.pt \
#             --model ../training/open_lm_configs/d=1024_l=24_h=8.json \
#             --config /project/flame/zichunyu/out/dclm_logs/baseline_01_0_fasttext_10000-data_influence_model-flan-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/params.txt \
#             --eval-yaml eval/mmlu_and_lowvar.yaml \
#             --output-file results/baseline_01_0_fasttext_10000-data_influence_model-flan-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5/random_200/epoch_$c/metrics_mmlu_and_lowvar.json \
#             --use-temp-working-dir
#     rm -rf eval_openlm_ckpt_temp_dirs_${SLURM_ARRAY_TASK_ID}
# done

method="baseline_01_0_fasttext_3.6B-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=28795904000"

for SLURM_ARRAY_TASK_ID in $(seq 9 -1 1); do
    export c=$SLURM_ARRAY_TASK_ID
    TMPDIR=/tmp PYTHONUNBUFFERED=1 torchrun --nproc_per_node 8 --master_port 47763 eval/eval_openlm_ckpt.py \
            --donot-compute-perplexity \
            --checkpoint /tmp/dclm_logs/${method}/checkpoints/epoch_$c.pt \
            --model ../training/open_lm_configs/d=1024_l=24_h=8.json \
            --config /tmp/dclm_logs/${method}/params.txt \
            --eval-yaml eval/mmlu_and_lowvar.yaml \
            --output-file results/${method}/epoch_$c/metrics_mmlu_and_lowvar.json \
            --use-temp-working-dir
    rm -rf eval_openlm_ckpt_temp_dirs_${SLURM_ARRAY_TASK_ID}
done