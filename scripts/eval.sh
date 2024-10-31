# less than 1.5 hours

# method=pythia-1b/fineweb/sample-350BT/mates
for m in "games-fineweb-102400-0.7-0.3"; do
    method=pythia-1b/$m

    torchrun --nproc_per_node 8 eval/eval_openlm_ckpt.py \
        --hf-model /data/users/zichunyu/lit-gpt/out/$method/step-00045200 \
        --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
        --eval-yaml eval/mmlu_and_lowvar.yaml \
        --output-file results/$method/step-00045200/metrics_mmlu_and_lowvar.json \
        --use-temp-working-dir
done

# method="refinedweb_01_0-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=8232325120"
# torchrun --nproc_per_node 8 eval/eval_openlm_ckpt.py \
#     --checkpoint ../logs/$method/checkpoints/epoch_6.pt \
#     --model open_lm_411m_v2.json \
#     --config logs/$method/params.txt \
#     --tokenizer /data/users/zichunyu/out/hf/pythia-410m \
#     --eval-yaml eval/mmlu_and_lowvar.yaml \
#     --output-file results/$method/epoch_6/metrics_mmlu_and_lowvar.json \
#     --use-temp-working-dir

# method="refinedweb_01_0_fasttext_dim-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=8232325120"

# for c in "5"; do
    # method="baseline_01_01_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"
#     method="10000-data_influence_model-flan_2-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"
#     torchrun --nproc_per_node 8 --master_port 23999 eval/eval_openlm_ckpt.py \
#         --checkpoint ../../tmp/logs/$method/checkpoints/epoch_$c.pt \
#         --model ../training/open_lm_configs/open_lm_1b_swiglutorch.json \
#         --config ../tmp/logs/$method/params.txt \
#         --tokenizer ../tokenization_configs/pythia-410m \
#         --eval-yaml eval/heavy.yaml \
#         --output-file results/$method/epoch_$c/metrics_heavy.json \
#         --use-temp-working-dir
# done
