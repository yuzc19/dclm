# cargo run --release -- \
#     --input /data/datasets/hf_cache/refinedweb_01_0/fasttext/fasttext_filter/processed_data \
#     --local-cell-dir /scratch/zichunyu/tmp \
#     --output /data/datasets/hf_cache/baseline_01_0_fasttext_tokenized \
#     --tokenizer "EleutherAI/gpt-neox-20b" \
#     --seqlen 2049 \
#     --wds-chunk-size 8192 \
#     --num-local-cells 512

cargo run --release -- \
    --input /data/datasets/hf_cache/out/refinedweb_01_0/fasttext/fasttext_filter/10000-data_influence_model-flan-prediction/processed_data \
    --local-cell-dir /scratch/zichunyu/tmp \
    --output /data/datasets/hf_cache/baseline_01_0_fasttext_10000-data_influence_model-flan_tokenized \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --seqlen 2049 \
    --wds-chunk-size 8192 \
    --num-local-cells 512