# mkdir -p /tmp/dataset
# gcloud storage cp -r gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/textfiles /tmp/dataset
# mkdir -p /tmp/data/fasttext_0.1/more_data/ && for f in /tmp/data/fasttext_0.1/processed_data/shard_*_processed.jsonl.zstd; do num=$(basename "$f" | sed 's/shard_0*\([0-9]*\)_processed.jsonl.zstd/\1/'); if [ "$num" -lt 180 ] || [ "$num" -gt 299 ]; then mv "$f" /tmp/data/fasttext_0.1/more_data/; fi; done

# 128 files = 7661393361 (original)
# 128 files = 4924417023 (model=Qwen3-30B-A3B-FP8, prompt=paraphrase)
# 176 files = 7057772304 (model=Qwen3-30B-A3B-FP8, prompt=paraphrase)
# 288 files = 8166875514 (model=Qwen3-1.7B sft by GPT-4o, prompt=paraphrase)
# 100 files = 5298330837 (model=Qwen3-30B-A3B-FP8, prompt=delete)
# 3933920178 (model=Qwen3-1.7B sft by GPT-4o-mini)

cargo run --release -- \
    --input /tmp/data/fasttext_0.1/fasttext_filter/processed_data \
    --local-cell-dir /tmp \
    --output /tmp/data/fasttext_0.1/tokenized_3.6B \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --use-tiktoken \
    --seqlen 2049 \
    --wds-chunk-size 8192 \
    --num-local-cells 512

# delete from 14.4B to 9.9B
# gcloud storage cp -r /tmp/fasttext_14.4B gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/Qwen3-4B-grpo-1020/