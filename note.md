### Environment

Important package version:

- nltk==3.8.1
- boto3==1.26.145
- moto==5.0.11

Run setup.py to download necessary files:

```bash
python setup.py install
```

### Ray

To launch a local ray cluster, use the following command:

```bash
ray start --head --port 6379
```

> TODO: Have not figured out how to set local `cluster_config` for `ray up`. Not sure if needed.

### Step1: Heuristic cleaning

Download [one test file](https://huggingface.co/datasets/mlfoundations/dclm-pool-400m-1x/blob/main/CC_shard_00000000.jsonl.zst) into the local directory.

Create `exp_data/datasets/raw_sources/test.json`:

```json
{
    "uuid": "f12dc026-cc4a-4203-ba9f-9ba08c5945f9",
    "name": "CC_shard_00000000",
    "dataset_url": "data/downloads/",
    "manifest_url": null,
    "sources": [],
    "tokenized": false,
    "tokenizer": null,
    "num_tokens": null,
    "dcnlp_commit_hash": null,
    "dcnlp_diff": null,
    "data_key": "jsonl.zst"
}
```

For local testing, change

- `shard_files = list_shard_files(working_dir, args.num_shards, args.shard_list_file)` to `shard_files = ["CC_shard_00000000.jsonl.zst"]` in `ray_processing/process.py`

> TODO: The original version only supports aws?

- `dataset_url` in `exp_data/datasets/raw_sources/test.json` to the local directory your data is stored
- `get_s3_dir_size` in `ray_processing/utils.py` to a local version

```bash
python ray_processing/process.py \
  --source_ref_paths exp_data/datasets/raw_sources/test.json \
  --readable_name refinedweb \
  --output_dir output/CC_shard_00000000/refinedweb/ \
  --config_path baselines/baselines_configs/refinedweb.yaml \
  --source_name cc \
  --overwrite
```

### Step2: Deduplication

> TODO: Use Rust-based [BFF](https://github.com/revbucket/bff/tree/ai2-fuzzy-substr/).

### Step3: Model-based ltering

Similar to step1, but use `baselines/baselines_configs/fasttext_filter.yaml` instead.
