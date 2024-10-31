### Environment

Important package version:

- nltk==3.8.1
- boto3==1.26.145
- moto==5.0.11

Run setup.py to download necessary files:

```bash
python setup.py install
```

Go to: http://aws.amazon.com/
Sign Up & create a new account (they'll give you the option for 1 year trial or similar)
Go to your AWS account overview
Account menu in the upper-right (has your name on it)
sub-menu: Security Credentials

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws configure

# DCLM-refinedweb	(one local shard = 405.6 GiB), so the total is around 41.6 TiB, while the entire is 279.6 TiB
# DCLM-baseline	6.6 TiB
# 1: 240T tokens, 279.6 TiB (5,047,684), 340TB, 370TB after gzip compression
# 2: 16.98T tokens (357,163)
# 3: 100B tokens (needed for 400M-1x at 2)
# 4: 8.2B tokens (final 400M-1x)
# The files in each shard were shuffled before the dataset was split into shards. The documents within each file were not further shuffled - this global shuffle occurs later in our pipeline, after filtering and tokenization of the dataset. If global shuffle before tokenization across all the documents is required by your processing scheme, make sure to take this into account.
# The documents were initially written into files as the were being read and processed from the CommonCrawl WARC files, so there was indeed no shuffling at this initial stage. After the files were created, we shuffled them (at the file level) and then split them into shards. However, because shuffling never happened at the document level at this stage, picking e.g. 300M documents at random from the entire dataset is not exactly the same as picking one shard.
with-proxy aws s3 ls --summarize --human-readable --recursive s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10/
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


- "output/cc_wet_2019_april_baselines/refinedweb/refinedweb/processed_data"


```
import os

import datasets

# /data/users/zichunyu/data/hf_cache/mlfoundations___json/mlfoundations--dclm-pool-400m-1x-36757e8d7b7ffd23/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51
datasets.load_dataset(
    "mlfoundations/dclm-pool-400m-1x",
    cache_dir="data",
    num_proc=os.cpu_count() - 1,
)

```
