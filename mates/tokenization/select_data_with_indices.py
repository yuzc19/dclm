import os
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from file_utils import read_jsonl, write_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/datasets/hf_cache")
    parser.add_argument("--ratio", type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    args.output_dir = f"{args.base_dir}/out/refinedweb_01_0/fasttext/fasttext_filter/1b-data_influence_model-prediction"

    data_dir = f"{args.base_dir}/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized"
    file_list = [
        os.path.abspath(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if not f.startswith(".")
    ]
    shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
    file_dir = "/data/datasets/hf_cache/refinedweb_01_0/fasttext/fasttext_filter/processed_data/{}.jsonl.zstd"

    out_dir = Path(f"{args.output_dir}/processed_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_sizes = []
    for shard_name in tqdm(shard_names):
        shard_file = file_dir.format(shard_name)
        count = sum(1 for _ in read_jsonl(shard_file))
        shard_sizes.append(count)
    dataset_size = sum(shard_sizes)
    print(f">> Total dataset size: {dataset_size}")

    indices = np.load(os.path.join(args.output_dir, f"selected_indices_{args.ratio}.npy"))
    print(f">> Max index: {max(indices)}")

    selected_indices_set = set(indices)
    global_offset = 0
    for shard_i, shard_name in tqdm(enumerate(shard_names)):
        in_file = file_dir.format(shard_name)
        out_file = out_dir / (shard_name + ".jsonl.zstd")
        out_data = []
        for line_idx, line in enumerate(read_jsonl(in_file)):
            global_idx = global_offset + line_idx
            if global_idx in selected_indices_set:
                out_data.append(line)
        write_jsonl(out_data, str(out_file))
        global_offset += shard_sizes[shard_i]
