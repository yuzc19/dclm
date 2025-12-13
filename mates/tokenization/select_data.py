import os
import argparse
import concurrent
from pathlib import Path
from multiprocessing import Pool

import datasets
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from file_utils import read_jsonl, write_jsonl


def mates_select(dataset_size, selection_size, args):
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(f"{args.output_dir}/{i}")
            for i in range(args.shard_num)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    # metrics = np.load(f"{args.output_dir}/prediction.npy")
    print(">> Metrics shape:", metrics.shape)
    metrics = metrics / args.temp
    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng(seed=1234)
    gumbel_noise = rng.gumbel(size=len(metrics))
    print(metrics.mean(), metrics.std(), gumbel_noise.mean(), gumbel_noise.std())
    metrics += gumbel_noise
    return np.argpartition(metrics, selection_size)[:selection_size]


def edu_select(dataset_size, selection_size, args):
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(f"{args.output_dir}/{i}")
            for i in range(args.shard_num)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)
    print(">> Metrics mean:", metrics.mean())
    return np.argpartition(-metrics, selection_size)[:selection_size]


def random_select(dataset_size, selection_size, args):
    rng = np.random.default_rng()
    return rng.choice(dataset_size, size=(selection_size,), replace=False)


METHODS = {
    "random": random_select,
    "fineweb-edu": edu_select,
    "mates": mates_select,
}


def get_indices(dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    select_it = METHODS[args.method]
    ls = select_it(dataset_size, selection_size, args)
    indices = list(map(int, ls))
    return indices


def process_jsonl(file_dir):
    return [d for d in read_jsonl(file_dir)]


def load_dataset(data_dir, shard_names, max_workers=None):
    dataset = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Using executor.map ensures order is preserved
        shards = list(
            tqdm(
                executor.map(process_jsonl, [data_dir.format(n) for n in shard_names]),
                total=len(shard_names),
            )
        )
        for shard in shards:
            dataset.extend(shard)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/zichunyu")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="mates")
    parser.add_argument("--shard_num", type=float, default=8)
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    # args.output_dir = f"{args.base_dir}/out/refinedweb_01_0/fasttext/fasttext_filter/10000-data_influence_model-flan-prediction"
    args.output_dir = f"{args.base_dir}/out/dclm_logs/baseline_01_1_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_2/pairwise-dim-lam-prediction-0"

    out_dir = Path(f"{args.output_dir}/processed_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = f"{args.base_dir}/data/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized"
    file_list = [
        os.path.abspath(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if not f.startswith(".")
    ]
    shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
    file_dir = "/home/zichunyu/data/refinedweb_01_0/fasttext/fasttext_filter/processed_data/{}.jsonl.zstd"

    # shard_sizes = []
    # for shard_name in tqdm(shard_names):
    #     shard_file = file_dir.format(shard_name)
    #     count = sum(1 for _ in read_jsonl(shard_file))
    #     shard_sizes.append(count)
    # dataset_size = sum(shard_sizes)
    # print(f">> Total dataset size: {dataset_size}")

    # 7860915 (from lam and flan)
    dataset_size = 29478616
    selection_size = dataset_size // args.ratio
    indices = get_indices(dataset_size, selection_size, args)
    print(f">> Max index: {max(indices)}")
    np.save(f"{args.output_dir}/indices-{args.ratio}-{args.temp}.npy", indices)
    exit(0)

    selected_indices_set = set(indices)
    global_offset = 0
    cnt = 0
    for shard_i, shard_name in tqdm(enumerate(shard_names)):
        in_file = file_dir.format(shard_name)
        out_file = out_dir / (shard_name + ".jsonl.zstd")
        out_data = []
        for line_idx, line in enumerate(read_jsonl(in_file)):
            global_idx = global_offset + line_idx
            if global_idx in selected_indices_set:
                out_data.append(line)
                cnt += 1
        write_jsonl(out_data, str(out_file))
        global_offset += shard_sizes[shard_i]
    print(cnt)
