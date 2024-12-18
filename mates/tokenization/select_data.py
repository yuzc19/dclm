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
    print(">> Metrics shape:", metrics.shape)
    metrics = metrics / args.temp
    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise
    return np.argpartition(metrics, selection_size)[:selection_size]


def random_select(dataset_size, selection_size, args):
    rng = np.random.default_rng()
    return rng.choice(dataset_size, size=(selection_size,), replace=False)


METHODS = {
    "random": random_select,
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
    parser.add_argument("--base_dir", type=str, default="/data/datasets/hf_cache")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--shard_num", type=float, default=16)
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    args.output_dir = f"{args.base_dir}/out/refinedweb_01_0/fasttext/fasttext_filter/10000-data_influence_model-flan-prediction"

    data_dir = f"{args.base_dir}/refinedweb_01_0/fasttext/fasttext_filter/processed_data"
    file_list = [
        os.path.abspath(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if not f.startswith(".") and not os.path.isdir(os.path.join(data_dir, f))
    ]
    shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
    shard_size = len(file_list) // args.shard_num
    file_dir = "/data/datasets/hf_cache/refinedweb_01_0/fasttext/fasttext_filter/processed_data/{}"

    out_dir = Path(f"{args.output_dir}/processed_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(file_dir, shard_names, max_workers=16)
    dataset_size = len(dataset)
    print(f">> Dataset size: {dataset_size}")
    selection_size = dataset_size // args.ratio
    indices = get_indices(dataset_size, selection_size, args)
    np.save(args.output_dir + "/mates-indices.npy", indices)
    print(f">> Max index: {max(indices)}")

    with Pool(16) as pool:
        process_args = [
            (
                [dataset[i] for i in indices[i::16]],
                f"{out_dir}/{i}_processed.jsonl.zstd",
            )
            for i in range(16)
        ]

        pool.starmap(write_jsonl, process_args)
        pool.close()
        pool.join()
