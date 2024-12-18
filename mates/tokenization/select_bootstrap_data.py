import os
import argparse
from functools import partial

import torch
import datasets
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from litdata import optimize
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from file_utils import read_jsonl, write_jsonl
from litdata.streaming import StreamingDataset, TokensLoader


def mates_select(dataset_size, selection_size, args):
    dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(f"{args.output_dir}/{i}") for i in range(8)]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(metrics.shape)

    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise

    indices = np.argsort(metrics)
    indices = np.concatenate([indices[:5000], indices[-5000:]])

    return indices


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


def are_tensors_equal(list1: list[torch.Tensor], list2: list[torch.Tensor]) -> bool:
    if len(list1) != len(list2):
        return False

    for tensor1, tensor2 in zip(list1, list2):
        if not torch.equal(tensor1, tensor2):
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/datasets/hf_cache")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="mates")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--use_doc", action="store_true")

    args = parser.parse_args()
    print(args)

    args.output_dir = f"{args.base_dir}/out/data_influence_model/pythia-1b/10000-prediction"

    data_dir = f"{args.base_dir}/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized"
    file_list = [
        os.path.abspath(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if not f.startswith(".")
    ]
    file_list = file_list[:300]
    file_dir = "/data/datasets/hf_cache/refinedweb_01_0/fasttext/fasttext_filter/processed_data/{}.jsonl.zstd"

    dataset = []
    for file in tqdm(file_list):
        shard_name = file.split("/")[-1].split(".")[0]
        for json_line in read_jsonl(file_dir.format(shard_name)):
            dataset.append(" ".join(json_line["text"].strip().splitlines()))
    dataset = Dataset.from_list([{"text": d} for d in dataset])
    print("Total examples:", len(dataset))

    indices = get_indices(len(dataset), 0, args)
    print(f">> Max index: {max(indices)}")

    tokenizer = Tokenizer("../lit-gpt/checkpoints/EleutherAI/pythia-1b")

    def tokenize(data: Dataset, index: int):
        yield tokenizer.encode(data[index]["text"], eos=True)

    optimize(
        fn=partial(tokenize, dataset),
        inputs=indices,
        output_dir=f"{args.base_dir}/data/data_influence_model/pythia-1b/10000/bs-1-sample",
        num_workers=(os.cpu_count() // 8),
        chunk_bytes="200MB",
    )
    train_dataset = StreamingDataset(
        input_dir=f"{args.base_dir}/data/data_influence_model/pythia-1b/10000/bs-1-sample",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    print(len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        pin_memory=True,
    )
    train_iter = iter(train_dataloader)

    out_data = []
    for train_data in tqdm(train_iter):
        out_data.append(train_data)
    torch.save(
        out_data,
        f"{args.base_dir}/data/data_influence_model/pythia-1b/10000/bs-1-sample"
        + "/train.pt",
    )
