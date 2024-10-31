import argparse
import os

import datasets
import numpy as np
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from tokenizer import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer


def mates_select(dataset, dataset_size, selection_size, args):
    metrics = np.array(dataset["scores"])[:, 0].reshape(-1)
    print(metrics.shape)

    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))

    # 0.1513 for temp=1, 0.2866 for temp=0.5, 0.4946 from temp=0.25, 0.7930 from temp=0.1
    print(spearmanr(metrics * 10, metrics * 10 + gumbel_noise))

    indices = np.argsort(metrics)
    indices = np.concatenate([indices[:selection_size], indices[-selection_size:]])

    return indices


METHODS = {"mates": mates_select}


def get_indices(dataset, dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    select_it = METHODS[args.method]
    ls = select_it(dataset, dataset_size, selection_size, args)
    indices = list(map(int, ls))
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="mates")
    parser.add_argument("--size", type=int, default=40000)
    parser.add_argument("--ckpt", type=int, default=10000)

    args = parser.parse_args()
    print(args)

    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                "/data/users/zichunyu/manifold/scaling_mates/tc_out/step-10000/pythia-1b"
                + f"/{i}"
            )
            for i in range(128)
        ]
    )

    dataset_size = len(dataset)
    print(f">> Dataset size: {dataset_size}")
    pythia_tokenizer = AutoTokenizer.from_pretrained("tokenization_configs/pythia-410m")

    def preprocess_data(examples):
        examples["text"] = pythia_tokenizer.batch_decode(
            examples["input_ids"],
            skip_special_tokens=True,
        )
        return examples

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=["input_ids"],
    )

    indices = get_indices(dataset, dataset_size, args.size, args)

    f = open(f"fasttext_train_data_oracle-{args.size}", "w")
    for cnt, i in tqdm(enumerate(indices)):
        if cnt < args.size:
            f.write("__label__hq " + dataset[i]["text"])
            f.write("\n")
        else:
            f.write("__label__lq " + dataset[i]["text"])
            f.write("\n")
