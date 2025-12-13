import os
import json
import glob
import gzip
import torch
import random
from tqdm import tqdm
import webdataset as wds
from multiprocessing import Pool
from transformers import AutoTokenizer

# input_dir = "gs://cmu-gpucloud-zichunyu/data/baseline_01_0_fasttext_tokenized"
input_dir_1 = "gs://cmu-gpucloud-zichunyu/data/baseline_01_0_fasttext_fineweb-edu_tokenized"
input_dir_2 = "gs://cmu-gpucloud-zichunyu/data/baseline_01_0_fasttext_epoch_2-data_influence_model-flan-group_tokenized"
output_dir = "."


def test_tokenize_shuffle_simple_do_sample():
    dss = [
        wds.WebDataset(os.path.join(input_dir_1, f"shard_{i:08d}.tar")).decode()
        for i in range(50)
    ]
    dss += [
        wds.WebDataset(os.path.join(input_dir_2, f"shard_{i:08d}.tar")).decode()
        for i in range(50)
    ]
    data = [torch.tensor([x["json.gz"]], dtype=torch.int32) for ds in dss for x in ds]
    print(len(data))
    random.seed(42)
    random.shuffle(data)
    torch.save(data, os.path.join(output_dir, "shard_0-49.pt"))


test_tokenize_shuffle_simple_do_sample()
