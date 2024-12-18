import torch
from tqdm import tqdm
from datasets import Dataset
from litdata import optimize
from functools import partial
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from file_utils import list_dir, read_jsonl
from litdata.streaming import StreamingDataset, TokensLoader

data_dir = "data/refinedweb_01_0/fasttext/fasttext_filter/processed_data"
file_list = list_dir(data_dir)

dataset = []
for file in tqdm(file_list):
    shard_name = file.split("/")[-1].split(".")[0]
    for json_line in read_jsonl(file):
        dataset.append(" ".join(json_line["text"].strip().splitlines()))
dataset = Dataset.from_list([{"text": d} for d in dataset])
print("Total examples:", len(dataset))

tokenizer = Tokenizer("checkpoints/EleutherAI/pythia-1b")


def tokenize(data: Dataset, index: int):
    yield tokenizer.encode(data[index]["text"], eos=True)


output_dir = data_dir + "/pythia_tokenized"
optimize(
    fn=partial(tokenize, dataset),
    inputs=list(range(len(dataset))),
    output_dir=output_dir,
    num_workers=8,
    chunk_bytes="200MB",
)
train_dataset = StreamingDataset(
    input_dir=str(output_dir),
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
torch.save(out_data, output_dir + "/train.pt")
