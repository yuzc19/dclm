from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import random
import torch
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from tokenizer import Tokenizer


def encode_pair(tokenizer: Tokenizer, context: str, continuation: str):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tokenizer.encode(context + continuation, bos=False, eos=False).tolist()
    context_enc = tokenizer.encode(context, bos=False, eos=False).tolist()
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    ignore_index: int,
) -> dict:
    context, target = example
    context_enc, continuation_enc = encode_pair(tokenizer, context, target)
    return {
        "input_ids": context_enc + continuation_enc,
        "labels": [ignore_index] * len(context_enc) + continuation_enc,
    }


def prepare(
    destination_path: Path = Path("/home/zichunyu/data"),
    tokenizer_dir: Path = Path("tokenization_configs/pythia-410m"),
    ignore_index: int = -100,
    task_name: str = "eli5",
) -> None:
    random.seed(1234)
    destination_path = destination_path / task_name
    destination_path.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(tokenizer_dir)

    print("Processing train split ...")
    train_set = load_dataset("rexarski/eli5_category", split="train")
    print(len(train_set))
    samples = []
    for data in train_set:
        if data["answers"]["score"][0] >= 5:
            samples.append(
                [
                    data["title"] + " " + data["selftext"],
                    data["answers"]["text"][0],
                ]
            )
    print(len(samples))
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
        )
        for sample in tqdm(samples)
    ]
    random.shuffle(train_set)
    torch.save(train_set, destination_path / "train.pt")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
