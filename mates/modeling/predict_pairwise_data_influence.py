import argparse
import fsspec
import os

import torch
import datasets
from datasets import Dataset
from transformers import AutoTokenizer
from modeling_seq_data_influence_model import BiEncoderModel, RolloutModel

# model_dir = "/home/zichunyu/out/dclm_logs/baseline_01_1_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000/checkpoints/epoch_2/pairwise-dim-flan"
# model_dir = "/home/zichunyu/out/dclm_logs/baseline_01_1_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_2/pairwise-dim-flan"
# model = BiEncoderModel.from_pretrained(model_dir)

model_dir = "/project/flame/zichunyu/out/oracle/pythia-410m/epoch_2/rollout-dim-flan-r10-bs1"
model = RolloutModel.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device)

print(model.temp, model.alpha)

pythia_tokenizer = AutoTokenizer.from_pretrained(
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
)
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    max_length=2048,
    padding="max_length",
)


@torch.no_grad()
def embed_batch(example):
    for key, value in example.items():
        bs = len(value)
        example[key] = torch.tensor(value, device="cuda").reshape(bs * 4, -1)
    outputs = model.model(
        example["input_ids"],
        attention_mask=example["attention_mask"],
        token_type_ids=example["token_type_ids"],
        return_dict=True,
    )
    p_reps = outputs.last_hidden_state[:, 0]
    p_reps = torch.nn.functional.normalize(p_reps, dim=-1).contiguous()
    p_reps = p_reps.reshape(-1, 4, p_reps.size(1)).mean(dim=1)
    pooled_output_p = model.dropout(p_reps)
    scores = model.classifier(pooled_output_p).squeeze()
    return p_reps.detach().float().cpu().numpy(), scores.detach().float().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/zichunyu")
    parser.add_argument("--model_name", type=str, default="pythia-410m")
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)
    parser.add_argument("--bootstrap", action="store_true")

    args = parser.parse_args()
    print(args)

    data_dir = f"{args.base_dir}/data/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized"
    output_dir = f"{model_dir}-prediction-0"

    if not args.bootstrap:
        # file_list = [
        #     os.path.abspath(os.path.join(data_dir, f))
        #     for f in os.listdir(data_dir)
        #     if not f.startswith(".")
        # ]
        # fs = fsspec.filesystem("gcs")
        fs = fsspec.filesystem("local")
        file_list = fs.glob(data_dir + "/*")
        shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
        shard_size = len(file_list) // args.shard[1]
        print(
            args.shard[0] * shard_size,
            (
                (args.shard[0] + 1) * shard_size
                if args.shard[0] + 1 < args.shard[1]
                else len(file_list)
            ),
        )
        dataset = datasets.concatenate_datasets(
            [
                # datasets.load_from_disk("gs://" + file_list[i])
                datasets.load_from_disk(file_list[i])
                for i in range(
                    args.shard[0] * shard_size,
                    (
                        (args.shard[0] + 1) * shard_size
                        if args.shard[0] + 1 < args.shard[1]
                        else len(file_list)
                    ),
                )
            ]
        )
    else:
        probe_data = "/data/datasets/hf_cache/dclm-baseline-1.0/global-shard_01_of_10/local-shard_0_of_10/pythia_tokenized/train.pt"
        dataset = torch.load(probe_data)
        shard_size = len(dataset) // args.shard[1]
        low, high = args.shard[0] * shard_size, (
            (args.shard[0] + 1) * shard_size
            if args.shard[0] + 1 < args.shard[1]
            else len(dataset)
        )
        print(low, high)
        dataset = dataset[low:high]
        bert_dataset = []
        for data in dataset:
            texts = pythia_tokenizer.batch_decode(data, skip_special_tokens=True)
            enc = tokenizer.batch_encode_plus(
                texts,
                max_length=2048,
                padding="max_length",
                truncation=True,
            )
            bert_dataset.append(enc)
        dataset = Dataset.from_list(bert_dataset)
        dataset.save_to_disk(data_dir + f"/{args.shard[0]}")

    print("Before annotation: Total number of examples:", len(dataset))

    dataset = dataset.map(
        lambda x: (lambda r, p: {"reps": r, "prediction": p})(*embed_batch(x)),
        batched=True,
        # with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
