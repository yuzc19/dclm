import argparse
import os

import datasets
import torch
from transformers import AutoModel
from modeling_data_influence_model import BertForSequenceClassification


class ModelAnnotator:
    def __init__(self, model_name, device_batch_size):
        self.model_name = model_name
        self.device_batch_size = device_batch_size

        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            problem_type="regression",
            num_labels=1,
        )
        self.model.eval()

        # self.bge = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
        # self.bge.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.model.to(self.device)
        # self.bge.to(self.device)

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "device_batch_size": self.device_batch_size,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @torch.no_grad()
    def __call__(self, example, indices):
        output = {"index": indices}

        outputs = self.model(
            torch.tensor(example["input_ids"], device=self.device),
            attention_mask=torch.tensor(example["attention_mask"], device=self.device),
            token_type_ids=torch.tensor(example["token_type_ids"], device=self.device),
            output_hidden_states=True,
        )
        output["prediction"] = outputs.logits.detach().float().cpu().numpy()

        # for key, value in example.items():
        #     bs = len(value)
        #     example[key] = torch.tensor(value, device=self.device).reshape(bs * 4, -1)
        # outputs = self.bge(
        #     example["input_ids"],
        #     attention_mask=example["attention_mask"],
        #     token_type_ids=example["token_type_ids"],
        #     return_dict=True,
        # )
        # p_reps = outputs.last_hidden_state[:, 0]
        # p_reps = torch.nn.functional.normalize(p_reps, dim=-1).contiguous()
        # p_reps = p_reps.reshape(-1, 4, p_reps.size(1)).mean(dim=1)
        # output["reps"] = p_reps.detach().float().cpu().numpy()

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/zichunyu")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    data_dir = f"/tmp/data/fasttext_0.1/3.6B_data/bert_tokenized"
    model_dir = f"/project/flame/zichunyu/out/10000-data_influence_model-flan"
    output_dir = f"/tmp/data/fasttext_0.1/3.6B_data/10000-data_influence_model-flan-prediction"

    file_list = [
        os.path.abspath(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if not f.startswith(".")
    ]
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
    print("Before annotation: Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator(model_dir, args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
