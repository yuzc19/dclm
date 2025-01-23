import argparse
import os
from dataclasses import dataclass

import datasets
import numpy as np
import torch
from datasets import Features, Sequence, Value
from modeling_seq_data_influence_model import BiEncoderModel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    def __call__(self, features):
        query = [f["query"] for f in features]
        passage = [f["passage"] for f in features]
        score = [f["score"] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer.batch_encode_plus(
            query,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.batch_encode_plus(
            passage,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for key, value in q_collated.items():
            bs = value.shape[0]
            q_collated[key] = value.reshape(bs * 4, -1)
        for key, value in d_collated.items():
            bs = value.shape[0]
            d_collated[key] = value.reshape(bs * 4, -1)

        return {
            "query": q_collated,
            "passage": d_collated,
            "label": torch.tensor(score, dtype=torch.float32),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b", required=False)
    parser.add_argument("--ckpt", type=int, default=10000, required=False)
    parser.add_argument("--temp", type=float, default=1.0, required=False)

    args = parser.parse_args()
    print(args)

    model_name = "BAAI/bge-base-en-v1.5"
    model = BiEncoderModel(model_name=model_name, temperature=args.temp)

    # torchrun --nproc-per-node 8 mates/modeling/train_pairwise_model.py
    args = TrainingArguments(
        "/home/zichunyu/out/oracle/pythia-410m/epoch_1/bs-1-sample/pairwise-dim-flan",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        # warmup_ratio=0.03,
        warmup_steps=50,
        logging_steps=5,
        eval_steps=50,
        save_steps=5000,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        bf16=True,
        report_to="wandb",
        run_name=f"pairwise-data-influence-model_bge-base_temp={args.temp}",
        remove_unused_columns=False,
    )

    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                # f"/home/zichunyu/out/oracle/pythia-410m/epoch_1/{i}"
                f"/home/zichunyu/out/oracle/pythia-410m/epoch_1/bs-1-sample/{i}"
            )
            # for i in [0, 4, 5, 7]
            for i in range(8)
        ]
    )
    mean_value = np.mean([s[1] for s in np.array(dataset["scores"])])
    std_value = np.std([s[1] for s in np.array(dataset["scores"])])
    print(mean_value, std_value)

    def preprocess_data(examples):
        queries = [input_ids[2048:] for input_ids in examples["input_ids"]]
        passages = [input_ids[:2048] for input_ids in examples["input_ids"]]
        queries = pythia_tokenizer.batch_decode(queries, skip_special_tokens=True)
        passages = pythia_tokenizer.batch_decode(passages, skip_special_tokens=True)
        scores = [(s[1] - mean_value) / std_value for s in examples["scores"]]
        return {"query": queries, "passage": passages, "score": scores}

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(test_size=1000, seed=1234, shuffle=True)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.select(range(0, len(train_dataset), 2))
    print("Training data size:", len(train_dataset))
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = EmbedCollator(tokenizer)

    def compute_metrics(eval_pred):
        print(model.temp)

        predictions, labels = eval_pred
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model()

    # Evaluate the best model
    eval_results = trainer.evaluate()

    # Print the evaluation results
    print("Best evaluation results:", eval_results)