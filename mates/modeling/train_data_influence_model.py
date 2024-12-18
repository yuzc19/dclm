import argparse
import os

import datasets
import numpy as np
from modeling_data_influence_model import BertForSequenceClassification
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, Trainer, TrainingArguments


def load_datasets_tar(oracle_dir):
    dataset = datasets.load_from_disk(oracle_dir)
    dataset = dataset.train_test_split(test_size=0.1, seed=1234, shuffle=True)
    train_dataset = dataset["train"].rename_column("input_ids", "ori_input_ids")
    print("Training data size:", len(train_dataset))
    eval_dataset = dataset["test"].rename_column("input_ids", "ori_input_ids")
    return train_dataset, eval_dataset


def load_eval_dataset(oracle_dir):
    dataset = datasets.load_from_disk(oracle_dir)
    dataset = dataset.train_test_split(test_size=0.1, seed=1234, shuffle=True)
    eval_dataset = dataset["test"].rename_column("input_ids", "ori_input_ids")
    return eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b", required=False)
    parser.add_argument("--ckpt", type=int, default=10000, required=False)

    args = parser.parse_args()
    print(args)

    train_dataset, eval_dataset = load_datasets_tar(
        f"output/oracle/{args.model_name}/{args.ckpt}/bs-1-sample"
        # f"/data/datasets/hf_cache/out/oracle/{args.model_name}/{args.ckpt}"
    )
    # eval_dataset = load_eval_dataset(
    #     f"output/oracle/{args.model_name}/{args.ckpt}/bs-1-sample"
    # )
    mean_value = np.mean(np.array(train_dataset["scores"])[:, 0])
    std_value = np.std(np.array(train_dataset["scores"])[:, 0])
    print(np.array(train_dataset["scores"])[:, 0].shape, mean_value, std_value)

    # Load pythia tokenizer
    # We are using 2048 right now
    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        max_length=2048,
        padding="max_length",
    )

    # Preprocess data function
    def preprocess_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["ori_input_ids"], skip_special_tokens=True
        )
        enc = tokenizer.batch_encode_plus(
            texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        # Convert the labels to float for regression
        scores = examples["scores"]
        enc["labels"] = [(float(s[0]) - mean_value) / std_value for s in scores]
        return enc

    # Process and encode the datasets
    train_dataset = train_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=["ori_input_ids", "scores"],
    )
    eval_dataset = eval_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=["ori_input_ids", "scores"],
    )
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # Load model for sequence classification with a regression head
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="regression",
        num_labels=1,
        cache_dir="/data/datasets/hf_cache",
    )

    # Training arguments
    batch_size = 16

    args = TrainingArguments(
        f"output/data_influence_model/{args.model_name}/{args.ckpt}",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        logging_steps=10,
        eval_steps=50,
        save_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        bf16=True,
        report_to="none",
    )

    # Define regression metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions[:, 0]
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model()

    # Evaluate the best model
    eval_results = trainer.evaluate()

    # Print the evaluation results
    print("Best evaluation results:", eval_results)
