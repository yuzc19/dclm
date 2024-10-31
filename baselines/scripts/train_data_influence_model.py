import argparse
import os

import datasets
import numpy as np
from .modeling_data_influence_model import BertForSequenceClassification
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, Trainer, TrainingArguments


# 'eval_spearman': 0.2391423268891367
def load_datasets_tar(oracle_dir):
    # step-1, 3153 for each
    # step-10000, 10000 for each
    # step-10000-10BT, 5000 for each
    # step-10000-100BT, 3153 for each
    dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(f"{oracle_dir}/{i}") for i in range(16)]
    )

    dataset = dataset.train_test_split(test_size=0.1, seed=1234, shuffle=True)
    train_dataset = dataset["train"].rename_column("input_ids", "ori_input_ids")
    print("Training data size:", len(train_dataset))
    eval_dataset = dataset["test"].rename_column("input_ids", "ori_input_ids")
    return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-410m", required=False)
    parser.add_argument("--ckpt", type=int, default=80000, required=False)

    args = parser.parse_args()
    print(args)

    train_dataset, eval_dataset = load_datasets_tar(
        # f"out/step-10000-flan"
        # f"out/step-{args.ckpt}-bs-1/pythia-1b"
        f"../manifold/scaling_mates/tc_out/step-{args.ckpt}-bs-1-sample/pythia-1b"
    )
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
        encoding = tokenizer.batch_encode_plus(
            texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        # Convert the labels to float for regression
        encoding["labels"] = [
            (float(score[0]) - mean_value) / std_value for score in examples["scores"]
        ]
        return encoding

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
        # f"/data/users/zichunyu/out/{args.model_name}/fineweb/sample-100BT/{args.ckpt}-data_influence_model-flan",
        problem_type="regression",
        num_labels=1,
    )

    # Training arguments
    batch_size = 32

    args = TrainingArguments(
        # accelerate 0.22.0
        # bs-1 (10000): 'eval_pearson': 0.4484906923255437, 'eval_spearman': 0.6899116513368633
        # bs-1 (20000): 'eval_pearson': 0.6661371438984552, 'eval_spearman': 0.683807173880882,
        f"/data/users/zichunyu/out/{args.model_name}/fineweb/sample-100BT/{args.ckpt}-data_influence_model-flan-bs-1-sample",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
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
