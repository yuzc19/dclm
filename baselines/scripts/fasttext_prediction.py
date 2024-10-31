import os

import datasets
import fasttext
import numpy as np
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

use_seq = True

base_dir = "/data/users/zichunyu"

if use_seq:
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"{base_dir}/manifold/scaling_mates/tc_out/step-10000/pythia-1b"
                + f"/{i}"
            )
            for i in range(128)
        ]
    )
    output_dir = "./fasttext-oh-eli5-prediction"
    os.makedirs(output_dir, exist_ok=True)
    print("Total number of examples:", len(dataset))

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("tokenization_configs/pythia-410m")
    # Finally, we observe that using a fairly strict threshold, which keeps the top-10% of examples, helps
    # over more permissive top-15% and top-20% thresholds. Larger -> Better
    model_dir = f"{base_dir}/tmp/quality_prediction_enrichment_models/fasttext_oh_eli5.bin"
    model = fasttext.load_model(model_dir)

    def process_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["input_ids"],
            skip_special_tokens=True,
        )
        texts = [t.replace("\n", " ") for t in texts]
        pred = []
        for l, s in zip(*model.predict(texts, k=-1)):
            pred_map = {}
            pred_map[l[0]] = s[0]
            pred_map[l[1]] = s[1]
            pred.append(pred_map["__label__hq"])
        return {"prediction": pred}


dataset = dataset.map(
    process_data,
    batched=True,
    batch_size=1024,
    num_proc=96,
)
print("After scoring: Total number of examples:", len(dataset))

print(f"Saving to {output_dir}")
dataset.save_to_disk(output_dir)

# PearsonRResult(statistic=0.12810353735354094, pvalue=0.0) SignificanceResult(statistic=0.15977095479526213, pvalue=0.0)
print(
    pearsonr(
        np.array(dataset["prediction"]).reshape(-1),
        -np.array(dataset["scores"])[:, 0].reshape(-1),
    ),
    spearmanr(
        np.array(dataset["prediction"]).reshape(-1),
        -np.array(dataset["scores"])[:, 0].reshape(-1),
    ),
)
