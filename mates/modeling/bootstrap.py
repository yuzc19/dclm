import argparse
import fsspec
import os

import torch
import datasets
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from modeling_seq_data_influence_model import RolloutModel

model_dir = "/project/flame/zichunyu/out/oracle/pythia-410m/epoch_2/rollout-dim-flan-r10"
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


def beam_search_select(
    reps,
    prediction,
    num_selected=10000,
    num_rollouts=10,
    num_beams=5,
    order=1,
):
    """
    Implements beam search to select indices based on representation diversity and prediction scores.

    Args:
        reps: Numpy array of representations with shape [N, dim]
        prediction: Numpy array of prediction scores with shape [N]
        num_selected: Number of candidates to retain at each step (default: 1000)
        num_rollouts: Number of selection iterations (default: 10)
        num_beams: Number of top indices to consider at each step (default: 5)

    Returns:
        List of selected indices with length num_rollouts
    """
    temp = float(model.temp)
    alpha = float(model.alpha)
    print(f">> Using beam width of {num_beams}")
    print(">> Reps shape:", reps.shape)
    print(">> Prediction shape:", prediction.shape)

    # Start with a single empty beam
    candidates = [{"indices": [], "avg_rel": np.zeros(reps.shape[0]), "score": 0.0}]

    # Iterate for num_rollouts steps
    for i in range(num_rollouts):
        print(f"Iteration {i+1}/{num_rollouts}")
        new_candidates = []

        # For each existing candidate
        for candidate in candidates:
            current_indices = candidate["indices"]
            current_avg_rel = candidate["avg_rel"]
            current_score = candidate["score"]

            # Calculate scores for all possible next indices
            scores = current_avg_rel * prediction
            # Mask already selected indices
            scores[current_indices] = order * float("inf")

            # Find the top num_beams indices to add
            top_indices = np.argsort(order * scores)[:num_beams]

            # Create new candidates for each of the top indices
            for selected_index in top_indices:
                # Update relevance with this new index
                cur_rel = alpha * (1 - np.matmul(reps, reps[selected_index].transpose()) / temp)
                new_avg_rel = (i * current_avg_rel + cur_rel) / (i + 1)

                # Create new candidate
                new_indices = current_indices.copy()
                new_indices.append(selected_index)
                new_score = current_score + scores[selected_index]

                new_candidates.append(
                    {
                        "indices": new_indices,
                        "avg_rel": new_avg_rel,
                        "score": new_score,
                    }
                )

        # Select the top num_selected candidates
        new_candidates.sort(key=lambda x: order * x["score"])
        candidates = new_candidates[:num_selected]

    # Return the indices from the best candidate
    return candidates


def group_select(reps, prediction, num_rollouts=10):
    temp = model.temp
    alpha = model.alpha
    print(">> Reps shape:", reps.shape)
    print(">> Prediction shape:", prediction.shape)

    selected_indices = []
    avg_rel = np.zeros(reps.shape[0])
    for i in range(num_rollouts):
        scores = avg_rel * prediction
        scores[selected_indices] = float("inf")
        selected_index = np.argmin(scores)
        selected_indices.append(selected_index)
        cur_rel = alpha * (1 - np.matmul(reps, reps[selected_index].transpose()) / temp)
        avg_rel = (i * avg_rel + cur_rel) / (i + 1)
    return selected_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print(args)

    if os.path.exists(f"{model_dir}-prediction-shard_0-9"):
        dataset = Dataset.load_from_disk(f"{model_dir}-prediction-shard_0-9")
    else:
        probe_data = "/project/flame/zichunyu/data/shard_0-9.pt"
        dataset = torch.load(probe_data)
        bert_dataset = []
        for data in tqdm(dataset):
            texts = pythia_tokenizer.batch_decode(data, skip_special_tokens=True)
            enc = tokenizer.batch_encode_plus(
                texts,
                max_length=2048,
                padding="max_length",
                truncation=True,
            )
            bert_dataset.append(enc)
        dataset = Dataset.from_list(bert_dataset)
        print("Before annotation: Total number of examples:", len(dataset))

        dataset = dataset.map(
            lambda x: (lambda r, p: {"reps": r, "prediction": p})(*embed_batch(x)),
            batched=True,
            # with_indices=True,
            batch_size=128,
            remove_columns=dataset.column_names,
        )
        print("After annotation: Total number of examples:", len(dataset))
        dataset.save_to_disk(f"{model_dir}-prediction-shard_0-9")

    candidates = beam_search_select(
        np.array(dataset["reps"]),
        np.array(dataset["prediction"]).reshape(-1),
    )
    candidates.extend(
        beam_search_select(
            np.array(dataset["reps"]),
            np.array(dataset["prediction"]).reshape(-1),
            order=-1,
        )
    )
    scores = np.array([i["score"] for i in candidates])
    print(scores[:50])
    bootstrap_indices = np.array([i["indices"] for i in candidates])
    print(">> Bootstrap indices shape:", bootstrap_indices.shape)
    np.save(f"{model_dir}-bootstrap.npy", bootstrap_indices)
