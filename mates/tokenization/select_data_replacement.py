import os
import argparse
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm
from file_utils import read_jsonl, write_jsonl


def mates_select(selection_size, args):
    """
    Select samples with replacement using MATES algorithm.
    Uses Gumbel-Top-k for sampling with replacement.

    Note: Lower scores are better (more likely to be sampled).
    """
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(f"{args.scores_dir}/{i}")
            for i in range(args.shard_num)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)
    print(">> Metrics mean:", metrics.mean())
    print(">> Metrics std:", metrics.std())

    # This converts "lower is better" to "higher is better" for Gumbel sampling
    metrics = -metrics / args.temp

    # Sample with replacement using categorical distribution
    rng = np.random.default_rng(seed=args.seed)

    # Convert to probabilities using softmax
    # Subtract max for numerical stability
    metrics_shifted = metrics - np.max(metrics)
    exp_metrics = np.exp(metrics_shifted)
    probabilities = exp_metrics / np.sum(exp_metrics)

    print(f">> Sampling {selection_size} indices with replacement...")
    # Use numpy's choice which is much faster
    sampled_indices = rng.choice(
        len(metrics),
        size=selection_size,
        replace=True,
        p=probabilities
    )

    return sampled_indices, metrics


def process_jsonl(file_dir):
    return [d for d in read_jsonl(file_dir)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textfiles_dir", type=str)
    parser.add_argument("--scores_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--shard_num", type=int, default=8)
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()
    print(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get list of shard files
    file_list = [
        os.path.abspath(os.path.join(args.textfiles_dir, f))
        for f in os.listdir(args.textfiles_dir)
        if f.endswith(".jsonl.zstd") and not f.startswith(".")
    ]
    shard_names = [os.path.basename(file).replace(".jsonl.zstd", "") for file in file_list]
    print(f">> Found {len(shard_names)} shards")

    # Count shard sizes
    shard_sizes = []
    for shard_name in tqdm(shard_names, desc="Counting shards"):
        shard_file = os.path.join(args.textfiles_dir, f"{shard_name}.jsonl.zstd")
        count = sum(1 for _ in read_jsonl(shard_file))
        shard_sizes.append(count)

    dataset_size = sum(shard_sizes)
    print(f">> Total dataset size: {dataset_size}")

    selection_size = dataset_size // args.ratio
    print(f">> Selection size: {selection_size}")

    # Get indices with replacement
    indices, metrics = mates_select(selection_size, args)
    print(f">> Selected {len(indices)} indices")
    print(f">> Unique indices: {len(set(indices))}")

    # Count occurrences of each index
    from collections import Counter

    index_counts = Counter(indices)

    # Save metrics to numpy file
    metrics_file = out_dir / "metrics.npy"
    np.save(metrics_file, metrics)
    print(f">> Saved metrics to {metrics_file}")

    # Save index counts as a numpy array
    # Create an array where index i contains the count for that index
    index_counts_array = np.zeros(dataset_size, dtype=np.int32)
    for idx, count in index_counts.items():
        index_counts_array[idx] = count

    index_counts_file = out_dir / "index_counts.npy"
    np.save(index_counts_file, index_counts_array)
    print(f">> Saved index counts to {index_counts_file}")

    # Process each shard and write selected samples
    global_offset = 0
    total_written = 0

    for shard_i, shard_name in tqdm(enumerate(shard_names), desc="Processing shards"):
        in_file = os.path.join(args.textfiles_dir, f"{shard_name}.jsonl.zstd")
        out_file = out_dir / f"{shard_name}.jsonl.zstd"

        # Load shard data
        shard_data = list(read_jsonl(in_file))
        shard_size = len(shard_data)

        # Collect samples for this shard (with repetitions)
        out_data = []
        for line_idx in range(shard_size):
            global_idx = global_offset + line_idx
            if global_idx in index_counts:
                # Add this sample multiple times based on count
                count = index_counts[global_idx]
                for _ in range(count):
                    out_data.append(shard_data[line_idx])
                total_written += count

        # Write output
        write_jsonl(out_data, str(out_file))
        global_offset += shard_size

    print(f">> Total samples written: {total_written}")
    print(f">> Expected: {selection_size}")
