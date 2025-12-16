import argparse
import json
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import numpy as np


def plot_distributions(metrics, index_counts, output_dir):
    """
    Plot KDE distributions for metrics and index_counts.

    Args:
        metrics: numpy array of prediction scores
        index_counts: numpy array of selection counts per index
        output_dir: directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Metrics distribution
    ax1 = axes[0]
    sns.kdeplot(data=metrics, ax=ax1, fill=True, color="blue", alpha=0.6)
    ax1.set_xlabel("Prediction Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(
        "KDE Distribution of Prediction Scores",
        fontsize=14,
        fontweight="bold",
    )
    ax1.axvline(
        metrics.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {metrics.mean():.4f}",
    )
    ax1.axvline(
        np.median(metrics),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(metrics):.4f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add stats text
    stats_text = (
        f"Min: {metrics.min():.4f}\nMax: {metrics.max():.4f}\nStd: {metrics.std():.4f}"
    )
    ax1.text(
        0.05,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 2: Index counts distribution (only for indices that were selected)
    ax2 = axes[1]
    selected_counts = index_counts[index_counts > 0]
    total_indices = len(index_counts)

    if len(selected_counts) > 0:
        sns.kdeplot(data=selected_counts, ax=ax2, fill=True, color="green", alpha=0.6)
        ax2.set_xlabel("Selection Count", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title(
            "KDE Distribution of Selection Counts (Selected Indices Only)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.axvline(
            selected_counts.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {selected_counts.mean():.4f}",
        )
        ax2.axvline(
            np.median(selected_counts),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(selected_counts):.4f}",
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add stats text
        selected_indices = len(selected_counts)
        selection_rate = selected_indices / total_indices * 100

        stats_text = (
            f"Total indices: {total_indices:,}\n"
            f"Selected: {selected_indices:,}\n"
            f"Selection rate: {selection_rate:.2f}%\n"
            f"Min count: {selected_counts.min()}\n"
            f"Max count: {selected_counts.max()}\n"
            f"Std: {selected_counts.std():.4f}"
        )
        ax2.text(
            0.05,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No indices selected",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )

    plt.tight_layout()

    # Save figure
    plot_file = output_dir / "selection_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f">> Saved plot to {plot_file}")


def plot_selection_count_distribution(index_counts, output_dir):
    """Plot percentage of indices for selection counts 0-13."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    max_bucket = 8
    total_indices = len(index_counts)
    percentages = [
        (index_counts == count).sum() / total_indices * 100
        for count in range(max_bucket + 1)
    ]

    bars = ax.bar(
        range(max_bucket + 1),
        percentages,
        tick_label=list(range(max_bucket + 1)),
        color="teal",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel("Repetition Count", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(
        "Repetition Count Distribution (21.6B Checkpoint)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(max_bucket + 1))
    ax.set_ylim(0, max(percentages + [0]) * 1.15 if percentages else 1)
    ax.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(percentages) * 0.01 if percentages else 0.5),
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    count_hist_file = output_dir / "repetition_count_distribution.png"
    plt.savefig(count_hist_file, dpi=300, bbox_inches="tight")
    print(f">> Saved repetition count distribution to {count_hist_file}")


def compute_correlation(metrics, index_counts):
    """
    Compute correlation between metrics and index counts.

    Args:
        metrics: numpy array of prediction scores (negated and temperature-scaled)
        index_counts: numpy array of selection counts per index

    Returns:
        dict with correlation statistics
    """
    # Only use indices that were selected at least once
    selected_mask = index_counts > 0

    if selected_mask.sum() == 0:
        print(">> Warning: No indices were selected!")
        return {}

    metrics_selected = metrics[selected_mask]
    counts_selected = index_counts[selected_mask]

    # Compute correlations
    pearson_corr, pearson_pval = pearsonr(metrics_selected, counts_selected)
    spearman_corr, spearman_pval = spearmanr(metrics_selected, counts_selected)

    correlation_stats = {
        "pearson_correlation": float(pearson_corr),
        "pearson_pvalue": float(pearson_pval),
        "spearman_correlation": float(spearman_corr),
        "spearman_pvalue": float(spearman_pval),
        "num_selected_indices": int(selected_mask.sum()),
        "total_indices": int(len(metrics)),
    }

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"Pearson correlation: {pearson_corr:.6f} (p-value: {pearson_pval:.6e})")
    print(f"Spearman correlation: {spearman_corr:.6f} (p-value: {spearman_pval:.6e})")
    print(f"Selected indices: {selected_mask.sum()} / {len(metrics)}")
    print("=" * 60 + "\n")

    return correlation_stats


def get_top_repeated_samples(index_counts, textfiles_dir, top_k=100):
    """
    Get top-k most repeated samples with text preview.

    Args:
        index_counts: numpy array of selection counts per index
        textfiles_dir: directory containing text files
        top_k: number of top samples to retrieve

    Returns:
        list of dicts with index, count, and text preview
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../tokenization"))
    from file_utils import read_jsonl

    print(f"\n>> Finding top {top_k} most repeated samples...")

    # Get non-zero counts and their indices
    nonzero_indices = np.nonzero(index_counts)[0]
    nonzero_counts = index_counts[nonzero_indices]

    # Sort by count descending
    sorted_idx = np.argsort(-nonzero_counts)
    top_indices = nonzero_indices[sorted_idx[:top_k]]
    top_counts = nonzero_counts[sorted_idx[:top_k]]

    # Get shard files
    shard_files = sorted(
        [
            f
            for f in os.listdir(textfiles_dir)
            if f.endswith(".jsonl.zstd") and not f.startswith(".")
        ]
    )

    # Build cumulative index for each shard
    cumsum = 0
    shard_offsets = [0]
    for shard_file in shard_files:
        count = sum(1 for _ in read_jsonl(os.path.join(textfiles_dir, shard_file)))
        cumsum += count
        shard_offsets.append(cumsum)

    # Load top samples
    top_samples = []
    for global_idx, count in tqdm(
        zip(top_indices, top_counts), total=len(top_indices), desc="Loading samples"
    ):
        global_idx = int(global_idx)

        # Find which shard this index belongs to
        shard_idx = np.searchsorted(shard_offsets[1:], global_idx, side="right")
        local_idx = global_idx - shard_offsets[shard_idx]
        shard_file = os.path.join(textfiles_dir, shard_files[shard_idx])

        # Load specific line
        for line_num, line in enumerate(read_jsonl(shard_file)):
            if line_num == local_idx:
                text = line.get("text", "")
                top_samples.append(
                    {
                        "index": global_idx,
                        "count": int(count),
                        "text_preview": text[:200],
                        "full_text": text,
                    }
                )
                break

    return top_samples


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Analyze selection with replacement results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing metrics.npy and index_counts.npy",
    )
    parser.add_argument(
        "--textfiles_dir",
        type=str,
        default=None,
        help="Directory containing original text files (for top repeated samples)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save plots (defaults to data_dir)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top repeated samples to retrieve",
    )
    parser.add_argument(
        "--skip_top_samples",
        action="store_true",
        help="Skip loading top repeated samples (faster)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.data_dir

    # Load data
    data_dir = Path(args.data_dir)
    metrics_file = data_dir / "metrics.npy"
    index_counts_file = data_dir / "index_counts.npy"

    print(f">> Loading metrics from {metrics_file}")
    metrics = np.load(metrics_file)

    print(f">> Loading index counts from {index_counts_file}")
    index_counts = np.load(index_counts_file)

    print(f">> Metrics shape: {metrics.shape}")
    print(f">> Index counts shape: {index_counts.shape}")

    # Plot distributions
    # plot_distributions(metrics, index_counts, args.output_dir)

    # Sample with replacement using categorical distribution
    rng = np.random.default_rng(seed=1234)

    # Convert to probabilities using softmax
    # Subtract max for numerical stability
    metrics /= 2.0
    metrics_shifted = metrics - np.max(metrics)
    exp_metrics = np.exp(metrics_shifted)
    probabilities = exp_metrics / np.sum(exp_metrics)

    # Use numpy's choice which is much faster
    sampled_indices = rng.choice(
        len(metrics),
        size=len(metrics),
        replace=True,
        p=probabilities
    )
     # Count occurrences of each index
    from collections import Counter

    index_counts = Counter(sampled_indices)
    index_counts_array = np.zeros(len(metrics), dtype=np.int32)
    for idx, count in index_counts.items():
        index_counts_array[idx] = count

    plot_selection_count_distribution(index_counts_array, args.output_dir)
    exit(0)

    # Compute correlation
    correlation_stats = compute_correlation(metrics, index_counts)

    # Save correlation stats
    output_dir = Path(args.output_dir)
    correlation_file = output_dir / "correlation_stats.json"
    with open(correlation_file, "w") as f:
        json.dump(correlation_stats, f, indent=2)
    print(f">> Saved correlation stats to {correlation_file}")

    # Get top repeated samples if textfiles_dir provided
    if args.textfiles_dir:
        top_repeated = get_top_repeated_samples(
            index_counts, args.textfiles_dir, top_k=args.top_k
        )

        # Save top repeated samples
        top_repeated_file = output_dir / "top_repeated_samples.json"
        with open(top_repeated_file, "w") as f:
            json.dump(top_repeated, f, indent=2)
        print(
            f">> Saved top {len(top_repeated)} repeated samples to {top_repeated_file}"
        )

        # Print summary of top 10
        print("\n" + "=" * 60)
        print("TOP 10 MOST REPEATED SAMPLES")
        print("=" * 60)
        for i, sample in enumerate(top_repeated[:10], 1):
            preview = sample["text_preview"].replace("\n", " ")
            print(f"{i}. Index: {sample['index']}, Count: {sample['count']}")
            print(f"   {preview}...")
        print("=" * 60 + "\n")

    print("\nDone!")
