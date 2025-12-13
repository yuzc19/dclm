from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
import datasets
import faiss
import torch
import glob
import os
import re

log_line = """2025-08-03,02:47:42 | INFO | => epoch 0, training on ['gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/tokenized-7.6B/{shard_00000152,shard_00000378,shard_00000223,shard_00000424,shard_00000010,shard_00000039,shard_00000225,shard_00000287,shard_00000212,shard_00000395,shard_00000428,shard_00000248,shard_00000200,shard_00000450,shard_00000415,shard_00000027,shard_00000032,shard_00000410,shard_00000121,shard_00000308,shard_00000402,shard_00000364,shard_00000057,shard_00000186,shard_00000090,shard_00000229,shard_00000433,shard_00000227,shard_00000218,shard_00000315,shard_00000386,shard_00000112,shard_00000295,shard_00000280,shard_00000234,shard_00000363,shard_00000028,shard_00000348,shard_00000154,shard_00000092,shard_00000087,shard_00000030,shard_00000122,shard_00000207,shard_00000031,shard_00000296,shard_00000155,shard_00000236,shard_00000047,shard_00000058,shard_00000210,shard_00000238,shard_00000083,shard_00000191,shard_00000341,shard_00000316,shard_00000099,shard_00000365,shard_00000163,shard_00000445,shard_00000451,shard_00000193,shard_00000095,shard_00000370,shard_00000284,shard_00000289,shard_00000292,shard_00000417,shard_00000270,shard_00000159,shard_00000164,shard_00000304,shard_00000306,shard_00000260,shard_00000041,shard_00000407,shard_00000116,shard_00000271,shard_00000392,shard_00000452,shard_00000256,shard_00000132,shard_00000294,shard_00000358,shard_00000419,shard_00000161,shard_00000001,shard_00000237,shard_00000268,shard_00000082,shard_00000118,shard_00000371,shard_00000209,shard_00000124,shard_00000144,shard_00000369,shard_00000025,shard_00000314,shard_00000434,shard_00000346,shard_00000276,shard_00000091,shard_00000149,shard_00000438,shard_00000182,shard_00000103,shard_00000394,shard_00000397,shard_00000141,shard_00000166,shard_00000312,shard_00000340,shard_00000038,shard_00000441,shard_00000239,shard_00000097,shard_00000129,shard_00000431,shard_00000373,shard_00000147,shard_00000351,shard_00000023,shard_00000125,shard_00000436,shard_00000387,shard_00000399,shard_00000040,shard_00000140,shard_00000148,shard_00000444,shard_00000178,shard_00000403,shard_00000261,shard_00000043,shard_00000245,shard_00000325,shard_00000137,shard_00000003,shard_00000319,shard_00000142,shard_00000157,shard_00000326,shard_00000007,shard_00000257,shard_00000336,shard_00000197,shard_00000393,shard_00000002,shard_00000264,shard_00000246,shard_00000301,shard_00000108,shard_00000344,shard_00000323,shard_00000281,shard_00000446,shard_00000079,shard_00000072,shard_00000222,shard_00000060}.tar', 'gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/Qwen3-30B-A3B-FP8/tokenized/{shard_00000222,shard_00000411,shard_00000278,shard_00000102,shard_00000027,shard_00000346,shard_00000097,shard_00000166,shard_00000026,shard_00000349,shard_00000248,shard_00000378,shard_00000085,shard_00000254,shard_00000141,shard_00000414,shard_00000070,shard_00000029,shard_00000225,shard_00000095,shard_00000207,shard_00000409,shard_00000367,shard_00000246,shard_00000276,shard_00000328,shard_00000263,shard_00000028,shard_00000036,shard_00000146,shard_00000135,shard_00000292,shard_00000229,shard_00000320,shard_00000373,shard_00000251,shard_00000385,shard_00000016,shard_00000335,shard_00000365,shard_00000108,shard_00000415,shard_00000273,shard_00000069,shard_00000397,shard_00000115,shard_00000081,shard_00000322,shard_00000191,shard_00000330,shard_00000247,shard_00000039,shard_00000393,shard_00000186,shard_00000168,shard_00000345,shard_00000392,shard_00000417,shard_00000289,shard_00000249,shard_00000329,shard_00000368,shard_00000317,shard_00000306,shard_00000261,shard_00000282,shard_00000038,shard_00000362,shard_00000228,shard_00000324,shard_00000224,shard_00000418,shard_00000410,shard_00000149,shard_00000239,shard_00000220,shard_00000132,shard_00000080,shard_00000104,shard_00000048,shard_00000204,shard_00000400,shard_00000022,shard_00000312,shard_00000091,shard_00000419,shard_00000244,shard_00000154,shard_00000234,shard_00000137,shard_00000298,shard_00000192,shard_00000041,shard_00000197,shard_00000176,shard_00000184,shard_00000155,shard_00000272,shard_00000269,shard_00000147,shard_00000370,shard_00000401,shard_00000043,shard_00000112,shard_00000084,shard_00000143,shard_00000037,shard_00000403,shard_00000212,shard_00000297,shard_00000208,shard_00000180,shard_00000003,shard_00000387,shard_00000134,shard_00000233,shard_00000377,shard_00000136,shard_00000001,shard_00000381,shard_00000009,shard_00000109,shard_00000169,shard_00000045,shard_00000210,shard_00000266,shard_00000151,shard_00000402,shard_00000002,shard_00000216,shard_00000161,shard_00000114,shard_00000160,shard_00000382,shard_00000256,shard_00000074,shard_00000075,shard_00000179,shard_00000252,shard_00000214,shard_00000218,shard_00000124,shard_00000352,shard_00000071,shard_00000277,shard_00000170,shard_00000395,shard_00000372,shard_00000209,shard_00000098,shard_00000031,shard_00000331,shard_00000379,shard_00000295,shard_00000107,shard_00000051,shard_00000175,shard_00000065,shard_00000060,shard_00000231}.tar']"""
log_line = "2025-09-06,01:09:21 | INFO | => epoch 0, training on ['gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/fasttext_0.1/tokenized_7.2B/{shard_00000149,shard_00000299,shard_00000336,shard_00000213,shard_00000014,shard_00000040,shard_00000230,shard_00000181,shard_00000384,shard_00000438,shard_00000428,shard_00000248,shard_00000329,shard_00000442,shard_00000405,shard_00000020,shard_00000036,shard_00000415,shard_00000118,shard_00000195,shard_00000337,shard_00000256,shard_00000065,shard_00000178,shard_00000096,shard_00000241,shard_00000215,shard_00000399,shard_00000330,shard_00000287,shard_00000261,shard_00000107,shard_00000214,shard_00000150,shard_00000385,shard_00000245,shard_00000028,shard_00000340,shard_00000144,shard_00000095,shard_00000083,shard_00000025,shard_00000124,shard_00000436,shard_00000034,shard_00000321,shard_00000146,shard_00000331,shard_00000044,shard_00000073,shard_00000379,shard_00000311,shard_00000079,shard_00000184,shard_00000161,shard_00000291,shard_00000089,shard_00000363,shard_00000425,shard_00000437,shard_00000164,shard_00000410,shard_00000098,shard_00000349,shard_00000271,shard_00000283,shard_00000396,shard_00000171,shard_00000190,shard_00000286,shard_00000254,shard_00000136,shard_00000294,shard_00000416,shard_00000045,shard_00000400,shard_00000108,shard_00000175,shard_00000380,shard_00000444,shard_00000325,shard_00000140,shard_00000445,shard_00000242,shard_00000426,shard_00000366,shard_00000001,shard_00000220,shard_00000279,shard_00000090,shard_00000109,shard_00000258,shard_00000304,shard_00000117,shard_00000142,shard_00000234,shard_00000029,shard_00000307,shard_00000427,shard_00000339,shard_00000169,shard_00000091,shard_00000151,shard_00000430,shard_00000262,shard_00000104,shard_00000154,shard_00000392,shard_00000141,shard_00000433,shard_00000303,shard_00000429,shard_00000038,shard_00000191,shard_00000356,shard_00000102,shard_00000120,shard_00000424,shard_00000432,shard_00000134,shard_00000326,shard_00000023,shard_00000113,shard_00000165,shard_00000226,shard_00000257,shard_00000043,shard_00000128,shard_00000156,shard_00000199,shard_00000324,shard_00000301,shard_00000246,shard_00000039,shard_00000401,shard_00000332,shard_00000282,shard_00000004,shard_00000320,shard_00000446,shard_00000318,shard_00000009,shard_00000358,shard_00000249,shard_00000348,shard_00000383,shard_00000002,shard_00000403,shard_00000250,shard_00000411,shard_00000110,shard_00000346,shard_00000255,shard_00000265,shard_00000376,shard_00000078,shard_00000071,shard_00000208,shard_00000056,shard_00000233,shard_00000218,shard_00000130,shard_00000378,shard_00000159,shard_00000225,shard_00000074,shard_00000421,shard_00000240,shard_00000222,shard_00000270,shard_00000362,shard_00000100,shard_00000033,shard_00000217,shard_00000377,shard_00000224,shard_00000123,shard_00000059,shard_00000206,shard_00000067,shard_00000050,shard_00000264,shard_00000355,shard_00000196,shard_00000185,shard_00000281,shard_00000182,shard_00000345,shard_00000013,shard_00000138,shard_00000351,shard_00000381,shard_00000088,shard_00000269,shard_00000021,shard_00000260,shard_00000408,shard_00000251,shard_00000019,shard_00000111,shard_00000235,shard_00000052,shard_00000273,shard_00000129,shard_00000179,shard_00000266,shard_00000418,shard_00000247,shard_00000042,shard_00000162,shard_00000285,shard_00000441,shard_00000391,shard_00000302,shard_00000055,shard_00000203,shard_00000101,shard_00000443,shard_00000005,shard_00000207,shard_00000066,shard_00000387,shard_00000041,shard_00000243}.tar', 'gs://cmu-gpucloud-zichunyu/data/refinedweb_01_0/Qwen3-4B-grpo-1020/fasttext_14.4B/tokenized/{shard_00000598,shard_00000846,shard_00000080,shard_00000903,shard_00000779,shard_00000386,shard_00000857,shard_00000569,shard_00000224,shard_00000549,shard_00000448,shard_00000333,shard_00000915,shard_00000658,shard_00000387,shard_00000471,shard_00000369,shard_00000870,shard_00000002,shard_00000502,shard_00000610,shard_00000004,shard_00000128,shard_00000367,shard_00000137,shard_00000727,shard_00000257,shard_00000827,shard_00000295,shard_00000850,shard_00000629,shard_00000362,shard_00000027,shard_00000021,shard_00000542,shard_00000573,shard_00000200,shard_00000159,shard_00000845,shard_00000322,shard_00000056,shard_00000378,shard_00000587,shard_00000014,shard_00000083,shard_00000505,shard_00000120,shard_00000148,shard_00000326,shard_00000916,shard_00000679,shard_00000230,shard_00000798,shard_00000409,shard_00000559,shard_00000646,shard_00000719,shard_00000123,shard_00000452,shard_00000349,shard_00000239,shard_00000007,shard_00000515,shard_00000420,shard_00000824,shard_00000623,shard_00000886,shard_00000126,shard_00000397,shard_00000763,shard_00000560,shard_00000546,shard_00000339,shard_00000054,shard_00000325,shard_00000625,shard_00000018,shard_00000298,shard_00000447,shard_00000262,shard_00000253,shard_00000557,shard_00000304,shard_00000389,shard_00000307,shard_00000221,shard_00000187,shard_00000688,shard_00000130,shard_00000139,shard_00000188,shard_00000475,shard_00000745,shard_00000186,shard_00000237,shard_00000359,shard_00000789,shard_00000219,shard_00000236,shard_00000550,shard_00000867,shard_00000184,shard_00000705,shard_00000193,shard_00000311,shard_00000506,shard_00000792,shard_00000097,shard_00000768,shard_00000038,shard_00000541,shard_00000747,shard_00000143,shard_00000616,shard_00000854,shard_00000520,shard_00000163,shard_00000393,shard_00000878,shard_00000044,shard_00000075,shard_00000133,shard_00000914,shard_00000794,shard_00000190,shard_00000874,shard_00000169,shard_00000299,shard_00000464,shard_00000121,shard_00000728,shard_00000127,shard_00000222,shard_00000574,shard_00000467,shard_00000114,shard_00000132,shard_00000059,shard_00000320,shard_00000279,shard_00000263,shard_00000543,shard_00000811,shard_00000370,shard_00000016,shard_00000073,shard_00000404,shard_00000037,shard_00000019,shard_00000281,shard_00000465,shard_00000492,shard_00000891,shard_00000005,shard_00000599,shard_00000873,shard_00000734,shard_00000152,shard_00000001,shard_00000048,shard_00000425,shard_00000168,shard_00000421,shard_00000249,shard_00000818,shard_00000628,shard_00000023,shard_00000275,shard_00000533,shard_00000880,shard_00000309,shard_00000530,shard_00000675,shard_00000572,shard_00000442,shard_00000676,shard_00000436,shard_00000146,shard_00000046,shard_00000065,shard_00000067,shard_00000486,shard_00000531,shard_00000247,shard_00000164,shard_00000450,shard_00000371,shard_00000481,shard_00000030,shard_00000710,shard_00000343,shard_00000639,shard_00000244,shard_00000411,shard_00000071,shard_00000047,shard_00000401,shard_00000433,shard_00000856,shard_00000892,shard_00000051,shard_00000858,shard_00000106,shard_00000428,shard_00000699,shard_00000040,shard_00000723,shard_00000721,shard_00000851,shard_00000317,shard_00000593,shard_00000098,shard_00000750,shard_00000382,shard_00000602,shard_00000289,shard_00000155,shard_00000917,shard_00000536,shard_00000402,shard_00000341,shard_00000246,shard_00000588,shard_00000347}.tar']"

# Extract all shard lists inside {...}
matches = re.findall(r"\{([^}]+)\}", log_line)

counts = [len(m.split(",")) for m in matches]
counts, sum(counts)

print(counts, sum(counts))

exit(0)

if False:
    # For MMLU + HellaSwag
    ocache = f"baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5/oracle/"
    odata = datasets.concatenate_datasets([datasets.load_from_disk(ocache + str(i)) for i in range(4)])
    print(len(odata))

    # i = 0
    # mmlu_path = "/project/flame/zichunyu/data/mmlu"
    # for val_path in sorted(glob.glob(os.path.join(mmlu_path, "*/val.pt"))):
    #     val_data = torch.load(val_path)[:5]
    #     if i == 26:
    #         print(len(val_data[0]["input_ids"]))
    #         print(len(val_data[0]["labels"]))
    #         exit(0)
    #     i += 1

    i = 0
    task_names = []
    mmlu_path = "/project/flame/zichunyu/data/mmlu"
    for val_path in sorted(glob.glob(os.path.join(mmlu_path, "*/val.pt"))):
        if i == 21 or i == 30:
            i += 1
            continue
        task_names.append(os.path.basename(os.path.dirname(val_path)))
        i += 1
    task_names.append("hellaswag")
    print(len(task_names))


    def process_scores(scores):
        first_part = scores[:, :275].reshape(scores.shape[0], 55, 5)
        avg_first = first_part.mean(axis=2)  # shape: (N, 55)

        last_part = scores[:, 275:]
        avg_last = last_part.mean(axis=1, keepdims=True)  # shape: (N, 1)

        return np.concatenate([avg_first, avg_last], axis=1)  # shape: (N, 56)


    scores = np.array(odata["scores"])
    scores = np.delete(scores, np.s_[105:110], axis=1)
    scores = np.delete(scores, np.s_[145:150], axis=1)
    print(scores.shape)  # should be (N, 307)

    processed_scores = process_scores(scores)

# Is there an emergence of some kind of the skills?
ocache = f"baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5/oracle-arce+hellaswag/"
odata = datasets.concatenate_datasets([datasets.load_from_disk(ocache + str(i)) for i in range(16)])
task_names = ["arce", "hellaswag"]
processed_scores = np.array(odata["scores"])
print(processed_scores.shape)  # should be (N, 64)


def select_indices(data):
    data = (data - data.mean()) / data.std()
    data = data / 0.5
    rng = np.random.default_rng(seed=42)
    gumbel_noise = rng.gumbel(size=len(data))
    data += gumbel_noise
    return np.argsort(data)[:102400]

    # ranked = np.apply_along_axis(
    #     lambda x: rankdata(x, method="average"),
    #     axis=0,
    #     arr=data,
    # )
    # min_rank = ranked.min(axis=1)
    # return np.argsort(min_rank)[:102400]

    indices_list = []
    for i in range(data.shape[1]):
        indices_list += np.argsort(data[:, i])[:6500].tolist()
    indices_list = list(set(indices_list))
    print(len(indices_list))
    return np.array(indices_list)


# indices = select_indices(processed_scores.mean(axis=1))
# indices = select_indices(processed_scores)
# np.random.seed(42)
# np.random.shuffle(indices)
# np.save("mean_indices_arce+hellaswag_102400_0.5.npy", indices)
# exit(0)


def rank_analysis(data_58d):
    ranked = np.apply_along_axis(
        lambda x: rankdata(x, method="average"),
        axis=0,
        arr=data_58d,
    )
    avg_rank = ranked.mean(axis=1)
    # calculate the number of avg_rank less than 4800
    print(f"Number of 10% avg_rank: {np.sum(avg_rank <= 4800)}")
    min_rank = ranked.min(axis=1)
    print(f"Number of 10% min_rank: {np.sum(min_rank <= 4800)}")
    max_rank = ranked.max(axis=1)
    print(f"Number of 10% max_rank: {np.sum(max_rank <= 4800)}")
    plt.figure(figsize=(12, 6))
    sns.kdeplot(avg_rank, label="Average Rank", linewidth=2)
    sns.kdeplot(min_rank, label="Min Rank", linewidth=2)
    sns.kdeplot(max_rank, label="Max Rank", linewidth=2)
    plt.xlabel("Rank Value")
    plt.ylabel("Density")
    plt.title("Distribution of Average, Min, and Max Ranks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rank_analysis.png")


# rank_analysis(processed_scores)
# exit(0)


def tsne_visualization(data_58d, labels=None, perplexity=30, random_state=42):
    """
    Reduce (N, 58) to (N, 2) using t-SNE and visualize it.

    Parameters:
        data_58d (ndarray): Input data of shape (N, 58)
        labels (ndarray or list): Optional labels for coloring the points
        perplexity (int): t-SNE perplexity parameter
        random_state (int): Random seed for reproducibility
    """
    # export OPENBLAS_NUM_THREADS=4
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    data_2d = tsne.fit_transform(data_58d)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        n_classes = 20
        cmap = plt.get_cmap("tab20")
        norm = mcolors.BoundaryNorm(
            boundaries=np.arange(n_classes + 1) - 0.5, ncolors=n_classes
        )
        _ = plt.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c=labels,
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.8,
        )
        handles = [
            mpatches.Patch(color=cmap(i), label=str(i)) for i in range(n_classes)
        ]
        plt.legend(
            handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
    else:
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=20, alpha=0.8)

    plt.title("t-SNE Visualization of Training Data (Feature=Influence Vector)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("tsne_visualization.png")


# check if nan in processed_scores
if np.isnan(processed_scores).any():
    # check where the nans are
    nan_indices = np.argwhere(np.isnan(processed_scores))
    print("NaN values found at indices:", nan_indices)


# processed_scores = np.load("/project/flame/zichunyu/data/shard_0-9.npy")
# d = processed_scores.shape[1]
# kmeans = faiss.Kmeans(d, k=20, niter=100, verbose=True)
# np.random.seed(42)
# indices = np.random.choice(processed_scores.shape[0], size=1000, replace=False)
# kmeans.train(processed_scores[indices])
# _, cluster_labels = kmeans.index.search(processed_scores[indices], 1)
# cluster_labels = cluster_labels.flatten()
# print(cluster_labels)
# print(f"Cluster distribution: {np.bincount(cluster_labels)}")
# tsne_visualization(processed_scores[indices], cluster_labels)
# exit(0)


def plot_spearman_heatmap(data_58d):
    """
    Plot a 58×58 heatmap of Spearman correlations between features.

    Parameters:
        data_58d (ndarray): Input array of shape (N, 58)
    """
    # Compute Spearman correlation matrix
    corr, _ = stats.spearmanr(data_58d)

    # corr = cosine_similarity(data_58d.T)
    # Manually compute cosine similarity between the first two features
    a = data_58d[:, 0]
    b = data_58d[:, 1]
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(cosine_sim)

    # [-1, 0)[0, 0.2)[0.2, 0.4)[0.4, 0.6)[0.6, 1)
    # Calculate fraction of correlations in each range
    total_pairs = corr.size - data_58d.shape[1]  # Exclude diagonal
    print(total_pairs)
    neg_one_to_zero = np.sum((corr >= -1) & (corr < 0)) / total_pairs
    zero_to_0p2 = np.sum((corr >= 0) & (corr < 0.2)) / total_pairs
    p2_to_0p4 = np.sum((corr >= 0.2) & (corr < 0.4)) / total_pairs
    p4_to_0p6 = np.sum((corr >= 0.4) & (corr < 0.6)) / total_pairs
    p6_to_1 = np.sum((corr >= 0.6) & (corr < 1)) / total_pairs

    print(f"Fraction in [-1, 0): {neg_one_to_zero:.4f}")
    print(f"Fraction in [0, 0.2): {zero_to_0p2:.4f}")
    print(f"Fraction in [0.2, 0.4): {p2_to_0p4:.4f}")
    print(f"Fraction in [0.4, 0.6): {p4_to_0p6:.4f}")
    print(f"Fraction in [0.6, 1]: {p6_to_1:.4f}")

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Spearman Correlation between Examples from ARC-E and HellaSwag (Feature=Influence Vector)"
    )
    plt.tight_layout()
    plt.savefig("heatmap_visualization.png")


plot_spearman_heatmap(processed_scores)
exit(0)


def plot_cosine_heatmap(sem_embeddings):
    # sem_embeddings is the size of (N, 768)
    # we want to calculate the mean of cosine similarity between the every 5 elements to next 5 elements, for example, the first 5 elements are the first task, the next 5 elements are the second task, and so on.
    # note, we cannot avergae the embeddings, we need to calculate the cosine similarity between the embeddings of each data point, then average the cosine similarity
    sim_matrix = np.zeros((56, 56))
    for i in range(56):
        for j in range(56):
            task_i_start = i * 5
            if i == 55:
                task_i_end = sem_embeddings.shape[0]
            else:
                task_i_end = task_i_start + 5
            task_j_start = j * 5
            if j == 55:
                task_j_end = sem_embeddings.shape[0]
            else:
                task_j_end = task_j_start + 5

            task_i_embeddings = sem_embeddings[task_i_start:task_i_end]
            task_j_embeddings = sem_embeddings[task_j_start:task_j_end]

            # calculate cosine similarity between all pairs from task i and task j
            cosine_similarities = cosine_similarity(
                task_i_embeddings, task_j_embeddings
            )
            # average all pairwise similarities
            sim_matrix[i, j] = np.max(cosine_similarities)

    total_pairs = sim_matrix.size - 56
    print(total_pairs)
    neg_one_to_zero = np.sum((sim_matrix >= -1) & (sim_matrix < 0)) / total_pairs
    zero_to_0p2 = np.sum((sim_matrix >= 0) & (sim_matrix < 0.2)) / total_pairs
    p2_to_0p4 = np.sum((sim_matrix >= 0.2) & (sim_matrix < 0.4)) / total_pairs
    p4_to_0p6 = np.sum((sim_matrix >= 0.4) & (sim_matrix < 0.6)) / total_pairs
    p6_to_1 = np.sum((sim_matrix >= 0.6) & (sim_matrix < 1)) / total_pairs

    print(f"Fraction in [-1, 0): {neg_one_to_zero:.4f}")
    print(f"Fraction in [0, 0.2): {zero_to_0p2:.4f}")
    print(f"Fraction in [0.2, 0.4): {p2_to_0p4:.4f}")
    print(f"Fraction in [0.4, 0.6): {p4_to_0p6:.4f}")
    print(f"Fraction in [0.6, 1]: {p6_to_1:.4f}")

    # print task name pairs that have high correlation
    high_corr_pairs = np.argwhere(sim_matrix > 0.6)
    for i, j in high_corr_pairs:
        if i < j:  # to avoid duplicate pairs
            print(
                f"{sim_matrix[i, j]:.4f} semantic similarity between {task_names[i]} and {task_names[j]}"
            )

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        cmap="vlag",
        center=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Cosine Similarity between Evaluation Tasks (Feature=GTE Embedding)")
    plt.tight_layout()
    plt.savefig("heatmap_visualization.png")


sem_embeddings = np.load("/project/flame/zichunyu/data/mmlu_hellaswag_embeddings.npy")
print(sem_embeddings.shape)
plot_cosine_heatmap(sem_embeddings)
