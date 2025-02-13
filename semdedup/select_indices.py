import pathlib
import random
import os

from tqdm import tqdm
import numpy as np
import datasets
import yaml


def collect_reward(out_dir, num_clusters):
    dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(f"{out_dir}/{i}") for i in range(8)]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)
    np.save(
        pathlib.Path(f"{out_dir}", "prediction.npy"),
        metrics,
    )

    cluster_average_reward = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        cluster_i = np.load(
            os.path.join(
                out_dir,
                params["sorted_clusters_file_loc"],
                f"cluster_{cluster_id}.npy",
            )
        )
        indices = cluster_i[:, 0].astype("int32")
        cluster_average_reward[cluster_id] = np.mean(metrics[indices])
    np.save(
        pathlib.Path(f"{out_dir}/cluster_info", "average_reward.npy"),
        cluster_average_reward,
    )


def group_select(out_dir, num_clusters, params, cluster_chose_ratio):
    prediction = np.load(pathlib.Path(f"{out_dir}", "prediction.npy"))

    temp = 0.9902
    alpha = 0.9496
    emb_memory = np.memmap(
        params["emb_memory_loc"],
        dtype="float32",
        mode="r",
    )
    emb_memory = emb_memory.reshape(-1, params["emb_size"])
    print(">> Reps shape:", emb_memory.shape)

    selected_indices = []
    for cluster_id in tqdm(range(num_clusters)):
        cluster_i = np.load(
            os.path.join(
                params["sorted_clusters_file_loc"],
                f"cluster_{cluster_id}.npy",
            )
        )
        indices = cluster_i[:, 0].astype("int32")
        selection_size = int(cluster_chose_ratio[cluster_id] * len(indices))
        metrics = prediction[indices]
        reps = emb_memory[indices]
        rel = np.zeros(reps.shape[0])
        tmp_indices = []
        for i in range(selection_size):
            scores = metrics - alpha * rel
            scores[tmp_indices] = float("inf")
            selected_index = np.argmin(scores)
            selected_indices.append(indices[selected_index])
            tmp_indices.append(selected_index)
            cur_sim = np.matmul(reps, reps[selected_index].transpose()) / temp - 1
            rel = (i * rel + metrics[selected_index] * cur_sim) / (i + 1)
    return selected_indices


def select(out_dir, num_clusters, params, cluster_chose_ratio):
    selected_indices = []
    prediction = np.load(pathlib.Path(f"{out_dir}", "prediction.npy"))
    for cluster_id in tqdm(range(num_clusters)):
        cluster_i = np.load(
            os.path.join(
                params["sorted_clusters_file_loc"],
                f"cluster_{cluster_id}.npy",
            )
        )
        indices = cluster_i[:, 0].astype("int32")
        metrics = prediction[indices]
        rng = np.random.default_rng()
        gumbel_noise = rng.gumbel(size=len(metrics))
        metrics += gumbel_noise
        size = int(cluster_chose_ratio[cluster_id] * len(indices))
        selected_indices += indices[np.argpartition(metrics, size)[:size]].tolist()
    return selected_indices


if __name__ == "__main__":
    confg_file = "clustering/configs/openclip/dclm_dim_411m.yaml"
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    SEED = params["seed"]
    random.seed(SEED)
    num_clusters = params["ncentroids"]
    selection_ratio = params["selection_ratio"]
    out_dir = params["emb_memory_loc"].split("/emb.npy")[0]

    if os.path.exists(f"{out_dir}/cluster_info/average_reward.npy"):
        print("Reward exists, skip collecting reward")
    else:
        collect_reward(out_dir, num_clusters)
    cluster_chose_ratio = np.ones(num_clusters) * selection_ratio
    selected_indices = group_select(out_dir, num_clusters, params, cluster_chose_ratio)
    print(">> Selected indices shape:", len(selected_indices))
    np.save(
        f"{out_dir}/group_selected_indices_{selection_ratio}.npy",
        selected_indices,
    )
