import pathlib
import random
import os

from tqdm import tqdm
import numpy as np
import datasets
import yaml

alpha = 0.0001
batch = 50


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


def mab(
    num_clusters,
    selection_ratio,
    cluster_average_reward,
):
    sum_chose = 0
    cluster_chose_ratio = np.zeros(num_clusters)
    cluster_chose_time = np.zeros(num_clusters)
    cluster_ucb = cluster_average_reward.copy()
    iterations = int(num_clusters * selection_ratio / (0.05 * batch))
    print(">> Iterations:", iterations)
    for _ in range(iterations):
        # 0.02 * 0.05 * 200
        # 1000 -> 0.2 selection ratio
        current_chose_num = 0
        current_chose = []
        for k in np.argsort(-cluster_ucb):
            if cluster_chose_ratio[k] < 1:
                cluster_chose_ratio[k] += 0.05
                cluster_chose_time[k] += 1
                current_chose_num += 1
                current_chose.append(k)
                if current_chose_num == batch:
                    break
        sum_chose += batch
        for k in range(num_clusters):
            ucb = alpha * np.sqrt(
                2 * (np.log(float(sum_chose))) / float(cluster_chose_time[k] + 1)
            )
            cluster_ucb[k] = cluster_average_reward[k] + ucb
    return cluster_chose_ratio


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
    confg_file = "clustering/configs/openclip/dclm_dim.yaml"
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
    # cluster_chose_ratio = mab(
    #     num_clusters,
    #     selection_ratio,
    #     np.load(f"{out_dir}/cluster_info/average_reward.npy"),
    # )
    cluster_chose_ratio = np.ones(num_clusters) * selection_ratio
    selected_indices = select(out_dir, num_clusters, params, cluster_chose_ratio)
    print(">> Selected indices shape:", len(selected_indices))
    np.save(
        f"{out_dir}/selected_indices_{selection_ratio}.npy",
        selected_indices,
    )
