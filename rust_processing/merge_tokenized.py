import shutil
import json
import os

# merge /tmp/fasttext_0.1/tokenized_7.2B and /tmp/fasttext_14.4B/tokenized
# 1. Create a new directory for the merged dataset
# 2. Copy the files from both datasets into the new directory (note that some files may have the same name, so you may need to rename them, e.g., /tmp/fasttext_0.1/tokenized_7.2B/shard_00000000.tar -> /tmp/fasttext_merged/shard_00000000_1.tar)
# 3. Create a new manifest file that includes entries from both datasets (each line in the manifest file is just like {"num_sequences":8192,"shard":"shard_00000914"})

# Create merged directory
merged_dir = "/tmp/data/fasttext-7.2B_prox-21.6B_merged/tokenized"
os.makedirs(merged_dir, exist_ok=True)


# Copy files from first dataset
src1 = "/tmp/data/fasttext_0.1/tokenized_7.2B"
for file in os.listdir(src1):
    if file.endswith(".tar"):
        shutil.copy2(f"{src1}/{file}", f"{merged_dir}/{file.replace('.tar', '_1.tar')}")

# Copy files from second dataset
src2 = "/tmp/data/prox/fasttext_21.6B/tokenized"
for file in os.listdir(src2):
    if file.endswith(".tar"):
        shutil.copy2(f"{src2}/{file}", f"{merged_dir}/{file.replace('.tar', '_2.tar')}")

manifest = []
# Read and process first manifest
manifest1_path = "/tmp/data/fasttext_0.1/tokenized_7.2B/manifest.jsonl"
with open(manifest1_path, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        entry["shard"] = entry["shard"] + "_1"
        manifest.append(entry)

# Read and process second manifest
manifest2_path = ("/tmp/data/prox/fasttext_21.6B/tokenized/manifest.jsonl")
with open(manifest2_path, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        # if int(entry["shard"].split("_")[1]) >= 418:
        #     continue
        entry["shard"] = entry["shard"] + "_2"
        manifest.append(entry)

with open(f"{merged_dir}/manifest.jsonl", "w") as f:
    for entry in manifest:
        f.write(json.dumps(entry) + "\n")

# from glob import glob
# from tqdm import tqdm
# import pandas as pd
# import os

# # hf download gair-prox/DCLM-pro --repo-type dataset --include "data/global-shard_01_of_10/018*" --local-dir /tmp/data/DCLM-pro
# parquet_files = glob(os.path.join("/tmp/data/DCLM-pro/", "**", "018*.parquet"), recursive=True)

# for p_file in tqdm(parquet_files, desc="Converting .parquet to .jsonl"):
#     try:
#         jsonl_dir = os.path.join(os.path.dirname(p_file), "jsonl")
#         os.makedirs(jsonl_dir, exist_ok=True)
#         base_filename = os.path.splitext(os.path.basename(p_file))[0]
#         jsonl_file = os.path.join(jsonl_dir, f"{base_filename}.jsonl")
#         pd.read_parquet(p_file).to_json(jsonl_file, orient="records", lines=True)
#     except Exception as e:
#         print(f"Error processing {p_file}: {e}")
