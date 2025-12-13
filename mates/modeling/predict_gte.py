from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
import datasets
import torch
import glob
import os

# Tokenize the input texts
i = 0
val_data = []
mmlu_path = "/project/flame/zichunyu/data/mmlu"
for val_path in sorted(glob.glob(os.path.join(mmlu_path, "*/val.pt"))):
    if i == 21 or i == 30:
        i += 1
        continue
    val_data += torch.load(val_path)[:5]
    i += 1
hellaswag_path = "/project/flame/zichunyu/data/hellaswag/train-1024.pt"
val_data += torch.load(hellaswag_path)[:32]
val_data = [d["input_ids"] for d in val_data]

ocache = f"baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5/oracle/"
odata = datasets.concatenate_datasets([datasets.load_from_disk(ocache + str(i)) for i in range(4)])
np.random.seed(42)
indices = np.random.choice(len(odata), size=1000, replace=False)
odata = odata.select(indices)
val_data = [d["input_ids"] for d in odata]

pythia_tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1")
input_texts = pythia_tokenizer.batch_decode(
    val_data,
    skip_special_tokens=True,
)
model_path = "Alibaba-NLP/gte-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()

with torch.no_grad():
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move batch to cuda
        batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
        
        outputs = model(**batch_dict)
        batch_embeddings = outputs.last_hidden_state[:, 0]
        all_embeddings.append(batch_embeddings)
    embeddings = torch.cat(all_embeddings, dim=0)
    print(embeddings.shape)

# Save embeddings to a npy
embeddings_path = "/project/flame/zichunyu/data/shard_0-9.npy"
np.save(embeddings_path, embeddings.detach().float().cpu().numpy())

# # (Optionally) normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print(scores.tolist())
