from scipy.stats import pearsonr, spearmanr
import numpy as np
import datasets

base_dir = "/home/zichunyu/out/dclm_logs/baseline_01_1_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_2"
dataset = datasets.concatenate_datasets([datasets.load_from_disk(f"{base_dir}/oracle/{i}").select(range(250)) for i in [0, 1, 2, 3]])
# base_dir = "/home/zichunyu/out/dclm_logs/baseline_01_1_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000/checkpoints/epoch_2"
dataset_2 = datasets.concatenate_datasets(
    [
        datasets.load_from_disk(f"{base_dir}/oracle-oh-eli5-8192/{i}").select(range(250))
        for i in range(4)
        # for i in [0, 2, 4, 6]
    ]
)
scores = np.array(dataset["scores"])[:, 0]
print(len(scores))
scores_2 = np.array(dataset_2["scores"])[:, 0]
print(len(scores_2))
print(spearmanr(scores, scores_2))
exit(0)

from modeling_seq_data_influence_model import BiEncoderModel
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import datasets
import torch

# ds_0 = datasets.load_from_disk("/home/zichunyu/out/oracle/pythia-410m/epoch_2/3")
# ds_1 = datasets.load_from_disk("/home/zichunyu/out/oracle/pythia-410m/epoch_2/1")
# ds_2 = datasets.load_from_disk("/home/zichunyu/out/oracle/pythia-410m/epoch_2/2")
ds_3 = datasets.load_from_disk("/home/zichunyu/out/oracle/pythia-410m/epoch_2/3")
ds_4 = datasets.load_from_disk("/home/zichunyu/out/oracle/pythia-410m/epoch_2/4")

base = 3.655113697052002

cnt = 0
first_scores = []
ids2index = {}
for d in ds_3:
    # print(d["scores"])
    first_scores.append(base - d["scores"][0])
    ids2index[tuple(d["input_ids"][:2048])] = cnt
    cnt += 1
print(first_scores)
print(max(first_scores))
print(cnt)


model_dir = (
    "/home/zichunyu/out/oracle/pythia-410m/epoch_1/bs-1-sample/pairwise-dim-flan"
)
model = BiEncoderModel.from_pretrained("BAAI/bge-base-en-v1.5")
model.load_state_dict(torch.load(model_dir + "/pytorch_model.bin"), strict=False)
model.cuda()
model.eval()


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
    return model.compute_similarity(p_reps[:1], p_reps[1:]).item()


pythia_tokenizer = AutoTokenizer.from_pretrained("tokenization_configs/pythia-410m")
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    max_length=2048,
    padding="max_length",
)
out = open("out.txt", "w")
rels = []
for d in tqdm(datasets.concatenate_datasets([ds_3, ds_4])):
    first_index = ids2index[tuple(d["input_ids"][:2048])]
    second_index = ids2index[tuple(d["input_ids"][2048:])]

    first_score = first_scores[first_index]
    second_score = first_scores[second_index]

    pairwise_oracle = base - d["scores"][1]
    # pairwise_oracle = first_score + (1-rel) * second_score
    rel = -(pairwise_oracle - first_score - second_score) / second_score
    if rel < 1 and rel > -1:
        # rels.append()
        cnt += 1

        texts = pythia_tokenizer.batch_decode(
            [d["input_ids"][:2048], ["input_ids"][2048:]],
            skip_special_tokens=True,
        )
        enc = tokenizer.batch_encode_plus(
            texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        rels.append(embed_batch(enc))
        print(rels[-1])

    # if rel > 3 or rel < -3:
    #     out.write(
    #         str(rel)
    #         + "\n"
    #         + pythia_tokenizer.decode(d["input_ids"][:2048])
    #         + "\n"
    #         + pythia_tokenizer.decode(d["input_ids"][2048:])
    #         + "\n\n"
    #     )

np.save("dim-rels.npy", rels)
print(cnt)
# print(rels)
# print(max(rels))

from scipy.stats import pearsonr, spearmanr
import numpy as np
import datasets

base_dir = "/home/zichunyu/out/dclm_logs/baseline_01_1_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000/checkpoints/epoch_2"
dataset = datasets.load_from_disk(f"{base_dir}/oracle/0")
base_dir = "/home/zichunyu/out/dclm_logs/baseline_01_1_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000/checkpoints/epoch_4"
dataset_2 = datasets.concatenate_datasets(
    [
        datasets.load_from_disk(f"{base_dir}/oracle/epoch_2/102012"),
        # datasets.load_from_disk(f"{base_dir}/oracle/epoch_1/0"),
    ]
)
# 0.42936883585415975 (32-1024)
# 0.553359333983349 (2-3)
# 0.4756888922223056 (2-4)
scores = np.array(dataset["scores"])[:200, 0]
print(len(scores))
scores_2 = np.array(dataset_2["scores"])[:, 0]
print(len(scores_2))
print(spearmanr(scores, scores_2))
