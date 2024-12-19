import os
import time
import torch
import random
import functools
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from open_lm.params import parse_args
from open_lm.precision import get_autocast
from open_lm.model import create_model, Block
from open_lm.file_utils import pt_load, check_exists
from open_lm.losses import CrossEntropyLossWithZLoss
from datasets import Dataset, Features, Sequence, Value
from litdata.streaming import StreamingDataset, TokensLoader
from open_lm.distributed import is_master, init_distributed_device
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def load_model(args, model):
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        if "_orig_mod" in next(iter(sd.items()))[0]:
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    else:
        sd = checkpoint
    return sd


def load_optimizer(args, model, optimizer):
    potential_checkpoint = args.resume.replace("epoch_", "optimizer_")
    if check_exists(potential_checkpoint):
        checkpoint = pt_load(potential_checkpoint, map_location="cpu")
    else:
        checkpoint = pt_load(args.resume, map_location="cpu")
    osd = checkpoint["optimizer"]
    osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
    return osd


def train(model, optimizer, train_data):
    optimizer.zero_grad()
    loss = CrossEntropyLossWithZLoss()
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        inputs, targets = (
            train_data[:, :-1].contiguous().long().cuda(),
            train_data[:, 1:].contiguous().long().cuda(),
        )
        out, _, _ = model(inputs)
        total_loss = loss(out.reshape(-1, model.vocab_size), targets.reshape(-1))
    total_loss.backward()
    if isinstance(model, FSDP):
        model.clip_grad_norm_(1, norm_type=2.0)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
    optimizer.step()


@torch.no_grad()
def evaluate(model, val_dataloader):
    model.eval()

    loss = torch.nn.CrossEntropyLoss(reduction="none")
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        total_loss = torch.tensor(0.0, device="cuda")
        cnt = 0
        for batch in val_dataloader:
            inputs, targets = batch["input_ids"][:, :-1], batch["labels"][:, 1:]
            out, _, _ = model(inputs)  # [bs, seq_len, vocab_size]
            targets = targets.reshape(-1)
            cur_loss = loss(out.reshape(-1, model.vocab_size), targets)
            total_loss += cur_loss[targets != -100].mean()
            cnt += 1

    model.train()
    return [(total_loss / cnt).item()]


def main(args):
    args = parse_args(args)
    # args.resume = "output/logs/baseline_01_0_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000/checkpoints/epoch_1.pt"
    # args.resume = "/data/datasets/hf_cache/dclm_logs/baseline_toy-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=8232325/checkpoints/epoch_1.pt"
    # args.resume = "/data/datasets/hf_cache/dclm_logs/baseline_toy-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=2232325120/checkpoints/epoch_5.pt"
    args.resume = "/data/datasets/hf_cache/dclm_logs/baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_1.pt"

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    random_seed(args.seed, 0)
    with torch.device(
        "meta" if args.experimental_meta_device and args.fsdp else args.device
    ):
        model = create_model(args)

    random_seed(args.seed, args.rank)
    if args.distributed:
        if args.fsdp:
            transformer_layer_cls = None

            transformer_layer_cls = {Block}
            transformer_auto_wrapper_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            )
            # tries to follow gopher...
            mp_policy = None
            if args.fsdp_amp:
                print("=> using bfloat16 params as part of fsdp amp policy.")
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
            elif args.fsdp_pure_bf16:
                print("=> using pure bfloat16 params as part of fsdp amp policy.")
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

            if args.rank == 0:
                print(
                    f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters()):,}"
                )
                print(f"Before FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB")

            fsdp_kwargs = {}
            assert not (
                args.fsdp_hybrid and args.fsdp_hybrid_o2
            ), "Only --fsdp-hybrid or --fsdp-hybrid-o2 should be set."
            if args.fsdp_backward_prefetch:
                fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            if args.fsdp_hybrid:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
            if args.fsdp_hybrid_o2:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy._HYBRID_SHARD_ZERO2
            print("=> FSDP kwargs: ", fsdp_kwargs)

            # Initialize FSDP. Use the same seed across workers to ensure reset_parameters is the same across workers.
            random_seed(args.seed, rank=0)
            model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrapper_policy,
                device_id=device,
                mixed_precision=mp_policy,
                cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
                use_orig_params=args.fsdp_use_orig_params,
                limit_all_gathers=args.fsdp_limit_all_gathers,
                **fsdp_kwargs,
            )

            print(
                f"After FSDP parameter num: {sum(p.numel() for p in model.parameters()):,} on rank {args.rank}"
            )
            print(
                f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
            )
        else:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args["static_graph"] = True
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], **ddp_args
            )

    sd = load_model(args, model)
    model.load_state_dict(sd)

    named_parameters = list(model.named_parameters())
    no_decay_params = []
    params = [p for n, p in named_parameters if p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    osd = load_optimizer(args, model, optimizer)

    probe_data = "/data/datasets/hf_cache/dclm-baseline-1.0/global-shard_01_of_10/local-shard_0_of_10/pythia_tokenized/train.pt"
    # probe_data = "/data/datasets/hf_cache/refinedweb_01_0/fasttext/fasttext_filter/processed_data/pythia_tokenized/train.pt"
    # probe_data = "/data/datasets/hf_cache/data/data_influence_model/pythia-1b/10000/bs-1-sample/train.pt"
    train_dataset = torch.load(probe_data)
    dataset_len = len(train_dataset)
    print(dataset_len)

    def val_collate_fn(batch):
        input_ids = [torch.tensor(s["input_ids"], device="cuda") for s in batch]
        labels = [torch.tensor(s["labels"], device="cuda") for s in batch]

        x = pad_sequence(input_ids, batch_first=True, padding_value=0)
        y = pad_sequence(labels, batch_first=True, padding_value=-100)

        x = x[:, :2048]
        y = y[:, :2048]

        return {"input_ids": x, "labels": y}

    val_dataloader = DataLoader(
        # torch.load("/data/datasets/hf_cache/data/lambada_openai/train-1024.pt")[:32],
        torch.load("/data/datasets/hf_cache/data/tulu/train-32.pt"),
        batch_size=16,
        collate_fn=val_collate_fn,
    )

    seed = int(os.environ.get("SEED"))
    print("SEED:", seed)
    np.random.seed(seed)
    ocache = f"/data/datasets/hf_cache/out/oracle/pythia-410m/epoch_1/{seed}"
    oracle = []
    cnt = 10000
    for i in tqdm(range(cnt)):
        model.load_state_dict(sd)
        optimizer.load_state_dict(osd)
        train_index, probe_index = np.random.permutation(dataset_len)[:2]
        train_data = train_dataset[train_index]
        probe_data = train_dataset[probe_index]
        scores = []
        train(model, optimizer, train_data)
        scores.append(evaluate(model, val_dataloader)[0])
        train(model, optimizer, probe_data)
        scores.append(evaluate(model, val_dataloader)[0])

        oracle.append(
            {
                "input_ids": train_data[0][:-1].cpu().numpy().tolist()
                + probe_data[0][:-1].cpu().numpy().tolist(),
                "scores": scores,
            }
        )
        if (i + 1) % 1000 == 0:
            if args.rank == 0:
                features = Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "scores": Sequence(Value("float32")),
                    }
                )
                processed_ds = Dataset.from_list(oracle, features=features)
                processed_ds.save_to_disk(ocache)
