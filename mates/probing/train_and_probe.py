import os
import time
import math
import copy
import wandb
import torch
import fsspec
import random
import functools
import numpy as np
from tqdm import tqdm
from torch import optim
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from open_lm.params import parse_args
from open_lm.precision import get_autocast
from open_lm.model import create_model, Block
from open_lm.file_utils import pt_load, check_exists
from open_lm.losses import CrossEntropyLossWithZLoss
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


def train(model, optimizer, train_data, accumulation=False, acc_steps=1):
    loss = CrossEntropyLossWithZLoss()
    autocast = get_autocast("amp_bfloat16")
    maybe_no_sync = nullcontext
    # Don't sync gradients until the final batch for FSDP.
    if isinstance(model, FSDP) and accumulation:
        maybe_no_sync = model.no_sync
    with maybe_no_sync():
        with autocast():
            inputs, targets = (
                train_data[:, :-1].contiguous().long().cuda(),
                train_data[:, 1:].contiguous().long().cuda(),
            )
            out, _, _ = model(inputs)
            total_loss = loss(out.reshape(-1, model.vocab_size), targets.reshape(-1)) / acc_steps
        total_loss.backward()
    if not accumulation:
        if isinstance(model, FSDP):
            model.clip_grad_norm_(1, norm_type=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
        optimizer.step()
    return total_loss.item()


def train_evaluate(model, optimizer, train_data, val_dataloader):
    loss_before_train = evaluate_train(model, train_data)

    optimizer.zero_grad()
    loss = CrossEntropyLossWithZLoss(reduction="none")
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        total_loss = None
        cnt = 0
        for batch in val_dataloader:
            inputs, targets = batch["input_ids"][:, :-1], batch["labels"][:, 1:]
            out, _, _ = model(inputs)  # [bs, seq_len, vocab_size]
            targets = targets.reshape(-1)
            cur_loss = loss(out.reshape(-1, model.vocab_size), targets)
            if total_loss:
                total_loss += cur_loss[targets != -100].mean()
            else:
                total_loss = cur_loss[targets != -100].mean()
            cnt += 1
    total_loss /= cnt
    total_loss.backward()
    if isinstance(model, FSDP):
        model.clip_grad_norm_(1, norm_type=2.0)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
    optimizer.step()

    scores = evaluate_train(model, train_data) - loss_before_train
    print(scores.size())

    return torch.argsort(scores)[:256].cpu()


@torch.no_grad()
def evaluate(model, val_dataloader):
    model.eval()

    loss = torch.nn.CrossEntropyLoss(reduction="none")
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        total_loss = 0.0
        cnt = 0
        for batch in val_dataloader:
            inputs, targets = batch["input_ids"][:, :-1], batch["labels"][:, 1:]
            out, _, _ = model(inputs)  # [bs, seq_len, vocab_size]
            targets = targets.reshape(-1)
            cur_loss = loss(out.reshape(-1, model.vocab_size), targets)
            total_loss += cur_loss[targets != -100].mean().item()
            cnt += 1

    model.train()
    return [total_loss / cnt]


@torch.no_grad()
def evaluate_train(model, train_data):
    model.eval()

    loss = torch.nn.CrossEntropyLoss(reduction="none")
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        all_loss = torch.tensor([], device="cuda")
        for batch in train_data.split(16):
            inputs, targets = (
                batch[:, :-1].contiguous().long().cuda(),
                batch[:, 1:].contiguous().long().cuda(),
            )
            out, _, _ = model(inputs)  # [bs, seq_len, vocab_size]
            cur_loss = loss(out.reshape(-1, model.vocab_size), targets.reshape(-1))
            cur_loss = cur_loss.reshape(16, -1).mean(dim=1)
            all_loss = torch.cat([all_loss, cur_loss])

    model.train()
    return all_loss


def main(args):
    args = parse_args(args)
    # args.resume = "/home/zichunyu/out/dclm_logs/baseline_01_0_fasttext_10000-data_influence_model-flan-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_2.pt"
    # args.resume = "/project/flame/zichunyu/out/dclm_logs/baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_5.pt"
    args.resume = "/project/flame/zichunyu/out/dclm_logs/baseline_01_0_fasttext_epoch_2-data_influence_model-flan-bs1-group-10-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_6.pt"

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    random_seed(args.seed, 0)
    with torch.device(args.device):
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

    if args.rank == 0:
        wandb.init(project="dcnlp", name="scale=400m_4x-step=30k-decay_data=oracle")
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

    probe_data = "/project/flame/zichunyu/data/shard_0-49_hq.pt"
    of = fsspec.open(probe_data, "rb")
    with of as f:
        train_dataset = torch.load(f)
    # indices = np.load("mean_indices_arce+hellaswag_102400_2.0.npy")
    # new_train_dataset = []
    # for i in indices:
    #     new_train_dataset.append(train_dataset[i])
    # train_dataset = new_train_dataset
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

    val_data = "gs://cmu-gpucloud-zichunyu/data/tulu/train-1024.pt"
    of = fsspec.open(val_data, "rb")
    with of as f:
        val_dataloader = DataLoader(
            torch.load(f)[:128],
            batch_size=64,
            collate_fn=val_collate_fn,
        )

    oracle = []
    model.load_state_dict(sd)
    optimizer.load_state_dict(osd)
    init_lr = optimizer.param_groups[0]["lr"]
    print("init_lr: ", init_lr)

    # def get_wsd_lr(learning_rate, it) -> float:
    #     if it < 30:
    #         return learning_rate * it / 30
    #     if it < 100:
    #         return learning_rate
    #     # math.pow(0.5, (it) / (50))
    #     return learning_rate * math.pow(0.5, (it - 50) / (50))

    def get_wsd_lr(base_lr, step):
        start_cooldown_step = 0
        if step < start_cooldown_step:
            lr = base_lr
        else:
            e = step - start_cooldown_step
            es = 1599 - start_cooldown_step
            # linear decay if power == 1; polynomial decay otherwise;
            decay = (1 - (e / es)) ** 1.0
            lr = decay * (base_lr - 3e-05) + 3e-05
        return lr

    # minus = np.linspace(0, 0.01, 200)
    model.train()
    base = 0
    for i in tqdm(range(1600)):
        lr = get_wsd_lr(init_lr, i)
        print(f"Step {i}, learning rate: {lr:.6f}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model_state_copy = copy.deepcopy(model.state_dict())
        optimizer_state_copy = copy.deepcopy(optimizer.state_dict())

        train_data = torch.cat(train_dataset[i * 512 + base : (i + 1) * 512 + base])

        # scores = []
        # for j in range(512):
        #     model.load_state_dict(model_state_copy)
        #     optimizer.load_state_dict(optimizer_state_copy)

        #     optimizer.zero_grad()
        #     train(model, optimizer, train_data[j : j + 1])
        #     scores.append(evaluate(model, val_dataloader)[0])

        # selected_data = train_data[:1]
        # selected_index = np.argmin(scores)
        # selected_data = train_data[selected_index : selected_index + 1]
        # selected_indices = np.random.choice(np.arange(512), size=128, replace=False)
        # selected_indices = np.argsort(scores)[:128]

        model.load_state_dict(model_state_copy)
        optimizer.load_state_dict(optimizer_state_copy)
        del model_state_copy, optimizer_state_copy

        train_loss = 0.0
        optimizer.zero_grad()
        for j in range(0, 512, 128):
            # cur_data = train_data[selected_indices[j + args.rank * 16 : j + (args.rank + 1) * 16]]
            cur_data = train_data[j + args.rank * 16 : j + (args.rank + 1) * 16]
            # train_loss += train(model, optimizer, cur_data)
            train_loss += train(model, optimizer, cur_data, accumulation=(j != 384), acc_steps=4)

        ref_loss = evaluate(model, val_dataloader)
        print("eval loss", ref_loss)

        if args.rank == 0:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "ref_loss": ref_loss[0],
                    "step": i + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # oracle.append(
        #     {
        #         "eval_loss": eval_loss,
        #         "selected_index": selected_index,
        #     }
        # )

        if (i + 1) % 400 == 0:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                os.makedirs(f"{args.resume[:-3]}/hq_{i+1}", exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict()},
                    f"{args.resume[:-3]}/hq_{i+1}/epoch_6.pt",
                )

    # torch.save(oracle, f"dim/dim-pointwise-{base}.pt")
    # torch.save(oracle, f"oracle-{base}.pt"

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        # mean_indices_arce+hellaswag_102400_2.0
        # random, random-128, random-800, random-1600
        os.makedirs(f"{args.resume[:-3]}/hq_{base}", exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict()},
            f"{args.resume[:-3]}/hq_{base}/epoch_6.pt",
        )
