import argparse
import json

import torch
import yaml
from open_lm.model import create_params
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from transformers import GPTNeoXTokenizerFast


def update_args_from_openlm_config(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        if k == "model" and args.model != None:
            continue
        if v == "None":
            v = None

        # we changed args
        if k == "batch_size":
            k = "per_gpu_batch_size"
        if k == "val_batch_size":
            k = "per_gpu_val_batch_size"
        if k == "val_data":
            continue

        setattr(args, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moe-freq",
        type=int,
        default=0,
        help="if set > 0, we will add MoE layer to every moe_freq layer.",
    )
    parser.add_argument(
        "--moe-num-experts",
        type=int,
        default=None,
        help="Number of experts for MoE",
    )

    parser.add_argument(
        "--moe-weight-parallelism",
        action="store_true",
        help="Add weight parallelism to MoE",
    )

    parser.add_argument(
        "--moe-expert-model-parallelism",
        action="store_true",
        help="Add expert model parallelism to MoE",
    )

    parser.add_argument(
        "--moe-capacity-factor",
        type=float,
        default=1.25,
        help="MoE capacity factor",
    )

    parser.add_argument(
        "--moe-loss-weight",
        type=float,
        default=0.1,
        help="MoE loss weight",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=2,
        help="MoE top k experts",
    )
    args = parser.parse_args()
    method = "10000-data_influence_model-flan_2-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"
    args.model = "training/open_lm_configs/open_lm_1b_swiglutorch.json"
    args.checkpoint = f"../tmp/logs/{method}/checkpoints/epoch_6.pt"
    args.config = f"../tmp/logs/{method}/params.txt"
    args.out_dir = "output/"
    update_args_from_openlm_config(args)
    checkpoint = torch.load(args.checkpoint)
    openlm_config = OpenLMConfig(create_params(args))
    lm = OpenLMforCausalLM(list(openlm_config))
    # hardcoded to NeoX Tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    lm.model.load_state_dict(state_dict)
    lm.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
