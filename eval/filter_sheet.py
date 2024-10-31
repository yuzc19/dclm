import argparse
import json
import math
import os

import pandas as pd


low_variance_datasets = [
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
    "mmlu_fewshot",
]


def gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_meta_data",
        default=f"{os.path.dirname(__file__)}/eval_meta_data.csv",
        help="Eval meta data file",
    )
    parser.add_argument("--eval_results", help="Eval results")
    return parser


def get_aggregated_results(data, eval_metadata):
    eval_metadata["results"] = eval_metadata["Eval Task"].map(
        data["eval_metrics"]["icl"]
    )
    eval_metadata["centered results"] = (
        eval_metadata["results"].astype(float)
        - 0.01 * eval_metadata["Random baseline"].astype(float)
    ) / (1.0 - 0.01 * eval_metadata["Random baseline"].astype(float))
    # print(eval_metadata)
    # result_list = []
    task_categories = sorted(eval_metadata["Task Category"].unique())
    for c in task_categories:
        eval_df = eval_metadata[eval_metadata["Task Category"] == c]
        sorted_df = eval_df.sort_values(by="Eval Task")
        filtered_df = sorted_df[sorted_df["Eval Task"].isin(low_variance_datasets)]
        c_sum, c_num = 0, 0
        for r in filtered_df["centered results"]:
            if not math.isnan(r):
                c_sum += r
                c_num += 1
        if c_num == 0:
            continue
        c_avg = c_sum / c_num
        print(c, c_avg)


def main():
    parser = gen_parser()
    args = parser.parse_args()

    eval_metadata = pd.read_csv(args.eval_meta_data)

    with open(args.eval_results, "r") as f:
        data = json.load(f)

    data = get_aggregated_results(data, eval_metadata)


if __name__ == "__main__":
    main()
