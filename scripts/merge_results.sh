

#!/usr/bin/env bash
set -euo pipefail

# Change this if you want a different run.
method="baseline_01_0_fasttext_7.2B-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=28795904000"
base_dir="results/${method}"

epochs=()
values=()

max_epoch="$(
  find "${base_dir}" -maxdepth 1 -type d -name 'epoch_*' \
    | sed 's#.*/epoch_##' | sort -n | tail -1
)"

if [[ -z "${max_epoch}" ]]; then
  echo "No epoch_* directories found under ${base_dir}" >&2
  exit 1
fi

for ((e = 1; e <= max_epoch; e++)); do
  csv_file="${base_dir}/epoch_${e}/metrics_mmlu_and_lowvar.csv"
  json_file="${base_dir}/epoch_${e}/metrics_mmlu_and_lowvar.json"

  if [[ -f "${json_file}" ]]; then
    python filter_sheet.py --eval_results "${json_file}"
  fi

  if [[ -f "${csv_file}" ]]; then
    # Prefer the csv output (python filter result).
    value="$(grep -Eo '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' "${csv_file}" | tail -1)"
  else
    value="NA"
  fi

  if [[ "${value}" == "NA" || -z "${value}" ]]; then
    continue
  fi

  epochs+=("${e}")
  values+=("${value}")
done

printf '%s\n' "$(printf '%s,' "${epochs[@]}")"
printf '%s\n' "$(printf '%s,' "${values[@]}")"
