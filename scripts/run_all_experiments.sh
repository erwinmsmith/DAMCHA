#!/usr/bin/env bash
# Run the repository's reconstruction and conditional text benchmarks.
# EPOCHS=1 PYTHON=python3 bash scripts/run_all_experiments.sh --dry-run
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-50}"
CV_BATCH="${CV_BATCH:-64}"
NLP_BATCH="${NLP_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-logs/experiments}"
DRY_RUN=0
if [[ "${1:-}" == '--dry-run' ]]; then DRY_RUN=1; fi
mkdir -p "$LOG_DIR"
failed=0
for dataset in cifar10 cifar100 commongen wmt; do
    batch="$CV_BATCH"
    if [[ "$dataset" == commongen || "$dataset" == wmt ]]; then batch="$NLP_BATCH"; fi
    for model in m_attention standard tha dcmha colmha mma moa; do
        log_file="$LOG_DIR/${dataset}_${model}.log"
        if [[ "${SKIP_DONE:-0}" == 1 && -f "${log_file}.done" ]]; then
            printf 'Skipping completed run: %s / %s\n' "$dataset" "$model"
            continue
        fi
        command=("$PYTHON" main.py --dataset "$dataset" --epochs "$EPOCHS"
                 --batch_size "$batch" --num_workers "$NUM_WORKERS" --seed "$SEED")
        if [[ "$model" == m_attention ]]; then
            command+=(--use_M --use_mlp --share_mlp)
        elif [[ "$model" != standard ]]; then
            command+=(--baseline "$model")
        fi
        printf '%q ' "${command[@]}"; printf '\n'
        if [[ "$DRY_RUN" == 1 ]]; then continue; fi
        rm -f "${log_file}.done"
        if "${command[@]}" 2>&1 | tee "$log_file"; then
            touch "${log_file}.done"
        else
            failed=$((failed + 1))
        fi
    done
done
printf 'Failed experiments: %s\n' "$failed"
[[ "$failed" == 0 ]]
