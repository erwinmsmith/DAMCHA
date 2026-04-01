#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# 6 models × 4 datasets = 24 experiments
#
# Models  : m_attention (ours), tha, dcmha, colmha, mma, moa
# Datasets: cifar10, cifar100  (CV)  |  commongen, wmt  (NLP)
#
# Usage:
#   bash run_all_experiments.sh              # run all 24
#   bash run_all_experiments.sh --dry-run    # print commands only
#   SKIP_DONE=1 bash run_all_experiments.sh  # skip if log already exists
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------
EPOCHS=50
CV_BATCH=64
NLP_BATCH=32
NUM_WORKERS=4
SEED=42
LOG_DIR="./logs/experiments"
CONDA_ENV="damcha"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
M_ATTENTION_ARGS="--use_M --use_mlp --share_mlp"
BASELINES=(tha dcmha colmha mma moa)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL:${NC} $*"; }

# Summary tracking
TOTAL=0; PASSED=0; FAILED=0
declare -a FAILED_LIST=()

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# run_experiment <dataset> <model_tag> <extra_args>
# ---------------------------------------------------------------------------
run_experiment() {
    local dataset="$1"
    local model_tag="$2"
    local extra_args="$3"
    local log_file="$LOG_DIR/${dataset}_${model_tag}.log"

    TOTAL=$((TOTAL + 1))

    # Skip if log already exists and SKIP_DONE=1
    if [[ "${SKIP_DONE:-0}" == "1" && -f "$log_file" ]]; then
        warn "Skipping (log exists): $dataset / $model_tag"
        PASSED=$((PASSED + 1))
        return 0
    fi

    # Dataset-specific settings
    local batch extra_data=""
    if [[ "$dataset" == "cifar10" || "$dataset" == "cifar100" ]]; then
        batch=$CV_BATCH
        extra_data="--in_ch 3 --img_size 32 --patch_size 4"
    else
        batch=$NLP_BATCH
        # Enable network accelerator for NLP (HuggingFace / dataset downloads)
        source /etc/network_turbo 2>/dev/null || true
    fi

    local cmd="conda run -n ${CONDA_ENV} python main.py \
        --dataset ${dataset} \
        --epochs ${EPOCHS} \
        --batch_size ${batch} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        ${extra_data} \
        ${extra_args}"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Experiment ${TOTAL}/24 │ dataset=${dataset} │ model=${model_tag}"
    log "Log  → ${log_file}"
    log "CMD  → $(echo $cmd | tr -s ' ')"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $cmd"
        PASSED=$((PASSED + 1))
        return 0
    fi

    # Run and tee to log file; capture exit code
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        log "✓ DONE  │ dataset=${dataset} │ model=${model_tag}"
        PASSED=$((PASSED + 1))
    else
        fail "✗ FAILED │ dataset=${dataset} │ model=${model_tag}  (see ${log_file})"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("${dataset}/${model_tag}")
    fi
    echo ""
}

# =============================================================================
# CV datasets
# =============================================================================
for dataset in cifar10 cifar100; do
    log "════════════════════════════════════"
    log "CV dataset: ${dataset}"
    log "════════════════════════════════════"

    # Our method
    run_experiment "$dataset" "m_attention" "$M_ATTENTION_ARGS"

    # 5 baselines
    for bl in "${BASELINES[@]}"; do
        run_experiment "$dataset" "$bl" "--baseline ${bl}"
    done
done

# =============================================================================
# NLP datasets
# =============================================================================
for dataset in commongen wmt; do
    log "════════════════════════════════════"
    log "NLP dataset: ${dataset}"
    log "════════════════════════════════════"

    # Our method
    run_experiment "$dataset" "m_attention" "$M_ATTENTION_ARGS"

    # 5 baselines
    for bl in "${BASELINES[@]}"; do
        run_experiment "$dataset" "$bl" "--baseline ${bl}"
    done
done

# =============================================================================
# Final summary
# =============================================================================
echo ""
log "════════════════════════════════════════════════════════"
log "All experiments finished."
log "  Total : ${TOTAL}"
log "  Passed: ${PASSED}"
log "  Failed: ${FAILED}"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    fail "Failed experiments:"
    for f in "${FAILED_LIST[@]}"; do
        fail "  - $f"
    done
fi
log "Logs saved to: ${LOG_DIR}/"
log "════════════════════════════════════════════════════════"
