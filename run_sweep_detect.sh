#!/bin/bash
# Sweep absolute values of the assistant-axis detection threshold.
#
# Self-contained: a warmup file is no longer required. If one isn't found
# at the requested path (or the default warmupaxes/<model>_warmup.pt),
# the Python script computes a fresh warmup (axes + compliance taus) and
# saves it there for re-use. After that, the cross-cap generation phase
# is run for each tau in the sweep, with baselines generated once and
# reused across all taus.
#
# After the sweep, optionally runs reclassify_refusals.py per tau-dir
# so each setting gets HarmBench-judged jailbreak labels and Claude-
# judged benign labels (same backends as run_llama.sh).
#
# Usage:
#   chmod +x run_sweep_detect.sh
#   ./run_sweep_detect.sh <output-dir> [taus] [n_jb] [n_bn] [reclassify] [warmup] [model]
#
# Examples:
#   ./run_sweep_detect.sh sweep_detect_llama
#   ./run_sweep_detect.sh sweep_detect_llama "-8,-4,0,4,8,12"
#   ./run_sweep_detect.sh sweep_detect_llama "-8,0,8,16" 100 100
#   ./run_sweep_detect.sh sweep_detect_llama default 50 50 no
#   ./run_sweep_detect.sh sweep_detect_qwen default 50 50 yes "" Qwen/Qwen3-32B
#   ./run_sweep_detect.sh sweep_detect_llama default 50 50 yes llama_full/warmup.pt
#
# Args (positional, all but the first optional):
#   1  OUTPUT_DIR    directory for per-tau results + summary.csv (required)
#   2  TAUS          comma-separated absolute detect-tau values
#                    (default: "-8,-4,0,4,8,12,16")
#   3  N_JAILBREAK   number of jailbreak prompts (default: 50)
#   4  N_BENIGN      number of benign prompts (default: 50)
#   5  RECLASSIFY    yes (default) or no -- judge each tau-dir after the sweep
#   6  WARMUP        explicit warmup.pt path. Empty (default) -> the Python
#                    script uses warmupaxes/<model>_warmup.pt and computes
#                    one if it doesn't exist.
#   7  MODEL         model id (default: meta-llama/Llama-3.3-70B-Instruct)

set -e

# Match run_llama.sh: enable hf_transfer for fast 70B shard pulls.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: paste a key here for unattended runs (only used by reclassify).
ANTHROPIC_API_KEY_OVERRIDE=""

if [ -z "$1" ]; then
    echo "Usage: $0 <output-dir> [taus] [n_jb] [n_bn] [reclassify] [warmup] [model]" >&2
    exit 2
fi

OUTPUT_DIR="$1"
TAUS_RAW="${2:-default}"
N_JB="${3:-50}"
N_BN="${4:-50}"
RECLASSIFY="${5:-yes}"
WARMUP="${6:-}"
MODEL="${7:-meta-llama/Llama-3.3-70B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [ "$TAUS_RAW" = "default" ]; then
    TAUS="-8,-4,0,4,8,12,16"
else
    TAUS="$TAUS_RAW"
fi

# Anthropic key resolution for the reclassify step (benign files only,
# since jailbreak files go through HarmBench locally). Same logic as
# run_llama.sh; skipped entirely if RECLASSIFY=no.
KEY_INFO="not needed (reclassify disabled)"
if [ "$RECLASSIFY" = "yes" ]; then
    if [ -n "$ANTHROPIC_API_KEY_OVERRIDE" ]; then
        ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_OVERRIDE"
    fi
    if [ -z "$ANTHROPIC_API_KEY" ] && [ -f .env ]; then
        # shellcheck disable=SC1091
        set -a
        . ./.env
        set +a
    fi
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        printf "Enter ANTHROPIC_API_KEY (hidden, or press Enter to skip reclassify): "
        read -rs ANTHROPIC_API_KEY
        echo ""
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "No key provided -- reclassify will be skipped."
            RECLASSIFY="no"
            KEY_INFO="skipped (no key)"
        else
            export ANTHROPIC_API_KEY
            KEY_INFO="set (${#ANTHROPIC_API_KEY} chars)"
        fi
    else
        export ANTHROPIC_API_KEY
        KEY_INFO="set (${#ANTHROPIC_API_KEY} chars)"
    fi
fi

# Resolve warmup display label for the banner. The Python script handles
# the actual path/default; we only need a human-readable line here.
if [ -n "$WARMUP" ]; then
    WARMUP_INFO="$WARMUP (forced)"
else
    WARMUP_INFO="warmupaxes/<model>_warmup.pt (auto-create if missing)"
fi

echo "============================================"
echo "  Detect-tau sweep"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Taus:          ${TAUS}"
echo "  Jailbreak N:   ${N_JB}"
echo "  Benign N:      ${N_BN}"
echo "  Max tokens:    ${MAX_NEW_TOKENS}"
echo "  Model:         ${MODEL}"
echo "  Warmup:        ${WARMUP_INFO}"
echo "  Reclassify:    ${RECLASSIFY}"
echo "  Anthropic key: ${KEY_INFO}"
echo "============================================"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p warmupaxes

# Skip baseline regen if a baselines.csv already exists in OUTPUT_DIR --
# matches the script's --skip-baselines flag so re-sweeps with a different
# tau list are cheap.
SKIP_BASELINES_FLAG=""
if [ -f "${OUTPUT_DIR%/}/baselines.csv" ]; then
    echo "Found existing baselines at ${OUTPUT_DIR%/}/baselines.csv -- reusing."
    SKIP_BASELINES_FLAG="--skip-baselines"
fi

# Pass --warmup only if explicitly provided; otherwise let the script use
# its default path under warmupaxes/.
WARMUP_FLAG=()
if [ -n "$WARMUP" ]; then
    WARMUP_FLAG=(--warmup "$WARMUP")
fi

LOG="${OUTPUT_DIR%/}/sweep_$(date +%Y-%m-%dT%H-%M-%S).log"

python sweep_detect_thresholds.py \
    --taus "$TAUS" \
    --output-dir "$OUTPUT_DIR" \
    --n-jailbreak "$N_JB" \
    --n-benign "$N_BN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --model "$MODEL" \
    "${WARMUP_FLAG[@]}" \
    $SKIP_BASELINES_FLAG 2>&1 | tee "$LOG"

if [ "$RECLASSIFY" = "yes" ]; then
    echo ""
    echo "── Reclassify each tau dir (HarmBench jailbreak, Claude benign) ───────"
    # Iterate the per-tau dirs created by the sweep. Each one looks like
    # the standard cross-cap output dir from run_crosscap, so the existing
    # reclassify pipeline works unchanged.
    for d in "${OUTPUT_DIR%/}"/tau_*/; do
        [ -d "$d" ] || continue
        echo ""
        echo "── ${d} ──"
        python reclassify_refusals.py \
            --input-dir "$d" \
            --backend harmbench
    done
fi

echo ""
echo "============================================"
echo "  Sweep complete."
echo "  Per-tau results: ${OUTPUT_DIR%/}/tau_*/"
echo "  Summary table:   ${OUTPUT_DIR%/}/summary.csv"
echo "  Log:             ${LOG}"
echo "============================================"
