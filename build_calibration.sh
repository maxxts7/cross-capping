#!/bin/bash
# Build an outcome-labelled calibration dataset by running Llama-3.3-70B
# uncapped on a batch of potentially-harmful prompts and judging each output.
# Produces clean refusing.csv / compliant.csv splits grounded in actual
# model behaviour, not dataset-of-origin labels.
#
# Note: Llama-3.3-70B in bf16 needs ~140 GB of GPU memory. device_map="auto"
# will spread layers across visible GPUs; make sure CUDA_VISIBLE_DEVICES
# exposes enough of them (typically 2+ H100/A100-80GB or 4+ smaller).
#
# Two judging backends:
#   anthropic  — Claude API. Needs ANTHROPIC_API_KEY (override below, env,
#                .env, or interactive prompt). Costs API tokens. Produces
#                4-class labels (refusal / compliance / partial_refusal / error).
#   harmbench  — local cais/HarmBench-Llama-2-13b-cls classifier. Needs
#                ~30 GB GPU memory loaded after the generation step. No
#                API cost. Binary refusal/compliance only.
#
# Usage:
#   chmod +x build_calibration.sh
#   ./build_calibration.sh                          # both, 200, anthropic
#   ./build_calibration.sh both 500                 # 500 prompts, anthropic
#   ./build_calibration.sh both 200 harmbench       # local classifier
#   ./build_calibration.sh jbb 100 harmbench        # JBB only, local
#
# Source:    jbb, wj, both     (default: both)
# n_prompts: total across selected sources (default: 200)
# backend:   anthropic, harmbench (default: anthropic)

set -e

# Paste your key between the quotes for quick runs. Leave empty to fall
# back to env var / .env / interactive prompt. Don't commit a real key --
# this file is tracked in git. Only used when backend=anthropic.
ANTHROPIC_API_KEY_OVERRIDE=""

SOURCE="${1:-both}"
N_PROMPTS="${2:-200}"
BACKEND="${3:-anthropic}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
OUTPUT_DIR="data/outcome_labeled_${SOURCE}_n${N_PROMPTS}_${BACKEND}"

if [ "$BACKEND" = "anthropic" ]; then
    # API key resolution: override constant -> existing env -> .env -> prompt.
    # First non-empty wins. Captured key is exported so the Python child
    # process inherits it.
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
        echo "ANTHROPIC_API_KEY not found in env, .env, or script override."
        printf "Enter ANTHROPIC_API_KEY (hidden): "
        read -rs ANTHROPIC_API_KEY
        echo ""
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "No key provided. Aborting."
            exit 1
        fi
    fi
    export ANTHROPIC_API_KEY
    KEY_INFO="set (${#ANTHROPIC_API_KEY} chars)"
elif [ "$BACKEND" = "harmbench" ]; then
    KEY_INFO="not needed (local classifier)"
else
    echo "Unknown backend: $BACKEND (expected: anthropic or harmbench)"
    exit 1
fi

echo "============================================"
echo "  Outcome-labelled calibration dataset"
echo "  Source:      ${SOURCE}"
echo "  N prompts:   ${N_PROMPTS}"
echo "  Generator:   ${MODEL}"
echo "  Judge:       ${BACKEND}"
echo "  Judge key:   ${KEY_INFO}"
echo "  Output:      ${OUTPUT_DIR}"
echo "============================================"
echo ""

python build_outcome_labeled_calibration.py \
    --source "$SOURCE" \
    --n-prompts "$N_PROMPTS" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL" \
    --judge-backend "$BACKEND" \
    --resume
