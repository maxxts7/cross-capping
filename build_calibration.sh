#!/bin/bash
# Build an outcome-labelled calibration dataset by running Llama-3.3-70B
# uncapped on a batch of potentially-harmful prompts and judging each output
# with Claude. Produces clean refusing.csv / compliant.csv splits grounded
# in actual model behaviour, not dataset-of-origin labels.
#
# Note: Llama-3.3-70B in bf16 needs ~140 GB of GPU memory. device_map="auto"
# will spread layers across visible GPUs; make sure CUDA_VISIBLE_DEVICES
# exposes enough of them (typically 2+ H100/A100-80GB or 4+ smaller).
#
# The script will prompt for ANTHROPIC_API_KEY (hidden input) if it's not
# already in env or in a .env file.
#
# Usage:
#   chmod +x build_calibration.sh
#   ./build_calibration.sh                      # defaults: both sources, 200 prompts
#   ./build_calibration.sh both 500             # scale up to 500 total
#   ./build_calibration.sh jbb 100              # JBB only, 100 prompts
#   ./build_calibration.sh wj 300               # WildJailbreak train only
#
# Source options:   jbb, wj, both  (default: both)
# n_prompts:        total across selected sources (default: 200)

set -e

# Paste your key between the quotes for quick runs. Leave empty to fall
# back to env var / .env / interactive prompt. Don't commit a real key --
# this file is tracked in git.
ANTHROPIC_API_KEY_OVERRIDE=""

SOURCE="${1:-both}"
N_PROMPTS="${2:-200}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
OUTPUT_DIR="data/outcome_labeled_${SOURCE}_n${N_PROMPTS}"

# API key resolution order: the override constant above, then an existing
# env var, then .env in the cwd, then interactive prompt. First non-empty
# wins. Whatever is captured gets exported so the Python child process
# inherits it and doesn't need to prompt mid-run.
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

echo "============================================"
echo "  Outcome-labelled calibration dataset"
echo "  Source:      ${SOURCE}"
echo "  N prompts:   ${N_PROMPTS}"
echo "  Model:       ${MODEL}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Judge key:   set (${#ANTHROPIC_API_KEY} chars)"
echo "============================================"
echo ""

python build_outcome_labeled_calibration.py \
    --source "$SOURCE" \
    --n-prompts "$N_PROMPTS" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL" \
    --resume
