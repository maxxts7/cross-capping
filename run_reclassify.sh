#!/bin/bash
# Re-run reclassify_refusals.py on an existing results directory. Useful when:
#   - the original run finished but reclassify was skipped (RECLASSIFY=no)
#   - reclassify failed/got rate-limited and you want to resume
#   - you want to re-judge with a different model or backend
#
# Backends:
#   harmbench  -- HarmBench-Llama-2-13b-cls (local, ~30 GB GPU). Used for
#                 jailbreak files. Benign files always go through Claude
#                 since HarmBench has no notion of "benign-unchanged."
#   anthropic  -- Claude for everything. Forces Claude on jailbreak files
#                 too; useful when no GPU is available.
#
# Usage:
#   chmod +x run_reclassify.sh
#   ./run_reclassify.sh <input-dir>
#   ./run_reclassify.sh <input-dir> harmbench
#   ./run_reclassify.sh <input-dir> anthropic
#   ./run_reclassify.sh <input-dir> anthropic 2 yes               # concurrency=2, resume
#   ./run_reclassify.sh <input-dir> anthropic 5 no claude-haiku-4-5   # different judge model
#
# Args (positional, all but the first optional):
#   1  INPUT_DIR    directory holding the result CSVs (required)
#   2  BACKEND      harmbench (default) | anthropic
#   3  CONCURRENCY  parallel API calls for anthropic backend (default 2,
#                   bump to 5+ if your tier has high rate limits)
#   4  RESUME       yes (default) -- skip rows already labelled
#                   no            -- relabel everything from scratch
#   5  MODEL        anthropic judge model (default claude-sonnet-4-6)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input-dir> [backend] [concurrency] [resume] [model]" >&2
    exit 2
fi

INPUT_DIR="$1"
BACKEND="${2:-harmbench}"
CONCURRENCY="${3:-2}"
RESUME="${4:-yes}"
MODEL="${5:-claude-sonnet-4-6}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input dir not found: $INPUT_DIR" >&2
    exit 1
fi

# Anthropic-key resolution. Required when the judge will hit Claude:
#   - benign files always do
#   - jailbreak files do too if --backend anthropic
#
# We always show the prompt so the key source is explicit on every run.
# Pressing Enter falls back to the script-pasted override / env / .env;
# typing a key uses what you typed for this run only (no persistence).
ANTHROPIC_API_KEY_OVERRIDE=""

EXISTING_KEY=""
if [ -n "$ANTHROPIC_API_KEY_OVERRIDE" ]; then
    EXISTING_KEY="$ANTHROPIC_API_KEY_OVERRIDE"
    KEY_SOURCE="script override"
elif [ -n "$ANTHROPIC_API_KEY" ]; then
    EXISTING_KEY="$ANTHROPIC_API_KEY"
    KEY_SOURCE="env var"
elif [ -f .env ]; then
    # shellcheck disable=SC1091
    set -a
    . ./.env
    set +a
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        EXISTING_KEY="$ANTHROPIC_API_KEY"
        KEY_SOURCE=".env"
    fi
fi

if [ -n "$EXISTING_KEY" ]; then
    printf "ANTHROPIC_API_KEY found (%s, %d chars). Press Enter to use it, " \
        "$KEY_SOURCE" "${#EXISTING_KEY}"
    printf "or type a different key (hidden): "
else
    printf "Enter ANTHROPIC_API_KEY (hidden, no fallback found -- required): "
fi
read -rs ENTERED_KEY
echo ""

if [ -n "$ENTERED_KEY" ]; then
    ANTHROPIC_API_KEY="$ENTERED_KEY"
    KEY_INFO="entered at prompt (${#ANTHROPIC_API_KEY} chars)"
elif [ -n "$EXISTING_KEY" ]; then
    ANTHROPIC_API_KEY="$EXISTING_KEY"
    KEY_INFO="from $KEY_SOURCE (${#ANTHROPIC_API_KEY} chars)"
else
    echo "No key provided -- aborting." >&2
    exit 1
fi
export ANTHROPIC_API_KEY

RESUME_FLAG=""
if [ "$RESUME" = "yes" ]; then
    RESUME_FLAG="--resume"
fi

echo "============================================"
echo "  Reclassify rerun"
echo "  Input dir:    ${INPUT_DIR}"
echo "  Backend:      ${BACKEND}"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Resume:       ${RESUME}"
echo "  Model:        ${MODEL}  (used for benign files always; for jailbreaks if backend=anthropic)"
echo "  Key:          ${KEY_INFO}"
echo "============================================"
echo ""

LOG="${INPUT_DIR%/}/reclassify_$(date +%Y-%m-%dT%H-%M-%S).log"

python reclassify_refusals.py \
    --input-dir "$INPUT_DIR" \
    --backend "$BACKEND" \
    --concurrency "$CONCURRENCY" \
    --model "$MODEL" \
    $RESUME_FLAG 2>&1 | tee "$LOG"

echo ""
echo "============================================"
echo "  Done. Log: $LOG"
echo "  Output:    ${INPUT_DIR%/}/*_reclassified.csv"
echo "============================================"
