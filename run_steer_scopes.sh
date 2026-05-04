#!/bin/bash
# Run the layer-range steer sweep four times -- once per scope -- so we can
# compare what the compliance axis does to generation when the clamp acts
# at different positions:
#
#   all                clamp every forward pass (prefill cursor + every token)
#   prefill_only       clamp only the post-instruction cursor (step 0)
#   first_token_only   clamp only the first decoded token (step 1)
#   cursor_plus_first  clamp steps 0 and 1
#
# Each run emits steer_layer_sweep.csv with a "scope" column tagging the rows.
# Outputs land in OUT_ROOT/<scope>/ so a quick `grep refusal` across them gives
# you the comparison.
#
# Usage:
#   chmod +x run_steer_scopes.sh
#   ./run_steer_scopes.sh                              # all defaults
#   ./run_steer_scopes.sh mixed                        # use the mixed-prompt sweep
#   ./run_steer_scopes.sh mixed "40-70" "8" 128        # range, target, max_new_tokens
#   ./run_steer_scopes.sh basic "20-71,40-70" "8,16"   # multiple ranges/targets

set -e

# Fast HF downloads if hf_transfer is installed.
export HF_HUB_ENABLE_HF_TRANSFER=1

VARIANT="${1:-mixed}"          # basic | mixed | jbb
LAYER_RANGES="${2:-40-70}"
TARGETS="${3:-8}"
MAX_NEW_TOKENS="${4:-128}"
MODEL="${5:-meta-llama/Llama-3.3-70B-Instruct}"
SCOPES=(all prefill_only first_token_only cursor_plus_first)

case "$VARIANT" in
    basic)   SCRIPT="steer_layer_sweep.py" ;;
    mixed)   SCRIPT="steer_layer_sweep_mixed.py" ;;
    jbb)     SCRIPT="steer_layer_sweep_jbb_compliance.py" ;;
    *) echo "VARIANT must be one of: basic | mixed | jbb"; exit 2 ;;
esac

STAMP=$(date +%Y-%m-%dT%H-%M-%S)
OUT_ROOT="steer_scope_sweep/${VARIANT}_${STAMP}"
AXES_DIR="${OUT_ROOT}/_axes"
mkdir -p "$AXES_DIR"

echo "============================================"
echo "  Steer-scope comparison sweep"
echo "  Script:        $SCRIPT"
echo "  Model:         $MODEL"
echo "  Layer ranges:  $LAYER_RANGES"
echo "  Targets:       $TARGETS"
echo "  Max tokens:    $MAX_NEW_TOKENS"
echo "  Scopes:        ${SCOPES[*]}"
echo "  Output root:   $OUT_ROOT"
echo "============================================"
echo ""

# Build axes once over the union of all requested layer ranges, reuse across
# every scope so the only thing that varies between runs is when the clamp
# fires.
echo "── Building shared axes ──────────────────────────────────────────────"
python "$SCRIPT" \
    --build \
    --model "$MODEL" \
    --layer-ranges "$LAYER_RANGES" \
    --targets "$TARGETS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --scope "all" \
    --include-baseline \
    --output-dir "$AXES_DIR"

AXES_PATH="${AXES_DIR}/axes.pt"
if [ ! -f "$AXES_PATH" ]; then
    echo "Build step did not produce $AXES_PATH -- aborting."
    exit 1
fi
echo "Built axes -> $AXES_PATH"
echo ""

# Move the build run's csv into the 'all' scope dir so we don't repeat work --
# the build step already ran the full sweep with --scope all.
ALL_DIR="${OUT_ROOT}/all"
mkdir -p "$ALL_DIR"
mv "${AXES_DIR}"/steer_layer_sweep*.csv "$ALL_DIR/" 2>/dev/null || true
mv "${AXES_DIR}"/steer_layer_sweep*_traces.pt "$ALL_DIR/" 2>/dev/null || true

# Now run the remaining scopes against the same axes file.
for scope in prefill_only first_token_only cursor_plus_first; do
    OUT_DIR="${OUT_ROOT}/${scope}"
    mkdir -p "$OUT_DIR"
    echo "── scope=${scope} ────────────────────────────────────────────────────"
    python "$SCRIPT" \
        --axes-path "$AXES_PATH" \
        --model "$MODEL" \
        --layer-ranges "$LAYER_RANGES" \
        --targets "$TARGETS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --scope "$scope" \
        --output-dir "$OUT_DIR"
    echo ""
done

echo "============================================"
echo "  All scopes complete"
echo "  Compare:  ${OUT_ROOT}/<scope>/steer_layer_sweep*.csv"
echo "============================================"
