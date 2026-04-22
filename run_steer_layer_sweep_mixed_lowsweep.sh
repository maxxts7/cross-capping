#!/bin/bash
# Mixed-prompt low-target sweep. Companion to run_steer_layer_sweep_mixed.sh.
#
# Same 15 prompts (5 alpaca + 5 jbb vanilla + 5 wj_eval adversarial) and
# same L40-L70 range, but sweeps targets across the gentle zone: 2, 4, 5, 7.
# Fills in the sub-8 band between zero push and the target=8 single-point
# result from the earlier mixed run -- where does harmful refusal start
# showing up, and where does benign collateral damage begin?
#
# Reuses steer_layer_sweep_mixed.py (which already accepts --targets).
#
# Usage:
#   chmod +x run_steer_layer_sweep_mixed_lowsweep.sh
#   ./run_steer_layer_sweep_mixed_lowsweep.sh                    # build PCA + sweep
#   ./run_steer_layer_sweep_mixed_lowsweep.sh pca                # explicit PCA
#   ./run_steer_layer_sweep_mixed_lowsweep.sh mean_diff          # mean-diff axes
#   ./run_steer_layer_sweep_mixed_lowsweep.sh reuse <axes.pt>    # reuse saved axes

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1

MODE="${1:-pca}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
LAYER_RANGES="40-70"
TARGETS="2,4,5,7"
CALIBRATION_DIR="${CALIBRATION_DIR:-Compliant-refusal}"

if [ "$MODE" = "pca" ] || [ "$MODE" = "mean_diff" ]; then
    AXIS_METHOD="$MODE"
    OUTPUT_DIR="steer_layer_sweep_mixed_lowsweep_${AXIS_METHOD}"
    echo "============================================"
    echo "  Mixed-prompt low-target sweep -- BUILD"
    echo "  Model:             $MODEL"
    echo "  Axis method:       $AXIS_METHOD"
    echo "  Calibration dir:   $CALIBRATION_DIR"
    echo "  Layer ranges:      $LAYER_RANGES"
    echo "  Targets:           $TARGETS"
    echo "  Prompts:           5 alpaca + 5 jbb vanilla + 5 wj_eval adversarial"
    echo "  Output:            $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_layer_sweep_mixed.py \
        --build \
        --model "$MODEL" \
        --axis-method "$AXIS_METHOD" \
        --calibration-dir "$CALIBRATION_DIR" \
        --layer-ranges="$LAYER_RANGES" \
        --targets="$TARGETS" \
        --output-dir "$OUTPUT_DIR" \
        --include-baseline
elif [ "$MODE" = "reuse" ]; then
    AXES_PATH="${2:?reuse mode needs an axes path as arg 2}"
    OUTPUT_DIR="steer_layer_sweep_mixed_lowsweep_reuse"
    echo "============================================"
    echo "  Mixed-prompt low-target sweep -- REUSE"
    echo "  Model:          $MODEL"
    echo "  Axes file:      $AXES_PATH"
    echo "  Layer ranges:   $LAYER_RANGES"
    echo "  Targets:        $TARGETS"
    echo "  Output:         $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_layer_sweep_mixed.py \
        --axes-path "$AXES_PATH" \
        --model "$MODEL" \
        --layer-ranges="$LAYER_RANGES" \
        --targets="$TARGETS" \
        --output-dir "$OUTPUT_DIR" \
        --include-baseline
else
    echo "Unknown mode: $MODE  (expected 'pca', 'mean_diff', or 'reuse')"
    exit 1
fi

echo ""
echo "Done. Eyeball $OUTPUT_DIR/steer_layer_sweep_mixed.csv -- group by 'source' and 'target' columns."
