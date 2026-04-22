#!/bin/bash
# Mixed-prompt layer sweep. Companion to run_steer_layer_sweep.sh.
#
# Fixed intervention: L40-L70 x target=8. Fifteen prompts spanning
# AlpacaEval benign + JBB bare harmful + WildJailbreak eval adversarial.
#
# Usage:
#   chmod +x run_steer_layer_sweep_mixed.sh
#   ./run_steer_layer_sweep_mixed.sh                    # build PCA axes + sweep
#   ./run_steer_layer_sweep_mixed.sh pca                # explicit PCA
#   ./run_steer_layer_sweep_mixed.sh mean_diff          # mean-diff axes
#   ./run_steer_layer_sweep_mixed.sh reuse <axes.pt>    # reuse saved axes (must cover L40-L70)

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1

MODE="${1:-pca}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
LAYER_RANGES="40-70"
TARGETS="8"
CALIBRATION_DIR="${CALIBRATION_DIR:-Compliant-refusal}"

if [ "$MODE" = "pca" ] || [ "$MODE" = "mean_diff" ]; then
    AXIS_METHOD="$MODE"
    OUTPUT_DIR="steer_layer_sweep_mixed_${AXIS_METHOD}"
    echo "============================================"
    echo "  Mixed-prompt layer sweep -- BUILD"
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
    OUTPUT_DIR="steer_layer_sweep_mixed_reuse"
    echo "============================================"
    echo "  Mixed-prompt layer sweep -- REUSE"
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
echo "Done. Eyeball $OUTPUT_DIR/steer_layer_sweep_mixed.csv -- group by 'source' column."
