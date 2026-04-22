#!/bin/bash
# Layer-range sweep on the compliance axis. Companion to run_steer_probe.sh.
#
# Fixes the target value and sweeps WHICH LAYERS receive the steering hook.
# Default: 4 runs = {L20-L71, L40-L70} x {target=8, target=16}.
#
# Question: is the refusal basin a property of the late band (L56-L71) only,
# or can a wider/earlier band also engage it -- and does a wide band at
# moderate target=8 match a narrower band at stronger target=16?
#
# Usage:
#   chmod +x run_steer_layer_sweep.sh
#   ./run_steer_layer_sweep.sh                    # build PCA axes + sweep
#   ./run_steer_layer_sweep.sh pca                # explicit PCA
#   ./run_steer_layer_sweep.sh mean_diff          # mean-diff axes
#   ./run_steer_layer_sweep.sh reuse <axes.pt>    # reuse saved axes (must cover L20-L71)

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1

MODE="${1:-pca}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
LAYER_RANGES="20-71,40-70"
TARGETS="8,16"
CALIBRATION_DIR="${CALIBRATION_DIR:-Compliant-refusal}"

if [ "$MODE" = "pca" ] || [ "$MODE" = "mean_diff" ]; then
    AXIS_METHOD="$MODE"
    OUTPUT_DIR="steer_layer_sweep_${AXIS_METHOD}"
    echo "============================================"
    echo "  Layer-range steering sweep -- BUILD"
    echo "  Model:             $MODEL"
    echo "  Axis method:       $AXIS_METHOD"
    echo "  Calibration dir:   $CALIBRATION_DIR"
    echo "  Layer ranges:      $LAYER_RANGES"
    echo "  Targets:           $TARGETS"
    echo "  Output:            $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_layer_sweep.py \
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
    OUTPUT_DIR="steer_layer_sweep_reuse"
    echo "============================================"
    echo "  Layer-range steering sweep -- REUSE"
    echo "  Model:          $MODEL"
    echo "  Axes file:      $AXES_PATH"
    echo "  Layer ranges:   $LAYER_RANGES"
    echo "  Targets:        $TARGETS"
    echo "  Output:         $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_layer_sweep.py \
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
echo "Done. Eyeball $OUTPUT_DIR/steer_layer_sweep.csv for per-(range,target) outputs."
