#!/bin/bash
# JBB-only compliance-side layer sweep. Mirror of run_steer_layer_sweep_mixed.sh.
#
# Fixed intervention: L40-L70 x negative targets {-4, -8}. Fifteen JBB
# bare harmful goals. Tests whether steering toward the compliance pole
# of the axis flips normally-refused prompts to compliance.
#
# Usage:
#   chmod +x run_steer_layer_sweep_jbb_compliance.sh
#   ./run_steer_layer_sweep_jbb_compliance.sh                    # build PCA axes + sweep
#   ./run_steer_layer_sweep_jbb_compliance.sh pca                # explicit PCA
#   ./run_steer_layer_sweep_jbb_compliance.sh mean_diff          # mean-diff axes
#   ./run_steer_layer_sweep_jbb_compliance.sh reuse <axes.pt>    # reuse saved axes (must cover L40-L70)

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1

MODE="${1:-pca}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
LAYER_RANGES="40-70"
TARGETS="-4,-8"
N_PROMPTS=15
CALIBRATION_DIR="${CALIBRATION_DIR:-Compliant-refusal}"

if [ "$MODE" = "pca" ] || [ "$MODE" = "mean_diff" ]; then
    AXIS_METHOD="$MODE"
    OUTPUT_DIR="steer_layer_sweep_jbb_compliance_${AXIS_METHOD}"
    echo "============================================"
    echo "  JBB compliance-side layer sweep -- BUILD"
    echo "  Model:             $MODEL"
    echo "  Axis method:       $AXIS_METHOD"
    echo "  Calibration dir:   $CALIBRATION_DIR"
    echo "  Layer ranges:      $LAYER_RANGES"
    echo "  Targets:           $TARGETS  (negative = compliance pole)"
    echo "  Prompts:           $N_PROMPTS JBB bare harmful goals"
    echo "  Output:            $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_layer_sweep_jbb_compliance.py \
        --build \
        --model "$MODEL" \
        --axis-method "$AXIS_METHOD" \
        --calibration-dir "$CALIBRATION_DIR" \
        --layer-ranges="$LAYER_RANGES" \
        --targets="$TARGETS" \
        --n-prompts "$N_PROMPTS" \
        --output-dir "$OUTPUT_DIR" \
        --include-baseline
elif [ "$MODE" = "reuse" ]; then
    AXES_PATH="${2:?reuse mode needs an axes path as arg 2}"
    OUTPUT_DIR="steer_layer_sweep_jbb_compliance_reuse"
    echo "============================================"
    echo "  JBB compliance-side layer sweep -- REUSE"
    echo "  Model:          $MODEL"
    echo "  Axes file:      $AXES_PATH"
    echo "  Layer ranges:   $LAYER_RANGES"
    echo "  Targets:        $TARGETS  (negative = compliance pole)"
    echo "  Prompts:        $N_PROMPTS JBB bare harmful goals"
    echo "  Output:         $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_layer_sweep_jbb_compliance.py \
        --axes-path "$AXES_PATH" \
        --model "$MODEL" \
        --layer-ranges="$LAYER_RANGES" \
        --targets="$TARGETS" \
        --n-prompts "$N_PROMPTS" \
        --output-dir "$OUTPUT_DIR" \
        --include-baseline
else
    echo "Unknown mode: $MODE  (expected 'pca', 'mean_diff', or 'reuse')"
    exit 1
fi

echo ""
echo "Done. Eyeball $OUTPUT_DIR/steer_layer_sweep_jbb_compliance.csv -- group by 'target' column."
