#!/bin/bash
# Direct-steering probe on the compliance axis for Llama-3.3-70B.
#
# Question: can we turn a BENIGN prompt into a refusal by forcing its
# compliance-axis projection to mean_refusing or beyond?
#
# If yes -> the axis is a causal refusal lever; cross-cap's threshold was
# just too low.
# If no  -> the axis is a correlational probe, not a lever; no threshold
# will flip output.
#
# Usage:
#   chmod +x run_steer_probe.sh
#   ./run_steer_probe.sh                    # build PCA axis + sweep (default)
#   ./run_steer_probe.sh pca                # explicit PCA
#   ./run_steer_probe.sh mean_diff          # mean-diff axis
#   ./run_steer_probe.sh reuse <axes.pt>    # reuse previously-saved axes
#
# Target range spans compliant-pool mean (~-3) through refusing-pool mean
# (~+12) and beyond. Rows worth eyeballing in the output CSV:
#   * benign prompt at target=+12 or +16 -- does it refuse?
#   * any target where output becomes garbled -- the over-steer boundary

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1

MODE="${1:-pca}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
TARGETS="-4,0,4,8,12,16,20"
CALIBRATION_DIR="${CALIBRATION_DIR:-Compliant-refusal}"

if [ "$MODE" = "pca" ] || [ "$MODE" = "mean_diff" ]; then
    AXIS_METHOD="$MODE"
    OUTPUT_DIR="steer_probe_${AXIS_METHOD}"
    echo "============================================"
    echo "  Direct-steering probe -- BUILD"
    echo "  Model:             $MODEL"
    echo "  Axis method:       $AXIS_METHOD"
    echo "  Calibration dir:   $CALIBRATION_DIR"
    echo "  Targets:           $TARGETS"
    echo "  Output:            $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_probe.py \
        --build \
        --model "$MODEL" \
        --axis-method "$AXIS_METHOD" \
        --calibration-dir "$CALIBRATION_DIR" \
        --targets "$TARGETS" \
        --output-dir "$OUTPUT_DIR" \
        --include-baseline
elif [ "$MODE" = "reuse" ]; then
    AXES_PATH="${2:-llama75/warmup (4).pt}"
    OUTPUT_DIR="steer_probe_reuse"
    echo "============================================"
    echo "  Direct-steering probe -- REUSE"
    echo "  Model:       $MODEL"
    echo "  Axes file:   $AXES_PATH"
    echo "  Targets:     $TARGETS"
    echo "  Output:      $OUTPUT_DIR"
    echo "============================================"
    echo ""
    python steer_probe.py \
        --axes-path "$AXES_PATH" \
        --model "$MODEL" \
        --targets "$TARGETS" \
        --output-dir "$OUTPUT_DIR" \
        --include-baseline
else
    echo "Unknown mode: $MODE  (expected 'pca', 'mean_diff', or 'reuse')"
    exit 1
fi

echo ""
echo "Done. Eyeball $OUTPUT_DIR/steer_probe.csv for per-target outputs."
