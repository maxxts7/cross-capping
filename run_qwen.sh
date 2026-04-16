#!/bin/bash
# Run the cross-axis capping experiment on Qwen3-32B (single GPU).
#
# Assistant-axis capping uses the original paper's exact vectors and
# thresholds from capping_config.pt (experiment: layers_46:54-p0.25).
#
# Usage:
#   chmod +x run_qwen.sh
#   ./run_qwen.sh                          # full run, default threshold (mean+std)
#   ./run_qwen.sh sanity                   # smoke test (10 jailbreak + 10 benign)
#   ./run_qwen.sh full optimal             # full run with midpoint threshold
#   ./run_qwen.sh full p25                 # full run with 25th percentile threshold
#
# Compliance threshold options: mean+std (default), optimal, mean, p25

set -e

PRESET="${1:-full}"
THRESHOLD="${2:-mean+std}"
OUTPUT_DIR="results/crosscap_qwen_${PRESET}_${THRESHOLD}"

echo "============================================"
echo "  Cross-Axis Capping -- Qwen3-32B"
echo "  Preset:               ${PRESET}"
echo "  Compliance threshold: ${THRESHOLD}"
echo "  Output:               ${OUTPUT_DIR}"
echo "============================================"
echo ""

python run_crosscap.py \
    --preset "$PRESET" \
    --compliance-threshold "$THRESHOLD" \
    --output-dir "$OUTPUT_DIR"
