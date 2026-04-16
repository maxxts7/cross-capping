#!/bin/bash
# Run the cross-axis capping experiment on Qwen3-32B (single GPU).
#
# Assistant-axis capping uses the original paper's exact vectors and
# thresholds from capping_config.pt (experiment: layers_46:54-p0.25).
#
# Usage:
#   chmod +x run_qwen.sh
#   ./run_qwen.sh              # full run (250 jailbreak + 100 benign)
#   ./run_qwen.sh sanity       # smoke test (5 jailbreak + 10 benign)

set -e

PRESET="${1:-full}"
OUTPUT_DIR="results/crosscap_qwen_${PRESET}"

echo "============================================"
echo "  Cross-Axis Capping -- Qwen3-32B"
echo "  Preset:     ${PRESET}"
echo "  Output:     ${OUTPUT_DIR}"
echo "============================================"
echo ""

python run_crosscap.py \
    --preset "$PRESET" \
    --output-dir "$OUTPUT_DIR"
