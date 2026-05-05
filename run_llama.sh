#!/bin/bash
# Run the cross-axis capping experiment on Llama-3.3-70B-Instruct, then
# auto-reclassify the result CSVs (HarmBench for jailbreak files, Claude
# for benign files).
#
# Auto-behaviours when MODEL is Llama:
#   * compliance axis is built from Compliant-refusal/{refusing,compliant}.csv
#     if that directory exists; falls back to JBB+WJ otherwise.
#   * reclassify_refusals.py runs with --backend harmbench (jailbreaks scored
#     locally; benign files still need Claude).
#
# Note: Llama-3.3-70B in bf16 needs ~140 GB of GPU memory. device_map="auto"
# will spread layers across visible GPUs; make sure CUDA_VISIBLE_DEVICES
# exposes enough of them (typically 2+ H100/A100-80GB or 4+ smaller).
# HarmBench-Llama-2-13b-cls loads after generation finishes and adds ~30 GB.
#
# Usage:
#   chmod +x run_llama.sh
#   ./run_llama.sh                                            # full run, defaults, reclassify
#   ./run_llama.sh sanity                                     # smoke test
#   ./run_llama.sh full optimal                               # midpoint threshold
#   ./run_llama.sh full p25                                   # 25th percentile threshold
#   ./run_llama.sh full mean+std benign-p1                    # tighter detect gate
#   ./run_llama.sh full optimal75 benign-p1 no                # skip reclassify step
#   ./run_llama.sh sanity optimal75 benign-p10 yes mean_diff  # mean-diff axis
#   ./run_llama.sh full 16                                    # literal tau=16, default L40-L70
#   ./run_llama.sh full optimal75 benign-p1 yes pca ""        # fall back to paper L56-L71 only
#   ./run_llama.sh full optimal75 benign-p1 yes pca 30-70     # different override range
#
# Compliance threshold options:  optimal75 (default), optimal, optimal90, optimal20, mean+std, mean, p25,
#                                 OR a literal number (e.g. 16) -> used as tau on every cap layer
# Cross-detect method options:   benign-p1 (default), benign-p5, benign-p10,
#                                 OR a literal number (e.g. 4) -> used as tau on every cap layer
# Reclassify options:            yes (default), no
# Axis method options:           pca (default), mean_diff
# Compliance-layer override:     default "40-70" inclusive -- Mode 3 cross-cap extends down to L40
#                                 (Mode 2 assistant-cap stays on the paper's L56-L71). Pass "" to
#                                 disable the override and use paper L56-L71 for both modes.

set -e

# Enable the fast Rust-based HF downloader. Without this, 70B shard pulls
# fall back to the slow single-threaded path and look stuck for minutes.
# hf_transfer must be installed in the active Python env (pip install hf_transfer).
export HF_HUB_ENABLE_HF_TRANSFER=1

# Paste your key between the quotes for unattended runs. Leave empty to
# fall back to env var / .env / interactive prompt. Don't commit a real
# key -- this file is tracked in git. Only used by the reclassify step
# (for benign files; jailbreak files use HarmBench locally).
ANTHROPIC_API_KEY_OVERRIDE=""

PRESET="${1:-full}"
THRESHOLD="${2:-optimal75}"
CROSS_DETECT="${3:-benign-p1}"
RECLASSIFY="${4:-yes}"
AXIS_METHOD="${5:-pca}"
# Default cross-cap (Mode 3) range is L40-L70 -- wider than the paper's L56-L71.
# Pass "" as the 6th arg to fall back to the paper's published range.
COMPLIANCE_LAYERS="${6-40-65}"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
LAYER_TAG=""
if [ -n "$COMPLIANCE_LAYERS" ]; then
    LAYER_TAG="_L${COMPLIANCE_LAYERS}"
fi
OUTPUT_DIR="results/crosscap_llama_${PRESET}_${THRESHOLD}_${CROSS_DETECT}_${AXIS_METHOD}${LAYER_TAG}"

# API key resolution for the reclassify step. Captured upfront so the long
# generation can run unattended; benign reclassify needs it at the end.
# Skipped entirely if RECLASSIFY=no.
KEY_INFO="not needed (reclassify disabled)"
if [ "$RECLASSIFY" = "yes" ]; then
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
        echo "Reclassify needs it for benign files (jailbreaks use HarmBench locally)."
        printf "Enter ANTHROPIC_API_KEY (hidden, or just press Enter to skip reclassify): "
        read -rs ANTHROPIC_API_KEY
        echo ""
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "No key provided -- reclassify will be skipped after the run."
            RECLASSIFY="no"
            KEY_INFO="skipped (no key)"
        else
            export ANTHROPIC_API_KEY
            KEY_INFO="set (${#ANTHROPIC_API_KEY} chars)"
        fi
    else
        export ANTHROPIC_API_KEY
        KEY_INFO="set (${#ANTHROPIC_API_KEY} chars)"
    fi
fi

CALIB_INFO="default (JBB + WildJailbreak)"
if [ -d Compliant-refusal ]; then
    CALIB_INFO="Compliant-refusal/ (outcome-labelled)"
fi

echo "============================================"
echo "  Cross-Axis Capping -- Llama-3.3-70B"
echo "  Preset:               ${PRESET}"
echo "  Compliance threshold: ${THRESHOLD}"
echo "  Cross-detect method:  ${CROSS_DETECT}"
echo "  Axis method:          ${AXIS_METHOD}"
echo "  Compliance layers:    ${COMPLIANCE_LAYERS:-paper default (L56-L71)}"
if [ -n "$COMPLIANCE_LAYERS" ]; then
    echo "  Mode 2 (assistant-cap): runs on paper layers (L56-L71); cross-cap (Mode 3) on the override range"
fi
echo "  Calibration:          ${CALIB_INFO}"
echo "  Reclassify:           ${RECLASSIFY}"
echo "  Anthropic key:        ${KEY_INFO}"
echo "  Output:               ${OUTPUT_DIR}"
echo "============================================"
echo ""

EXTRA_ARGS=()
if [ -n "$COMPLIANCE_LAYERS" ]; then
    EXTRA_ARGS+=(--compliance-layers "$COMPLIANCE_LAYERS")
fi

python run_crosscap.py \
    --preset "$PRESET" \
    --model "$MODEL" \
    --compliance-threshold "$THRESHOLD" \
    --cross-detect-method "$CROSS_DETECT" \
    --axis-method "$AXIS_METHOD" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

if [ "$RECLASSIFY" = "yes" ]; then
    echo ""
    echo "── Reclassify (HarmBench for jailbreaks, Claude for benigns) ───────"
    python reclassify_refusals.py \
        --input-dir "$OUTPUT_DIR" \
        --backend anthropic
fi
