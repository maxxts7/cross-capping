#!/bin/bash
# Download a HuggingFace model into the local cache, standalone.
#
# Useful for:
#   * Pre-warming the cache before a long experiment run.
#   * Diagnosing download issues in isolation from run_crosscap.py.
#   * Resuming a partial download after a killed run.
#
# Usage:
#   chmod +x download_model.sh
#   ./download_model.sh                                       # default: Llama-3.3-70B
#   ./download_model.sh meta-llama/Llama-3.3-70B-Instruct     # same, explicit
#   ./download_model.sh cais/HarmBench-Llama-2-13b-cls        # any other repo
#
# Expects either HF_TOKEN or RUNPOD_SECRET_hf_token in the environment
# (or equivalent already-logged-in state via `hf auth login`).

set -e

MODEL="${1:-meta-llama/Llama-3.3-70B-Instruct}"

# If the bootstrap installer placed hf under ~/.local/bin, make sure the
# current shell can find it without a re-login.
export PATH="$HOME/.local/bin:$PATH"
hash -r

# Token resolution -- pick up the RunPod-style secret if HF_TOKEN isn't set.
if [ -z "${HF_TOKEN:-}" ] && [ -n "${RUNPOD_SECRET_hf_token:-}" ]; then
    export HF_TOKEN="$RUNPOD_SECRET_hf_token"
fi

# Enable the fast Rust downloader if installed. Without this, sharded 70B
# pulls can drop to single-digit MB/s and look stuck for long stretches.
if python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    TRANSFER_INFO="installed + enabled (fast downloads)"
else
    TRANSFER_INFO="NOT installed -- downloads will be slow. Run: pip install hf_transfer"
fi

# Cache path resolution mirrors what run_crosscap.py's preflight does.
if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
    HUB_DIR="$HUGGINGFACE_HUB_CACHE"
elif [ -n "${HF_HOME:-}" ]; then
    HUB_DIR="$HF_HOME/hub"
else
    HUB_DIR="$HOME/.cache/huggingface/hub"
fi
# "owner/name" -> "models--owner--name" (HF's on-disk naming convention)
SAFE_NAME="models--${MODEL//\//--}"
MODEL_CACHE="$HUB_DIR/$SAFE_NAME"

echo "============================================"
echo "  HF model download"
echo "  Model:        $MODEL"
echo "  Cache path:   $MODEL_CACHE"
echo "  hf_transfer:  $TRANSFER_INFO"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF token:     set (${#HF_TOKEN} chars)"
else
    echo "  HF token:     not set in env -- relying on prior hf auth login"
fi
echo "============================================"
echo ""

if [ -d "$MODEL_CACHE" ]; then
    echo "Existing cache: $(du -sh "$MODEL_CACHE" | awk '{print $1}')"
    echo "(Existing files will be verified by hash; new shards will be fetched.)"
    echo ""
fi

hf download "$MODEL"

echo ""
echo "============================================"
echo "  Download complete"
if [ -d "$MODEL_CACHE" ]; then
    POST_SIZE=$(du -sh "$MODEL_CACHE" | awk '{print $1}')
    echo "  Total on disk: $POST_SIZE"
fi
echo "============================================"
