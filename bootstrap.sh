#!/bin/bash
# Bootstrap a fresh machine (e.g. RunPod) for the cross-capping experiments.
# Installs the HuggingFace CLI, logs in with the RunPod-provided token,
# and installs Python dependencies including flash-attn.
#
# Run this from INSIDE the already-cloned cross-capping repo. Clone the
# repo yourself first, cd into it, then:
#     chmod +x bootstrap.sh
#     ./bootstrap.sh
#
# Expects RUNPOD_SECRET_hf_token to be set in the environment (RunPod
# exposes pod secrets with this prefix). If you're not on RunPod, just
# export RUNPOD_SECRET_hf_token=<your-hf-token> before invoking.
#
# Requires torch to be pre-installed in the base environment (the RunPod
# PyTorch image provides it). flash-attn's --no-build-isolation flag
# compiles against whatever torch is already there, so it has to come
# before the main requirements install.

set -e

if [ -z "${RUNPOD_SECRET_hf_token:-}" ]; then
    echo "ERROR: RUNPOD_SECRET_hf_token is not set."
    echo "On RunPod, add it in the pod's Secrets configuration."
    echo "Elsewhere: export RUNPOD_SECRET_hf_token=<your-hf-token>"
    exit 1
fi

echo "── Installing HuggingFace CLI ────────────────────────────────"
curl -LsSf https://hf.co/cli/install.sh | bash
# The installer writes hf to ~/.local/bin; make sure the current shell
# can find it without a re-login.
export PATH="$HOME/.local/bin:$PATH"
hash -r

echo ""
echo "── Installing hf_transfer (Rust-based fast downloader) ──────"
# The default HF downloader is single-threaded per shard. On cloud
# bandwidth-capped networks, Llama-3.3-70B can take >30 min to download
# and often appears hung for long stretches. hf_transfer parallelises
# byte ranges within each shard and is typically 5-10x faster. The
# env var is required -- HF won't use it just because it's installed.
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# Persist into the user's shell rc so future sessions pick it up too.
# Only appends if the line isn't already there (idempotent across re-runs).
if ! grep -qs "HF_HUB_ENABLE_HF_TRANSFER" "$HOME/.bashrc" 2>/dev/null; then
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> "$HOME/.bashrc"
fi

echo ""
echo "── Logging in to HuggingFace ─────────────────────────────────"
hf auth login "$RUNPOD_SECRET_hf_token"

echo ""
echo "── Installing flash-attn (slow; compiles against torch) ─────"
pip install flash-attn --no-build-isolation

echo ""
echo "── Installing project requirements ──────────────────────────"
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Bootstrap complete"
echo "  cwd: $(pwd)"
echo ""
echo "  Next steps:"
echo "    chmod +x run_llama.sh build_calibration.sh"
echo "    ./build_calibration.sh both 200 harmbench   # produces Compliant-refusal/"
echo "    ln -s data/outcome_labeled_both_n200_harmbench Compliant-refusal"
echo "    ./run_llama.sh sanity                        # then the real run"
echo "============================================"
