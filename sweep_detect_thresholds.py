"""Sweep absolute values of the assistant-axis detection threshold.

Goal: study the detection capability of the assistant axis in isolation.
Keep the cross-cap correction (compliance axis + compliance tau) exactly
the way the main experiment uses it; vary only the detect-gate tau.

For each tau in the sweep, the cross-cap hook fires whenever
  (h @ v_assist) < tau
at any cap layer. When it fires, correction along the compliance axis is
applied identically to the main run. So:
  - very low tau  -> gate almost never fires; cross-cap text == baseline
  - very high tau -> gate always fires; cross-cap text == compliance-cap

Sweeping tau across this range traces out the model's response curve as
detection sensitivity increases, which is the diagnostic the user wants.

Inputs:
  --warmup PATH       optional. If given and the file exists, load
                      assistant_axes, compliance_axes, compliance_taus,
                      cap_layers from it. If given but missing, compute
                      a fresh warmup and save to that exact path.
                      If omitted, default to
                          warmupaxes/<model-slug>_warmup.pt.
                      No external warmup.pt is required.
  --taus LIST         comma-separated floats, e.g. "-8,-4,0,4,8,12,16".
                      Each value is broadcast to every cap layer; sweep
                      is over a single scalar tau, the same way the user
                      reasons about the gate.

Outputs (rooted at --output-dir):
  tau_<v>/cross_cap_jailbreak.csv
  tau_<v>/cross_cap_benign.csv
  tau_<v>/gen_trace.pt
  baselines.csv         one row per prompt: prompt_idx, prompt_type,
                        prompt_text, baseline_text. Generated once and
                        reused across all tau values.
  summary.csv           one row per (tau, prompt_type): tau, prompt_type,
                        n_prompts, n_fired, fire_rate, mean_fires_per_prompt.
                        This is the detection PR-style table.
  metadata.json

Usage:
  python sweep_detect_thresholds.py \
      --warmup llama_full/warmup.pt \
      --taus -8,-4,0,4,8,12,16 \
      --n-jailbreak 50 --n-benign 50 \
      --output-dir sweep_detect_llama
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from crosscap_experiment import (
    SteeringExperiment,
    generate_baseline,
    generate_cross_capped,
)
from run_crosscap import (
    MODEL_NAME,
    AXIS_PATH,
    CALIBRATION_PROMPTS,
    build_prompts,
    preflight_hf_access,
    start_download_monitor,
    _compute_warmup_state,
    _format_push_trace,
)


WARMUP_DIR = Path("warmupaxes")  # default location for newly-computed warmups


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("sweep_detect")


def _parse_taus(raw: str) -> list[float]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("--taus must contain at least one value")
    return out


def _tau_dirname(tau: float) -> str:
    """Filesystem-safe label. Negative sign kept; '.' replaced so paths
    stay readable on Windows (no funny shell interpretation either)."""
    s = f"{tau:+.3f}".rstrip("0").rstrip(".")
    return f"tau_{s.replace('.', 'p')}"


def _model_already_cached(model_name: str) -> Path | None:
    """Return the local snapshot path if every weight file the model
    needs is already on disk; None otherwise.

    Uses snapshot_download(local_files_only=True) -- the same mechanism
    from_pretrained relies on, so a positive answer here guarantees the
    subsequent load is a pure-disk op. We deliberately ignore the same
    .bin/.pth artifacts as crosscap_experiment so the safetensors-only
    cache produced by run_llama.sh isn't reported incomplete just
    because the alternate-format weights are missing."""
    try:
        path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            ignore_patterns=["*.bin", "*.bin.index.json", "*.pth", "*.pt"],
        )
        return Path(path)
    except (LocalEntryNotFoundError, FileNotFoundError):
        return None
    except Exception:
        # Be conservative: any other failure (e.g. corrupted cache index)
        # falls through to the full download path so the user gets the
        # same diagnostics they would have gotten before.
        return None


def _load_model(model_name: str) -> "SteeringExperiment":
    """Instantiate SteeringExperiment, skipping preflight + download
    monitor when the model is already on disk. Both helpers are noisy
    (network probes + repeated cache-size prints) and only earn their
    keep on first download; once the cache is populated they're pure
    overhead."""
    cached = _model_already_cached(model_name)
    if cached is not None:
        logger.info("Model %s already cached at %s -- skipping download monitor.",
                    model_name, cached)
        # Force from_pretrained off the network entirely so even the
        # internal snapshot_download verification call inside
        # SteeringExperiment.__init__ becomes a no-op disk lookup.
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return SteeringExperiment(model_name, axis_path=AXIS_PATH)
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

    logger.info("Model not fully cached; running preflight + download monitor.")
    preflight_hf_access(model_name)
    _monitor_stop = start_download_monitor(model_name)
    try:
        return SteeringExperiment(model_name, axis_path=AXIS_PATH)
    finally:
        _monitor_stop.set()


def _model_slug(model_name: str) -> str:
    """Filesystem-safe label for cache filenames."""
    return model_name.replace("/", "__").replace(":", "_")


def _default_warmup_path(model_name: str) -> Path:
    return WARMUP_DIR / f"{_model_slug(model_name)}_warmup.pt"


def _build_warmup_cfg(args) -> dict:
    """Mirror the cfg keys _compute_warmup_state expects.

    Defaults match run_crosscap.py's "full" preset. The cross-detect
    method we pass here is irrelevant for the sweep itself (we overwrite
    the resulting cross_detect_taus with the swept values), but the
    function still needs it to populate cross_detect_stats. We pick
    'benign-p5' as a neutral middle option."""
    cfg = {
        "N_COMPLIANCE":            args.n_compliance,
        "N_DETECT_CAL":            min(args.n_detect_cal, len(CALIBRATION_PROMPTS)),
        "AXIS_METHOD":             args.axis_method,
        "ORTHOGONALIZE":           False,
        "CALIBRATION_DIR":         args.calibration_dir,
        "COMPLIANCE_THRESHOLD":    args.compliance_threshold,
        "CROSS_DETECT_METHOD":     "benign-p5",
        "COMPLIANCE_LAYERS_OVERRIDE": None,
    }
    if args.compliance_layers:
        lo, hi = map(int, args.compliance_layers.split("-"))
        if lo > hi:
            raise ValueError(f"--compliance-layers '{args.compliance_layers}': start > end")
        cfg["COMPLIANCE_LAYERS_OVERRIDE"] = (lo, hi)
    return cfg


def _load_or_compute_warmup(exp, args) -> tuple[dict, Path]:
    """If --warmup points at an existing file, load it. Otherwise compute
    a fresh warmup and save it to the requested path (or the default
    warmupaxes/ location). Returns (state, resolved_path)."""
    if args.warmup:
        warmup_path = Path(args.warmup)
    else:
        warmup_path = _default_warmup_path(args.model)

    if warmup_path.exists():
        logger.info("Loading warmup state from %s", warmup_path)
        return torch.load(warmup_path, map_location="cpu", weights_only=False), warmup_path

    logger.info("No warmup at %s -- computing a fresh one (this takes a while)", warmup_path)
    cfg = _build_warmup_cfg(args)
    state = _compute_warmup_state(exp, cfg)
    warmup_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, warmup_path)
    logger.info("Saved fresh warmup to %s", warmup_path)
    return state, warmup_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warmup", type=str, default=None,
                        help="Optional path to warmup.pt. If the file exists, "
                             "it is loaded; if it doesn't, a fresh warmup is "
                             "computed and saved there. If omitted, defaults "
                             "to warmupaxes/<model-slug>_warmup.pt.")
    parser.add_argument("--taus", type=str, required=True,
                        help="Comma-separated detect-tau values, e.g. '-8,-4,0,4,8'")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write per-tau CSVs and summary.csv")
    parser.add_argument("--n-jailbreak", type=int, default=50)
    parser.add_argument("--n-benign", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Override the model loaded for generation. "
                             "Must match the warmup file's model if one exists.")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Reuse baselines.csv from --output-dir if it "
                             "exists; skip the baseline generation pass.")
    # Warmup-construction knobs (only used when no warmup file exists yet).
    parser.add_argument("--axis-method", choices=["pca", "mean_diff"], default="pca",
                        help="Compliance-axis construction method (default: pca). "
                             "Only used when computing a fresh warmup.")
    parser.add_argument("--compliance-threshold", type=str, default="optimal75",
                        help="Compliance-tau method (default: optimal75). "
                             "Only used when computing a fresh warmup.")
    parser.add_argument("--compliance-layers", type=str, default=None,
                        help="Cross-cap layer override 'lo-hi' (e.g. '40-70'). "
                             "Only used when computing a fresh warmup.")
    parser.add_argument("--calibration-dir", type=str, default=None,
                        help="Outcome-labelled calibration directory "
                             "(refusing.csv + compliant.csv). Only used "
                             "when computing a fresh warmup.")
    parser.add_argument("--n-compliance", type=int, default=50,
                        help="Per-side prompts for compliance-axis build "
                             "(only used when computing a fresh warmup).")
    parser.add_argument("--n-detect-cal", type=int, default=50,
                        help="Benign-prompt count for detect-tau calibration "
                             "(only used when computing a fresh warmup).")
    args = parser.parse_args()

    taus = _parse_taus(args.taus)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build prompts: jailbreak (WildJailbreak eval) + benign (AlpacaEval).
    # Use the same loaders run_crosscap uses so the prompt distribution
    # matches the main experiment exactly. Done before model load so
    # dataset-download failures show up fast.
    prompt_cfg = {
        "N_PROMPTS": args.n_jailbreak,
        "N_BENIGN_EVAL": args.n_benign,
    }
    prompts = build_prompts(prompt_cfg)
    n_jb = sum(1 for p in prompts if p["type"] == "jailbreak")
    n_bn = sum(1 for p in prompts if p["type"] == "benign")
    logger.info("Loaded %d prompts: %d jailbreak, %d benign", len(prompts), n_jb, n_bn)

    # 2. Load model. _load_model first checks whether the HF cache
    # already has every weight file -- if so it skips preflight +
    # the download monitor and forces HF_HUB_OFFLINE=1 around the
    # SteeringExperiment instantiation, so a fully-cached model
    # never makes a network call. On a cache miss we fall back to
    # the run_crosscap.py preflight + monitor path.
    print(f"Loading model: {args.model}")
    exp = _load_model(args.model)

    # 3. Resolve warmup state: load existing or compute + persist a fresh
    # one to warmupaxes/.
    state, warmup_path = _load_or_compute_warmup(exp, args)
    assistant_axes = state["assistant_axes"]
    compliance_axes = state["compliance_axes"]
    compliance_taus = state["compliance_taus"]
    cap_layers = list(state["cap_layers"])
    logger.info("Warmup cap layers: L%d-L%d (%d layers)",
                cap_layers[0], cap_layers[-1], len(cap_layers))

    # 4. Baselines: generate once per prompt and reuse across all taus.
    # Saves N_taus-1 redundant forward passes per prompt.
    baselines_path = output_dir / "baselines.csv"
    if args.skip_baselines and baselines_path.exists():
        logger.info("Reusing existing baselines from %s", baselines_path)
        baselines_df = pd.read_csv(baselines_path)
    else:
        logger.info("Generating baselines (%d prompts)...", len(prompts))
        rows = []
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            input_ids = exp.tokenize(prompt["text"])
            prompt_len = input_ids.shape[1]
            try:
                bl_ids = generate_baseline(exp, input_ids, args.max_new_tokens)
                bl_text = exp.tokenizer.decode(bl_ids[0, prompt_len:], skip_special_tokens=True)
            except Exception:
                logger.exception("Baseline failed for prompt %d", prompt["idx"])
                bl_text = "NA"
            rows.append({
                "prompt_idx":   prompt["idx"],
                "prompt_type":  prompt["type"],
                "prompt_text":  prompt["text"],
                "baseline_text": bl_text,
            })
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                logger.info("  baselines %d/%d  (%.1f min)", i + 1, len(prompts), (time.time() - t0) / 60)
        baselines_df = pd.DataFrame(rows)
        baselines_df.to_csv(baselines_path, index=False)
        logger.info("Saved %s", baselines_path)

    # Index baselines by (idx, type) so we can attach them to every per-tau row.
    baseline_lookup = {
        (r["prompt_idx"], r["prompt_type"]): r["baseline_text"]
        for _, r in baselines_df.iterrows()
    }

    # 5. Sweep loop. For each tau: build a constant detect-tau dict, run
    # cross-cap on every prompt, write one CSV per (tau, prompt_type),
    # accumulate a summary row.
    summary_rows = []
    sweep_start = time.time()
    for tau_idx, tau in enumerate(taus):
        tau_dir = output_dir / _tau_dirname(tau)
        tau_dir.mkdir(parents=True, exist_ok=True)
        # Broadcast the scalar to every cap layer.
        detect_taus = {li: float(tau) for li in cap_layers}
        logger.info("─" * 60)
        logger.info("[%d/%d] tau=%+.3f  (dir=%s)", tau_idx + 1, len(taus), tau, tau_dir.name)

        per_prompt_rows = []
        traces = []
        t_tau = time.time()
        for i, prompt in enumerate(prompts):
            input_ids = exp.tokenize(prompt["text"])
            prompt_len = input_ids.shape[1]
            try:
                cross_ids, n_triggered, n_corrected, cross_active, per_layer_events, per_layer_trace = (
                    generate_cross_capped(
                        exp, input_ids, cap_layers,
                        per_layer_detect_axes=assistant_axes,
                        correct_axes=compliance_axes,
                        detect_thresholds=detect_taus,
                        correct_thresholds=compliance_taus,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
                cross_text = exp.tokenizer.decode(cross_ids[0, prompt_len:], skip_special_tokens=True)
            except Exception:
                logger.exception("cross-cap failed for prompt %d at tau=%g", prompt["idx"], tau)
                cross_ids = None
                n_triggered = n_corrected = 0
                cross_active = []
                cross_text = "NA"
                per_layer_events = {}
                per_layer_trace = {}

            per_prompt_rows.append({
                "prompt_idx":   prompt["idx"],
                "prompt_type":  prompt["type"],
                "prompt_text":  prompt["text"],
                "baseline_text": baseline_lookup.get((prompt["idx"], prompt["type"]), ""),
                "tau_detect":   tau,
                "correction_applied": "Yes" if n_corrected > 0 else "No",
                "n_triggered":  n_triggered,
                "n_corrected":  n_corrected,
                "layers":       ",".join(f"L{li}" for li in cross_active),
                "capped_text":  cross_text if n_corrected > 0 else "NA",
                "fires_per_layer": ";".join(
                    f"L{li}={len(events)}" for li, events in per_layer_events.items()
                ),
                "push_trace": _format_push_trace(per_layer_events, cross_ids, prompt_len, exp.tokenizer)
                              if cross_ids is not None else "",
            })
            traces.append({
                "prompt_idx":  prompt["idx"],
                "prompt_type": prompt["type"],
                "prompt_len":  prompt_len,
                "tau_detect":  tau,
                "per_layer_trace": per_layer_trace,
            })
            del cross_ids
            torch.cuda.empty_cache()

        # Save per-tau CSVs split by prompt type.
        df_tau = pd.DataFrame(per_prompt_rows)
        jb = df_tau[df_tau["prompt_type"] == "jailbreak"]
        bn = df_tau[df_tau["prompt_type"] == "benign"]
        jb.to_csv(tau_dir / "cross_cap_jailbreak.csv", index=False)
        bn.to_csv(tau_dir / "cross_cap_benign.csv", index=False)
        torch.save(traces, tau_dir / "gen_trace.pt")

        # Summary entries: fire rate per prompt type at this tau.
        for label, sub in (("jailbreak", jb), ("benign", bn)):
            n = len(sub)
            n_fired = int((sub["correction_applied"] == "Yes").sum())
            mean_fires = float(sub["n_corrected"].mean()) if n > 0 else 0.0
            summary_rows.append({
                "tau_detect":   tau,
                "prompt_type":  label,
                "n_prompts":    n,
                "n_fired":      n_fired,
                "fire_rate":    n_fired / n if n > 0 else 0.0,
                "mean_fires_per_prompt": mean_fires,
            })

        elapsed = (time.time() - t_tau) / 60
        logger.info(
            "tau=%+.3f done in %.1f min:  jb fired %d/%d  bn fired %d/%d",
            tau, elapsed,
            int((jb["correction_applied"] == "Yes").sum()), len(jb),
            int((bn["correction_applied"] == "Yes").sum()), len(bn),
        )

    # 6. Write summary + metadata.
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "model":          args.model,
        "warmup":         str(warmup_path),
        "taus":           taus,
        "cap_layers":     f"L{cap_layers[0]}-L{cap_layers[-1]}",
        "n_cap_layers":   len(cap_layers),
        "n_jailbreak":    n_jb,
        "n_benign":       n_bn,
        "max_new_tokens": args.max_new_tokens,
        "compliance_taus_method": "loaded from warmup",
        "timestamp":      datetime.now().isoformat(),
        "elapsed_minutes": (time.time() - sweep_start) / 60,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Console summary table.
    print("\n" + "=" * 60)
    print(" Detection-tau sweep summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {output_dir / 'metadata.json'}")
    print(f"Per-tau dirs:  {len(taus)} under {output_dir}/")


if __name__ == "__main__":
    main()
