"""Generate uncapped baseline outputs for a batch of potentially-harmful
prompts. Stage one of the outcome-labelled calibration pipeline; the
classify_calibration.py script handles the labelling step afterwards.

Output: <output-dir>/raw_generations.csv with columns
    prompt_idx, source, prompt_text, baseline_text

Usage:
    python generate_calibration.py \\
        --source both \\
        --n-prompts 200 \\
        --output-dir data/outcome_labeled

Prompts come from JBB-Behaviors (jbb), WildJailbreak train (wj), or both.
Generation uses the same SteeringExperiment + generate_baseline machinery
as the main run_crosscap.py, with no capping applied.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from crosscap_experiment import SteeringExperiment, generate_baseline
from run_crosscap import (
    MODEL_NAME,
    AXIS_PATH,
    load_jbb_behaviors,
    load_wildjailbreak_train,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_calib")


def load_prompts(source: str, n_prompts: int) -> list[dict]:
    """Load prompts from JBB-Behaviors, WildJailbreak train, or both combined.

    Returns a list of {prompt_idx, source, prompt_text}. prompt_idx is a
    string keyed by source so splits from different datasets don't collide.
    """
    prompts: list[dict] = []

    if source in ("jbb", "both"):
        per_source = n_prompts if source == "jbb" else n_prompts // 2
        jbb = load_jbb_behaviors(n_prompts=per_source)
        for i, text in enumerate(jbb):
            prompts.append({
                "prompt_idx":  f"jbb_{i}",
                "source":      "jbb",
                "prompt_text": text,
            })

    if source in ("wj", "both"):
        per_source = n_prompts if source == "wj" else n_prompts // 2
        wj = load_wildjailbreak_train(n_prompts=per_source)
        for i, text in enumerate(wj):
            prompts.append({
                "prompt_idx":  f"wj_{i}",
                "source":      "wildjailbreak_train",
                "prompt_text": text,
            })

    logger.info("Loaded %d prompts (source=%s)", len(prompts), source)
    return prompts


def run_generation(
    prompts: list[dict],
    model_name: str,
    axis_path: str,
    max_new_tokens: int,
    output_path: Path,
    resume: bool,
) -> pd.DataFrame:
    """Run uncapped generation for every prompt and save to output_path.

    Writes the CSV incrementally after each prompt so an interrupted run
    can resume without re-generating anything.
    """
    existing: dict[str, str] = {}
    if resume and output_path.exists():
        df_prev = pd.read_csv(output_path)
        if "prompt_idx" in df_prev.columns and "baseline_text" in df_prev.columns:
            existing = dict(zip(df_prev["prompt_idx"].astype(str), df_prev["baseline_text"].fillna("")))
            logger.info("Resuming: %d prompts already generated", len(existing))

    pending = [p for p in prompts if p["prompt_idx"] not in existing]
    logger.info("To generate: %d new prompts (max_new_tokens=%d)", len(pending), max_new_tokens)

    # Carry forward what we already have so the output CSV is always complete
    rows: list[dict] = [
        {**p, "baseline_text": existing[p["prompt_idx"]]}
        for p in prompts if p["prompt_idx"] in existing
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pending:
        logger.info("Nothing to generate; output is already complete")
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return df

    logger.info("Loading model: %s", model_name)
    exp = SteeringExperiment(model_name, axis_path=axis_path)
    logger.info("  Layers: %d, hidden_dim: %d", exp.num_layers, exp.hidden_dim)

    for p in tqdm(pending, desc="generating", unit="prompt"):
        try:
            input_ids = exp.tokenize(p["prompt_text"])
            prompt_len = input_ids.shape[1]
            out_ids = generate_baseline(exp, input_ids, max_new_tokens)
            text = exp.tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)
        except Exception as e:
            logger.exception("FAILED generation for %s: %s", p["prompt_idx"], e)
            text = ""

        rows.append({**p, "baseline_text": text})

        try:
            del input_ids, out_ids
        except Exception:
            pass
        torch.cuda.empty_cache()

        # Incremental save so Ctrl-C doesn't lose progress
        pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d generations to %s", len(df), output_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["jbb", "wj", "both"], default="both",
        help="Which calibration source to pull from.",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=200,
        help="How many prompts total. When source=both, split evenly.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/outcome_labeled"),
        help="Directory to write raw_generations.csv into.",
    )
    parser.add_argument(
        "--model-name", default=MODEL_NAME,
        help=f"HF model ID for generation (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Max generated tokens per prompt.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip prompts already present in the raw generations CSV.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_generations.csv"

    prompts = load_prompts(args.source, args.n_prompts)
    df = run_generation(
        prompts,
        args.model_name,
        AXIS_PATH,
        args.max_new_tokens,
        raw_path,
        resume=args.resume,
    )

    print()
    print(f"Generated {len(df)} rows -> {raw_path}")
    print(f"Next: python classify_calibration.py --input {raw_path} --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
