"""Bootstrap CIs for the cross-capping headline metrics.

Reproduces the analysis in qwen_full/.../bootstrap.txt for any results
directory. Auto-discovers the four reclassified CSVs (with or without
the Windows-style ' (N)' suffix), computes label rates among judgeable
rows, builds 95% percentile CIs from N=10,000 prompt-level resamples,
and emits a paired lift between cross-cap and assistant-cap on the
shared jailbreak set.

Usage:
  python bootstrap_results.py llama_full
  python bootstrap_results.py llama_full --n-resamples 20000
  python bootstrap_results.py llama_full --label-col llm_label_opus

What gets bootstrapped:
  - Per-mode label rates over judgeable rows (NaN = no fire,
    'error' = judge abstained -- both excluded by construction).
  - Paired lift: resample prompt indices once per draw, recompute
    each mode's refusal rate over the resampled judgeable subset,
    take the difference. Pairing across modes is preserved so the
    CI captures within-prompt agreement, not just marginal noise.
  - Benign rates over fired rows (matching the qwen RESULTS.md
    convention).

Outputs:
  <dir>/bootstrap.txt  -- text-format summary, same layout as the
                          existing qwen_full bootstrap.txt.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


JAILBREAK_LABELS = ["refusal", "partial_refusal", "compliance", "degraded"]
BENIGN_LABELS = ["benign_unchanged", "benign_false_refusal", "benign_degraded"]

JUDGE_NONLABELS = {"error", "nan", ""}  # NaN handled separately via .isna()


def _strip_index_suffix(stem: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", stem)


def _find_csv(directory: Path, base: str) -> Path | None:
    """Find <directory>/<base>.csv, allowing a trailing ' (N)' suffix
    (Windows download/copy naming). Returns the most-recently modified
    match if multiple exist."""
    candidates = []
    for p in directory.iterdir():
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if _strip_index_suffix(p.stem) == base:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _normalise_label(s):
    """Treat NaN, 'error', and empty string as 'unjudged'. Keep
    everything else as-is."""
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    if s in JUDGE_NONLABELS:
        return None
    return s


def _judgeable_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Return a Series of normalised labels, NaN where unjudgeable."""
    return df[label_col].map(_normalise_label)


def _rate_ci(labels: np.ndarray, target: str, n: int, rng: np.random.Generator):
    """Bootstrap a single-rate 95% CI by resampling row indices with
    replacement. Rate is computed over rows with a non-None label.
    Returns (point_estimate, lo, hi) all as percentages."""
    judgeable_mask = labels != None  # noqa: E711
    judgeable = labels[judgeable_mask]
    if len(judgeable) == 0:
        return float("nan"), float("nan"), float("nan")
    point = 100.0 * (judgeable == target).mean()
    if n <= 0:
        return point, point, point
    idx_pool = np.flatnonzero(judgeable_mask)
    draws = rng.integers(0, len(idx_pool), size=(n, len(idx_pool)))
    sampled_labels = labels[idx_pool[draws]]
    rates = 100.0 * (sampled_labels == target).mean(axis=1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return point, lo, hi


def _label_block(label_arr: np.ndarray, labels_to_report, n: int, rng):
    """Run _rate_ci for a list of label classes and return [(name, pt, lo, hi), ...]
    plus an additional 'refusal_any' synthesis if both refusal labels are present."""
    rows = []
    for lbl in labels_to_report:
        rows.append((lbl, *_rate_ci(label_arr, lbl, n, rng)))
    if "refusal" in labels_to_report and "partial_refusal" in labels_to_report:
        # refusal_any = refusal + partial_refusal, treated as a synthetic class.
        synth = np.where(np.isin(label_arr, ["refusal", "partial_refusal"]),
                         "refusal_any",
                         label_arr)
        rows.append(("refusal_any", *_rate_ci(synth, "refusal_any", n, rng)))
    return rows


def _paired_lift(
    a_labels: np.ndarray, b_labels: np.ndarray, target: str,
    n: int, rng: np.random.Generator,
):
    """Paired-lift CI on (b_rate - a_rate). Resample prompt indices
    once per draw and recompute each mode's rate on the resampled
    judgeable subset (which may differ between modes if some prompts
    are unjudged in one but not the other -- standard practice)."""
    a_judge = a_labels != None  # noqa: E711
    b_judge = b_labels != None  # noqa: E711
    n_prompts = len(a_labels)
    a_pt = 100.0 * (a_labels[a_judge] == target).mean() if a_judge.any() else float("nan")
    b_pt = 100.0 * (b_labels[b_judge] == target).mean() if b_judge.any() else float("nan")
    if n <= 0:
        return a_pt, b_pt, b_pt - a_pt, b_pt - a_pt, b_pt - a_pt, float("nan")
    a_rates = np.empty(n)
    b_rates = np.empty(n)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        a_sub = a_labels[idx]
        b_sub = b_labels[idx]
        a_jm = a_sub != None  # noqa: E711
        b_jm = b_sub != None  # noqa: E711
        a_rates[i] = 100.0 * (a_sub[a_jm] == target).mean() if a_jm.any() else 0.0
        b_rates[i] = 100.0 * (b_sub[b_jm] == target).mean() if b_jm.any() else 0.0
        diffs[i] = b_rates[i] - a_rates[i]
    a_lo, a_hi = np.percentile(a_rates, [2.5, 97.5])
    b_lo, b_hi = np.percentile(b_rates, [2.5, 97.5])
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    p_pos = float((diffs > 0).mean())
    return (
        (a_pt, a_lo, a_hi),
        (b_pt, b_lo, b_hi),
        (b_pt - a_pt, d_lo, d_hi),
        p_pos,
    )


def _format_block(title: str, rows) -> str:
    lines = [f"== {title} =="]
    for name, pt, lo, hi in rows:
        lines.append(f"  {name:25s} {pt:5.1f}%  [{lo:5.1f}, {hi:5.1f}]")
    return "\n".join(lines)


def _benign_fired_subset(df: pd.DataFrame, label_col: str) -> np.ndarray:
    """Benign rates are reported over rows where correction fired
    (matches the qwen RESULTS convention)."""
    fired = df["correction_applied"].astype(str).str.strip().str.lower() == "yes"
    sub = df.loc[fired, label_col].map(_normalise_label).to_numpy()
    return sub


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=str, help="Results directory (e.g. llama_full)")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--label-col", type=str, default="llm_label",
                   help="CSV column holding the judge label (default: llm_label).")
    p.add_argument("--seed", type=int, default=20260505)
    p.add_argument("--output", type=str, default=None,
                   help="Output path. Default: <directory>/bootstrap.txt")
    args = p.parse_args()

    d = Path(args.directory)
    if not d.is_dir():
        print(f"ERROR: {d} is not a directory", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.output) if args.output else d / "bootstrap.txt"

    rng = np.random.default_rng(args.seed)
    n = args.n_resamples

    csvs = {
        "assistant_cap_jailbreak": _find_csv(d, "assistant_cap_jailbreak_reclassified"),
        "assistant_cap_benign":    _find_csv(d, "assistant_cap_benign_reclassified"),
        "cross_cap_jailbreak":     _find_csv(d, "cross_cap_jailbreak_reclassified"),
        "cross_cap_benign":        _find_csv(d, "cross_cap_benign_reclassified"),
    }

    found = {k: v for k, v in csvs.items() if v is not None}
    missing = [k for k, v in csvs.items() if v is None]
    print(f"Loading from {d}")
    for k, v in found.items():
        print(f"  {k:30s} {v.name}")
    if missing:
        print(f"  (missing: {', '.join(missing)} -- those sections will be skipped)")

    sections = []

    # 1. Per-mode jailbreak rates
    for key in ("assistant_cap_jailbreak", "cross_cap_jailbreak"):
        if key not in found:
            continue
        df = pd.read_csv(found[key])
        if args.label_col not in df.columns:
            print(f"  WARN: {found[key].name} has no '{args.label_col}' column; skipping")
            continue
        labels = _judgeable_labels(df, args.label_col).to_numpy(dtype=object)
        rows = _label_block(labels, JAILBREAK_LABELS, n, rng)
        title = key.replace("_", "-") + " jailbreak (rates among judgeable)"
        # Cleaner title:
        mode = "assistant-cap" if key.startswith("assistant") else "cross-cap"
        sections.append(_format_block(f"{mode} jailbreak (rates among judgeable)", rows))

    # 2. Paired jailbreak lift
    if "assistant_cap_jailbreak" in found and "cross_cap_jailbreak" in found:
        a_df = pd.read_csv(found["assistant_cap_jailbreak"])
        b_df = pd.read_csv(found["cross_cap_jailbreak"])
        if args.label_col in a_df.columns and args.label_col in b_df.columns:
            # Align on prompt_idx so the paired resample is on shared prompts.
            joined = a_df[["prompt_idx", args.label_col]].rename(
                columns={args.label_col: "a_label"}
            ).merge(
                b_df[["prompt_idx", args.label_col]].rename(
                    columns={args.label_col: "b_label"}
                ),
                on="prompt_idx", how="inner",
            )
            n_paired = len(joined)
            a_labels = joined["a_label"].map(_normalise_label).to_numpy(dtype=object)
            b_labels = joined["b_label"].map(_normalise_label).to_numpy(dtype=object)

            for target, target_label in [("refusal", "refusal rate"), ("refusal_any", "refusal_any")]:
                if target == "refusal_any":
                    a_use = np.where(np.isin(a_labels, ["refusal", "partial_refusal"]),
                                     "refusal_any", a_labels)
                    b_use = np.where(np.isin(b_labels, ["refusal", "partial_refusal"]),
                                     "refusal_any", b_labels)
                    pair_target = "refusal_any"
                else:
                    a_use = a_labels
                    b_use = b_labels
                    pair_target = "refusal"
                (a_pt, a_lo, a_hi), (b_pt, b_lo, b_hi), (d_pt, d_lo, d_hi), p_pos = _paired_lift(
                    a_use, b_use, pair_target, n, rng,
                )
                if target == "refusal":
                    block = [
                        f"== paired lift  cross-cap minus assistant-cap  ({target_label}, judgeable both) ==",
                        f"  Paired prompts (intersection): {n_paired}",
                        f"  assistant-cap refusal: {a_pt:.1f}%  [{a_lo:.1f}, {a_hi:.1f}]",
                        f"  cross-cap refusal:     {b_pt:.1f}%  [{b_lo:.1f}, {b_hi:.1f}]",
                        f"  paired lift (c-a):     {d_pt:+.1f} pp  [{d_lo:+.1f}, {d_hi:+.1f}]",
                        f"  lift>0 probability:    {100*p_pos:.1f}%",
                    ]
                    sections.append("\n".join(block))
                else:
                    block = [
                        f"  paired lift ({target_label}, c-a): {d_pt:+.1f} pp  [{d_lo:+.1f}, {d_hi:+.1f}]",
                        f"  lift_any>0 probability: {100*p_pos:.1f}%",
                    ]
                    # Append onto the previous section's text.
                    sections[-1] = sections[-1] + "\n" + "\n".join(block)

    # 3. Benign rates (over fired rows)
    for key in ("assistant_cap_benign", "cross_cap_benign"):
        if key not in found:
            continue
        df = pd.read_csv(found[key])
        if args.label_col not in df.columns:
            print(f"  WARN: {found[key].name} has no '{args.label_col}' column; skipping")
            continue
        sub = _benign_fired_subset(df, args.label_col)
        rows = _label_block(sub, BENIGN_LABELS, n, rng)
        n_fired = int((sub != None).sum())  # noqa: E711
        mode = "assistant-cap" if key.startswith("assistant") else "cross-cap"
        title = f"{mode} benign (over fired rows only, n={n_fired})"
        sections.append(_format_block(title, rows))

    # 4. Header
    n_paired = 0
    if "assistant_cap_jailbreak" in found and "cross_cap_jailbreak" in found:
        a_df = pd.read_csv(found["assistant_cap_jailbreak"])
        b_df = pd.read_csv(found["cross_cap_jailbreak"])
        n_paired = len(set(a_df["prompt_idx"]).intersection(set(b_df["prompt_idx"])))
    header = [
        f"Bootstrap N: {n} resamples, percentile 95% CI",
        f"Label column: {args.label_col}",
        f"Paired jailbreak prompts: {n_paired}",
        "",
    ]

    text = "\n".join(header) + "\n\n".join(sections) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
