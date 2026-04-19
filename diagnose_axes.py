"""Load a warmup.pt (with diagnostic fields) and optionally gen_trace.pt,
print compact per-layer summary tables.

This is a thin loader/printer, not an analysis tool. Real pattern-finding
happens in a notebook against the same files. The point of this script is
to make the persisted diagnostic data easy to eyeball from a terminal.

Usage:
    python diagnose_axes.py --warmup <results_dir>/warmup.pt
    python diagnose_axes.py --warmup <results_dir>/warmup.pt \
                            --trace  <results_dir>/gen_trace.pt
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _fmt(x: float, width: int = 8, dp: int = 3) -> str:
    """Fixed-width float format with sign, for aligning columns."""
    return f"{x:+{width}.{dp}f}"


def print_per_layer_axis_table(state: dict[str, Any]) -> None:
    """One row per cap layer. Columns come from the new diagnostic fields
    populated in _compute_warmup_state -- pulled straight, no computation
    other than formatting.
    """
    cap_layers = list(state["cap_layers"])
    stats = state["compliance_stats"]
    cos_ca = state["per_layer_cos_compliance_assistant"]
    taus = state["compliance_taus"]

    # Header
    cols = [
        ("layer",         6),
        ("tau",           8),
        ("mean_r",        8),
        ("std_r",         7),
        ("mean_c",        8),
        ("std_c",         7),
        ("sep",           7),
        ("cos(a,c)",      10),
        ("var_expl%",     10),
    ]
    header = "  ".join(f"{name:>{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))

    for li in cap_layers:
        s = stats[li]
        row = [
            f"L{li:<5d}",
            f"{taus[li]:>8.2f}",
            f"{s['mean_refusing']:>8.2f}",
            f"{s['std_refusing']:>7.2f}",
            f"{s['mean_compliant']:>8.2f}",
            f"{s['std_compliant']:>7.2f}",
            f"{s['separation']:>7.2f}",
            f"{cos_ca[li]:>+10.3f}",
            f"{s.get('var_explained', float('nan')):>10.1f}",
        ]
        print("  ".join(row))


def print_adjacent_layer_cosines(state: dict[str, Any]) -> None:
    cap_layers = list(state["cap_layers"])
    adj = state["adjacent_layer_cos_compliance"]
    print("\nAdjacent-layer compliance-axis cosines:")
    for i in range(len(cap_layers) - 1):
        a = cap_layers[i]
        b = cap_layers[i + 1]
        print(f"  cos(L{a}, L{b}) = {adj[a]:+.3f}")


def print_projection_shape_summary(state: dict[str, Any]) -> None:
    """Quick shape check on the per-prompt calibration projections:
    for each layer, min/median/max of refusing and compliant projection
    distributions on the compliance axis. Raw lists are in
    state["compliance_stats"][li]["refusing_projs" / "compliant_projs"].
    """
    cap_layers = list(state["cap_layers"])
    stats = state["compliance_stats"]

    print("\nCalibration projections on compliance axis (min / median / max):")
    print(f"  {'layer':>6}  {'refusing (min / med / max)':>30}   {'compliant (min / med / max)':>30}")
    for li in cap_layers:
        r = stats[li].get("refusing_projs", [])
        c = stats[li].get("compliant_projs", [])
        def triple(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            return f"{min(vals):7.2f} / {statistics.median(vals):7.2f} / {max(vals):7.2f}"
        print(f"  L{li:<5d}  {triple(r):>30}   {triple(c):>30}")


def print_calibration_norms(state: dict[str, Any]) -> None:
    """Per-layer mean L2 norms on the two calibration sets (signal 6)."""
    cap_layers = list(state["cap_layers"])
    rn = state["per_prompt_norms_refusing"]
    cn = state["per_prompt_norms_compliant"]

    print("\nCalibration activation L2 norms, per layer (mean +/- std):")
    print(f"  {'layer':>6}  {'refusing':>22}   {'compliant':>22}")
    for li in cap_layers:
        r = rn[li]
        c = cn[li]
        rmean, rstd = (statistics.mean(r), statistics.pstdev(r)) if r else (float('nan'), float('nan'))
        cmean, cstd = (statistics.mean(c), statistics.pstdev(c)) if c else (float('nan'), float('nan'))
        print(f"  L{li:<5d}  {rmean:>10.2f} +/- {rstd:>6.2f}   {cmean:>10.2f} +/- {cstd:>6.2f}")


def print_trace_summary(traces: list[dict]) -> None:
    """Aggregate per-layer per-outcome firing + projection summary from
    the per-token trace. Doesn't touch outcome labels (those are in the
    reclassified CSV, not the trace); just shows dynamics by (prompt, layer).
    """
    if not traces:
        print("\n(Trace file empty.)")
        return

    # Collapse: for each (prompt_idx, layer) compute total fires, total steps,
    # fire rate, mean and sd of compliance projection across all steps, mean
    # of detect projection, mean norm.
    print(f"\nPer-token trace summary ({len(traces)} prompts):")
    print(f"  {'prompt':>7}  {'type':>10}  {'layer':>6}  {'steps':>6}  "
          f"{'fires':>6}  {'fire%':>6}  {'mean_cproj':>10}  {'mean_dproj':>10}  {'mean_norm':>10}")

    for rec in traces:
        pid = rec["prompt_idx"]
        ptype = rec["prompt_type"]
        per_layer = rec["per_layer_trace"]
        for li in sorted(per_layer.keys()):
            steps = per_layer[li]
            if not steps:
                continue
            fires = sum(1 for s in steps if s.get("fired"))
            cprojs = [s["correct_proj"] for s in steps]
            dprojs = [s["detect_proj"] for s in steps]
            norms = [s["norm"] for s in steps]
            print(
                f"  {pid:>7}  {ptype:>10}  L{li:<5d}  {len(steps):>6}  "
                f"{fires:>6}  {100*fires/len(steps):>5.1f}%  "
                f"{statistics.mean(cprojs):>10.2f}  "
                f"{statistics.mean(dprojs):>10.2f}  "
                f"{statistics.mean(norms):>10.2f}"
            )


def print_clamp_persistence_summary(traces: list[dict]) -> None:
    """Signal 2: for each fire event at layer L, gather the projection at
    subsequent cap layers onto L's axis. A clamp that 'sticks' lands the
    projection near the fire's src_tau; one that gets rolled back drifts
    well above src_tau.
    """
    if not traces:
        return

    # src_layer -> dst_layer -> list of (proj_at_dst, src_tau)
    bucket: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for rec in traces:
        for dst_li, steps in rec["per_layer_trace"].items():
            for s in steps:
                for entry in s.get("proj_onto_prev_axes", []):
                    src = entry["src_layer"]
                    bucket[src][dst_li].append((entry["proj"], entry["src_tau"]))

    if not bucket:
        print("\n(No fires recorded; no clamp-persistence data.)")
        return

    print("\nClamp persistence (signal 2): projection at downstream layer onto upstream layer's axis")
    print("  a 'stuck' clamp lands the projection near the fire's src_tau; drift means the clamp was rolled back.")
    for src in sorted(bucket.keys()):
        for dst in sorted(bucket[src].keys()):
            values = bucket[src][dst]
            projs = [v[0] for v in values]
            taus = [v[1] for v in values]
            n = len(projs)
            mean_proj = statistics.mean(projs)
            mean_tau = statistics.mean(taus)
            delta = mean_proj - mean_tau
            print(
                f"  src=L{src}  dst=L{dst}  n={n:>5}  "
                f"mean_proj={mean_proj:>7.2f}  src_tau={mean_tau:>7.2f}  "
                f"delta={delta:>+7.2f}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=Path, required=True, help="Path to warmup.pt")
    p.add_argument("--trace",  type=Path, default=None, help="Optional path to gen_trace.pt")
    args = p.parse_args()

    state = torch.load(args.warmup, map_location="cpu", weights_only=False)

    missing = [
        k for k in (
            "compliance_stats",
            "per_layer_cos_compliance_assistant",
            "adjacent_layer_cos_compliance",
            "per_prompt_norms_refusing",
            "per_prompt_norms_compliant",
        ) if k not in state
    ]
    if missing:
        print(f"WARNING: warmup.pt is missing new diagnostic fields: {missing}")
        print("  This file was produced before the diagnostic logging was added.")
        print("  Re-run warmup to regenerate.")
        return

    print(f"Warmup loaded: {args.warmup}")
    print(f"  cap_layers = L{state['cap_layers'][0]}-L{state['cap_layers'][-1]} "
          f"({len(state['cap_layers'])} layers)")
    print()

    print_per_layer_axis_table(state)
    print_adjacent_layer_cosines(state)
    print_projection_shape_summary(state)
    print_calibration_norms(state)

    if args.trace is not None:
        if not args.trace.exists():
            print(f"\nERROR: trace file not found: {args.trace}")
            return
        traces = torch.load(args.trace, map_location="cpu", weights_only=False)
        print_trace_summary(traces)
        print_clamp_persistence_summary(traces)


if __name__ == "__main__":
    main()
