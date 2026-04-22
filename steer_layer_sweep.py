"""
Layer-range sweep on the compliance axis -- companion to steer_probe.py.

Where steer_probe.py sweeps TARGET values across a fixed cap-layer range
(L56-L71 for Llama), this script does the opposite: fixes target(s) and
sweeps which LAYERS receive the unconditional steering hook.

Question this answers: is the refusal basin a property of the late-network
band specifically (as the paper's L56-L71 suggests), or can it also be
engaged from a wider/earlier band? And does that interact with push
strength -- does a wide band at target=8 match a narrow band at target=16?

Default sweep (4 runs): cross product of
    LAYER RANGES = [(20, 71), (40, 70)]
    TARGETS      = [8, 16]

Axes are built once over the union range (20-71 inclusive) so each run
reuses the same per-layer compliance directions; only the subset of
layers hooked and the injected target value change.

Output: one merged CSV with a layer_range + target column, plus a single
traces .pt file. No judge pass -- eyeball the texts.
"""
import argparse
import csv
import logging
from pathlib import Path
from contextlib import ExitStack

import torch

from crosscap_experiment import (
    SteeringExperiment,
    compute_pca_compliance_axis,
    compute_mean_diff_compliance_axis,
)
import run_crosscap

from steer_probe import (
    _SteerToTargetHook,
    generate_baseline,
    decode_new_tokens,
    default_harmful_prompts,
    load_prompts_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default sweep -- 4 runs = {(20,71), (40,70)} x {8, 16}.
DEFAULT_LAYER_RANGES = [(20, 71), (40, 70)]
DEFAULT_TARGETS = [8.0, 16.0]


def parse_layer_ranges(s: str) -> list[tuple[int, int]]:
    """Parse '20-71,40-70' -> [(20, 71), (40, 70)]. Inclusive on both ends."""
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split("-")
        lo, hi = int(a), int(b)
        if lo > hi:
            raise ValueError(f"Layer range {chunk}: start {lo} > end {hi}")
        out.append((lo, hi))
    return out


def parse_targets(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def union_layers(ranges: list[tuple[int, int]]) -> list[int]:
    """Union of all (lo, hi) inclusive ranges, sorted."""
    covered: set[int] = set()
    for lo, hi in ranges:
        covered.update(range(lo, hi + 1))
    return sorted(covered)


def build_axes_over(exp: SteeringExperiment, args, cap_layers: list[int]) -> dict:
    """Build per-layer compliance axes over the full union `cap_layers`.
    Thresholds/detection taus are not computed -- the sweep doesn't use them.
    """
    calib_dir = args.calibration_dir
    if not calib_dir and "llama" in args.model.lower():
        default = run_crosscap.DEFAULT_LLAMA_CALIBRATION_DIR
        if not Path(default).exists():
            raise FileNotFoundError(
                f"Llama needs outcome-labelled calibration CSVs at {default}/. "
                f"Run build_calibration.sh or pass --calibration-dir."
            )
        calib_dir = default

    if calib_dir:
        logger.info("Calibration source: %s", calib_dir)
        refusing, compliant = run_crosscap._load_outcome_calibration(
            calib_dir, args.n_compliance,
        )
    else:
        logger.info("Calibration source: JBB-Behaviors + WildJailbreak train")
        refusing = run_crosscap.load_jbb_behaviors(n_prompts=args.n_compliance)
        compliant = run_crosscap.load_wildjailbreak_train(n_prompts=args.n_compliance)

    if args.axis_method == "pca":
        axes, stats, _, _ = compute_pca_compliance_axis(
            exp, refusing, compliant, cap_layers, axis_name="pca_compliance",
        )
    else:
        axes, stats, _, _ = compute_mean_diff_compliance_axis(
            exp, refusing, compliant, cap_layers, axis_name="mean_diff_compliance",
        )

    return {
        "cap_layers": cap_layers,
        "compliance_axes": axes,
        "compliance_stats": stats,
        "axis_method": args.axis_method,
        "calibration_dir": calib_dir,
    }


def generate_steered_on_range(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    layer_range: list[int],
    per_layer_axes: dict[int, torch.Tensor],
    target: float,
    max_new_tokens: int,
) -> tuple[torch.Tensor, dict[int, list[dict]]]:
    """Apply _SteerToTargetHook at every layer in `layer_range` with the same
    target. Uses each layer's own compliance axis from per_layer_axes.
    """
    hooks = [
        _SteerToTargetHook(exp.layers[li], li, per_layer_axes[li], target)
        for li in layer_range
    ]
    with ExitStack() as stack:
        for h in hooks:
            stack.enter_context(h)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            sequences = exp.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    per_layer_trace = {li: h.trace for li, h in zip(layer_range, hooks)}
    return sequences, per_layer_trace


def main():
    parser = argparse.ArgumentParser(
        description="Layer-range sweep on the compliance axis (fixed targets).",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--axes-path",
                     help="Load existing axes (warmup.pt / axes.pt). Must cover "
                          "every layer in the requested union range.")
    src.add_argument("--build", action="store_true",
                     help="Build fresh axes over the union of layer ranges and "
                          "save to <output>/axes.pt.")

    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--axis-method", default="pca", choices=["pca", "mean_diff"])
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--n-compliance", type=int, default=50,
                        help="Prompts per pool for axis construction.")
    parser.add_argument(
        "--layer-ranges", default="20-71,40-70", type=parse_layer_ranges,
        help="Comma-separated inclusive layer ranges, e.g. '20-71,40-70'.",
    )
    parser.add_argument(
        "--targets", default="8,16", type=parse_targets,
        help="Comma-separated target projection values.",
    )
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="steer_layer_sweep_results")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Also generate an uncapped baseline per prompt.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", args.model)
    exp = SteeringExperiment(args.model)

    union_lrs = union_layers(args.layer_ranges)
    logger.info(
        "Layer ranges: %s  ->  union covers %d layers (L%d..L%d)",
        ["L%d-L%d" % r for r in args.layer_ranges],
        len(union_lrs), union_lrs[0], union_lrs[-1],
    )
    logger.info("Targets: %s", args.targets)

    if args.build:
        logger.info("Building %s compliance axes over union range", args.axis_method)
        state = build_axes_over(exp, args, union_lrs)
        axes_save = output_dir / "axes.pt"
        torch.save(state, axes_save)
        logger.info("Saved axes -> %s", axes_save)
    else:
        logger.info("Loading axes: %s", args.axes_path)
        state = torch.load(args.axes_path, map_location="cpu", weights_only=False)

    per_layer_axes = state["compliance_axes"]
    missing = [li for li in union_lrs if li not in per_layer_axes]
    if missing:
        raise ValueError(
            f"Axes file is missing layers needed by the sweep: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
            f"Rebuild with --build and --layer-ranges covering those layers."
        )

    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)
    else:
        logger.info("Loading default harmful prompts: 5 JBB + 5 WildJailbreak...")
        prompts = default_harmful_prompts(n_per_source=5)
        logger.info("  idx 0-4: JBB bare goals   idx 5-9: WildJailbreak eval wrappers")

    n_runs = len(prompts) * len(args.layer_ranges) * len(args.targets)
    logger.info(
        "%d prompts x %d layer ranges x %d targets = %d generations",
        len(prompts), len(args.layer_ranges), len(args.targets), n_runs,
    )

    results_path = output_dir / "steer_layer_sweep.csv"
    trace_path = output_dir / "steer_layer_sweep_traces.pt"
    all_traces = []

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_idx", "layer_range", "layer_start", "layer_end",
            "n_layers", "target", "prompt_text", "steered_text",
            "L_first_pre_mean", "L_first_post_mean", "L_first_push_mean",
            "L_last_pre_mean", "L_last_post_mean", "L_last_push_mean",
        ])

        for pi, text in enumerate(prompts):
            input_ids = exp.tokenize(text)
            prompt_len = input_ids.shape[1]

            if args.include_baseline:
                seqs = generate_baseline(exp, input_ids, max_new_tokens=args.max_new_tokens)
                baseline = decode_new_tokens(exp.tokenizer, seqs, prompt_len)
                writer.writerow([
                    pi, "baseline", "", "", "", "baseline", text, baseline,
                    "", "", "", "", "", "",
                ])
                f.flush()
                logger.info("[%d baseline] %s", pi, baseline[:80].replace("\n", " "))

            for (lo, hi) in args.layer_ranges:
                layer_range = list(range(lo, hi + 1))
                range_label = f"L{lo}-L{hi}"

                for target in args.targets:
                    seqs, per_layer_trace = generate_steered_on_range(
                        exp, input_ids, layer_range, per_layer_axes,
                        target=target, max_new_tokens=args.max_new_tokens,
                    )
                    out = decode_new_tokens(exp.tokenizer, seqs, prompt_len)

                    def layer_stats(li):
                        t = per_layer_trace[li]
                        if not t:
                            return 0.0, 0.0, 0.0
                        pre = sum(x["pre_proj"] for x in t) / len(t)
                        post = sum(x["post_proj"] for x in t) / len(t)
                        push = sum(x["push_applied"] for x in t) / len(t)
                        return pre, post, push

                    f_pre, f_post, f_push = layer_stats(layer_range[0])
                    l_pre, l_post, l_push = layer_stats(layer_range[-1])

                    writer.writerow([
                        pi, range_label, lo, hi, len(layer_range),
                        f"{target:.2f}", text, out,
                        f"{f_pre:.3f}", f"{f_post:.3f}", f"{f_push:.3f}",
                        f"{l_pre:.3f}", f"{l_post:.3f}", f"{l_push:.3f}",
                    ])
                    f.flush()
                    all_traces.append({
                        "prompt_idx": pi,
                        "layer_range": (lo, hi),
                        "target": target,
                        "prompt_len": prompt_len,
                        "per_layer_trace": per_layer_trace,
                    })
                    logger.info(
                        "[%d %s target=%+.1f] L%d pre=%+.2f post=%+.2f | %s",
                        pi, range_label, target, layer_range[0],
                        f_pre, f_post, out[:80].replace("\n", " "),
                    )

    torch.save(all_traces, trace_path)
    logger.info("Wrote %s and %s", results_path, trace_path)


if __name__ == "__main__":
    main()
