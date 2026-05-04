"""
JBB-only compliance-side variant of steer_layer_sweep_mixed.py.

Where the mixed sweep pushes TOWARD refusal (positive target) across a
mixed prompt set, this script does the mirror: steers along the negative
end of the compliance axis on 15 JailbreakBench bare harmful goals.

    idx 0..14: JBB-Behaviors bare harmful goals

Question this answers: on prompts the model would normally refuse, does
steering toward the compliance pole of the axis actually flip the output
to compliance? This validates that the learned axis is causal in both
directions -- refusal and compliance -- not just defensively useful.

Reuses the axis build + steering hook + stats plumbing from
steer_layer_sweep.py; only the prompt loader, default targets, and
output layout change.
"""
import argparse
import csv
import logging
from pathlib import Path

import torch

import run_crosscap
from crosscap_experiment import SteeringExperiment
from steer_probe import (
    decode_new_tokens,
    generate_baseline,
    load_prompts_file,
)
from steer_layer_sweep import (
    build_axes_over,
    generate_steered_on_range,
    parse_layer_ranges,
    parse_targets,
    union_layers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def jbb_prompts(n: int = 15) -> list[str]:
    """Return the first `n` bare harmful goals from JailbreakBench."""
    prompts = list(run_crosscap.load_jbb_behaviors(n_prompts=n))
    if len(prompts) < n:
        logger.warning("JBB returned %d of %d requested prompts", len(prompts), n)
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="JBB-only compliance-side layer sweep (negative targets).",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--axes-path",
                     help="Load existing axes. Must cover every layer in the "
                          "requested union range.")
    src.add_argument("--build", action="store_true",
                     help="Build fresh axes over the union of layer ranges.")

    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--axis-method", default="pca", choices=["pca", "mean_diff"])
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--n-compliance", type=int, default=50,
                        help="Prompts per pool for axis construction.")
    parser.add_argument(
        "--layer-ranges", default="40-70", type=parse_layer_ranges,
        help="Inclusive layer ranges, e.g. '40-70'.",
    )
    parser.add_argument(
        "--targets", default="-4,-8", type=parse_targets,
        help="Comma-separated target projection values. Negative values "
             "push toward the compliance end of the axis.",
    )
    parser.add_argument("--n-prompts", type=int, default=15,
                        help="Number of JBB bare harmful goals to use.")
    parser.add_argument("--prompts-file", default=None,
                        help="Override JBB prompts with one-per-line file. "
                             "Source tag will be 'file'.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="steer_layer_sweep_jbb_compliance_results")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument(
        "--scope", default="all",
        choices=["all", "prefill_only", "first_token_only", "cursor_plus_first"],
        help="Steering scope. 'all' clamps every forward pass; "
             "'prefill_only' clamps only the post-instruction cursor (step 0); "
             "'first_token_only' clamps only the first decoded token (step 1); "
             "'cursor_plus_first' clamps both step 0 and step 1.",
    )
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
    logger.info("Targets (compliance side = negative): %s", args.targets)

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
        source_tags = ["file"] * len(prompts)
    else:
        logger.info("Loading %d JBB bare harmful goals...", args.n_prompts)
        prompts = jbb_prompts(n=args.n_prompts)
        source_tags = ["jbb_vanilla"] * len(prompts)
        logger.info("  idx 0-%d: JBB-Behaviors bare harmful goals", len(prompts) - 1)

    n_runs = len(prompts) * len(args.layer_ranges) * len(args.targets)
    logger.info(
        "%d prompts x %d layer ranges x %d targets = %d generations",
        len(prompts), len(args.layer_ranges), len(args.targets), n_runs,
    )

    results_path = output_dir / "steer_layer_sweep_jbb_compliance.csv"
    trace_path = output_dir / "steer_layer_sweep_jbb_compliance_traces.pt"
    all_traces = []

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_idx", "source", "layer_range", "layer_start", "layer_end",
            "n_layers", "target", "scope", "prompt_text", "steered_text",
            "L_first_pre_mean", "L_first_post_mean", "L_first_push_mean",
            "L_last_pre_mean", "L_last_post_mean", "L_last_push_mean",
        ])

        for pi, text in enumerate(prompts):
            src_tag = source_tags[pi]
            input_ids = exp.tokenize(text)
            prompt_len = input_ids.shape[1]

            if args.include_baseline:
                seqs = generate_baseline(exp, input_ids, max_new_tokens=args.max_new_tokens)
                baseline = decode_new_tokens(exp.tokenizer, seqs, prompt_len)
                writer.writerow([
                    pi, src_tag, "baseline", "", "", "", "baseline", "", text, baseline,
                    "", "", "", "", "", "",
                ])
                f.flush()
                logger.info("[%d %s baseline] %s", pi, src_tag,
                            baseline[:80].replace("\n", " "))

            for (lo, hi) in args.layer_ranges:
                layer_range = list(range(lo, hi + 1))
                range_label = f"L{lo}-L{hi}"

                for target in args.targets:
                    seqs, per_layer_trace = generate_steered_on_range(
                        exp, input_ids, layer_range, per_layer_axes,
                        target=target, max_new_tokens=args.max_new_tokens,
                        scope=args.scope,
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
                        pi, src_tag, range_label, lo, hi, len(layer_range),
                        f"{target:.2f}", args.scope, text, out,
                        f"{f_pre:.3f}", f"{f_post:.3f}", f"{f_push:.3f}",
                        f"{l_pre:.3f}", f"{l_post:.3f}", f"{l_push:.3f}",
                    ])
                    f.flush()
                    all_traces.append({
                        "prompt_idx": pi,
                        "source": src_tag,
                        "layer_range": (lo, hi),
                        "target": target,
                        "scope": args.scope,
                        "prompt_len": prompt_len,
                        "per_layer_trace": per_layer_trace,
                    })
                    logger.info(
                        "[%d %s %s target=%+.1f] L%d pre=%+.2f post=%+.2f | %s",
                        pi, src_tag, range_label, target, layer_range[0],
                        f_pre, f_post, out[:80].replace("\n", " "),
                    )

    torch.save(all_traces, trace_path)
    logger.info("Wrote %s and %s", results_path, trace_path)


if __name__ == "__main__":
    main()
