"""
Direct-steering probe on the compliance axis.

Question this script answers: if we unconditionally rewrite the hidden state
so its projection onto the compliance axis equals some target value, at every
generation step and every cap layer -- where on the axis does output flip
from compliance to refusal? No detection gate, no threshold, no "fire only
if below" logic. Just:

    h <- h + (target - dot(h, v)) * v

Default prompt set is 10 HARMFUL prompts (5 JailbreakBench bare goals +
5 WildJailbreak eval adversarial wrappers). The question: for prompts the
model COMPLIES with at baseline, does steering to high target produce
refusal? That tells us how far along the axis the "refusal basin" starts,
and whether cross-cap's optimal75 threshold was simply too low to engage it.

Two modes:

1. Build a fresh compliance axis (pca or mean_diff), save it, and run the sweep:

    python steer_probe.py --build --axis-method pca \\
        --output-dir steer_probe_pca

2. Reuse an existing axes.pt or warmup.pt:

    python steer_probe.py --axes-path "llama75/warmup (4).pt" \\
        --output-dir steer_probe_reuse

Outputs CSV + per-step traces + (if building) axes.pt. No judge pass;
eyeball the texts.
"""
import argparse
import csv
import logging
from pathlib import Path
from contextlib import ExitStack

import torch
import torch.nn as nn

from crosscap_experiment import (
    SteeringExperiment,
    compute_pca_compliance_axis,
    compute_mean_diff_compliance_axis,
)
import run_crosscap  # for data loaders + outcome-calibration helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default prompt set: 10 harmful prompts -- 5 bare goals from JailbreakBench
# and 5 adversarial wrappers from WildJailbreak eval. JBB is held out from
# Llama's calibration (which uses Compliant-refusal/, not JBB); the WJ eval
# split is disjoint from the WJ train split used in calibration.
#
# Convention: indices 0-4 are JBB bare goals, 5-9 are WildJailbreak eval
# wrappers. See default_harmful_prompts() below.
#
# Lazy-loaded at runtime -- the HF datasets pull adds ~15s the first time.
DEFAULT_PROMPTS = None  # sentinel; see default_harmful_prompts()


def default_harmful_prompts(n_per_source: int = 5) -> list[str]:
    """Load `n_per_source` bare goals from JailbreakBench and `n_per_source`
    adversarial wrappers from WildJailbreak eval. Returns a flat list with
    JBB first, WJ second so indices map to source.
    """
    bare = list(run_crosscap.load_jbb_behaviors(n_prompts=n_per_source))
    wj_rows = run_crosscap.load_jailbreak_dataset(n_prompts=n_per_source)
    wrapped = [r["goal"] for r in wj_rows]
    if len(bare) < n_per_source:
        logger.warning("JBB returned %d of %d requested bare goals",
                       len(bare), n_per_source)
    if len(wrapped) < n_per_source:
        logger.warning("WJ eval returned %d of %d requested wrappers",
                       len(wrapped), n_per_source)
    return bare + wrapped


# Cap layer range. For Llama-3.3-70B-Instruct the paper uses L56-L71; this
# matches what run_crosscap.py loads via load_original_capping and what all
# the llama75/llama90/llama20 runs used.
LLAMA_CAP_LAYERS = list(range(56, 72))


class _SteerToTargetHook:
    """Unconditional steering hook.

    At every forward pass, computes the last-token projection onto `axis` and
    rewrites h so that dot(h, axis) == target. Applied regardless of where h
    currently sits. No detect gate.

    Records per-step (step, pre_proj, post_proj, push_applied, norm_pre) so
    we can verify the push was actually applied and measure the delta.
    post_proj should equal target within floating-point tolerance.
    """

    def __init__(
        self,
        layer_module: nn.Module,
        layer_idx: int,
        axis: torch.Tensor,
        target: float,
    ) -> None:
        self._layer = layer_module
        self._layer_idx = layer_idx
        self._axis = axis.float()
        self._target = float(target)
        self._axis_device: torch.Tensor | None = None
        self._handle = None
        self._step = 0
        self.trace: list[dict] = []

    def __enter__(self):
        def hook_fn(module, inputs, output):
            if torch.is_tensor(output):
                h = output
            else:
                h = output[0]

            if self._axis_device is None:
                self._axis_device = self._axis.to(h.device)

            axis = self._axis_device
            h_last = h[0, -1, :].float()
            pre_proj = (h_last @ axis).item()
            norm_pre = h_last.norm().item()
            delta = self._target - pre_proj
            h[0, -1, :].add_(delta * axis.to(h.dtype))
            post_proj = (h[0, -1, :].float() @ axis).item()
            self.trace.append({
                "step": self._step,
                "pre_proj": pre_proj,
                "post_proj": post_proj,
                "push_applied": delta,
                "norm_pre": norm_pre,
            })
            self._step += 1

            if torch.is_tensor(output):
                return h
            return (h, *output[1:])

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def generate_steered(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    cap_layers: list[int],
    per_layer_axes: dict[int, torch.Tensor],
    target: float,
    max_new_tokens: int,
) -> tuple[torch.Tensor, dict[int, list[dict]]]:
    hooks = [
        _SteerToTargetHook(exp.layers[li], li, per_layer_axes[li], target)
        for li in cap_layers
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
    per_layer_trace = {li: h.trace for li, h in zip(cap_layers, hooks)}
    return sequences, per_layer_trace


def generate_baseline(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        return exp.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )


def decode_new_tokens(tokenizer, sequences: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(sequences[0, prompt_len:], skip_special_tokens=True)


def parse_targets(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_prompts_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]


def build_axes(exp: SteeringExperiment, args) -> dict:
    """Build the compliance axis from the calibration pool. Skips all threshold
    and detection-tau work -- the probe doesn't use those. Returns a minimal
    dict with cap_layers + compliance_axes + compliance_stats (stats retained
    so we can print a sensible target range).
    """
    # Calibration source: explicit --calibration-dir wins; else Llama uses
    # Compliant-refusal/, other models use JBB + WildJailbreak.
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

    cap_layers = LLAMA_CAP_LAYERS  # Hardcoded: matches paper's Llama range.

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


def main():
    parser = argparse.ArgumentParser(description="Direct-steering probe on the compliance axis.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--axes-path", help="Load existing axes from warmup.pt or axes.pt (fast path).")
    src.add_argument("--build", action="store_true",
                     help="Build a fresh compliance axis and save to <output>/axes.pt.")

    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--axis-method", default="pca", choices=["pca", "mean_diff"])
    parser.add_argument("--calibration-dir", default=None,
                        help="Outcome-labelled refusing.csv/compliant.csv directory. "
                             "Defaults to Compliant-refusal/ for Llama.")
    parser.add_argument("--n-compliance", type=int, default=50,
                        help="Prompts per pool for axis construction.")
    parser.add_argument(
        "--targets", default="-4,0,4,8,12,16,20", type=parse_targets,
        help="Comma-separated target projection values for the sweep.",
    )
    parser.add_argument("--prompts-file", default=None,
                        help="Optional prompts file (one per line). Falls back to DEFAULT_PROMPTS.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="steer_probe_results")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Also generate an uncapped baseline for each prompt.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", args.model)
    exp = SteeringExperiment(args.model)

    if args.build:
        logger.info("Building %s compliance axis", args.axis_method)
        state = build_axes(exp, args)
        axes_save = output_dir / "axes.pt"
        torch.save(state, axes_save)
        logger.info("Saved axes -> %s", axes_save)
    else:
        logger.info("Loading axes: %s", args.axes_path)
        state = torch.load(args.axes_path, map_location="cpu", weights_only=False)

    cap_layers = list(state["cap_layers"])
    per_layer_axes = state["compliance_axes"]
    logger.info("Using %d compliance axes (L%d-L%d)",
                len(per_layer_axes), cap_layers[0], cap_layers[-1])

    if "compliance_stats" in state:
        s = state["compliance_stats"][cap_layers[0]]
        logger.info(
            "L%d pool stats: mean_refusing=%+.2f  mean_compliant=%+.2f  sep=%+.2f",
            cap_layers[0], s["mean_refusing"], s["mean_compliant"], s["separation"],
        )
        logger.info("Suggested target sweep covers range "
                    "[mean_compliant, mean_refusing, beyond].")

    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)
    else:
        logger.info("Loading default harmful prompts: 5 JBB + 5 WildJailbreak...")
        prompts = default_harmful_prompts(n_per_source=5)
        logger.info("  idx 0-4: JBB bare goals   idx 5-9: WildJailbreak eval wrappers")
    logger.info("%d prompts x %d targets = %d generations",
                len(prompts), len(args.targets), len(prompts) * len(args.targets))

    results_path = output_dir / "steer_probe.csv"
    trace_path = output_dir / "steer_probe_traces.pt"
    all_traces = []

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_idx", "target", "prompt_text", "steered_text",
            "L_first_pre_mean", "L_first_post_mean", "L_first_push_mean",
            "L_last_pre_mean", "L_last_post_mean", "L_last_push_mean",
        ])

        for pi, text in enumerate(prompts):
            input_ids = exp.tokenize(text)
            prompt_len = input_ids.shape[1]

            if args.include_baseline:
                seqs = generate_baseline(exp, input_ids, max_new_tokens=args.max_new_tokens)
                baseline = decode_new_tokens(exp.tokenizer, seqs, prompt_len)
                writer.writerow([pi, "baseline", text, baseline,
                                 "", "", "", "", "", ""])
                f.flush()
                logger.info("[%d baseline] %s", pi, baseline[:80].replace("\n", " "))

            for target in args.targets:
                seqs, per_layer_trace = generate_steered(
                    exp, input_ids, cap_layers, per_layer_axes,
                    target=target, max_new_tokens=args.max_new_tokens,
                )
                out = decode_new_tokens(exp.tokenizer, seqs, prompt_len)

                def layer_stats(li):
                    t = per_layer_trace[li]
                    if not t: return 0.0, 0.0, 0.0
                    pre = sum(x["pre_proj"] for x in t) / len(t)
                    post = sum(x["post_proj"] for x in t) / len(t)
                    push = sum(x["push_applied"] for x in t) / len(t)
                    return pre, post, push

                f_pre, f_post, f_push = layer_stats(cap_layers[0])
                l_pre, l_post, l_push = layer_stats(cap_layers[-1])

                writer.writerow([
                    pi, f"{target:.2f}", text, out,
                    f"{f_pre:.3f}", f"{f_post:.3f}", f"{f_push:.3f}",
                    f"{l_pre:.3f}", f"{l_post:.3f}", f"{l_push:.3f}",
                ])
                f.flush()
                all_traces.append({
                    "prompt_idx": pi,
                    "target": target,
                    "prompt_len": prompt_len,
                    "per_layer_trace": per_layer_trace,
                })
                logger.info("[%d target=%+.1f] pre=%+.2f post=%+.2f | %s",
                            pi, target, f_pre, f_post, out[:80].replace("\n", " "))

    torch.save(all_traces, trace_path)
    logger.info("Wrote %s and %s", results_path, trace_path)


if __name__ == "__main__":
    main()
