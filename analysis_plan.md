# Cross-cap Diagnostic — Analysis Plan

This document is about what to look at first once the diagnostic data lands. Priority ordering is deliberate: later steps only make sense if earlier ones check out, so working through them in sequence avoids building interpretations on top of foundational issues you haven't yet ruled out.

The mental model: before we can interpret anything about *how* the cap mechanism is performing, we need to know whether the pieces the mechanism depends on are themselves well-behaved. If the compliance axis isn't really capturing refusal, nothing downstream matters. If the calibration distribution is bimodal, the threshold is the wrong mechanism. If magnitude is confounding direction, every projection-based measurement is suspect. Foundation first, dynamics second, causal effect last.

## Step 1 — Is the compliance axis even measuring refusal?

Open `warmup.pt` and look at `per_layer_cos_compliance_assistant`. This is the cosine between the unsupervised PCA axis (what the cap pushes along) and the paper's supervised assistant axis (known to separate refusal from compliance). If these two are measuring the same thing, they should agree — values near +1 across all cap layers mean the unsupervised construction found the refusal direction. Values near 0 mean PCA found a different direction, and any "cross-cap" interpretation of firing behaviour at that layer is on shaky ground. Negative values would be alarming — it means PC1 points the opposite way from refusal, and our sign-flip logic missed it.

Expected shape of a healthy result: uniform high-positive cosines across all 16 layers. Surprising: a layer or two with cos near 0 or a dip in the middle of the band. Those are candidates for "dead axis" layers that should probably be dropped from the cap before you interpret any of their firing behaviour.

Next look at `adjacent_layer_cos_compliance`. Each entry is the cosine between layer L's compliance axis and layer L+1's. If the refusal direction is genuinely stable across depth, these should all be high-positive. Low or negative adjacent cosines mean the PCA found different directions at different layers — possibly because the refusal representation genuinely rotates with depth, or possibly because the calibration set is under-specifying the axis and noise dominates. Either way, it undermines the "uniform cap across 16 layers" design.

This step takes five minutes. Don't proceed until you've done it.

## Step 2 — Are the calibration distributions well-behaved?

In `compliance_stats[li]` you now have `refusing_projs` and `compliant_projs` — the full per-prompt projection lists on the compliance axis for each calibration prompt, at every layer. Draw histograms of these in a notebook, one plot per layer, refusing and compliant overlaid.

What you're looking for: two clean clusters with modest overlap. The midpoint threshold (`optimal` = alpha 0.5) is designed for that shape. Problems show up when the shape violates the assumption — refusing is bimodal (hard-refuse cluster plus soft-refuse cluster with a gap between them), compliant has a long tail that stretches into refusing territory, either class has outliers that inflate the mean and drag the midpoint around. Any of these means the threshold sweep (alpha 0.2 / 0.5 / 0.9) was exploring a parameter that was the wrong shape for the data.

The same file holds `per_prompt_projections_assistant_refusing` and `per_prompt_projections_assistant_compliant` — same prompts projected onto the supervised assistant axis. If the two axes disagree on which prompts are "refusing-like" and "compliant-like," that's a more nuanced failure of the unsupervised construction than cosine alone reveals. Scatter plot compliance-axis projection against assistant-axis projection per prompt; correlated clouds are good, orthogonal ones are bad.

## Step 3 — Is magnitude confounding direction?

Open `per_prompt_norms_refusing` and `per_prompt_norms_compliant`. Plot the mean L2 norm per layer for each class, overlaid. If refusing activations are systematically larger than compliant, every projection-based threshold fires more often on refusing-direction activations because they're literally bigger, not because the direction is different. This matters because it changes what the cap mechanism is actually doing — if the true signal is magnitude, clamping along a direction is a roundabout way of clamping norm.

Healthy shape: the two classes have similar norm distributions at each layer, ideally overlapping heavily. Unhealthy: systematic offset, or one class with much wider spread. A quick check is to divide each projection by the corresponding activation norm (turns projection into cosine with the axis) and re-plot step 2's histograms. If the separation survives normalisation, direction is real. If it collapses, direction was mostly an artefact of magnitude.

## Step 4 — What do the per-token trajectories look like?

Only once steps 1-3 pass do the per-token dynamics become interpretable. Load `gen_trace.pt`, index by `(prompt_idx, layer_idx, step)`, and look at `correct_proj` over time for individual prompts.

Start with one refusal prompt and one compliance prompt, same layer, same alpha. Plot `correct_proj` vs step for both. The refusal prompt should show the compliance projection either starting low (model was going to refuse anyway — idx 210 style) or dipping under tau mid-generation and staying there. The compliance prompt should show the projection riding just above tau most of the time, maybe dipping under and triggering fires but then rising again.

Then add a third line for a stuck-compliance prompt (one of 213, 214, 217, 219 from the earlier runs). If its trajectory looks qualitatively similar to the successful-compliance prompt but the cap just happens not to convert it, the problem is in cap strength or location — we can hope to move it by retuning. If its trajectory looks qualitatively *different* — e.g. projection stays well above tau, or fires cluster at a specific token range — the problem is more structural and a different lever is needed.

After individual inspection, aggregate across prompts. For each outcome (refusal / compliance / error), plot mean projection trajectory ± one standard deviation. If refusal and compliance trajectories are indistinguishable in the aggregate, outcome is being determined by something other than the signal we're measuring. If they diverge clearly at a specific token range, that's where the cap's effectiveness is concentrated.

## Step 5 — Does the clamp actually stick?

The deepest question, and the one with the most leverage if it comes back surprising. In `gen_trace.pt`, each fire event at layer L has `proj_onto_prev_axes` entries recorded at every downstream cap layer L'. These tell you: immediately after the clamp at L pushed the activation to `tau_L`, what does the projection onto L's axis look like at L+1, L+2, ..., L+N?

By construction, the post-clamp projection at L itself equals `tau_L`. If layer L+1 still reads close to `tau_L` on L's axis, the clamp survived L+1's residual + attention ops — the intervention is causally persistent. If L+1's projection has drifted well above `tau_L`, the model reconstructed the refusing-direction signal via L+1's processing, and the clamp was cosmetic.

`diagnose_axes.py --trace <file>` prints a summary of this (mean proj at dst vs mean src_tau, with the delta). Look at the delta column. A delta close to zero across many src→dst pairs means clamping is effective. A persistently positive delta means the clamps aren't holding, and the next-most-promising direction is the clamp scale / over-shoot experiment — push past tau so that even after partial rollback the activation stays below the intended threshold.

One caveat on step 5: compliance projection onto L's axis at a downstream layer isn't a perfect measure of clamp persistence, because L+1's operations can shift the residual stream along L's axis without necessarily undoing the "refusal" effect (the projection is a proxy, not the outcome). Take the delta as suggestive, not conclusive. If it's large, the clamp-scale experiment is worth running; if it's small, that avenue is probably closed and the next question is about axis construction (axis_method = mean-diff, or orthogonalisation) rather than clamp intensity.

## What to do with the findings

Each step has a decision: does this foundation hold, or does it need fixing before the next step is meaningful? In practice most projects find that one or two of the foundational checks reveal a real issue, and the interesting analysis becomes "here's what we thought was happening, here's what was actually happening." Don't chase every surprise — the goal is to identify the *next experiment*, not to fully characterise the system.

Realistically the first concrete output of this analysis is probably one of: (a) drop layers X and Y from the cap because their axis cos is ~0, re-run; (b) the calibration set is bimodal, split it before computing an axis; (c) clamps don't stick, try over-shoot; (d) axes rotate across depth, try mean-diff or per-layer PCA calibration. Pick whichever the data most clearly points to, run that experiment, and come back.

The tracking file for results: a scratch notebook (not checked in) that loads `warmup.pt` + `gen_trace.pt` and produces the step 1-5 plots inline. Reproducibility is cheap — the diagnostic scripts are deterministic given the same model + calibration set.
