# Cross-Axis Capping: An Exploratory Study on Separating Jailbreak Detection from Correction in Activation Space

Exploratory project testing whether activation capping improves when detection and correction use separate directions in activation space.

## Motivation

Lu et al. (2026) show that capping activations along a single "Assistant Axis" reduces harmful responses by ~60% without degrading capabilities. But they note the axis is incomplete: "there are likely other subtleties left uncaptured." The axis that best detects jailbreaks need not be the best axis to correct along.

Zhao et al. (2025) provide the mechanistic basis: LLMs encode harmfulness and refusal as separate concepts on distinct directional axes. Steering harmfulness changes the model's risk assessment; steering refusal changes its output behaviour. These are different computations.

This motivates a simple modification: **detect on the assistant axis, correct on a separately-computed compliance axis**. The assistant axis gates when to intervene; the compliance axis controls what the intervention does.

AdaSteer (Liu et al., 2025) uses two directions (Rejection + Harmfulness) but applies both as continuous additive steering to every input. Our approach is threshold-gated: intervene only when a jailbreak is detected, leave everything else untouched.

## Setup

- **Model**: Qwen3-32B, 64 layers, bfloat16
- **Assistant axis**: Pre-computed from Lu et al., via [lu-christina/assistant-axis-vectors](https://huggingface.co/lu-christina/assistant-axis-vectors)
- **Capping layers**: L46--L53 (8 layers, 72--84% depth)
- **Jailbreak prompts**: 100 from JailbreakBench
- **Benign prompts**: 50 calibration questions
- **Compliance axis data**: 50 refusing (JBB bare goals) + 50 compliant (WildJailbreak successful jailbreaks)
- **Detection threshold**: midpoint of benign/jailbreak projection distributions
- **Correction threshold**: mean jailbreak projection
- **Generation**: greedy, max 256 tokens

## Method

**Single-axis capping** (baseline): `h <- h - v * min(<h, v> - tau, 0)`. Detect and correct on the same assistant axis.

**Cross-axis capping**: two-stage gated intervention at each capping layer:

```
if <h, v_detect> < tau_detect:          # assistant axis gates
    if <h, v_correct> < tau_correct:    # compliance axis corrects
        h <- h + (tau_correct - <h, v_correct>) * v_correct
```

Detection and correction are factored apart. The gate can be strict without making the correction aggressive, or vice versa.

**Compliance axis variants tested**:
- **PCA**: PC1 of pooled refusing + compliant activations at L53, sign-corrected to point from compliant toward refusing.
- **Mean-difference**: normalised `(mean_refusing - mean_compliant)`. Simpler, noisier.

## Results

All results below use the PCA compliance axis, judged by Claude Sonnet 4.6. Numbers exclude judge errors (Sonnet refused to classify 12--13 of the most harmful jailbreak prompts). Rows where capping did not fire are excluded from benign stats.

### Jailbreak prompts (n=100, 87--88 classified)

| Metric | Assistant-axis | Cross-axis (PCA) |
|---|---|---|
| Refusal | 32.2% (28/87) | **62.5%** (55/88) |
| Partial refusal | 0% | 3.4% (3/88) |
| Compliance | 67.8% (59/87) | 33.0% (29/88) |
| Degraded | 0% | 1.1% (1/88) |

Cross-axis roughly doubles the refusal rate and halves compliance. Head-to-head on the 86 prompts where both methods received a valid label: 29 flipped from compliance to refusal under cross-axis capping, 0 went the other direction.

### Benign prompts (n=50, 34 where correction fired)

| Metric | Assistant-axis | Cross-axis (PCA) |
|---|---|---|
| Unchanged | 91.2% (31/34) | 91.2% (31/34) |
| False refusal | 0% | 0% |
| Degraded | 8.8% (3/34) | 8.8% (3/34) |

Identical capability preservation. Cross-axis Pareto dominates single-axis: strictly better jailbreak refusal with no benign cost.

## Experiment Log

**Exp 1 -- Baseline replication.** Reproduced Lu et al. assistant-axis capping on Qwen3-32B with their published axis vectors and layer range.

**Exp 2 -- PCA compliance axis.** Computed PC1 from 50 refusing + 50 compliant activations. Cosine similarity with assistant axis was moderate, confirming related but distinct directions. Cross-axis capping showed clear improvement over baseline.

**Exp 3 -- Mean-difference axis.** Tested the raw centroid difference as a simpler alternative. Underperformed PCA on both jailbreak refusal and benign preservation.

**Exp 4 -- Classifier iteration.** Ran three LLM judges. Iterated the Sonnet prompt to require quality/meaning checks before labelling "degraded" -- this dropped false degradation from ~30-48% to near zero, revealing that most "degraded" outputs were simply shorter, not broken.

**Design decisions**: Our midpoint threshold is more aggressive than Lu et al.'s 25th-percentile approach. Single reference layer (L53) for axis construction is a simplification -- the compliance direction may rotate across layers. WildJailbreak prompts used for both axis construction and threshold calibration introduces data leakage.

## Failure Modes

**A -- Benign degradation.** The central tension. Capping fires on 68% of benign prompts (34/50). Of those, 8.8% (3/34) show degraded output. The correction pushes along a direction unrelated to what the model was doing, damaging coherence. The problem is structural: threshold-gated methods will always have false positives.

**B -- Residual compliance.** 33% of jailbreak prompts (29/88 classified) still produce compliant outputs under cross-axis capping. Two sub-patterns: (1) the jailbreak keeps the detection projection above threshold, so the gate never fires; (2) the gate fires but the correction is undone by subsequent layers. The second pattern is specific to cross-axis capping -- the model's residual computation partially reverses the perturbation.

**C -- Fictional framing bypasses detection entirely.** This is the most significant failure mode. Jailbreaks that use fictional framing ("write a story where the protagonist explains how to...") reframe harmful content as creative output. The model can generate harmful information while maintaining a high assistant-axis projection -- because it genuinely *is* acting as a helpful assistant. It is fulfilling a creative writing request. It isn't exiting assistant mode; it is using assistant mode to comply with a creatively-framed harmful request. The detection gate sees a high projection, never fires, and the model complies uncorrected. This is a fundamental limitation of any detection scheme based on the assistant axis: fictional framing attacks don't suppress the assistant signal, they exploit it. Cross-axis capping inherits this blindspot because it still relies on the assistant axis for detection. Addressing this likely requires a detection axis that tracks harmfulness of the *output content* rather than the model's *behavioural mode* -- which connects directly to the harmfulness/refusal separation found by Zhao et al. (2025).
## Limitations

- **n=100 jailbreak, n=50 benign** -- not statistically significant. Error bars on 60% with n=100 are ~+/-10pp.
- **Single model** (Qwen3-32B). No evidence of cross-architecture generalisation.
- **Single threshold strategy**. Did not explore the cross-axis framework's natural advantage: independently tuning detection sensitivity and correction strength.
- **50+50 prompts for axis construction** in a 5120-dimensional space. The compliance axis is under-determined.
- **Calibration data overlap** between axis construction and threshold computation.

## Next Steps

The PCA compliance axis learns a direction with real signal -- cross-axis capping doubles the refusal rate (32% to 63%) with no benign capability cost. But the direction is noisy. It captures compliance/refusal mixed with confounds (style, length, topic). This is why the effect is promising but not reliable enough in practice.

The path forward is to **de-noise the learned directions**:

1. **Linear probe**: train a supervised classifier on compliance vs refusal activations. This gives a direction that maximally discriminates rather than maximally varies -- directly optimised for the separation we care about.
2. **Analyse confounds**: project activations onto the probe direction and check what else correlates -- response length, topic, prompt structure. Identify what the direction has learned besides the safety-relevant signal.
3. **De-confound**: subtract out components that correlate with non-safety features. If the probe direction loads on response length, regress that out. Isolate the pure compliance signal.

4. **Fictional-framing axis**: compute a contrast vector between default prompts and hypothetical/fictional scenario prompts ("write a story where..."). This direction would capture the model's shift into fictional compliance mode -- the same mode that failure mode C exploits. Capping along this axis could detect fictional framing attacks that the assistant axis is structurally blind to, potentially serving as either a replacement or supplementary detection gate.

This is the path from "shows promising signal in an exploratory setting" to "works reliably as an intervention."

## References

- Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. arXiv:2601.10387.
- Zhao, J., Huang, J., Wu, Z., Bau, D., & Shi, W. (2025). LLMs Encode Harmfulness and Refusal Separately. arXiv:2507.11878.
- Liu, Q., et al. (2025). AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender. EMNLP 2025. arXiv:2504.09466.
- Arditi, A., Obeso, O., Surnachev, A., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717.
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
