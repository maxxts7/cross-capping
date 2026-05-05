# Cross-Axis Capping — full-run results

Two complete runs are in scope here:

- **Qwen3-32B**, paper layers, axes calibrated on JBB+WildJailbreak. Run dir: `qwen_full/Qwen3-32B_L46-L53_nj250_nb100_tok256_pca_optimal75_2026-04-17T19-16-29/`.
- **Llama-3.3-70B-Instruct**, extended cap range L40-L65, axes calibrated on outcome-labelled `Compliant-refusal/`. Run dir: `llama_full/`.

Both used PCA compliance axes, the WildJailbreak eval split for the 250 jailbreak prompts, and AlpacaEval for the 100 benign prompts. Judges are Claude Sonnet (HarmBench was not used here). Bootstrap CIs are N=10,000 percentile-method draws produced by `bootstrap_results.py`.

## Side-by-side headlines

Refusal rate is over judgeable rows (NaN/no-fire and judge-error rows excluded). Lift is the paired cross-cap minus assistant-cap difference on the shared prompt set.

| Run | Assistant-cap refusal | Cross-cap refusal | Paired lift | P(lift > 0) | Benign unchanged (cross-cap) |
|---|---|---|---|---|---|
| qwen_full   | 32.9% [26.6, 39.3] | 54.5% [47.5, 61.4] | **+21.5 pp** [+15.1, +28.0] | 100.0% | 91.0% [83.6, 97.0] |
| llama_full  | 20.6% [15.2, 26.5] | 48.1% [41.6, 54.7] | **+27.5 pp** [+21.0, +34.3] | 100.0% | 98.0% [95.0, 100.0] |

Both runs show a clean, large defensive lift that doesn't overlap zero. Llama shows the larger absolute lift in percentage points, partly because the assistant-cap baseline on Llama refuses less (20.6% vs Qwen's 32.9%) — there's more room to grow.

## Qwen3-32B

### Configuration

| Field | Value |
|---|---|
| Model | Qwen/Qwen3-32B |
| Cap layers | L46-L53 (8 layers, paper's published range) |
| Prompts | 250 jailbreak + 100 benign |
| Max new tokens | 256 |
| Axis method | PCA |
| Compliance threshold | optimal75 |
| Cross-detect method | benign-p1 |
| Calibration source | default JBB-Behaviors + WildJailbreak train |
| cos(assistant, compliance) | 0.156 |
| Judge | claude-sonnet-4-6 |

### Jailbreak (rates over judgeable rows)

| Mode | Fire rate | Refusal | Partial refusal | Compliance | Degraded |
|---|---|---|---|---|---|
| assistant-cap (n_judged=219/250) | 100% | **32.9%** [26.6, 39.3] | 1.8% [0.4, 3.7] | 65.3% [58.9, 71.7] | 0% |
| cross-cap (n_judged=204/250) | 92.4% | **54.5%** [47.5, 61.4] | 3.4% [1.0, 6.2] | 41.6% [34.8, 48.5] | 0.5% [0, 1.5] |

Paired lift, cross-cap minus assistant-cap, same prompts:
- Refusal: **+21.5 pp** [+15.1, +28.0], P(lift > 0) = 100.0%
- Refusal_any (refusal + partial): **+23.1 pp** [+16.7, +29.7], P(lift > 0) = 100.0%

Raw label counts (all rows, including NaN no-fires and judge-errors):
- assistant_cap_jailbreak (250 rows, 100% fire): refusal 72, partial 4, compliance 143, error 31.
- cross_cap_jailbreak (250 rows, 231 fires): refusal 111, partial 7, compliance 85, degraded 1, error 27, NaN 19.

### Benign (rates over fired rows)

| Mode | Fire rate | Unchanged | False refusal | Degraded |
|---|---|---|---|---|
| assistant-cap (fired n=100) | 100% | 90.0% [84.0, 95.0] | 0% [0, 0] | 10.0% [5.0, 16.0] |
| cross-cap (fired n=67) | 67% | 91.0% [83.6, 97.0] | 0% [0, 0] | 9.0% [3.0, 16.4] |

Both modes preserve output quality on benigns: zero false refusals, ~9-10% degradation among fired rows. Cross-cap's detect gate leaves 33% of benigns completely untouched.

### Per-layer fires (cross-cap jailbreaks, 231/250 prompts fired)

| Layer | Fires | Share |
|---|---|---|
| L46 | 9,696 | 41.0% |
| L47 | 5,325 | 22.5% |
| L48 | 2,323 | 9.8% |
| L49 | 2,552 | 10.8% |
| L50 | 1,184 | 5.0% |
| L51 | 1,729 | 7.3% |
| L52 | 313 | 1.3% |
| L53 | 547 | 2.3% |
| **Total** | **23,669** | |

Cross-cap benigns: 419 fires total across the same layers, 67/100 prompts. Two orders of magnitude fewer fires than on jailbreaks — the decoupled-axis design is doing what it's supposed to do here.

## Llama-3.3-70B-Instruct

### Configuration

| Field | Value |
|---|---|
| Model | meta-llama/Llama-3.3-70B-Instruct |
| Mode 2 (assistant-cap) layers | L56-L71 (paper's range) |
| Mode 3 (cross-cap) layers | L40-L65 (`--compliance-layers` override) |
| Prompts | 250 jailbreak + 100 benign |
| Max new tokens | 256 |
| Axis method | PCA |
| Compliance threshold | optimal |
| Cross-detect method | benign-p1 |
| Calibration source | `Compliant-refusal/` (outcome-labelled refusing.csv + compliant.csv) |
| cos(assistant, compliance) | 0.157 |
| Judge | claude-sonnet-4-6 |

### Jailbreak (rates over judgeable rows)

| Mode | Fire rate | Refusal | Partial refusal | Compliance | Degraded |
|---|---|---|---|---|---|
| assistant-cap (n_judged=204/250) | 100% | **20.6%** [15.2, 26.5] | 1.5% [0.0, 3.4] | 77.9% [72.1, 83.8] | 0% |
| cross-cap (n_judged=214/250) | 100% | **48.1%** [41.6, 54.7] | 1.4% [0.0, 3.3] | 50.5% [43.9, 57.0] | 0% |

Paired lift, cross-cap minus assistant-cap, same prompts:
- Refusal: **+27.5 pp** [+21.0, +34.3], P(lift > 0) = 100.0%
- Refusal_any: **+27.5 pp** [+20.9, +34.1], P(lift > 0) = 100.0%

Raw label counts:
- assistant_cap_jailbreak (250 rows, 100% fire): refusal 42, partial 3, compliance 159, error 46.
- cross_cap_jailbreak (250 rows, 250 fires): refusal 103, partial 3, compliance 108, error 36.

### Benign (rates over fired rows)

| Mode | Fire rate | Unchanged | False refusal | Degraded |
|---|---|---|---|---|
| assistant-cap (fired n=100) | 100% | 99.0% [97.0, 100.0] | 0% [0, 0] | 1.0% [0.0, 3.0] |
| cross-cap (fired n=100) | 100% | 98.0% [95.0, 100.0] | 0% [0, 0] | 2.0% [0.0, 5.0] |

Both modes are essentially indistinguishable on benign utility. Note: cross-cap has 100% fire rate on benigns here (it always fires), unlike Qwen where the detect gate left 33% of benigns alone. That's a consequence of the `--compliance-layers 40-65` override — see Interpretation below.

### Per-layer fires (cross-cap, all 250 jailbreak prompts fired)

| Layer | Jailbreak fires | Benign fires |
|---|---|---|
| L40 | 34,839 | 24,437 |
| L41 | 34,058 | 24,316 |
| L42 | 25,163 | 19,526 |
| L43 | 28,561 | 20,487 |
| L44 | 29,382 | 23,176 |
| L45 | 34,428 | 24,419 |
| L46 | 34,532 | 24,365 |
| L47 | 34,897 | 24,429 |
| L48 | 34,579 | 24,407 |
| L49 | 34,978 | 24,422 |
| L50 | 33,638 | 23,857 |
| L51 | 35,134 | 24,401 |
| L52 | 31,757 | 23,418 |
| L53 | 32,799 | 23,104 |
| L54 | 34,984 | 24,369 |
| L55 | 35,285 | 24,391 |
| L56 | 4,111 | 0 |
| L57 | 4,177 | 0 |
| L58 | 4,089 | 0 |
| L59 | 3,951 | 0 |
| L60 | 5,098 | 4 |
| L61 | 6,882 | 15 |
| L62 | 4,641 | 1 |
| L63 | 3,663 | 2 |
| L64 | 6,232 | 14 |
| L65 | 11,644 | 152 |
| **Total** | **583,502** | **377,712** |

The L40-L55 firing pattern is uniformly heavy on both jailbreak and benign because at those override-extra layers the detect gate was forced to +inf (always-pass) under benign-p1 mode — the assistant axis isn't tuned at L40-L55, so the calibration logic substituted the compliance axis as a placeholder and made the gate permissive. Inside the paper's range (L56-L65), the gate behaves the way the design intends: heavy firing on jailbreaks, near-zero firing on benigns.

## Interpretation

### Both runs

- The decoupled detect/correct axes deliver a real, statistically-clean defensive lift on both models. CIs don't cross zero, and 100.0% of bootstrap draws show positive lift.
- Benign utility is preserved at the headline level. Zero false refusals, single-digit degradation rates, indistinguishable across the two modes.

### Qwen-specific

- The design intent shows up cleanly: cross-cap fires 56× more on jailbreaks than benigns (23,669 vs 419 fires). The detect gate is genuinely selective.
- Fires are front-loaded at L46-L47 (63.5% of all fires).

### Llama-specific

- Cross-cap fires on every benign prompt because the override layers L40-L55 have an always-pass gate by design. The 98% benign-unchanged rate isn't because the gate was selective — it's because the corrections at those layers are gentle enough that they don't break the model's benign reasoning.
- This is a different equilibrium from Qwen: Qwen's defensive gain comes from a discriminating gate; Llama's comes from gentle uniform clamping at extras plus a real two-gate cross-cap inside the paper's range.
- Inside the paper's range L56-L65, fire counts are an order of magnitude lower on benign than on jailbreak — the design works as intended where the assistant axis is genuinely tuned.
- For sweeps studying the assistant axis's intrinsic detection capability, run with `--compliance-layers ""` so the cross-cap range = paper's L56-L71 only. The L40-L55 layers are a different mechanism (uniform clamp on the compliance axis) that conflates with detection studies.

## Bootstrap details

- N = 10,000 resamples, percentile 95% CI (2.5 / 97.5).
- Jailbreak paired lift: resample prompt indices with replacement, then compute each mode's refusal rate over the resampled judgeable rows; take the difference. Per-prompt pairing is preserved across modes via inner-join on `prompt_idx`.
- Rates are over judgeable rows only; NaN (no-fire) and `error` (judge abstain) are excluded.
- Reproduce: `python bootstrap_results.py <run-dir>`.
- Raw output for each run lives at `<run-dir>/bootstrap.txt`.

## Artifacts

- `qwen_full/.../RESULTS.md` — original per-run write-up for Qwen, with the same headline numbers plus prose interpretation.
- `qwen_full/.../bootstrap.txt`, `llama_full/bootstrap.txt` — raw bootstrap output per run.
- `bootstrap_results.py` — script that produces the `bootstrap.txt` files.
- Per-run reclassified CSVs sit alongside each run: `assistant_cap_{jailbreak,benign}_reclassified.csv`, `cross_cap_{jailbreak,benign}_reclassified.csv`.
- `metadata.json` in each run dir captures the exact experiment configuration.
