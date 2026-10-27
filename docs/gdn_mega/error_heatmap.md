# GDN absolute-error heatmaps

Measurements dated 2026-10-26; original **10748eb**,
current **aac5a4d**. Both receive identical inputs and reference
results. [Overview](README.md) · [Latency heatmap](heatmap.md).

**Version scope:** “Current” in the figures means the plotted checkpoint,
not automatically the latest production build. See [accuracy regimes](#accuracy-regimes-where-the-current-kernel-passes-and-fails)
for the fallback-free kernel's broader passing and failing cases.


![Maximum absolute raw-output error](figures/error_heatmap_max_abs.png)

**Current passes 63/63 shapes; original passes
0/63**, checking both output and final state.
Finite failures remain visible on the shared scale; gray marks NaN/Inf.

## Elementwise acceptance

```text
abs(actual[i] - reference[i]) <= 1e-05 + 0.001 * abs(reference[i])
```

This is a mixed absolute/relative bound, **not** a uniform maximum absolute error
of 1e-05. The ratio below divides each element's error by its allowance; a maximum
ratio ≤ 1 means every element satisfies that bound. Finiteness, relative RMSE < 0.005
and final-state checks are additional requirements. No outliers are discarded.

![Worst elementwise tolerance ratio](figures/error_heatmap_tolerance.png)

## H16/Hg4, raw output units

| Tokens | Original max abs | Current max abs | Original worst ratio | Current worst ratio |
|---:|---:|---:|---:|---:|
| 128 | 5.000e-04 | 2.441e-04 | 11.858 | 0.468 |
| 256 | 6.482e-04 | 2.441e-04 | 10.407 | 0.468 |
| 1024 | 7.446e-04 | 2.433e-04 | 13.150 | 0.465 |
| 4096 | 7.587e-04 | 2.439e-04 | 22.756 | 0.478 |
| 8192 | 7.230e-04 | 4.796e-04 | 21.355 | 0.472 |
| 16384 | 8.791e-04 | 3.465e-04 | 26.775 | 0.475 |
| 32768 | 9.809e-04 | 4.634e-04 | 23.493 | 0.476 |

## Measurement contract

CPU FP64 token recurrence; seed 0; D128; H/Hg=8/2, 12/3, 16/4, 24/6, 24/8, 32/8, 48/12, 48/16, 64/16;
nonzero initial state; normalized Gaussian Q/K, gate scale1.0.
Maxima include **every output element and all 3 fresh launches**.
Outputs and final state are filled with NaN before each launch. Caller scratch
initialization is preserved. Input/reference fingerprints match between libraries,
and inputs are checked for mutation. Every repeat and final-state metric is saved.
Statistics are reused only for byte-identical output and state tensors.
Both library workers share one fresh CPU golden per cell within this report run;
input/source identity and full golden-content hashes are verified before reuse.

The current algorithm retains FP32 Gram matrices, stable block solves, output
accumulation and authoritative state. Projections multiply original FP16 inputs
by two FP16 state components; the second captures the first component's rounding
residual. Final FP16 output rounding remains unavoidable. The recorded rounding
floor is a maximum statistic, not proof that every output is correctly rounded.
This ordinary-input grid is not an exhaustive input-range guarantee.

## Accuracy regimes: where the current kernel passes and fails

**This section concerns the fallback-free kernel `677c0ee`**,
library `dfb01e1bc24f`, audited 2026-10-27.

### Current approved acceptance and regular regression coverage

The user accepted these small absolute errors. The current fixed policy is:

```text
abs(error[i]) <= 0.0001 + 0.001 * abs(reference[i])
```

**Atol increased from 1e-05 to 0.0001; rtol remains
0.001. All 160 synthetic cases now pass**, including the
23 former failures. Every case is part of ordinary pytest discovery;
there is no opt-in stress marker. The complete GDN suite passes
**245 tests**.

At unchanged rtol, the largest absolute floor needed by the former failures is
8.293e-05, below the fixed 0.0001 budget.
Finiteness, raw-output and final-state bounds, independent CPU FP64 reference,
all five poisoned-scratch launches, input/guard integrity and relative RMSE
< 0.005 remain enforced. No clipping, outlier removal,
case-specific tolerance or adaptive acceptance is used. Boundary tests inject
one bad element while RMSE still passes, checking that a regression beyond the
approved elementwise budget is rejected.

This is a **user-approved tolerance change, not a kernel accuracy improvement**.
The original tighter-policy measurements and their failures remain intact below
as diagnostics. Passing model captures already satisfied the older, tighter bound.
[Policy calibration and full-suite evidence](data/accepted_precision.json).

### Historical characterization under the earlier tighter tolerance

The figures above retain their own dated checkpoint; their green cells do not
certify the additional regimes below. At the recorded diagnostic tolerance
`atol=1e-05, rtol=0.001`, the synthetic audit has
**137/160 pass, 23 failing cases**, with five launches per case.
The separate real-input replay has **126/126 pass**.
Case counts are coverage, not estimates of how often production inputs fail.
Within beta in [0,1], g <= 0 and normalized-key setups, **157/157 pass current policy; 134/157 pass under the old bound**.
The other 3 exploratory cases remain in the full report;
they are not used to establish safety for this primary workload.

### Parameters and measured regimes

`beta` is write strength. The kernel receives the FP32 **log gate `g`**;
**`alpha = exp(g)`** is per-token retention, so alpha and g are the same control
on different scales. More-negative g means more forgetting. Tiny negative g
means alpha is almost 1 and memory persists. Our primary workload has beta in
[0,1], g <= 0, and L2-normalized keys (with ordinary FP16 rounding).

These are **measured combinations, not universal safe intervals**. There is no
established rule such as “beta below X always passes” or “alpha above Y fails.”
Key alignment, V magnitude, incoming state and the duration of weak decay matter
together. Beta=0 is not automatically safe: the initial state still propagates.

| Regime | Beta, gate / retention | Keys, values/state and length | Evidence and interpretation |
|---|---|---|---|
| **Pass: observed model workload** | Captured beta median 0.353; g median -0.006174, alpha about 0.99385; natural per-head/token variation retained | Actual correlated Qwen keys, max abs(V) 19.41; supplied-state max about 12.60; 346–1024 tokens | All 108 full traces and 18 continuations pass. This is the strongest evidence for the measured workload, not all models or longer contexts. |
| **Pass: ordinary long sequences** | Uniform beta in [0,1), ordinary log-sigmoid Gaussian gates | Independent normalized Gaussian keys, unit-scale V, initial-state standard deviation 0.1; 256–32768 tokens, H16/Hg4 | All 7 current-kernel shapes pass. Does not establish long-context accuracy with correlated, nearly lossless heads. |
| **Pass: controlled correlation at unit scale** | beta=0.5 or 1; g=0, -0.001, -0.1 or -1; alpha=1, 0.9990005, 0.90484 or 0.36788 | Mean key cosine 0–1; V standard deviation 1, initial-state standard deviation 0.1 or 1; mostly 4K, two 32K no-decay cases | **56/56 pass current policy; 56/56 pass under the old bound.** Correlation, beta=1, or alpha=1 alone is not enough to cause failure. |
| **Stress: sustained near-lossless memory at 32K** | Tested g=-1e-8, -1e-7, -1e-6; alpha approximately 0.99999999, 0.9999999, 0.999999; beta=0, beta=1 and uniform[0,1) probes | Includes identical keys and supplied unit-scale state; old-bound failures have max abs(V) 3.90 and max abs(initial) 3.89 | **6/6 pass current policy; 3/6 pass under the old bound**. The previous tighter bound can fail without large amplitudes. Real long-context coverage is still useful. |
| **Stress: high-amplitude combinations** | Old-bound failures use beta=1 and g=0, -1e-6 or -0.001; alpha=1, about 0.999999 or 0.9990005 | V/state scale factors 32 or 128; keys range from uncorrelated to nearly identical; typical old-bound failing max abs(V) about 125–500 | **28/28 pass current policy; 8/28 pass under the old bound**. Magnitude is not a proven failure threshold; keep these cases in regression coverage. |
| **Unestablished: other joint distributions** | Arbitrary persistent gate/beta patterns, other checkpoints; beta>1 or positive g are outside the primary workload above | Real correlated 8K–32K+ traces, other key norms and incoming-state distributions | The small set of exploratory cases and short model captures cannot establish blanket safety. No values are clipped or excluded to obtain a pass. |

### How large were the tighter-policy violations?

**All 160 cases remain finite; no NaN/Inf or overflow was observed.**
All ideal FP16-rounded goldens pass. The failures are additional internal
arithmetic error, not an inadequate final FP16 output range.

**Small absolute errors; they exceed a strict elementwise tolerance. No massive
numerical blow-up was observed.** At the worst normalized witness, reference
is -0.001068957 and the kernel returns
-0.001142502: the absolute difference is
**7.355e-05**. The recorded old allowance is
**1.107e-05**, so the error is
**6.644× the allowance**. That ratio measures a test-threshold overrun;
it does **not** mean the output is 6.64× wrong or establish a large
effect on the model. Keep the strict failure visible, but do not equate its
multiplier with practical severity.

The table shows the element with the worst normalized error in each selected
case, using whichever tensor (output or state) is worse. “Excess” is the absolute
amount beyond its allowance. The violation counts include every element, not a
sample. These witness errors are **not** the global maximum absolute error.

| Case | Violating output / state elements under old bound | Worst tensor | Absolute error at witness | Old allowance | Error / old allowance | Excess over old allowance | Approved allowance | Error / approved allowance |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `critical_tiny_gate_-1e-08_32768` | 1,478 / 0 | output | 2.314e-05 | 1.057e-05 | 2.189× | 1.257e-05 | 1.006e-04 | 0.230× |
| `critical_tiny_gate_-1e-07_32768` | 21 / 0 | output | 1.331e-05 | 1.014e-05 | 1.313× | 3.177e-06 | 1.001e-04 | 0.133× |
| `control_coordinate_beta_zero` | 28 / 0 | output | 1.520e-05 | 1.083e-05 | 1.404× | 4.377e-06 | 1.008e-04 | 0.151× |
| `nearly_correlated_beta1` | 1 / 0 | output | 1.324e-05 | 1.249e-05 | 1.060× | 7.506e-07 | 1.025e-04 | 0.129× |
| `delayed_nearly_correlated` | 2 / 1 | output | 2.006e-05 | 1.334e-05 | 1.503× | 6.714e-06 | 1.033e-04 | 0.194× |
| `critical_nearly_correlated_3023_128.0` | 95 / 26 | output | 7.355e-05 | 1.107e-05 | 6.644× | 6.248e-05 | 1.011e-04 | 0.728× |

For the unit-scale 32K, g=-1e-8 example, 1,478 of
33,554,432 outputs fail (0.00440%),
with zero state violations. Its worst witness error is
2.314e-05, or 2.189× its allowance.
That is a small absolute error, not the large drift of the removed sequential
fallback, but it fails the recorded older contract. We have not measured
its effect on model logits or quality.

The largest raw-output max-abs statistic among failing cases is 0.249755;
the ideal FP16-rounding maximum on that same case is 0.249755.
Large max-abs numbers at large activation magnitudes can be normal FP16 rounding.
Use the witness, allowance and counts above to judge the extra numerical defect;
RMSE or the global maximum alone cannot establish acceptance.

### Which failures are important enough to fix?

The measured Qwen workload currently passes every checked element. **Do not call
weak gates themselves rare or extreme:** 36.97%
of its gates exceed -0.001 and 1.60% exceed -1e-6.
The unverified part is sustained weak forgetting over a long sequence with the
relevant beta, key and state history. Global percentiles do not measure that joint
behavior. Captured initial states already reach about 12.60, so the failing
amplitude-32 tests' state maximum of about 12.5 is not, by itself, unrealistic.

1. **First priority: obtain long-context real captures with their incoming states.**
   Check the 32K near-lossless regime: it exceeded the old bound at unit scale, so it is
   potentially relevant even when V never reaches the large stress magnitudes.
   A failure beyond the current approved budget would justify prioritizing a parallel numerical fix.
2. **Lower immediate priority: the amplitude-32/128 stress-only combinations.**
   Their failing V maxima are roughly 125–500 versus 19.41 in these captures.
   Establish their joint frequency in the target model before paying a general
   runtime cost. This is a prioritization judgment, not a claim that they are safe
   or impossible; retain their failures and test larger models/contexts.
3. **Keep the decision tied to elementwise errors and model impact.** A few
   small violations and an overflow need different responses. Any repair should
   preserve the fast parallel execution. The approved policy update above is
   explicit; historical metrics are not relabeled. These small stress-test errors alone
   do not justify a costly fallback or a general performance penalty.

[All refreshed cases, five repeats and fingerprints](data/numerical_regimes.json) ·
[Captured-input and ordinary-control evidence](data/fallback_removal.json) ·
[Input ranges](input_ranges.md).

Run all regular regressions and optionally save a fresh audit under the current
policy. Preserve the historical diagnostic artifact instead of overwriting it:

```bash
bash scripts/gdn_mega_docker.sh python -m pytest \
  tests/test_gdn_mega.py tests/test_gdn_mega_validation.py \
  tests/test_gdn_mega_errors.py tests/test_gdn_mega_benchmark.py -q
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite all --output benchmark_data/gdn_mega/current_accuracy.json
bash scripts/report_gdn_mega.sh render
```

## Reproduce

```bash
bash scripts/report_gdn_mega.sh errors
bash scripts/report_gdn_mega.sh render
```

The errors command requires the original library at
`benchmark_data/gdn_mega/baseline_lib/libno_workspace_kernel.so` and current at
`build/lib/libno_workspace_kernel.so`. Numerical failures are retained as data.

[Measurements, repeats and hashes](data/error_heatmap.json) ·
[CSV](data/error_heatmap.csv) ·
[Maximum-error SVG](figures/error_heatmap_max_abs.svg) ·
[Tolerance SVG](figures/error_heatmap_tolerance.svg)
