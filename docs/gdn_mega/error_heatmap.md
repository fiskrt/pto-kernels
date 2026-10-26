# GDN absolute-error heatmaps

Measurements dated 2026-10-26; original **10748eb**,
current **aac5a4d**. Both receive identical inputs and reference
results. [Overview](README.md) · [Latency heatmap](heatmap.md).

![Maximum absolute raw-output error](figures/error_heatmap_max_abs.png)

**Current passes 63/63 shapes; original passes
0/63**, checking both output and final state.
Finite failures remain visible on the shared scale; gray marks NaN/Inf.

## Elementwise acceptance

```text
abs(actual[i] - reference[i]) <= 1e-5 + 1e-3 * abs(reference[i])
```

This is a mixed absolute/relative bound, **not** a uniform maximum absolute error
of 1e-5. The ratio below divides each element's error by its allowance; a maximum
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
