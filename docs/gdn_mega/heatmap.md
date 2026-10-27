# GDN speedup heatmap

Generated from measurements dated 2026-10-27.
Original **10748eb**, current **365b105**. [Overview](README.md).

![Original/current latency](figures/heatmap.png)

Each cell shows three lines: **runtime ratio**, **ours: median total µs**, and
**base: median total µs**. The times cover the full kernel, not an individual
stage; allocation and validation are excluded.
`*` appears only beside a failing implementation's runtime, never on the ratio.
Passing runtimes stay unmarked. Accuracy checks here belong to these timed
libraries and the policy below; the [error heatmaps](error_heatmap.md) retain
their separately labeled checkpoint and policy.

Original fails strict numerical checks on **63/63 cells**.
Ratios involving a starred runtime are **not accuracy-equivalent speedups**.

Current passes **63/63 shapes**. There are 63 measured runtime
ratios, ranging from **1.29× to 4.26×**.
For 4K and longer, the range is **1.29× to 1.92×**.
There are 0 ratios below 1×. Gray cells mean
that a timing is unavailable, not merely that numerical validation failed.

## Original latency, milliseconds

Columns are token counts. `*` identifies a numerical failure.

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.209* | 0.262* | 0.388* | 0.960* | 1.774* | 3.344* | 6.756* |
| 12/3 | 0.253* | 0.330* | 0.461* | 1.214* | 2.234* | 4.343* | 8.781* |
| 16/4 | 0.109* | 0.148* | 0.311* | 0.862* | 1.662* | 3.447* | 6.782* |
| 24/6 | 0.136* | 0.173* | 0.355* | 1.101* | 2.233* | 4.641* | 9.139* |
| 24/8 | 0.131* | 0.166* | 0.346* | 1.129* | 2.210* | 4.657* | 9.122* |
| 32/8 | 0.153* | 0.208* | 0.483* | 1.680* | 3.397* | 6.801* | 13.456* |
| 48/12 | 0.157* | 0.223* | 0.581* | 2.242* | 4.624* | 9.119* | 18.051* |
| 48/16 | 0.159* | 0.226* | 0.584* | 2.238* | 4.663* | 9.165* | 18.088* |
| 64/16 | 0.197* | 0.300* | 0.794* | 3.071* | 6.194* | 12.252* | 24.457* |

## Current latency, milliseconds

Columns are token counts. Current timings remain available even where the original fails.

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.064 | 0.082 | 0.177 | 0.519 | 0.926 | 1.900 | 3.627 |
| 12/3 | 0.060 | 0.079 | 0.192 | 0.646 | 1.260 | 2.531 | 5.077 |
| 16/4 | 0.060 | 0.080 | 0.193 | 0.642 | 1.271 | 2.540 | 5.147 |
| 24/6 | 0.063 | 0.083 | 0.199 | 0.674 | 1.332 | 2.701 | 5.408 |
| 24/8 | 0.061 | 0.082 | 0.198 | 0.674 | 1.354 | 2.762 | 5.510 |
| 32/8 | 0.091 | 0.131 | 0.356 | 1.303 | 2.564 | 5.182 | 10.333 |
| 48/12 | 0.093 | 0.132 | 0.360 | 1.329 | 2.707 | 5.418 | 10.828 |
| 48/16 | 0.092 | 0.133 | 0.360 | 1.353 | 2.803 | 5.556 | 11.092 |
| 64/16 | 0.125 | 0.182 | 0.520 | 1.998 | 4.025 | 8.042 | 16.170 |

## Measurement and acceptance

H/Hg=8/2, 12/3, 16/4, 24/6, 24/8, 32/8, 48/12, 48/16, 64/16; D=chunk size 128;
nonzero initial state; seed 0; gate scale1.0.
Normalized Gaussian Q/K and ordinary benchmark inputs; this grid does not cover
all FLA distributions or replace the separate correlated-key stress tests.

5 checked warmups, median of 15 trials ×
20 graph calls. Allocation and CPU reference work are outside timing.
The loaded library hash is checked for every worker.

The numerical pass criterion for every raw-output and final-state element is
`abs(error) <= 0.0001 + 0.001*abs(reference)`, plus finiteness and relative RMSE < 0.005,
before and after timing. The baseline is timed even when it fails, with its
failures retained. Current-kernel failures still fail the report command.
Cached statistics are reused only after byte equality
of every output/state element. The current grid uses the independent CPU FP64
token recurrence, computed from the actual rounded inputs. No inverse, decay
threshold or kernel intermediate participates in the oracle. Reference/check
wall times are saved separately from device timings.
See the [matched error heatmaps](error_heatmap.md) for absolute errors and violations.

## Reproduce

```bash
bash scripts/report_gdn_mega.sh latency
# Regenerate Markdown and plots from saved data without an NPU run:
bash scripts/report_gdn_mega.sh render
```

[Measurements and fingerprints](data/heatmap.json) ·
[CSV](data/heatmap.csv) ·
[SVG](figures/heatmap.svg)
