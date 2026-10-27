# GDN speedup heatmap

Generated from measurements dated 2026-10-27.
Original **10748eb**, current **677c0ee**. [Overview](README.md).

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
ratios, ranging from **1.26× to 4.53×**.
For 4K and longer, the range is **1.26× to 1.80×**.
There are 0 ratios below 1×. Gray cells mean
that a timing is unavailable, not merely that numerical validation failed.

## Original latency, milliseconds

Columns are token counts. `*` identifies a numerical failure.

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.204* | 0.260* | 0.376* | 0.976* | 1.726* | 3.441* | 6.587* |
| 12/3 | 0.280* | 0.339* | 0.479* | 1.244* | 2.287* | 4.433* | 8.917* |
| 16/4 | 0.112* | 0.146* | 0.314* | 0.890* | 1.721* | 3.422* | 6.934* |
| 24/6 | 0.136* | 0.168* | 0.356* | 1.158* | 2.257* | 4.726* | 9.277* |
| 24/8 | 0.132* | 0.168* | 0.351* | 1.165* | 2.301* | 4.704* | 9.279* |
| 32/8 | 0.156* | 0.212* | 0.497* | 1.726* | 3.474* | 6.988* | 13.738* |
| 48/12 | 0.161* | 0.228* | 0.595* | 2.292* | 4.759* | 9.314* | 18.416* |
| 48/16 | 0.158* | 0.226* | 0.593* | 2.319* | 4.705* | 9.339* | 18.491* |
| 64/16 | 0.199* | 0.305* | 0.819* | 3.147* | 6.351* | 12.477* | 25.014* |

## Current latency, milliseconds

Columns are token counts. Current timings remain available even where the original fails.

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.063 | 0.083 | 0.199 | 0.654 | 1.288 | 2.503 | 5.088 |
| 12/3 | 0.062 | 0.083 | 0.206 | 0.692 | 1.341 | 2.675 | 5.292 |
| 16/4 | 0.061 | 0.082 | 0.204 | 0.692 | 1.356 | 2.666 | 5.377 |
| 24/6 | 0.063 | 0.085 | 0.208 | 0.702 | 1.393 | 2.785 | 5.626 |
| 24/8 | 0.064 | 0.086 | 0.209 | 0.720 | 1.424 | 2.872 | 5.699 |
| 32/8 | 0.094 | 0.135 | 0.376 | 1.374 | 2.696 | 5.407 | 10.707 |
| 48/12 | 0.096 | 0.137 | 0.380 | 1.401 | 2.798 | 5.636 | 11.131 |
| 48/16 | 0.096 | 0.139 | 0.383 | 1.423 | 2.883 | 5.712 | 11.405 |
| 64/16 | 0.129 | 0.191 | 0.555 | 2.084 | 4.154 | 8.269 | 16.587 |

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
