# GDN speedup heatmap

Generated from measurements dated 2026-10-28.
Original **10748eb**, current **f381ae8**. [Overview](README.md).

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
ratios, ranging from **1.29× to 4.66×**.
For 4K and longer, the range is **1.29× to 1.87×**.
There are 0 ratios below 1×. Gray cells mean
that a timing is unavailable, not merely that numerical validation failed.

## Scheduling tradeoff

Production was simplified to one head-parallel schedule for all shapes. A custom
H8 schedule in `365b105` used 16 recurrence groups (two independent value-column
slices per head) and eight preparation helpers. Matched H8/Hg2 measurements at
4K–32K showed **22–27% lower latency (1.29–1.38× throughput)**. This is an
additional gain over the uniform schedule, not over the original baseline.
We removed it to keep the code simple: one schedule and 153 fewer source lines.
The shared factored/asynchronous inverse joins remain. The removed schedule
is available in Git history; it is not a runtime tuning option.
[Measured comparison and numerical validation](algorithm_search.md#schedule-simplification).

## Original latency, milliseconds

Columns are token counts. `*` identifies a numerical failure.

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.213* | 0.269* | 0.386* | 0.980* | 1.723* | 3.433* | 6.607* |
| 12/3 | 0.280* | 0.339* | 0.472* | 1.226* | 2.283* | 4.456* | 8.925* |
| 16/4 | 0.110* | 0.148* | 0.317* | 0.892* | 1.726* | 3.519* | 6.948* |
| 24/6 | 0.137* | 0.171* | 0.357* | 1.142* | 2.268* | 4.730* | 9.275* |
| 24/8 | 0.135* | 0.172* | 0.357* | 1.143* | 2.271* | 4.732* | 9.297* |
| 32/8 | 0.158* | 0.213* | 0.499* | 1.725* | 3.473* | 6.969* | 13.740* |
| 48/12 | 0.159* | 0.227* | 0.593* | 2.296* | 4.754* | 9.332* | 18.441* |
| 48/16 | 0.155* | 0.224* | 0.592* | 2.285* | 4.707* | 9.324* | 18.490* |
| 64/16 | 0.200* | 0.304* | 0.815* | 3.149* | 6.352* | 12.481* | 25.022* |

## Current latency, milliseconds

Columns are token counts. Current timings remain available even where the original fails.

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.061 | 0.079 | 0.192 | 0.625 | 1.242 | 2.426 | 5.045 |
| 12/3 | 0.060 | 0.080 | 0.195 | 0.657 | 1.285 | 2.590 | 5.200 |
| 16/4 | 0.059 | 0.080 | 0.197 | 0.664 | 1.312 | 2.621 | 5.339 |
| 24/6 | 0.062 | 0.083 | 0.200 | 0.681 | 1.362 | 2.774 | 5.607 |
| 24/8 | 0.062 | 0.084 | 0.202 | 0.696 | 1.390 | 2.846 | 5.684 |
| 32/8 | 0.093 | 0.133 | 0.366 | 1.334 | 2.659 | 5.394 | 10.688 |
| 48/12 | 0.095 | 0.134 | 0.368 | 1.377 | 2.789 | 5.615 | 11.141 |
| 48/16 | 0.093 | 0.133 | 0.366 | 1.391 | 2.885 | 5.717 | 11.415 |
| 64/16 | 0.127 | 0.187 | 0.535 | 2.059 | 4.153 | 8.277 | 16.569 |

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
