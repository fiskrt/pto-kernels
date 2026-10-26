# GDN speedup heatmap

Generated from measurements dated 2026-10-26.
Original **10748eb**, current **aac5a4d**. [Overview](README.md).

![Original/current latency](figures/heatmap.png)

Original timings are historical RMSE-only measurements. The original fails the
current elementwise policy; this is **not an accuracy-equivalent comparison**.

Current passes **49/49 shapes**. There are 48 valid latency
ratios, ranging from **1.18× to 4.12×**.
For 4K and longer, the range is **1.18× to 1.71×**.
There are 0 ratios below 1×. Gray cells identify
an explicitly failed implementation; no speedup is inferred for them.

## H16/Hg4

| Tokens | Original ms | Current ms | Speedup × |
|---:|---:|---:|---:|
| 128 | 0.106 | 0.066 | 1.60 |
| 256 | 0.145 | 0.088 | 1.65 |
| 1024 | 0.314 | 0.214 | 1.47 |
| 4096 | 0.883 | 0.724 | 1.22 |
| 8192 | 1.727 | 1.420 | 1.22 |
| 16384 | 3.500 | 2.850 | 1.23 |
| 32768 | 6.925 | 5.766 | 1.20 |

## Measurement and acceptance

H=8,12,16,24,32,48,64; Hg=H/4; D=chunk size 128;
nonzero initial state; seed 0; gate scale1.0.
Normalized Gaussian Q/K and ordinary benchmark inputs; this grid does not cover
all FLA distributions or characterize the slower strong-coupling fallback.

5 checked warmups, median of 15 trials ×
20 graph calls. Allocation and CPU reference work are outside timing.
The loaded library hash is checked for every worker.

Every raw-output and final-state element must satisfy
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005,
before and after timing. Cached statistics are reused only after byte equality
of every output/state element. The grid uses a CPU FP32 chunked reference;
the production tests separately include an independent FP64 token oracle.
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
