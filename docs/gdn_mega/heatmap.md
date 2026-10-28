# GDN speed heatmap

Measured 2026-10-28; baseline library `4bf4218a7473`, current `0bf6f7832dd1`.
Nonzero initial state: True; seed 0; gate scale 1.0; D=C=128.
Independent CPU FP64 token recurrence on actual rounded inputs. Every output/state element must satisfy
`abs(error) <= 0.0001 + 0.001*abs(reference)`, plus finiteness and relative RMSE < 0.005.
Current passes 63/63 shapes; baseline passes 0/63.

![Latency](figures/heatmap.png)

Each cell shows runtime ratio and total median µs for both kernels. Asterisks
mark only failed implementations; those ratios are not accuracy-equivalent
speedups.

## Current median total microseconds

| H/Hg | 128 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/2 | 61.3 | 79.0 | 192.2 | 625.0 | 1242.1 | 2425.9 | 5044.9 |
| 12/3 | 60.2 | 79.9 | 194.5 | 657.1 | 1285.5 | 2590.4 | 5200.4 |
| 16/4 | 59.1 | 79.6 | 197.0 | 663.8 | 1311.7 | 2621.0 | 5339.3 |
| 24/6 | 62.3 | 82.7 | 200.2 | 681.2 | 1362.4 | 2773.6 | 5606.8 |
| 24/8 | 62.4 | 83.6 | 201.9 | 695.6 | 1390.1 | 2846.0 | 5683.8 |
| 32/8 | 92.6 | 132.5 | 366.0 | 1334.5 | 2659.1 | 5394.4 | 10688.3 |
| 48/12 | 94.9 | 134.5 | 368.4 | 1376.8 | 2789.4 | 5615.4 | 11140.7 |
| 48/16 | 92.9 | 133.2 | 366.3 | 1390.9 | 2884.8 | 5717.4 | 11415.2 |
| 64/16 | 126.5 | 186.6 | 535.1 | 2058.8 | 4153.1 | 8277.0 | 16569.2 |

## Scheduling tradeoff

The removed H8-specific schedule (`365b105`) used sixteen state-column workers
and eight preparation helpers. Matched H8/Hg2 measurements at 4K–32K showed
**22–27% lower latency (1.29–1.38× throughput)** than the uniform schedule.
It was removed for simplicity, saving 153 source lines; shared factored and
asynchronous joins remain. This is a historical comparison, not a guarantee
for arbitrary candidate libraries supplied to this script.

See [README](README.md) for reproduction. Raw run files are intentionally
excluded from the documentation tree.
