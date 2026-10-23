# Paired FP16 checkpoint, 2026-10-23

Stable 16-token diagonal solves and block-triangular joins replace inverse powers.
FP16 high/low products accumulate in FP32; recurrent state stays FP32.
Inverse preparation overlaps the previous chunk. Strongly coupled heads restart
with compensated FP32 token recurrence. This path has input-dependent latency.
The public ABI, input/output dtypes and caller allocation sizes are unchanged.

Strict tests: **70 passed**, unchanged elementwise absolute/relative tolerances,
independent FP64 recurrence and poisoned workspace checks. See tests.txt.

H16/Hg4, D128, nonzero initial state, median of 15 trials × 20 graph calls:

| Tokens | Latency (ms) |
|---:|---:|
| 128 | 0.076513 |
| 4096 | 0.968760 |
| 8192 | 1.915925 |
| 16384 | 3.907988 |
| 32768 | 7.935392 |

4K is 1.30× faster than the correct FP32 checkpoint (1.259212 ms),
but remains slower than the historical original (~0.87 ms). The old ~0.25 ms
FP16 version did not pass the current strict accuracy policy. Target not reached.
Benchmark JSON includes error metrics before/after graph replay and binary hash.
