# Original algorithm versus current stable kernel

The current stable kernel is about7–10% faster at4K–32K than the original
algorithm, not3×. The previously reported1.58× compared with a slower, correct
FP32 intermediate checkpoint, not the original.

Ascend910B2;16 value heads/4 key heads;D128;nonzero initial state;ordinary
Gaussian normalized Q/K workload, gate_scale1. Median of15 trials ×20 graph calls;
allocations, reference calculations and output scaling excluded from timing.

| Tokens | Original ms | Current stable ms | Speedup |
|---:|---:|---:|---:|
| 256 | 0.147 | 0.092 | 1.61× |
| 1024 | 0.312 | 0.231 | 1.35× |
| 4096 | 0.869 | 0.789 | 1.10× |
| 8192 | 1.720 | 1.581 | 1.09× |
| 16384 | 3.502 | 3.205 | 1.09× |
| 32768 | 6.965 | 6.534 | 1.07× |

Original is10748eb, from the saved2026-10-23 measurements in
[production_latency.csv](production_latency.csv). Current production code is
bfc99e1. Current256/1K/4K were measured again on2026-10-26;8K/16K/32K use the
saved2026-10-23 run of the identical binary. The new4K result0.788741ms is within
1.3% of the saved0.798337ms. No kernel changes were made for this comparison.
[New short-length raw records](original_comparison_short_20261026.json),
[saved long-length records](current_benchmark.json), [table CSV](original_vs_current.csv).

Current measurements pass pointwise checks before and after graph replay; the
production checkpoint passes80 strict tests. Original timings were collected
under the historical weaker acceptance policy: the original is not certified
against the current full numerical suite. This table is for the ordinary
benchmark distribution; positive-key FLA cases can select the slower compensated
path, as documented in [current_status.md](current_status.md).
