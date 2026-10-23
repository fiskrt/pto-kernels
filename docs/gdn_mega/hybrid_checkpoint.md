# Verified native/paired hybrid checkpoint

80 strict tests pass, including the FLA-derived distributions and strong-decay
tail regressions. No tolerance or reference was relaxed. Full test log:
[hybrid_tests.txt](hybrid_tests.txt). Raw timing/error records:
[hybrid_benchmark.json](hybrid_benchmark.json),
[hybrid_fla_benchmark.json](hybrid_fla_benchmark.json).
The independent [full numerical audit](hybrid_accuracy.json) passes all 36 cases
with five poisoned-workspace repeats each. It saves maximum absolute errors,
percentiles, every pointwise violation count, guard/input checks and binary hashes.

Changes from the paired checkpoint ff86de2:

1. Native FP32 block joins and residual solve replace paired arithmetic inside
   the inverse. Original FP16 K/Q still multiply paired FP32 state/residuals.
2. Native FP32 attention removes residual splitting/reconstruction and its
   identity-matrix multiplication. Full inverse FP16-pair buffers are no longer
   written; only coupling markers are materialized globally.
3. Invalid rows are masked before exp, preventing padded-prefix overflow and
   0*infinity/0*NaN propagation into valid outputs.

H16/Hg4, D128, nonzero initial state; median of15 trials × 20 graph calls:

| Tokens | Hybrid ms | Paired checkpoint ms | Speedup |
|---:|---:|---:|---:|
| 128 | 0.066423 | 0.076513 | 1.152× |
| 4096 | 0.905793 | 0.968760 | 1.070× |
| 8192 | 1.783001 | 1.915925 | 1.075× |
| 16384 | 3.615756 | 3.907988 | 1.081× |
| 32768 | 7.520225 | 7.935392 | 1.055× |

4K is 1.390× faster than the correct FP32 checkpoint (1.259212 ms), but still
slower than the historical original (~0.87 ms). The historical ~0.25 ms FP16
path failed the stricter numerical policy. The requested target is not reached.

The conservative strong-coupling path remains slow. FLA-derived examples below
use H8/Hg2, D128 and5 trials × 5 graph calls; they are different distributions/shapes,
not direct speedup comparisons with the Gaussian H16 table above. These use the
independent CPU FP64 token oracle before/after graph replay.

| FLA-derived case | Tokens | Initial state | ms |
|---|---:|---|---:|
| fla_positive_no_decay | 1024 | False | 3.686692 |
| fla_positive_weak | 1500 | False | 5.393116 |
| fla_positive_strong | 257 | False | 0.124532 |
| fla_gaussian_initial | 1024 | True | 0.259980 |
| fla_negative_eigenvalues | 513 | False | 1.873560 |

There are still two global phases and one launch. Public arguments, dtypes,
output scaling, allocation sizes and initial-state behavior are unchanged.
Previously reserved per-core buffers now hold FP32 inverse/RHS data; `w` is
reserved and `A` holds only coupling flags. See the kernel README for the full
workspace map. No debug kernels or selectable tuning configurations were added.

The broader experiments and rejected timings are documented in
[radical_experiments.md](radical_experiments.md). Input ranges and why inverse
powers overflow despite bounded KK/inverse entries are in
[input_ranges.md](input_ranges.md). Passing tests is evidence for the tested
cases, not a proof for arbitrary unbounded inputs.
