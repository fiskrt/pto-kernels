# Current stable GDN result

Production code: **bfc99e1** (preceded by shared-operand fusion **a329112**).
All80 strict tests pass. The restored library hash matches the tested checkpoint;
there are no uncommitted kernel changes or rejected experimental kernels retained.
The requested3× speed target is **not reached**.

## Input ranges and the actual numerical problem

The pinned FLA main tests use positive normalized Q/K, not just zero-mean Gaussian
keys. Their typical off-diagonal KKᵀ is about0.75. Beta includes sigmoid(uniform),
sigmoid(Gaussian), and a doubled family approaching2; gates include zero decay.
Gaussian V and initial state have no finite mathematical generator bound.
[Full source-linked range analysis and derivations](input_ranges.md).

For the bounded main generator with zero initial state, the exact physical state
norm is bounded below1555 at32K/D128, before accounting for rounding perturbations.
Yet a measured FLA-style block has intermediate inverse powers up to4.76e18
while its final inverse has maximum magnitude1. Scaling those powers can prevent
exponent overflow but cannot recover bits lost through cancellation. Stable block
substitution avoids the enormous intermediates; strict near-zero errors still
require FP32/compensated arithmetic. The1555 bound does not cover every FLA family.

## Retained structural changes

1. Stable16×16 FP32 diagonal substitution and block joins replace unstable
   inverse powers. Native FP32 joins/solve/attention remove costly paired
   reconstruction; original K/Q multiply two FP16 components of FP32 state.
2. Stack K/Q projections to share state input. Precompute Gamma=exp(G_last-G_i)*K,
   then stack attention*R and Gammaᵀ*R. This removes a dependent residual scaling
   exchange and three synchronization flags per ordinary chunk.
3. Share one decay matrix between QK and KK weighting. G_i-G_j remains, with
   invalid rows masked before exp. No decay-dependent term truncation is used.

There is one launch, two global phases, and seven used production source files.
Arguments, dtypes, output scaling and caller allocation sizes are unchanged.
The current scratch map and persistent-kernel explanation are in the
[kernel README](../../csrc/kernel/gdn_mega/README.md).

## Final checked benchmark

Ascend910B2, H16/Hg4, D128, nonzero initial state,15 trials ×20 graph calls.
References/allocations are outside timing. Every benchmark checks all output and
state elements before and after replay. [Raw timings and errors](current_benchmark.json).

| Tokens | Prior native/paired checkpoint ms | Current ms | Speedup |
|---:|---:|---:|---:|
| 128 | 0.066423 | 0.068649 | 0.968× |
| 4096 | 0.905793 | 0.798337 | 1.135× |
| 8192 | 1.783001 | 1.580666 | 1.128× |
| 16384 | 3.615756 | 3.205259 | 1.128× |
| 32768 | 7.520225 | 6.534212 | 1.151× |

The repeated4K measurement is0.798337ms; the acceptance run measured0.794724ms.
That is about1.58× faster than the correct FP32 predecessor at1.259212ms, and
1.09× versus the historical original0.8688ms at the same shape/workload. The old
~0.25ms FP16 optimization failed the strict numerical checks and is not a valid
accuracy-equivalent comparator.128-token latency regresses against the prior
hybrid; longer sequences improve.

## FLA-derived workload timings

H8/Hg2, D128,5 trials ×5 graph calls, independent CPU FP64 token oracle.
These shapes/distributions differ from the Gaussian H16 table above.
[Raw checked records](current_fla_benchmark.json).

| Case | Tokens | Initial state | Current ms |
|---|---:|---|---:|
| fla_positive_no_decay | 1024 | False | 3.716416 |
| fla_positive_weak | 1500 | False | 5.428900 |
| fla_positive_strong | 257 | False | 0.120476 |
| fla_gaussian_initial | 1024 | True | 0.236840 |
| fla_negative_eigenvalues | 513 | False | 1.883372 |

The conservative compensated path still makes positive-key/no-decay and doubled
beta cases slow. The ordinary benchmark does not establish FP16-like speed for
these distributions. This remains a major unresolved performance limitation.

## What was tried and rejected

[Full experiment table and raw records](radical_experiments.md) cover
chunk-parallel inverses, direct panel solves, FP16 inverse plus refinement,
larger diagonal blocks, a block-preconditioned finite series, projection fusion
alone, batched small joins, and full affine chunk maps. Native FP32 solves and
the shared-operand pipeline were the productive directions. Full affine maps
passed15 selected tests but took1.253943ms, so they were removed. Rejected variants
are not left as selectable production configurations.

## Accuracy evidence and throughput interpretation

[80-test log](shared_decay_tests.txt). Acceptance checks every raw-output and
final-state element against1e-5+1e-3*abs(reference), plus relative RMSE<0.005,
finiteness, exact shapes, guarded/read-only memory and five poisoned-workspace
repeats for each of36 independent audit cases. No tolerance was weakened.
Maximum absolute error is not always tiny: the amplitude32 near-correlated case
has about0.03125 output quantization error, while all pointwise criteria pass.
[Its corrected detailed audit](fused_control_fixed.json) includes near-zero values.
Passing these cases is not a proof for arbitrary unbounded inputs.

For the ordinary4K path, executed dense matrix work (2MNK, vector work excluded):

| Operation | GFLOPs |
|---|---:|
| Shared KKᵀ/QKᵀ |1.073742|
| Stacked paired-state projections |8.589935|
| Stable inverse block joins |0.704643|
| Native residual solve |2.147484|
| Stacked residual output/state products |4.294967|
| Total |16.810770|

Aggregate matrix work divided by total runtime is about21.1TFLOP/s; it mixes
FP16 and FP32 products and includes waits/vector work in the denominator. It is
not a pure GEMM throughput measurement. Independent per-operation milliseconds
would be misleading because preparation and cube work overlap in the persistent
phase. The [preceding fused hardware profile](fused_profile.json) showed only
~20% cube arithmetic activity, substantial vector work and transfer/wait costs;
those per-pipe counters overlap and cannot be summed as stage durations.
