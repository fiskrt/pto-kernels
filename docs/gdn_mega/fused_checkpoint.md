# Shared-operand pipeline checkpoint

All80 strict tests pass, including five poisoned-workspace repeats per independent
FP64 audit case. [Test log](fused_tests.txt); [raw checked timings](fused_benchmark.json).
Acceptance remains every raw-output/final-state element within
`1e-5 + 1e-3*abs(reference)`, relative RMSE <0.005, finite exact shapes,
unchanged inputs/guards and scratch-independent repeated results.

1. Stack K and Q to share paired-state input in their projections.
2. Prepare Gamma = exp(G_last-G_i)*K independently of recurrent state.
3. Compute [attention; Gammaᵀ]*R as a stacked native FP32 product. This removes
   the dependent residual-weighting/splitting exchange and three synchronization
   flags from each ordinary chunk. FP32 accumulation/state and stable inverse remain.

H16/Hg4, D128, nonzero initial state,15 trials ×20 graph calls:

| Tokens | Previous stable hybrid ms | Fused ms | Speedup |
|---:|---:|---:|---:|
| 128 | 0.066423 | 0.069667 | 0.953× |
| 4096 | 0.905793 | 0.844893 | 1.072× |
| 8192 | 1.783001 | 1.672497 | 1.066× |
| 16384 | 3.615756 | 3.392108 | 1.066× |
| 32768 | 7.520225 | 6.872840 | 1.094× |

At4K the result is1.490× faster than the correct FP32 predecessor's1.259212 ms,
and slightly faster than the historical original (~0.87 ms). It is still far
from the requested3× target and the historical inaccurate ~0.25 ms FP16 result.
The128-token case regresses; the larger preparation has a startup cost.
Strong-coupling fallback remains conservative and slow; this is not a claim of
similar performance for FLA's positive-key/no-decay distributions.

The first full test run caught a stale scalar cache line in the reused coupling
marker slot:79 passed, one late-coupling case failed. Explicit invalidation and
volatile reads fix it; the five-repeat [failure](fused_control_failure.json) and
[corrected audit](fused_control_fixed.json) remain recorded. Max absolute errors
in that amplitude32 case are about0.03125, largely final FP16 rounding; the
corrected run has zero pointwise violations, including near-zero values.

The kernel still has one launch, two global phases and seven used source files.
Public arguments, dtypes, shapes, output scaling and allocation sizes are unchanged.
`A/w` now hold the two FP32 column halves of Gamma; `wy_workspace_a1/a2` are
reserved compatibility buffers. The second half of `o_workspace_qk` holds
coupling flags; the first half holds inverse-join intermediates. No new tuning
options or debug kernels are retained. See the kernel README for the full map.
