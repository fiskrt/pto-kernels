# Algorithm experiments, 2026-10-23

Reference checkpoint: `ff86de2`, H16/Hg4, D128, T4096, nonzero initial state,
**0.968760 ms**. It passes 70 original tests and the eight new FLA-derived cases.
The correct FP32 predecessor was 1.259212 ms. The historical original was ~0.87 ms;
the ~0.25 ms historical optimized FP16 version fails the strict accuracy policy.

All timings below use the same checked graph benchmark, 15 trials × 20 calls.
Reported timings were collected without another NPU job running. An accidentally launched timing
process during the first scheduling test was terminated; only the later isolated
run is recorded. Rejected variants are not retained as unused production kernels.

| Experiment | Structural change | 4K ms | Validation | Decision |
|---|---|---:|---|---|
| Chunk-parallel inverse phase | Complete all independent inverses before recurrent work | 1.228863 | 78/78 full tests | Reject: added exports, reloads and synchronization outweigh parallelism |
| Direct 32-row panel solve | Four forward panel solves; no full inverse | 1.251790 | 6 selected strict audit cases + checked benchmark | Reject: additional RHS dependencies/products outweigh saved inverse work |
| FP16 approximate inverse + RHS refinement | Cheap inverse, accurate solve residual, one correction | 1.278160 | 7 selected strict audit cases + checked benchmark | Reject: extra large products and vector/cube exchanges dominate |

| Native FP32 solve, paired state/output | Native joins and RHS product replace software precision inside inverse | 0.945859 | 7 selected audits + benchmark | Promising; extend to attention |
| Native solve and attention | Remove residual split/reconstruction and its identity multiply | 0.931576 | 7 selected audits + benchmark | Best interim timing; full suite later caught tail overflow |
| Two 64-row diagonal solves | One native block join | 1.291802 | 7 selected audits + benchmark | Reject: vector substitution dominates |
| Four 32-row diagonal solves | Three native block joins | 1.217353 | 7 selected audits + benchmark | Reject: larger vector solves still dominate |
| Block-preconditioned finite series | Six dense native products; exact nilpotent eight-block formula | 1.216864 | 7 selected audits + benchmark | Reject: extra dense work/spills outweigh fewer joins |

These are exploratory timings of the benchmarked shape, not full-suite certificates.
The final hybrid candidate masks invalid rows before exponentiation after its
first full run exposed three nonfinite-output tail failures (75 passed, 3 failed).
Final validation and timings must be checked separately from these interim runs.

The CPU feasibility screen (`algorithm_screen.json`) compares direct FP32 solve,
16-row panels and one/two refinement steps. All pass sampled ordinary and FLA
families, but even direct FP32 fails the near-identical-key amplitude-32 stress
case. This screens algorithms; it does not simulate exact NPU rounding or timing.
Hardware candidates keep the compensated recurrence for strong coupling.

The precomputed-inverse experiment initially exposed a missing cross-subblock
handshake when reading coupling markers. Poisoned scratch caught it. Adding the
handshake resolved it before the reported 78-test run; the timed variant includes
that synchronization. A stalled/incomplete run is not counted as passing.

Raw timing JSON and test logs are adjacent to this file. Public ABI, allocation
sizes and tolerance policy were unchanged in all three variants. No target-speed
claim is made: the rejected variants do not improve the saved checkpoint.


## Shared projection/residual operands

Stacking K and Q into one paired-state product alone measured 0.909957 ms
versus the 0.905793 ms hybrid checkpoint: no improvement. Moving residual
weighting into independent preparation, Gamma = exp(G_last-G_i)*K, permits
one stacked native product [attention; Gammaᵀ]*R. This removes the runtime
residual split/scale exchange and flags 3/4/10 from the ordinary chunk path.
Initial smoke/benchmark passed at 0.844531 ms, but the full suite caught one
late-coupling failure (79 passed). The reused control slot needed explicit
scalar-cache invalidation after DMA updates. With that fixed, all80 tests pass;
final corrected timings are saved separately in fused_benchmark.json.
The failed and fixed five-repeat audits are retained as fused_control_failure.json
and fused_control_fixed.json. No acceptance policy changed.


## Remove duplicate decay preparation

Native attention stores preserve the decay tile. Sharing it with KK weighting
removes one full chunk decay construction and exponentiation, without changing
G_i-G_j or dropping any terms. Final H16/Hg4/4K nonzero-state timing0.794724 ms
versus0.844893 ms:1.063×. All80 strict tests pass; logs and raw data are
shared_decay_tests.txt/shared_decay_benchmark.json. The first timing before code
formatting was0.797781 ms. Existing ABI, allocations, files and phases unchanged.

Batching four16×16 inverse joins through a shared export/reload gave0.806571 ms
with8 selected tests and the checked benchmark passing. It is rejected; the
separate stable joins remain. Raw result:batched_join_benchmark.json.

The preceding fused hardware profile is saved in fused_profile.json. Active-head
median cube time was845.05us, with175.58us arithmetic,205.25us MTE2,148.49us FIX,
and166.43us RHS/state waits. Vector0 had358.29us vector work. These overlapping
per-pipe counters cannot be added as stage durations. They support removing
redundant preparation/transfers; they do not show a compute-saturated kernel.
