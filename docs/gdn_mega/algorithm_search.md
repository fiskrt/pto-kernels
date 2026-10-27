# GDN structural search: current findings

Production remains the C=128, C_inv=16 kernel (`677c0ee`); no experimental
arithmetic or workspace contract has been promoted. This report records measured
combinations, not a proof that the current configuration is globally optimal.
[Measurements, numerical metrics and source/library hashes](data/algorithm_search.json).

## Full-pipeline NPU results

Ascend910B2, D=128, 16 value heads / 4 key heads, T=4096, nonzero initial state.
Times are median device microseconds from 15 graph trials of 20 calls. Ranges
show the two latest measurements in opposite variant order. The nine-case
screen retains exact regular-suite inputs and five poisoned-workspace launches;
it does not replace the complete 246-test suite.

| C | C_inv | A: leaf inversion | J: joining/application | Total µs | Numerical evidence |
|---:|---:|---|---|---:|---|
| 128 | 16 | Forward substitution, vector FP32 | Full inverse, cube FP32 (production) | 697–700 | Full 246 tests pass |
| 128 | 16 | Forward substitution, vector FP32 | Factored final join + asynchronous independent joins | **652–666** | Full 246 tests pass |
| 64 | 16 | Forward substitution, vector FP32 | Factored final join | 835 | 9/9 screen pass |
| 64 | 32 | Forward substitution, vector FP32 | Factored final join | 960 | 9/9 screen pass |
| 128 | 32 | Forward substitution, vector FP32 | Full or factored final join | Not measured | Both 8/9; large nearly-correlated case fails |
| 128 | 32 | Reverse substitution, vector FP32 | Full inverse | 843 | 9/9 screen pass |
| 128 | 32 | Reverse substitution, vector FP32 | Factored final join | 851 | 9/9 screen pass |
| 128 | 64 | Forward substitution, vector FP32 | Factored final join | Not measured | 8/9; large nearly-correlated case fails |
| 256 | 16 | Forward substitution, vector FP32 | Factored final join | 868 | 8/9; large nearly-correlated case fails |
| 256 | 16 | Reverse substitution, vector FP32 | Factored final join | Not measured | Same screen case fails |

The C=256 prototype needs larger scratch allocations, streaming vector panels,
separate Q/K projections and separate output/state products. It also stores
intermediate state/output bases in global memory to fit on chip. Its time is a
result for that concrete schedule, not evidence that every C=256 design is slow.
It was only built/screened for H=8/16. The C=64/128 prototypes retain the existing
allocation sizes. Initial C=64 layout and C=256 capacity bugs were fixed before
these numerical results; failed device executions were never timed.

A separate chunk-worker rotation prototype uses otherwise idle groups but adds
state transfer and readiness polling. It passed 9/9 and took 904 µs at 4K, so this
implementation is rejected.

## What the A experiments establish

The CPU screen covers 1,920 case/configuration combinations: C=64/128/256,
C_inv=8/16/32/64, full/factored joining, eight fixed distributions, and several
leaf algorithms/arithmetic choices. It tests correlated and nearly correlated
keys, beta=1, zero/tiny gates, high-amplitude inputs and FLA input families.
These are sampled matrices, not complete input-domain guarantees.

For L=I-N with strictly lower-triangular N, the inverse is the *finite* polynomial
I+N+...+N^(C_inv-1). None of these models drops powers based on decay.
Horner, Estrin, Paterson–Stockmeyer and band-limited Newton have **not yet been
implemented and benchmarked as complete NPU pipelines in this sweep**.

Maximum absolute leaf-inverse error against FP64, with FP32 arithmetic:

| A | C_inv=8 | C_inv=16 | C_inv=32 | C_inv=64 |
|---|---:|---:|---:|---:|
| Forward substitution | 2.53e-08 | 3.31e-08 | 3.31e-08 | 4.34e-08 |
| Horner | 2.53e-08 | 3.31e-08 | 3.31e-08 | 4.34e-08 |
| Estrin / power doubling | 2.78e-06 | 4.48e-04 | 2.00e+01 | 8.59e+10 |
| Paterson–Stockmeyer | 8.94e-07 | 2.24e-04 | 8.88e+00 | 2.77e+10 |
| Band-limited Newton | 2.53e-08 | 3.50e-08 | 3.31e-08 | 4.34e-08 |

- **Forward/reverse substitution:** the best demonstrated choices so far.
  Larger leaves remove cube joins but lengthen vector dependencies. Reversing
  the solve direction changes rounding and rescued the C_inv=32 screen, but was
  slower than C_inv=16. A nine-case pass is not a full accuracy certificate.
- **Horner:** accurate in this CPU screen, but has C_inv−2 dependent dense matrix
  products after forming I+N: 14 at size 16, 30 at size 32. Its intermediate
  magnitude reached 2.33e17 at size 64 even though its final inverse was accurate.
  Thus accurate FP32 results do not establish safety of an FP16 implementation.
- **Estrin and Paterson–Stockmeyer:** fewer matrix stages, but severe cancellation
  on some correlated inputs. Size-16 inverse errors are already about 2e-4–4e-4;
  size-32 errors reach order 10. Their unmodified FP32 forms are poor candidates
  for the full supported numerical regime. This does not rule out stabilized
  or differently preconditioned variants of these methods.
- **Band-limited Newton:** the most interesting unmeasured A candidate. Starting
  with the diagonal and first subdiagonal, apply X←X(2I−LX), then retain the
  diagonal band known to be correct. The band grows 1→3→7→15→31→63. In exact
  arithmetic, if I−LX is zero through band b, its square is zero through band
  2b+1; truncating outside the new band preserves those correct entries. This
  is a structural construction, not a decay threshold. Sampled intermediate
  magnitudes stayed ≤1 and FP32 errors remained near forward substitution.
  It needs 6 dense products at size 16 and 8 at size 32, plus masking and data
  movement. Whether cube execution beats vector substitution is still unknown.

Single-FP16-operand polynomial models also fail multiple sampled regimes.
Neither their failure nor the CPU FP32 success substitutes for NPU elementwise
verification.

## What J and C establish

For a two-block unit lower-triangular system, let D0 and D1 be the diagonal-block
inverses and F=−D1 L10. The original schedule forms F D0 and then applies the
full inverse. The factored schedule computes

```
R0 = D0 B0
R1 = D1 B1 + F R0
```

It avoids forming F D0. The successful candidate also overlaps independent
joins and their transfers with explicit pipeline dependencies. Its measured
improvement combines the algorithm and scheduling changes; it is not a pure
J-only ablation. An earlier isolated factored-only screen gave about 674 µs
versus 692 µs for its matched baseline.

The same A=32 forward-substitution failure occurs with both full and factored
J. Changing A to reverse substitution fixes that screen for both joins. Thus
accuracy belongs to the complete (C,C_inv,A,J) combination, not an independent
certificate for either A or J.

C=64 increases the number of recurrent steps/state exchanges; reduced inverse
work did not offset that cost in either measured configuration. C=256 reduces
those exchanges but increases matrix work and forces substantial retiling.
**C=128, C_inv=16 remains the best measured complete configuration.** The modest
factored/asynchronous gain is not the requested breakthrough, and its long-context
gain is smaller and variable; no 2× or 3× new improvement has been demonstrated.

## HF32 precision/runtime tradeoff

This is a separate arithmetic experiment on the factored/asynchronous C=128,
C_inv=16 schedule. It enables the hardware's HF32 mode for all FP32 cube
products; tensors still use the original public dtypes. The [Ascend C documentation](https://asc.gitcode.com/api/SIMD-API/c_api/cube_compute/mmad_compute_key_features/HF32.html)
describes operand rounding and increased matrix arithmetic throughput. That is
not a guarantee of a corresponding full-kernel speedup.

Fresh matching inputs and CPU FP64 token recurrence; 13 selected numerical cases,
five poisoned launches each, plus checked timing replays. Both FP32 variants pass
13/13; HF32 passes 3/13. All remain finite and preserve inputs/guards. This is
selected coverage, not a full HF32 regression audit.

| T | Production µs | Matched FP32 candidate µs | HF32 µs |
|---:|---:|---:|---:|
| 256 | 82.1–84.3 | 79.6–80.6 | 76.6–78.1 |
| 4096 | 697.2–700.2 | 651.6–665.9 | 620.6–624.0 |
| 32768 | 5371.0–5456.1 | 5154.9–5363.4 | 4948.2–4980.7 |

Keeping rtol=1e-3, the following absolute floors would be necessary to accept
both output and final state for each HF32 case (current atol is 1e-4):

| Regime | Required atol | HF32 max output abs error | HF32 max state abs error |
|---|---:|---:|---:|
| Ordinary weak gates | 0.000346494 | 0.000811651 | 0.000764642 |
| Identical keys, beta=1, no decay | 0.00394666 | 0.00106033 | 0.00402832 |
| Identical keys, beta=1, g=−1e-7, 32K | 0.00295667 | 0.00233753 | 0.00390625 |
| Nearly correlated, beta=1, g=−1e-6, amplitude 1 | 0.00234357 | 0.00290486 | 0.00192744 |
| Same regime, amplitude 32 | 0.0749942 | 0.0929554 | 0.0616779 |
| Same regime, amplitude 128 | 0.299977 | 0.371822 | 0.246712 |

Maximum error alone can hide damage near zero: in the amplitude-128 case,
production's maximum output error is 0.249755 (FP16 output rounding), versus
HF32's 0.371822. But at a reference value of −0.00812336, HF32 has absolute error
0.124456. HF32 violates the elementwise bound at 176,009 of 1,049,600 output
elements in this case (~16.8%), despite relative RMSE≈3.47e-4 passing the RMSE
criterion. Accepting all 13 sampled cases at fixed rtol would require atol≈0.30.
That would not establish acceptance of the full suite or model-quality equivalence.
No tolerances have been changed and HF32 has not been promoted.

## Reproduction and next work

The committed fast build/verification entry point is:

```bash
bash scripts/gdn_mega_docker.sh bash scripts/check_gdn_mega.sh --quick
bash scripts/gdn_mega_docker.sh bash scripts/check_gdn_mega.sh
```

`--source` and `--output` build isolated source snapshots. Raw experiments,
generators and logs remain under `benchmark_data/gdn_mega/optimize_20261027/`;
none are production kernel includes. The C=256 prototype additionally requires
its explicitly experimental allocation harness. Production has seven sources,
one schedule and its unchanged caller allocation contract.

The next meaningful A experiment is a hardware mapping of stable, band-limited
Newton, including the masking/transfer cost and overlap with other stages.
More pipelined C=256 schedules and other joining algorithms remain open; the
current sweep does not exhaust this search space.
