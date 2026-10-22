# Validated GDN improvements

## 905ef0d — parallel gate scan and exact inverse doubling

Replaced 128 serial prefix-sum steps with seven parallel vector passes and
fused layout conversion. Replaced recursive dense inverse bookkeeping with
an exact nilpotent polynomial doubling algorithm that retains the FP32
accumulator, avoiding repeated multiplication by identity matrices.

All 21 tests pass on Ascend910B2, covering every supported value-head count,
initial states, ragged packed sequences, weak decay, and repeated execution.
Production build supports all heads; H=16, Hg=4 benchmark, device graph median:

| Tokens | Baseline us | New us | Speedup |
|---|---:|---:|---:|
| 128 | 104.892 | 93.739 | 1.119x |
| 256 | 141.561 | 125.622 | 1.127x |
| 1024 | 306.957 | 258.306 | 1.188x |
| 4096 | 863.501 | 779.279 | 1.108x |

Output relative RMSE remains ~0.00038; final-state relative RMSE ~0.00032.
Commands and setup are in memory.md. Raw results: baseline.json and
first_acceptance.json under benchmark_data/gdn_mega/ (local ignored artifacts).
The 3x target has not yet been reached.

## 6453a98 — host dispatch

Choose one templated kernel entrypoint per head count on the host. All seven
specializations remain available, but unused specializations no longer enlarge
the launched kernel. All 21 tests pass. Latencies: 76.919 / 98.894 / 228.839 /
750.282 us at T=128/256/1024/4096, respectively (1.36x / 1.43x / 1.34x / 1.15x
against the original baseline). Numerical errors unchanged.

## 4a68a0b — shared products, cube recurrence, and optional fused pipeline

Share KK/QK Gram products across grouped value heads; invert two 64x64
diagonal blocks and join them exactly; overlap WY and state-input preparation
with inversion; retain recurrent state on the cube. The default staged path
passed all 23 tests across all supported head counts on Ascend910B2.

The optional `GDN_FUSED_PIPELINE` path uses persistent head workers, two
producer/consumer slots, contiguous intermediate layouts, and L1/L0-resident
inverse/WY/state/output intermediates. It passed 16 selected H=16 tests,
including packed tails, initial states, weak gates, and repeated execution.
Other head counts have not yet been validated for this optional path.

H=16, Hg=4 device-graph medians (microseconds):

| Tokens | Original baseline | Default staged | Optional fused | Fused speedup |
|---|---:|---:|---:|---:|
| 128 | 104.892 | 52.708 | 39.185 | 2.68x |
| 256 | 141.561 | 75.084 | 51.217 | 2.76x |
| 1024 | 306.957 | 169.278 | 124.193 | 2.47x |
| 4096 | 863.501 | 553.351 | 411.824 | 2.10x |

Fused output relative RMSE is at most 0.000432 and final-state relative RMSE
at most 0.000389 on these benchmarks. Existing acceptance thresholds were
unchanged. Raw records: `benchmark_data/gdn_mega/output_acceptance.json` and
`benchmark_data/gdn_mega/fused_contiguous.json` (local ignored artifacts).
These measurements and device tests completed before the user paused work;
this commit records that validated checkpoint. The 3x target remains unmet.
The unfinished adaptive-inverse experiment and its unrun tests are excluded.
