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
