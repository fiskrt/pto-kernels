# GDN mega optimization notebook

Target: 3x faster than commit `10748eb`, with correctness preserved.
Hardware: Ascend910B2, 24 cube / 48 vector cores. CANN 9.1.0,
PyTorch / torch-npu 2.10.0. Container: `gdn-mega-opt`.

## Reproduction

```sh
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest -q tests/test_gdn_mega.py
bash scripts/gdn_mega_docker.sh timeout 120 python benchmarking/gdn_mega.py --output benchmark_data/gdn_mega/current.json
```

Build options: `GDN_MEGA_ONLY=ON` excludes unrelated kernels and bindings.
`GDN_MEGA_HEAD=16` compiles one head specialization for experiments (~12s);
clear it for full acceptance tests. Full compilation is ~35s, initial host
binding compilation ~44s. CMake reconfiguration adds ~10s; ordinary source
edits do not reconfigure. Fixed pre-existing CANN include-path loss on
reconfiguration and incorrectly shell-expanded `$ORIGIN` linker flags.

The benchmark allocates once and uses device graphs with 20 calls, 15 event
timing trials, five correctness/warmup calls. References run on CPU.
Warm-cache timing; report output AND state errors. Existing acceptance
tolerances are unchanged; actual observed errors are reported as well.

## Baseline

H=16, Hg=4, D=C=128; no initial state; normalized q/k, random value and beta,
logsigmoid gates. All 7 original tests pass. Expanded suite also passes all
17 tests, adding packed ragged sequences, lengths 1/127/129/257, initial state,
weak decay gates, and five consecutive calls.

| T | Baseline us | Parallel scan us | Scan + exact doubling us |
|---|---:|---:|---:|
| 128 | 104.892 | 80.348 | 76.662 |
| 256 | 141.561 | 108.560 | 104.495 |
| 1024 | 306.957 | 269.530 | 234.732 |
| 4096 | 863.501 | 827.078 | 759.152 |

Doubling measurements include optional stage-counter instrumentation.
Baseline binary and raw JSON retained under ignored `benchmark_data/gdn_mega/`.
Baseline output relative RMSE ~0.00038; final-state relative RMSE ~0.00032.

## Findings / experiments

- Original gate prefix sum has 128 serial row operations with extra row copies,
  then a global barrier and separate transpose pass. Seven-step parallel scan
  fuses both transposes and removes one global barrier. Verified on H=16
  expanded suite (14 selected tests); full head coverage remains pending.
- Original inverse constructs I and zero using matmuls and repeatedly copies
  X through X@I. Exact geometric doubling uses A^128=0, 14 matmuls, and retains
  the running X in FP32 L0C; no convergence assumption or truncation.
- Stage instrumentation (`GDN_STAGE_TIMING`) stores seven core-0 timestamps
  in the otherwise unused A_inv_f32 workspace. Differences include stage
  barriers; optional debug path, not production timing.
- At T=4096 after scan, stage tick counts: gates 917, KKT 5400, inverse 14150,
  WY 3389, recurrent state 10116, output 7331. Inverse is largest but not enough
  alone to deliver 3x. Need reduce data transfers, synchronization, and state
  critical path too.
- Two consecutive global barriers remain between inverse and WY (old cast
  stage removed but barrier retained).
- Compiler ASC language defaults to automatic synchronization in addition
  to explicit set/wait/barriers. Experiment with disabling automatic sync
  requires full correctness validation: some vector dependencies may currently
  rely on compiler-inserted synchronization.
- Potential next algorithm: block triangular inverse with TINSERT Acc->Mat,
  performing only nonzero recursive blocks instead of dense sparse matmuls.
- chunk_h serializes W@S, vector residual, Ktilde^T@Vnew, vector state update,
  with GM roundtrips on every chunk. State could remain resident in L1; vector
  preparation can be moved ahead of the recurrent dependency.

## First full acceptance

All 21 tests pass, including H=8/12/16/24/32/48/64 and packed tails.
Full-dispatch build timings: 93.739 / 125.622 / 258.306 / 779.279 us for
T=128/256/1024/4096. These are the honest production comparison: the
single-head experimental binary is ~20 us faster due to code/dispatch layout.
Future changes must measure full supported dispatch too. Investigate separate
kernel entrypoints per head, chosen by the host, to avoid putting all seven
large specializations in a single device entrypoint.
