> Current numerical/performance status: [stable checkpoint and final measurements](current_status.md). The measurements below are historical.

# GDN mega-kernel correctness audit — 2026-10-23

**The existing test passes did not establish pointwise accuracy.** I found no
reference-output leakage, seed-specific kernel branch, or test-specific device
path in the inspected production sources. I did find inadequate acceptance
criteria, an output-view pointer bug, and an optimized-kernel numerical regression.
The pointer bug is fixed. The numerical failures remain visible and unresolved.

The full updated suite reports **16 passed, 48 failed**: the original 32 cases
now give 1 pass / 31 failures; 20 new independent numerical cases give 3 passes /
17 failures; 12 checker, reference, pointer and workspace-reuse tests pass.
There are no `xfail` or skip annotations hiding the numerical failures.

## What was wrong with testing

1. `_assert_close` calculated maximum absolute error but only enforced relative
   RMSE < 0.005. An isolated bad element could pass. The new checker includes a
   mutation test where one error of 0.1 among a million unit values passes the
   old RMSE threshold and fails the new pointwise rule.
2. Nine head-configuration tests discarded final state. The correlated-key
   test had a separate RMSE-only checker and only tested beta 0 and 0.02.
   Both gaps are corrected.
3. The old FP32 reference used the same chunked inverse formulation as the
   optimized implementation. The new oracle is CPU FP64, one token at a time:
   decay state, compute `kᵀ state`, apply the delta update, then compute `qᵀ state`.
   It has no inverse, chunk truncation, or FP16 intermediate rounding. Analytic
   two-token and beta=1 correlated-key cases validate the oracle; three seeds
   cross-check the old reference against it on ordinary, weak- and zero-decay
   inputs. The reference never goes to the NPU or enters the kernel arguments.
4. The helper scales raw output by `1/sqrt(128)`. That is the existing attention
   convention, but it also shrinks reported absolute output errors by about
   11.314. The audit records both raw and scaled errors, plus unscaled state.
5. Zero-initialized workspaces and a narrow input distribution left blind spots.
   The audit covers new seeds, nonzero/absent initial state, packed empty
   sequences, tails, 8K/32K lengths, value/state amplitudes 0, 0.001, 1 and 32,
   correlated keys, beta 0/0.02/0.5/1, mixed gates, and additional head groupings.

## Acceptance and memory checks

Each raw output and final-state element must satisfy
`abs(actual - reference) <= 1e-5 + 1e-3 * abs(reference)`, and relative RMSE
must remain below 0.005. Shapes must match exactly, and both actual and reference
must be finite. This is the repository testing skill's FP16 default policy,
selected before observing the audit results; it is not a newly agreed model
accuracy budget. The absolute floor is 1e-5, not a universal maximum absolute
error of 1e-5. No tolerance was relaxed to obtain passes.

The audit reports maximum/mean absolute error, exact nearest-rank percentiles,
counts above 1e-4 and 1e-3, pointwise violations, worst-error values/index, RMSE,
and R². Every element participates; no fitting, clipping, discarded outliers,
or subsampled checks. Rounding the FP64 oracle directly to FP16 passes this
policy for all 20 cases, so the output dtype alone does not explain the failures.

Every case runs five times with all scratch and output buffers filled with
0, NaN, 17, -9 and NaN respectively. Inputs/constants are checked unchanged;
output/state allocations have prefix/suffix canaries; repeated outputs are
compared bit for bit. Statistics are reused only after that full bit comparison.
All 20 cases pass these memory/determinism checks after the wrapper fix,
including the cases whose numerical outputs are consistently nonfinite.
A separate test changes inputs and initial state while reusing workspace and
compares the result with a fresh allocation.

## Fixed pointer bug; unchanged arithmetic

`csrc/host/utils.h` passed `tensor.storage().data()` to kernels, which ignores a
contiguous view's storage offset. Guarded output views therefore wrote into the
prefix guard and returned shifted data. The wrapper now uses `tensor.data_ptr()`.
A regression test compares offset views with ordinary allocations bit for bit
over five launches and checks guards. It passes.

This fixes the shared host argument-conversion helper. No function signatures,
input/output dtypes, allocation sizes, or device arithmetic changed. Only the
GDN runtime path was tested here; other operators using this shared helper were
not rebuilt as device kernels or exercised. The rebuilt GDN device library's
SHA-256 is unchanged from the pre-audit binary:
`a1e1be91cfb087f8ab4f1fc1d4df9f185135fd672ba66c11780110533155ec60`.

## Numerical results

Measured on Ascend 910B2, CANN 9.1.0, torch/torch-npu 2.10.0, NPU 0.
The table shows the first of five identical repeats. Both outputs must pass;
for example, zero query gives exact zero output but still fails final state.
Full settings, hashes and per-repeat statistics are in [JSON](accuracy_audit.json);
the flat table is also saved as [CSV](accuracy_audit.csv).

| Case | T; H/Hg | Max abs raw output | Max abs scaled output | Max abs state | Raw relative RMSE | Pointwise violations: output / state | Result |
|---|---|---:|---:|---:|---:|---:|---|
| ordinary_seed2027 | 257; 16/4 | 0.000646875 | 5.71762e-05 | 0.000516337 | 0.000427549 | 14025 / 4632 | FAIL |
| weak_seed119 | 513; 16/4 | 0.00142906 | 0.000126313 | 0.00139174 | 0.000529721 | 278419 / 67547 | FAIL |
| no_decay | 257; 8/8 | 0.00169936 | 0.000150204 | 0.00210182 | 0.000481068 | 61058 / 33897 | FAIL |
| zero_state_tail | 129; 12/3 | 0.000478776 | 4.23182e-05 | 0.000457552 | 0.000432845 | 4988 / 1766 | FAIL |
| packed_empty | 257; 16/4 | 0.00135222 | 0.000119521 | 0.00136716 | 0.000500097 | 120604 / 129344 | FAIL |
| packed_empty_zero_state | 130; 8/2 | 0.000490942 | 4.33936e-05 | 0.000506999 | 0.000426316 | 3640 / 2541 | FAIL |
| large_values_and_state | 257; 8/2 | 0.0191199 | 0.00168998 | 0.0121574 | 0.000430708 | 31654 / 13520 | FAIL |
| tiny_values_and_state | 257; 8/2 | 5.6282e-07 | 4.97467e-08 | 4.05641e-07 | 0.000493795 | 0 / 0 | PASS |
| zero_beta_no_decay | 257; 8/2 | 0.000122034 | 1.07864e-05 | 0 | 0.000207356 | 0 / 0 | PASS |
| zero_query | 257; 8/2 | 0 | 0 | 0.000310088 | 0 | 0 / 2975 | FAIL |
| zero_values_and_state | 257; 8/2 | 0 | 0 | 0 | 0 | 0 / 0 | PASS |
| correlated_beta002 | 256; 8/2 | 0.000131008 | 1.15796e-05 | 0.000207643 | 0.000210796 | 291 / 271 | FAIL |
| correlated_beta05 | 256; 8/2 | nonfinite | nonfinite | nonfinite | nonfinite | all nonfinite | FAIL |
| correlated_beta1 | 256; 8/2 | nonfinite | nonfinite | nonfinite | nonfinite | all nonfinite | FAIL |
| mixed_decay | 513; 8/2 | 0.00133967 | 0.000118411 | 0.000574472 | 0.000501822 | 69473 / 1256 | FAIL |
| equal_heads64 | 129; 64/64 | 0.000787766 | 6.96293e-05 | 0.000533194 | 0.000428343 | 29362 / 24540 | FAIL |
| single_key_head | 129; 24/1 | 0.000716478 | 6.33283e-05 | 0.000537555 | 0.000434762 | 10423 / 8719 | FAIL |
| keys_norm2_stable_beta | 257; 8/2 | 0.000159305 | 1.40807e-05 | 0.000137727 | 0.000429695 | 2097 / 431 | FAIL |
| long_8k_weak | 8192; 8/2 | 0.00161894 | 0.000143095 | 0.000948407 | 0.000537543 | 2315395 / 32214 | FAIL |
| long_32k | 32768; 8/2 | 0.00136296 | 0.00012047 | 0.000501734 | 0.00042988 | 911222 / 1708 | FAIL |

## Nonfinite regression versus the original kernel

For identical exact unit keys `k=e₀`, zero log decay, and beta 0.5 or 1,
the optimized kernel produces nonfinite values in **all 262,144 output elements
and all 131,072 state elements**. The FP64 oracle stays finite. At beta=1 the
recurrence simply replaces the first state row with the current value vector;
the analytic oracle test verifies this directly.

I reran the same two cases against the frozen original `10748eb` device library,
using the fixed host wrapper and identical five poisoned-scratch repetitions.
The original stays finite and satisfies the old RMSE policy, although it also
fails the new pointwise criterion. This establishes a regression in the
optimized device algorithm, rather than just stricter testing:

| beta | Original max abs raw output | Original max abs state | Original output/state relative RMSE | Optimized |
|---:|---:|---:|---:|---|
| 0.5 | 0.00101662 | 0.00452194 | 0.000842936 / 0.00112923 | All output/state nonfinite |
| 1 | 0.00250371 | 0.0136719 | 0.00172505 / 0.00241278 | All output/state nonfinite |

[Original-kernel evidence and hashes](accuracy_baseline_correlated.json).
The full 64-term fallback does not make a polynomial inverse numerically safe.
For a 64×64 strictly lower matrix with entries `-beta`, the bottom-left entry
of `A^32` has magnitude `choose(62,31) * beta^32`: about 108 million at beta=0.5
and 4.65e17 at beta=1, above FP16's largest finite value 65,504. The implementation
stores intermediate powers in FP16. This is a concrete overflow mechanism
consistent with the observed regression, despite a bounded exact inverse.
The small-norm truncation bound does not bound intermediate rounding/overflow
or end-to-end output error. No device-internal tracing was added in this audit.

## Benchmark safeguards and historical results

The benchmark now enforces the pointwise/RMSE policy on raw/scaled output and
final state before timing and after graph replay. Failed cases retain structured
error statistics, get no latency or speedup, and return a nonzero exit code.
The sweep fingerprints the checker, reference and worker; an old RMSE-only
report cannot silently resume as a new-policy pass. The sweep also returns
failure if any comparison fails.

A one-cell H8/T128 comparison was run: both libraries failed the new policy;
both latencies and the speedup are null, and the sweep exited 1. A resume file
with its validation fingerprint removed was rejected with exit 2.
A one-token H8/Hg2 case with zero initial state passed the unchanged strict
policy, completed timing, and passed the post-replay check (2 trials × 2 calls).
The branch report preserves its historical RMSE-only results. The
[heatmap](heatmap.md) was refreshed on 2026-10-26 with the current stable kernel
and strict checks, compared against explicitly labeled historical original
latencies. The original measurements have not been reclassified as pointwise
passes; fresh original strict spot-check failures are preserved alongside it.

## Reproduce and recorded checks

Run from the repository root. The numerical audit and full suite are expected
to return exit 1 until the unresolved numerical issues are fixed.

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 600 python benchmarking/audit_gdn_mega.py \
  --output benchmark_data/gdn_mega/accuracy_audit.json
bash scripts/gdn_mega_docker.sh timeout 600 python -m pytest \
  tests/test_gdn_mega.py tests/test_gdn_mega_validation.py -q --tb=no
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest \
  tests/test_gdn_mega_validation.py -q -k 'not independent_recurrence_and_memory'
bash scripts/gdn_mega_docker.sh bash -c \
  'env LD_LIBRARY_PATH="/mounted_home/autokernel_gdn/pto-kernels/benchmark_data/gdn_mega/baseline_lib:${LD_LIBRARY_PATH:-}" timeout 120 python benchmarking/audit_gdn_mega.py --cases correlated_beta05 correlated_beta1 --output benchmark_data/gdn_mega/accuracy_baseline_correlated.json'
bash scripts/gdn_mega_docker.sh timeout 120 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib --heads 8 --lengths 128 \
  --output benchmark_data/gdn_mega/strict_sweep.json
bash scripts/gdn_mega_docker.sh timeout 120 python benchmarking/gdn_mega.py \
  --heads 8 --key-heads 2 --lengths 1 --trials 2 --inner 2 \
  --output benchmark_data/gdn_mega/strict_benchmark_one_token.json
```

Recorded: GDN-only build passed (48.27 s); full suite **16 passed / 48 failed**
(46.23 s); independent audit **3 passed / 17 failed**; original correlated-case
audit **0 pointwise passes / 2 failures**, both finite; pointer/checker/oracle/reuse
checks all pass. Python compilation, figure generation, and `git diff --check`
also pass. Logs report failures rather than changing tolerances or removing cases.

This is evidence from the listed cases, not proof over every input or device.
The unresolved nonfinite regression and pointwise errors need a numerical fix
before renewed performance acceptance. No new speed improvement is claimed.
