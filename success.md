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

## e900fcd — bounded adaptive inverse

`GDN_ADAPTIVE_INVERSE` measures the infinity norm of the actual FP16 triangular
matrix and selects 2/4/8/16/32 polynomial terms only when a geometric tail
bound is at most 3.2e-6 per diagonal block. Otherwise the exact 64-term block
inverse is retained. This is a truncation bound; floating-point rounding is
validated separately by reference comparisons.

Full-head build with `GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE` passed all 25
tests on Ascend910B2, including new zero-beta and correlated-key, zero-decay
cases, with five repeated executions. Benchmark medians at T=128/256/1024/4096:
39.143 / 49.481 / 112.795 / 372.280 us. The long case improves 9.6% over the
previous fused checkpoint; actual output/state error metrics are unchanged.
Raw results: `benchmark_data/gdn_mega/adaptive_acceptance.json`. No tolerance
relaxation; 3x target remains unmet. This path remains optional at this commit.

## 3a19c00 — joint output/state products

Stack output and next-state equations into two shared-RHS matrix products:
`[O; Snext] = [Qscaled; decayI] S + [attention; Ktilde^T] R`. This replaces four
separate GEMMs with two taller GEMMs, sharing RHS transfers and synchronization.
The algebra is unchanged and works with nonzero initial state. Optional flag:
`GDN_JOINT_PRODUCTS`, together with fused/adaptive flags.

18 selected H16 tests pass, including initial state, weak gates, packed tails,
and correlated keys. Initial-state benchmark: 39.382 / 50.061 / 111.211 /
359.178 us at T=128/256/1024/4096. Original initial-state baseline: 107.532 /
145.567 / 309.691 / 863.765 us (2.73x / 2.91x / 2.78x / 2.40x). Actual
output/state error remains approximately .00043/.00039. No tolerances changed.
Raw records: joint_output_initial.json and baseline_initial.json under
benchmark_data/gdn_mega/. All-head acceptance remains pending for this flag.

## d41e12a — in-place signed inverse

The vector preparation now emits -A directly. The cube inverts the two
diagonal blocks in place with +I and then overwrites the untouched lower
offdiagonal block with its inverse join. This removes a full128x128 GEMM
against -I and two intermediate copies per chunk. The inverse algebra and
measured numerical errors are unchanged. Enable with `GDN_NEGATIVE_A` in
addition to fused/adaptive/joint-product flags.

All 18 selected H16 tests pass. Initial-state medians: 38.915 / 48.909 /
104.472 / 331.824 us; T4096 improves 7.6% over the preceding joint-product
checkpoint and is 2.60x faster than the original initial-state baseline.
Raw results: benchmark_data/gdn_mega/negative_a_initial.json. Full-head
acceptance is still pending for these optional fused changes.

## f666654 — overlap operand loads with inversion

The in-place inverse frees L1 operand buffers early. Load W/U, scaled keys,
queries, attention, and decay during inversion; narrow inverse synchronization
to MTE1/cube/FIX instead of blocking independent MTE2 transfers.
`GDN_PREFETCH` requires `GDN_NEGATIVE_A`.

18 selected H16 tests pass. Initial-state medians: 37.458 / 45.992 / 96.432 /
299.105 us, with unchanged numerical errors. T4096 improves 9.9% over the
preceding checkpoint. Speedups over original initial-state baseline:
2.87x / 3.17x / 3.21x / 2.89x. Full-head acceptance is pending. Raw results:
benchmark_data/gdn_mega/prefetch_initial.json.

## 3eae108 — adaptive-depth publication fix and strict full acceptance

A stricter reference check exposed a race in scalar depth metadata publication.
The vector's dcci cache flush lacked dsb(DSB_DDR) before the ready notification;
the cube could consequently use an insufficient depth for a short packed tail.
Forcing64 terms removed the failure. The affected matrices have row norms
2.7–3.6 and should always select64, so this was an implementation bug rather
than a legitimate bounded truncation. Added volatile metadata accesses and
DDR fences. Removed the dead initial-state identity product for nonempty
sequences.

All31 tests pass in the full-head build, including50 repeated executions of
the weak-decay packed regression. Every reference comparison now requires
finite results and relativeRMSE<0.005; the previous0.01 absolute-error early
exit and5% relative threshold are removed. Earlier entries reported passes
under those older checks; this acceptance supersedes them.

With initial state:39.651/48.429/98.719/301.066us atT128/256/1024/4096.
Without initial state:38.417/47.582/97.472/299.847us. All optional fused flags
(fused/adaptive/joint-products/negative-A/prefetch) are included. Output RMSE
~0.00043, state~0.00039 on benchmark inputs. Raw records:fenced_acceptance*.json
under benchmark_data/gdn_mega/. 3x is not yet achieved across all lengths.

## 9e91150 — reuse resident operands

Reuse S already resident in L0B after W*S. When a chunk's stored half decay
matrix is exactly zero and the preceding L1 copy is also zero, retain that
L1 copy instead of reloading it. This does not assume S is zero and introduces
no additional approximation. L0A must still reload decay because the second
joint product overwrites that region with Ktilde^T.

25 selected strict H16 tests pass, including a new test alternating finite
and zero half-decay matrices. Initial-state medians:37.468/46.568/95.733/
292.657us; actual errors unchanged. T4096 is2.95x faster than the original
initial-state baseline. Flag:GDN_REUSE_OPERANDS (requires adaptive metadata
and joint products). Raw results:benchmark_data/gdn_mega/reuse_initial.json.

## 073ae91 — share prefix exponentials and simplify decay construction

W, Q, and final decay reuse one exp(g) vector. Construct decayI directly in
half precision instead of converting the entire identity to float and back;
fill zero directly when its half value is zero. A vector-to-scalar event
orders exp(g) before short-tail scalar reads. exp(g_i-g_j) remains unchanged.
25 selected strict tests pass. Initial-state medians:36.726/45.199/94.346/
292.951us; shorter cases improve, long case is unchanged. Actual errors are
unchanged. Raw results:benchmark_data/gdn_mega/vector_trim_initial.json.

## 28929db — omit products with exactly zero half decay

When exp(g_last) rounds to FP16 zero, omit decay-matrix construction/load
and the full S*0 cube product. Use three ordinary products for output/state;
retain joint products when decay is nonzero. This supports nonzero incoming
state and adds no approximation to the existing half-decay calculation.
25 selected strict H16 tests pass, including changing decay and the repeated
packed regression. Initial-state medians:35.877/44.158/91.991/285.193us.
Speedups:2.997x/3.296x/3.367x/3.029x. T128 remains just below target.
Raw results:benchmark_data/gdn_mega/skip_decay_initial.json.
Validation: bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh;
then timeout 120 python -m pytest tests/test_gdn_mega.py -q -k
'not head_configs or H16'; then timeout 120 python benchmarking/gdn_mega.py
--initial-state --output benchmark_data/gdn_mega/skip_decay_initial.json.

## b315f03 — overlap first chunk preparation with cube setup

Seed the first producer credits before loading identity/state, since vector
preparation has no dependency on those cube tiles. Subsequent head credits
retain their existing ordering. 25 selected strict tests pass.
Initial-state medians:35.337/42.725/90.698/283.634us, exceeding3x against the
original initial-state baseline at all four lengths. Full-head default-build
acceptance remains pending. Raw:benchmark_data/gdn_mega/early_initial.json.
Commands match preceding entry with GDN_EARLY_PREPARE added to definitions.

## f71a883 — enable validated fused pipeline by default

Full production build (all head specializations, empty experiment definitions)
passes all32 strict tests on Ascend910B2. Initial-state medians:
36.354/44.813/92.734/284.608us; no-state:35.750/43.582/91.639/285.068us.
Unlike the earlier single-head experimental build, this production measurement
falls short of3x atT128; long-sequence speedup remains above3x. No claim of
universal3x is made. Raw results:final_initial.json,final_no_initial.json.
Commands: cmake -S . -B build -DGDN_MEGA_HEAD= -DGDN_MEGA_DEFINITIONS=
-DGDN_MEGA_MANUAL_SYNC=OFF; bash scripts/build_gdn_mega.sh; timeout120
python -m pytest tests/test_gdn_mega.py -q; benchmark with/without --initial-state.
User requests the next structural optimization target: another2x from this
checkpoint, emphasizing cube utilization, global pipeline, and movement reuse.
Frozen library:benchmark_data/gdn_mega/structural_baseline_lib/.

## 1290ec8 — pipeline the six recurrence products

Use two L0A/L0B tiles and explicit producer/consumer dependencies to overlap
Q*S with residual export, load subsequent operands during accumulator transfers,
and overlap state export with the final attention product. Arithmetic is
unchanged. GDN_PIPELINED_RECURRENCE selects the new zero-half-decay branch;
finite-decay and first zero-state chunks keep the existing path.
25 selected strict H16 tests pass. Initial-state medians:
33.705/41.230/85.323/261.891us. T4096 improves8.0% versus the production
structural baseline284.608us; this is1.087x, not the requested additional2x.
Numerical errors are unchanged. Raw:benchmark_data/gdn_mega/recurrence_initial.json.
Commands: configure HEAD=16, DEFINITIONS=GDN_PIPELINED_RECURRENCE; build script;
timeout120 pytest tests/test_gdn_mega.py -q -k 'not head_configs or H16';
timeout120 python benchmarking/gdn_mega.py --initial-state --output above.

## a8fe7ff — interleave independent inverse solves

GDN_INTERLEAVED_INVERSE queues the two diagonal solves together, overlapping
cube products with FIX exports using separate operand/accumulator tiles. It
uses exactly the same polynomial terms and rounding; no decay threshold is
added. All25 selected strict tests pass. With gate_scale0.01 and nonzero initial
state, T1024/4096 improve105.278/337.722us to99.331/308.240us on the same npu:0
(1.060x/1.096x). Ordinary-gate timings36.052/42.528/86.675/264.372us are slightly
slower than the preceding recurrence-only build, so this remains optional.
Actual output and state errors are unchanged. Raw files:interleaved_initial,
interleaved_weak,structural_baseline_weak_npu0.json under benchmark_data/gdn_mega.
Build:HEAD=16,DEFINITIONS=GDN_PIPELINED_RECURRENCE;GDN_INTERLEAVED_INVERSE,
MANUAL_SYNC=OFF. Same selected pytest and benchmark commands; benchmark now
accepts --gate-scale0.01 (with a space before the numeric argument).

Added a diagnostic GDN_MATMUL_PROBE build and benchmarking/gdn_matmul_probe.py.
All24 cores return exact identity products across five runs. For512 products,
median resident/reload/feedback ticks:64x64=913/4310/5748;
128x128=7284/15800/18676. Operand reload plus feedback costs6.30x and2.56x
resident arithmetic respectively. Probe command: configure DEFINITIONS=
GDN_MATMUL_PROBE, build, timeout120 python benchmarking/gdn_matmul_probe.py.
This diagnostic replaces GDN execution and must not be used for GDN results.

## 1e9fee7 — extend recurrence scheduling to finite decay

The pipelined recurrence now includes the incoming-state decay product when
its coefficient is nonzero. It no longer selects the optimized schedule based
on decay. Track the final accumulator layout explicitly instead of inferring
it from the coefficient. No additional approximation or term truncation.
25 selected strict tests pass; numerical errors unchanged. Initial-state
T128/4096:32.934/260.210us. Weak-decay T1024/4096:93.816/286.148us, improving
7.2% at4096 over308.240us before generalizing the schedule, and18.0% speedup
(1.180x) versus337.722us structural baseline. Ordinary4096 is1.094x versus
284.608us. The additional2x target remains unmet.
Raw:general_initial.json,general_weak.json. Build and validation commands match
preceding entry. Disabling compiler auto-sync did not materially help; restored
MANUAL_SYNC=OFF. Full-head acceptance for these optional schedules is pending.

## 3e4c336 — production acceptance of general matrix schedules

Enable pipelined recurrence and interleaved inverse by default. Full production
build (all head specializations, no experiment definitions) passes all32 strict
tests. Initial-state T128/256/1024/4096:33.582/40.720/85.264/263.232us.
Weak-decay T1024/4096:91.686/282.969us. Errors unchanged; ordinary output RMSE
~0.00043, weak~0.00054, both below0.005. Ordinary4096 is1.081x and weak4096
1.193x versus the new structural checkpoint; another2x is still the goal.
Raw:general_full_initial.json,general_full_weak.json. Configure HEAD empty,
DEFINITIONS empty,MANUAL_SYNC=OFF; build; timeout120 pytest -q
 tests/test_gdn_mega.py; benchmark with --initial-state and optionally
--gate-scale 0.01. Added validated store probe:contiguous/strided/L1 ticks
3831/5506/2048 for256 stores (24-core diagnostic). Changing output layout is
not justified by this modest penalty. Rejected RHS-first reassociation is
not included in production.

## 8cfbfa4 — overlap shared Gram transfers and products

Pipeline shared K*K^T and Q*K^T across tiles: prefetch the next K/Q pair while
current products run, retain K^T, and export the first result during the second
product. The single-wave path keeps its shorter setup. Arithmetic, grouped-head
reuse, decay handling, and error thresholds are unchanged.

All 32 strict tests pass in the full-head production build. Fresh same-device
before/after T4096 initial-state medians: ordinary 261.734 -> 254.176 us (1.030x),
weak decay 283.597 -> 276.511 us (1.026x). T128 is 32.402 us versus 32.896 us.
An earlier A/B run without single-wave dispatch confirmed long-sequence gains
but regressed startup; that regression is avoided by the final dispatch.
Raw results: gram_before_initial.json, gram_before_weak.json,
gram_dispatch_initial.json, gram_dispatch_weak.json under benchmark_data/gdn_mega.

Validation: configure GDN_MEGA_HEAD and GDN_MEGA_DEFINITIONS empty,
GDN_MEGA_MANUAL_SYNC=OFF; `bash scripts/build_gdn_mega.sh`;
`timeout 120 python -m pytest tests/test_gdn_mega.py -q`;
`timeout 120 python benchmarking/gdn_mega.py --initial-state --lengths 128 4096`
and `--initial-state --gate-scale 0.01 --lengths 4096`, all through
`scripts/gdn_mega_docker.sh`. The additional 2x target remains unmet.

Final revalidation of 8cfbfa4 after rejecting all further experiments: full-head
production build passes all32 tests. Initial-state T128/256/1024/4096 medians
32.912/40.444/83.016/254.862us; no-state33.323/40.772/83.348/253.609us.
Weak-decay initial-state T1024/4096:90.317/276.833us. Initial-state gains versus
original10748eb are3.267x/3.599x/3.730x/3.389x. Gains versus the newer f71a883
checkpoint at4096 are1.117x ordinary and1.220x weak; the further2x is not achieved.
Raw files:production_final_initial.json,production_final_no_initial.json,
production_final_weak.json. No relaxed checks or experimental compile flags.

Real-device profile of this exact production build:
`msprof op --kernel-name="*gdn_mega_specialized*" --aic-metrics=Roofline
--output=benchmark_data/gdn_mega/device_gram_pipeline python benchmarking/gdn_mega.py
--initial-state --profile-once --lengths 4096` (source CANN first, timeout180).
Profile OPPROF_20261022172022_WQDJDGCBKTHCAPFG; active-core medians: cube busy31.06%,
MTE1 busy26.67%, FIX busy21.26%; cube wait54.57%, MTE1 wait51.50%. The recurrence
still has substantial transfer/dependency stalls. Larger core-splitting,
column-tiling, direct-load, and deferred-store experiments are recorded with
rejection evidence in memory.md; none are in production.

## 5141f56 — production cleanup and saved 8K/16K/32K comparison

Removed private legacy mega stages, unused inverse variants, matmul_probe and
its runner, timing/early-exit hooks, and algorithm tuning preprocessor switches.
The validated production schedule is now unconditional; architecture guards and
runtime shape/numerical decisions remain. Removed the experiment-only CMake
options; retained GDN_MEGA_ONLY for fast builds. Mega source drops from 19 files /
7,126 lines to seven used files / 1,104 lines (84.5% fewer). One launch contains
two global phases: concurrent gate/layout + shared KK/QK, then persistent
preparation/inverse/WY/state/output. Source map and buffer roles are documented.

Public signature, shapes, dtypes, and allocation sizes are unchanged. Corrected
host comments: named intermediate buffers are opaque scratch; reserved arguments
remain for compatibility. This is a maintainability improvement, not a claimed
new algorithmic speedup. Ordinary latency varies -2.75% to +1.16% against 9ea497c,
and its reported numerical errors match the pre-cleanup checkpoint.

Final full-head production build: 32 tests passed in 5.85 s on Ascend910B2,
CANN9.1.0 / torch and torch-npu2.10.0, NPU0. Benchmarking checks output and final
state against the CPU FP32 reference for five launches per case, with unchanged
finite-value and relative-RMSE <0.005 requirements. Ordinary lengths
128/256/1024/4096/8192/16384/32768 and weak-decay lengths4096/8192/16384/32768 all
pass with a nonzero initial state. No relaxed checks. Source include/guard audit,
Python syntax, report links, and git diff --check pass.

Original10748eb -> cleaned5141f56 medians (ms), H16/Hg4,D=C128:

| Tokens | Ordinary original | Ordinary cleaned | Speedup | Weak original | Weak cleaned | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0.8688 | 0.2576 | 3.37x | 0.8663 | 0.2780 | 3.12x |
| 8192 | 1.7197 | 0.5346 | 3.22x | 1.6848 | 0.5826 | 2.89x |
| 16384 | 3.5022 | 1.0938 | 3.20x | 3.3755 | 1.1761 | 2.87x |
| 32768 | 6.9650 | 2.2583 | 3.08x | 6.8061 | 2.4120 | 2.82x |

The weak-decay long-sequence results do not meet a universal3x target. The 32K
ordinary output/state relative RMSE is0.000430/0.000381; weak is0.000538/0.000487.

Validation commands, through scripts/gdn_mega_docker.sh:

```bash
bash scripts/build_gdn_mega.sh
timeout 120 python -m pytest tests/test_gdn_mega.py -q
timeout 300 python benchmarking/gdn_mega.py --initial-state \
  --lengths 128 256 1024 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/branch_report_20261023/cleanup_initial.json
timeout 300 python benchmarking/gdn_mega.py --initial-state --gate-scale 0.01 \
  --lengths 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/branch_report_20261023/cleanup_weak.json
```

Tracked report and all tables: docs/gdn_mega/optimization_report.md,
checkpoint_latency.csv, checkpoint_stages.csv, production_latency.csv, and
measurements.json. JSON preserves raw timing/error metrics, environment, dense
matrix FLOP accounting, and measured kernel-library SHA-256 fingerprints.

## 2a5cce3 — checked shape sweep, heatmap, and execution explanation

Added a reusable benchmark sweep and a separate CPU plotting script. The sweep
varies H=8/12/16/24/32/48/64 and T=128/256/1024/4096/8192/16384/32768 with fixed
GQA group4, D=C128, ordinary logsigmoid gates, seed0, and nonzero initial states.
Original10748eb and cleaned5141f56 libraries run in separate processes. Their
actually loaded SHA-256 fingerprints match the saved branch-report artifacts.
Each passing implementation/shape checks output and final state over five
warmups and after timed graph replay; medians use15trials of20calls. Allocation
and reference checks are outside timing. The runner releases per-shape graphs
and tensors, checkpoints results, and supports validated resume.

All49 optimized shapes pass (largest output/state relative RMSE0.000434155,
threshold0.005 unchanged). There are48 valid speedups ranging2.90654x–7.52798x.
The original baseline fails output correctness atH64/Hg16/T32768 in two processes:
relative RMSE0.087156 then0.087424. Its cell is explicitly gray/failed, without
baseline latency or a speedup. The optimized kernel passes that shape at7.51260ms,
with output/state relative RMSE0.000430010/0.000377453. The baseline failure's
cause has not been diagnosed; no tolerance relaxation or baseline modification.

Validated real-device normal/failure paths, matching CSV/JSON shape grids and
ratios, resume without retiming completed rows, rejection of incomplete plotting
grids, PNG visual inspection, SVG XML parsing, Python syntax, and git diff
--check. No production kernel or interface changes. Existing32-test acceptance
remains in the previous entry; this change adds the larger benchmark coverage.

Commands (repository root):

```bash
bash scripts/gdn_mega_docker.sh timeout 1800 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib \
  --output docs/gdn_mega/heatmap.json
bash scripts/gdn_mega_docker.sh timeout 900 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib \
  --output docs/gdn_mega/heatmap.json --resume
bash scripts/gdn_mega_docker.sh python benchmarking/plot_gdn_mega.py \
  docs/gdn_mega/heatmap.json
```

Saved docs/gdn_mega/heatmap.{png,svg,csv,json,md}. The Markdown explains executed
GFLOPs versus TFLOP/s, the two global phases inside one launch, head-persistent
chunk processing, and its head-count scheduling implications. Initial sweep
stopped on the baseline numerical failure; the runner now records numerical
failures explicitly and the validated resume completed the remaining row.

## 2026-10-23 — correctness audit and tested tensor-view fix

Implementation/evidence commit: **d1396f9** (`Audit GDN pointwise accuracy and
expose inverse overflow regression`). This entry records a testing improvement
and a verified host-pointer fix, **not numerical acceptance or a new speedup**.

The old checker printed maximum absolute error but enforced only relative RMSE;
nine head-configuration tests omitted final state. The suite now checks every
element using abs(error) <= 1e-5 + 1e-3 * abs(reference), plus RMSE < 0.005,
finite values and exact shapes. New CPU FP64 token recurrence avoids the shared
chunked/inverse reference and reports raw output, attention-scaled output, and
final-state errors separately. No tolerance relaxation, skip or xfail hides the
failures. No expected-output leakage or test/seed-specific production branch
was found in the inspected source.

Guarded output views exposed a real pointer bug: shared host conversion used
storage().data() and ignored a tensor's storage offset. Changed it to data_ptr().
The GDN-only build passes, the contiguous-view regression passes five launches,
and the independent checker/oracle/workspace-reuse checks pass. Device arithmetic,
ABI and required allocation sizes are unchanged. Only the GDN runtime path of
this shared wrapper fix was tested. The device library hash remains
`a1e1be91cfb087f8ab4f1fc1d4df9f185135fd672ba66c11780110533155ec60`.

**Actual results:** full suite **16 passed / 48 failed** in 46.23 s, exit 1.
Original cases: 1/32 pass under the stricter policy. New independent cases:
3/20 pass; all 20 pass five-repeat input-integrity, output-canary and poisoned-
scratch determinism checks after the pointer fix. The remaining 12 checker,
reference, pointer and reused-workspace tests pass. Rounding the independent
oracle directly to FP16 passes for all 20 numerical cases.

The optimized kernel returns entirely nonfinite output/state for identical unit
keys, no decay, beta 0.5 or 1. The original 10748eb library stays finite on the
same two cases and passes the old RMSE criterion, although it fails the new
pointwise criterion. This is a real optimized-device regression, consistent with
FP16 inverse powers overflowing despite the full 64-term fallback. It is still
unfixed. Typical finite cases also violate pointwise tolerances: T32768/H8 raw
output max abs error 0.00136296; weak-decay T8192/H8 max abs error 0.00161894.

Saved the full findings, all 20 case tables, source/library hashes and original-
kernel comparison in `docs/gdn_mega/accuracy_audit.{md,json,csv}` and
`docs/gdn_mega/accuracy_baseline_correlated.json`. Previous report/heatmap passes
are now explicitly labeled historical RMSE-only acceptance, including the figure.
The benchmark rejects pointwise failures before timing and validates graph replay;
the sweep fingerprints validation code and rejects incompatible resume artifacts.
Verified H8/T128 failure handling: null latencies/speedup and exit 1. Removing a
resume validation fingerprint produces exit 2. A passing H8/Hg2/T1 case with zero
initial state completes timing and post-replay validation without changing policy.
Python syntax, SVG parsing/visual inspection and git diff --check pass.

Reproduction commands and detailed failure metrics are in
`docs/gdn_mega/accuracy_audit.md`. Numerical corrections are required before
claiming correctness-qualified performance improvements again.

## 2026-10-23 — first complete numerical correction

Commit **41f0549** replaces inverse powers with stable FP32 forward substitution
and preserves FP32 Gram products, gated factors, WY intermediates, residuals and
recurrent state. Only public output/final-state stores round to FP16. Internal
32-token tiles and reuse of consumed producer slots preserve the existing ABI
and allocation sizes. A2 Acc-to-L1 copies narrow to FP16/BF16, so exact FP32
intermediate transfers use scratch export/reload with the slot held until done.

Validation: `timeout 180 python -m pytest tests/test_gdn_mega.py
 tests/test_gdn_mega_validation.py -q --tb=short` inside the prepared Docker
container: **64 passed in 72.14 s**, unchanged tolerances, including all 20
independent numerical cases and 32K tokens. The previously nonfinite beta 0.5/1
cases now pass; beta=1 final state is exact. Build passes in 21.79 s.

Initial correctness-qualified H16/Hg4 timing with nonzero initial state:
T128 **0.068475 ms**, T4096 **1.259212 ms** (15 trials × 20 graph calls).
This is slower than the historical FP16 optimized path; no speedup is claimed.
Raw results: `benchmark_data/gdn_mega/fp32_first_benchmark.json`.
Further work targets the dense scalar-decay product and FP32 transfer overhead,
without relaxing the numerical policy.

## 2026-10-23 — paired FP16 stable checkpoint

Commit **ff86de2**: stable block inverse, paired FP16 products with FP32 state,
overlapped inverse preparation, and compensated restart for strongly coupled heads.
**70/70 tests pass** with unchanged strict absolute/relative criteria.
H16/Hg4, nonzero initial state, 4K: **0.968760 ms**, versus correct FP32
1.259212 ms (**1.30×**). 8K/16K/32K: 1.915925/3.907988/7.935392 ms.
ABI/dtypes/caller allocation sizes unchanged. Strong-coupling fallback can be slow.
Historical ~0.87 ms original and ~0.25 ms inaccurate FP16 timings are not beaten;
the requested target remains unmet. Full evidence: docs/gdn_mega/paired_checkpoint/.

## 2026-10-23 — verified hybrid and FLA range audit

Commit **1f5e19e** keeps FP16 tensor products for original K/Q times paired
state/residual, while native FP32 handles inverse joins, residual solve and
attention. This removes inverse pair materialization and identity products.
Invalid rows are masked before exponentiation, fixing padded-prefix 0*infinity
NaNs discovered by the full suite. Public ABI/allocation sizes are unchanged.

**80 tests pass** in 89.90 s with unchanged tolerances; the independent saved
FP64 audit passes **36/36 cases × 5 poisoned-workspace repeats**. This includes
FLA distributions, beta's (0,2) mode, and new 75/92-token strong-decay regressions.
All output/state pointwise violation counts are zero. Large-amplitude cases can
still have ~0.03125 absolute error from final FP16 rounding; detailed per-case
errors are saved, not hidden by RMSE.

H16/Hg4 with nonzero initial state: 4K **0.905793 ms**, versus paired checkpoint
0.968760 ms (**1.070×**) and correct FP32 checkpoint 1.259212 ms (**1.390×**).
8K/16K/32K: **1.783001 / 3.615756 / 7.520225 ms**. The historical original
~0.87 ms and inaccurate ~0.25 ms FP16 timing are still not beaten. Target unmet.
FLA positive-key zero/weak-decay cases remain slow due to compensated recurrence;
these timings are explicitly saved in docs/gdn_mega/hybrid_checkpoint.md.

Input-range derivation, measured inverse-power growth (4.76e18 for FLA-style
positive/no-decay keys despite inverse maxabs 1), rejected major algorithm
experiments, raw timings and accuracy evidence are in docs/gdn_mega/.

2026-10-23 — `a329112` — shared-operand projection and residual pipeline.
All80 strict tests pass after correcting scalar-cache invalidation for reused
coupling flags. Preweight K independently, then share R between output/state
products; remove the dependent residual-scale exchange and three flags per chunk.
H16/Hg4, nonzero initial state,128/4K/8K/16K/32K:
0.069667/0.844893/1.672497/3.392108/6.872840ms. At4K1.072× vs prior stable
0.905793ms,1.490× vs correct FP32 predecessor1.259212ms. The128-token case
regresses; conservative strong-coupling fallback remains slow.3× target unmet.
ABI/dtypes/allocation sizes unchanged. Full evidence and failed/fixed regression
records: docs/gdn_mega/fused_checkpoint.md and adjacent JSON/logs.

2026-10-23 — `bfc99e1` — share the prepared decay matrix between QK and KK
weighting. Removes a redundant full matrix construction/exp; keeps G_i-G_j and
every inverse term. All80 strict tests pass. H16/Hg4/4K with initial state:
0.794724ms,1.063× faster than the fused checkpoint0.844893ms,1.583× faster than
the correct FP32 predecessor1.259212ms.3× target still unmet. ABI/allocation
sizes unchanged. Raw records: docs/gdn_mega/shared_decay_benchmark.json and
shared_decay_tests.txt. Batched-small-join variant was slower and was removed.

2026-10-26 — `6fb5757` — refreshed the original/current stable heatmaps and
removed the obsolete FP16 speedup summary. All 49 current shape measurements
pass strict checks; 48 historical-original latency ratios range from 1.04× to
3.85×, or 1.04× to 1.54× at 4K and longer. The original's failed H64/32K cell
remains gray. Baseline timings are explicitly historical RMSE-only measurements,
not relabeled strict passes. Tooling `f071140` adds guarded historical-baseline
reuse and validated byte-equality statistics caching; six checker tests and
plot/resume integrity checks pass. No production kernel or tolerance changed.
Evidence: docs/gdn_mega/heatmap.md and heatmap_validation.txt.

2026-10-26 — matched original/current accuracy heatmaps (measurement tooling
`8e5e7f0`). Fresh 49-cell H/T sweep, identical input/reference fingerprints,
three NaN-initialized output/final-state launches per implementation. Current
`bfc99e1` passes49/49 with bitwise identical repeats; original `10748eb` passes0/49.
Current max absolute output error2.3705e-4–4.8440e-4, worst tolerance ratio≤0.47921;
47/49 maxima equal the FP16 output rounding floor. Original H64 at8K/16K/32K
also has repeat-dependent large errors (maxima0.55745/0.58078/0.50206), retained
in the plots; their cause is not established. Independent FP64 H16/128 and4K
crosschecks: current2/2, original0/2. Twelve CPU checker tests and six plot
integrity checks pass. Saved both shared-scale plots, all repeats/CSV, explanation
and validation in docs/gdn_mega/error_heatmap*. No kernel, tolerance, interface,
allocation, or performance change is claimed by this reporting improvement.

2026-10-26 — saved fresh original/current hardware profiles and diagnostic
phase measurements in docs/gdn_mega/profile_comparison.md. Production current
strict checks pass, as do both current diagnostic accuracy checks and the T256
simulator check. Production remains unchanged; this is a profiling result, not
a new speedup. Original890.307us/current794.432us (1.121x); shared setup34.20us,
persistent phase768.41us. Identified vector preparation, dependent FP32 inverse
scratch round trips,129us state feedback waits and16/24 active persistent groups.
Raw counters, clock samples, reproducible diagnostic patch and traceviz timeline
are saved. No tolerance/ABI/allocation change.3x target remains unmet.

2026-10-26 — `aac5a4d`: reduce stable GDN solve serialization and overlap
next-chunk preparation. Queue independent FP32 inverse joins by level; gather
triangular-solve coefficients directly and broadcast the solved row through
vector instruction strides; retain pending weights in UB before projection
acknowledgement; transfer whole FP32 attention/Gamma tiles in the existing
allocations. All seven production source files remain used, with no new tuning
options, entry arguments, allocations, precision shortcuts or global stages.
Production GDN-only build succeeded; all80 strict tests passed in92.18s, including
FP64/poisoned-workspace audits. Fresh H16/Hg4/nonzero-state production graph sweep
(9x20) passes7/7 shapes: T256 .094064->.086807ms;512 .141884->.127788;
1024 .235302->.211988;4096 .803037->.720410;8192 1.577844->1.418542;
16384 3.208187->2.833265;32768 6.528661->5.766021. These are1.084–1.132x
speedups against the previous *correct* production kernel, not the inaccurate
historical original. The screened isolated candidate reached .701ms at4K;
report the cleaned production build's .720ms instead. Library SHA256:
619b7df351e839f58d853d3ca639cafca44127a18e270626dfe2ddc79f5ed89d.
Raw measurements, numerical errors and rejected structural experiments are in
docs/gdn_mega/optimization_20261026.json/csv and the accompanying Markdown report.
The requested3x target is still unmet; this is a validated improvement, not a
claim of the requested2x/3x breakthrough.
Additional matched4K checks for `aac5a4d`: H8/Hg2 .787242->.680237ms,
H24/Hg6 .826423->.718642ms, H64/Hg16 2.511402->2.144606ms, all strict checks pass.
Fresh hardware profiler task duration794.432->700.678us is recorded separately
from graph timing. Later two-chunk lookahead passed its initial checks but did
not improve latency and was not retained.

2026-10-26 — refreshed and consolidated the reports for kernel `aac5a4d`.
Fresh latency grid:49/49 current cells pass elementwise/finiteness/RMSE checks
before and after timing;48 historical-original ratios,1.18–4.12x overall and
1.18–1.71x at4K+. Original64-head/32K timing failure remains gray.
Fresh matched error grid: current49/49, original0/49; all elements and3 launches
per shape retained. Independent FP64 crosschecks: current2/2, original0/2.
Zero current raw-output/final-state elementwise violations. The unchanged kernel
and checker hashes match the saved80-test production pass;12 report/checker tests
also pass. No tolerances or kernel implementation changed.
`bash scripts/report_gdn_mega.sh all` measures both reports; `render` regenerates
Markdown, PNG and SVG from saved JSON. Generated pages and plots now use the same
data. docs/gdn_mega contains only README.md, heatmap.md, error_heatmap.md and
input_ranges.md, plus their data/figures. Removed obsolete checkpoint/experiment
reports and replaced the solver-screening script with a focused input-range
script. Historical experiments remain available in Git at a83187b. Complete-grid,
source/library fingerprint, repeat/bounds and local-link checks pass; figures
were visually inspected. Evidence is consolidated in data/validation.json.

2026-10-26 — `e9714dd`: faster exhaustive validation and independent FP64 grid
references. At T4096/H16/Hg4, median first-check CPU wall time falls from
1.379s to0.222s (6.22x), by making percentile sorting optional and checking raw
output directly instead of duplicating scaled statistics. Every element,
finiteness, maximum absolute error and final-state bound remain checked; policy
unchanged. FP64 scratch reuse gives3.182->2.995s on the same case, with identical
output/state. Changed outputs, changed or mutated references, NaNs, and altered
shared-golden cache contents are covered by regression tests. The85 kernel/oracle
checks and24 checker/report checks pass across the complete and targeted runs.
Explicit24/8 and48/16 rows are supported; every heatmap cell displays current
median total microseconds beneath its speedup/failure. All63 current latency
cells pass FP64 elementwise checks through32K. This improves validation speed,
not production kernel speed; kernel aac5a4d and its ABI are unchanged.

2026-10-26 — follow-up to `e9714dd`: refreshed the full63-cell latency/error
reports with FP64 goldens, including24/8 and48/16 through32K. Current63/63 pass;
original0/63 pass the elementwise policy, with every element and3 launches per
error cell retained. Timed the15 previously missing original cells despite
numerical failure, preserving their failed statuses. All63 runtime ratios are
now available; stars are decided per cell and implementation, and current
median total microseconds appear beneath each ratio. Passing entries are never
starred just because another entry fails. Runtime tables retain failing baseline
timings; these comparisons are explicitly not accuracy equivalent. The final
suite passes115 tests in58.79s, including mixed pass/fail annotation cases and
protection against reusing a permissive failed result as a strict pass. The
single-cell sweep integration check confirms failed baseline timing and a
passing current result. Verified all63 displayed current medians, input/oracle/
library/source fingerprints, complete grids, zero current violations, local
links and plots. Removed redundant selected-shape FP64 reports; maintained docs
remain four Markdown pages. Kernel code and numerical tolerances are unchanged.

2026-10-26 — heatmap display correction: removed asterisks from runtime ratios.
Each cell now explicitly labels our median total microseconds and the baseline's
median total microseconds. Only a numerically failing implementation's own time
receives a marker. Verified63 unmarked ratios,63 unmarked current times and63
marked failing baseline times against saved measurements. The5 mixed pass/fail
annotation tests pass; PNG visually inspected. Numerical data and kernel are
unchanged. Full-suite evidence remains tied to e7c916d; display checks are saved
separately in data/validation.json.

2026-10-26 — audit tooling commits0a1797e,810b731,f2019b9 add beta-bounded stress
cases, matched controls and exact worst-tolerance witnesses. The checker tests
pass31/31 and the existing regression suite passes116/116. This is NOT a kernel
precision success: the additional56 cases uncover23 unresolved failures within
beta[0,1], all missed by RMSE-only acceptance. Production kernel unchanged.
Results and reproducible failing cases are saved in docs/gdn_mega/README.md and
data/precision_audit.json/csv; the failures are retained without xfail markers,
filtered outliers or relaxed tolerances. The critical audit exits nonzero.

2026-10-26 — correlation audit tooling (`f89c376`, expanded in the accompanying
report commit): controlled key-correlation sweeps preserve all non-key inputs,
record exact input/library fingerprints, and compare unchanged production with
an isolated parallel-only diagnostic. The117 regression tests pass (62.49s),
and49 focused tooling tests pass (2.38s). Saved710 device launches across71
matched cases in docs/gdn_mega/data/correlation_risk.json/csv. This is a testing
improvement, NOT a kernel speed or universal precision success: production
passes60/71 stress cases, parallel-only58/71; all56 unit-amplitude cases in this
new study pass both. Earlier tiny-decay failures remain unresolved. No production
kernel, tolerance, interface or allocation changes; failures remain visible.

2026-10-27 — `4ce8af3`: added real captured-input replay to the existing audit
script, with a regression test that prevents normalization/clipping and verifies
independently derived continuation state. Full118-test suite passes in71.37s.
On the user's Qwen3.5-0.8B captures, unchanged production and the isolated
parallel-only build both pass126/126 cases (108 actual full traces plus18
supplied-state continuations), five launches each, zero output/state elementwise
violations. Saved exact fingerprints, headwise metrics and all repeats in
docs/gdn_mega/data/qwen_capture.json/csv. This improves validation coverage;
it is not a kernel speedup or proof beyond the captured346..1024-token contexts.
Production fallback triggers69.0% of captured full-trace head instances despite
being unnecessary for passing this replay. No production code or tolerances changed.

2026-10-27 — `677c0ee`: remove the sequential fallback and all coupling-detection
reductions, flag stores/cache reads and dispatch. Captured-workload performance
improvement with known strict synthetic regressions (not a universally green
suite):126/126 actual Qwen capture/continuation cases pass all output/state
bounds on both libraries, including post-timing checks. Median paired speedup
across108 full traces17.00x (12.05..18.23x);18 continuations15.02x. For1K code,
roughly3.7ms ->208us per kernel. H16/Hg4 ordinary256..32K controls7/7 pass,
1.019..1.095x speedup;32K5900.52->5390.88us. Fixed15 trials x20 graph calls,
five checked warmups, NPU jobs serialized, exact inputs/library hashes matched.
The complete final regression suite has116 passes and2 failures (74.92s):
nearly_correlated_beta1 and delayed_nearly_correlated. They expose3 violating
output elements and1 state element in total, amplitude32,beta1,g0. No tests,
failures or outliers were removed and no tolerances relaxed. User explicitly
requested removal based on the measured real-input accuracy; these failures
remain to be solved in parallel. Source shrinks1302->1171 lines, same7 files,
2 phases/1 launch, unchanged public inputs/outputs/allocations. Evidence and
reproduction: docs/gdn_mega/README.md and data/fallback_removal.json/csv.

2026-10-27 — `ea51f0f`: added regenerable accuracy regimes and numerical severity
to the error heatmap. The31 report/checker tests pass; deterministic regeneration,
provenance matching and rejection of incomplete/mixed-kernel evidence verified.
Refreshed160-case/800-launch audit on unchanged677c0ee:137pass23fail. This is a
reporting/coverage improvement, not a kernel precision fix.56/56 controlled
unit-scale correlation cases and126/126 real captures pass;3unit-scale32K
near-lossless cases and20larger-scale probes fail. Failures remain finite and
explicit. Worst normalized witness7.35452e-5 against1.10690e-5 allowance (6.644x)
is described as a small absolute error, not a massive output error. No tolerance
changes, outlier removal or fallback. Data: docs/gdn_mega/data/numerical_regimes.json.

2026-10-27 — `df37822`: user-approved precision budget and regular stress coverage.
The user explicitly accepted the small measured errors and requested a tolerance
change plus normal regression tests. Fixed ATOL is now1e-4 (was1e-5), RTOL remains
1e-3; finiteness, RMSE<.005, every raw-output/state element, CPU FP64 golden,
input/guard integrity and five poisoned-workspace launches remain mandatory.
All160 exact existing cases now come from tests/gdn_mega_cases.py and run under
ordinary pytest (36standard+56critical+68correlation), with no opt-in stress flag.
Full245-test suite passes501.87s;34 reporting/checker tests pass0.27s. Three new
boundary tests reject a single beyond-budget element even when RMSE passes.
All23 former failures were freshly recalibrated: largest required absolute floor
at unchanged rtol8.2929952887e-5, below1e-4; worst new ratio in those cases.841616.
This is a policy/coverage improvement, NOT improved kernel arithmetic. Inputs and
tolerance-independent error statistics match the older audit; kernel binary
remainsdfb01e1. Historical figures and failure data retain their old policy.
Calibration/full log/collection: docs/gdn_mega/data/accepted_precision.json;
current policy and old diagnostics coexist in error_heatmap.md and README.md.

2026-10-27 — `08d331a`: refreshed the complete speed heatmap against freshly timed
original10748eb and unchanged current kernel677c0ee (binarydfb01e1). All63 current
cells pass CPU FP64 output/final-state elementwise checks before/after timing at
atol1e-4,rtol1e-3 plus finiteness/RMSE. All63 original cells fail that same policy;
their measured runtimes remain visible, with asterisks only on original runtimes.
All63 current cells are faster: original/current median1.6321x, range1.2562..4.5306x;
4K..32K range1.2562..1.7974x. At32K current H16/Hg4=5377.20us,
H24/Hg8=5699.19us,H48/Hg16=11405.29us,H64/Hg16=16586.58us.
Nine head pairs including24/8 and48/16, lengths128..32768, nonzero initial state,
five checked warmups and15x20 graph calls; both library hashes verified. Current
is median1.0559x faster than the older optimized heatmap's saved timings (a
cross-run comparison). This is fresh verification/reporting, not a new kernel
optimization or accuracy improvement.34 benchmark/report tests pass0.28s;
complete-grid/policy/library checks, deterministic Markdown regeneration and
visual inspection pass. Updated JSON/CSV/PNG/SVG/heatmap.md and reproduction
script; it now freshly measures both libraries and never borrows old error-grid
failure markers. Error heatmaps retain their historical checkpoint/policy.

2026-10-27 — `9e35bcb`: current-kernel profiling/reporting improvement; no new
kernel speedup or production change. Added docs/gdn_mega/profile.md with seven
figures (four explanatory/hardware charts, three traceviz timelines), complete
hardware CSV contents, diagnostic samples, numerical checks and source hashes.
Production remains677c0ee/dfb01e1. Fresh hardware task times: H16/Hg4/T4096
681.299us; H24/Hg8/T4096 715.177us; H48/Hg16/T4096 1362.438us;
H16/Hg4/T32768 5288.183us. At4K, cube active169.46us (24.9%), cubeMTE2
167.42us, FIX116.20us, vector239.54us; counters overlap.16/24 groups persist
atH16. Cube/vector L2 read-hit medians98.97%/97.66%, so GM traffic is not all HBM.
Isolated clock diagnostic: phase1~22.37us, persistent~641.74us; joins127.18us,
state-feedback wait106.41us,reload19.76us,RHS wait.02us,output/state products
140.48us. Instrumented profiler task686.14us is0.7% above production; clocks
remain approximate and overlap across cube/vector streams. Inverse joins are
only4.5% of persistent matrix FLOPs. Production simulator T512 passes FP64;
1,205,863events rendered with sibling traceviz22664d1, separate from hardware.
All4 production hardware checks, simulator check, first/last diagnostic checks
and diagnostic profiler check pass approved1e-4+1e-3 elementwise bounds against
CPU FP64.34 benchmark/report tests pass0.26s. Every clock-scope budget fits its
per-core phase; Markdown determinism, links, script hashes and unchanged
production sources/library verified. The full245-test suite was not rerun for
this report-only change. Reproduce with scripts/profile_gdn_mega.sh collect;
render regenerates from ignored raw dumps. Diagnostic code is generated only
under benchmark_data, with no production macros/probe files. Ranked structural
hypotheses: shorten state feedback, direct blocked RHS solve instead of inverse
materialization, bounded preparation helpers, affine chunk-state scan. These
are opportunities, not measured speedups or a claim of2x.
## 400151f — fast isolated builds and verified FP64 reference caching

Added `scripts/check_gdn_mega.sh`: head specializations compile concurrently in
an isolated directory, preserving the production source, ABI and installed
library. All seven compiled in 10.66 seconds. The tool verifies the loaded
library hash and records source hashes, coverage and timings. Its nine-case quick
screen uses unchanged regular-suite cases and adds checked 256/4096-token timings;
the full mode still runs every regression test.

Optional CPU-golden caching keys exact input bytes, oracle source and PyTorch
version, and hashes every loaded reference tensor. Device outputs and acceptance
results are never cached across runs. All 160 numerical cases retain five
poisoned-workspace launches, every-element bounds, input checks and guards.
Unchanged production passed 246 tests in 505.95 seconds while populating the cache;
the isolated asynchronous-join candidate passed the same 246 tests in 148.71
seconds with cached goldens. These are different kernel binaries, not a matched
kernel-latency comparison. The 35 checker/benchmark tests also pass (0.29 seconds),
including cache invalidation and corruption rejection. Quick screens take about
2.2–2.4 seconds after process startup and before the checked timing sweep.

No production kernel optimization is promoted by this commit. After the requested
exploration cleanup, evidence is retained in `docs/gdn_mega/data/algorithm_search.json`;
the verified faster candidate library/manifest are in
`benchmark_data/gdn_mega/verified_candidate/`. Rejected raw experiments were removed.


## 2026-10-27 — Concurrent token preparation and value-column recurrence (`365b105`)

Promoted factored/asynchronous inverse joining and an H8 schedule that uses
16 groups for two independent 64-value-column state slices per head, with eight
helper groups preparing future chunk/key-head work. Helpers share cached KK/QK
across grouped value heads. RHS preparation overlaps the inverse wait/load.
Dispatch depends only on shape and available work; C=128, C_inv=16, FP32
arithmetic, the caller ABI and all allocation sizes are unchanged.

The exact installed binary (`4dc29f4de899315b167f958138895eeca7f9a71c2c16a5b3e4bff36a50979f1c`)
passed **264 tests in 171.38 seconds**, including 18 new worker-layout cases.
Every output/state element is checked against the independent CPU FP64
recurrence using unchanged atol=1e-4, rtol=1e-3, with five poisoned-scratch
launches, integrity/guard checks and finite/RMSE checks. All seven head
specializations compile in **14.53 seconds**.

Matched medians against the previously installed `dfb01e1` production binary,
15 graph trials × 20 calls, supplied initial state, D=128:

| H/Hg | Tokens | Before µs | After µs | Speedup |
|---|---:|---:|---:|---:|
| 8/2 | 1024 | 205.68 | 180.18 | 1.142× |
| 8/2 | 4096 | 685.76 | 520.10 | 1.319× |
| 8/8 | 4096 | 693.98 | 517.22 | 1.342× |
| 8/2 | 8192 | 1257.32 | 967.97 | 1.299× |
| 8/2 | 16384 | 2558.70 | 1834.33 | 1.395× |
| 8/2 | 32768 | 4956.42 | 3862.76 | 1.283× |
| 16/4 | 4096 | 696.09 | 665.36 | 1.046× |

Short H8 medians vary by roughly ±2%; this is not a uniform gain over every
shape. Gate-panel rescaling passed the previous 246-test suite but did not
provide a repeatable speed improvement, so it is not included. Full tables,
rejected schedules and provenance are recorded in `docs/gdn_mega/algorithm_search.md`
and its existing JSON companion.

The refreshed original/current heatmap measures all 63 cells with the same
installed `4dc29f4d` binary: current passes 63/63, with its median total µs in
every cell. Original runtimes remain visible and are marked only for their
actual numerical failures. Report scripts now enable the existing validated
CPU-golden cache without reusing kernel outputs or acceptance decisions.
The follow-up 48+48 immediate / 32 delayed helper-reuse schedule was slower
(782.02us at H8/4K) and was discarded after its nine-case screen passed.
Cleanup removed 556 generated files (340.1 MiB); no experimental build/source
folders or obsolete candidate libraries remain.

## 2026-10-28 — simplify scheduling (`f381ae8`)

At the user's request, removed the H8-specific token helpers and value-column
workers. Restored one full-width head schedule for all shapes while retaining
factored/asynchronous joins. Production shrank from 1,342 to 1,189 lines across
the same seven files; ABI, allocations, numerical policy and all regression
cases remain unchanged. This is a code simplification, not a speed improvement:
the removed H8 schedule reduced 4K–32K latency by 22–27% in matched tests.

All 264 regression tests pass in 169.30s against the independent CPU FP64
reference, including elementwise output/state bounds, correlated-key cases,
poisoned scratch and integrity checks. All seven specializations build in
10.53s. The updated profiler compiles and its measured outputs/states pass;
35 CPU benchmark/report tests pass. Rejected experiments and diagnostic build
artifacts were removed. Measurements and fingerprints are in
`docs/gdn_mega/data/algorithm_search.json`; the heatmap generator preserves a
note about the removed schedule's benefit.

Fresh full heatmap: **63/63 restored-kernel cells pass**, including 24/8 and
48/16 through 32K. Figures and tables now compare uniform `f381ae8` with
original `10748eb`; the removed H8 benefit is documented separately.
