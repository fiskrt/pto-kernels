## Current workflow after merge cleanup (`f0bcf32`, 2026-10-28)

Earlier shell-wrapper commands and separate GDN test/helper modules below are
historical: they have been removed. Use `benchmarking/gdn_mega.py` subcommands
`build`, `test [--quick]`, `bench`, `heatmap`, `render`; use
`benchmarking/profile_gdn_mega.py` for hardware/simulator/render. All 264 test IDs
and 178 fixed cases now live in `tests/test_gdn_mega.py`. Canonical commands and
Docker setup are in `docs/gdn_mega/README.md`. No docs/data directory or profile.md
remains; figures contain only speed/error PNGs. Raw runs go to ignored
benchmark_data; retained folders are baseline_lib, reference_cache, qwen_capture,
plus latest heatmap.json. Historical raw profile/experimental builds removed.
All 264 tests pass (540.91s cold cache), warm quick 9 passes in 2.44s; full build 10.40s.
Hardware and simulator checks plus traceviz rendering pass. Production kernel
sources are unchanged from f381ae8. Do not recreate deleted helper files.

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
Warm-cache timing; report output AND state errors. Current acceptance requires
finite values and relative RMSE < 0.005 for both output and state. This aggregate
budget accounts for FP16 intermediate matrix products versus the FP32 reference;
the former 0.01 absolute-error shortcut and 5% RMSE budget were removed.
Actual observed errors are reported alongside the threshold.

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

## Ongoing experiments after 6453a98

- Shared KKT Gram computation once per key head, concurrent with gate scan;
  vector gating uses all cores. Tests pass for H16 including ragged/weak gates.
- Blocked inverse (two 64x64 diagonal inverses + one offdiagonal join) is faster
  than four 32x32 blocks: fewer tiny matrix instructions outweigh extra FLOPs.
- WY input-side scaling overlaps vector preparation with cube inverse work.
  Numerical output relative RMSE rises slightly from .00038 to .000435.
- Cube-only state recurrence keeps state in L1. Uses negative W, U@I + (-W)@S,
  then S@decayI + Ktilde.T@Vnew. Prep stores scaled K in A_inv_f32 scratch and
  decay matrices temporarily in s; cube overwrites s with actual incoming states.
  All 14 selected H16 tests pass. No tolerance relaxation. Timings with all
  experimental macros: 54.288 / 78.317 / 187.010 / 623.784 us. Long workload
  stage ticks: gates1082, KKT3541, inverse8756, WY3100, state7322, output6926.
- Current experiment CMake config: HEAD=16, MANUAL_SYNC=OFF,
  DEFINITIONS=GDN_SHARED_GRAMS;GDN_BLOCKED_INVERSE;GDN_INVERSE_BASE64;GDN_WY_INPUTS;GDN_CUBE_STATE;GDN_STAGE_TIMING.
  These are experimental flags, not defaults yet. Full-head acceptance pending.
- Simulator profile running with kernel-name `gdn_mega_specialized<16>`, using
  `benchmarking/gdn_mega.py --profile-once --lengths 128`; output in
  benchmark_data/gdn_mega/profile_state, process log /tmp/gdn-simulator.log.

## Fusion checkpoint

Full production acceptance after shared-output work: 23 tests pass, all heads,
T up to 1024 with weak gates and initial state. H16/Hg4 latencies
52.708 / 75.084 / 169.278 / 553.351 us. Simulator profile_output also passes
numerics after creating masks/identity on CPU and checking raw output on CPU
(msprof's kernel filter otherwise skips NPU setup/check operations).
Traceviz plot: benchmark_data/gdn_mega/pipeline.png; matched profile root
profile_output/OPPROF_20261022133337_UMHOVJSARRABVLPS. Trace reveals many serial
MTE2 and FIX transfers, substantial wait time, low cube utilization.
Disabling compiler automatic sync made no measurable difference; keep OFF for
GDN_MEGA_MANUAL_SYNC normally. Static full-tile DMA descriptors and overlapping
X/power matmuls in the blocked inverse gave negligible measured changes.

New optional experiment GDN_FUSED_PIPELINE (fused_pipeline.cpp):
- raw KK stored in v_new scratch (NOT A_inv_f32, which scaled K would overwrite
  before other grouped heads consume raw KK); raw QK in second half-span of
  A_inv_f32, scaled K in first half-span.
- one persistent worker per (sequence, value head), vector subblocks prepare
  all chunk factors together, sharing decay matrix coefficients across KK/QK.
- ready flags 5/6 and free flags 7/8 make a two-credit producer/consumer loop.
  Cube seeds both free flags, waits ready, replenishes free immediately; vector
  drains two leftover free signals at end of each head. No global barriers in
  the recurrent loop. Buffers are per chunk, so ahead-of-consumer writes do not
  alias older chunks.
- cube retains inverse, W/U/residual, state, and output accumulators on chip,
  avoiding intermediate GM stores/loads. L1 state lives at 8*32768, other
  operands at 9..13*32768; inverse scratch is below 6*32768, total <512KiB.
- direct fused path returns after initial gate scan/shared-Gram barrier and
  skips the staged path. Currently compiling/testing H16 with this flag.
- source defaults now enable all validated staged improvements unless
  GDN_LEGACY_STAGES is defined. Fused pipeline is NOT default yet.

## Paused checkpoint and commit discipline

Commit every measured improvement once its correctness tests pass, and record
the commit ID, validation scope, and measurements in success.md immediately.
Do not accumulate further validated gains without recording them.

Optional fused pipeline with contiguous head-major intermediates passed 16
H=16 tests: 39.185 / 51.217 / 124.193 / 411.824 us at T=128/256/1024/4096.
The two free-slot credits are replenished only AFTER the cube loads both slot
matrices into L1. This supersedes the earlier immediate-replenishment note.
The default staged implementation passed the full 23-test suite.

Optimization is paused by the user. The commit checkpoint excludes the
adaptive-inverse experiment, which remains in the working tree. Its latest
build failed with two tile-constructor argument errors; two new correlated-key
tests remain unrun. Do not treat this experiment as a validated improvement.
CMake cache currently selects HEAD=16 and definitions
GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE;GDN_FUSED_TIMING. The existing shared
library is from the preceding exact fused timing build, not the failed build.
On resumption, inspect /tmp/gdn-adaptive-build.log inside gdn-mega-opt.
The adaptive helper adds max_terms and four conditional blocks; their changes
are intentionally unstaged relative to the exact-inverse checkpoint.

## Resumed: bounded adaptive inverse

User resumed optimization. Fixed the two dynamic-tile constructors (one
dynamic dimension takes one constructor argument). Full-head acceptance:

```sh
bash scripts/gdn_mega_docker.sh bash -lc 'cmake -S . -B build -DGDN_MEGA_HEAD= -DGDN_MEGA_DEFINITIONS="GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE" && bash scripts/build_gdn_mega.sh'
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest -q tests/test_gdn_mega.py
bash scripts/gdn_mega_docker.sh timeout 120 python benchmarking/gdn_mega.py --output benchmark_data/gdn_mega/adaptive_acceptance.json
```

25 tests pass. H16/Hg4 latencies: 39.143 / 49.481 / 112.795 / 372.280 us.
During single-head testing, use `-k 'not head_configs or H16'`; matching `16`
alone accidentally includes H64-Hg16, unsupported by a HEAD=16 binary.
Counter build: inverse 6470 ticks vs exact 8754, rest ~9572 at T4096; vector
preparation ~11427 ticks. Cube remains critical. Full uninstrumented build
is ~3us faster. Next test: full128 polynomial avoids two block inversions and
join overhead when only a few terms are needed; retain blocked fallback for
high norm.

Full128 adaptive polynomial experiment rejected: all 18 selected tests passed,
but latencies 41.222 / 51.750 / 117.201 / 383.706 us were slower than the
blocked adaptive inverse. Larger matrix arithmetic outweighed reduced dispatch.
Next: overlap chunk RHS and output operand DMA with inverse computation;
current PIPE_ALL barriers unnecessarily include independent MTE2 transfers.

Direct residual polynomial (no dense inverse/W/U) passed 18 selected tests,
but initial-state timings 41.805 / 53.032 / 119.838 / 392.198 us are slower.
More full128 matmuls outweigh avoided smaller inverse operations. Rejected;
source snapshot in benchmark_data/gdn_mega/direct_residual.cpp.
Initial-state baseline now measured explicitly: 107.532 / 145.567 / 309.691 /
863.765 us. Next exact algebraic change concatenates shared-factor products:
[U,-W] = inverse [betaV,-beta exp(g)K]; [O;Snext] = [Qexp;decayI]S + [Att;Kt^T]R.
Both fit physical L0 buffers (128KiB L0C, 64KiB L0A/L0B).

Joint output/state products pass 18 selected H16 tests and reduce T4096 to
359.178us WITH initial state. A first attempt also concatenated two L0B tiles
by simple pointer offset; this is invalid because L0B is row-major in its
outer blocks. Removed that branch. Concatenate W/U in column-major L1 instead
and use one full-width TMOV to correctly interleave L0B. Current build flags
GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE;GDN_JOINT_PRODUCTS, HEAD=16.

Correctly interleaved joint W/U GEMM passes 18 tests but no meaningful gain:
358.615us vs359.178us atT4096, with short cases slightly slower. Removed it.
New exact simplification: vector emits -A directly; cube inverts its diagonal
blocks in place using +I, then consumes the untouched offdiagonal -A in the
join. Removes a full128 negative-identity GEMM and two L1 copies per chunk.

Prefetch operands into L1 buffers0/1/9..12 while in-place inverse uses2/5.
Inverse helper synchronizes only MTE1/M/FIX; final PIPE_ALL waits for operand
loads before padding and releasing slot credit. 18 tests pass, initial-state
latencies37.458/45.992/96.432/299.105us. GDN_PREFETCH requires GDN_NEGATIVE_A.
Next algebraic reduction: exp(g_i-g_j) can be computed as the outer product
exp(g_i-center)*exp(center-g_j). Check range <=120 to keep individual factors
finite; otherwise retain original matrix exponential. Reduces vector exp work
from8192 elements per stripe to192 without dropping matrix entries.

Factored centered decay passes 24 selected tests (adding gate_scale4 fallback),
but does not materially improve runtime:38.129/46.269/95.580/298.966us. Dropped
it; keep exp(g_i-g_j) for its simpler numerical behavior. Extra extreme-gate
tests retained. Next remove dead initial-state identity multiply and prepare
I-A in vector registers, replacing one identity GEMM per diagonal inverse.

Vector inverse seed experiment passes24tests but regresses to320.483us atT4096:
extra vector/DMA work outweighs removed cube identity products. Removed it.
Kept exact removal of dead initial-state*I accumulator copy. Tightening all
reference checks to0.5% relativeRMSE (from5%) and removing1e-2 absolute-error
early exit; no more near-zero-output false passes. Next inverse version uses
only the specific producer/consumer edges per operation, avoiding redundant
MTE1/M/FIX round trips while preserving operand lifetimes.

Strict checks found packed weak-decay short-tail errors in some heads. Full64
terms fixes them; CPU norms of the actual half A are2.7–3.6, requiring64.
Failing heads vary between runs. CANN TSYNC_CVID publishes scalar GM stores
with volatile,dcci,dsb(DSB_DDR); our producer lacked the dsb. Added producer
and consumer DDR fences and volatile accesses. All31 full-head tests now
pass, including50 repeats of the packed regression. Tight per-operation
inverse synchronization was not faster and was reverted.

Full acceptance command flags:GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE;
GDN_JOINT_PRODUCTS;GDN_NEGATIVE_A;GDN_PREFETCH, HEAD empty. Initial-state
latencies39.651/48.429/98.719/301.066us. No-state38.417/47.582/97.472/299.847us.
Next exact transfer reuse: after W*S, L0B already contains S; the joint output
product need not load it again. For a half decay matrix that is exactly zero,
its L1/L0A copies can also be reused across chunks (not a state-zero assumption).

Operand reuse passes25selectedstrict tests. Only L1 decay is reusable across
chunks; L0A bottom half gets overwritten by Ktilde^T in second joint product.
Caught that mistake in initial attempt and fixed it. S in L0B survives from
W*S to first joint product. Added finite/zero-decay alternation regression.
Latencieswithinitial37.468/46.568/95.733/292.657us. Next interleave independent
64x64 diagonal inverses, preserving arithmetic but amortizing pipeline flags.

Paired diagonal inverses pass25strict tests but save only~2us at4096 and
regress128 startup; reverted extra scheduling code. Next vector trim caches
exp(g) once for W/Q/decay and constructs decayI directly inhalf, avoiding
float conversions of an identity matrix. Exact-zero decay uses vector fill.


Latest scheduling reductions: reuse exp(g) without factoring exp(g_i-g_j);
omit S*0 when the stored FP16 decay is exactly zero; seed first vector credits
before cube setup. Both initial-state and no-state H16 benchmark sweeps now
exceed 3x at all four standard lengths. All32 all-head/default tests pass. Production medians regress slightly versus
the single-head build (T128 initial36.354us), so3x is not universal. New user
target: another2x from this production checkpoint through structural changes.
Default promotion enables fused/adaptive/joint/negative-A/prefetch/reuse/
zero-decay/early-preparation. GDN_STAGED_PIPELINE selects the older staged path;
GDN_LEGACY_STAGES retains the earlier legacy comparison path. For individual
fusion ablations, GDN_STAGED_PIPELINE disables automatic fused definitions so
specific GDN_MEGA_DEFINITIONS can be selected explicitly.

Structural pass baseline f71a883 (full-head): initial-state36.354/44.813/
92.734/284.608us. Simulator T256 trace shows14% cube utilization and vector
producer waits. Profile:profile_structural/OPPROF_20261022152508_YOOUMPNXUMCJBEFC;
plot:structural_pipeline.png. Flattening matrix TCVT views passes25tests but
has no meaningful speedup (284.885us at4096), discarded. Three-half-buffer
streaming preparation also passes25tests, but only startup improves slightly
(35.000/42.714/91.256/283.335us). It does not reduce the long steady-state cost.
Need distinguish simulator serialization from hardware critical path using
phase counters before further scheduling changes.

Hardware counter correction: streaming producer cube [wait221,inverse3353,
rest8403] ticks; vector prepare8017/8190ticks, waits3374/3200. At32chunks,
cube wait is only~2%; simulator's startup producer stall is not the hardware
steady-state bottleneck. Dropped streaming preparation (snapshot ignored).
Six-product cube schedule now261.891us versus284.608us, with unchanged math.
Next work must remove matrix operations or state dependencies, not just flags.

Real-device Roofline profile of recurrence schedule1290ec8 atT4096:
device_roofline/OPPROF_20261022154557_CGPVAFKBAWGHKVLN. Active-core medians:
cube busy32.47%, MTE1 busy28.64%, MTE2 busy18.45%, FIX busy20.99%; cube wait
58.34%, MTE1 wait56.43%. Profiler diagnoses pipeline latency, not bandwidth.
Command: sourceCANN; timeout180 msprof op --kernel-name='*gdn_mega_specialized*'
--aic-metrics=Roofline --output=benchmark_data/gdn_mega/device_roofline
python benchmarking/gdn_mega.py --initial-state --profile-once --lengths4096.
Matmul probe confirms high movement/feedback penalty. Interleaved inverses
help weak gates~9.6%, slightly regress ordinary gates; optional, arithmetic
unchanged. User explicitly requests general pipeline/throughput optimization,
not additional decay-conditioned specializations. Next compare explicit
scheduling with compiler auto-sync disabled, then broaden recurrence scheduling
to finite decay and profile independent matrix-product transfer/compute overlap.

General recurrence schedule1e9fee7: finite decay included, no new decay shortcut.
25strict tests pass, initial ordinaryT4096=260.210us, weakT4096=286.148us.
Moving the state refill earlier gives only noise-scale differences; reverted.
Matmul store probe (256 stores) median ticks:contiguous3831,strided5506,L12048.
Only~1.44x output-store penalty; a new vector scatter stage would add substantial
traffic for a small possible saving, so not pursued. Probe raw:store_probe.jsonl.
RHS-first reassociation [betaV,-beta expK]*[I;S], then inverse*RHS passes25strict
tests but is slower263.750/288.095us atT4096 ordinary/weak and increases weak
outputRMSE0.000537→0.000575. Reverted; snapshot:rhs_recurrence.cpp. The original
residual arithmetic remains. Full-head acceptance of general schedules running.

Cross-chunk inverse lookahead (upper L0 banks, alternate state L1, next operands
prefetched during inverse) passes25 selected tests but is slower:265.828us
ordinary/286.730us weak at4096. Rejected; ignored snapshots preserve the experiment.
Reordering W before U to overlap W export also produces no robust improvement,
especially for weak decay; rejected. Shared Gram pipelining does help: full32
strict tests pass, fresh same-device ordinary261.734->254.176us and weak
283.597->276.511us. Single-wave Gram work uses original setup to avoid startup
regression. Next structural experiment: use otherwise idle cube cores for
output products alongside the state recurrence, with full per-chunk buffers.

2026-10-22 structural experiments after the Gram pipeline (all rejected):
- Offload output to the eight idle cube cores for a single H16 sequence.
  State producers publish rounded residual/state buffers; output workers run
  Q*S + attention*R. Selected25 tests pass with unchanged errors, but ordinary
  T4096 is265.334us and weak286.147us versus production~254/277us.
  Counters show producer waits~1861ticks, inverse~1946, recurrence~7456;
  vector preparation~10846/11011; output workers~7900work+3700wait ticks.
  This moved the bottleneck instead of removing it.
- Restoring contiguous two-slot attention plus explicit consumer acknowledgments
  does not recover the vector slowdown (preparation~10800ticks), disproving a
  store-layout-only explanation. Earlier residual publication still gives
  265.170/290.575us ordinary/weak. Ring snapshots:output_workers_ring/.
- Decouple preparation across all48 vector cores, with16 recurrence and8 output
  cube workers. Reuse each saved-state slot for its decay matrix before state
  publication. First attempt bypassed vector mask initialization and faulted;
  fixing initialization makes25selected tests pass onNPU1. NPU1 T4096 timings
  253.782us ordinary/305.935us weak are not a useful general improvement.
  No cross-device speedup claim. Snapshots:global_prepare/. Device0 subsequently
  reported healthy and passed25 tests against the frozen known-good library.
- Direct aligned load2d operand transfers instead of general img2colv2 loaders
  pass25 selected tests onNPU1;257.588/277.452us ordinary/weak. No demonstrated
  gain; rejected. Snapshots:load2d/.
- Split value columns into64-wide tiles, retaining four FP32 accumulators and
  interleaving residual/state exports with independent products. All25 selected
  tests pass with unchanged errors, but255.942/276.574us ordinary/weak are not
  better than production. Snapshot:recurrence_tiled.cpp.
- Let an output FIX store overlap the next inverse. Missing M->MTE1 ordering
  first caused an L0A conflict; adding the dependency and retaining full tail
  barriers passed25selected tests, but the4096 benchmark then failed with
  output relativeRMSE0.065378. This is NOT accepted. Full chunk-boundary barriers
  remain in production. Snapshots:deferred_output/.

All experimental files are ignored snapshots, not production source. Final
production rebuild uses all head specializations, empty experiment definitions,
and compiler automatic synchronization. No new decay-dependent shortcuts were
introduced in this structural pass; only the tested Gram pipeline is retained.

Final production recovery and acceptance: all32 tests passed onNPU0 after the
experimental faults; all-head build, no extra definitions, automatic syncON.
Initial32.912/40.444/83.016/254.862us; weak90.317/276.833us at1024/4096.
No-initial33.323/40.772/83.348/253.609us. All actual errors unchanged.
Device profile:device_gram_pipeline/OPPROF_20261022172022_WQDJDGCBKTHCAPFG,
1800MHz; active cube0..15 median busy31.06%,MTE126.67%,FIX21.26%; waits
cube54.57%,MTE151.50%. Summary:gram_profile_summary.json. The further2x target
is not achieved; source and build are restored to the tested Gram improvement.

## 2026-10-23 — precision/performance experiments after the FP32 correction

The strict pointwise policy remains unchanged. The 64-case FP32 checkpoint is
`41f0549` (1.259212 ms at H16/Hg4/T4096 with initial state). Four additional
adversarial cases were subsequently fixed before testing: random correlated
keys/beta=.9/amplitude32, nearly correlated keys/beta=1/amplitude32, positive
log gates, and 16K no decay. The nearly correlated case exposes further errors;
it must not be removed or accepted through RMSE alone.

Measured experimental H16/Hg4/T4096 candidates (not accepted releases):

| Candidate | ms | Status |
|---|---:|---|
| FP32 32-token, parallel vector inverses | 1.189890 | original audit passed; added near-collinear case fails |
| Paired FP16, 32-token | 1.392073 | no improvement |
| Paired FP16, 128-token, diagonal16 | 1.278445 | 67/68 tests pass |
| Same, diagonal32 | 1.368102 | benchmark passes; no improvement |
| Same, diagonal64 | 1.334599 | benchmark passes; no improvement |
| Residual before inverse, original FP16 K/Q | 1.057365 | ordinary and overflow smoke cases pass; near-collinear case fails |
| Bounded single-FP16 inverse + Newton residual correction | 1.102940 | ordinary/weak/no-decay/overflow pass; near-collinear case fails |

Residual-first identity: `R = inv * [beta * (V - exp(G) * K*S)]` replaces
`inv*(beta*V) - [inv*(beta*exp(G)*K)]*S`. Original K and Q need only two products
against a high/low state pair. Scaling R for the state update on vector permits
another exact-input half product. Output and authoritative state stay FP32 on
vector; only public stores round to FP16. Existing workspace sizes are retained.

Latest profiler: `profile_residual/OPPROF_20261023144109_OWBQIAEZWKUOQDLE`.
Block0 cube: 1.026 ms total, 203.7 us cube active (~19.6%), 141.3 us MTE1,
186.9 us MTE2, 110.6 us FIX, 246.3 us global vector-preparation wait, 65.5 us
RHS/state wait. Vector0: 256.3 us vector, 305.6 us waiting for inverse/RHS solve.
These are overlapping per-pipe counters, not additive stage timings.

Two actual implementation hazards found and corrected: vector temporary reuse
requires V->MTE2 ordering before a new DMA load, and short dependent cube MADs
require a PIPE_M barrier (otherwise a 16x16 product retained earlier Gram data).
The poisoned-workspace tests detected the latter; integer beta=1 diagnostics
localized it. Do not remove either dependency to improve timings.

Overflow mechanism: for identical unit keys, beta=1, g=0, signed strict-lower
A has entries -1. Its powers contain binomial coefficients despite `(I-A)^-1`
being just a unit diagonal and a -1 first subdiagonal. Scaling the enormous
power-series terms prevents infinities but cannot undo catastrophic cancellation.
Stable diagonal substitution plus block joins avoids those powers altogether.
A CPU experiment with bounded FP16 block inversion + one Newton correction
reduced inverse error on a no-decay sample from 1.52e-4 to 1.20e-7, but the NPU
correction cost outweighed its savings. Raw results are in ignored
`benchmark_data/gdn_mega/`; the candidate remains experimental.


2026-10-23, FLA ranges and radical algorithm screening:
- User requested larger algorithm/pipeline changes, no micro-optimization loop.
  Pinned FLA source analyzed in docs/gdn_mega/input_ranges.md/json. Main chunk
  tests use positive normalized keys (mean KK off-diagonal ~0.75), beta
  sigmoid(uniform), g log-sigmoid(uniform) scaled/masked, including zero decay.
  FLA beta-logit mode can double sigmoid into (0,2); FP32 initial state is adapted
  to this kernel's FP16 ABI. No FLA RMS-only/CI-warning tolerance was imported.
- Positive no-decay 128-token example has inverse powers up to4.76e18 while
  true inverse maxabs1. Weak decay8.58e15; identical beta1 keys6.03e36. Large
  intermediates/cancellation, not large KK entries, cause the old overflow.
- Added8FLA distribution cases and2strong-tail regressions; now80tests expected.
  Positive Q/K use half normalization matching FLA, rather than float-normalize
  then half. All old adversarial cases retained. Source hashes saved.
- Rejected NPU algorithms, 4KH16/Hg4initial: precomputed inverse1.228863ms
  (78old tests pass), direct32-row panels1.251790, halfinverse+RHSrefinement
  1.278160, native64diags1.291802, native32diags1.217353, exact eight-block
  preconditioned series1.216864. Last variants only selected audits+benchmark;
  they are not full-support certificates. Snapshots in /tmp/gdn_*; no unused
  experiment kernels are kept in production. Results in radical_experiments.md.
- NativeFP32 inverse+RHS .945859ms; adding nativeattention .931576ms. This is
  modest, not target reached. Removed unused full inverse FP16-pair writes,
  unused pair-product helpers and identity products from retained candidate.
  Pair representation now only original-half K/Q times paired state/scaledR.
- FIRST FULL HYBRID RUN: 75pass/3fail, nonfinite outputs at127-token/strong
  packed tails. Root cause padded zero G minus negative real G overflowedexp;
  invalid0*inf poisoned full native inverse rows then0*NaN spread through output.
  Fix invalid-row mask BEFOREexp (load causal mask with count rows, not64).
  Added FLAstrongT75 andstrongT92 cases. Full80-test rerun PASSED in89.90s.
- Current candidate preserves ABI/allocation sizes. A onlycouplingmarkers,w
  unused; A_inv/u FP32attentioncolumnhalves; o_qs/o_gated FP32inverse rowhalves;
  o_qk FP32join scratch; h_ws second64KB FP32RHS. Read currentREADME forcontract.
- CPU affine-map screen passes sampled ordinary/no-decay/FLA cases, fails
  nearcorrelatedamp32 like other FP32 chunk methods. Potential next big change:
  precompute F,B,Qprime,Uprime perchunk then recurrent phase uses only
  S_next=F*S+B andO=Qprime*S+Uprime (2paired products,1exchange roundtrip).
  Need measure precompute overhead; no NPU implementation yet. Existing
  compensated head restart must remain for strong coupling. See algorithm_screen.json (affine_maps).

Final retained hybrid after80passes:4KH16/Hg4withinitial .905793ms;
T128 .066423ms,8K1.783001ms,16K3.615756ms,32K7.520225ms.
FLApositiveH8/Hg2zero-decayT1024 3.686692ms,weakT1500 5.393116ms,
strongT257 .124532ms,GaussianinitialT1024 .259980ms,beta2T5131.873560ms.
Strong path latency remains a major limitation, not a target-speed result.

2026-10-23 shared-operand pipeline:
- K/Q projection stacking alone0.909957ms: no win vs0.905793ms.
- Prepare Gamma=exp(G_last-G_i)*K, then stack attention/Gamma^T against R.
  Removes dependent residual scaling/splitting and flags3/4/10.
- Initial fullsuite79/80: late-coupling marker was stale in scalar cache after
  reusing per-core control address. dcci+volatile read fixes it;80/80 pass.
- Corrected H16 initial-state timings128/4K/8K/16K/32K:
  0.069667/0.844893/1.672497/3.392108/6.872840ms.
  Public ABI/alloc unchanged; A/w now Gamma, wy slots unused compatibility.
- Next: native attention stores no longer overwrite decay tile, so its repeated
  calculation is redundant. Also batch independent inverse joins by level to
  reduce serial export/reload latency; do not confuse packed MAD output layout
  with a larger-stride accumulator view.

2026-10-23 reuse prepared decay:
- Removing the duplicate C128 decay matrix gives0.794724ms at4K (1.063x vs
  fused0.844893ms),80/80 tests pass. No truncation or G_i-G_j change.
- Batched16-block joins0.806571ms,8smoke+benchmark pass; rejected, removed.
- Current experiment draft /tmp/gdn_affine_edit.py + /tmp/gdn_affine_pipeline.hpp:
  three phases/private raw Gram copies, chunk-parallel affine maps, one stacked
  native product per recurrent chunk. Existing allocations reused; unaligned
  sequences retain fused path. Not yet hardware tested at this entry.

2026-10-23 affine maps hardware result:
- Corrected map implementation15 selectedtests pass;4K1.253943ms: reject vs
  retained0.794724ms. Initial missing set_mask_norm caused UB exception; fixed
  before valid timing. No tolerance/selection changes.
- Snapshot /tmp/gdn_affine_candidate and generation scripts /tmp/gdn_affine_edit.py,
  /tmp/gdn_affine_pipeline.hpp are experimental only. Production affine file
  removed; restored exact bfc99e1 source and rebuilt library hash
  92c7bfe1d266cf8a3fc5a17c632fb83e6cc03a10624b910e91423d8877dc92ef,
  matching the80-test/shared_decay_benchmark checkpoint.

2026-10-26 stable heatmap refresh:
- User requested replacing obsolete FP16 speedup plots. Measured all49 current
  H8/12/16/24/32/48/64 x T128/256/1K/4K/8K/16K/32K cells; all pass strict
  pointwise/RMSE/finiteness checks before and after graph replay.
- Frozen original10748eb timings retained with explicit historical RMSE-only
  provenance. Fresh original H16/T128,T4096 strict checks fail; cannot silently
  call those strict passes. Original H64/T32768 remains gray/no ratio.
-48 valid latency ratios1.04–3.85x;4K-and-longer1.04–1.54x; no regressions.
  H16 long ratios~1.10x at4K/8K/16K,1.06x at32K.
- Tooling f071140 supports --baseline-report with settings/environment/input
  generator/library guards. Byte-identical validated outputs reuse statistics;
  six focused checker tests pass. No tolerance or production kernel changes.
- Updated docs/gdn_mega/heatmap.{png,svg,json,csv,md}; validation checks in
  heatmap_validation.txt. Long-shape baseline provenance/known failure preserved.

2026-10-26: fresh original/current hardware profiling, H16/Hg4/T4096/initial.
Production durations original890.307us vs stable794.432us (1.121x). Current
library92c7bfe... unchanged. All current profile/diagnostic strict checks pass;
original fails, never relabeled. Docs: profile_comparison.md plus raw JSON.
Current phases from isolated diagnostic copies: shared grams+gates34.20us,
persistent768.41us. Cube active-group median arithmetic175.58us (22.1%),
vector0 arithmetic316.97us vs original165.33us. Only16/24 groups persist.
Current cube section medians: paired projections100.27us; inverse loads/joins
223.60us; attention loads22.71us; RHS wait0.02us; residual solve/export97.94us;
output/state products143.75us; state feedback wait129.00us; state reload19.05us.
Vector prep435.53us: weighting/masks/checks214.04, diagonal solves162.49,
export/Gamma56.21us. Cube/vector counters overlap; do not sum their totals.
Each chunk's7 inverse joins do14 GM store/reload round trips (7168 overall).
Current cube MTE2 instructions846 vs original377.5; direct L0C->L1 traffic0
vs original9216KB median. GM->L1 traffic16544 vs11969KB. Cached GM traffic is
not necessarily HBM traffic; diagnosis is dependencies/transfer latency.
T256 simulator + traceviz shows FIXPIPE/CUBE serialization, but simulator
startup timing is not a hardware critical-path estimate. Diagnostics source
patch and exact measurement script saved; production has no profiling code.
Next promising structural targets: keep stable inverse intermediates closer to
compute; take independent prep off late state feedback with bounded buffers
and idle core groups. Prior global-precompute design was slower: do not simply
repeat it. Native FP32 projection ablation only addresses~100us, not entire gap.

2026-10-26, follow-up structural optimization (`aac5a4d`):
- Batched dependent inverse levels: independent operands/accumulators, six
  synchronized export/reload phases. Small MAD accumulator layout MUST be native
  BxB per group; treating it as a RowsxB subview corrupts later column panels.
- Direct byte-offset coefficient gathers eliminate diagonal transposes. Vector
  vmla with src1 repeat stride0 consumes solved rows without broadcast copies.
- Weight next chunk before flag5, preserve its coefficients in UB96..128 through
  RHS (V converts in UB128, input reuses0); finish inverse only after ack.
- Whole attention/Gamma tiles partition existing allocations by halves of heads;
  all supported H are even. No caller allocation/ABI/stage changes.
- Normal production build:80tests pass92.18s; strict256..32K sweep7/7. H16/Hg4
  4K803.037->720.410us;32K6528.661->5766.021us. Additional4K H8/H24/H64 speedups
  1.157/1.150/1.171x. Screening final_base701us is NOT production's720us.
- Hardware profile:794.432->700.678us (separate from graph timing); cube active
  175.58->169.46, MTE2 193.66->166.26, FIX148.04->114.58, vector316.97->282.95us.
  Vec flag5 wait114.85->3.56, flag9 wait12.07->87.86us; counters overlap.
- Rejected measured4K candidates: headmajor810us; all-core precomputed inverse
  1147; split weights with extra copy801; factored final64 solve812; paired
  on-chip inverse/native RHS830; packed FP32 GM799.7; native projections874;
  spare-core full preparation824 (H8 709 vs retained680); helpers weights-only
  748. Two-chunk lookahead with pending weights preserved through output/state
  passed128/4K but730us, so rejected. No inactive versions in production.
- Compiler rejects FP32 cc->cbuf intrinsic destinations even bypassing PTO guard;
  supported direct accumulator copies narrow. Do not claim native FP32 L1 path.
- Failed cached-identity and early source-layout variants retained as failures
  in report. One early packed run loaded baseline before candidate finished;
  explicitly excluded, launcher now tests candidate exists, compare loaded SHA.
- Primary reproducible evidence: docs/gdn_mega/optimization_20261026.md/json/csv.
  Isolated experiments remain ignored under benchmark_data/gdn_mega/breakthrough_20261026.
  Two global phases, seven production files,1302 physical lines. Target3x NOT met.

2026-10-26, user-requested report cleanup:
- Maintained docs entry point is docs/gdn_mega/README.md. Only four Markdown
  pages remain: README, heatmap, error_heatmap, input_ranges. Raw inputs/results
  are in data/ and plot images in figures/. Obsolete report paths in earlier
  log entries are historical; recover them from Git commit a83187b if needed.
- scripts/report_gdn_mega.sh [all|latency|errors|render] is the report entry point.
  Plotters now generate Markdown from the same JSON as their PNG/SVG. Shared
  text renderer: benchmarking/gdn_mega_report_text.py. Input-only range script
  gdn_mega_ranges.py replaces retired experimental gdn_mega_algorithms.py.
- Fresh AAC latency/error grids49/49 current passes, zero pointwise violations;
  original error grid0/49, independent FP64 current2/2 original0/2. Keep these
  distinctions: latency baseline is historical RMSE-only, error baseline fresh.
- No C++/tolerance changes. Saved80-test pass hashes reverified;12 report/checker
  CPU tests passed. New data/validation.json records report integrity checks.
- Source-version lookup excludes README-only commits, so report cleanup cannot
  relabel the kernel version. Heatmap render uses current actual library hashes.

2026-10-26: Critical precision audit of unchanged aac5a4d / library619b7df:
standard116-test suite passes, but56 additional beta[0,1] cases yield33 pass,
23 fail. All finite, ideal FP16 rounding passes, poisoned-workspace/guards intact;
all failures would pass RMSE-only. Failures include beta0.6000977,0.8999023,
random beta[0,1), and beta1: not merely beta>1 or an endpoint. Tiny negative
log gates accumulate decay error in compensated recurrence: state and its
compensation are separately multiplied by rounded FP32 exp(g), without retaining
multiplication error. k=e0,beta1 gives an analytic witness: initial3.013671875,
g=-1.000000011686e-7,T32768 => exact3.00381283675, kernel2.998046875. Scalar
FP32 repeated multiply then FP16 matches exactly. Six independent analytic
output checks agree with the FP64 oracle within1.11e-12. No-decay matched controls
pass; nearly correlated zero-initial-state case still fails with tiny decay.
Normal chunked beta0/T32768 also has28 near-zero output violations (worst1.40x),
so fixing fallback decay alone may not finish numerical work. No kernel fix or
relaxed tolerance applied. Full evidence and commands: docs/gdn_mega/README.md
critical audit section and data/precision_audit.json/csv. Source commits0a1797e,
810b731,f2019b9 retain the exact generators used for the successive audit batches.

2026-10-26 — matched correlation study after user-requested repair revert:
- Production remains source checkpoint1a4244d, kernel aac5a4d, library619b7df.
  Isolated parallel-only diagnostic disables only the fallback dispatch;
  library3fa652f. Never confuse that binary with the deployed library.
- Generator/tooling f89c376 preserves all non-key inputs across correlation
  changes and records fingerprints, actual cosine statistics and exit flags.
  Exit flags count executed fallback heads only for production, not ablation.
- 71 cases x2 binaries x5 repeats=710 launches. All56 unit-amplitude cases pass
  both, including32K and h0std1. These are NOT the old56 tiny-g critical cases,
  whose23 failures remain. Gates here0,-.001,-.1,-1; mostly one seed.
- At amplitude32,beta1,g0,T4K, parallel output violations at rho0/.95 are41/333
  of4,194,304; worst-allowance witnesses2.17e-5/4.76e-5. Global maxabs is often
  FP16 rounding, not these near-zero cancellation errors. Identical keys can
  be easier than nearly correlated; no monotonic correlation-error rule.
- Production60/71 passes vs parallel58/71. Fallback not uniformly safer:
  rho.95,g-.001,amp32 has6506 production output violations vs231 parallel.
  Existing nearly_correlated_beta1 loses one element by6% over bound without
  fallback. All finite, guards intact, scratch repeats identical, FP16 ideal
  floors pass. All failures pass RMSE-only, so keep strict policy.
- Production triggers all8 heads at rho.25,beta1,g0 (worst pair, not mean).
  Verified63 saved heatmap inputs never trigger: conservative max bound.3751.
- Evidence: docs/gdn_mega/data/correlation_risk.json/csv and README section.
  Regression117 pass62.49s; tooling49 pass2.38s. No kernel fix or timing claim.

2026-10-27 — real Qwen capture replay, tooling4ce8af3:
- User capture directory /tmp/claude-10182/-home-fskogh/27b86c35-f06f-4387-bf22-935eb7e98fff/scratchpad/gdn
  includes gdn_keys.pt: six prompts x18 layers, T346..1024,H16/Hg16,D128.
  Local ignored copy: benchmark_data/gdn_mega/qwen_capture/gdn_keys.pt.
- Use audit_gdn_mega.py --suite capture --capture PATH --output PATH. No extra
  script or synthetic distribution fit. Q/K/V already FP16; beta cast to ABI
  FP16 before both golden/kernel; g retained FP32. Never renormalize/clip.
- 108 full traces +18 code_python suffixes with FP64-derived, FP16-rounded
  prefix state. Both unchanged production619b7df and isolated parallel3fa652f
  pass126/126, five launches each (1260total); every output/state checked.
  198,623,232 output and33,030,144 state elements per full pass per binary.
- Production output maxabs.00195618 vs parallel.00195250; latter equals ideal
  FP16 rounding max. State maxabs.00337935 equals floor on both. Worst output
  allowance ratio.484879 both; state.479069 production/.476524 parallel.
- Fallback occurs1192/1728 full-trace head instances (69.0%), despite parallel
  passing all. Flags agree with independent FP64 coupling threshold in all
  heads. This is not a rare branch on the actual workload! No speed measured.
- Real g median-.006174,36.97% above-.001,1.60% above-1e-6; beta median.353,
  Vabsmax19.40625. Preserve head/token patterns; constant tiny gates at32K
  are different. Do not extrapolate these up-to1K captures to32K or all models.
- All guards/inputs/repeats intact;118 regression tests pass71.37s. Evidence
  docs/gdn_mega/data/qwen_capture.json/csv; no production/tolerance/ABI changes.

2026-10-27 — fallback removed on explicit user request, commit677c0ee:
- Every head uses parallel chunks. Deleted compensated token recurrence,
  private balanced dot helpers, max-coupling reduction, flag writes/dcci/branch,
  internal g_raw/control/bool args. No dormant fallback/config remains.
- Current binarydfb01e1; old619b7df preserved under
  benchmark_data/gdn_mega/fallback_removal/before_lib. Same ABI/allocations,
  seven production sources,1171lines vs1302. o_workspace_qk second half now
  reserved/unused; tests no longer interpret those obsolete scratch markers.
- Existing audit --timing reuses gdn_mega.time_launch (15x20 graph calls),
  checks all outputs/states again after timing. No duplicate golden per case.
- Captures126/126 pass before/after, all five poisoned repetitions and timed
  graphs. Metrics exactly equal prior isolated parallel-only capture audit.
  Full108 trace speedup median16.9957x,range12.0518..18.2344x.18 continuation
  median15.0173x.1Kcode3.68..3.73ms->208us. Not end-to-end model speedup or
  improvement over original six-stage algorithm; baseline is pre-removal.
- Ordinary H16/Hg4 with h0,256/512/1K/4K/8K/16K/32K all pass. Speedups
  1.021/1.019/1.030/1.040/1.048/1.068/1.095x;32K5900.52->5390.88us.
- Final full regression116pass2fail,74.92s. nearly_correlated_beta1:1 output
  violation of526336,ratio1.06010,error1.324e-5. delayed_nearly_correlated:
  2of394240 outputs,ratio1.50320,error2.006e-5;1of131072 state,ratio1.07987,
  error1.087e-5. Both Vabsmax124.9375,h0max~12.5,beta1,g0; FP16 floors pass.
  These are real retained strict failures, not xfails or relaxed tests.
- Historical critical33/56 result was pre-removal; do not present its exact
  counts or fallback-specific mechanism as current. Heatmaps remain labeled
  aac5a4d checkpoint, not silently refreshed. Latest report:
  docs/gdn_mega/data/fallback_removal.json/csv with README section.

2026-10-27 — accuracy regime/severity report ea51f0f:
- Refreshed ALL160 fixed synthetic cases on unchanged fallback-free677c0ee /
  librarydfb01e1:137pass23fail,5 launches each. Primary beta[0,1],g<=0,
  normalized-key setups134/157pass;3 exploratory cases outside these assumptions
  all pass but do not establish primary-workload safety. All finite, guards and
  scratch repeats intact; all ideal FP16 rounding floors pass.
- Controlled unit-scale correlation56/56pass. Long32K fixed tiny-g group3/6pass:
  g=-1e-8,beta1 has1478of33554432 output violations,ratio2.18890,witness2.314e-5;
  g=-1e-7,beta1 has21; beta0 matchedcontrol28. All three final states pass.
  This is not only a large-value issue; long real captures are next evidence need.
- High-amplitude/large-state probes8/28pass.20fail plus3unit-long=23total.
  Worst normalized witness: critical_nearly_correlated_3023_128.0,output
  reference-.001068956603 vsactual-.001142501831,error7.35452e-5,
  allowance1.10690e-5,ratio6.64428. Small absolute defect exceeding strict test
  threshold; NOT output6.64x wrong or proof of major model impact. Largest
  maxabs among failures.249755 equals that case's ideal FP16-rounding max.
- User emphasized small absolute errors vs massive ones. Lead with absolute
  error; ratio is only threshold overrun. No model-quality effect measured.
  Weak g not rare:36.97% >-.001,1.60% >-1e-6 inQwen. Actual captured h0max12.60
  matches stress h0max~12.5, so state magnitude alone is not an extremeness test.
- docs/gdn_mega/error_heatmap.md now has passing/caution/untested regimes,
  witness/error/allowance/excess/count table, real-workload relevance and priority.
  gdn_mega_report_text.py regenerates it from numerical_regimes.json plus
  current captured evidence, rejecting incomplete/mixed-kernel data. Figures
  remain historicalaac5a4d and are explicitly separated from currentregimes.
-31 CPU report/checker tests pass. Deterministic rendering,complete800 launches,
  provenance,local links and four Markdown pages verified. No kernel/tolerance
  changes; do not claim universal green suite or catastrophic errors.

2026-10-27 — IMPORTANT current policy supersedes historical no-relaxation notes:
- User explicitly said all measured errors are acceptable and requested changing
  atol/rtol to accept them and moving these cases into regular suites.
- Commitdf37822: ATOL=1e-4, RTOL=1e-3, MAX_RMSE_RATIO=.005. Fixed global policy
  for every raw output and state element; never adapt thresholds to candidates.
  Kernel677c0ee / librarydfb01e1 unchanged. Do NOT call this a numerical kernel fix.
- Shared pure-Python catalog tests/gdn_mega_cases.py:STANDARD36+CRITICAL56+
  CORRELATION68=AUDIT_CASES160. Pytest now parametrizes all160 by default;
  audit --suite standard still selects36; --suite all exactly160 without duplicates.
  Original dictionaries/seeds are identical to historical numerical_regimes.json.
- Fresh23-former-failure audit allpass, identical input hashes and unchanged
  tolerance-independent statistics. Max required_atol_at_fixed_rtol8.2929952887e-5
  (critical_nearly_correlated_3011_128.0 finalstate); new worst ratio.841616.
  The137 older passing cases already needed<=1e-5. Full fresh suite245pass501.87s;
  ordinary collection explicitly lists160 parameter names, total245tests.
- Every-element checks, FP64 independent reference, finiteness, exact shapes,
  guards/inputs and five scratch-poison repetitions kept. Three new boundary
  tests at reference0/1/100 reject one element1% over budget despite passing RMSE.
- Report formulas now use their recorded policy, not hardcoded/live tolerance.
  Old error grids/numerical_regimes data stay immutable withatol1e-5 diagnostics.
  accepted_precision.json records authorization, calibration, all245test log,
  collection, source hashes. Renderer verifies matching kernel/cases/input hashes.
- Canonical auto-loop test command is the four GDN files in docs/gdn_mega/README.md;
  no old240s timeout (expanded full suite takes~8m21s, mostly CPU FP64 work).
  No new runtime tuning options, fallback, caller allocation or production-code changes.

2026-10-27 speed heatmap refresh (`08d331a`): docs/gdn_mega/heatmap.md now measures
current677c0ee/dfb01e1 against freshly timed original10748eb/4bf4218, all63 cells.
Current63/63 pass approved1e-4+1e-3 bound; original0/63 pass (timed regardless).
Median original/current1.6321x; 4K+range1.2562..1.7974x. Each cell retains medianus.
scripts/report_gdn_mega.sh latency now freshly measures both libraries; render
uses the latency workers' own accuracy status. Do not overlay the older error
heatmap: its current library and tighter policy differ.34 report/benchmark tests
passed, full grid verified. No production-kernel changes in this refresh.

2026-10-27 current profile (`9e35bcb`): docs/gdn_mega/profile.md is the maintained
profile, generated by benchmarking/profile_gdn_mega.py and scripts/profile_gdn_mega.sh.
Raw root benchmark_data/gdn_mega/profile_current_20261027; production677c0ee/dfb01e1
unchanged. Four hardware profiles4K H16/H24/H48 and32K H16, simulator512tokens,
plus isolated stage clocks. All profiled FP64 elementwise checks pass;34 CPU
benchmark/report tests pass. Device4K681.30us, cube active24.9%;16/24 persistent
groups. Diagnosticjoins127.18us,statefeedback106.41us,reload19.76us,RHSwait.02us;
vector nextweights115.58us,nextdiag/Gamma164.32us,finalization83.87us. Use the
full data for both vector subcores. Stage clocks include transfers/waits and
cannot be added across core types. Diagnostic profiler686.14us (+0.7%).
Simulator trace181.81us is512tokens, not4K; traceviz aggregate occupancy counts
wait instructions and is NOT arithmetic utilization. No matmul probe or
instrumentation added to production. Main opportunities and previous failed
precompute/factored64 caveats are in profile.md; no optimization was implemented.
2026-10-27 — fast iteration tooling 400151f:
- `bash scripts/gdn_mega_docker.sh bash scripts/check_gdn_mega.sh [--quick]`
  snapshots source and compiles each H separately, parallel, then links a launch
  dispatcher. No production #ifdef/config proliferation. All7 specializations
  10.66s; quick H8/H16 builds ~10s. --source/--output accept isolated directories;
  --build-only permits CPU compilation alongside an NPU test. Never run NPU jobs
  concurrently. Every worker verifies loaded library SHA against its manifest.
- Full246test baseline505.95s fills6.2GB reference cache; cached candidate148.71s.
  Cache is opt-in GDN_REFERENCE_CACHE, enabled by tool. Hash inputs/source/torch
  version, validate every golden byte; no stored device outputs/pass results.
  Corruption raises, atomic write;35 checker/benchmark tests pass. Unset env for
  fresh FP64. Quick9fixedcatalog cases+checked256/4K timings is partial only.
- Production source677c0ee/librarydfb01e1 unchanged. Isolated async_joins removes
  final inverse-block materialization, applies F*R0, overlaps independent joins
  using dependency-specific events. Full246pass; initialH16singlehead screen
  4K654us vsbase692us, not yet matched allshape production evidence or promotion.
- User steered search toward (C,C_inv,A,J) with A including forward solves,
  Horner/Estrin/Paterson-Stockmeyer and cube/vector mapping. Continue structural
  search, not merely the~6% scheduling result. All experiments currently ignored
  under benchmark_data/gdn_mega/optimize_20261027. Keep numerical policy fixed.


2026-10-27 — structural search findings, docs/gdn_mega/algorithm_search.md/json:
- Production unchanged. C128/C_inv16 forward vector A remains best measured.
  Factored top join + asynchronous independent joins passes246tests; matched4K
  latest652–666us vsproduction697–700. Long32K gain smaller/variable; no breakthrough.
- Nine-case fixed screen: C64/inv16 835us, C64/inv32 960us pass9/9. C128/inv32
  forward A fails large nearly-correlated3011 amp128 with full and factored J;
  reverse A passes9/9 but843us fullJ/851us factored. C128/inv64forward8/9,untimed.
  Its initial UB-index-at176KB fault fixed by moving indices168KB; no device fault timed.
- C256/inv16 retiled prototype868us,8/9 (same high-amplitude case);reversealso8/9.
  Requires explicit enlarged scratch via ignored large_chunk_worker.py, streamed
  vector stripes, separate Q/K and output/state products, GM state/base residency.
  A2 L0C is128KiB, NOT256KiB: initial oversized tiles faulted and were retiled.
  OnlyH8/16 built. No production allocation or ABI change. Do not generalize its
  failure/slowness to every C256 schedule.
- Rotating chunk workers reuse spare KK-upper storage for FP32state and unused
  wy scratch for readiness counters (H<=16,B1,T>=512,GQA>=2). It passed9/9 but
  904us4K; reject this implementation, not evidence of numeric shortcut/fallback.
- CPU polynomial_sweep.py:1920 case/config combinations,8fixeddistributions,
  C64/128/256,Cinv8/16/32/64,Jfull/factored. Isolates fixedFP32 coefficient solve
  againstFP64, NOT full recurrence/kernel certificate. Forward/Horner inverse
  error~3–4e-8; Horner intermediates2.33e17 at64, manydependentmatmuls. Estrin/PS
  error~2e-4..4e-4 at16,order9..20 at32,~1e10 at64 fromcancellation.
- New CPU-only promising A: band-limited Newton X<-X(2I-LX),mask to growing
  known-correct band1->3->7->15...; samples peak<=1, inverse errors~forward.
  Exact reasoning: residual E has zero bands<=b;E² zero<=2b+1;mask preserves
  newlycorrectband. No decay truncation. NOT implemented/timed onNPU yet.
- HF32all onasync candidate fails6/9,then10/13expandedchecks,allfinite/integrity
  intact. Matched4K620–624vsFP32652–666;32K4948–4981vsFP325155–5363. Required
  atol atfixedrtol.001:weakg.0003465;identicalbeta1g0state.0039467;nearlycorr
  amp1 .00234357,amp32 .0749942,amp128 .2999769. Lastnearzero witnessref-.00812336
  err.124456;176009/1049600outputviolations despiteRMSE.000347. No tolerancechange.
  All13current/FP32candidatepass. Ordinary timingcellsHF32allfail butstilltimed.
- Detailed raw scripts/sources/logs ignored under benchmark_data/gdn_mega/optimize_20261027;
  reusable fastbuild/check tooling committed400151f,successentry57f0593. Latest
  user asks conclusions across A,C,Cinv,J; emphasize measured vs CPU-only scope.


2026-10-27 — cube Newton completed; user explicitly requested exploration cleanup:
- Native FP32 band-Newton cube A16 and A32, C128, factored/asynchronous J, both
  pass all246tests (154.60/155.61s), unchanged1e-4+1e-3*abs(FP64) policy. No ABI
  or allocation change. Initial bug: distance at184KiB clobbered by PTO's8KiB
  reserved scratch. Fixed to168KiB, identity176KiB; vectorize initial constants.
- Matched H16/Hg4 h0, T256/4096/32768, two variant orders: production82.4/693–695/
  5348–5430us; faster vectorcandidate78.6–79.2/659–662/5328–5331us;
  Newton16 105/1183–1191/9518–9620us; Newton32 109–111/1281–1286/10147–10205us.
- Hardware cube arithmetic158->182/184us; new cube wait8 373/432us, vector wait10
  249/322us. L2 read hit~99%. Reject roundtrip-heavy mapping, not numerical method.
  Extra9case matched audit passesall; amp128near-correlated worst error/allowance
  .9794 vector, .8823 Newton16, .8160 Newton32. No precision policy changes.
- All Newton/rejected prototypes, rawprofilerlogs, scripts and build/gdn_* dirs
  removed at explicit user request:2551files,6038218773bytes. Existing reportdata
  preserves measurements/hashes; historical rawpaths there are provenance, not live.
- benchmark_data/gdn_mega now ONLY baseline_lib,profile_current_20261027 (render
  inputs; removed diagnostic source/build),qwen_capture(userinputs+README),
  reference_cache(activefasttests),verified_candidate(2files:checked.so+manifest).
  The one verified speed improvement's exact source patch/basehashes are under
  retained_improvement in docs/gdn_mega/data/algorithm_search.json. It rebuilds
  against current7production sources using check_gdn_mega.sh. Productionlibhash
  dfb01e1 unchanged, candidate00144c5. Newtonsourcepatches explicitly discarded.

## 2026-10-27: token helpers + independent value columns promoted (`365b105`)

Production now includes the factored/asynchronous join candidate. H8, one
sequence and sufficiently many chunk/key-head tasks use 16 recurrence groups
(two 64-column state slices/head) plus eight token-preparation helpers. Other
shapes retain local preparation. The work test requires T>=1024 and at least
two chunk/key-head tasks per helper; Hg1 starts at 2048 on Ascend910B2.
No recurrent state handoff, numerical dispatch, gate clipping or new allocation.
Prepared inverses reuse cached raw-QK storage; row0/col127 is an unused ready
flag. Initialize flags before concurrent readers/writers, publish only after
vector exports and cube joins, and let vector RHS overlap the inverse wait/load.

Exact production binary 4dc29f4d passes 264 tests (178 numerical cases; 18 new
worker-layout regressions), unchanged FP64 elementwise policy, 171.38s pytest;
all seven head specializations build in 14.53s. Quick verification remains
nine cases and now benchmarks H8/H16, both T256/T4096. No additional production
files: seven files, 1342 physical source lines. Profile diagnostic generator
was adapted and its H16 measured outputs pass; existing profile figures remain
explicitly historical and do not describe the H8 producer/consumer path.

H8/Hg2 matched before/after production: 4K 685.76→520.10us (1.319x),
8K 1257.32→967.97us (1.299x), 16K 2558.70→1834.33us (1.395x),
32K 4956.42→3862.76us (1.283x). H16 gains 1.046x at4K and only1.011x
at32K. T256 H8 results fluctuate roughly ±2%; no uniform small-shape gain.
Early prototype group sweeps showed ~1.2–1.4x over the join-only candidate.
Do not confuse those comparisons with either pre-turn production or10748eb.

Rejected full token-prep barrier (H16/4K867us), streaming prep without column
splits (745us), leaf-only helpers (679us), and leaf-only early RHS (684us).
H12 two-way splitting leaves no helpers and did not win. Gate-panel16/64
factorization was implemented without reciprocal exponentials; panel64 passed
then-full246 tests but matched4K speedups were ~±1% and short inputs slower.
Discarded. Existing Gamma already uses exp(G_end-G_i); C16/-5 safety guidance
from FlashKDA uses BF16, not FP16. Gate work was not the limiting stage here.

Detailed tables, numerical metrics, build/source/library hashes and per-trial
times live in docs/gdn_mega/algorithm_search.md and its JSON. Rejected source,
generator, build and log artifacts are removed; the old isolated verified
candidate is obsolete because its improvement is now production.

Follow-up helper reuse: 48+48 immediate columns plus 32 delayed columns on each
K1 helper passed the nine-case screen but took 782.02us at H8/4K, versus ~520us
production. Helpers' combined preparation/recurrence tail dominates. Rejected
and removed; more busy cores alone does not mean a shorter critical path.

## 2026-10-28 — user requested unified-schedule rollback (`f381ae8`)

Current production has ONE head-parallel schedule, 1,189 physical lines in
seven sources. H8 helpers, state-column splitting, readiness flags and their
shape dispatch are removed. Factored/asynchronous joins remain. Sources match
retained_improvement.verification.source_sha256 exactly. Installed library
0bf6f7832dd1688b55f0962c92cbd42d048a31385a0a36f6d1e2e0a05e628691;
fresh 264 tests pass in169.30s, including all178 numerical cases. No ABI,
allocation or tolerance changes. Keep the18 former helper-layout cases.

Two unification prototypes passed9-case screens but were rejected: global
preparation plus head interleaving604/861/1918us at4K H8/H16/H48; local two-half
pipeline658/673/1393us, losing H8 gain and4–6% onshort inputs. Evidence saved in
existing algorithm_search.json/md; no experimental source/builds retained.

Fresh matched H8/Hg2 uniform vs removed custom at4K/8K/16K/32K:
658.61/1242.03/2515.80/4878.08 vs502.31/964.54/1829.29/3750.30us.
Custom gives1.29–1.38x throughput or22–27% lower latency. User explicitly accepts
this tradeoff for simpler code. Heatmap renderer now includes the historical
benefit note. Full speed grid is refreshed on installed uniform binary:63/63 current cells
pass; original10748eb fails63/63 but retains measured runtimes.
Profiler diagnostic generator adapted and smoke check passes. Old profile and
error figures retain their explicit historical checkpoint labels.

## 2026-10-29 — heatmap rerun and timing contract

Fresh full grid is in ignored `benchmark_data/gdn_mega/heatmap.json`; both
committed PNGs/Markdown reports now show 2026-10-29. Current library
`0bf6f7832dd1` passes 63/63; original `4bf4218a7473` fails 63/63 but is timed.
All input/reference/environment identities match. No kernel or timing changes.

Timing uses preallocated/reused inputs, outputs and workspace, the same initial
state on every call, five checked warmups, a 20-call graph and one untimed
replay, then 15 NPU-event timings divided by 20; report their median.
No external clearing or cache flush between calls. Final timed output/state
are checked outside timing. README and generated heatmap.md explain this.
Current medians differ from the October 28 run by -4.6% to +2.0% across cells
(median -2.3%); do not attribute these variations to an optimization.

## 2026-10-29 — merged upstream/main 13a6870

Upstream added the original mega kernel independently, causing add/add
conflicts. Keep the optimized seven-source implementation and consolidated
264-test suite; the six original stage .cpp files are superseded. Preserve
upstream tri_inv_col_sweep's is_bsnd_format/cu_seqlens binding arguments.

All repository lint passes with upstream defaults. clang-format sorts includes:
fused_pipeline.hpp now explicitly includes inverse.hpp and recurrence_pipeline.hpp.
The fast builder matches dispatch whitespace independently. Kernel tokens
(apart from include dependencies) and numerical/timing ASTs are unchanged.
Fresh CMake build, full host binding type check, seven-head fast build (10.48s)
and 264 tests (167.11s) pass. No ABI, numerical-policy or schedule change.
