# GDN mega kernel

The maintained reports are:

- [Sequential fallback removal](#sequential-fallback-removal): latest captured-input speedups, validation and retained failures.

- [Speedup heatmap](heatmap.md): original versus the pre-removal `aac5a4d` checkpoint across heads and sequence lengths.
- [Error heatmaps](error_heatmap.md): maximum absolute output error and worst elementwise tolerance ratio at that checkpoint.
- [Input ranges](input_ranges.md): FLA-derived distributions, inverse overflow and cancellation.
- [Qwen capture replay](#qwen-forward-pass-capture-replay): actual correlated keys and gates, production versus parallel-only accuracy.

`data/` contains the measurements and provenance used by these pages; `figures/`
contains their PNG/SVG plots. Experiment dumps and superseded checkpoint reports
are available in Git history (before this cleanup: `a83187b`). They are not part
of the maintained report set.

## Current implementation and correctness

**Production now always uses the parallel chunk pipeline; the sequential
fallback and its detection overhead have been removed.** The final regression
run after removal passes **116 tests and fails 2**: `nearly_correlated_beta1`
and `delayed_nearly_correlated`, both synthetic amplitude-32 cases with beta=1
and no decay. These failures remain ordinary failing tests, with unchanged
tolerances; the whole suite is not green.

The expanded current-kernel audit passes **137/160 synthetic cases** and fails
23, including three unit-scale 32K near-lossless-memory cases and 20 larger-scale
stress cases. All remain finite. The [accuracy regimes and failure severity](error_heatmap.md#accuracy-regimes-where-the-current-kernel-passes-and-fails)
separate these from the 126/126 passing real-input cases and show absolute witness
errors, allowances, violation counts and priorities for further validation.

Historical results below retain their original library fingerprints. Before
removal, `aac5a4d` passed 118 regression tests but failed 23/56 additional
critical cases, even with beta in [0,1]. Its saved FP64 heatmap passes 63/63
shapes, while the original implementation fails all 63 under this policy.
Those grids have not been relabeled as measurements of the new kernel.
[Historical validation evidence](data/validation.json) and
[critical audit](#critical-precision-audit-beta-in-01).
The tests include an independent FP64 token oracle, supplied initial state,
ragged/empty sequences, FLA-derived stress cases and poisoned/repeated workspace.

Every raw-output and final-state element must meet
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005.
This is an elementwise mixed absolute/relative bound, not RMSE-only acceptance
or a uniform maximum absolute error of 1e-5. Original-kernel failures remain visible.

The current kernel batches independent inverse joins, eliminates vector-solve
transposes/broadcast copies, starts next-chunk weighting earlier and transfers
whole attention/Gamma tiles. It retains FP32 state and stable block solves.
The ordinary-input 3× speed target remains unmet; removing the fallback targets
the additional cost on correlated real inputs.

There is **one launch, two global phases and seven production source files**
(1171 physical lines after fallback removal, versus 1302 at `aac5a4d` and 5241 originally). Arguments, tensor dtypes,
output scaling and caller allocation sizes are unchanged. Internal scratch
layouts and the execution model are documented in the
[kernel README](../../csrc/kernel/gdn_mega/README.md).

## Reproduce

From the repository root, using the prepared CANN container:

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh python -m pytest tests/test_gdn_mega.py tests/test_gdn_mega_validation.py -q
bash scripts/report_gdn_mega.sh all
```

Use `latency` or `errors` instead of `all` to regenerate one report. Use `render`
to regenerate both Markdown pages and all plots from saved data without running
NPU measurements. Markdown statistics, tables, version labels and figures come
from the same JSON; do not edit the generated pages by hand.

Latency uses the frozen original timing data in `data/original_latency.json`;
it explicitly labels the original's older acceptance policy. Fresh matched error
measurements require the original library at
`benchmark_data/gdn_mega/baseline_lib/libno_workspace_kernel.so` and current at
`build/lib/libno_workspace_kernel.so`. The scripts verify actual loaded hashes,
matching inputs/reference results, complete grids and visible failures. NPU jobs
run serially. Every current latency cell and every matched error cell now uses
the independent CPU PyTorch FP64 token recurrence. The separate selected-shape
FP64 report is superseded by the full FP64 grid.

The small entry scripts are `gdn_mega.py` (single-shape timing),
`gdn_mega_sweep.py` (latency grid), `gdn_mega_errors.py` (matched accuracy), and
the two `plot_gdn_mega*.py` renderers. Shared Markdown generation lives in
`gdn_mega_report_text.py`. `gdn_mega_ranges.py` measures the documented input ranges;
it contains no experimental solver implementations.

## Golden, input coverage and fast checks

The golden consumes the actual rounded input tensors, converts them to CPU FP64,
and computes each token in order:

```text
X = exp(g_t) * S
residual = beta_t * (v_t - k_t^T X)
S = X + k_t * residual^T
out_t = q_t^T S
```

There is no chunk inverse, power truncation, decay threshold, or production
intermediate in this reference. Supplied initial states and sequence boundaries
are honored. The ordinary unit tests also retain an FP32 chunked reference,
which is separately crosschecked against this independent recurrence.

Beta is update strength; `exp(g)` is the state decay. The ordinary performance
heatmap uses uniform beta in [0,1) and log-sigmoid Gaussian gates. It is not a
worst-case performance claim. The independent audit additionally covers zero
and weak decay, beta=1, doubled sigmoid beta approaching 2, correlated/positive
keys, large values/states, positive log gates, and packed/empty sequences.
Strong coupling triggered a slower token recurrence in the pre-removal kernel;
production now keeps every head on the parallel path.
See [input ranges](input_ranges.md) for the precise distribution limits.

Checking remains exhaustive: finiteness, maximum absolute error, every-element
mixed bound, and final state. RMSE cannot rescue a single violating element.
Percentile sorting is optional diagnostic work and is disabled by default.
The timing worker checks raw output directly, removing an unnecessary scale/
unscale round trip and duplicate scaled-output statistics. Since the output
scale is less than one, the raw bound also implies the corresponding scaled
bound. Repeat statistics
are reused only after comparing every output/state bit and checking that the
oracle tensors have not changed. A changed element, NaN, or changed golden
forces revalidation; regression tests cover these cases.

FP64 golden construction and CPU checking are outside kernel timings. The
single-pair benchmark records their wall times as `validation_seconds`, making
slow validation visible during optimization. The golden's serial token loop
and FP64 state arithmetic still cost substantially more than the NPU kernel.

The default grid includes 24/8 and 48/16 as well as the previous group-of-four
pairs. To run just those rows, use the same options on either grid script:

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega_sweep.py \
  --baseline-report docs/gdn_mega/data/original_latency.json \
  --head-configs 24/8 48/16 --output benchmark_data/gdn_mega/pairs_latency.json
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega_errors.py \
  --head-configs 24/8 48/16 --output benchmark_data/gdn_mega/pairs_errors.json
```

There were no saved historical original timings for the new rows. The script
checks and times the original even when accuracy fails, retaining the failed
status. Each heatmap cell shows the ratio, our median total µs, and the baseline
median total µs. Only a failing implementation's runtime gets an asterisk; ratios
and passing runtimes remain unmarked. Numerical
failures and runtime are separate measurements, and all errors remain visible.

### Measured checker cost

CPU wall time for T4096, H16/Hg4, D128; four CPU threads; median of three
runs on identical tensors. This isolates checking from NPU transfer and timing.

| Work | Before | Current | Speedup |
|---|---:|---:|---:|
| First output/final-state check | 1.379 s | 0.222 s | 6.22× |
| FP64 reference | 3.182 s | 2.995 s | 1.06× |
| Unchanged-output recheck | 4.59 ms | 4.91 ms | approximately unchanged |

The old/new FP64 output and state were identical on this measurement. Oracle
regression tests separately cover weak/no decay, beta near 2, nonzero state,
and packed empty sequences. Exact timings and source fingerprints are retained
in [validation evidence](data/validation.json). These are validation speedups;
the production NPU kernel is unchanged.

For matched error reports, both library workers share one fresh CPU golden per
cell through a temporary file. Input/source identity and the full golden-content
hash must match before reuse. The directory is deleted after the report, and
changed or corrupted entries fail closed. This avoids running the serial FP64
recurrence twice without reusing any device output or skipping any comparison.

FP64 is an audit golden, not a requirement for the production kernel's arithmetic.
The earlier ordinary H16/Hg4 crosscheck measured a maximum raw-output difference
of 3.62e-6 between the FP32 chunked reference and independent FP64 recurrence at
T4096 (1.53e-6 at T128). This is appreciable relative to the 1e-5 absolute tolerance
floor, although the allowance also includes the relative term. The independent
token algorithm avoids sharing the production kernel's chunking errors. Full
FP64 grids are useful for milestone validation; short fixed cases and reuse of
an unchanged golden are preferable for frequent optimization iterations.

## Critical precision audit: beta in [0,1]

This section records the **pre-removal** `aac5a4d` kernel. Its counts and
fallback-specific diagnosis do not describe the new kernel after removal.

**Outcome: 33 passed, 23 failed out of 56 additional cases**, with five launches
per case against the independent CPU FP64 recurrence. The original 36-case audit
also passes; its doubled-beta case is outside the requested beta range and is
excluded from conclusions about production inputs. The separate 116-test pytest
suite passes but does not include the new critical suite. These results expose
coverage gaps rather than establish that the kernel is precision-clean.

The additional cases test beta endpoints, interior rates 0.25/0.5/0.6/0.9, random
beta in [0,1), keys around the coupling threshold, exact/nearly correlated and
positive keys, log gates 0/-1e-8/-1e-7/-1e-6/-80, lengths through 32768, supplied
and zero initial state, packed tails, and 24/8, 48/16, 64/16 head configurations.
All 56 cases have beta in [0,1] and nonpositive log gates. All kernel outputs are
finite. All input/guard checks and scratch-independence checks pass. The ideal
FP16-rounded reference passes every case, so the failures are avoidable internal
arithmetic errors, not an insufficient output range. **Every failing case would
pass the RMSE-only threshold.**

Selected raw-output results (unit-scale values/state unless noted):

| Case | Tokens | beta | Log gate | Max abs error | Worst error/allowance | Violating elements |
|---|---:|---:|---:|---:|---:|---:|
| Exact correlated keys | 4097 | 0.6 nominal | -1e-7 | 2.147e-3 | 29.20 | 99,251 |
| Exact correlated keys | 4097 | uniform[0,1) | -1e-7 | 2.122e-3 | 29.82 | 99,480 |
| Exact correlated keys | 32768 | 1 | -1e-7 | 5.482e-3 | 265.81 | 7,764,502 |
| Nearly correlated keys, seed 3011 | 1025 | 1 | -1e-6 | 1.951e-3 | 1.56 | 4 |
| Nearly correlated keys, zero initial state | 4097 | 1 | -1e-6 | 1.058e-3 | 4.05 | 3,670 |
| No updates, supplied initial state | 32768 | 0 | -1e-7 | 1.965e-3 | 1.40 | 28 |

The stored FP16 beta for nominal 0.6 is 0.60009765625. The failures above do not
require the larger amplitude 32/128 stress cases. For the 32K beta=1 example,
inputs satisfy exactly unit key norm, with max|V|=3.9043 and max|initial|=3.8945.
At one output element the FP64 golden is -0.000680124823 and the kernel returns
-0.00351905823: error 2.83893e-3 against an allowance of 1.06801e-5.

### Confirmed mechanism and limits of the diagnosis

For exact keys k=e0 and beta=1, the state has an independent analytic solution:
row 0 is the current value vector, while all other rows are their initial values
multiplied by exp((t+1)*g). Direct FP64 evaluation agrees with the token oracle
to at most 1.11e-12 on the six selected output witnesses.

The correlated-key fallback rounds exp(g) to FP32 and repeatedly multiplies the
FP32 state by it. Its compensation covers the additive update, not the lost
precision of decay multiplication. At g=-1e-8 the rounded factor is exactly 1.
For g=-1e-7, an initial state element 3.013671875 should become 3.00381283675 after
32768 steps; the kernel returns 2.998046875. A scalar FP32 repeated-multiplication
emulation reproduces this value exactly, and matches all six selected final-state
witnesses. Removing decay makes the matched correlated-key controls pass.

The beta=0 case also reveals a smaller failure in the normal chunked path;
that path needs further investigation. No production fix has been applied,
and no tolerance has been relaxed. The ordinary heatmap's 63/63 current passes
remain valid for its stated input distribution; they do not cover these cases.

[All cases, five repeats, source/binary hashes and analytic witnesses](data/precision_audit.json) ·
[Per-case CSV](data/precision_audit.csv).

Reproduce the supported-range audit (currently exits 1 because of numerical failures):

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite critical --output benchmark_data/gdn_mega/critical_precision.json
# Smaller failing case, beta strictly inside [0,1]:
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite critical --cases interior_beta_0.6_correlated \
  --output benchmark_data/gdn_mega/precision_repro.json
```

## How much does key correlation affect precision?

This section records the pre-removal kernel and its isolated parallel-only
diagnostic. The saved source/library fingerprints identify those versions.

We compared the unchanged production kernel with an isolated build that disables
only its sequential fallback. Both builds consumed identical input tensors and
were checked against the independent CPU FP64 token recurrence, using the same
elementwise bounds for output and final state. There were **71 matched cases,
five launches per case per build (710 launches)**. The production library and
source were not modified. [Full evidence and build patch](data/correlation_risk.json)
and [per-case results](data/correlation_risk.csv) retain failures and fingerprints.

The controlled sweep uses normalized keys with a shared direction and independent
orthogonal noise. Target average cosine similarity ranges from 0 to 1; changing
correlation preserves Q, V, beta, gates and initial state exactly. Most cases use
4096 tokens and 8 value / 2 key heads, beta 0.5 or 1, and constant log gates
0, -0.001, -0.1 or -1. Two cases extend to 32768 tokens. The zero-correlation
control uses noise in a 127-dimensional subspace, so it is not exactly the
heatmap's independent 128-dimensional Gaussian distribution.

**All 56 unit-amplitude cases pass on both builds**, including the 32K cases
and six supplied-initial-state controls with state standard deviation 1.
At unit scale, the worst absolute output error is about 0.002, predominantly
the unavoidable final FP16 rounding. This new set of 56 cases is distinct from
the earlier 56 critical cases above: tiny log gates (-1e-8 to -1e-6) still have
known failures and are not certified by this sweep.

Larger values expose cancellation errors. The following matched controls use
4096 tokens, beta=1, no decay, V standard deviation 32 and initial-state standard
deviation 3.2. Each row checks 4,194,304 raw-output elements.

| Mean key cosine target | Production output violations | Parallel-only output violations | Parallel-only worst error / allowance | Parallel-only state violations |
|---|---:|---:|---:|---:|
| 0 | 41 | 41 | 1.84 | 6 |
| 0.25 | 1 | 70 | 2.51 | 3 |
| 0.50 | 0 | 212 | 3.05 | 9 |
| 0.75 | 3 | 431 | 3.69 | 21 |
| 0.95 | 11 | 333 | 4.58 | 15 |
| 1.00 | 227 | 1 | 1.03 | 0 |

For the parallel path, increasing mean correlation from 0 to 0.95 raises the
error at the worst tolerance violation from **2.17e-5 to 4.76e-5**. Violating
outputs rise from 0.00098% to 0.00794%. The latter witness is reference
0.0004119455 versus output 0.0003643036, with allowance 1.04e-5. These witness
errors are not the global maximum absolute error: at correlation 0.95 the global
maximum is 0.05975, equal to the maximum error from simply rounding the FP64
golden to FP16. All ideal FP16-rounded goldens pass the elementwise policy.

Correlation does not monotonically worsen precision: exactly identical keys
can be easier than nearly correlated keys. Nor is the fallback uniformly safer.
With correlation 0.95 and log gate -0.001 at amplitude 32, production has
**6,506 output violations versus 231 on the parallel path**. Its worst witness
has absolute error 2.11e-4 and exceeds its allowance by 20.71 times, versus a
worst ratio of 2.41 for the parallel path. In contrast, the existing
`nearly_correlated_beta1` regression passes production but fails without fallback
at just one of 526,336 outputs: error 1.324e-5 versus allowance 1.249e-5 (6% over).

Across all 71 cases, production passes 60 and the parallel-only build passes 58.
All outputs are finite, inputs and guards remain intact, and repeated launches
are bitwise identical. **Every failure would pass RMSE-only acceptance.** No
overflow or explosive error appeared in this study, but the strict failures
remain real. This is a controlled synthetic sensitivity study, mostly one seed,
not a measured distribution of model activations or a guarantee for all inputs.

The production fallback is activated for all eight heads even at target cosine
0.25 with beta=1 and no decay: it checks the worst pair, not the average cosine.
All 63 saved performance-heatmap inputs avoid fallback; a conservative bound on
their coupling coefficients is at most 0.3751, below the 0.5 trigger. That grid
therefore does not establish correlated-key performance. The verified input
hashes and bounds are included in the full evidence.

Reproduce the 68 controlled cases on the current production library (exits 1
because the strict failures are retained):

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite correlation --output benchmark_data/gdn_mega/correlation.json
```

The three additional existing regressions, isolated diagnostic patch, compiler
command, library selection and exact case lists are recorded in the JSON. The
ordinary regression suite passes 117 tests; this does not override the separate
stress-audit failures. No production fallback was removed in this study.

## Qwen forward-pass capture replay

This initial comparison used the pre-removal production library and an isolated
build with fallback dispatch disabled. Their saved measurements remain unchanged.

**Both pre-removal production and the isolated parallel-only build pass all 126 cases,
with zero elementwise violations in output and final state.** This study uses
the user's actual `Qwen/Qwen3.5-0.8B` forward-pass captures, rather than fitting
a synthetic gate distribution. [Full results and capture provenance](data/qwen_capture.json)
and [per-case CSV](data/qwen_capture.csv) retain exact input and library hashes,
headwise gate/correlation/error statistics, all five repeats and the diagnostic
build patch. Production source, allocations, interfaces and tolerances are unchanged.

Coverage is all six captured prompts and all 18 linear-attention layers:
**108 full traces, 346–1024 tokens each**. An additional 18 continuation cases
cover every layer of `code_python`: process its first 512 tokens with the
independent CPU FP64 recurrence, round the resulting initial state to the
kernel's FP16 interface, and check the remaining 512 tokens against an FP64
oracle starting from that same rounded state. No device result supplies the golden.
There are 16 value and 16 captured key/query heads, each dimension 128.

Captured Q/K/V are already FP16; they are replayed without normalization or
scaling. Gates remain their captured FP32 values. Beta is converted from FP32
to FP16 as required by our ABI, before both kernel and golden consume it.
Thus this measures kernel arithmetic on its actual inputs, not end-to-end
agreement with the original FP32 model. No gates or beta values are clipped,
filtered, replaced by constants or shuffled across heads/tokens.

| Build | Cases passing | Output violations | State violations | Max abs output error | Max abs state error | Worst output error / allowance | Worst state error / allowance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production, with fallback | 126/126 | 0 | 0 | 0.00195618 | 0.00337935 | 0.484879 | 0.479069 |
| Parallel-only diagnostic | 126/126 | 0 | 0 | 0.00195250 | 0.00337935 | 0.484879 | 0.476524 |
| Ideal FP16 rounding of the FP64 golden | 126/126 | 0 | 0 | 0.00195250 | 0.00337935 | — | — |

The error allowance is unchanged: `1e-5 + 1e-3*abs(reference)` for every raw
output and state element. The global maximum errors are dominated by final
FP16 rounding. Each build checks 198,623,232 output and 33,030,144 state elements
per complete pass, with five launches per case: **1,260 total device launches**.
All results are finite, inputs/guards remain intact, and scratch-poisoned repeats
are bitwise identical.

Correlation is substantial in these real inputs: some within-chunk keys are
identical, and production takes the sequential fallback for **1,192/1,728
full-trace head instances (69.0%)**. The observed dispatch flags agree with an
independent FP64 calculation of the coupling trigger in every head. The
parallel-only build disables that dispatch and still passes every case.
The fallback was therefore unnecessary for meeting our accuracy policy on
these captures; it is not a rare path in this workload. No latency measurement
or fallback removal is included in this audit.

The measured gate median is -0.006174; 36.97% of gates exceed -0.001, and 1.60%
exceed -1e-6. Weak gates really occur, but these traces retain their natural
temporal/head structure. They do not impose one tiny constant gate across a
32K sequence. Beta's median is 0.353 and maximum |V| is 19.40625. See
[measured ranges](input_ranges.md#measured-qwen-forward-pass-inputs) for details.

These results cover one model checkpoint, six prompts and contexts up to 1024
tokens. They do not resolve the earlier synthetic long-context failures or
establish that every correlated workload is safe without fallback.

Reproduce the production audit using the supplied capture file:

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite capture --capture benchmark_data/gdn_mega/qwen_capture/gdn_keys.pt \
  --output benchmark_data/gdn_mega/qwen_capture/production.json
```

The capture is an untracked local copy of the user's `gdn_keys.pt`; its original
path, SHA256, capture script and source README are recorded in the JSON. The
large tensor file is not committed. Replaying the parallel-only diagnostic
uses the isolated library selected through `LD_LIBRARY_PATH`, with its exact
patch, build commands and loaded-library fingerprint recorded alongside results.

The updated replay tooling passes the complete **118-test regression suite**
(71.37 seconds); the log and source fingerprints are saved with the results.

## Sequential fallback removal

Commit `677c0ee` makes production always use the parallel chunk algorithm. The sequential
recompute routine, its private dot-reduction helpers, the per-chunk maximum
coupling reduction, flag stores, cache invalidation and threshold dispatch are
removed. No dormant branch or tuning option remains. This removes 131 production
lines (1302 → 1171), with the same seven source files, one launch and two global
phases. Input/output types, shapes, caller allocations and numerical tolerances
are unchanged. Reserved scratch capacity remains allocated for ABI compatibility.

### Captured Qwen performance and accuracy

All **126 cases pass before and after removal**, with zero elementwise output
or final-state violations, including five poisoned-scratch launches and checks
after timed graph replay. These are the same 108 full traces and 18 supplied-state
continuations described above. New output/state numerical metrics exactly match
the previously audited parallel-only diagnostic on every case.

The **median paired speedup over 108 full traces is 17.00×** (range 12.05–18.23×).
The 18 continuation cases have median 15.02×; all 126 together have median 16.06×.
This compares against the immediately preceding production kernel with fallback,
not the branch's original six-stage algorithm or end-to-end model inference.

Each kernel time is the median of 15 NPU-event trials × 20 allocation-free
calls in a graph, following five checked warmups. NPU jobs run serially; CPU
reference, allocation and checking are outside the timed region. The table
summarizes the 18 layer calls for each full prompt. Its last column is the median
of paired speedups, which need not equal the ratio of the two displayed medians.

| Captured prompt | Tokens | Before: median total µs | After: median total µs | Median paired speedup |
|---|---:|---:|---:|---:|
| prose_wiki_style | 709 | 2567.30 | 167.60 | 15.37× |
| prose_story | 749 | 2709.21 | 168.88 | 16.06× |
| chat_qa | 346 | 1285.60 | 105.16 | 12.24× |
| code_python | 1024 | 3734.57 | 208.46 | 17.93× |
| code_c | 1024 | 3691.59 | 208.31 | 17.71× |
| code_rust_ts | 1024 | 3682.41 | 208.43 | 17.66× |

The old threshold triggered fallback on 69.0% of full-trace head instances,
although the parallel computation passes on those inputs. Removing that path
eliminates the full-sequence token-by-token recomputation. No new input scaling,
gate assumptions, clipping or reference adjustment contributes to these speedups.

### Ordinary independent-key controls

All seven shapes pass the same FP64 elementwise checks before/after graph
replay. These use normalized independent Gaussian Q/K, H16/Hg4, supplied initial
state and the ordinary benchmark gates/beta. They avoided fallback previously.
Removing detection overhead gives 1.019–1.095× speedups (1.8–8.6% less time);
these are seven selected controls, not a regenerated full heatmap.

| Tokens | Before total µs | After total µs | Speedup |
|---|---:|---:|---:|
| 256 | 86.31 | 84.57 | 1.021× |
| 512 | 127.84 | 125.52 | 1.019× |
| 1024 | 212.22 | 205.98 | 1.030× |
| 4096 | 721.66 | 693.81 | 1.040× |
| 8192 | 1433.07 | 1366.83 | 1.048× |
| 16384 | 2869.37 | 2687.06 | 1.068× |
| 32768 | 5900.52 | 5390.88 | 1.095× |

### Remaining strict failures

The complete regression suite has **116 passing tests and 2 failing tests**.
The failures are retained unchanged, without xfail markers or looser bounds:

| Synthetic case | Output violations | State violations | Worst output error / allowance | Worst state error / allowance |
|---|---:|---:|---:|---:|
| `nearly_correlated_beta1` | 1 / 526,336 | 0 / 262,144 | 1.0601 | 0.6643 |
| `delayed_nearly_correlated` | 2 / 394,240 | 1 / 131,072 | 1.5032 | 1.0799 |

Both use amplitude 32, beta=1 and no decay; maximum |V| is 124.9375 and maximum
initial-state magnitude is about 12.5. The first failing output witness has
error 1.324e-5 versus allowance 1.249e-5. The delayed case's worst output witness
has error 2.006e-5 versus allowance 1.334e-5; its failing state witness has error
1.087e-5 versus allowance 1.006e-5. These are internal arithmetic errors: ideal
FP16 rounding passes. These cases passed with the old fallback, so removal is
not an accuracy improvement for every supported input. The user requested
removal after the real-input audit; the captured-workload results do not justify
hiding the remaining stress failures or claiming universal precision.

[Matched measurements, source/library fingerprints and failure witnesses](data/fallback_removal.json) ·
[Captured per-case latency/error CSV](data/fallback_removal.csv).
Identical numerical dictionaries across five repetitions are stored once in
this report, after verifying equality, alongside each individual integrity and
scratch-independence result. All timed replay outputs are checked separately.

Reproduce the current captured accuracy/latency run with the existing script:

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite capture --timing \
  --capture benchmark_data/gdn_mega/qwen_capture/gdn_keys.pt \
  --output benchmark_data/gdn_mega/fallback_removal/capture_after.json
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega.py \
  --lengths 256 512 1024 4096 8192 16384 32768 \
  --heads 16 --key-heads 4 --initial-state \
  --output benchmark_data/gdn_mega/fallback_removal/ordinary_after.json
```

The old library is preserved locally under
`benchmark_data/gdn_mega/fallback_removal/before_lib/`; its SHA256 starts `619b7df`.
Select it through `LD_LIBRARY_PATH` to reproduce the before measurements.
The current library SHA256 starts `dfb01e1`. The older speed/error heatmaps
remain explicitly tied to their pre-removal checkpoint.
