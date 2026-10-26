# GDN mega kernel

The maintained reports are:

- [Speedup heatmap](heatmap.md): original versus current latency across heads and sequence lengths.
- [Error heatmaps](error_heatmap.md): maximum absolute output error and worst elementwise tolerance ratio for both implementations.
- [Input ranges](input_ranges.md): FLA-derived distributions, inverse overflow and cancellation.

`data/` contains the measurements and provenance used by these pages; `figures/`
contains their PNG/SVG plots. Experiment dumps and superseded checkpoint reports
are available in Git history (before this cleanup: `a83187b`). They are not part
of the maintained report set.

## Current implementation and correctness

**Known precision failures remain in kernel `aac5a4d`, even with beta in [0,1].**
The additional critical audit passes 33/56 cases and fails 23/56; see
[the critical precision audit below](#critical-precision-audit-beta-in-01).

The existing regression suite now passes **117 tests**; the latest log is in
[the correlation study](data/correlation_risk.json). The earlier 116-test run
remains recorded in `data/validation.json`.
The full FP64 grid passes **63/63 current shapes**, with zero elementwise violations;
the original fails all63 shapes under this stricter policy.
[Validation log, source hashes and library fingerprint](data/validation.json).
Kernel sources and numerical tolerances have not changed during this audit.
A green ordinary-input suite is not a precision signoff for slow decay.
The tests include an independent FP64 token oracle, supplied initial state,
ragged/empty sequences, FLA-derived stress cases and poisoned/repeated workspace.

Every raw-output and final-state element must meet
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005.
This is an elementwise mixed absolute/relative bound, not RMSE-only acceptance
or a uniform maximum absolute error of 1e-5. Original-kernel failures remain visible.

The current kernel batches independent inverse joins, eliminates vector-solve
transposes/broadcast copies, starts next-chunk weighting earlier and transfers
whole attention/Gamma tiles. It retains FP32 state and stable block solves.
The requested 3× speed target remains unmet.

There is **one launch, two global phases and seven production source files**
(1302 physical lines at `aac5a4d`, versus 5241 originally). Arguments, tensor dtypes,
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
Strong coupling can trigger the production kernel's slower token recurrence.
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
