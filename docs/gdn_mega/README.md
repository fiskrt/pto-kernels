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

Kernel **`a365dad` fixes the precision failures found within beta in [0,1]**.
All **56 critical cases pass**, compared with 23 failures before the repair.
The full audit passes 101 case entries across 505 launches, and the regression
suite passes **125 tests**. Four critical cases are also retained in the ordinary
regression suite; five additional cases exercise changing decay and numerical
formula boundaries. See [the precision audit](#critical-precision-audit-beta-in-01).
[Validation log, source hashes and library fingerprint](data/validation.json).

The saved heatmap pages still describe measurements of **`aac5a4d`**, which passed
its 63 ordinary-input shapes but failed the later critical audit. They are not
new measurements of this repair. Fresh repair timings appear below.

Every raw-output and final-state element must meet
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005.
This is an elementwise mixed absolute/relative bound, not RMSE-only acceptance
or a uniform maximum absolute error of 1e-5. Original-kernel failures remain visible.

The current kernel batches independent inverse joins, eliminates vector-solve
transposes/broadcast copies, starts next-chunk weighting earlier and transfers
whole attention/Gamma tiles. It retains FP32 state and stable block solves.
The requested 3× speed target remains unmet.

There is **one launch, two global phases and seven production source files**
(1390 physical lines at `a365dad`, versus 5241 originally). Arguments, tensor dtypes,
output scaling and caller allocation sizes are unchanged. Internal scratch
layouts and the execution model are documented in the
[kernel README](../../csrc/kernel/gdn_mega/README.md).

## Reproduce

From the repository root, using the prepared CANN container:

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh python -m pytest tests/test_gdn_mega.py tests/test_gdn_mega_validation.py tests/test_gdn_mega_errors.py tests/test_gdn_mega_benchmark.py -q
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

**Repair `a365dad`: 56/56 critical cases pass; previously 33 passed and 23 failed.**
The complete audit passes 101 case entries (45 standard/regression cases plus
56 critical cases), each launched five times. Four critical inputs also appear
under regression names in the standard suite. The separate pytest suite passes
125 tests, including the newly retained failures and five gate-transition cases.

Every launch checks every raw-output and final-state element against the
independent CPU FP64 token recurrence. All outputs are finite, all input/guard
checks pass, and all scratch-poison repeats are bitwise identical. The policy
remains `abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus RMSE < 0.005.
The worst error/allowance ratio across the complete audit is **0.6043 for output**
and **0.4878 for final state**; ratios above one fail. These are measured bounds
on these cases, not a proof for every possible FP16 input.

The original 56 case definitions, measured input ranges, and FP16-rounded FP64
reference metrics match before and after the repair. Beta remains in [0,1] for
all critical cases. The standard suite also retains one FLA beta>1 stress case,
which passes but is outside the requested production range. No tolerances,
outlier filters, clipping, xfail markers or reference-dependent paths were added.

### Which inputs are numerically difficult?

Here **alpha = exp(g)** is state decay, **g** is the log gate, and **beta** is
update strength. They have different roles.

| Quantity | Difficult inputs in the audit | Reason |
|---|---|---|
| Alpha / log gate | g = -1e-8, -1e-7, -1e-6, hence alpha almost one | Errors persist over long sequences; FP32 exp(-1e-8) rounds to exactly one |
| Beta | 0.6, 0.9, 1, and random beta in [0,1) | Strong updates make cancellation and residual/update rounding consequential |
| Keys | Identical or nearly parallel, including exactly unit-norm k=e0 | Similar update directions expose chunk-solve sensitivity; other directions retain long-lived state |
| Sequence length | Up to 32768 tokens | Repeated decay rounding accumulates |
| Values / initial state | Unit-scale failures existed; amplitudes 32 and 128 exposed additional errors | Large intermediates can cancel to small results where absolute error dominates |

Beta=0 also exposed a smaller chunked-path failure: repeated decay alone was
inaccurate. Beta=0.25 and 0.5 passed the original interior-rate samples, but that
is not a universal safe-beta threshold. More negative gates usually shorten
error persistence; strongly decaying and zero-decay cases remain in the tests.

For a fixed unit key, the homogeneous state transition has eigenvalue alpha
perpendicular to the key, and alpha*(1-beta) along it. Thus alpha near one
preserves the orthogonal state even when beta is one. With beta in [0,1] and
alpha <= 1 these exact unit-key cases are mathematically stable: the observed
failures were arithmetic errors, not an intrinsically exploding recurrence.
If keys are not normalized, the parallel factor is alpha*(1-beta*||k||²), so
key norm matters in addition to correlation.

### What changed

1. Small decay uses **S + S*expm1(g)** with compensated addition. A degree-seven
   polynomial evaluates expm1 on |g| <= 1/8, avoiding subtraction from a rounded
   exponential. The same method applies to each chunk's accumulated gate.
2. The chunked state now carries FP32 addition compensation across chunks in
   the existing `wy_workspace_a1/a2` buffers. This fixes the beta=0 long-sequence
   drift without new caller allocations.
3. The conservative token path retains low-order bits through dot products,
   residuals and rank-one updates. Splitting FP32 operands into two 12-bit
   components gives exact FP32 products with the original FP16 inputs;
   error-free TwoSum reductions and additions retain their rounding errors.
   The components keep the full FP32 exponent range.

Public arguments, input/output dtypes, shapes, scaling and allocation sizes are
unchanged. There is still one launch, two global phases and seven production
source files. The existing strong-coupling dispatch rule is unchanged.

Selected before/after **worst output error/allowance ratios**:

| Case | Tokens | Before | Repaired |
|---|---:|---:|---:|
| Exact correlated keys, beta=1, g=-1e-7 | 32768 | 265.815 | 0.486 |
| Exact correlated keys, beta=0.6, g=-1e-7 | 4097 | 29.202 | 0.484 |
| Nearly correlated keys, beta=1, amplitude 128, seed 3019 | 1025 | 58.528 | 0.488 |
| Beta=0, supplied state, g=-1e-7 | 32768 | 1.404 | 0.485 |

The original decay diagnosis also has an independent analytic check: for k=e0
and beta=1, state row zero is the current value, while other rows equal their
initial value times exp((t+1)*g). Six selected analytic output witnesses agree
with the FP64 recurrence within 1.11e-12. For example, initial state 3.013671875
with g=-1.000000011686e-7 should become 3.00381283675 after 32768 steps; the old
kernel returned 2.998046875. Repeated FP32 multiplication reproduces that old
error exactly. The saved report retains these original witnesses and failures.

Fresh median total kernel times, H/Hg=16/4, nonzero initial state, ordinary input
distribution, 15 trials of 20 graph-captured launches:

| Tokens | Before (µs) | Repaired (µs) | Increase |
|---|---:|---:|---:|
| 256 | 84.812 | 92.251 | 8.8% |
| 4096 | 693.686 | 761.910 | 9.8% |
| 32768 | 5561.383 | 6091.685 | 9.5% |

Both implementations pass these ordinary timing inputs. This is a correctness
repair with measured overhead, not a speedup. The requested 3× target remains
unmet. Timings and loaded-library hashes are saved with the audit evidence.

Two conservative-path cases were also timed with 7 trials of 3 graph-captured
launches. The old runtimes remain visible despite failing accuracy:

| Input | Before (µs) | Repaired (µs) | Runtime ratio |
|---|---:|---:|---:|
| Correlated keys, beta=0.6, 4097 tokens | 14643.33* | 40878.86 | 2.79× |
| Nearly correlated keys, beta=1, 1025 tokens | 3695.31* | 10269.09 | 2.78× |

`*` Only the pre-fix timings fail elementwise acceptance. Both repaired rows
pass before and after timed graph replay. The more accurate conservative path
has a substantial performance cost; this repair does not claim FP16 speed there.

[Before/after audit, five repeats, hashes and analytic witnesses](data/precision_audit.json) ·
[All 56 critical cases, before/after CSV](data/precision_audit.csv).

Reproduce:

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite all --output benchmark_data/gdn_mega/precision_audit.json
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega.py \
  --initial-state --lengths 256 4096 32768 \
  --output benchmark_data/gdn_mega/precision_performance.json
```
