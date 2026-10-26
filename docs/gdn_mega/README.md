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

Kernel **`aac5a4d`** and its validation/report tooling pass **115 tests**.
The full FP64 grid passes **63/63 current shapes**, with zero elementwise violations;
the original fails all63 shapes under this stricter policy.
[Validation log, source hashes and library fingerprint](data/validation.json).
Kernel sources and numerical tolerances have not changed during this report refresh.
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
status. Starred ratios and baseline timings denote numerical problems; the
current median total µs remains below the ratio in every heatmap cell. Numerical
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
