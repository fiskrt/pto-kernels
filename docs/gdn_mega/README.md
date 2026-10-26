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

Kernel **`aac5a4d`** passed all **80 tests** in the last complete production run.
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
run serially. Two selected shapes additionally use the independent FP64 oracle.

The small entry scripts are `gdn_mega.py` (single-shape timing),
`gdn_mega_sweep.py` (latency grid), `gdn_mega_errors.py` (matched accuracy), and
the two `plot_gdn_mega*.py` renderers. Shared Markdown generation lives in
`gdn_mega_report_text.py`. `gdn_mega_ranges.py` measures the documented input ranges;
it contains no experimental solver implementations.
