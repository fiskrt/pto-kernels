# Current stable GDN result

Production kernel: **`aac5a4d`**, 2026-10-26. The normal production build passes
**all80 strict tests**, including independent FP64 and poisoned-workspace cases.
No test or tolerance was changed. The requested3× speed target remains unmet.

Latest results and all four optimization targets:
[structural optimization report](optimization_20261026.md),
[raw measurements and provenance](optimization_20261026.json),
[latency CSV](optimization_20261026.csv).

Against the previous correct production kernel, H16/Hg4 with nonzero state:

| Tokens | Previous ms | Current ms | Speedup |
|---:|---:|---:|---:|
| 4096 | 0.8030 | 0.7204 | 1.115× |
| 8192 | 1.5778 | 1.4185 | 1.112× |
| 16384 | 3.2082 | 2.8333 | 1.132× |
| 32768 | 6.5287 | 5.7660 | 1.132× |

The report also covers256/512/1024 tokens and matched4K checks atH8/H24/H64.
The production changes batch independent inverse joins, remove solve transposes
and broadcasts, start next-chunk weighting earlier, and transfer whole
attention/Gamma tiles. Larger precomputation/helper-core/native-projection
alternatives were measured and rejected when slower.

There is still one launch, two global phases and seven used source files.
Arguments, dtypes, output scaling and caller allocation sizes are unchanged.
Internal scratch layout changes are documented in the
[kernel README](../../csrc/kernel/gdn_mega/README.md).

The [latency heatmap](heatmap.md), [absolute-error heatmaps](error_heatmap.md)
and [original profiler diagnosis](profile_comparison.md) remain explicitly
versioned snapshots of `bfc99e1`. They are not measurements of `aac5a4d`.
The historical original fails strict numerical acceptance, so its timing is
not an accuracy-equivalent baseline. See [FLA input ranges](input_ranges.md)
for the overflow/cancellation analysis and strong-coupling limitations.
