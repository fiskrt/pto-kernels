# Structural GDN optimization results — 2026-10-26

Implementation **`aac5a4d`**, compared with the previous correct production
kernel (`bfc99e1`, unchanged at starting HEAD `5c5c43d`). All80 strict tests pass.
The production H16 sweep improves by **1.084–1.132×**; representative H8/H24/H64
4K checks improve by **1.150–1.171×**. **The requested2×/3× breakthrough has not
been achieved.** The larger alternatives were implemented and measured, not
assumed to help.

## What changed

1. **Batch independent inverse joins by dependency level.** Seven stable block
   joins still perform the same14 FP32 matrix products. Independent products
   now use disjoint operand/accumulator regions and queue together. Their
   exports/reloads have six synchronized phases instead of fourteen serial
   product round trips. This does not mean only six DMA instructions or fewer
   mathematical products; FP32 intermediates still use scratch.
2. **Remove transposes and broadcast matrices from the vector solve.** Gather
   each elimination coefficient directly from the original stripe using byte
   offsets prepared once. The vector FMA reuses its solved row through a zero
   repeat stride. This preserves FP32 forward substitution while removing four
   block transposes per vector stripe and15 materialized row broadcasts.
3. **Start next-chunk weighting before the current projection acknowledgement.**
   Retain pending coefficients at UB96–128KiB through RHS formation. Finish and
   publish their inverse only after the cube has consumed the previous inverse.
   This moves useful work into a wait without adding a copy or caller storage.
4. **Transfer whole attention/Gamma matrices.** Partition each existing pair of
   scratch allocations by heads, rather than by matrix columns. Full tiles are
   contiguous, reducing transfer instructions and scattered row accesses.

No inverse powers are truncated; no gate-dependent approximation was added.
Gate differences, the strong-coupling restart and authoritative FP32 state remain.
Shared KKᵀ/QKᵀ still run once per key head.

## Checked production timing

Ascend910B2, D128, H16/Hg4, nonzero initial state, ordinary benchmark distribution.
Median of9 trials×20 graph calls; allocation and reference work are excluded.
Every output/state element is checked before and after replay. Both columns
use the same strict acceptance policy. This is not a comparison against the
historical original that fails that policy.

| Tokens | Previous stable ms | New production ms | Speedup |
|---:|---:|---:|---:|
| 256 | 0.0941 | 0.0868 | 1.084× |
| 512 | 0.1419 | 0.1278 | 1.110× |
| 1024 | 0.2353 | 0.2120 | 1.110× |
| 4096 | 0.8030 | 0.7204 | 1.115× |
| 8192 | 1.5778 | 1.4185 | 1.112× |
| 16384 | 3.2082 | 2.8333 | 1.132× |
| 32768 | 6.5287 | 5.7660 | 1.132× |

The isolated screening build reached0.701ms at4K. After source cleanup and the
normal production build, the measured result is0.720ms; the table reports that
production binary, not the best experimental number. Library hashes and all
numerical statistics are saved in the [raw report](optimization_20261026.json)
and [compact CSV](optimization_20261026.csv).

Additional matched4K checks, with nonzero state and the same timing protocol:

| Value heads | Key heads | Previous stable ms | New production ms | Speedup |
|---:|---:|---:|---:|---:|
| 8 | 2 | 0.7872 | 0.6802 | 1.157× |
| 24 | 6 | 0.8264 | 0.7186 | 1.150× |
| 64 | 16 | 2.5114 | 2.1446 | 1.171× |

These are selected head counts, not a new full49-cell heatmap. The existing
[latency heatmap](heatmap.md) and [error heatmaps](error_heatmap.md) identify their
older `bfc99e1` snapshot explicitly.

## What the four profiler targets taught us

**Inverse transfers:** the compiler rejects direct FP32 accumulator→L1 copies
on this A2 target, including a call below PTO's static guard. The supported
copy narrows to FP16/BF16. A hybrid with paired FP16 inverse joins passed the
initial pointwise checks but took0.830ms; converting and reconstructing pairs
cost more than the transfers it removed. Packed FP32 scratch alone took0.800ms.
The retained batching improves dependency scheduling without narrowing data.

**Preparation/state feedback:** moving preparation alone was insufficient.
A version that copied pending coefficients took0.801ms. Direct coefficient
gathers, batched joins and a buffer layout that retains pending coefficients
were useful together. This is why isolated changes did not predict the combined
gain. A subsequent two-chunk lookahead experiment retained pending weights
through output/state finishing; it passed the128/4K checks but measured0.730ms
at4K, so it was not retained.

**Idle core groups:** all-core precomputation of full inverses took1.147ms.
A bounded producer/consumer prototype assigned idle vector groups to future
chunk preparation without extra allocations. It passed the4K check but took
0.824ms atH16; offloading only weights took0.748ms. AtH8 the full-helper variant
was0.709ms, versus0.680ms for retained production. These schedules add transfer,
coordination and producer-throughput costs; they are not retained. The ordinary
H16 recurrence still uses16 of24 cube groups.

**Projection precision:** native FP32 projections passed the initial checks but
cost0.874ms. Keeping paired-state projections was faster. No projection-precision
option or unused alternative implementation was added to production.

Other screened alternatives included head-major split scratch (0.810ms) and
applying the final64-row inverse join as a factored solve (0.812ms). Failed
prototype versions remain in the raw evidence; they are not reported as passes.
Only the retained production implementation has the full80-test certificate.
One early packed-scratch timing loaded the baseline before compilation finished;
that run is explicitly excluded, and the launcher now checks the library exists.
Measured library hashes, rather than candidate names, establish provenance.

## Fresh hardware profiler check

Same H16/Hg4/T4096/nonzero-state workload. Before uses the saved hardware profile
of the identical baseline library; after is a new `msprof op` run of production.
Except whole-kernel time, values are medians across active physical groups0–15.

| Measurement | Before µs | After µs |
|---|---:|---:|
| Whole kernel, profiler task duration | 794.43 | 700.68 |
| Cube arithmetic active | 175.58 | 169.46 |
| Cube input DMA active, MTE2 | 193.66 | 166.26 |
| Cube result export active, FIXPIPE | 148.04 | 114.58 |
| Vector0 arithmetic active | 316.97 | 282.95 |
| Vector0 wait for projection acknowledgement, flag5 | 114.85 | 3.56 |
| Vector0 wait for output products, flag9 | 12.07 | 87.86 |

Counters overlap and cannot be summed into runtime. Profiler task duration is
separate from graph timing; it must not replace the production benchmark above.
The changed waits show work moved into the old projection wait, with slack now
appearing later. Cube arithmetic time changes little, while transfers/vector work
shrink: this is a scheduling/data-movement gain, not a new FLOP-count shortcut.
Raw profiler CSV contents and checked profiler output are embedded in the report.

## Correctness, interface and source footprint

The normal GDN-only build completed, followed by **80 passed in92.18s**. Tests
include independent CPU FP64 token recurrence, poisoned/repeated scratch,
input-mutation guards, ragged/empty sequences, supplied initial state, weak/no
and positive gates, FLA-derived inputs, overflow and late strong-coupling cases.
Every raw-output and final-state element must satisfy
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, with finiteness and relative RMSE checks.
No test or tolerance was changed. All7 sequence-sweep cells and the6 additional
before/after head-count runs passed their pointwise checks.

There is still **one launch, two global phases and seven used production source
files**, totaling1302 physical lines (previous1221; original5241). No experimental
kernel, probe or tuning macro was added to production. The extra source supports
explicit overlap and batched dependencies.

Arguments, input/output dtypes, output scaling and caller allocation sizes are
unchanged. Internal scratch layout changes: `A/w` and `A_inv/u` each hold whole
FP32 streams for the first/second halves of heads. All supported H are even;
`(H/2)*T*C*4 == H*T*C*2`, exactly the existing buffer capacity. Scratch contents
are not a public intermediate-output API. See the [workspace contract](../../csrc/kernel/gdn_mega/README.md).

## Reproduction

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 240 python -m pytest \
  tests/test_gdn_mega.py tests/test_gdn_mega_validation.py -q
bash scripts/gdn_mega_docker.sh timeout 300 python benchmarking/gdn_mega.py \
  --heads 16 --key-heads 4 --initial-state --trials 9 --inner 20 \
  --lengths 256 512 1024 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/recheck.json
```

Run NPU jobs serially. For a baseline comparison, load the separately built
`5c5c43d` library through `LD_LIBRARY_PATH` and verify the hash reported by the
benchmark. Exact hashes, compiler flags, test log, generator source hashes and
experimental generator scripts are retained in the raw report.
