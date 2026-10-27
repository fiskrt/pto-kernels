# Production GDN mega kernel

One launch with shared setup followed by recurrence on Ascend 910B2:

1. Vector computes compensated gate prefixes and transposes beta while cube
   computes FP32 KKᵀ and QKᵀ, shared by grouped value heads. A global barrier
   joins these independent producers.
2. Normally each active cube/vector group retains one head across its chunks. Vector
   weights the next chunk before waiting for current projections, retains its
   coefficients through RHS preparation, then finishes its inverse while cube
   processes the current chunk. The recurrent state stays in vector-local FP32 storage.

For one sequence with H=8, at least 1024 tokens, and at least two chunk/key-head
tasks per spare group, the second phase uses a different work partition:
16 groups retain two independent 64-value-column slices of each head; the
remaining eight groups prepare future chunks. The helpers share cached KK/QK
across grouped heads and compute weights, leaf inverses, cube joins and Gamma.
Consumers form the RHS while waiting for/loading those prepared inverses.
State never moves between workers. An extra global barrier initializes readiness
flags before producers and consumers start concurrently. Shorter shapes and
other head/batch counts use the local preparation schedule above.

The inverse uses stable 16×16 FP32 triangular substitution followed by
block-triangular joins. The vector solve gathers coefficients directly without
transposing or expanding solved rows. Independent cube joins are queued by
level. The final join stays factored: with D0/D1 the diagonal inverses and
F=-D1*L10, the solve computes R0=D0*B0, then R1=D1*B1+F*R0. It avoids
materializing F*D0. It never forms the large powers of A that caused the
previous FP16 overflow. Inverse joins, RHS solves and attention products use
native FP32 operands. A stacked K/Q product shares two FP16 components of
state and accumulates in FP32. Preparation computes Gamma = exp(G_last-G_i)*K
in FP32. A second stacked product shares R between attention*R and Gammaᵀ*R,
removing the dependent residual-scaling exchange. QK and KK weighting share
one prepared decay matrix.
The pair representation is approximate; the authoritative recurrent state is FP32.
Native FP32 joins export/reload intermediates because A2 accumulator-to-L1
copies narrow to FP16/BF16. No identity-matrix products are needed in this path.

The residual is formed before applying the inverse:

`R = inverse * [beta * (V - exp(G) * K*S)]`.

This avoids separately multiplying the inverse by both WY operands. Original
FP16 K and Q remain exact inputs to their products with paired state.
Output accumulation and state updates remain FP32, with final FP16 stores.
Invalid rows and future columns are masked before exponentiation, preventing
`0*inf` from padded prefixes. Gate differences retain `G_i-G_j`; there is no inverse-power truncation based
on decay and no division by a potentially underflowed gate exponential.

Every head uses this parallel chunk algorithm. There is no coupling threshold,
per-chunk detection reduction, or sequential token fallback. Real Qwen capture
replays pass without that fallback. The fixed numerical regression cases
pass the user-approved atol=1e-4, rtol=1e-3 policy, including the small errors
previously rejected by atol=1e-5. Historical tighter-policy results remain
recorded. See the
[current validation report](../../../docs/gdn_mega/README.md) for measured scope.

A persistent kernel means that the same hardware group loops over successive
chunks of a head, retaining state locally. It does not mean a background process
or an additional launch. File count does not equal stage count.

| File | Used for |
|---|---|
| `kernel_gdn_mega.cpp` | Setup, shape-based worker partition, entry and head dispatch |
| `common.hpp` | Synchronization, packed I/O and sequence indexing |
| `gate_scan.hpp` | Compensated prefix scan and gate/beta layout conversion |
| `shared_grams.hpp` | Shared FP32 KKᵀ/QKᵀ with pipelined input loads |
| `inverse.hpp` | Batched 16×16 triangular substitution |
| `recurrence_pipeline.hpp` | Original-input × paired-state products and native FP32 block joins |
| `fused_pipeline.hpp` | Preparation overlap, recurrent state and output |

All seven files are used by the entry point. Shape-based scheduling is internal;
there are no probe kernels, runtime accuracy knobs, or selectable tuning configurations. Compiler architecture and
cube/vector guards remain necessary. Hardware validation is on Ascend 910B2;
the inherited A5 synchronization path is not a verified implementation.

## Caller contract

`pto_gdn_mega` keeps the same arguments, tensor shapes, dtypes, allocation sizes
and in-place output behavior. No additional caller allocation is required.
T and packed sequence count N are dynamic; H supports 8/12/16/24/32/48/64, Hg
must divide H, and D=C=128. The launch uses the runtime physical cube-core count
(24 on the tested device). Nonzero initial states, absent initial state, empty
packed sequences and short tails are supported.

Only `out` and `final_state` are primary outputs. The existing caller applies
`128**-0.5` to raw output; the device does not introduce that scaling.

| Supplied scratch | Current role |
|---|---|
| `g_t`, `g_sum`, `beta_t` | Head-major high/low gate prefixes and beta |
| `A`, `w` | Whole FP32 Gamma tiles: first/second halves of heads, head-major |
| `A_inv`, `u` | Whole FP32 attention tiles: first/second halves of heads, head-major |
| `A_inv_f32` | FP32 QKᵀ shared by grouped heads; token helpers reuse it for per-head factored inverses and readiness flags after caching QK |
| `v_new`, `s` | Two column halves of FP32 KKᵀ |
| `wy_workspace_a1/a2` | Reserved compatibility storage |
| `kkt_workspace` | FP32 output-product exchange |
| `h_workspace` | FP32 projection/residual/state exchange, saved FP32 RHS and paired state |
| `o_workspace_qk` | Native FP32 inverse join scratch (first 16 KiB/core); remaining capacity reserved for ABI compatibility |
| `o_workspace_qs/gated` | Two row halves of prepared native FP32 inverse |

For attention and Gamma, each existing half allocation holds `H/2` complete
FP32 head streams. All supported head counts are even, so this exactly fills
the existing allocations. No columns are split across buffers.

Scratch contents are not a post-call intermediate-results API. Continue allocating
existing shapes, as demonstrated by `_prepare_gdn_mega` in
[`test_gdn_mega.py`](../../../tests/test_gdn_mega.py). The
[`host declaration`](../../host/torch_gdn_mega.h) describes the public arguments.

## Build and verify

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh python -m pytest \
  tests/test_gdn_mega.py tests/test_gdn_mega_validation.py \
  tests/test_gdn_mega_errors.py tests/test_gdn_mega_benchmark.py -q
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \
  --suite all --output benchmark_data/gdn_mega/current_accuracy.json
bash scripts/gdn_mega_docker.sh timeout 600 python benchmarking/gdn_mega.py \
  --initial-state --lengths 128 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/current_performance.json
```

The independent oracle is CPU FP64 token recurrence. Acceptance checks every
raw-output and final-state element against `1e-4 + 1e-3*abs(reference)`, as well
as relative RMSE < 0.005, finiteness, exact shapes, guarded outputs, unchanged
inputs, and repeated launches with poisoned scratch. The user approved raising
atol from1e-5 to1e-4; rtol and the other checks are unchanged. All160 cases run
by default, with explicit boundary tests rejecting errors beyond the new budget.
Historical RMSE-only timing results in the
[report index](../../../docs/gdn_mega/README.md) are not certificates
for this stricter policy. Current milestones and remaining limitations are
recorded in [`success.md`](../../../success.md) and
[`memory.md`](../../../memory.md).
