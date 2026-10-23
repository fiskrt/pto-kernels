# Production GDN mega kernel

One launch and two global phases, with a fixed compiled schedule on Ascend 910B2:

1. Vector computes compensated gate prefixes and transposes beta while cube
   computes FP32 KKᵀ and QKᵀ, shared by grouped value heads. A global barrier
   joins these independent producers.
2. Each active cube/vector group retains one head across its chunks. Vector
   prepares the next chunk's inverse and attention while cube processes the
   current chunk. The recurrent state stays in vector-local FP32 storage.

The inverse uses stable 16×16 FP32 triangular substitution followed by
block-triangular joins. It never forms the large powers of A that caused the
previous FP16 overflow. Inverse joins, RHS solves and attention products use
native FP32 operands. A stacked K/Q product shares two FP16 components of
state and accumulates in FP32. Preparation computes Gamma = exp(G_last-G_i)*K
in FP32. A second stacked product shares R between attention*R and Gammaᵀ*R,
removing the dependent residual-scaling exchange.
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

Strong coupling can amplify small chunk-solve errors. When any strict-lower
coefficient `abs(beta_i * exp(G_i-G_j) * dot(k_i,k_j))` exceeds 0.5, the head
is recomputed from its supplied initial state using compensated FP32 token
recurrence and balanced dot reductions. This also corrects a previously computed
prefix when coupling appears later. Selection depends on input arithmetic, not
seeds, test identifiers, or reference outputs. This conservative path is slower;
latency consequently depends on the values as well as the shape.

A persistent kernel means that the same hardware group loops over successive
chunks of a head, retaining state locally. It does not mean a background process
or an additional launch. File count does not equal stage count.

| File | Used for |
|---|---|
| `kernel_gdn_mega.cpp` | Two-phase orchestration, entry and head dispatch |
| `common.hpp` | Synchronization, packed I/O and sequence indexing |
| `gate_scan.hpp` | Compensated prefix scan and gate/beta layout conversion |
| `shared_grams.hpp` | Shared FP32 KKᵀ/QKᵀ with pipelined input loads |
| `inverse.hpp` | Batched 16×16 triangular substitution |
| `recurrence_pipeline.hpp` | Original-input × paired-state products and native FP32 block joins |
| `fused_pipeline.hpp` | Preparation overlap, recurrent state, output and compensated path |

All seven files are used by the entry point. There are no probe kernels, runtime
accuracy knobs, or selectable tuning configurations. Compiler architecture and
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
| `A`, `w` | Two column halves of native FP32 decay-weighted keys Gamma |
| `A_inv`, `u` | Two column halves of native FP32 attention |
| `A_inv_f32` | FP32 QKᵀ shared by grouped heads |
| `v_new`, `s` | Two column halves of FP32 KKᵀ |
| `wy_workspace_a1/a2` | Reserved compatibility storage |
| `kkt_workspace` | FP32 output-product exchange |
| `h_workspace` | FP32 projection/residual/state exchange, saved FP32 RHS and paired state |
| `o_workspace_qk` | Native FP32 inverse join scratch (first 16 KiB/core), coupling markers (second half) |
| `o_workspace_qs/gated` | Two row halves of prepared native FP32 inverse |

Scratch contents are not a post-call intermediate-results API. Continue allocating
existing shapes, as demonstrated by `_prepare_gdn_mega` in
[`test_gdn_mega.py`](../../../tests/test_gdn_mega.py). The
[`host declaration`](../../host/torch_gdn_mega.h) describes the public arguments.

## Build and verify

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 240 python -m pytest \
  tests/test_gdn_mega.py tests/test_gdn_mega_validation.py -q
bash scripts/gdn_mega_docker.sh timeout 600 python benchmarking/audit_gdn_mega.py \
  --output benchmark_data/gdn_mega/current_accuracy.json
bash scripts/gdn_mega_docker.sh timeout 600 python benchmarking/gdn_mega.py \
  --initial-state --lengths 128 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/current_performance.json
```

The independent oracle is CPU FP64 token recurrence. Acceptance checks every
raw-output and final-state element against `1e-5 + 1e-3*abs(reference)`, as well
as relative RMSE < 0.005, finiteness, exact shapes, guarded outputs, unchanged
inputs, and repeated launches with poisoned scratch. No tolerance was relaxed.
Historical RMSE-only results in the
[branch report](../../../docs/gdn_mega/optimization_report.md) are not certificates
for this stricter policy. Current milestones and remaining limitations are
recorded in [`success.md`](../../../success.md) and
[`memory.md`](../../../memory.md).
