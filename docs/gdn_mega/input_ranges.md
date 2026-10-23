# GDN input ranges and numerical design

Sources pinned to FLA commit `12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e`:
[test generators](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/tests/ops/test_gdn.py),
[comparison helper](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/fla/utils/testing.py),
[gate activation](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/fla/ops/gated_delta_rule/gate.py).

## What FLA tests

The main chunk tests normalize positive uniform Q/K; recurrent and FlashQLA
families also use Gaussian Q/K. V is uniform or Gaussian. Initial state is zero
or Gaussian FP32. Main beta is sigmoid(uniform), while logit tests use
sigmoid(Gaussian), optionally doubled. Gates use log-sigmoid inputs, divided by
normalizers 0.01, 0.1, 1, or 10; masking includes completely zero decay. Fused
activation also tests negative exp(A_log) times softplus(raw gate + bias).
FLA includes FP16, FP32 and BF16 paths, dimensions beyond our D128, optional
normalization/activation, output scaling, and backward tests. Our ABI consumes
already transformed FP16 Q/K/V/beta and FP32 log gates, with FP16 initial/final
state. The eight initial added cases adapt input distributions, not the full FLA API.

Its comparison helper normally accepts an RMS error ratio (plus an absolute
small-error shortcut); its CI environment can turn some failures into warnings.
We retain our own stricter elementwise policy and independent CPU FP64 oracle.
No FLA tolerance or warning behavior was imported.

## Bounds derived from those generators

These are mathematical generator ranges, not measured bounds for trained models.
Normal random variables have no finite theoretical magnitude bound.

| Quantity | Range before floating-point rounding |
|---|---|
| Normalized key/query norm | 1 |
| Key/query component | [-1,1], or [0,1] for positive inputs |
| KKᵀ entry | [-1,1]; [0,1] for positive keys |
| sigmoid(uniform[0,1)) beta | [0.5, 0.731059) |
| sigmoid(Gaussian) beta | (0,1) |
| doubled sigmoid beta | (0,2) |
| log-sigmoid(uniform)/normalizer c | approximately [-0.693147/c, -0.313262/c] |
| masked log gate | same interval or exactly 0 |
| log-sigmoid(Gaussian), activated gate | negative, no finite lower bound |
| Uniform V | [0,1) |
| Gaussian V / initial state | no finite mathematical bound |

FP16 rounding can make a normalized key's norm slightly above 1 and beta
round to an endpoint. Compute references from the actual supplied rounded
inputs. Positive random keys are strongly correlated: for independent uniform
coordinates, the large-D dot-product limit is E[X]^2/E[X²] = 3/4, whereas
independent zero-mean normalized Gaussian keys have dot-product scale 1/sqrt(D).
Thus positive-key coverage is essential, not an exotic stress distribution.

## Why small KKᵀ can still overflow an inverse expansion

Write L = I - A with A_ij = -beta_i exp(G_i-G_j) k_iᵀ k_j for i>j.
For identical unit keys, beta=1 and zero decay, A is minus the strict lower
triangle of ones. Then |(A^p)_ij| = binomial(i-j-1,p-1). At C128 these
intermediate powers vastly exceed 65504, although L⁻¹ is just a diagonal of
ones and a first subdiagonal of minus ones. The large terms cancel exactly
in exact arithmetic. Scaling the powers can prevent exponent overflow but
cannot recover mantissa bits lost to that cancellation. Beta below one still
permits huge binomial intermediates; a bound on individual entries is insufficient.

A stronger identity explains why the desired answer itself can remain bounded.
Define P_l = I-beta_l k_l k_lᵀ. For i>j,

    (L⁻¹)_ij = -beta_i exp(G_i-G_j)
                k_iᵀ P_(i-1) ... P_(j+1) k_j.

If log gates <=0 and 0<=beta_l ||k_l||²<=2, each P_l is a contraction, giving
|(L⁻¹)_ij| <= beta_i ||k_i|| ||k_j||. This avoids the giant inverse-power terms.
It motivates stable block solves and physical-state formulations. It does not
promise tiny rounding error near zero outputs, nor bound arbitrary V/state.
The physical recurrence has the conservative bound
||S_t||_F <= ||S_0||_F + sum_l beta_l ||k_l|| ||v_l|| under those conditions.
Positive log gates, unnormalized keys or beta outside that condition need a
separate bound; the existing extended stress tests remain enabled.

## Algorithm screening

`benchmarking/gdn_mega_algorithms.py` evaluates alternative solves against the independent
FP64 token oracle, with FP32 surrounding arithmetic. It compares direct FP32
triangular solve, 16-row panel substitution with paired FP16 products, an
FP16 approximate inverse with one/two residual corrections, an exact finite
block-preconditioned series, and precomputed affine chunk maps. Results are saved in
`algorithm_screen.json`. This is numerical feasibility evidence only, not an NPU
speed measurement or a complete hardware error simulation.

All candidates pass the sampled ordinary, no-decay and three positive-key FLA
cases. All fail the near-identical-key, amplitude-32 stress case, including direct
FP32 substitution. Therefore ordinary FP16 overflow is not the only issue:
Gram-product rounding and recurrent cancellation can violate absolute tolerance.
The compensated physical recurrence remains necessary until a better general
formulation also passes that case. No test is removed to make an approach pass.

## Measured first-block example (D128, 2026-10-23)

These are FP64 measurements on the exact FP16 test inputs, first key/value head
and first 128-token block only. They are examples, not full-distribution maxima.
Raw observed input ranges and individual powers are in `input_ranges.json`.

| Case | Mean off-diagonal KKᵀ | Largest of A¹,A²,…,A⁶⁴ | max abs(L⁻¹) |
|---|---:|---:|---:|
| Gaussian ordinary | -0.00203 | 0.103 | 1.0 |
| FLA positive, no decay | 0.75066 | 4.76e18 | 1.0 |
| FLA positive, weak decay | 0.74945 | 8.58e15 | 1.0 |
| FLA positive, masked decay | 0.75048 | 8.87e5 | 1.0 |
| Identical unit keys, beta=1 | 1.0 | 6.03e36 | 1.0 |

The first two FLA examples exceed even BF16's useful cancellation precision by
many orders of magnitude, despite a benign final inverse. Wider exponent range
alone is not a numerically stable algorithm.

## Reproduction and tail regression

```bash
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega_algorithms.py \
  --ranges-only --output docs/gdn_mega/input_ranges.json
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega_algorithms.py \
  --output benchmark_data/gdn_mega/algorithm_screen.json
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega.py \
  --audit-cases fla_positive_no_decay fla_positive_weak fla_gaussian_initial \
  --output benchmark_data/gdn_mega/fla_timings.json
```

Positive Q/K normalization now follows the FP16 normalization used by FLA's
main chunk tests. The initial distribution audit used FP32 normalization before
casting to FP16; its historical JSON records its source hash. Current range
results include the exact generator hash and are regenerated with FP16 normalization.
These are distribution adaptations to our ABI, not byte-identical FLA random seeds.

The native-solve full-suite run exposed invalid padded rows evaluating
exp(0-G_j), which can overflow with strong decay. Zero multiplication afterwards
does not repair this: 0*infinity becomes NaN. Invalid rows are now masked before
exponentiation, alongside future columns. Added 75-token FLA-style strong-gate
and 92-token strong-decay cases preserve this regression independently of the
existing 63/127-token failures. Tolerances and the CPU FP64 oracle are unchanged.
