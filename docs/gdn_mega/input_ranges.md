# GDN input ranges and numerical design

Sources pinned to FLA commit `12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e`:
[test generators](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/tests/ops/test_gdn.py),
[comparison helper](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/fla/utils/testing.py),
[gate activation](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/fla/ops/gated_delta_rule/gate.py).

## Measured Qwen forward-pass inputs

The user's `Qwen/Qwen3.5-0.8B` FP32-model capture contains six prompts,
346–1024 tokens each, across 18 linear-attention layers and 16 value heads:
1,404,288 token-head samples. These are observed values, not mathematical
bounds or a guarantee for other checkpoints/context lengths.

| Captured quantity | Minimum | p1 | Median | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Log gate g (FP32) | -9.31484 | -3.13377 | -0.006174 | -4.827e-7 | -5.989e-11 |
| Beta before ABI conversion (FP32) | 1.182e-12 | 0.002551 | 0.352994 | 0.980206 | 0.999992 |
| Beta consumed by kernel/golden (FP16) | 0 | 0.002550 | 0.353027 | 0.979980 | 1 |

36.97% of gates lie above -0.001, and 1.60% above -1e-6. Very weak decay is
present in real data; it should not automatically be dismissed as an artificial
edge case. Per-head and temporal structure matters: a constant tiny gate over
32K tokens is a different experiment from these captured sequences. Beta can
round to 0 or 1 in our FP16 interface; 0.00363% of captured beta values round to 1.
Captured Q/K are already normalized and rounded to FP16, and captured V is FP16
with maximum absolute value 19.40625. The replay does not renormalize or rescale.

The [capture replay](README.md#qwen-forward-pass-capture-replay) checks all actual
Q/K/V/g/beta tensors, retaining correlations between heads, tokens and fields.
It passes all 108 full-trace and 18 supplied-state continuation cases on both
production and an isolated build without sequential fallback. All comparisons
use the unchanged elementwise policy against the independent FP64 recurrence.
The pre-removal production kernel nevertheless took fallback for 69.0% of
full-trace head instances. Production now always uses the parallel chunk path.
These measurements are separate from the synthetic speed/error heatmaps and do
not establish long-context accuracy. [Data and provenance](data/qwen_capture.json).

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

## A tighter bound on the physical state

The contraction bound above can be sharpened using the update's energy. This is
an algebraic bound derived here, not a measured model-trace limit. Let
`X = exp(g_i)*S_(i-1)`, `a = k_iᵀ X`, `r² = ||k_i||²`, and `rho = beta_i*r²`.
For beta>0 and rho<2, completing the square gives

    ||S_i||_F² = ||X||_F²
      - beta*(2-rho)*||a - (1-rho)/(2-rho)*v_i||²
      + beta/(2-rho)*||v_i||².

With nonpositive log gates,

    ||S_t||_F² <= ||S_0||_F² + sum_i beta_i/(2-beta_i*||k_i||²) * ||v_i||².

For the main positive-uniform FLA generator before rounding, unit keys,
beta <= sigmoid(1), V components in[0,1), D128 and zero initial state, this
bounds the physical state norm by **1554.49 at32768 tokens**. Unit queries give
the same bound on each output vector's norm, so individual state/output entries
are also below that number. The energy factor is beta/(2-beta) <=0.576117.
Substituting a supplied squared-key-norm bound of1.002 gives about1555.38 instead;
actual rounded norms and beta must be used when applying this to stored inputs.
The derivation bounds exact recurrence arithmetic, not rounding error in a kernel.

Thus the bounded main generator does not need enormous physical states to
produce an overflowing inverse power series. The desired values can fit FP16
while an unstable construction does not. This supports changing the algorithm,
but does not justify FP16-only arithmetic: cancellation and near-zero absolute
errors still require enough precision.

This bound is conditional. Gaussian V/initial states have no finite generator
bound. As doubled beta approaches2/||k||², the denominator tends to zero;
at equality use the earlier triangle bound, and above it the contraction
assumption fails. FP16 rounding can cross that endpoint. These families cannot
be covered by the1554.49 figure, nor is it a blanket production input limit.

## Measured first-block example (D128, 2026-10-23)

These are FP64 measurements on the exact FP16 test inputs, first key/value head
and first 128-token block only. They are examples, not full-distribution maxima.
Raw observed input ranges and individual powers are in [input_ranges.json](data/input_ranges.json).

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
bash scripts/gdn_mega_docker.sh python benchmarking/gdn_mega_ranges.py
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
