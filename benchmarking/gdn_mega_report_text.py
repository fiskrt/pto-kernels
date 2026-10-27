"""Markdown summaries derived from the same validated data as the GDN plots."""
import os
import json
from gdn_mega_sweep import head_configs
from pathlib import Path


def link(path, markdown):
    return Path(os.path.relpath(Path(path).resolve(), markdown.parent.resolve())).as_posix()


def number(value, spec='.3e'):
    return format(value, spec) if value is not None else 'NaN/Inf'


def numerical_failure(row, variant):
    """Mark this implementation/cell only when a check actually failed."""
    return row[variant]['status'] == 'failed' or row.get(f'{variant}_accuracy_status') == 'failed'


def write_latency(data, source, stem, markdown):
    rows = data['records']
    policy = rows[0]['current']['validation_policy']
    ratios = [r['speedup'] for r in rows if r['speedup'] is not None]
    long = [r['speedup'] for r in rows if r['length'] >= 4096 and r['speedup'] is not None]
    passed = sum(r['current']['status'] == 'passed' for r in rows)
    current = data['current_kernel_commit'][:7]
    baseline = data.get('baseline_source', {}).get('source_commit', 'original')[:7]
    historical = data.get('baseline_source', {}).get('kind') == 'historical_latency'
    configs = head_configs(data)
    lengths = sorted(data['lengths'])
    cells = {(r['heads'], r['key_heads'], r['length']): r for r in rows}
    def latency_table(variant):
        table = '| H/Hg | ' + ' | '.join(map(str, lengths)) + ' |\n'
        table += '|---|' + '---:|' * len(lengths) + '\n'
        table += '\n'.join(
            f'| {h}/{hg} | ' + ' | '.join(
                (number(cells[h, hg, n][f'{variant}_ms'], '.3f')
                 + ('*' if numerical_failure(cells[h, hg, n], variant) else ''))
                for n in lengths) + ' |' for h, hg in configs)
        return table
    current_table, baseline_table = latency_table('current'), latency_table('baseline')
    failures = sum(numerical_failure(r, 'baseline') for r in rows)
    warning = ('Original timings combine frozen historical measurements with fresh timings\n'
               'for previously missing cells. ' if historical else '')
    warning += (f'Original fails strict numerical checks on **{failures}/{len(rows)} cells**.\n'
                'Ratios involving a starred runtime are **not accuracy-equivalent speedups**.\n')
    text = f'''# GDN speedup heatmap

Generated from measurements dated {data['created_utc'][:10]}.
Original **{baseline}**, current **{current}**. [Overview](README.md).

![Original/current latency]({link(stem.with_suffix('.png'), markdown)})

Each cell shows three lines: **runtime ratio**, **ours: median total µs**, and
**base: median total µs**. The times cover the full kernel, not an individual
stage; allocation and validation are excluded.
`*` appears only beside a failing implementation's runtime, never on the ratio.
Passing runtimes stay unmarked. See the [matched error report](error_heatmap.md).

{warning}
Current passes **{passed}/{len(rows)} shapes**. There are {len(ratios)} measured runtime
ratios{f", ranging from **{min(ratios):.2f}× to {max(ratios):.2f}×**" if ratios else ""}.
'''
    if long:
        text += f'For 4K and longer, the range is **{min(long):.2f}× to {max(long):.2f}×**.\n'
    text += f'''There are {sum(x < 1 for x in ratios)} ratios below 1×. Gray cells mean
that a timing is unavailable, not merely that numerical validation failed.

## Original latency, milliseconds

Columns are token counts. `*` identifies a numerical failure.

{baseline_table}

## Current latency, milliseconds

Columns are token counts. Current timings remain available even where the original fails.

{current_table}

## Measurement and acceptance

H/Hg={', '.join(f'{h}/{hg}' for h, hg in configs)}; D=chunk size 128;
{'nonzero' if data['initial_state'] else 'zero'} initial state; seed {data['seed']}; gate scale{data['gate_scale']}.
Normalized Gaussian Q/K and ordinary benchmark inputs; this grid does not cover
all FLA distributions or characterize the slower strong-coupling fallback.

{data['warmups']} checked warmups, median of {data['trials']} trials ×
{data['calls_per_trial']} graph calls. Allocation and CPU reference work are outside timing.
The loaded library hash is checked for every worker.

The numerical pass criterion for every raw-output and final-state element is
`abs(error) <= {policy['atol']:g} + {policy['rtol']:g}*abs(reference)`, plus finiteness and relative RMSE < {policy['max_relative_rmse']:g},
before and after timing. The baseline is timed even when it fails, with its
failures retained. Current-kernel failures still fail the report command.
Cached statistics are reused only after byte equality
of every output/state element. The current grid uses the independent CPU FP64
token recurrence, computed from the actual rounded inputs. No inverse, decay
threshold or kernel intermediate participates in the oracle. Reference/check
wall times are saved separately from device timings.
See the [matched error heatmaps](error_heatmap.md) for absolute errors and violations.

## Reproduce

```bash
bash scripts/report_gdn_mega.sh latency
# Regenerate Markdown and plots from saved data without an NPU run:
bash scripts/report_gdn_mega.sh render
```

[Measurements and fingerprints]({link(source, markdown)}) ·
[CSV]({link(source.with_suffix('.csv'), markdown)}) ·
[SVG]({link(stem.with_suffix('.svg'), markdown)})
'''
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(text)
    print(f'Saved {markdown}')


def accuracy_regimes(source, markdown):
    """Keep the measured scope and stress failures in regenerated error reports."""
    directory = Path(source).parent
    audit_path = directory / 'numerical_regimes.json'
    if not audit_path.exists():
        return ''
    audit = json.loads(audit_path.read_text())
    evidence_path = directory / 'fallback_removal.json'
    evidence = json.loads(evidence_path.read_text())
    capture = json.loads((directory / 'qwen_capture.json').read_text())['input_summary']
    cases = audit['cases']
    if ([c['case']['name'] for c in cases] != audit['requested_cases'] or
            len({c['case']['name'] for c in cases}) != len(cases)):
        raise ValueError('Numerical regimes require a complete, unique audit')
    if audit['kernel_library_sha256'] != evidence['after_metadata']['kernel_library_sha256']:
        raise ValueError('Regime audit and captured-input evidence use different kernels')
    if audit['policy'] != evidence['after_metadata']['policy']:
        raise ValueError('Regime audit and captured-input evidence use different tolerances')
    failed = [c for c in cases if not c['passed']]
    approved_path = directory / 'accepted_precision.json'
    approved = json.loads(approved_path.read_text()) if approved_path.exists() else None
    policy_update = ''
    if approved:
        if (approved['kernel_library_sha256'] != audit['kernel_library_sha256'] or
                approved['previous_policy'] != audit['policy'] or
                approved['regular_suite']['audit_case_names'] != audit['requested_cases'] or
                approved['regular_suite']['failed'] != 0):
            raise ValueError('Approved-policy evidence does not match the kernel, old policy or complete regression suite')
        policy = approved['policy']
        recalibrated = approved['previous_failures']['cases']
        if ({c['case']['name'] for c in recalibrated} != {c['case']['name'] for c in failed}
                or not all(c['passed'] for c in recalibrated)):
            raise ValueError('Approved policy must cover every previously failing case')
        original_inputs = {c['case']['name']: c['input_sha256'] for c in failed}
        if any(c['input_sha256'] != original_inputs[c['case']['name']] or c['policy'] != policy
               for c in recalibrated):
            raise ValueError('Policy comparison must retain identical inputs and the approved tolerance')
        policy_update = f'''
### Current approved acceptance and regular regression coverage

The user accepted these small absolute errors. The current fixed policy is:

```text
abs(error[i]) <= {policy['atol']:g} + {policy['rtol']:g} * abs(reference[i])
```

**Atol increased from {audit['policy']['atol']:g} to {policy['atol']:g}; rtol remains
{policy['rtol']:g}. All {len(cases)} synthetic cases now pass**, including the
{len(failed)} former failures. Every case is part of ordinary pytest discovery;
there is no opt-in stress marker. The complete GDN suite passes
**{approved['regular_suite']['passed']} tests**.

At unchanged rtol, the largest absolute floor needed by the former failures is
{approved['maximum_required_atol']:.6g}, below the fixed {policy['atol']:g} budget.
Finiteness, raw-output and final-state bounds, independent CPU FP64 reference,
all five poisoned-scratch launches, input/guard integrity and relative RMSE
< {policy['max_relative_rmse']:g} remain enforced. No clipping, outlier removal,
case-specific tolerance or adaptive acceptance is used. Boundary tests inject
one bad element while RMSE still passes, checking that a regression beyond the
approved elementwise budget is rejected.

This is a **user-approved tolerance change, not a kernel accuracy improvement**.
The original tighter-policy measurements and their failures remain intact below
as diagnostics. Passing model captures already satisfied the older, tighter bound.
[Policy calibration and full-suite evidence]({link(approved_path, markdown)}).

### Historical characterization under the earlier tighter tolerance
'''
    primary = [c for c in cases if c['beta_in_production_range']
               and c['input_ranges']['log_gate'][1] <= 0 and c['case'].get('key_scale', 1.) == 1.]
    controlled = [c for c in cases if c['case']['name'].startswith('correlation_')]
    unit = [c for c in controlled if c['case'].get('amplitude', 1.) == 1.]
    large = [c for c in cases if c['case'].get('amplitude', 1.) >= 32. or c['case'].get('state_scale', .1) > 1.]
    large_failed = [c for c in large if not c['passed']]
    slow = [c for c in cases if max(c['case']['lengths']) >= 32768
            and -1e-6 <= c['case'].get('constant_log_gate', -1.) < 0
            and c['case'].get('amplitude', 1.) <= 1.]
    slow_failed = [c for c in slow if not c['passed']]
    def count(group):
        return f"{sum(c['passed'] for c in group)}/{len(group)} pass"
    def regime_count(group):
        return (f"{len(group)}/{len(group)} pass current policy; {count(group)} under the old bound"
                if approved else count(group))
    def worst(case):
        return max(((field, m[field]) for m in case['repeats'] for field in ('raw_output', 'final_state')),
                   key=lambda pair: pair[1]['max_tolerance_ratio'])
    if not failed or not large_failed or not slow_failed:
        raise ValueError('Regime evidence changed; review the measured-failure interpretation')
    named = {c['case']['name']: c for c in cases}
    selected = [named[name] for name in (
        'critical_tiny_gate_-1e-08_32768', 'critical_tiny_gate_-1e-07_32768',
        'control_coordinate_beta_zero', 'nearly_correlated_beta1', 'delayed_nearly_correlated')]
    selected += [max(large_failed, key=lambda c: worst(c)[1]['max_tolerance_ratio'])]
    severity = []
    current_column = ' | Approved allowance | Error / approved allowance' if approved else ''
    for c in selected:
        field, m = worst(c)
        out, state = [c['repeats'][0][f] for f in ('raw_output', 'final_state')]
        severity.append(
            f"| `{c['case']['name']}` | {out['pointwise_violations']:,} / {state['pointwise_violations']:,} | "
            f"{'output' if field == 'raw_output' else 'state'} | {m['abs_error_at_worst_tolerance']:.3e} | "
            f"{m['allowed_at_worst_tolerance']:.3e} | {m['max_tolerance_ratio']:.3f}× | "
            f"{m['abs_error_at_worst_tolerance']-m['allowed_at_worst_tolerance']:.3e}"
            + (f" | {policy['atol']+policy['rtol']*abs(m['reference_at_worst_tolerance']):.3e} | "
               f"{m['abs_error_at_worst_tolerance']/(policy['atol']+policy['rtol']*abs(m['reference_at_worst_tolerance'])):.3f}×" if approved else '') + ' |')
    finite = all(m[f]['finite'] for c in cases for m in c['repeats'] for f in ('raw_output', 'final_state'))
    floors = all(c[f]['passed'] for c in cases for f in ('fp16_output_rounding_floor', 'fp16_state_rounding_floor'))
    max_ratio = max(worst(c)[1]['max_tolerance_ratio'] for c in failed)
    worst_witness = worst(max(failed, key=lambda c: worst(c)[1]['max_tolerance_ratio']))[1]
    tiny = named['critical_tiny_gate_-1e-08_32768']['repeats'][0]['raw_output']
    largest = max(failed, key=lambda c: c['repeats'][0]['raw_output']['max_abs'])
    largest_error = largest['repeats'][0]['raw_output']['max_abs']
    largest_floor = largest['fp16_output_rounding_floor']['max_abs']
    if not finite or not floors:
        raise ValueError('New nonfinite or output-representation failures require a different severity summary')
    return f'''
## Accuracy regimes: where the current kernel passes and fails

**This section concerns the fallback-free kernel `{evidence['after_kernel_commit'][:7]}`**,
library `{audit['kernel_library_sha256'][:12]}`, audited {audit['created_utc'][:10]}.
{policy_update}
The figures above retain their own dated checkpoint; their green cells do not
certify the additional regimes below. At the recorded diagnostic tolerance
`atol={audit['policy']['atol']:g}, rtol={audit['policy']['rtol']:g}`, the synthetic audit has
**{count(cases)}, {len(failed)} failing cases**, with five launches per case.
The separate real-input replay has **{count([dict(passed=c['after']['passed']) for c in evidence['cases']])}**.
Case counts are coverage, not estimates of how often production inputs fail.
Within beta in [0,1], g <= 0 and normalized-key setups, **{regime_count(primary)}**.
The other {len(cases)-len(primary)} exploratory cases remain in the full report;
they are not used to establish safety for this primary workload.

### Parameters and measured regimes

`beta` is write strength. The kernel receives the FP32 **log gate `g`**;
**`alpha = exp(g)`** is per-token retention, so alpha and g are the same control
on different scales. More-negative g means more forgetting. Tiny negative g
means alpha is almost 1 and memory persists. Our primary workload has beta in
[0,1], g <= 0, and L2-normalized keys (with ordinary FP16 rounding).

These are **measured combinations, not universal safe intervals**. There is no
established rule such as “beta below X always passes” or “alpha above Y fails.”
Key alignment, V magnitude, incoming state and the duration of weak decay matter
together. Beta=0 is not automatically safe: the initial state still propagates.

| Regime | Beta, gate / retention | Keys, values/state and length | Evidence and interpretation |
|---|---|---|---|
| **Pass: observed model workload** | Captured beta median {capture['captured_beta_quantiles'][4]:.3f}; g median {capture['log_gate_quantiles'][4]:.6f}, alpha about 0.99385; natural per-head/token variation retained | Actual correlated Qwen keys, max abs(V) {capture['value_abs_max']:.2f}; supplied-state max about 12.60; 346–1024 tokens | All 108 full traces and 18 continuations pass. This is the strongest evidence for the measured workload, not all models or longer contexts. |
| **Pass: ordinary long sequences** | Uniform beta in [0,1), ordinary log-sigmoid Gaussian gates | Independent normalized Gaussian keys, unit-scale V, initial-state standard deviation 0.1; 256–32768 tokens, H16/Hg4 | All {len(evidence['ordinary_summary'])} current-kernel shapes pass. Does not establish long-context accuracy with correlated, nearly lossless heads. |
| **Pass: controlled correlation at unit scale** | beta=0.5 or 1; g=0, -0.001, -0.1 or -1; alpha=1, 0.9990005, 0.90484 or 0.36788 | Mean key cosine 0–1; V standard deviation 1, initial-state standard deviation 0.1 or 1; mostly 4K, two 32K no-decay cases | **{regime_count(unit)}.** Correlation, beta=1, or alpha=1 alone is not enough to cause failure. |
| **Stress: sustained near-lossless memory at 32K** | Tested g=-1e-8, -1e-7, -1e-6; alpha approximately 0.99999999, 0.9999999, 0.999999; beta=0, beta=1 and uniform[0,1) probes | Includes identical keys and supplied unit-scale state; old-bound failures have max abs(V) 3.90 and max abs(initial) 3.89 | **{regime_count(slow)}**. The previous tighter bound can fail without large amplitudes. Real long-context coverage is still useful. |
| **Stress: high-amplitude combinations** | Old-bound failures use beta=1 and g=0, -1e-6 or -0.001; alpha=1, about 0.999999 or 0.9990005 | V/state scale factors 32 or 128; keys range from uncorrelated to nearly identical; typical old-bound failing max abs(V) about 125–500 | **{regime_count(large)}**. Magnitude is not a proven failure threshold; keep these cases in regression coverage. |
| **Unestablished: other joint distributions** | Arbitrary persistent gate/beta patterns, other checkpoints; beta>1 or positive g are outside the primary workload above | Real correlated 8K–32K+ traces, other key norms and incoming-state distributions | The small set of exploratory cases and short model captures cannot establish blanket safety. No values are clipped or excluded to obtain a pass. |

### How large were the tighter-policy violations?

**All {len(cases)} cases remain finite; no NaN/Inf or overflow was observed.**
All ideal FP16-rounded goldens pass. The failures are additional internal
arithmetic error, not an inadequate final FP16 output range.

**Small absolute errors; they exceed a strict elementwise tolerance. No massive
numerical blow-up was observed.** At the worst normalized witness, reference
is {worst_witness['reference_at_worst_tolerance']:.9f} and the kernel returns
{worst_witness['actual_at_worst_tolerance']:.9f}: the absolute difference is
**{worst_witness['abs_error_at_worst_tolerance']:.3e}**. The recorded old allowance is
**{worst_witness['allowed_at_worst_tolerance']:.3e}**, so the error is
**{max_ratio:.3f}× the allowance**. That ratio measures a test-threshold overrun;
it does **not** mean the output is {max_ratio:.2f}× wrong or establish a large
effect on the model. Keep the strict failure visible, but do not equate its
multiplier with practical severity.

The table shows the element with the worst normalized error in each selected
case, using whichever tensor (output or state) is worse. “Excess” is the absolute
amount beyond its allowance. The violation counts include every element, not a
sample. These witness errors are **not** the global maximum absolute error.

| Case | Violating output / state elements under old bound | Worst tensor | Absolute error at witness | Old allowance | Error / old allowance | Excess over old allowance{current_column} |
|---|---:|---|---:|---:|---:|---:|{'---:|---:|' if approved else ''}
''' + '\n'.join(severity) + f'''

For the unit-scale 32K, g=-1e-8 example, {tiny['pointwise_violations']:,} of
{tiny['elements']:,} outputs fail ({100*tiny['pointwise_violations']/tiny['elements']:.5f}%),
with zero state violations. Its worst witness error is
{tiny['abs_error_at_worst_tolerance']:.3e}, or {tiny['max_tolerance_ratio']:.3f}× its allowance.
That is a small absolute error, not the large drift of the removed sequential
fallback, but it fails the recorded older contract. We have not measured
its effect on model logits or quality.

The largest raw-output max-abs statistic among failing cases is {largest_error:.6f};
the ideal FP16-rounding maximum on that same case is {largest_floor:.6f}.
Large max-abs numbers at large activation magnitudes can be normal FP16 rounding.
Use the witness, allowance and counts above to judge the extra numerical defect;
RMSE or the global maximum alone cannot establish acceptance.

### Which failures are important enough to fix?

The measured Qwen workload currently passes every checked element. **Do not call
weak gates themselves rare or extreme:** {100*capture['fraction_g_above_minus_0.001']:.2f}%
of its gates exceed -0.001 and {100*capture['fraction_g_above_minus_1e_6']:.2f}% exceed -1e-6.
The unverified part is sustained weak forgetting over a long sequence with the
relevant beta, key and state history. Global percentiles do not measure that joint
behavior. Captured initial states already reach about 12.60, so the failing
amplitude-32 tests' state maximum of about 12.5 is not, by itself, unrealistic.

1. **First priority: obtain long-context real captures with their incoming states.**
   Check the 32K near-lossless regime: it exceeded the old bound at unit scale, so it is
   potentially relevant even when V never reaches the large stress magnitudes.
   A failure beyond the current approved budget would justify prioritizing a parallel numerical fix.
2. **Lower immediate priority: the amplitude-32/128 stress-only combinations.**
   Their failing V maxima are roughly 125–500 versus 19.41 in these captures.
   Establish their joint frequency in the target model before paying a general
   runtime cost. This is a prioritization judgment, not a claim that they are safe
   or impossible; retain their failures and test larger models/contexts.
3. **Keep the decision tied to elementwise errors and model impact.** A few
   small violations and an overflow need different responses. Any repair should
   preserve the fast parallel execution. The approved policy update above is
   explicit; historical metrics are not relabeled. These small stress-test errors alone
   do not justify a costly fallback or a general performance penalty.

[All refreshed cases, five repeats and fingerprints]({link(audit_path, markdown)}) ·
[Captured-input and ordinary-control evidence]({link(evidence_path, markdown)}) ·
[Input ranges](input_ranges.md).

Run all regular regressions and optionally save a fresh audit under the current
policy. Preserve the historical diagnostic artifact instead of overwriting it:

```bash
bash scripts/gdn_mega_docker.sh python -m pytest \\
  tests/test_gdn_mega.py tests/test_gdn_mega_validation.py \\
  tests/test_gdn_mega_errors.py tests/test_gdn_mega_benchmark.py -q
bash scripts/gdn_mega_docker.sh python benchmarking/audit_gdn_mega.py \\
  --suite all --output benchmark_data/gdn_mega/current_accuracy.json
bash scripts/report_gdn_mega.sh render
```
'''


def write_errors(data, source, stem, markdown):
    rows = data['records']
    policy = rows[0]['current']['policy']
    commits = data.get('kernel_commits', {})
    versions = {v: (commits.get(v) or data['library_sha256'][v])[:7] for v in ('baseline', 'current')}
    passes = {v: sum(r[v]['status'] == 'passed' for r in rows) for v in versions}
    configs = head_configs(data)
    heads, key_heads = (16, 4) if (16, 4) in configs else configs[0]
    table = []
    for row in rows:
        if (row['heads'], row['key_heads']) != (heads, key_heads):
            continue
        a, b = [row[v]['summary']['raw_output'] for v in versions]
        table.append(f"| {row['length']} | {number(a['max_abs'])} | {number(b['max_abs'])} | "
                     f"{number(a['max_tolerance_ratio'], '.3f')} | {number(b['max_tolerance_ratio'], '.3f')} |")
    oracle = 'CPU FP64 token recurrence' if data['reference'] == 'recurrent-fp64' else 'CPU FP32 chunked reference'
    regimes = accuracy_regimes(source, markdown)
    scope_note = ("\n**Version scope:** “Current” in the figures means the plotted checkpoint,\n"
                  "not automatically the latest production build. See [accuracy regimes](#accuracy-regimes-where-the-current-kernel-passes-and-fails)\n"
                  "for the fallback-free kernel's broader passing and failing cases.\n" if regimes else '')
    text = f'''# GDN absolute-error heatmaps

Measurements dated {data['created_utc'][:10]}; original **{versions['baseline']}**,
current **{versions['current']}**. Both receive identical inputs and reference
results. [Overview](README.md) · [Latency heatmap](heatmap.md).
{scope_note}

![Maximum absolute raw-output error]({link(stem.with_name(stem.name+'_max_abs.png'), markdown)})

**Current passes {passes['current']}/{len(rows)} shapes; original passes
{passes['baseline']}/{len(rows)}**, checking both output and final state.
Finite failures remain visible on the shared scale; gray marks NaN/Inf.

## Elementwise acceptance

```text
abs(actual[i] - reference[i]) <= {policy['atol']:g} + {policy['rtol']:g} * abs(reference[i])
```

This is a mixed absolute/relative bound, **not** a uniform maximum absolute error
of {policy['atol']:g}. The ratio below divides each element's error by its allowance; a maximum
ratio ≤ 1 means every element satisfies that bound. Finiteness, relative RMSE < {policy['max_relative_rmse']:g}
and final-state checks are additional requirements. No outliers are discarded.

![Worst elementwise tolerance ratio]({link(stem.with_name(stem.name+'_tolerance.png'), markdown)})

## H{heads}/Hg{key_heads}, raw output units

| Tokens | Original max abs | Current max abs | Original worst ratio | Current worst ratio |
|---:|---:|---:|---:|---:|
''' + '\n'.join(table) + f'''

## Measurement contract

{oracle}; seed {data['seed']}; D128; H/Hg={', '.join(f'{h}/{hg}' for h, hg in configs)};
{'nonzero' if data['initial_state'] else 'zero'} initial state; normalized Gaussian Q/K, gate scale{data['gate_scale']}.
Maxima include **every output element and all {data['repeats']} fresh launches**.
Outputs and final state are filled with NaN before each launch. Caller scratch
initialization is preserved. Input/reference fingerprints match between libraries,
and inputs are checked for mutation. Every repeat and final-state metric is saved.
Statistics are reused only for byte-identical output and state tensors.
Both library workers share one fresh CPU golden per cell within this report run;
input/source identity and full golden-content hashes are verified before reuse.

The current algorithm retains FP32 Gram matrices, stable block solves, output
accumulation and authoritative state. Projections multiply original FP16 inputs
by two FP16 state components; the second captures the first component's rounding
residual. Final FP16 output rounding remains unavoidable. The recorded rounding
floor is a maximum statistic, not proof that every output is correctly rounded.
This ordinary-input grid is not an exhaustive input-range guarantee.
'''
    text += regimes
    text += f'''
## Reproduce

```bash
bash scripts/report_gdn_mega.sh errors
bash scripts/report_gdn_mega.sh render
```

The errors command requires the original library at
`benchmark_data/gdn_mega/baseline_lib/libno_workspace_kernel.so` and current at
`build/lib/libno_workspace_kernel.so`. Numerical failures are retained as data.

[Measurements, repeats and hashes]({link(source, markdown)}) ·
[CSV]({link(source.with_suffix('.csv'), markdown)}) ·
[Maximum-error SVG]({link(stem.with_name(stem.name+'_max_abs.svg'), markdown)}) ·
[Tolerance SVG]({link(stem.with_name(stem.name+'_tolerance.svg'), markdown)})
'''
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(text)
    print(f'Saved {markdown}')
