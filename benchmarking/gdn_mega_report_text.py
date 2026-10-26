"""Markdown summaries derived from the same validated data as the GDN plots."""
import os
import json
from pathlib import Path


def link(path, markdown):
    return Path(os.path.relpath(Path(path).resolve(), markdown.parent.resolve())).as_posix()


def number(value, spec='.3e'):
    return format(value, spec) if value is not None else 'NaN/Inf'


def write_latency(data, source, stem, markdown):
    rows = data['records']
    ratios = [r['speedup'] for r in rows if r['speedup'] is not None]
    long = [r['speedup'] for r in rows if r['length'] >= 4096 and r['speedup'] is not None]
    passed = sum(r['current']['status'] == 'passed' for r in rows)
    current = data['current_kernel_commit'][:7]
    baseline = data.get('baseline_source', {}).get('source_commit', 'original')[:7]
    historical = data.get('baseline_source', {}).get('kind') == 'historical_latency'
    heads = 16 if 16 in data['heads'] else data['heads'][0]
    table = '\n'.join(f"| {r['length']} | {number(r['baseline_ms'], '.3f')} | "
                      f"{number(r['current_ms'], '.3f')} | {number(r['speedup'], '.2f')} |"
                      for r in rows if r['heads'] == heads)
    warning = ('Original timings are historical RMSE-only measurements. The original fails the\n'
               'current elementwise policy; this is **not an accuracy-equivalent comparison**.\n'
               if historical else 'Both libraries were measured under the current acceptance policy.\n')
    text = f'''# GDN speedup heatmap

Generated from measurements dated {data['created_utc'][:10]}.
Original **{baseline}**, current **{current}**. [Overview](README.md).

![Original/current latency]({link(stem.with_suffix('.png'), markdown)})

{warning}
Current passes **{passed}/{len(rows)} shapes**. There are {len(ratios)} valid latency
ratios, ranging from **{min(ratios):.2f}× to {max(ratios):.2f}×**.
'''
    if long:
        text += f'For 4K and longer, the range is **{min(long):.2f}× to {max(long):.2f}×**.\n'
    text += f'''There are {sum(x < 1 for x in ratios)} ratios below 1×. Gray cells identify
an explicitly failed implementation; no speedup is inferred for them.

## H{heads}/Hg{heads // data['group_size']}

| Tokens | Original ms | Current ms | Speedup × |
|---:|---:|---:|---:|
{table}

## Measurement and acceptance

H={','.join(map(str, data['heads']))}; Hg=H/{data['group_size']}; D=chunk size 128;
{'nonzero' if data['initial_state'] else 'zero'} initial state; seed {data['seed']}; gate scale{data['gate_scale']}.
Normalized Gaussian Q/K and ordinary benchmark inputs; this grid does not cover
all FLA distributions or characterize the slower strong-coupling fallback.

{data['warmups']} checked warmups, median of {data['trials']} trials ×
{data['calls_per_trial']} graph calls. Allocation and CPU reference work are outside timing.
The loaded library hash is checked for every worker.

Every raw-output and final-state element must satisfy
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005,
before and after timing. Cached statistics are reused only after byte equality
of every output/state element. The grid uses a CPU FP32 chunked reference;
the production tests separately include an independent FP64 token oracle.
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


def write_errors(data, source, stem, markdown, fp64=None):
    rows = data['records']
    commits = data.get('kernel_commits', {})
    versions = {v: (commits.get(v) or data['library_sha256'][v])[:7] for v in ('baseline', 'current')}
    passes = {v: sum(r[v]['status'] == 'passed' for r in rows) for v in versions}
    heads = 16 if 16 in data['heads'] else data['heads'][0]
    table = []
    for row in rows:
        if row['heads'] != heads:
            continue
        a, b = [row[v]['summary']['raw_output'] for v in versions]
        table.append(f"| {row['length']} | {number(a['max_abs'])} | {number(b['max_abs'])} | "
                     f"{number(a['max_tolerance_ratio'], '.3f')} | {number(b['max_tolerance_ratio'], '.3f')} |")
    oracle = 'CPU FP64 token recurrence' if data['reference'] == 'recurrent-fp64' else 'CPU FP32 chunked reference'
    text = f'''# GDN absolute-error heatmaps

Measurements dated {data['created_utc'][:10]}; original **{versions['baseline']}**,
current **{versions['current']}**. Both receive identical inputs and reference
results. [Overview](README.md) · [Latency heatmap](heatmap.md).

![Maximum absolute raw-output error]({link(stem.with_name(stem.name+'_max_abs.png'), markdown)})

**Current passes {passes['current']}/{len(rows)} shapes; original passes
{passes['baseline']}/{len(rows)}**, checking both output and final state.
Finite failures remain visible on the shared scale; gray marks NaN/Inf.

## Elementwise acceptance

```text
abs(actual[i] - reference[i]) <= 1e-5 + 1e-3 * abs(reference[i])
```

This is a mixed absolute/relative bound, **not** a uniform maximum absolute error
of 1e-5. The ratio below divides each element's error by its allowance; a maximum
ratio ≤ 1 means every element satisfies that bound. Finiteness, relative RMSE < 0.005
and final-state checks are additional requirements. No outliers are discarded.

![Worst elementwise tolerance ratio]({link(stem.with_name(stem.name+'_tolerance.png'), markdown)})

## H{heads}/Hg{heads // data['group_size']}, raw output units

| Tokens | Original max abs | Current max abs | Original worst ratio | Current worst ratio |
|---:|---:|---:|---:|---:|
''' + '\n'.join(table) + f'''

## Measurement contract

{oracle}; seed {data['seed']}; D128; GQA group {data['group_size']};
{'nonzero' if data['initial_state'] else 'zero'} initial state; normalized Gaussian Q/K, gate scale{data['gate_scale']}.
Maxima include **every output element and all {data['repeats']} fresh launches**.
Outputs and final state are filled with NaN before each launch. Caller scratch
initialization is preserved. Input/reference fingerprints match between libraries,
and inputs are checked for mutation. Every repeat and final-state metric is saved.
Statistics are reused only for byte-identical output and state tensors.

The current algorithm retains FP32 Gram matrices, stable block solves, output
accumulation and authoritative state. Projections multiply original FP16 inputs
by two FP16 state components; the second captures the first component's rounding
residual. Final FP16 output rounding remains unavoidable. The recorded rounding
floor is a maximum statistic, not proof that every output is correctly rounded.
This ordinary-input grid is not an exhaustive input-range guarantee.
'''
    if fp64:
        checks = json.loads(fp64.read_text())
        if checks['library_sha256'] != data['library_sha256'] or checks['reference'] != 'recurrent-fp64':
            raise ValueError('FP64 crosschecks must use the same libraries and the independent oracle')
        counts = {v: sum(r[v]['status'] == 'passed' for r in checks['records']) for v in versions}
        text += f'''
Independent FP64 token-oracle crosschecks: current **{counts['current']}/{len(checks['records'])}**,
original **{counts['baseline']}/{len(checks['records'])}**. These selected shapes
are separate from the full-grid oracle above. [FP64 measurements]({link(fp64, markdown)}).
'''
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
