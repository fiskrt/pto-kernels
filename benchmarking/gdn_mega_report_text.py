"""Markdown summaries derived from the same validated data as the GDN plots."""
import os
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
                'Starred timings and ratios are **runtime comparisons, not accuracy-equivalent speedups**.\n')
    text = f'''# GDN speedup heatmap

Generated from measurements dated {data['created_utc'][:10]}.
Original **{baseline}**, current **{current}**. [Overview](README.md).

![Original/current latency]({link(stem.with_suffix('.png'), markdown)})

Each cell shows the runtime ratio above our kernel's **median total time
in µs per call**, including cells where the original fails. This covers the full
kernel, not an individual stage; allocation and validation are excluded.
`*` marks a numerical failure; its runtime is retained and its failed accuracy
status is unchanged. See the [matched error report](error_heatmap.md).

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
`abs(error) <= 1e-5 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005,
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


def write_errors(data, source, stem, markdown):
    rows = data['records']
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
