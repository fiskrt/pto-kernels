"""Plot matched original/current output errors; failed finite cells stay visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np


VARIANTS = ('baseline', 'current')


def validate(data):
    heads, lengths = sorted(data['heads']), sorted(data['lengths'])
    cells = {(r['heads'], r['length']): r for r in data['records']}
    if len(cells) != len(data['records']) or set(cells) != {(h, n) for h in heads for n in lengths}:
        raise ValueError('Error heatmap requires exactly one matched comparison per requested cell')
    for cell in cells.values():
        before, after = cell['baseline'], cell['current']
        for name in ('input_sha256', 'reference_sha256', 'policy', 'environment'):
            if before[name] != after[name]:
                raise ValueError(f'Mismatched comparison: {name}')
        for variant in VARIANTS:
            row = cell[variant]
            if row['kernel_library_sha256'] != data['library_sha256'][variant]:
                raise ValueError('Mismatched library fingerprint')
            if len(row['repeats']) != data['repeats']:
                raise ValueError('Missing repeats')
            metric = row['summary']['raw_output']
            finite = all(r['raw_output']['finite'] for r in row['repeats'])
            if metric['finite'] != finite:
                raise ValueError('Nonfinite repeat hidden by summary')
            for name in ('max_abs', 'max_tolerance_ratio'):
                value = metric[name]
                if finite:
                    expected = max(r['raw_output'][name] for r in row['repeats'])
                    if value is None or not np.isfinite(value) or value < 0 or value != expected:
                        raise ValueError('Summary must retain the maximum across all repeats')
                elif value is not None:
                    raise ValueError('Nonfinite errors must be explicit, not replaced by a finite maximum')
    return heads, lengths, cells


def plot(data, heads, lengths, cells, metric, output):
    values = np.array([[[cells[h, n][variant]['summary']['raw_output'][metric]
                         if cells[h, n][variant]['summary']['raw_output']['finite'] else np.nan
                         for n in lengths] for h in heads] for variant in VARIANTS], dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    smallest = float(positive.min()) if positive.size else 1e-8
    largest = float(positive.max()) if positive.size else 1.
    if metric == 'max_abs':
        vmin, vmax = smallest, max(largest, smallest * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        shown = np.where(values == 0, vmin, values)
        cmap = plt.get_cmap('magma').copy()
        title = 'Maximum absolute output error: original vs current'
        label = 'max |out − reference|, raw output units (shared log scale)'
    else:
        lower = min(-1., np.floor(np.log10(smallest)))
        upper = max(1., np.ceil(np.log10(largest)))
        norm = TwoSlopeNorm(vmin=lower, vcenter=0., vmax=upper)
        shown = np.log10(np.where(values == 0, 10**lower, values))
        cmap = plt.get_cmap('RdYlGn_r').copy()
        title = 'Worst elementwise tolerance ratio: original vs current'
        label = 'max [|error| / (1e−5 + 1e−3·|reference|)] — limit = 1'
    cmap.set_bad('#d9dde3')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), sharey=True, layout='constrained')
    fig.suptitle(title, fontsize=18)
    for index, (axis, variant) in enumerate(zip(axes, VARIANTS)):
        mesh = axis.imshow(np.ma.masked_invalid(shown[index]), norm=norm, cmap=cmap, aspect='auto')
        commit = data.get('kernel_commits', {}).get(variant)
        version = commit[:7] if commit else data['library_sha256'][variant][:10]
        axis.set_title(f'{"Original" if variant == "baseline" else "Current"} {version}', fontsize=14, pad=10)
        axis.set_xticks(range(len(lengths)), [f'{n//1024}K' if n >= 1024 and n % 1024 == 0 else str(n) for n in lengths])
        axis.set_yticks(range(len(heads)), [f'{h}/{h//data["group_size"]}' for h in heads])
        axis.set_xlabel('Sequence length (tokens)')
        if index == 0:
            axis.set_ylabel('Value heads H / key heads Hg')
        for i, h in enumerate(heads):
            for j, n in enumerate(lengths):
                value = values[index, i, j]
                if np.isnan(value):
                    text, color = 'NaN/Inf', '#374151'
                else:
                    text = f'{value:.2e}' if metric == 'max_abs' else f'{value:.2f}'
                    rgba = cmap(norm(shown[index, i, j]))
                    luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                    color = 'white' if luminance < .5 else '#17202a'
                axis.text(j, i, text, ha='center', va='center', fontsize=9, color=color, weight='bold')
        axis.set_xticks(np.arange(-.5, len(lengths)), minor=True)
        axis.set_yticks(np.arange(-.5, len(heads)), minor=True)
        axis.grid(which='minor', color='white', linewidth=1.5)
        axis.tick_params(which='minor', bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=axes, label=label, shrink=.82, pad=.02)
    if metric == 'max_abs':
        ticks = np.geomspace(norm.vmin, norm.vmax, 5)
        bar.set_ticks(ticks, labels=[f'{v:.1e}' for v in ticks])
        bar.minorticks_off()
    else:
        ticks = np.arange(norm.vmin, norm.vmax + 1)
        bar.set_ticks(ticks, labels=[f'{10**v:g}' for v in ticks])
    oracle = 'CPU FP64 token recurrence' if data['reference'] == 'recurrent-fp64' else 'CPU FP32 chunked reference'
    hardware = data['records'][0]['current']['environment']['device_name']
    state = 'nonzero' if data['initial_state'] else 'zero'
    fig.supxlabel(
        f'{hardware} · identical seeded inputs and reference · D = 128 · GQA group {data["group_size"]} · {state} initial state\n'
        f'{oracle} · maximum over ALL output elements and {data["repeats"]} fresh launches · lower is better\n'
        'Outputs filled with NaN before each launch · finite failures remain visible · gray means a nonfinite output\n'
        + ('Absolute maximum alone does not determine pointwise acceptance; see the tolerance-ratio figure.'
           if metric == 'max_abs' else 'Green: worst ratio ≤ 1; red: worst ratio > 1. RMSE and final state are checked separately.'),
        fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(f'Saved {output.with_suffix(".png")} and {output.with_suffix(".svg")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, help='Output filename stem; produces max-abs and tolerance-ratio PNG/SVG pairs')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    heads, lengths, cells = validate(data)
    stem = args.output or args.input.with_suffix('')
    plot(data, heads, lengths, cells, 'max_abs', stem.with_name(stem.name+'_max_abs'))
    plot(data, heads, lengths, cells, 'max_tolerance_ratio', stem.with_name(stem.name+'_tolerance'))


if __name__ == '__main__':
    main()
