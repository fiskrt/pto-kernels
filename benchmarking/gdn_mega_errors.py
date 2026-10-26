"""Fresh, matched GDN error measurements; numerical failures remain in the plots.

Run in the CANN container. Both libraries use identical inputs, oracle, output
units and repeats. No kernel timing or historical numerical records are reused.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADS = [8, 12, 16, 24, 32, 48, 64]
LENGTHS = [128, 256, 1024, 4096, 8192, 16384, 32768]


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(memoryview(tensor.contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def summarize(repeats, name):
    metrics = [r[name] for r in repeats]
    finite = all(m["finite"] for m in metrics)
    return {
        "finite": finite,
        "passed": all(m["passed"] for m in metrics),
        "elements": metrics[0]["elements"],
        "max_abs": max(m["max_abs"] for m in metrics) if finite else None,
        "max_tolerance_ratio": max(m["max_tolerance_ratio"] for m in metrics) if finite else None,
        "pointwise_violations_max": max(m["pointwise_violations"] for m in metrics) if finite else None,
        "relative_rmse_max": max(m["relative_rmse"] for m in metrics) if finite else None,
        "actual_nonfinite_max": max(m.get("actual_nonfinite", 0) for m in metrics),
    }


def worker(args):
    import torch
    import torch_npu
    sys.path.insert(0, str(ROOT / "tests"))
    from test_gdn_mega import _make_inputs, _expand_kv_heads, _prepare_gdn_mega, _reference
    from gdn_mega_validation import POLICY, error_metrics, recurrent_reference
    from pto_kernels import pto_gdn_mega

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if line.split()[-1].endswith('/libno_workspace_kernel.so')}
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one loaded library, got {libraries}")
    fingerprint = hashlib.sha256(Path(libraries.pop()).read_bytes()).hexdigest()
    props = torch.npu.get_device_properties(args.device)
    environment = dict(device_name=props.name, cube_cores=props.cube_core_num,
                       vector_cores=props.vector_core_num, torch_version=torch.__version__,
                       torch_npu_version=torch_npu.__version__)
    heads = args.heads[0]
    rows = []

    def measure(length):
        inputs = _make_inputs(length, heads, heads // args.group_size)
        inputs[3].mul_(args.gate_scale)
        q, k = _expand_kv_heads(inputs[0], inputs[1], heads)
        initial = (torch.randn(1, heads, 128, 128, dtype=torch.float16) * .1
                   if args.initial_state else None)
        input_hash = tensor_hash([*inputs, initial])
        expected_scaled, expected_state32 = _reference(q, k, *inputs[2:], initial_state=initial)
        expected32 = expected_scaled.double() * 128**.5
        del q, k, expected_scaled
        crosscheck = None
        if args.reference == 'recurrent-fp64':
            expected, expected_state = recurrent_reference(*inputs, initial)
            crosscheck = {
                'raw_output_max_abs': (expected - expected32).abs().max().item(),
                'final_state_max_abs': (expected_state - expected_state32.double()).abs().max().item(),
            }
            del expected32, expected_state32
        else:
            expected, expected_state = expected32, expected_state32
        reference_hash = tensor_hash([expected, expected_state])
        # Nearest FP16 rounding gives the smallest possible error at each element
        # for this fixed oracle; it does not measure internal kernel precision.
        rounding_floor = (expected.half().double() - expected).abs().max().item()
        launch = _prepare_gdn_mega(*(x.to(args.device) for x in inputs),
                                  torch.tensor([0, length], dtype=torch.int32, device=args.device),
                                  initial.to(args.device) if initial is not None else None)
        first = previous = None
        repeats = []
        for repeat in range(args.repeats):
            # Poison only primary outputs. Keep the caller's documented scratch
            # initialization, so both implementations receive the same contract.
            launch['out'].fill_(float('nan'))
            launch['final_state'].fill_(float('nan'))
            pto_gdn_mega(**launch)
            torch.npu.synchronize()
            actual = launch['out'].cpu().clone()
            state = launch['final_state'].cpu().view_as(expected_state).clone()
            if first is None:
                first = (actual, state)
            same_first = all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                             for a, b in zip((actual, state), first))
            same_previous = previous is not None and all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip((actual, state), previous[:2]))
            if same_previous:
                metrics = previous[2]
            else:
                metrics = {'raw_output': error_metrics(actual, expected, include_percentiles=False),
                           'final_state': error_metrics(state, expected_state, include_percentiles=False)}
            repeats.append(dict(repeat=repeat + 1, bitwise_same_as_first=same_first, **metrics))
            previous = (actual, state, metrics)
        unchanged = all(torch.equal(launch[name].cpu(), value)
                        for name, value in zip(('q', 'k', 'v', 'g', 'beta'), inputs))
        if initial is not None:
            unchanged &= torch.equal(launch['initial_state'].cpu(), initial)
        summary = {name: summarize(repeats, name) for name in ('raw_output', 'final_state')}
        passed = unchanged and all(m['passed'] for m in summary.values())
        return dict(length=length, heads=heads, key_heads=heads // args.group_size,
                    status='passed' if passed else 'failed', inputs_unchanged=unchanged,
                    repeats=repeats, summary=summary, fp16_output_rounding_floor=rounding_floor,
                    input_sha256=input_hash, reference_sha256=reference_hash,
                    fp32_reference_vs_fp64=crosscheck)

    for length in args.lengths:
        row = measure(length)
        row.update(kernel_library_sha256=fingerprint, environment=environment, policy=POLICY)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: row[key] for key in ('heads', 'length', 'status', 'summary')}), flush=True)
    # Numerical failures are data for this error comparison. Exceptions still
    # terminate the worker and invalidate an incomplete run.


def save(report, path):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fields = ['heads', 'key_heads', 'length', 'variant', 'status', 'max_abs_raw_output',
              'max_tolerance_ratio_raw_output', 'pointwise_violations_raw_output',
              'nonfinite_raw_output', 'max_abs_final_state', 'fp16_output_rounding_floor']
    with path.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for cell in report['records']:
            for variant in ('baseline', 'current'):
                row = cell[variant]
                m = row['summary']['raw_output']
                writer.writerow(dict(heads=cell['heads'], key_heads=cell['key_heads'], length=cell['length'],
                    variant=variant, status=row['status'], max_abs_raw_output=m['max_abs'],
                    max_tolerance_ratio_raw_output=m['max_tolerance_ratio'],
                    pointwise_violations_raw_output=m['pointwise_violations_max'],
                    nonfinite_raw_output=m['actual_nonfinite_max'],
                    max_abs_final_state=row['summary']['final_state']['max_abs'],
                    fp16_output_rounding_floor=row['fp16_output_rounding_floor']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-lib', type=Path, default=ROOT/'benchmark_data/gdn_mega/baseline_lib')
    parser.add_argument('--current-lib', type=Path, default=ROOT/'build/lib')
    parser.add_argument('--heads', type=int, nargs='+', default=HEADS)
    parser.add_argument('--lengths', type=int, nargs='+', default=LENGTHS)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--gate-scale', type=float, default=1.)
    parser.add_argument('--initial-state', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reference', choices=['chunked-fp32', 'recurrent-fp64'], default='chunked-fp32')
    parser.add_argument('--device', default=os.environ.get('NPU_DEVICE', 'npu:0'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.group_size <= 0 or args.repeats < 1 or any(h not in HEADS or h % args.group_size for h in args.heads)
            or any(n <= 0 for n in args.lengths) or len(set(args.heads)) != len(args.heads)
            or len(set(args.lengths)) != len(args.lengths)):
        parser.error('Use distinct supported shapes, valid grouping, and positive repeats')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(args.heads) != 1:
            parser.error('A worker handles one head count')
        return worker(args)
    libraries = {'baseline': args.baseline_lib.resolve(), 'current': args.current_lib.resolve()}
    hashes = {name: hashlib.sha256((path/'libno_workspace_kernel.so').read_bytes()).hexdigest()
              for name, path in libraries.items()}
    known_path = ROOT/'docs/gdn_mega/data/heatmap.json'
    known = json.loads(known_path.read_text()) if known_path.exists() else {}
    commits = {
        'baseline': (known.get('baseline_source', {}).get('source_commit')
                     if hashes['baseline'] == known.get('library_sha256', {}).get('baseline') else None),
        'current': (known.get('current_kernel_commit')
                    if hashes['current'] == known.get('library_sha256', {}).get('current') else None),
    }
    sources = ['benchmarking/gdn_mega_errors.py', 'tests/gdn_mega_validation.py', 'tests/test_gdn_mega.py']
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), heads=args.heads, lengths=args.lengths,
                  group_size=args.group_size, gate_scale=args.gate_scale, initial_state=args.initial_state,
                  repeats=args.repeats, reference=args.reference, seed=0, head_dim=128, chunk_size=128,
                  device=args.device, output_units='raw unscaled kernel output', library_sha256=hashes,
                  kernel_commits=commits,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                  primary_outputs_initialized='NaN before every repeat; caller scratch initialization preserved',
                  aggregation='Maximum over every element and every repeat; no subsampling or excluded failures',
                  records=[])
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for key in report:
            if key not in ('created_utc', 'records') and previous.get(key) != report[key]:
                parser.error(f'Cannot resume: {key} differs')
        report = previous
    save(report, args.output)
    with tempfile.TemporaryDirectory(prefix='gdn_errors_') as temporary:
        for i, heads in enumerate(args.heads):
            if {r['length'] for r in report['records'] if r['heads'] == heads} == set(args.lengths):
                continue
            runs = {}
            for variant in (('baseline', 'current') if i % 2 == 0 else ('current', 'baseline')):
                path = Path(temporary)/f'{variant}_{heads}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--heads', str(heads),
                           '--lengths', *map(str, args.lengths), '--group-size', str(args.group_size),
                           '--gate-scale', str(args.gate_scale), '--repeats', str(args.repeats),
                           '--reference', args.reference, '--device', args.device, '--output', str(path),
                           '--initial-state' if args.initial_state else '--no-initial-state']
                env = dict(os.environ)
                env['LD_LIBRARY_PATH'] = str(libraries[variant]) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
                env['PYTHONPATH'] = str(ROOT/'python') + os.pathsep + env.get('PYTHONPATH', '')
                print(f'{variant}: H={heads}', flush=True)
                result = subprocess.run(command, env=env, cwd=ROOT, timeout=1800)
                if result.returncode or not path.exists():
                    raise RuntimeError(f'Incomplete worker: {variant}, H={heads}')
                rows = json.loads(path.read_text())
                if len(rows) != len(args.lengths) or {r['length'] for r in rows} != set(args.lengths):
                    raise RuntimeError(f'Incomplete shape grid: {variant}, H={heads}')
                if any(r['kernel_library_sha256'] != hashes[variant] for r in rows):
                    raise RuntimeError(f'Wrong library loaded: {variant}')
                runs[variant] = {r['length']: r for r in rows}
            for length in args.lengths:
                before, after = runs['baseline'][length], runs['current'][length]
                for key in ('input_sha256', 'reference_sha256', 'environment', 'policy'):
                    if before[key] != after[key]:
                        raise RuntimeError(f'Comparison mismatch: H={heads}, T={length}, {key}')
                report['records'].append(dict(heads=heads, key_heads=heads//args.group_size, length=length,
                                              baseline=before, current=after))
            save(report, args.output)
    print(f'Saved {len(report["records"])} matched comparisons to {args.output}', flush=True)


if __name__ == '__main__':
    main()
