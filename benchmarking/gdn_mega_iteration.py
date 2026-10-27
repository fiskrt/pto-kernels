"""Build an isolated GDN candidate and run the ordinary precision tests.

Compile head specializations in parallel, without changing production sources,
the installed library, or the kernel schedule. Run through check_gdn_mega.sh.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HEADS = (8, 12, 16, 24, 32, 48, 64)
QUICK_CASES = (
    'ordinary_seed2027', 'weak_seed119', 'packed_empty',
    'packed_empty_zero_state', 'correlated_beta1', 'nearly_correlated_beta1',
    'fla_positive_strong_tail75',
    'critical_nearly_correlated_3011_128.0', 'critical_tiny_gate_-1e-07_4097',
)
TESTS = ('tests/test_gdn_mega.py', 'tests/test_gdn_mega_validation.py',
         'tests/test_gdn_mega_errors.py', 'tests/test_gdn_mega_benchmark.py')
FLAGS = ['-DMEMORY_BASE', '-Dno_workspace_kernel_EXPORTS',
         '-I/usr/local/Ascend/driver/kernel/inc', '-O0', '-g', '-fPIC',
         '-D_FORTIFY_SOURCE=2', '-O2', '-std=c++17', '-Wno-macro-redefined',
         '-Wno-ignored-attributes', '-fstack-protector-strong',
         '--npu-arch=dav-2201', '--asc-aicore-lang']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, destination, heads):
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in source.iterdir() if p.suffix in ('.hpp', '.cpp'))
    hashes = {p.name: sha256(p) for p in sources}
    for path in sources:
        shutil.copyfile(path, destination / path.name)
    source_text = (source / 'kernel_gdn_mega.cpp').read_text()
    signature = re.search(r'extern "C" void pto_launch_gdn_mega\([\s\S]*?\)\s*{', source_text)
    if signature is None:
        raise ValueError('Cannot identify the production launch signature')
    declaration = signature.group()[:-1].strip()
    arguments = [p.strip().split()[-1].lstrip('*')
                 for p in declaration.split('(', 1)[1][:-1].split(',')]
    calls = re.findall(r'    case (\d+): LAUNCH_MEGA\(\1\); break;', source_text)
    if tuple(map(int, calls)) != HEADS:
        raise ValueError('Expected the complete production head dispatch')

    def compile_head(head):
        text = re.sub(r'    case (\d+): LAUNCH_MEGA\(\1\); break;\n',
                      lambda m: m.group() if int(m[1]) == head else '', source_text)
        text = text.replace('void pto_launch_gdn_mega(', f'void pto_launch_gdn_mega_{head}(')
        path = destination / f'head_{head}.cpp'
        path.write_text(text)
        with (destination / f'head_{head}.log').open('w') as log:
            subprocess.run(['bisheng', *FLAGS, '-c', str(path), '-o', str(path.with_suffix('.o'))],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(heads)) as executor:
        list(executor.map(compile_head, heads))
    shim = '#include <cstdint>\n' + '\n'.join(
        declaration.replace('void pto_launch_gdn_mega(', f'void pto_launch_gdn_mega_{h}(') + ';'
        for h in heads)
    shim += '\n' + declaration + '{\n switch(num_heads){\n'
    shim += '\n'.join(f' case {h}: pto_launch_gdn_mega_{h}({", ".join(arguments)}); break;'
                      for h in heads)
    shim += '\n }\n}\n'
    (destination / 'launch.cpp').write_text(shim)
    subprocess.run(['bisheng', '-fPIC', '-O2', '-c', str(destination / 'launch.cpp'),
                    '-o', str(destination / 'launch.o')], check=True)
    library = destination / 'libno_workspace_kernel.so'
    subprocess.run(['bisheng', '-shared', '-s', '-Wl,-z,relro', '-Wl,-z,now',
                    '-Wl,-soname,libno_workspace_kernel.so', '-o', str(library),
                    str(destination / 'launch.o'),
                    *(str(destination / f'head_{h}.o') for h in heads)], check=True)
    return dict(source_sha256=hashes, heads=heads, library_sha256=sha256(library),
                build_seconds=time.perf_counter()-started)


def verify(manifest_path):
    # Import only after the new process's loader path selects the candidate.
    import torch
    import torch_npu  # noqa: F401
    import pto_kernels  # noqa: F401
    import pytest
    manifest = json.loads(manifest_path.read_text())
    loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if line.split()[-1].endswith('/libno_workspace_kernel.so')}
    if len(loaded) != 1 or sha256(Path(loaded.pop())) != manifest['library_sha256']:
        raise RuntimeError('Loaded library does not match the newly compiled candidate')
    torch.set_num_threads(4)
    if manifest['quick']:
        sys.path.insert(0, str(ROOT / 'tests'))
        from gdn_mega_cases import AUDIT_CASES
        selected = {c['name']: c for c in AUDIT_CASES if c['name'] in QUICK_CASES}
        assert len(selected) == len(QUICK_CASES)
        assert all(c['heads'] in manifest['heads'] for c in selected.values())
        tests = [f'tests/test_gdn_mega_validation.py::test_independent_recurrence_and_memory[{name}]'
                 for name in QUICK_CASES]
    else:
        tests = list(TESTS)
    started = time.perf_counter()
    status = pytest.main([*tests, '-q'])
    manifest.update(test_seconds=time.perf_counter()-started, test_exit_code=int(status),
                    coverage='partial fixed screen' if manifest['quick'] else 'full regression suite')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    if status == 0 and manifest['quick']:
        import gdn_mega
        # Cover both the shared local-preparation path and the spare-core path.
        for heads in (8, 16):
            filename = 'benchmark_h8.json' if heads == 8 else 'benchmark.json'
            sys.argv = ['gdn_mega.py', '--heads', str(heads), '--key-heads', str(heads // 4),
                        '--lengths', '256', '4096', '--initial-state',
                        '--output', str(manifest_path.parent/filename)]
            gdn_mega.main()
    raise SystemExit(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Partial fixed screen; full suite is required before promotion')
    parser.add_argument('--source', type=Path, default=ROOT/'csrc/kernel/gdn_mega')
    parser.add_argument('--output', type=Path, default=ROOT/'build/gdn_mega_iteration')
    parser.add_argument('--build-only', action='store_true', help='Compile without starting an NPU job')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        verify(args.worker)
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Candidate build must be outside the production source directory')
    report = build(source, output, (8, 16) if args.quick else HEADS)
    report.update(quick=args.quick, source=str(source))
    manifest = output/'verification.json'
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Built {len(report['heads'])} head specializations in {report['build_seconds']:.2f}s: {output}", flush=True)
    if not args.build_only:
        environment = dict(os.environ, LD_LIBRARY_PATH=f"{output}:{os.environ.get('LD_LIBRARY_PATH', '')}")
        environment.setdefault('GDN_REFERENCE_CACHE', str(ROOT/'benchmark_data/gdn_mega/reference_cache'))
        subprocess.run([sys.executable, __file__, '--worker', str(manifest)], env=environment, check=True)


if __name__ == '__main__':
    main()
