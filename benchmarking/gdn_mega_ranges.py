"""Measure GDN test-input ranges and inverse-power growth on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_gdn_mega_validation import AUDIT_CASES, make_audit_inputs


def describe_inputs(case, inputs, initial):
    q, k, v, g, beta = inputs
    key = k[0, :128, 0].double()
    prefix = g[0, :128, 0].double().cumsum(0)
    n = key.shape[0]
    gram = key @ key.T
    eye = torch.eye(n, dtype=torch.float64)
    a = -(gram * torch.exp(torch.tril(prefix[:, None] - prefix[None, :])) * beta[0, :n, 0].double()[:, None]).tril(-1)
    power, powers = a.clone(), {}
    for degree in (1, 2, 4, 8, 16, 32, 64):
        if degree > 1:
            power = power @ power
        powers[degree] = float(power.abs().max())
    inverse = torch.linalg.solve_triangular(eye - a, eye, upper=False, unitriangular=True)
    return dict(case=case['name'], config=case, q_abs_max=float(q.abs().max()),
        k_norm_min=float(k.float().norm(dim=-1).min()), k_norm_max=float(k.float().norm(dim=-1).max()),
        v_abs_max=float(v.abs().max()), state_abs_max=float(initial.abs().max()) if initial is not None else 0.,
        g_min=float(g.min()), g_max=float(g.max()), beta_min=float(beta.min()), beta_max=float(beta.max()),
        measured_block_tokens=n, first_head_first_chunk_kk_offdiag_mean=float(gram[~eye.bool()].mean()),
        first_head_first_chunk_inverse_abs_max=float(inverse.abs().max()), first_head_first_chunk_powers=powers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/gdn_mega/data/input_ranges.json'))
    parser.add_argument('--cases', nargs='+', choices=[c['name'] for c in AUDIT_CASES])
    args = parser.parse_args()
    torch.set_num_threads(4)
    names = args.cases or [c['name'] for c in AUDIT_CASES
                          if 'fla' in c or c['name'] in ('ordinary_seed2027', 'correlated_beta1')]
    report = {'kind': 'CPU input ranges', 'fla_source_commit': '12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e',
              'input_generator_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] /
                  'tests/test_gdn_mega_validation.py').read_bytes()).hexdigest(), 'cases': []}
    for name in names:
        case = next(c for c in AUDIT_CASES if c['name'] == name)
        inputs, initial, _ = make_audit_inputs(case)
        report['cases'].append(describe_inputs(case, inputs, initial))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {len(report["cases"])} input-range cases to {args.output}')


if __name__ == '__main__':
    main()
