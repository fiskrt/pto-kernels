"""CPU numerical screening of alternative chunk solves; never times the NPU.

Uses the strict independent FP64 token oracle and exact public input tensors.
Candidates compare triangular solves and precomputed affine chunk maps,
with other arithmetic in FP32. --ranges-only measures the input generators.
These results are feasibility screens, not hardware performance claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_gdn_mega_validation import AUDIT_CASES, make_audit_inputs
from gdn_mega_validation import POLICY, error_metrics, recurrent_reference


def half(x):
    return x.half().float()


def pair(x):
    hi = half(x)
    return hi, half(x - hi)


def product(a, b):
    ah, al = pair(a)
    bh, bl = pair(b)
    return (ah @ bl + al @ bh) + ah @ bh


def approximate_inverse(a):
    n = a.shape[-1]
    x = half(a)
    eye = torch.eye(n, dtype=a.dtype)
    for i in range(0, n, 16):
        sl = slice(i, i + 16)
        x[..., sl, sl] = half(torch.linalg.solve_triangular(
            eye[sl, sl] - a[..., sl, sl], eye[sl, sl].expand_as(a[..., sl, sl]),
            upper=False, unitriangular=True))
    for block in (16, 32, 64):
        for i in range(0, n, 2 * block):
            lo, hi = slice(i + block, i + 2 * block), slice(i, i + block)
            x[..., lo, hi] = half(half(x[..., lo, lo] @ x[..., lo, hi]) @ x[..., hi, hi])
    return x


def solve(a, b, method):
    eye = torch.eye(a.shape[-1])
    if method == 'fp32_direct':
        return torch.linalg.solve_triangular(eye - a, b, upper=False, unitriangular=True)
    if method == 'block_series':
        diagonal = torch.zeros_like(a)
        off = a.clone()
        for i in range(0, a.shape[-1], 16):
            sl = slice(i, i + 16)
            diagonal[..., sl, sl] = torch.linalg.solve_triangular(eye[sl, sl] - a[..., sl, sl],
                eye[sl, sl].expand_as(a[..., sl, sl]), upper=False, unitriangular=True)
            off[..., sl, sl] = 0
        power = off @ diagonal
        result = diagonal @ (eye + power)
        for _ in range(2):
            power = power @ power
            result = result + result @ power
        return result @ b
    if method == 'panel16':
        result = torch.zeros_like(b)
        for i in range(0, a.shape[-1], 16):
            sl = slice(i, i + 16)
            rhs = b[..., sl, :] + product(a[..., sl, :i], result[..., :i, :])
            diagonal = torch.linalg.solve_triangular(eye[sl, sl] - a[..., sl, sl],
                eye[sl, sl].expand_as(a[..., sl, sl]), upper=False, unitriangular=True)
            result[..., sl, :] = product(diagonal, rhs)
        return result
    if method.startswith('refine'):
        inverse = approximate_inverse(a)
        result = product(inverse, b)
        for _ in range(int(method[-1])):
            residual = (b - result) + product(a, result)
            result += product(inverse, residual)
        return result
    raise ValueError(method)


def run(inputs, initial, cu, method):
    q, k, v, g, beta = [x.float()[0] for x in inputs]
    repeat = v.shape[1] // k.shape[1]
    q, k = [x.repeat_interleave(repeat, 1) for x in (q, k)]
    output, states = torch.empty_like(v), []
    for seq, (begin, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        state = initial[seq].float().clone() if initial is not None else torch.zeros(v.shape[1], 128, 128)
        for pos in range(begin, end, 128):
            size = min(128, end - pos)
            def padded(x):
                t = torch.zeros((128,) + x.shape[1:], dtype=x.dtype)
                t[:size] = x[pos:pos+size]
                return t.transpose(0, 1)
            keys, queries, values, rates = [padded(x) for x in (k, q, v, beta)]
            prefix = padded(g).double().cumsum(-1)
            decay = torch.exp(torch.tril(prefix[..., :, None] - prefix[..., None, :])).float().tril()
            a = -(keys @ keys.transpose(-1, -2) * decay * rates[..., :, None]).tril(-1)
            expg = prefix.exp().float()
            if method == 'affine_maps':
                u = solve(a, rates[..., :, None] * values, 'fp32_direct')
                w = solve(a, rates[..., :, None] * expg[..., :, None] * keys, 'fp32_direct')
                attention = (queries @ keys.transpose(-1, -2)) * decay
                tail = (prefix[..., size-1, None] - prefix).exp().float()
                tail[..., size:] = 0
                transition = expg[..., size-1, None, None] * torch.eye(128) - keys.transpose(-1, -2) @ (tail[..., None] * w)
                injection = keys.transpose(-1, -2) @ (tail[..., None] * u)
                projection = expg[..., :, None] * queries - attention @ w
                local = attention @ u
                output[pos:pos+size] = (product(projection, state) + local).transpose(0, 1)[:size]
                state = product(transition, state) + injection
                continue
            rhs = rates[..., :, None] * (values - expg[..., :, None] * (keys @ state))
            residual = solve(a, rhs, method)
            attention = (queries @ keys.transpose(-1, -2)) * decay
            output[pos:pos+size] = (expg[..., :, None] * (queries @ state) + product(attention, residual)).transpose(0, 1)[:size]
            scale = (prefix[..., size-1, None] - prefix).exp().float()
            scale[..., size:] = 0
            state = expg[..., size-1, None, None] * state + keys.transpose(-1, -2) @ (scale[..., None] * residual)
        states.append(state)
    return output[None].half(), torch.stack(states).half()


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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['fp32_direct', 'panel16', 'refine1', 'refine2', 'block_series', 'affine_maps'])
    parser.add_argument('--cases', nargs='+', choices=[c['name'] for c in AUDIT_CASES])
    parser.add_argument('--ranges-only', action='store_true', help='Measure input ranges and inverse-power growth; skip algorithm screening')
    args = parser.parse_args()
    torch.set_num_threads(4)
    names = args.cases or ([c['name'] for c in AUDIT_CASES if 'fla' in c or c['name'] in ('ordinary_seed2027', 'correlated_beta1')]
        if args.ranges_only else ['ordinary_seed2027', 'no_decay', 'nearly_correlated_beta1',
            'fla_positive_no_decay', 'fla_positive_weak', 'fla_negative_eigenvalues'])
    report = {'kind': 'CPU input ranges' if args.ranges_only else 'CPU numerical screen; no NPU timing',
        'policy': POLICY, 'fla_source_commit': '12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e',
        'input_generator_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'tests' / 'test_gdn_mega_validation.py').read_bytes()).hexdigest(),
        'cases': []}
    for name in names:
        case = next(c for c in AUDIT_CASES if c['name'] == name)
        inputs, initial, cu = make_audit_inputs(case)
        if args.ranges_only:
            row = describe_inputs(case, inputs, initial)
            report['cases'].append(row)
            print(name, 'max power', max(row['first_head_first_chunk_powers'].values()), flush=True)
            continue
        expected = recurrent_reference(*inputs, initial, cu)
        for method in args.methods:
            actual = run(inputs, initial, cu, method)
            metrics = [error_metrics(a, e) for a, e in zip(actual, expected)]
            row = {'case': name, 'method': method, 'output': metrics[0], 'state': metrics[1]}
            report['cases'].append(row)
            print(name, method, [(m['pointwise_violations'], m['max_abs']) for m in metrics], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
