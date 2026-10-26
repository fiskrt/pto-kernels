"""Independent GDN oracle and explicit numerical acceptance policy.

The oracle is CPU FP64 token recurrence: no chunking, inverse, norm threshold,
FP16 intermediate rounding, or dependency on the production kernel.
"""
import math

import torch


ATOL = 1e-5
RTOL = 1e-3
MAX_RMSE_RATIO = 0.005
POLICY = {"atol": ATOL, "rtol": RTOL, "max_relative_rmse": MAX_RMSE_RATIO,
          "criterion": "every element: abs(error) <= atol + rtol * abs(reference), AND relative RMSE"}


def recurrent_reference(q, k, v, g, beta, initial_state=None, cu_seqlens=None):
    """Return raw (unscaled) output and final state from exact supplied inputs."""
    q, k, v, g, beta = [x.detach().cpu().double() for x in (q, k, v, g, beta)]
    _, tokens, heads, dim = v.shape
    group = heads // k.shape[2]
    q, k = q.repeat_interleave(group, 2), k.repeat_interleave(group, 2)
    boundaries = [0, tokens] if cu_seqlens is None else cu_seqlens.cpu().tolist()
    output = torch.empty_like(v)
    states = []
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (torch.zeros(heads, dim, dim, dtype=torch.float64) if initial_state is None
                 else initial_state[seq].detach().cpu().double().clone())
        for token in range(start, end):
            state = state * g[0, token].exp()[:, None, None]
            prediction = (k[0, token, :, :, None] * state).sum(dim=1)
            delta = beta[0, token, :, None] * (v[0, token] - prediction)
            state = state + k[0, token, :, :, None] * delta[:, None, :]
            output[0, token] = (q[0, token, :, :, None] * state).sum(dim=1)
        states.append(state)
    return output, torch.stack(states)


def error_metrics(actual, expected, *, include_percentiles=True):
    """No broadcasting, rescaling, fitting, or outlier removal is allowed."""
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"passed": False, "finite": False, "elements": actual.numel(),
                "actual_nonfinite": int((~torch.isfinite(actual)).sum()),
                "reference_nonfinite": int((~torch.isfinite(expected)).sum()),
                "reference_abs_max": expected.abs().max().item() if bool(torch.isfinite(expected).all()) else None}
    diff = (actual - expected).abs()
    if not diff.numel():
        return {"passed": True, "finite": True, "elements": 0, "max_abs": 0.0,
                "relative_rmse": 0.0, "pointwise_violations": 0}
    allowed = ATOL + RTOL * expected.abs()
    relative_rmse = (diff.square().mean() / expected.square().mean().clamp_min(1e-16)).sqrt().item()
    violations = int((diff > allowed).sum())
    max_index = int(diff.flatten().argmax())
    # Exact nearest-rank percentiles. torch.quantile rejects tensors above 2**24
    # elements (already exceeded by T=32768, H=8); never subsample outliers.
    percentiles = ({name: diff.flatten().kthvalue(max(1, math.ceil(p * diff.numel()))).values.item()
                    for name, p in (("p50_abs", .5), ("p99_abs", .99), ("p999_abs", .999))}
                   if include_percentiles else {})
    centered = (expected - expected.mean()).square().sum()
    r_squared = 1 - diff.square().sum().item() / max(centered.item(), 1e-30)
    return {
        "passed": violations == 0 and relative_rmse < MAX_RMSE_RATIO,
        "finite": True, "elements": diff.numel(), "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(), **percentiles,
        "relative_rmse": relative_rmse, "r_squared": r_squared,
        "pointwise_violations": violations, "max_tolerance_ratio": (diff / allowed).max().item(),
        "reference_abs_max": expected.abs().max().item(), "max_error_flat_index": max_index,
        "actual_at_max_error": actual.flatten()[max_index].item(),
        "reference_at_max_error": expected.flatten()[max_index].item(),
        "abs_gt_1e-4": int((diff > 1e-4).sum()), "abs_gt_1e-3": int((diff > 1e-3).sum()),
    }


def assert_numerically_close(name, actual, expected):
    metrics = error_metrics(actual, expected)
    assert metrics["passed"], f"{name}: policy={POLICY}; errors={metrics}"
