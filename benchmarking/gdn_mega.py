"""Allocation-free GDN timing and numerical validation on a real NPU.

PYTHONPATH=python timeout 120 python benchmarking/gdn_mega.py --output baseline.json
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_gdn_mega import (  # noqa: E402
    _assert_close, _expand_kv_heads, _make_inputs, _prepare_gdn_mega, _reference,
)
from pto_kernels import pto_gdn_mega  # noqa: E402


def errors(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    diff = actual - expected
    return {
        "max_abs": diff.abs().max().item(),
        "relative_rmse": (diff.square().mean() / expected.square().mean().clamp_min(1e-16)).sqrt().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--initial-state", action="store_true", help="Benchmark with a nonzero incoming recurrent state")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-checks", action="store_true", help="For deliberately truncated profiling kernels only")
    parser.add_argument("--stage-timing", action="store_true", help="Read GDN_STAGE_TIMING instrumentation")
    parser.add_argument("--profile-once", action="store_true", help="One checked launch, without a timing graph")
    parser.add_argument("--fused-timing", action="store_true", help="Read GDN_FUSED_TIMING counters")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    records = []
    for length in args.lengths:
        inputs = _make_inputs(length, args.heads, args.key_heads)
        q, k = _expand_kv_heads(inputs[0], inputs[1], args.heads)
        initial = torch.randn(1, args.heads, 128, 128, dtype=torch.float16) * 0.1 if args.initial_state else None
        expected, expected_state = _reference(q, k, *inputs[2:], initial_state=initial)
        launch = _prepare_gdn_mega(
            *(t.to(args.device) for t in inputs),
            torch.tensor([0, length], dtype=torch.int32, device=args.device),
            initial.to(args.device) if initial is not None else None,
        )
        for _ in range(1 if args.profile_once else 5):
            pto_gdn_mega(**launch)
            torch.npu.synchronize()
            actual = launch["out"].cpu().float() * 128**-0.5
            state = launch["final_state"].cpu().view_as(expected_state)
            if not args.skip_checks:
                _assert_close("out", actual, expected)
                _assert_close("state", state, expected_state)
        if args.profile_once:
            print(json.dumps(dict(length=length, output_error=errors(actual, expected),
                                  state_error=errors(state, expected_state))), flush=True)
            continue
        # Capture launches so Python dispatch cannot leave gaps between kernels.
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.inner):
                pto_gdn_mega(**launch)
        graph.replay()
        torch.npu.synchronize()
        times = []
        for _ in range(args.trials):
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / args.inner)
        record = dict(length=length, heads=args.heads, key_heads=args.key_heads, initial_state=args.initial_state,
                      median_us=statistics.median(times), min_us=min(times), max_us=max(times),
                      output_error=errors(actual, expected), state_error=errors(state, expected_state))
        records.append(record)
        if args.fused_timing:
            counters = launch["kkt_workspace"].view(torch.int64).flatten()[:48].cpu().tolist()
            record["fused_ticks"] = {name: counters[i * 16:i * 16 + 4]
                                     for i, name in enumerate(["cube", "vec0", "vec1"])}
        if args.stage_timing:
            ticks = launch["A_inv_f32"].view(torch.int64).flatten()[:7].cpu().tolist()
            record["stage_ticks"] = dict(zip(
                ["gates", "kkt", "inverse", "wy", "state", "output"],
                [b - a for a, b in zip(ticks, ticks[1:])],
            ))
        print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
