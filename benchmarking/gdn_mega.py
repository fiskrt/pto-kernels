"""Allocation-free GDN timing and numerical validation on a real NPU.

PYTHONPATH=python timeout 120 python benchmarking/gdn_mega.py --output baseline.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_gdn_mega import (  # noqa: E402
    _expand_kv_heads, _make_inputs, _prepare_gdn_mega, _reference,
)
from gdn_mega_validation import POLICY, error_metrics, recurrent_reference
from test_gdn_mega_validation import AUDIT_CASES, make_audit_inputs  # noqa: E402
from pto_kernels import pto_gdn_mega  # noqa: E402


class AccuracyFailure(AssertionError):
    def __init__(self, metrics):
        self.metrics = metrics
        super().__init__("Pointwise/RMSE acceptance failed; see numerical_errors")


def check_outputs(launch, expected, expected_state, previous=None):
    """Validate against fixed oracle tensors; cache only within that benchmark case."""
    # Keep independent snapshots: even CPU test inputs may be mutated in place.
    raw = launch["out"].cpu().clone()
    state = launch["final_state"].cpu().view_as(expected_state).clone()
    if previous is not None and previous[3] is expected and previous[4] is expected_state and all(
            a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip((raw, state), previous[:2])):
        # Every bit was checked. Recomputing exact percentiles for identical
        # tensors adds minutes to large sweeps without changing acceptance.
        return previous
    raw_double = raw.double()
    metrics = {"raw_output": error_metrics(raw_double, expected.double() * 128**.5),
               "scaled_output": error_metrics(raw_double * 128**-.5, expected),
               "final_state": error_metrics(state, expected_state)}
    if not all(value["passed"] for value in metrics.values()):
        raise AccuracyFailure(metrics)
    return raw, state, metrics, expected, expected_state


def benchmark_case(args, length, audit_case=None):
    """Check and time one shape; release its graph/tensors on return."""
    if audit_case is None:
        inputs = _make_inputs(length, args.heads, args.key_heads)
        inputs[3].mul_(args.gate_scale)
        q, k = _expand_kv_heads(inputs[0], inputs[1], args.heads)
        initial = torch.randn(1, args.heads, 128, 128, dtype=torch.float16) * 0.1 if args.initial_state else None
        expected, expected_state = _reference(q, k, *inputs[2:], initial_state=initial)
        cu = torch.tensor([0, length], dtype=torch.int32)
    else:
        inputs, initial, cu = make_audit_inputs(audit_case)
        expected, expected_state = recurrent_reference(*inputs, initial, cu)
        expected = expected * 128**-.5
    launch = _prepare_gdn_mega(
        *(t.to(args.device) for t in inputs), cu.to(args.device),
        initial.to(args.device) if initial is not None else None,
    )
    checked = None
    for _ in range(1 if args.profile_once else 5):
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        checked = check_outputs(launch, expected, expected_state, checked)
    metrics = checked[2]
    if args.profile_once:
        print(json.dumps(dict(length=length, numerical_errors=metrics, validation_policy=POLICY)), flush=True)
        return None
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
    record = dict(length=length, heads=args.heads, key_heads=args.key_heads, initial_state=args.initial_state, gate_scale=args.gate_scale,
                  median_us=statistics.median(times), min_us=min(times), max_us=max(times),
                  numerical_errors=metrics)
    if audit_case is not None:
        record.update(audit_case=audit_case["name"], input_distribution=audit_case,
                      heads=audit_case["heads"], key_heads=audit_case["key_heads"],
                      initial_state=initial is not None, gate_scale=audit_case.get("gate_scale", 1.))
    # Verify the timed graph also leaves correct outputs, outside its timing.
    record["after_replay_errors"] = check_outputs(launch, expected, expected_state, checked)[2]
    record["validation_reuse"] = "Statistics reused only after byte equality of every output/state element"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 1024, 4096])
    parser.add_argument("--audit-cases", nargs="+", choices=[c["name"] for c in AUDIT_CASES],
                        help="Time fixed audit distributions with the CPU FP64 oracle; overrides shape/input options")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--initial-state", action="store_true", help="Benchmark with a nonzero incoming recurrent state")
    parser.add_argument("--gate-scale", type=float, default=1.0, help="Scale log gates to exercise different recurrent memory lengths")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-once", action="store_true", help="One checked launch, without a timing graph")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    # Verify which implementation the dynamic loader actually selected.
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if line.split()[-1].endswith("/libno_workspace_kernel.so")}
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one loaded kernel library, found {libraries}")
    library_hash = hashlib.sha256(Path(libraries.pop()).read_bytes()).hexdigest()
    properties = torch.npu.get_device_properties(args.device)
    environment = dict(device_name=properties.name, cube_cores=properties.cube_core_num,
                       vector_cores=properties.vector_core_num, torch_version=torch.__version__,
                       torch_npu_version=torch_npu.__version__)
    records = []
    cases = [(sum(c["lengths"]), c) for c in AUDIT_CASES if c["name"] in args.audit_cases] if args.audit_cases else [(n, None) for n in args.lengths]
    for length, audit_case in cases:
        try:
            record = benchmark_case(args, length, audit_case)
            if record is not None:
                record["status"] = "passed"
        except AssertionError as error:
            # Preserve numerical failures, never assign them a timing/speedup.
            record = dict(length=length, heads=args.heads, key_heads=args.key_heads,
                          initial_state=args.initial_state, gate_scale=args.gate_scale,
                          status="failed", error=str(error))
            if isinstance(error, AccuracyFailure):
                record["numerical_errors"] = error.metrics
        if record is not None:
            record["kernel_library_sha256"] = library_hash
            record["validation_policy"] = POLICY
            record["input_generator_sha256"] = hashlib.sha256((Path(__file__).resolve().parents[1] / "tests" /
                ("test_gdn_mega_validation.py" if audit_case else "test_gdn_mega.py")).read_bytes()).hexdigest()
            if audit_case is not None:
                record.update(audit_case=audit_case["name"], input_distribution=audit_case,
                              heads=audit_case["heads"], key_heads=audit_case["key_heads"],
                              initial_state=audit_case.get("initial", True), gate_scale=audit_case.get("gate_scale", 1.))
            record["reference"] = "CPU FP64 token recurrence" if audit_case else "CPU FP32 chunked; independent FP64 audit in audit_gdn_mega.py"
            record["environment"] = environment
            records.append(record)
            print(json.dumps(record), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(records, indent=2) + "\n")

    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
