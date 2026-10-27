"""Run every independent GDN accuracy case and save failures without hiding them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import torch
import torch_npu  # noqa: F401
import pto_kernels.pto_kernels_ops as extension

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from gdn_mega_validation import POLICY  # noqa: E402
from gdn_mega_cases import AUDIT_CASES, STANDARD_CASES, CRITICAL_CASES, CORRELATION_CASES
from test_gdn_mega_validation import make_capture_inputs, run_audit_case  # noqa: E402


def capture_cases(store):
    cases = []
    longest = max(store, key=lambda sample: next(iter(store[sample].values()))["q"].shape[0])
    for sample, layers in store.items():
        for layer, record in sorted(layers.items()):
            tokens, key_heads, dim = record["q"].shape
            heads = record["v"].shape[1]
            assert dim == record["v"].shape[-1] == 128
            for start in ([0, tokens // 2] if sample == longest else [0]):
                cases.append(dict(name=f"capture_{sample}_L{layer}_start{start}",
                                  sample=sample, layer=layer, start=start,
                                  lengths=(tokens-start,), heads=heads, key_heads=key_heads,
                                  initial=bool(start), per_head=True))
    return cases


def capture_statistics(inputs):
    """Headwise distributions and actual strict-lower 128-token coupling."""
    _, keys, values, gates, beta = (x[0].double() for x in inputs)
    probabilities = torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)
    group = beta.shape[1] // keys.shape[1]
    result = []
    for head in range(beta.shape[1]):
        key = keys[:, head // group]
        cosines, couplings = [], []
        for start in range(0, len(key), 128):
            k = key[start:start+128]
            rows, cols = torch.tril_indices(len(k), len(k), offset=-1)
            gram = k @ k.T
            norm = k.norm(dim=-1)
            cosines.append(gram[rows, cols] / (norm[rows] * norm[cols]).clamp_min(1e-30))
            prefix = gates[start:start+128, head].cumsum(0)
            couplings.append(gram[rows, cols].abs() * beta[start:start+128, head][rows]
                             * (prefix[rows]-prefix[cols]).exp())
        cosines, couplings = torch.cat(cosines), torch.cat(couplings)
        result.append({"head": head,
                       "beta_quantiles": torch.quantile(beta[:, head], probabilities).tolist(),
                       "log_gate_quantiles": torch.quantile(gates[:, head], probabilities).tolist(),
                       "alpha_above_0.999": float((gates[:, head].exp() > .999).double().mean()),
                       "within_chunk_cosine_mean": cosines.mean().item(),
                       "within_chunk_abs_cosine_quantiles": torch.quantile(cosines.abs(), probabilities).tolist(),
                       "coupling_abs_max": couplings.max().item(),
                       "coupling_above_0.5": int((couplings > .5).sum()),
                       "value_abs_max": values[:, head].abs().max().item()})
    return {"quantile_probabilities": probabilities.tolist(), "per_head": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--suite", choices=["standard", "critical", "correlation", "capture", "all"], default="standard")
    parser.add_argument("--timing", action="store_true", help="Also measure 15 trials of 20 graph calls and recheck every output/state element")
    parser.add_argument("--capture", type=Path, help="Captured gdn_keys.pt; required with --suite capture")
    all_cases = AUDIT_CASES
    parser.add_argument("--cases", nargs="+",
                        help="Explicit subset for reproducing failures; default runs every case")
    args = parser.parse_args()
    from gdn_mega import time_launch
    if (args.suite == "capture") != (args.capture is not None):
        parser.error("--capture is required exactly when --suite capture is selected")
    captured = torch.load(args.capture, map_location="cpu", weights_only=True)["store"] if args.capture else None
    replay_cases = capture_cases(captured) if captured is not None else []
    cases = {"standard": STANDARD_CASES, "critical": CRITICAL_CASES,
             "correlation": CORRELATION_CASES, "capture": replay_cases, "all": all_cases}[args.suite]
    if args.cases:
        available = replay_cases if captured is not None else all_cases
        unknown = set(args.cases) - {c["name"] for c in available}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in available if c["name"] in args.cases]
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if line.split()[-1].endswith("/libno_workspace_kernel.so")}
    assert len(libraries) == 1
    root = Path(__file__).resolve().parents[1]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "device": torch.npu.get_device_properties(args.device).name,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "requested_cases": [c["name"] for c in cases],
              "suite": args.suite, "production_beta_range": [0., 1.],
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                  ("tests/gdn_mega_validation.py", "tests/gdn_mega_cases.py", "tests/test_gdn_mega_validation.py", "tests/test_gdn_mega.py",
                   "benchmarking/audit_gdn_mega.py", "benchmarking/gdn_mega.py", "csrc/host/utils.h")},
              "host_extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
              "policy": POLICY, "oracle": "CPU FP64 token recurrence, raw output",
              "kernel_library_sha256": hashlib.sha256(Path(libraries.pop()).read_bytes()).hexdigest(), "cases": []}
    if captured is not None:
        with args.capture.open("rb") as stream:
            capture_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        report["capture"] = {"path": str(args.capture), "sha256": capture_hash,
                             "model": "Qwen/Qwen3.5-0.8B (user-supplied capture)",
                             "conversion": "Captured FP16 Q/K/V unchanged; beta cast FP32 to FP16 for kernel ABI; g stays FP32. No renormalization, clipping, repetition or synthetic input scaling.",
                             "initial_state": "Full traces start from zero. Continuation on all layers of the first longest prompt uses FP16-rounded independent FP64 prefix state, identically in kernel and suffix oracle."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        prepared = (make_capture_inputs(captured[case["sample"]][case["layer"]], case["start"])
                    if captured is not None else None)
        result = run_audit_case(case, args.device, prepared_inputs=prepared,
                                benchmark=time_launch if args.timing else None)
        if prepared is not None:
            result["capture_statistics"] = capture_statistics(prepared[0])
        bounds = result["input_ranges"]["beta"]
        result["beta_in_production_range"] = 0. <= bounds[0] <= bounds[1] <= 1.
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        sample = result["repeats"][0]
        print(json.dumps({"case": case["name"], "passed": result["passed"],
                          "raw_output": sample["raw_output"], "final_state": sample["final_state"]}), flush=True)
    failed = sum(not c["passed"] for c in report["cases"])
    print(f"{len(report['cases']) - failed} passed; {failed} failed; saved {args.output}", flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
