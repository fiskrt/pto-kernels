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
from test_gdn_mega_validation import AUDIT_CASES, make_capture_inputs, run_audit_case  # noqa: E402


# Fixed before evaluation. All critical cases obey the user's beta range [0,1].
# Keep these separate from the standard suite so newly found failures remain
# reproducible without silently redefining the passing regression suite.
CRITICAL_CASES = [
    dict(name=f'critical_boundary_{rate}_{amplitude}', seed=3001, lengths=(513,), heads=8, key_heads=2,
         correlated=True, beta=rate, gate_scale=0., amplitude=amplitude, state_scale=1.)
    for rate in (.499755859375, .5, .50048828125, 1.) for amplitude in (1., 32.)
] + [
    dict(name=f'critical_tiny_gate_{gate}_{length}', seed=3007, lengths=(length,), heads=8, key_heads=2,
         correlated=True, beta=1., constant_log_gate=gate, state_scale=1.)
    for gate in (-1e-8, -1e-7, -1e-6) for length in (4097, 32768)
] + [
    dict(name=f'critical_nearly_correlated_{seed}_{amplitude}', seed=seed, lengths=(1025,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=-1e-6, amplitude=amplitude, state_scale=1.)
    for seed in (3011, 3019, 3023) for amplitude in (1., 32., 128.)
] + [
    dict(name=f'critical_gaussian_{rate}_{gate}', seed=3037, lengths=(4097,), heads=8, key_heads=2,
         beta=rate, constant_log_gate=gate, state_scale=1.)
    for rate in (0., .5, 1.) for gate in (0., -1e-6, -80.)
] + [
    dict(name=f'critical_group_{heads}_{keys}', seed=3041, lengths=(1025,), heads=heads, key_heads=keys,
         beta=1., constant_log_gate=-1e-6, state_scale=1.)
    for heads, keys in ((24, 8), (48, 16), (64, 16))
] + [
    dict(name=f'critical_positive_{seed}', seed=seed, lengths=(4097,), heads=8, key_heads=2,
         fla='positive', beta=1., constant_log_gate=-1e-6, state_scale=1.)
    for seed in (3049, 3061)
] + [
    dict(name='critical_random_beta_32k', seed=3067, lengths=(32768,), heads=8, key_heads=2,
         constant_log_gate=-1e-6, state_scale=1.),
    dict(name='critical_alternating_beta', seed=3079, lengths=(4097,), heads=8, key_heads=2,
         nearly_correlated=True, alternating_beta=True, constant_log_gate=-1e-7, state_scale=1.),
    dict(name='critical_packed', seed=3083, lengths=(0, 127, 128, 129, 0, 513), heads=24, key_heads=8,
         beta=1., constant_log_gate=-1e-6, state_scale=1.),
    dict(name='critical_zero_initial', seed=3089, lengths=(4097,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=-1e-6, initial=False),
    dict(name='critical_large_static_state', seed=3109, lengths=(1025,), heads=8, key_heads=2,
         beta=0., gate_scale=0., state_scale=128.),
]


# Matched controls added after the initial critical sweep to isolate failures.
CRITICAL_CASES += [
    dict(name='control_nearly_correlated_no_decay', seed=3011, lengths=(1025,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=0., state_scale=1.),
    dict(name='control_zero_initial_no_decay', seed=3089, lengths=(4097,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=0., initial=False),
    dict(name='control_coordinate_zero_initial', seed=3007, lengths=(32768,), heads=8, key_heads=2,
         correlated=True, beta=1., constant_log_gate=-1e-7, initial=False),
    dict(name='control_coordinate_beta_zero', seed=3007, lengths=(32768,), heads=8, key_heads=2,
         correlated=True, beta=0., constant_log_gate=-1e-7, state_scale=1.),
    dict(name='control_coordinate_no_decay', seed=3007, lengths=(32768,), heads=8, key_heads=2,
         correlated=True, beta=1., constant_log_gate=0., state_scale=1.),
]


# Interior rates distinguish endpoint behavior from the supported interval.
CRITICAL_CASES += [
    dict(name=f'interior_beta_{rate}_{kind}', seed=3163, lengths=(4097,), heads=8, key_heads=2,
         **{kind: True}, beta=rate, constant_log_gate=-1e-7, state_scale=1.)
    for rate in (.25, .5, .6, .9) for kind in ('correlated', 'nearly_correlated')
] + [
    dict(name='interior_random_beta_correlated', seed=3163, lengths=(4097,), heads=8, key_heads=2,
         correlated=True, constant_log_gate=-1e-7, state_scale=1.)
]


# Matched unit-scale sweep: correlation is varied without changing Q/V/h0.
# These are synthetic sensitivity measurements, not measured model inputs.
CORRELATION_CASES = [
    dict(name=f"correlation_rho{rho}_beta{rate}_g{gate}", seed=3827,
         lengths=(4096,), heads=8, key_heads=2, key_correlation=rho,
         beta=rate, constant_log_gate=gate, state_scale=.1)
    for rho in (0., .25, .5, .75, .95, 1.)
    for rate in (.5, 1.) for gate in (0., -.001, -.1, -1.)
] + [
    dict(name=f"correlation_long_rho{rho}", seed=3827, lengths=(32768,),
         heads=8, key_heads=2, key_correlation=rho, beta=1., constant_log_gate=0., state_scale=.1)
    for rho in (.95, 1.)
] + [
    dict(name=f"correlation_large_rho{rho}_g{gate}", seed=3827, lengths=(4096,),
         heads=8, key_heads=2, key_correlation=rho, beta=1., constant_log_gate=gate,
         amplitude=32., state_scale=.1)
    for rho in (0., .25, .5, .75, .95, 1.) for gate in (0., -.001)
] + [
    dict(name=f"correlation_initial_rho{rho}_g{gate}", seed=3827, lengths=(4096,),
         heads=8, key_heads=2, key_correlation=rho, beta=1., constant_log_gate=gate,
         state_scale=1.)
    for rho in (.75, .95, 1.) for gate in (0., -.001)
]


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
                                  initial=bool(start), observe_coupling=True, per_head=True))
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
    parser.add_argument("--capture", type=Path, help="Captured gdn_keys.pt; required with --suite capture")
    all_cases = AUDIT_CASES + CRITICAL_CASES + CORRELATION_CASES
    parser.add_argument("--cases", nargs="+",
                        help="Explicit subset for reproducing failures; default runs every case")
    args = parser.parse_args()
    if (args.suite == "capture") != (args.capture is not None):
        parser.error("--capture is required exactly when --suite capture is selected")
    captured = torch.load(args.capture, map_location="cpu", weights_only=True)["store"] if args.capture else None
    replay_cases = capture_cases(captured) if captured is not None else []
    cases = {"standard": AUDIT_CASES, "critical": CRITICAL_CASES,
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
                  ("tests/gdn_mega_validation.py", "tests/test_gdn_mega_validation.py", "tests/test_gdn_mega.py",
                   "benchmarking/audit_gdn_mega.py", "csrc/host/utils.h")},
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
        result = run_audit_case(case, args.device, prepared_inputs=prepared)
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
