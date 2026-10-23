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
from test_gdn_mega_validation import AUDIT_CASES, run_audit_case  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--cases", nargs="+", choices=[c["name"] for c in AUDIT_CASES],
                        help="Explicit subset for reproducing failures; default runs every case")
    args = parser.parse_args()
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
              "requested_cases": args.cases or [c["name"] for c in AUDIT_CASES],
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                  ("tests/gdn_mega_validation.py", "tests/test_gdn_mega_validation.py", "tests/test_gdn_mega.py",
                   "benchmarking/audit_gdn_mega.py", "csrc/host/utils.h")},
              "host_extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
              "policy": POLICY, "oracle": "CPU FP64 token recurrence, raw output",
              "kernel_library_sha256": hashlib.sha256(Path(libraries.pop()).read_bytes()).hexdigest(), "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in AUDIT_CASES:
        if args.cases and case["name"] not in args.cases:
            continue
        result = run_audit_case(case, args.device)
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
