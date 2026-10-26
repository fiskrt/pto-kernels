"""Compare two compiled GDN libraries over sequence length and head count.

Runs gdn_mega.py in isolated processes so each implementation is actually loaded.
Every cell checks output and final state; allocation/reference time is excluded.
"""
import argparse
import ast
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
SUPPORTED_HEADS = [8, 12, 16, 24, 32, 48, 64]


def input_generator_hash():
    tree = ast.parse((ROOT / "tests/test_gdn_mega.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_make_inputs")
    if ast.get_docstring(node):
        node.body = node.body[1:]
    # Python 3.12 adds an empty type_params AST field to ordinary functions.
    for child in ast.walk(node):
        if hasattr(child, "type_params") and not child.type_params:
            del child.type_params
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def save_report(report, output):
    for row in report["records"]:
        if row.get("status") not in ("passed", "failed"):
            raise ValueError("Every comparison must have an explicit validation status")
    output.write_text(json.dumps(report, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as f:
        fields = ["heads", "key_heads", "length", "status", "baseline_ms", "current_ms", "speedup"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["records"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-lib", type=Path, help="Directory containing original libno_workspace_kernel.so")
    baseline.add_argument("--baseline-report", type=Path,
                          help="Frozen historical-baseline timings; preserves their original validation limitations")
    parser.add_argument("--current-lib", type=Path, default=ROOT / "build/lib")
    parser.add_argument("--heads", type=int, nargs="+", default=SUPPORTED_HEADS)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--group-size", type=int, default=4, help="Value heads per key head; Hg = H / group_size")
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--initial-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_data/gdn_mega/sweep.json")
    parser.add_argument("--resume", action="store_true", help="Reuse completed head rows with identical settings and library hashes")
    args = parser.parse_args()
    if (args.group_size <= 0 or any(h not in SUPPORTED_HEADS or h % args.group_size for h in args.heads)
            or any(n <= 0 for n in args.lengths) or len(set(args.heads)) != len(args.heads)
            or len(set(args.lengths)) != len(args.lengths)):
        parser.error("Use distinct supported heads divisible by group-size and distinct positive lengths")
    libraries = {"current": args.current_lib.resolve()}
    if args.baseline_lib:
        libraries["baseline"] = args.baseline_lib.resolve()
    fingerprints = {name: hashlib.sha256((path / "libno_workspace_kernel.so").read_bytes()).hexdigest()
                    for name, path in libraries.items()}
    historical = json.loads(args.baseline_report.read_text()) if args.baseline_report else None
    if historical:
        if historical.get("kind") != "historical_baseline":
            parser.error("Expected a frozen historical_baseline report, not an old optimized sweep")
        fingerprints["baseline"] = historical["library_sha256"]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "heads": args.heads, "lengths": args.lengths, "group_size": args.group_size,
        "gate_scale": args.gate_scale, "initial_state": args.initial_state, "device": args.device,
        "chunk_size": 128, "head_dim": 128, "seed": 0, "warmups": 5, "trials": 15, "calls_per_trial": 20,
        "timing": "Median device-event time per graph-replayed call, warm cache; allocation and checks excluded",
        "library_sha256": fingerprints, "records": [],
        # Never reuse historical RMSE-only passes after changing acceptance or
        # its oracle. Include both the shared checker and the worker/reference.
        "validation_sha256": hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in
            ("tests/gdn_mega_validation.py", "tests/test_gdn_mega.py", "benchmarking/gdn_mega.py"))).hexdigest(),
        "validation": "pointwise atol=1e-5, rtol=1e-3 AND relative RMSE < 0.005; raw output and final state",
        "current_kernel_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "log", "-1", "--format=%H", "--", "csrc/kernel/gdn_mega"],
            cwd=ROOT, text=True).strip(),
    }
    historical_cells = {}
    if historical:
        for key in ["group_size", "gate_scale", "initial_state", "device", "chunk_size", "head_dim",
                    "seed", "warmups", "trials", "calls_per_trial"]:
            if historical.get(key) != report[key]:
                parser.error(f"Historical baseline mismatch: {key}")
        if historical.get("input_generator_ast_sha256") != input_generator_hash():
            parser.error("Historical baseline input generator differs")
        historical_cells = {(row["heads"], row["length"]): row for row in historical["records"]}
        if len(historical_cells) != len(historical["records"]):
            parser.error("Historical baseline has duplicate cells")
        for h in args.heads:
            for n in args.lengths:
                row = historical_cells.get((h, n))
                if row is None or row.get("key_heads") != h // args.group_size:
                    parser.error(f"Historical baseline missing or mismatched cell: H={h}, T={n}")
                if row.get("kernel_library_sha256") != fingerprints["baseline"]:
                    parser.error("Historical baseline contains a different library hash")
                if row.get("status") not in ("passed", "failed") or (row["status"] == "passed" and not row.get("median_us", 0) > 0):
                    parser.error("Historical baseline has an invalid validation/timing record")
        report["baseline_source"] = {
            "kind": "historical_latency", "file": str(args.baseline_report),
            "file_sha256": hashlib.sha256(args.baseline_report.read_bytes()).hexdigest(),
            "source_commit": historical["source_commit"], "created_utc": historical["created_utc"],
            "validation": historical["validation"],
        }
        report["validation"] = "Current: " + report["validation"] + "; baseline: historical RMSE-only, not current acceptance"
    if args.resume and output.exists():
        previous = json.loads(output.read_text())
        for key in ["heads", "lengths", "group_size", "gate_scale", "initial_state", "device",
                    "chunk_size", "head_dim", "seed", "warmups", "trials", "calls_per_trial", "library_sha256",
                    "validation_sha256", "validation", "baseline_source", "current_kernel_commit"]:
            if previous.get(key) != report[key]:
                parser.error(f"Cannot resume: {key} differs")
        report = previous
    save_report(report, output)
    with tempfile.TemporaryDirectory(prefix="gdn_mega_sweep_") as temporary:
        runs_dir = Path(temporary)
        for index, heads in enumerate(args.heads):
            if {row["length"] for row in report["records"] if row["heads"] == heads} == set(args.lengths):
                continue
            runs = ({"baseline": {n: historical_cells[heads, n] for n in args.lengths}}
                    if historical else {})
            # Alternate order across head counts to avoid always measuring one variant first.
            variants = ["current"] if historical else (["baseline", "current"] if index % 2 == 0 else ["current", "baseline"])
            for variant in variants:
                path = runs_dir / f"{variant}_h{heads}.json"
                command = [sys.executable, str(ROOT / "benchmarking/gdn_mega.py"),
                           "--heads", str(heads), "--key-heads", str(heads // args.group_size),
                           "--lengths", *map(str, args.lengths), "--gate-scale", str(args.gate_scale),
                           "--device", args.device, "--output", str(path)]
                if args.initial_state:
                    command.append("--initial-state")
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = str(libraries[variant]) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
                print(f"\n{variant}: H={heads}, Hg={heads // args.group_size}", flush=True)
                process = subprocess.run(command, cwd=ROOT, env=env, timeout=900)
                if process.returncode not in (0, 1) or not path.exists():
                    raise RuntimeError(f"Worker failed before completing: {variant}, H={heads}")
                runs[variant] = {row["length"]: row for row in json.loads(path.read_text())}
                report["environment"] = next(iter(runs[variant].values()))["environment"]
                if historical and report["environment"] != historical["environment"]:
                    raise RuntimeError("Current hardware/software environment differs from historical baseline")
                if set(runs[variant]) != set(args.lengths):
                    raise RuntimeError(f"Incomplete run: {path}")
                failures = any(row.get("status") == "failed" for row in runs[variant].values())
                if process.returncode != int(failures):
                    raise RuntimeError(f"Unexpected worker failure: {variant}, H={heads}")
                if any(row["kernel_library_sha256"] != fingerprints[variant] for row in runs[variant].values()):
                    raise RuntimeError(f"Dynamic loader selected the wrong library: {variant}")
            for length in args.lengths:
                before, after = runs["baseline"][length], runs["current"][length]
                passed = all(row.get("status") == "passed" for row in (before, after))
                report["records"].append({
                    "heads": heads, "key_heads": heads // args.group_size, "length": length,
                    "status": "passed" if passed else "failed",
                    "baseline_ms": before["median_us"] / 1000 if "median_us" in before else None,
                    "current_ms": after["median_us"] / 1000 if "median_us" in after else None,
                    "speedup": before["median_us"] / after["median_us"] if passed else None,
                    "baseline": before, "current": after,
                })
            save_report(report, output)
    save_report(report, output)
    print(f"\nSaved {len(report['records'])} comparison results to {output}", flush=True)
    raise SystemExit(any(row["status"] == "failed" for row in report["records"]))


if __name__ == "__main__":
    main()
