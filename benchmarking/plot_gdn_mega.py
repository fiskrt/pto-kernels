"""Plot a GDN speedup heatmap from gdn_mega_sweep.py JSON; no NPU needed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
from gdn_mega_report_text import write_latency  # noqa: E402

from gdn_mega_sweep import head_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Filename stem; writes PNG and SVG beside input by default")
    parser.add_argument("--markdown", type=Path, help="Generated Markdown report; defaults beside the plots")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    heads, lengths = sorted(head_configs(data)), sorted(data["lengths"])
    cells = {((r["heads"], r["key_heads"]), r["length"]): r for r in data["records"]}
    if len(cells) != len(data["records"]) or set(cells) != {(h, n) for h in heads for n in lengths}:
        raise ValueError("Heatmap requires one checked result for every requested cell")
    for row in cells.values():
        if row.get("status") not in ("passed", "failed"):
            raise ValueError("Every comparison must have an explicit validation status")
        if row.get("status") == "failed":
            if row["speedup"] is not None or not any(row[v].get("status") == "failed" for v in ["baseline", "current"]):
                raise ValueError("Failed comparisons must identify a failed implementation and have no speedup")
        elif row["speedup"] is None or not np.isfinite(row["speedup"]) or row["speedup"] <= 0:
            raise ValueError("Passing speedups must be finite and positive")
        elif (any(row[v].get("status") == "failed" for v in ["baseline", "current"])
              or not np.isclose(row["speedup"], row["baseline_ms"] / row["current_ms"], rtol=1e-10)):
            raise ValueError("Passing comparisons must match their latencies and contain no failed implementation")
    values = np.array([[np.nan if cells[h, n].get("status") == "failed" else cells[h, n]["speedup"]
                        for n in lengths] for h in heads], dtype=float)
    valid = values[~np.isnan(values)]
    if not valid.size:
        valid = np.array([1.0])
    # A fixed 1x neutral point makes regressions visibly different from wins.
    norm = TwoSlopeNorm(vmin=min(0.5, float(valid.min())), vcenter=1.0,
                        vmax=max(1.5, float(valid.max())))
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9dde3")
    mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(lengths)), [f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n) for n in lengths])
    ax.set_yticks(range(len(heads)), [f"{h} / {hg}" for h, hg in heads])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Value heads H / key heads Hg")
    historical = data.get("baseline_source", {}).get("kind") == "historical_latency"
    title = ("GDN: historical original / current stable latency" if historical else
             "GDN mega kernel: original / current latency")
    ax.set_title(title, fontsize=16, pad=16)
    for i, h in enumerate(heads):
        for j, n in enumerate(lengths):
            value = values[i, j]
            if np.isnan(value):
                failed = [name for name in ["baseline", "current"] if cells[h, n][name].get("status") == "failed"]
                label = (failed[0] if len(failed) == 1 else "both") + " FAIL"
                color = "#374151"
            else:
                label = f"{value:.2f}×"
                color = "white" if norm(value) > 0.8 or norm(value) < 0.15 else "#17202a"
            ax.text(j, i - .12, label, ha="center", va="center", fontsize=10 if np.isnan(value) else 12,
                    weight="bold", color=color)
            current_ms = cells[h, n]["current_ms"]
            latency = f"{current_ms * 1000:,.1f} µs" if current_ms is not None else "no current timing"
            ax.text(j, i + .21, latency, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(np.arange(-0.5, len(lengths)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heads)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, label="Speedup (higher is faster)", shrink=0.8)
    ticks = sorted({norm.vmin, 1.0, norm.vmax,
                    *(v for v in range(2, int(np.ceil(norm.vmax))) if norm.vmax-v >= .25)})
    bar.set_ticks(ticks, labels=[f'{v:.2g}' for v in ticks])
    state = "nonzero" if data["initial_state"] else "zero"
    hardware = data.get("environment", {}).get("device_name", data["device"])
    acceptance = ("Pointwise + RMSE acceptance" if data.get("validation_sha256") else
                  "Historical RMSE-only acceptance; no pointwise accuracy guarantee")
    if historical:
        acceptance = "Current: strict pointwise + RMSE; original: historical RMSE-only (fails strict spot-checks)"
    current_commit = data.get("current_kernel_commit", data.get("source_commit", ""))[:7]
    baseline_commit = data.get("baseline_source", {}).get("source_commit", "original")[:7]
    fig.supxlabel(
        f"{hardware} · D = chunk size = 128 · explicit H/Hg rows · {state} initial state · gate scale {data['gate_scale']:g}\n"
        f"5 checked warmups · median of 15 × 20 graph calls · allocation excluded · gray = correctness failure\n"
        "Each cell: speedup above; current kernel median total µs per call below\n"
        f"{acceptance}\n"
        f"Original {baseline_commit} / current {current_commit} · 1× = equal latency; below 1× = current slower",
        fontsize=9,
    )
    output = args.output or args.input.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    svg = output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    write_latency(data, args.input, output, args.markdown or output.with_suffix('.md'))
    print(f"Saved {output.with_suffix('.png')} and {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
