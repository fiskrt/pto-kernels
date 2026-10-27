#!/usr/bin/env bash
# Measure and render the maintained GDN reports; NPU jobs run serially.
set -euo pipefail
cd "$(dirname "$0")/.."
report_dir=docs/gdn_mega
mkdir -p "$report_dir/data" "$report_dir/figures"
run() { bash scripts/gdn_mega_docker.sh "$@"; }
latency_plot() {
    run python benchmarking/plot_gdn_mega.py "$report_dir/data/heatmap.json" \
        --output "$report_dir/figures/heatmap" --markdown "$report_dir/heatmap.md"
}
error_plot() {
    run python benchmarking/plot_gdn_mega_errors.py "$report_dir/data/error_heatmap.json" \
        --output "$report_dir/figures/error_heatmap" --markdown "$report_dir/error_heatmap.md"
}
latency() {
    local status=0
    run python benchmarking/gdn_mega_sweep.py \
        --baseline-lib benchmark_data/gdn_mega/baseline_lib --baseline-commit 10748eb --current-lib build/lib \
        --output "$report_dir/data/heatmap.json" || status=$?
    # Numerical failures remain explicit even when their runtimes are measured.
    # The plot rejects incomplete data; current failures still fail this command.
    if (( status > 1 )); then return "$status"; fi
    latency_plot
    python - "$report_dir/data/heatmap.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))['records']
raise SystemExit(any(r['current']['status'] != 'passed' for r in rows))
PY
}
errors() {
    run python benchmarking/gdn_mega_errors.py --output "$report_dir/data/error_heatmap.json"
    error_plot
    python - "$report_dir/data/error_heatmap.json" <<'PY'
import json, sys
rows = [row for path in sys.argv[1:] for row in json.load(open(path))['records']]
raise SystemExit(any(r['current']['status'] != 'passed' for r in rows))
PY
}
case "${1:-all}" in
    latency) latency ;;
    errors) errors ;;
    all) errors; latency ;;
    render) latency_plot; error_plot ;;
    *) echo "Usage: $0 [all|latency|errors|render]" >&2; exit 2 ;;
esac
