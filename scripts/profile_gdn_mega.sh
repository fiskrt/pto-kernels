#!/usr/bin/env bash
# Profile serially; keep all instrumented sources and raw traces out of production.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-collect}
profile_root=${2:-benchmark_data/gdn_mega/profile_$(date -u +%Y%m%d_%H%M%S)}
run() { bash scripts/gdn_mega_docker.sh "$@"; }
if [[ $mode == render ]]; then
    if [[ $# -lt 2 ]]; then
        profile_root=$(python -c 'import json; print(json.load(open("docs/gdn_mega/data/profile.json"))["raw_directory"])')
    fi
    run python benchmarking/profile_gdn_mega.py render "$profile_root"
    exit
fi
if [[ $mode != collect ]]; then echo "Usage: $0 [collect|render] [raw_directory]" >&2; exit 2; fi
mkdir -p "$profile_root"
run bash scripts/build_gdn_mega.sh
for shape in 16:4:4096 24:8:4096 48:16:4096 16:4:32768; do
    IFS=: read -r profile_h profile_hg profile_t <<< "$shape"
    run bash -c 'source /usr/local/Ascend/cann/set_env.sh
        msprof op --kernel-name="*gdn_mega_specialized*" --aic-metrics=Roofline --output="$1" \
            python benchmarking/gdn_mega.py --heads "$2" --key-heads "$3" \
            --initial-state --profile-once --lengths "$4"' \
        _ "$profile_root/device_h${profile_h}_t${profile_t}" "$profile_h" "$profile_hg" "$profile_t" \
        > "$profile_root/device_h${profile_h}_t${profile_t}.log" 2>&1
done
run bash -c 'source /usr/local/Ascend/cann/set_env.sh
    export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B2/lib:$LD_LIBRARY_PATH"
    msprof op simulator --aic-metrics=PipeUtilization --kernel-name="*gdn_mega_specialized*" --output="$1" \
        python benchmarking/gdn_mega.py --heads 16 --key-heads 4 --initial-state --profile-once --lengths 512' \
    _ "$profile_root/simulator_h16_t512" > "$profile_root/simulator_h16_t512.log" 2>&1
run python benchmarking/profile_gdn_mega.py prepare-diagnostic "$profile_root/diagnostic"
run bash -c 'set -e; source /usr/local/Ascend/cann/set_env.sh
    bisheng -DMEMORY_BASE -Dno_workspace_kernel_EXPORTS -I/usr/local/Ascend/driver/kernel/inc \
        -O0 -g -fPIC -D_FORTIFY_SOURCE=2 -O2 -std=c++17 -Wno-macro-redefined \
        -Wno-ignored-attributes -fstack-protector-strong --npu-arch=dav-2201 \
        -c --asc-aicore-lang "$1/kernel_gdn_mega.cpp" -o "$1/kernel.o"
    bisheng -O0 -g -shared -s -Wl,-z,relro -Wl,-z,now -Wl,-soname,libno_workspace_kernel.so \
        -o "$1/libno_workspace_kernel.so" "$1/kernel.o"
    export LD_LIBRARY_PATH="$(realpath "$1"):$LD_LIBRARY_PATH"
    python benchmarking/profile_gdn_mega.py measure-diagnostic "$2"' \
    _ "$profile_root/diagnostic" "$profile_root/diagnostic.json" > "$profile_root/diagnostic.log" 2>&1
run bash -c 'source /usr/local/Ascend/cann/set_env.sh
    export LD_LIBRARY_PATH="$(realpath "$1"):$LD_LIBRARY_PATH"
    msprof op --kernel-name="*gdn_mega_specialized*" --aic-metrics=Roofline --output="$2" \
        python benchmarking/gdn_mega.py --heads 16 --key-heads 4 --initial-state --profile-once --lengths 4096' \
    _ "$profile_root/diagnostic" "$profile_root/device_diagnostic" > "$profile_root/device_diagnostic.log" 2>&1
run chmod -R a+rX "$profile_root"
run python benchmarking/profile_gdn_mega.py render "$profile_root"
