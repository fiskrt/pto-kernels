#!/usr/bin/env bash
# Run inside the persistent CANN container via gdn_mega_docker.sh.
set -eo pipefail
cd "$(dirname "$0")/.."
source /usr/local/Ascend/cann/set_env.sh
exec python benchmarking/gdn_mega_iteration.py "$@"
