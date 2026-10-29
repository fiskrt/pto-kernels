# GDN mega kernel

One NPU launch, two phases: shared gate/Gram preparation, then a persistent
head-parallel inverse/residual/state/output pipeline. Every shape uses the same
schedule. C=D=128; inverse leaves are 16×16. There is no decay-dependent
truncation or sequential fallback. Authoritative state and sensitive matrix
operations remain FP32; projections use two FP16 components of state.

[Kernel design and caller allocations](../../csrc/kernel/gdn_mega/README.md) ·
[Speed heatmap](heatmap.md) · [Absolute-error heatmap](error_heatmap.md) ·
[Input ranges](input_ranges.md)

The caller ABI, output scaling and allocation sizes are unchanged. The kernel
is split across seven source files. Hardware validation is on Ascend 910B2
(24 cube/48 vector cores); other architectures are not certified by these runs.

## Files

| File | Purpose |
|---|---|
| `benchmarking/gdn_mega.py` | Fast build, quick/full tests, timing, matched grid and both heatmaps |
| `benchmarking/profile_gdn_mega.py` | msprof hardware/simulator collection and traceviz rendering |
| `tests/test_gdn_mega.py` | All 264 tests, 178 fixed numerical cases, independent oracle and allocation helpers |

There are no GDN-specific shell wrappers. Raw JSON, caches, profiler traces and
experimental builds belong in ignored `benchmark_data/` or `build/`. The docs
retain Markdown and two PNGs only. Historical exploration data is available in
Git history before this consolidation; it is not a runtime/test dependency.

## Environment and initial build

Use the existing prepared container:

```bash
docker exec -it gdn-mega-opt bash
# From the repository root inside the container:
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export NPU_DEVICE=npu:0
export OMP_NUM_THREADS=4
```

For a new checkout, a container can instead be started from the host repository
root as follows. The repository will be `/workspace` inside it:

```bash
docker run --name gdn-mega-opt -it --rm --privileged \
  --network=host --ipc=host --shm-size=16g \
  --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/queue_schedule:/var/queue_schedule \
  -v "$PWD":/workspace -w /workspace \
  quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.11 bash
```

The tested Python environment uses PyTorch/torch-npu 2.10.0. Initial setup and
binding compilation, if they have not already been done:

```bash
source /usr/local/Ascend/cann/set_env.sh
python -m pip install torch==2.10.0 torch-npu==2.10.0 pytest pybind11 cmake ninja matplotlib numpy
cmake -S . -B build -G Ninja -DGDN_MEGA_ONLY=ON -DSOC_VERSION=Ascend910B2 \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch, pybind11; print(torch.utils.cmake_prefix_path + ";" + pybind11.get_cmake_dir())')"
cmake --build build -j4
gdn_extension_suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ln -sf "../../pto_kernels_ops${gdn_extension_suffix}" \
  "python/pto_kernels/pto_kernels_ops${gdn_extension_suffix}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export NPU_DEVICE=npu:0
export OMP_NUM_THREADS=4
```

The PTO headers/toolchain must be installed as described by the repository's
main build documentation. These commands use the prepared CANN image's paths.

## Fast optimization loop

```bash
# Isolated build; all seven supported head specializations, about 10–11 seconds.
python benchmarking/gdn_mega.py build

# Nine fixed numerical cases; partial screen, not full acceptance.
python benchmarking/gdn_mega.py test --library build/gdn_candidate --quick

# Checked 256/4096-token timings on two representative head configurations.
python benchmarking/gdn_mega.py bench --library build/gdn_candidate \
  --head-configs 8/2 16/4 --lengths 256 4096

# Full acceptance before retaining an optimization.
python benchmarking/gdn_mega.py test --library build/gdn_candidate
```

`build --source PATH --output PATH` selects an isolated source snapshot/build.
`build --quick` compiles only H8/H16; that binary cannot run the full head suite.
Builds do not replace the installed kernel. After validation, install explicitly
with `cp build/gdn_candidate/libno_workspace_kernel.so build/lib/`.

Tests verify the loaded library hash. The full suite retains all 264 test IDs
and all 178 fixed case definitions from before consolidation. Its independent
FP64 checks, five poisoned-scratch launches, input/guard checks and tolerances
are unchanged. The new oracle-cache identity uses the complete oracle function
source, exact input bytes and PyTorch version, and hashes all loaded golden
bytes. Device outputs and pass results are never cached. The first run after
consolidation creates new cache entries; subsequent runs reuse them.

Run the test file directly to recompute references without the CLI's default
cache, or explicitly set a cache:

```bash
python -m pytest tests/test_gdn_mega.py -q
GDN_REFERENCE_CACHE=benchmark_data/gdn_mega/reference_cache \
  python -m pytest tests/test_gdn_mega.py -q
```

Acceptance is **every raw-output and final-state element** satisfying
`abs(error) <= 1e-4 + 1e-3*abs(reference)`, plus finiteness and relative RMSE < 0.005.
The golden is an independent CPU FP64 token recurrence on the rounded ABI
inputs. A separate FP32 chunked reference is retained as an additional check.
The tests cover tails, packed/empty sequences, initial/absent state, beta
endpoints/interior values, correlated keys, strong/tiny/zero gates and 32K
contexts. Tolerances are fixed, never fitted to a candidate.

## Timing and heatmaps

```bash
# Installed library by default. Initial state is supplied by default.
python benchmarking/gdn_mega.py bench --head-configs 48/16 --lengths 256 4096

# Full 63-cell comparison, including 24/8 and 48/16, through 32K.
python benchmarking/gdn_mega.py heatmap \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib

# Smaller matched comparison, or compare two experimental libraries.
python benchmarking/gdn_mega.py heatmap --library build/gdn_candidate \
  --baseline-lib build/lib --head-configs 8/2 16/4 48/16 --lengths 256 4096 \
  --report-dir benchmark_data/gdn_mega/comparison

# Re-render both PNGs and Markdown without another NPU run.
python benchmarking/gdn_mega.py render benchmark_data/gdn_mega/heatmap.json
```

The full grid compares against the original 10748eb library when that is supplied
as `--baseline-lib`. That historical binary is not shipped in the repository;
it must be built from the historical source. Any other compatible library can
be supplied, and both hashes are recorded. Rendering does not require an NPU.

For **each shape and each implementation**, inputs are generated/transferred
and all outputs, masks, intermediates and workspace are allocated once using
the same caller helper. Five calls warm the kernel, with FP64 output/state
checks after each. The benchmark then captures 20 calls in an NPU graph, performs
one untimed replay and synchronizes. It times 15 graph replays with NPU events,
divides each elapsed graph time by 20, and reports the median of those 15 averages
(300 timed calls). Output/state are checked again after the final replay.

Every call reuses the same allocations and input values. The original supplied
initial state is reused, not the previous call's final state. Outputs and scratch
are not externally cleared between calls; caller-required initialization happens
before warmup. Kernel-internal computation, transfers and synchronization are
timed. Allocation, input transfer, CPU oracle/checks, graph capture and the
caller's output scaling are excluded. Graph execution overhead remains, but
per-call Python dispatch is avoided. There is no cache flush, so this measures
warmed repeated-buffer device performance rather than end-to-end request latency.

Current and original use separate serial worker processes; their order alternates
by head pair. Both are freshly timed. Workers verify loaded-library hashes and
must agree on input/reference hashes, environment and policy. Shape, initial
state, gate scale and repetitions are CLI arguments; see `--help` and the
[heatmap's benchmark procedure](heatmap.md#benchmark-procedure-identical-for-both-libraries).

Both figures use the same checked runs. Error cells show the maximum over all
checked warmups and the post-timing check; raw output and final state are both
validated. Baseline failures still have timings and are starred. A failed
current cell makes the command fail. Asterisks never mark passing runtimes.
The ordinary-input grid does not replace the adversarial test suite.

Do not run NPU jobs concurrently. Parallel CPU compilation is fine. Large-grid
CPU checks, especially H64/32K, are much slower than the nine-case screen.

## Profiling

```bash
python benchmarking/profile_gdn_mega.py hardware --head-config 48/16 --length 4096 \
  --output benchmark_data/gdn_mega/profile_h48
python benchmarking/profile_gdn_mega.py simulator --head-config 16/4 --length 512 \
  --output benchmark_data/gdn_mega/profile_sim

# Use the OPPROF directory produced by the simulator above.
python benchmarking/profile_gdn_mega.py render \
  --input benchmark_data/gdn_mega/profile_sim/OPPROF_REPLACE_WITH_RUN \
  --traceviz ../traceviz-mcp --output benchmark_data/gdn_mega/profile_view
```

`--library` can select an isolated candidate. Hardware mode collects Roofline
counters; simulator mode collects PipeUtilization traces. Each launches the
actual kernel once and checks output/state against FP64. Rendering writes
`timeline.png` and `summary.txt`; `--units` and `--window` select cores/time ranges.
The sibling traceviz checkout is optional until rendering. With a fresh
`/workspace` container, mount/install that checkout separately or pass its path.

Use hardware timings for performance. Simulator traces explain scalar/control,
cube/vector and MTE overlap; their time scale and waits are not hardware FLOP
utilization. There is no generated instrumented kernel or committed profile report.
