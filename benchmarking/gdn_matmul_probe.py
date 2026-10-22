"""Run only with GDN_MATMUL_PROBE; checks isolated products, not GDN output."""
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_gdn_mega import _make_inputs, _prepare_gdn_mega
from pto_kernels import pto_gdn_mega

torch.set_num_threads(4)
device = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.set_device(device)
torch.npu.config.allow_internal_format = False
inputs = _make_inputs(128, 16, 4)
args = _prepare_gdn_mega(*(x.to(device) for x in inputs),
    torch.tensor([0, 128], dtype=torch.int32, device=device))
samples = []
for repeat in range(5):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    pto_gdn_mega(**args)
    end.record()
    end.synchronize()
    result = args["wy_workspace_a1"].cpu()
    torch.testing.assert_close(result, torch.eye(128, dtype=result.dtype).expand_as(result), rtol=0, atol=0)
    counters = args["kkt_workspace"].view(torch.int64).flatten()[:24 * 16].reshape(24, 16)[:, :6].cpu()
    samples.append(counters)
    print(json.dumps({"repeat": repeat, "elapsed_us": start.elapsed_time(end) * 1000,
                      "median_ticks": counters.median(0).values.tolist()}), flush=True)
ticks = torch.stack(samples[1:]).flatten(0, 1).median(0).values.tolist()
for offset, size in [(0, 64), (3, 128)]:
    resident, reload, feedback = ticks[offset:offset + 3]
    print(json.dumps({"size": size, "matmuls": 512, "resident_ticks": resident,
        "reload_ticks": reload, "feedback_ticks": feedback,
        "reload_over_resident": reload / resident, "feedback_over_resident": feedback / resident}), flush=True)
