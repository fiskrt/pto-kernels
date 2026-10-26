"""Accuracy-plot reductions must preserve the ordinary acceptance checks."""
import importlib.util
from pathlib import Path

import pytest
import torch

from gdn_mega_validation import error_metrics


@pytest.mark.parametrize("case", ["rounded", "spike", "nan", "inf", "empty"])
def test_omitting_percentiles_preserves_every_acceptance_metric(case):
    torch.manual_seed(123)
    expected = torch.randn(3, 17) * 32
    actual = expected.half()
    if case == "spike":
        expected[0, 0], actual[0, 0] = 0, .01
    elif case in ("nan", "inf"):
        actual[0, 0] = float(case)
    elif case == "empty":
        expected, actual = expected[:0], actual[:0]
    full = error_metrics(actual, expected)
    compact = error_metrics(actual, expected, include_percentiles=False)
    assert compact == {k: v for k, v in full.items() if k not in ("p50_abs", "p99_abs", "p999_abs")}
    if case in ("spike", "nan", "inf"):
        assert not compact["passed"]


def test_repeat_summary_keeps_worst_error_and_nonfinite_failure():
    spec = importlib.util.spec_from_file_location(
        "gdn_error_worker", Path(__file__).resolve().parents[1] / "benchmarking/gdn_mega_errors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = torch.zeros(2, 4)
    repeats = [{"raw_output": error_metrics(torch.full_like(expected, value), expected, include_percentiles=False)}
               for value in (0., .01, .001)]
    result = module.summarize(repeats, "raw_output")
    assert result["max_abs"] == repeats[1]["raw_output"]["max_abs"]
    assert result["pointwise_violations_max"] == 8
    assert not result["passed"]
    repeats.append({"raw_output": error_metrics(torch.full_like(expected, float("nan")), expected)})
    result = module.summarize(repeats, "raw_output")
    assert not result["finite"] and result["max_abs"] is None
    assert result["actual_nonfinite_max"] == 8
