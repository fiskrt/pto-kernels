"""A benchmark cache must never hide changed outputs or a changed oracle."""
import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "gdn_benchmark_worker", Path(__file__).resolve().parents[1] / "benchmarking/gdn_mega.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def inputs():
    launch = {"out": torch.zeros(1, 2, 1, 128, dtype=torch.float16),
              "final_state": torch.zeros(1, 1, 128, 128, dtype=torch.float16)}
    return launch, torch.zeros_like(launch["out"]), torch.zeros_like(launch["final_state"])


def test_identical_outputs_reuse_only_the_validated_statistics(monkeypatch):
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)

    def unexpected_recalculation(*args):
        raise AssertionError("Byte-identical tensors do not need repeated percentiles")

    monkeypatch.setattr(benchmark, "error_metrics", unexpected_recalculation)
    assert benchmark.check_outputs(launch, expected, state, previous) is previous


@pytest.mark.parametrize("field", ["out", "final_state"])
@pytest.mark.parametrize("bad", [0.01, float("nan")])
def test_changed_element_is_rechecked_even_after_a_cached_pass(field, bad):
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)
    launch[field].flatten()[-1] = bad
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected, state, previous)


def test_a_different_reference_invalidates_cached_metrics():
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)
    expected = expected.clone()
    expected.flatten()[-1] = 0.01
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected, state, previous)
