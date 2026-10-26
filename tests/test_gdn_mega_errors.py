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
    full = error_metrics(actual, expected, include_percentiles=True)
    compact = error_metrics(actual, expected, include_percentiles=False)
    assert compact == {k: v for k, v in full.items() if k not in ("p50_abs", "p99_abs", "p999_abs")}
    if case in ("spike", "nan", "inf"):
        assert not compact["passed"]


def test_repeat_summary_keeps_worst_error_and_nonfinite_failure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarking"))
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


@pytest.mark.parametrize('arguments,expected', [
    ([], [(8, 2), (12, 3), (16, 4), (24, 6), (24, 8), (32, 8), (48, 12), (48, 16), (64, 16)]),
    (['--head-configs', '24/6', '24/8', '48/16'], [(24, 6), (24, 8), (48, 16)]),
    (['--heads', '24', '48', '--group-size', '3'], [(24, 8), (48, 16)]),
])
def test_explicit_head_pairs_keep_distinct_gqa_rows(monkeypatch, arguments, expected):
    import argparse
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarking'))
    from gdn_mega_sweep import add_shape_arguments, resolve_head_configs, head_configs
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args(arguments)
    assert resolve_head_configs(args, parser) == expected
    assert head_configs(vars(args)) == expected


@pytest.mark.parametrize('pairs', [['24/0'], ['24/7'], ['24/8', '24/8'], ['24'], ['24/8/1']])
def test_invalid_head_pairs_are_rejected(monkeypatch, pairs):
    import argparse
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarking'))
    from gdn_mega_sweep import add_shape_arguments, resolve_head_configs
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    with pytest.raises(SystemExit):
        resolve_head_configs(parser.parse_args(['--head-configs', *pairs]), parser)


def test_shared_golden_requires_matching_inputs_source_and_contents(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarking'))
    from gdn_mega_errors import shared_reference
    path = tmp_path / 'golden.pt'
    identity = {'inputs': 'input-hash', 'oracle_sha256': 'oracle-hash'}
    expected = (torch.arange(17, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
    calls = []
    def compute():
        calls.append(1)
        return expected
    first = shared_reference(path, identity, compute)
    second = shared_reference(path, identity, compute)
    assert len(calls) == 1
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    for field in identity:
        with pytest.raises(RuntimeError, match='mismatch'):
            shared_reference(path, {**identity, field: 'changed'}, compute)
    saved = torch.load(path, weights_only=True)
    saved['tensors'][0][-1] += 1
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match='mismatch'):
        shared_reference(path, identity, compute)
