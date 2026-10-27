"""Accuracy-plot reductions must preserve the ordinary acceptance checks."""
import importlib.util
from pathlib import Path

import pytest
import torch

from gdn_mega_validation import ATOL, RTOL, MAX_RMSE_RATIO, error_metrics


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


def test_regression_golden_cache_invalidates_and_rejects_corruption(monkeypatch, tmp_path):
    import gdn_mega_validation as validation
    monkeypatch.setenv('GDN_REFERENCE_CACHE', str(tmp_path))
    calls = []
    def compute(value, initial):
        calls.append(1)
        return value.double() + (0 if initial is None else initial), value.double().sum()
    monkeypatch.setattr(validation, 'recurrent_reference', compute)
    value = torch.arange(8, dtype=torch.float16)
    first = validation.cached_recurrent_reference(value, None)
    second = validation.cached_recurrent_reference(value.clone(), None)
    assert len(calls) == 1 and all(torch.equal(a, b) for a, b in zip(first, second))
    validation.cached_recurrent_reference(value + 1, None)
    validation.cached_recurrent_reference(value, torch.tensor(0.))
    assert len(calls) == 3
    for path in tmp_path.glob('*.pt'):
        saved = torch.load(path, weights_only=True)
        saved['tensors'][0][0] += 1
        torch.save(saved, path)
    with pytest.raises(RuntimeError, match='mismatch'):
        validation.cached_recurrent_reference(value, None)
    monkeypatch.delenv('GDN_REFERENCE_CACHE')
    validation.cached_recurrent_reference(value, None)
    assert len(calls) == 4


@pytest.mark.parametrize('baseline,current,additional,expected', [
    ('passed', 'passed', {}, (False, False)),
    ('failed', 'passed', {}, (True, False)),
    ('passed', 'failed', {}, (False, True)),
    ('passed', 'passed', {'baseline_accuracy_status': 'failed'}, (True, False)),
    ('failed', 'passed', {'baseline_accuracy_status': 'passed'}, (True, False)),
])
def test_numerical_asterisk_is_per_cell_and_implementation(monkeypatch, baseline, current, additional, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarking'))
    from gdn_mega_report_text import numerical_failure
    row = {'baseline': {'status': baseline}, 'current': {'status': current}, **additional}
    assert tuple(numerical_failure(row, v) for v in ('baseline', 'current')) == expected


def test_worst_tolerance_witness_is_distinct_from_maximum_absolute_error():
    expected = torch.tensor([0., .5, 1.], dtype=torch.float64)
    actual = torch.tensor([2*ATOL, .5004, 1.0009], dtype=torch.float64)
    m = error_metrics(actual, expected)
    assert not m['passed'] and m['pointwise_violations'] == 1
    assert m['max_error_flat_index'] == 2
    assert m['worst_tolerance_flat_index'] == 0
    assert m['actual_at_worst_tolerance'] == 2*ATOL
    assert m['reference_at_worst_tolerance'] == 0.
    assert m['abs_error_at_worst_tolerance'] == 2*ATOL
    assert m['allowed_at_worst_tolerance'] == ATOL
    assert m['max_tolerance_ratio'] == 2.


@pytest.mark.parametrize("reference_value", [0., 1., 100.])
def test_approved_elementwise_policy_rejects_a_single_regressing_element(reference_value):
    # Thousands of exact elements make the RMSE check pass even for the spike.
    # The elementwise bound must still distinguish the two sides of the limit.
    assert ATOL == 1e-4 and RTOL == 1e-3
    expected = torch.ones(4096, dtype=torch.float64)
    expected[0] = reference_value
    allowed = ATOL + RTOL * abs(reference_value)
    for factor, passed in ((.99, True), (1.01, False)):
        actual = expected.clone()
        actual[0] += factor * allowed
        result = error_metrics(actual, expected)
        assert result["relative_rmse"] < MAX_RMSE_RATIO
        assert result["passed"] is passed
        assert result["pointwise_violations"] == (0 if passed else 1)
