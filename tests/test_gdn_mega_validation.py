"""Adversarial correctness checks independent of the chunked GDN algorithm."""
import pytest
import torch

from pto_kernels import pto_gdn_mega
from gdn_mega_validation import POLICY, assert_numerically_close, error_metrics, recurrent_reference
from test_gdn_mega import D, _expand_kv_heads, _make_inputs, _prepare_gdn_mega, _reference


# Fixed before evaluating the kernel: no tuning the oracle or tolerances to results.
AUDIT_CASES = [
    dict(name="ordinary_seed2027", seed=2027, lengths=(257,), heads=16, key_heads=4),
    dict(name="weak_seed119", seed=119, lengths=(513,), heads=16, key_heads=4, gate_scale=.01),
    dict(name="no_decay", seed=911, lengths=(257,), heads=8, key_heads=8, gate_scale=0.),
    dict(name="zero_state_tail", seed=1709, lengths=(129,), heads=12, key_heads=3, initial=False),
    dict(name="packed_empty", seed=31, lengths=(0, 1, 127, 0, 129, 0), heads=16, key_heads=4, gate_scale=.01),
    dict(name="packed_empty_zero_state", seed=313, lengths=(0, 1, 129, 0), heads=8, key_heads=2, initial=False),
    dict(name="large_values_and_state", seed=191, lengths=(257,), heads=8, key_heads=2, amplitude=32.),
    dict(name="tiny_values_and_state", seed=509, lengths=(257,), heads=8, key_heads=2, amplitude=1e-3),
    dict(name="zero_beta_no_decay", seed=809, lengths=(257,), heads=8, key_heads=2, gate_scale=0., beta=0.),
    dict(name="zero_query", seed=811, lengths=(257,), heads=8, key_heads=2, zero_query=True),
    dict(name="zero_values_and_state", seed=821, lengths=(257,), heads=8, key_heads=2, amplitude=0.),
    dict(name="correlated_beta002", seed=823, lengths=(256,), heads=8, key_heads=2, correlated=True, gate_scale=0., beta=.02),
    dict(name="correlated_beta05", seed=827, lengths=(256,), heads=8, key_heads=2, correlated=True, gate_scale=0., beta=.5),
    dict(name="correlated_beta1", seed=829, lengths=(256,), heads=8, key_heads=2, correlated=True, gate_scale=0., beta=1.),
    dict(name="mixed_decay", seed=839, lengths=(513,), heads=8, key_heads=2, mixed=True),
    dict(name="equal_heads64", seed=853, lengths=(129,), heads=64, key_heads=64),
    dict(name="single_key_head", seed=857, lengths=(129,), heads=24, key_heads=1),
    dict(name="keys_norm2_stable_beta", seed=859, lengths=(257,), heads=8, key_heads=2, key_scale=2., beta=.1),
    dict(name="long_8k_weak", seed=863, lengths=(8192,), heads=8, key_heads=2, gate_scale=.01),
    dict(name="long_32k", seed=877, lengths=(32768,), heads=8, key_heads=2),
    dict(name="correlated_random_beta09", seed=883, lengths=(513,), heads=8, key_heads=2,
         correlated_random=True, gate_scale=0., beta=.9, amplitude=32.),
    dict(name="nearly_correlated_beta1", seed=887, lengths=(257,), heads=16, key_heads=4,
         nearly_correlated=True, gate_scale=0., beta=1., amplitude=32.),
    dict(name="positive_log_gates", seed=907, lengths=(257,), heads=8, key_heads=2, gate_scale=-.01),
    dict(name="long_16k_no_decay", seed=919, lengths=(16384,), heads=8, key_heads=2, gate_scale=0.),
    dict(name="delayed_nearly_correlated", seed=929, lengths=(385,), heads=8, key_heads=2,
         nearly_correlated=True, correlated_after=128, gate_scale=0., beta=1., amplitude=32.),
    dict(name="strong_correlated_decay", seed=937, lengths=(257,), heads=8, key_heads=2,
         nearly_correlated=True, gate_scale=.01, beta=.9, amplitude=32.),
    # Distribution families from FLA test_gdn.py at 12cb9d4, adapted to our
    # FP16 input/state ABI and D128. Keep our independent FP64 oracle/policy.
    dict(name="fla_positive_no_decay", seed=42, lengths=(1024,), heads=8, key_heads=2,
         fla="positive", gate_mask=1., initial=False),
    dict(name="fla_positive_weak", seed=43, lengths=(1500,), heads=8, key_heads=2,
         fla="positive", gate_normalizer=10., initial=False),
    dict(name="fla_positive_strong", seed=44, lengths=(257,), heads=8, key_heads=2,
         fla="positive", gate_normalizer=.01, initial=False),
    dict(name="fla_positive_masked", seed=45, lengths=(513,), heads=8, key_heads=2,
         fla="positive", gate_mask=.5, initial=False),
    dict(name="fla_gaussian_initial", seed=46, lengths=(1024,), heads=8, key_heads=2,
         fla="gaussian", gate_normalizer=10., state_scale=1.),
    dict(name="fla_negative_eigenvalues", seed=47, lengths=(513,), heads=8, key_heads=2,
         fla="positive", beta_logits=True, negative_eigenvalues=True, initial=False),
    dict(name="fla_gate_activation", seed=48, lengths=(513,), heads=8, key_heads=2,
         fla="positive", gate_activation=True, initial=False),
    dict(name="fla_gaussian_logits", seed=49, lengths=(513,), heads=8, key_heads=2,
         fla="gaussian", beta_logits=True, gate_logits=True, state_scale=1.),
    dict(name="fla_positive_strong_tail75", seed=51, lengths=(75,), heads=8, key_heads=2,
         fla="positive", gate_normalizer=.01, initial=False),
    dict(name="strong_decay_tail92", seed=53, lengths=(92,), heads=8, key_heads=2, gate_scale=4.),
]
READ_ONLY = ("q", "k", "v", "g", "beta", "mask_lower", "mask_full", "minus_identity", "cu_seqlens")


def make_audit_inputs(case):
    inputs = list(_make_inputs(sum(case["lengths"]), case["heads"], case["key_heads"], seed=case["seed"]))
    q, k, v, g, beta = inputs
    if "fla" in case:
        if case["fla"] == "positive":
            q.copy_(torch.nn.functional.normalize(torch.rand_like(q), dim=-1))
            k.copy_(torch.nn.functional.normalize(torch.rand_like(k), dim=-1))
            v.copy_(torch.rand_like(v))
        beta_source = torch.randn_like(g) if case.get("beta_logits") else torch.rand_like(g)
        beta.copy_((beta_source.sigmoid() * (2 if case.get("negative_eigenvalues") else 1)).half())
        gate_source = torch.randn_like(g) if case.get("gate_logits") else torch.rand_like(g)
        g.copy_(torch.nn.functional.logsigmoid(gate_source) / case.get("gate_normalizer", 1.))
        if "gate_mask" in case:
            g.mul_(torch.rand_like(g) > case["gate_mask"])
        if case.get("gate_activation"):
            a_log = torch.randn(case["heads"])
            bias = torch.randn(case["heads"])
            g.copy_(-a_log.exp() * torch.nn.functional.softplus(torch.randn_like(g) + bias))
    g.mul_(case.get("gate_scale", 1.))
    amplitude = case.get("amplitude", 1.)
    v.mul_(amplitude)
    if case.get("correlated"):
        # Exact unit vectors: stable delta-rule recurrence even with beta=1.
        k.zero_()
        k[..., 0] = 1
    if case.get("correlated_random") or case.get("nearly_correlated"):
        keys = k[:, :1].float().expand_as(k).clone()
        if case.get("nearly_correlated"):
            keys += .01 * torch.randn_like(keys)
            keys /= keys.norm(dim=-1, keepdim=True)
        after = case.get("correlated_after", 0)
        k[:, after:].copy_(keys[:, after:].half())
    k.mul_(case.get("key_scale", 1.))
    if "beta" in case:
        beta.fill_(case["beta"])
    if case.get("zero_query"):
        q.zero_()
    if case.get("mixed"):
        for chunk, start in enumerate(range(0, q.shape[1], 128)):
            g[:, start:start + 128].mul_([4., .01, 0., 1.][chunk % 4])
    initial = None
    if case.get("initial", True):
        initial = torch.randn(len(case["lengths"]), case["heads"], D, D, dtype=torch.float16) * (case.get("state_scale", .1) * amplitude)
    cu = torch.tensor([0, *torch.tensor(case["lengths"]).cumsum(0).tolist()], dtype=torch.int32)
    return inputs, initial, cu


def run_audit_case(case, device):
    inputs, initial, cu = make_audit_inputs(case)
    # Reference stays on the CPU and is never passed to the device/launch helper.
    expected, expected_state = recurrent_reference(*inputs, initial, cu)
    launch = _prepare_gdn_mega(*(x.to(device) for x in inputs), cu.to(device),
                               initial.to(device) if initial is not None else None)
    snapshots = {name: launch[name].clone() for name in READ_ONLY}
    if initial is not None:
        snapshots["initial_state"] = launch["initial_state"].clone()
    guards = []
    for name in ["out", "final_state"]:
        old = launch[name]
        backing = torch.full((old.numel() + 128,), -123.25, dtype=old.dtype, device=device)
        launch[name] = backing[64:-64].view_as(old)
        guards.append(backing)
    if initial is None:
        launch["initial_state"] = launch["final_state"]  # Deliberately poisoned, must be ignored.
    metrics, first = [], None
    for fill in [0., float("nan"), 17., -9., float("nan")]:
        for name, tensor in launch.items():
            if isinstance(tensor, torch.Tensor) and name not in snapshots and name != "initial_state":
                tensor.fill_(fill)
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        actual = launch["out"].cpu().clone()
        state = launch["final_state"].cpu().view_as(expected_state).clone()
        integrity = all(torch.equal(launch[name], before) for name, before in snapshots.items())
        integrity &= all(bool((x[:64] == -123.25).all() and (x[-64:] == -123.25).all()) for x in guards)
        if first is None:
            first = (actual.clone(), state.clone())
        # Byte comparison remains meaningful even if numerical checks expose NaNs.
        independent = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip((actual, state), first))
        # Reuse statistics only after checking every output/state bit. All five
        # device launches and all integrity checks still execute independently.
        numerical = ({key: metrics[0][key] for key in ("raw_output", "scaled_output", "final_state")}
                     if metrics and independent else
                     {"raw_output": error_metrics(actual, expected),
                      "scaled_output": error_metrics(actual.double() * D**-.5, expected * D**-.5),
                      "final_state": error_metrics(state, expected_state)})
        metrics.append({**numerical,
                        "inputs_and_guards_intact": integrity, "scratch_independent": independent})
    return {"case": case, "policy": POLICY, "repeats": metrics,
            "fp16_output_rounding_floor": error_metrics(expected.half(), expected),
            "fp16_state_rounding_floor": error_metrics(expected_state.half(), expected_state),
            "passed": all(m["raw_output"]["passed"] and m["final_state"]["passed"] and
                          m["inputs_and_guards_intact"] and m["scratch_independent"] for m in metrics)}


@pytest.mark.parametrize("case", AUDIT_CASES, ids=[c["name"] for c in AUDIT_CASES])
def test_independent_recurrence_and_memory(case, npu_device):
    result = run_audit_case(case, npu_device)
    assert result["passed"], result


def test_reference_analytic_two_tokens():
    q = k = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
    v = torch.tensor([[[[1., 0.]], [[2., 0.]]]])
    g = torch.zeros(1, 2, 1)
    beta = torch.full((1, 2, 1), .5)
    out, state = recurrent_reference(q, k, v, g, beta)
    torch.testing.assert_close(out, torch.tensor([[[[.5, 0.]], [[1.25, 0.]]]], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(state, torch.tensor([[[[1.25, 0.], [0., 0.]]]], dtype=torch.float64), rtol=0, atol=0)


def test_reference_correlated_beta_one_is_stable():
    case = next(c for c in AUDIT_CASES if c["name"] == "correlated_beta1")
    inputs, initial, cu = make_audit_inputs(case)
    out, state = recurrent_reference(*inputs, initial, cu)
    q, _ = _expand_kv_heads(*inputs[:2], case["heads"])
    # With k=e_0, beta=1 and g=0, each step replaces precisely the first
    # state row with v_t; the other rows retain their initial values.
    analytic = torch.einsum("bthd,hde->bthe", q.double(), initial[0].double())
    analytic += q[..., :1].double() * (inputs[2].double() - initial[0, :, 0, :].double())
    final = initial.double().clone()
    final[0, :, 0, :] = inputs[2][0, -1].double()
    torch.testing.assert_close(out, analytic, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state, final, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed,gate_scale", [(0, 1.), (42, .01), (93, 0.)])
def test_chunked_reference_against_recurrence(seed, gate_scale):
    inputs = list(_make_inputs(257, 8, 2, seed))
    inputs[3].mul_(gate_scale)
    initial = torch.randn(1, 8, D, D, dtype=torch.float16) * .1
    raw, state64 = recurrent_reference(*inputs, initial)
    q, k = _expand_kv_heads(*inputs[:2], 8)
    out, state = _reference(q, k, *inputs[2:], initial)
    torch.testing.assert_close(out.double(), raw * D**-.5, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(state.double(), state64, atol=1e-6, rtol=1e-5)


def test_checker_rejects_spike_hidden_by_rmse():
    expected = torch.ones(1_000_000)
    actual = expected.clone()
    actual[123] += .1
    assert error_metrics(actual, expected)["relative_rmse"] < .005
    with pytest.raises(AssertionError):
        assert_numerically_close("spike", actual, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_checker_rejects_nonfinite(bad):
    with pytest.raises(AssertionError):
        assert_numerically_close("nonfinite", torch.tensor([bad]), torch.zeros(1))


def test_checker_rejects_broadcasting():
    with pytest.raises(AssertionError):
        assert_numerically_close("shape", torch.ones(8), torch.ones(1))


def test_contiguous_views_respect_storage_offset(npu_device):
    """Regression for storage().data() ignoring a contiguous view's offset.

    Compare identical launches bit for bit; this tests pointer handling, without
    relying on or relaxing the separate numerical acceptance criterion.
    """
    inputs, initial, cu = make_audit_inputs(AUDIT_CASES[0])
    launch = _prepare_gdn_mega(*(x.to(npu_device) for x in inputs), cu.to(npu_device), initial.to(npu_device))
    pto_gdn_mega(**launch)
    torch.npu.synchronize()
    expected = {name: launch[name].clone() for name in ("out", "final_state")}
    guards = []
    for name in (*READ_ONLY[:-1], "initial_state", "out", "final_state"):
        old = launch[name]
        backing = torch.full((old.numel() + 128,), -123.25, dtype=old.dtype, device=npu_device)
        launch[name] = backing[64:-64].view_as(old)
        launch[name].copy_(old if name not in expected else torch.zeros_like(old))
        guards.append(backing)
    for _ in range(5):
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        for name in expected:
            assert torch.equal(launch[name], expected[name]), name
        assert all(bool((x[:64] == -123.25).all() and (x[-64:] == -123.25).all()) for x in guards)


def test_changed_inputs_on_reused_workspace(npu_device):
    """A second request must depend on its inputs, not previous scratch data."""
    case = AUDIT_CASES[0]
    inputs, initial, cu = make_audit_inputs(case)
    launch = _prepare_gdn_mega(*(x.to(npu_device) for x in inputs), cu.to(npu_device), initial.to(npu_device))
    pto_gdn_mega(**launch)
    torch.npu.synchronize()
    inputs2, initial2, _ = make_audit_inputs({**case, "seed": 6599})
    for name, tensor in zip(("q", "k", "v", "g", "beta"), inputs2):
        launch[name].copy_(tensor)
    launch["initial_state"].copy_(initial2)
    pto_gdn_mega(**launch)
    torch.npu.synchronize()
    reused = {name: launch[name].clone() for name in ("out", "final_state")}
    fresh = _prepare_gdn_mega(*(x.to(npu_device) for x in inputs2), cu.to(npu_device), initial2.to(npu_device))
    pto_gdn_mega(**fresh)
    torch.npu.synchronize()
    for name in reused:
        assert torch.equal(reused[name], fresh[name]), name


@pytest.mark.parametrize('gate,beta_value,initial_supplied', [(1., .9, True), (0., 1., True),
    (.01, 1.999, True), (0., 1.999, False), (-.01, .5, True)])
def test_reused_scratch_oracle_against_literal_fp64(gate, beta_value, initial_supplied):
    """The faster golden must preserve the independent literal recurrence."""
    inputs = list(_make_inputs(129, 8, 2, seed=185))
    inputs[3].mul_(gate)
    inputs[4].fill_(beta_value)
    initial = torch.randn(3, 8, D, D).half() if initial_supplied else None
    cu = torch.tensor([0, 63, 63, 129], dtype=torch.int32)
    snapshots = [x.clone() for x in inputs]
    initial_snapshot = initial.clone() if initial is not None else None
    q, k, v, g, beta = [x.double() for x in inputs]
    q, k = _expand_kv_heads(q, k, 8)
    expected = torch.empty_like(v)
    states = []
    for i, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        state = initial[i].double().clone() if initial is not None else torch.zeros(8, D, D, dtype=torch.float64)
        for t in range(start, end):
            state = state * g[0, t].exp()[:, None, None]
            prediction = (k[0, t, :, :, None] * state).sum(1)
            residual = beta[0, t, :, None] * (v[0, t] - prediction)
            state = state + k[0, t, :, :, None] * residual[:, None, :]
            expected[0, t] = (q[0, t, :, :, None] * state).sum(1)
        states.append(state)
    actual, final = recurrent_reference(*inputs, initial, cu)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(final, torch.stack(states), atol=1e-14, rtol=1e-14)
    assert all(torch.equal(a, b) for a, b in zip(inputs, snapshots))
    if initial is not None:
        assert torch.equal(initial, initial_snapshot)
