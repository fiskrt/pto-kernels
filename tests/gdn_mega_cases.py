"""Fixed GDN regression inputs shared by pytest, audits and benchmarks.

All cases are regular tests. Never select inputs based on kernel output or
silently exclude stress failures from the optimization regression suite.
"""

# Fixed before evaluating the kernel: no tuning the oracle or tolerances to results.
STANDARD_CASES = [
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

# Fixed before evaluation. All critical cases obey the user's beta range [0,1].
# Named groups allow standalone audit subsets; ordinary pytest runs all groups.
CRITICAL_CASES = [
    dict(name=f'critical_boundary_{rate}_{amplitude}', seed=3001, lengths=(513,), heads=8, key_heads=2,
         correlated=True, beta=rate, gate_scale=0., amplitude=amplitude, state_scale=1.)
    for rate in (.499755859375, .5, .50048828125, 1.) for amplitude in (1., 32.)
] + [
    dict(name=f'critical_tiny_gate_{gate}_{length}', seed=3007, lengths=(length,), heads=8, key_heads=2,
         correlated=True, beta=1., constant_log_gate=gate, state_scale=1.)
    for gate in (-1e-8, -1e-7, -1e-6) for length in (4097, 32768)
] + [
    dict(name=f'critical_nearly_correlated_{seed}_{amplitude}', seed=seed, lengths=(1025,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=-1e-6, amplitude=amplitude, state_scale=1.)
    for seed in (3011, 3019, 3023) for amplitude in (1., 32., 128.)
] + [
    dict(name=f'critical_gaussian_{rate}_{gate}', seed=3037, lengths=(4097,), heads=8, key_heads=2,
         beta=rate, constant_log_gate=gate, state_scale=1.)
    for rate in (0., .5, 1.) for gate in (0., -1e-6, -80.)
] + [
    dict(name=f'critical_group_{heads}_{keys}', seed=3041, lengths=(1025,), heads=heads, key_heads=keys,
         beta=1., constant_log_gate=-1e-6, state_scale=1.)
    for heads, keys in ((24, 8), (48, 16), (64, 16))
] + [
    dict(name=f'critical_positive_{seed}', seed=seed, lengths=(4097,), heads=8, key_heads=2,
         fla='positive', beta=1., constant_log_gate=-1e-6, state_scale=1.)
    for seed in (3049, 3061)
] + [
    dict(name='critical_random_beta_32k', seed=3067, lengths=(32768,), heads=8, key_heads=2,
         constant_log_gate=-1e-6, state_scale=1.),
    dict(name='critical_alternating_beta', seed=3079, lengths=(4097,), heads=8, key_heads=2,
         nearly_correlated=True, alternating_beta=True, constant_log_gate=-1e-7, state_scale=1.),
    dict(name='critical_packed', seed=3083, lengths=(0, 127, 128, 129, 0, 513), heads=24, key_heads=8,
         beta=1., constant_log_gate=-1e-6, state_scale=1.),
    dict(name='critical_zero_initial', seed=3089, lengths=(4097,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=-1e-6, initial=False),
    dict(name='critical_large_static_state', seed=3109, lengths=(1025,), heads=8, key_heads=2,
         beta=0., gate_scale=0., state_scale=128.),
]


# Matched controls added after the initial critical sweep to isolate failures.
CRITICAL_CASES += [
    dict(name='control_nearly_correlated_no_decay', seed=3011, lengths=(1025,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=0., state_scale=1.),
    dict(name='control_zero_initial_no_decay', seed=3089, lengths=(4097,), heads=8, key_heads=2,
         nearly_correlated=True, beta=1., constant_log_gate=0., initial=False),
    dict(name='control_coordinate_zero_initial', seed=3007, lengths=(32768,), heads=8, key_heads=2,
         correlated=True, beta=1., constant_log_gate=-1e-7, initial=False),
    dict(name='control_coordinate_beta_zero', seed=3007, lengths=(32768,), heads=8, key_heads=2,
         correlated=True, beta=0., constant_log_gate=-1e-7, state_scale=1.),
    dict(name='control_coordinate_no_decay', seed=3007, lengths=(32768,), heads=8, key_heads=2,
         correlated=True, beta=1., constant_log_gate=0., state_scale=1.),
]


# Interior rates distinguish endpoint behavior from the supported interval.
CRITICAL_CASES += [
    dict(name=f'interior_beta_{rate}_{kind}', seed=3163, lengths=(4097,), heads=8, key_heads=2,
         **{kind: True}, beta=rate, constant_log_gate=-1e-7, state_scale=1.)
    for rate in (.25, .5, .6, .9) for kind in ('correlated', 'nearly_correlated')
] + [
    dict(name='interior_random_beta_correlated', seed=3163, lengths=(4097,), heads=8, key_heads=2,
         correlated=True, constant_log_gate=-1e-7, state_scale=1.)
]


# Matched unit-scale sweep: correlation is varied without changing Q/V/h0.
# These are synthetic sensitivity measurements, not measured model inputs.
CORRELATION_CASES = [
    dict(name=f"correlation_rho{rho}_beta{rate}_g{gate}", seed=3827,
         lengths=(4096,), heads=8, key_heads=2, key_correlation=rho,
         beta=rate, constant_log_gate=gate, state_scale=.1)
    for rho in (0., .25, .5, .75, .95, 1.)
    for rate in (.5, 1.) for gate in (0., -.001, -.1, -1.)
] + [
    dict(name=f"correlation_long_rho{rho}", seed=3827, lengths=(32768,),
         heads=8, key_heads=2, key_correlation=rho, beta=1., constant_log_gate=0., state_scale=.1)
    for rho in (.95, 1.)
] + [
    dict(name=f"correlation_large_rho{rho}_g{gate}", seed=3827, lengths=(4096,),
         heads=8, key_heads=2, key_correlation=rho, beta=1., constant_log_gate=gate,
         amplitude=32., state_scale=.1)
    for rho in (0., .25, .5, .75, .95, 1.) for gate in (0., -.001)
] + [
    dict(name=f"correlation_initial_rho{rho}_g{gate}", seed=3827, lengths=(4096,),
         heads=8, key_heads=2, key_correlation=rho, beta=1., constant_log_gate=gate,
         state_scale=1.)
    for rho in (.75, .95, 1.) for gate in (0., -.001)
]


# Exercise the independent value-column workers and producer streams across
# grouped, single-key and ordinary multi-head layouts, including partial tiles.
# Existing critical cases cover H8/Hg2 at 1K–32K with extreme gates and beta.
SCHEDULING_CASES = [
    dict(name=f'parallel_columns_hg{keys}_t{length}_initial{initial}',
         seed=4111, lengths=(length,), heads=8, key_heads=keys,
         nearly_correlated=True, beta=1., constant_log_gate=-1e-6,
         amplitude=32., initial=initial)
    for keys in (1, 4, 8) for length in (1024, 2048, 2139) for initial in (False, True)
]

AUDIT_CASES = STANDARD_CASES + CRITICAL_CASES + CORRELATION_CASES + SCHEDULING_CASES
