"""Calibration through the digitizer: a pulsed source's recorded charge, and fitting it.

``pulse_charge_response`` turns the digitizer (SPE spectrum, threshold curve, charge precision) and
dark noise into a smooth per-pulse response; ``sample_pulse_charge_sums`` digitizes pulses one by
one. ``PulseResponse`` puts the response into the calibration model. With no response the fit is
the published estimator (gated by the other calibration tests, which pass none).
"""
import numpy as np
import jax
import jax.numpy as jnp

from lucid.fitting import FieldParams, PulseResponse, calibrate, closure_data, pulsed_closure_data
from lucid.fitting.calib import CalibrationForward
from lucid.simulation.digitizer import (pulse_charge_response, resolve_model_config,
                                        sample_pulse_charge_sums)
from tests import _calib_toy as toy
from tests._calib_toy import FIELDS

HK = resolve_model_config({"model": "hk", "pe_precision": 0.3})
DARK = 4.6725e3 * 1350e-9                # HK dark rate x full readout, per pulse


def test_response_matches_pulse_by_pulse_digitization():
    mu = np.array([1e-3, 0.05, 1.0, 5.0])
    mean, var, p_hit = pulse_charge_response(HK, mu, dark_mu=DARK)
    reps, n = 20, 40_000
    s, s2 = sample_pulse_charge_sums(np.repeat(mu, reps), n, HK, np.random.default_rng(3), DARK)
    s, s2 = s.reshape(len(mu), reps), s2.reshape(len(mu), reps)
    np.testing.assert_allclose(s.mean(1) / n, mean, rtol=0.02)
    # the sampled variance leaves out the prompt x dark cross term (~1% at 5 p.e.)
    np.testing.assert_allclose((s2 / n - (s / n) ** 2).mean(1), var, rtol=0.03)
    assert abs(p_hit[0] / mu[0] - 0.899) < 0.01        # single-p.e. regime: WCSim's threshold curve


def test_pulse_response_interpolates_the_table_and_is_linear_below_it():
    resp = PulseResponse(HK, n_pulses=1, pulse_scale=[1.0], dark_mu=DARK)
    x = np.array([2e-3, 0.3, 7.0])
    mean, _, _ = pulse_charge_response(HK, np.concatenate([[0.0], x]), dark_mu=DARK)
    np.testing.assert_allclose(np.asarray(resp.per_pulse(jnp.asarray(x))), mean[1:] - mean[0],
                               rtol=5e-3)
    assert abs(resp.pedestal - mean[0]) < 1e-3 * mean[0]
    tiny = jnp.asarray([1e-9, 2e-9])
    r = np.asarray(resp.per_pulse(tiny) / tiny)
    assert abs(r[0] - r[1]) < 1e-6 and 0.8 < r[0] < 0.9
    g = jax.grad(lambda v: resp.per_pulse(v).sum())(jnp.asarray([0.0, 0.5]))
    assert np.isfinite(np.asarray(g)).all()


def test_pulse_profile_gains_recovers_the_gains_that_made_the_data():
    rng = np.random.default_rng(0)
    ns, s = 50, 3
    mu = jnp.asarray(rng.uniform(0.01, 2.0, (s, ns)))
    k = np.exp(0.2 * rng.standard_normal(ns)); k /= np.exp(np.mean(np.log(k)))
    resp = PulseResponse(HK, n_pulses=1000, pulse_scale=[1.0, 0.5], dark_mu=DARK)
    wl = jnp.asarray([[0], [1], [1]])
    data = resp.charge(jnp.asarray(k)[None, :] * mu, wl)
    k_hat = np.asarray(resp.profile_gains(mu, data.sum(0), wl, gauge='log'))
    np.testing.assert_allclose(k_hat, k, rtol=1e-4)


# --- closure through the digitizer on the synthetic problem ---------------------------------------

NS = 24
GAINS = np.exp(0.2 * np.random.default_rng(5).standard_normal(NS))
GAINS /= np.exp(np.mean(np.log(GAINS)))
FIT_KW = dict(steps=120, refresh=4, jacobian_draws=1, n_forward_draws=1,
              max_step=0.3, lam=1e-4, mu=0.02)


def _stub_sim(source, dp, key):
    return toy.charge(source, dp), jnp.zeros(NS)


def _setup(pulse_mu_max):
    params = FieldParams(toy.dp_true(NS), FIELDS, n_sensors=NS)
    theta_true = params.theta_from_physical()
    sources = toy.sources(NS, seed=11, b_spread=0.5, b_scale=2.3)
    fwd = CalibrationForward(_stub_sim, sources, params, NS)
    mu = closure_data(fwd, theta_true, GAINS, n_draws=1)
    # per-pulse scale putting the brightest sensor at `pulse_mu_max` p.e. per pulse
    resp = PulseResponse(HK, n_pulses=50_000, pulse_scale=[pulse_mu_max / float(mu.max())],
                         dark_mu=DARK)
    start = np.asarray(theta_true) + np.array([0.2, -0.15, 0.1])
    return params, theta_true, sources, fwd, resp, start


def test_closure_on_the_expected_digitized_charge_is_exact():
    params, theta_true, sources, fwd, resp, start = _setup(3.0)
    mu = closure_data(fwd, theta_true, GAINS, n_draws=1)
    data = resp.charge(mu, jnp.zeros((mu.shape[0], 1), dtype=jnp.int32))
    res = calibrate(_stub_sim, sources, params, data, start, response=resp, **FIT_KW)
    np.testing.assert_allclose(np.asarray(res['theta']), np.asarray(theta_true), atol=1e-3)


def test_ignoring_the_digitizer_biases_the_fit():
    """The same data fitted as gain x forward: threshold, rounding and dark noise displace it."""
    params, theta_true, sources, fwd, resp, start = _setup(3.0)
    mu = closure_data(fwd, theta_true, GAINS, n_draws=1)
    data = resp.charge(mu, jnp.zeros((mu.shape[0], 1), dtype=jnp.int32))
    res = calibrate(_stub_sim, sources, params, data, start, **FIT_KW)
    assert np.abs(np.asarray(res['theta']) - np.asarray(theta_true)).max() > 1e-2


def test_closure_on_pulse_by_pulse_digitized_data():
    """Data digitized pulse by pulse (sampling noise, measured variance) fit back to truth within
    its statistics."""
    params, theta_true, sources, fwd, resp, start = _setup(3.0)
    data, var = pulsed_closure_data(fwd, theta_true, resp, GAINS, n_draws=1, seed=2)
    res = calibrate(_stub_sim, sources, params, data, start, response=resp, data_var=var,
                    **FIT_KW)
    np.testing.assert_allclose(np.asarray(res['theta']), np.asarray(theta_true), atol=2e-2)
