"""The expected first-hit time distribution (sensor_response.first_hit_density, build_make_hits_first_hit).

It must sum to the chance of a hit, equal the waveform at a low charge, move to the earliest arrivals at a high
charge, and agree with a simulated Poisson process, the many-pulse emulation the data are.
"""
import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.digitizer import resolve_model_config, single_pe_time_response
from lucid.simulation.sensor_response import apply_time_response, build_make_hits_first_hit, first_hit_density


def _gauss_rate(mu, t0=100.0, sigma=10.0, n=300):
    """A Gaussian-in-time expected waveform of total charge mu, 1 ns bins."""
    t = np.arange(n) + 0.5
    shape = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return mu * shape / shape.sum(), t


def _mean_time(p, t):
    return float((p * t).sum() / p.sum())


def test_sums_to_the_chance_of_a_hit():
    for mu in (0.01, 1.0, 50.0):
        rate, _ = _gauss_rate(mu)
        g = np.asarray(first_hit_density(jnp.asarray(rate)))
        assert np.isclose(g.sum(), 1.0 - np.exp(-mu), rtol=1e-4)


def test_low_charge_is_the_waveform_and_high_charge_is_earlier():
    rate, t = _gauss_rate(0.01)
    g = np.asarray(first_hit_density(jnp.asarray(rate)))
    assert abs(_mean_time(g, t) - _mean_time(rate, t)) < 0.2

    rate, t = _gauss_rate(100.0)
    g = np.asarray(first_hit_density(jnp.asarray(rate)))
    assert _mean_time(g, t) < _mean_time(rate, t) - 15.0      # the first of ~100 PEs sits ~2.3 sigma early


def test_shift_grows_with_charge():
    shifts = []
    for mu in (0.1, 1.0, 10.0, 100.0):
        rate, t = _gauss_rate(mu)
        g = np.asarray(first_hit_density(jnp.asarray(rate)))
        shifts.append(_mean_time(g, t) - _mean_time(rate, t))
    assert all(a > b for a, b in zip(shifts, shifts[1:]))


def test_matches_a_poisson_process():
    rng = np.random.default_rng(0)
    rate, t = _gauss_rate(3.0)
    g = np.asarray(first_hit_density(jnp.asarray(rate)))
    n_pulses = 40_000
    hit = rng.poisson(rate, size=(n_pulses, rate.size)) > 0
    has = hit.any(axis=1)
    p = has.mean()
    assert abs(p - g.sum()) < 4 * np.sqrt(p * (1 - p) / n_pulses)
    first = t[hit[has].argmax(axis=1)]
    se = first.std() / np.sqrt(len(first))
    assert abs(first.mean() - _mean_time(g, t)) < 4 * se


def test_factory_charge_waveform_and_window():
    fn = build_make_hits_first_hit(window_ns=100.0, bin_width_ns=1.0, smear_time=False)
    w = jnp.array([1.0, 2.0, 3.0, 4.0])
    idx = jnp.array([0, 0, 1, 1])
    times = jnp.array([10.2, 20.7, 30.1, 500.0])            # the last is outside the window
    charge, g, wf = fn(w, idx, times, 3, jax.random.PRNGKey(0), jnp.full(4, 0.5), jnp.ones(3))
    assert np.allclose(np.asarray(charge), [1.5, 3.5, 0.0])
    wf = np.asarray(wf)
    assert np.isclose(wf[0, 10], 0.5) and np.isclose(wf[0, 20], 1.0) and np.isclose(wf[1, 30], 1.5)
    assert np.isclose(wf.sum(), 3.0)                         # the 500 ns slot is in the charge, not in the histograms
    assert np.allclose(np.asarray(g).sum(1), 1.0 - np.exp(-wf.sum(1)), atol=1e-6)


def test_tts_smearing_keeps_the_charge_and_widens_the_waveform():
    sharp = build_make_hits_first_hit(window_ns=200.0, bin_width_ns=1.0, smear_time=False)
    smear = build_make_hits_first_hit(window_ns=200.0, bin_width_ns=1.0, tts_sigma_ns=3.0)
    args = (jnp.array([1.0]), jnp.array([0]), jnp.array([100.4]), 1, jax.random.PRNGKey(0), jnp.array([1.0]), jnp.ones(1))
    _, _, wf0 = sharp(*args)
    _, _, wf1 = smear(*args)
    wf0, wf1 = np.asarray(wf0)[0], np.asarray(wf1)[0]
    t = np.arange(200) + 0.5
    assert np.isclose(wf1.sum(), 1.0, atol=1e-5)
    var = (wf1 * (t - (wf1 * t).sum()) ** 2).sum()
    assert np.isclose(var, 3.0 ** 2 + 1.0 / 12.0, rtol=0.05)  # the TTS variance plus the bin's own
    assert (wf0 * (t - (wf0 * t).sum()) ** 2).sum() < 1e-6


def test_gradient_flows_through_the_weights():
    fn = build_make_hits_first_hit(window_ns=100.0, bin_width_ns=1.0, tts_sigma_ns=1.0)
    idx = jnp.array([0, 0])
    times = jnp.array([40.3, 55.8])

    def mean_first_time(scale):
        _, g, _ = fn(jnp.array([1.0, 1.0]) * scale, idx, times, 1, jax.random.PRNGKey(0), jnp.array([4.0, 4.0]), jnp.ones(1))
        t = jnp.arange(100) + 0.5
        return (g[0] * t).sum() / g[0].sum()

    grad = jax.grad(mean_first_time)(1.0)
    assert np.isfinite(float(grad)) and float(grad) < 0.0    # more charge, earlier first hit


def test_time_response_is_a_convolution_with_the_zero_offset_at_the_centre():
    rng = np.random.default_rng(0)
    rate = rng.random((3, 60)).astype(np.float32)
    kernel, centre = np.array([0.1, 0.5, 0.2, 0.1, 0.05], dtype=np.float32), 1       # a late tail: more weight after the zero offset
    out = np.asarray(apply_time_response(jnp.asarray(rate), kernel, centre))
    expected = np.stack([np.convolve(r, kernel, mode='full')[centre:centre + rate.shape[1]] for r in rate])
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    spike = np.zeros((1, 40), np.float32)
    spike[0, 20] = 1.0
    out = np.asarray(apply_time_response(jnp.asarray(spike), kernel, centre))[0]
    np.testing.assert_allclose(out[19:24], kernel, atol=1e-6)                        # the kernel's offsets land after, as well as before, the spike: bin 20 + (j - centre)


def test_single_pe_time_response_is_the_digitisers_delay_and_discriminator_loss():
    model = resolve_model_config('hk_wcsim')
    kernel, centre = single_pe_time_response(model, bin_ns=0.5, n_draws=400_000, chunk=200_000)
    assert 0.85 < kernel.sum() < 0.95                                                # the S-curve keeps ~90% of single PEs in HK
    offsets = (np.arange(len(kernel)) - centre) * 0.5
    p = kernel / kernel.sum()
    mean, median = (p * offsets).sum(), offsets[np.searchsorted(np.cumsum(p), 0.5)]
    assert mean > median > 0.0                                                       # a late exponential tail: delayed, and skewed to late times
    assert kernel[offsets > 20.0].sum() < 1e-3 * kernel.sum()
