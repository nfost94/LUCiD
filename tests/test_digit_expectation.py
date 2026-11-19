"""The analytic digitiser (digitizer.digit_response_kernels, sensor_response.digit_expectation) against the digitiser itself.

Each test draws pulses of Poisson PEs with a known time profile, plus dark, and runs them through the real windowing (digitizer.digitize_event, with the HK dead time), the readout
(apply_readout_resolution), the discriminator and the charge step. The expected digits, from the profile's rate alone, must match them: their number, their time, their charge and the
distribution of that charge, in a single window and across a dead time with later windows. The gradient through the digit expectation must be finite, also for a bright PMT.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

from lucid.simulation.digitizer import (
    _remove_bin_spread, apply_charge_precision, apply_discriminator, apply_readout_resolution, digit_response_kernels, digitize_event, generate_dark_noise,
    resolve_model_config, single_pe_time_response)
from lucid.simulation.sensor_response import digit_expectation

BIN = 1.0
MODEL = resolve_model_config({'model': 'hk_wcsim', 'deadtime_ns': 560.0, 'tdc_ns': 0.1, 'pe_precision': 0.3})
DARK_KHZ = 4.6725
WINDOW_BINS, DEAD_BINS = int(MODEL['integration_window_ns'] / BIN), int(MODEL['deadtime_ns'] / BIN)
_KERNELS = {}


def _pe_cut(top):                                          # the rule of digitizer.pulse_charge_response
    return int(np.ceil(top + 10.0 * np.sqrt(top) + 10.0))


def _kernels(n_max):
    if n_max not in _KERNELS:
        _KERNELS[n_max] = digit_response_kernels(MODEL, BIN, n_max, n_draws=100_000)
    return _KERNELS[n_max]


def _rate(peaks, n_bins):
    """Expected PEs per bin: Gaussian peaks (mu, centre ns, sigma ns), and the dark rate."""
    edges = np.arange(n_bins + 1) * BIN
    rate = np.full(n_bins, DARK_KHZ * 1e-6 * BIN)
    for mu, centre, sigma in peaks:
        rate += mu * np.diff(ndtr((edges - centre) / sigma))
    return rate


def _expect(rate, kernels):
    d, c, f, w = digit_expectation(jnp.asarray(rate[None], jnp.float32), jnp.asarray(kernels.time, jnp.float32), jnp.asarray(kernels.charge_time, jnp.float32),
                                   jnp.asarray(kernels.charge_pmf, jnp.float32), kernels.centre, WINDOW_BINS, DEAD_BINS)
    return np.asarray(d[0], float), np.asarray(c[0], float), np.asarray(f[0], float), np.asarray(w[0], float)


def _simulate(peaks, n_bins, n_pulses, rng):
    """The digits of `n_pulses` pulses: each pulse is one sensor with Poisson PEs in each peak and dark over the window. Returns (pulse, time, charge) of every digit."""
    return _simulate_ns(peaks, n_bins * BIN, n_pulses, rng)


def _simulate_ns(peaks, length_ns, n_pulses, rng):
    sensor, times = [], []
    for mu, centre, sigma in peaks:
        counts = rng.poisson(mu, size=n_pulses)
        sensor.append(np.repeat(np.arange(n_pulses), counts))
        times.append(rng.normal(centre, sigma, size=int(counts.sum())))
    d_s, d_t, _ = generate_dark_noise(n_pulses, DARK_KHZ, 0.0, length_ns, rng)
    sensor, times = np.concatenate(sensor + [d_s]), np.concatenate(times + [d_t])
    res = digitize_event(sensor, times, np.ones(len(times)), n_pulses, MODEL)
    pe, t = apply_readout_resolution(res.digit_pe_true, res.digit_time, MODEL, rng)
    keep = apply_discriminator(pe, MODEL, rng)
    q = np.asarray(apply_charge_precision(pe, MODEL), dtype=float)
    return res.digit_sensor_idx[keep], t[keep], q[keep]


def _check(peaks, n_bins, n_pe, n_pulses=40_000, seed=0, regions=((0.0, 1e9),)):
    rng = np.random.default_rng(seed)
    kernels = _kernels(n_pe)
    rate = _rate(peaks, n_bins)
    d, c, f, w = _expect(rate, kernels)
    assert np.all(np.isfinite(d)) and np.all(np.isfinite(c)) and np.all(np.isfinite(f))
    _, t, q = _simulate(peaks, n_bins, n_pulses, rng)
    centres = (np.arange(n_bins) + 0.5) * BIN
    for lo, hi in regions:                                        # digits in each time region
        m, ms = (centres >= lo) & (centres < hi), (t >= lo) & (t < hi)
        exp_n, sim_n = n_pulses * d[m].sum(), ms.sum()
        assert abs(sim_n - exp_n) <= 5 * np.sqrt(max(exp_n, 1.0)) + 0.01 * exp_n, (lo, hi, exp_n, sim_n)
    exp_n, sim_n = n_pulses * d.sum(), len(t)
    assert abs(sim_n - exp_n) <= 5 * np.sqrt(exp_n) + 0.01 * exp_n, (exp_n, sim_n)
    exp_q, sim_q = c.sum() / d.sum(), q.mean()                     # the mean charge of a digit
    assert abs(sim_q - exp_q) <= 5 * q.std() / np.sqrt(len(q)) + 0.01 * exp_q, (exp_q, sim_q)
    exp_t, sim_t = (centres * d).sum() / d.sum(), t.mean()         # the mean time of a digit
    assert abs(sim_t - exp_t) <= 5 * t.std() / np.sqrt(len(t)) + 0.3, (exp_t, sim_t)
    level = np.minimum(np.rint(q / kernels.charge_step).astype(int), len(kernels.charge_levels) - 1)
    sim_h, exp_h = np.bincount(level, minlength=len(f)), n_pulses * f # the distribution of the charge
    big = exp_h >= 30
    assert big.any()
    assert np.all(np.abs(sim_h - exp_h)[big] <= 5 * np.sqrt(exp_h[big]) + 0.02 * exp_h[big]), np.c_[exp_h[big], sim_h[big]]


def test_one_window_matches_the_digitiser_at_low_and_high_charge():
    for mu in (0.4, 3.0, 15.0):
        _check([(mu, 100.0, 8.0)], n_bins=300, n_pe=_pe_cut(mu))


def test_dead_time_and_later_windows_match_the_digitiser():
    # A first window opened at 100 ns is dead until 860 ns: the 400 ns peak only gives a digit if the first peak had no PE, the 900 ns peak after it can open a window again.
    _check([(1.0, 100.0, 8.0), (1.0, 400.0, 8.0), (1.0, 900.0, 8.0)], n_bins=1500, n_pe=_pe_cut(3.0), regions=((0.0, 250.0), (250.0, 700.0), (700.0, 1500.0)))


def test_the_charge_of_the_second_peak_is_in_the_early_digits():
    # Two close peaks: the digit is timed at the first peak and carries both peaks' charge; a digit at the second peak needs no PE in the first.
    peaks = [(2.0, 100.0, 5.0), (2.0, 140.0, 5.0)]
    rate = _rate(peaks, 300)
    d, c, f, w = _expect(rate, _kernels(_pe_cut(4.0)))
    early, late = (np.arange(300) < 120), (np.arange(300) >= 120)
    assert d[late].sum() < 0.2 * d[early].sum()
    assert c[early].sum() / d[early].sum() > 3.0                  # both peaks' PEs in the early digit: about 4 PE less the discriminator's and the charge step's loss
    _check(peaks, n_bins=300, n_pe=_pe_cut(4.0), regions=((0.0, 120.0), (120.0, 300.0)))


def test_kernels_are_consistent_with_the_digitiser_and_each_other():
    k = _kernels(12)
    assert np.allclose(k.time.sum(1), k.charge_pmf.sum(1), rtol=2e-3)
    assert np.allclose(k.charge_time.sum(1), k.charge_pmf @ k.charge_levels, rtol=2e-3)
    one, centre = single_pe_time_response(MODEL, BIN, n_draws=400_000)
    assert abs(k.time[0].sum() - one.sum()) < 0.01 and k.centre == centre
    assert k.time[0].sum() < k.time[1].sum() < 1.0 + 1e-9         # two PEs are kept more often than one


def test_gradient_through_the_digit_expectation_is_finite_for_a_bright_pmt():
    kernels = _kernels(_pe_cut(25.0))
    rate = jnp.asarray(_rate([(25.0, 100.0, 8.0)], 300), jnp.float32)
    args = (jnp.asarray(kernels.time, jnp.float32), jnp.asarray(kernels.charge_time, jnp.float32), jnp.asarray(kernels.charge_pmf, jnp.float32))

    def total_charge(scale):
        return digit_expectation((scale * rate)[None], *args, kernels.centre, WINDOW_BINS, DEAD_BINS)[1].sum()

    def mean_time(scale):
        d = digit_expectation((scale * rate)[None], *args, kernels.centre, WINDOW_BINS, DEAD_BINS)[0][0]
        return (d * (jnp.arange(300) + 0.5)).sum() / d.sum()

    g_q, g_t = jax.grad(total_charge)(1.0), jax.grad(mean_time)(1.0)
    assert np.isfinite(float(g_q)) and np.isfinite(float(g_t))
    assert 12.0 < float(g_q) < 38.0                               # the recorded charge grows with the rate, about one PE per PE
    assert float(g_t) < 0.0                                       # more light, an earlier first PE and so an earlier digit


def test_digit_time_histogram_matches_the_digitiser_bin_by_bin():
    """A sharp peak (sigma 1.5 ns) on 1 ns bins, 400k pulses through the real digitiser: the expected digits per bin must match the simulated histogram, in particular on the late side where the jitter's
    exponential tail is. The PE times are continuous; the binned rate does not say where in its bin a PE is, and the kernels (started from the bin's centre) have the bin's spread removed."""
    rng = np.random.default_rng(5)
    peaks, n_bins, n_pulses = [(0.6, 100.0, 1.5)], 300, 400_000
    kernels = digit_response_kernels(MODEL, BIN, _pe_cut(0.6), n_draws=2_000_000)
    d, c, f, w = _expect(_rate(peaks, n_bins), kernels)
    _, t, q = _simulate(peaks, n_bins, n_pulses, rng)
    sim_h = np.histogram(t, bins=np.arange(n_bins + 1) * BIN)[0].astype(float)
    exp_h = n_pulses * d
    big = exp_h >= 30
    z = (sim_h - exp_h)[big] / np.sqrt(exp_h[big])
    late = (np.arange(n_bins) >= 102) & (np.arange(n_bins) < 140)                      # beyond the peak's own width: the jitter's tail
    ratio_late = sim_h[late].sum() / exp_h[late].sum()
    sigma_late = 1.0 / np.sqrt(exp_h[late].sum())
    print(f'chi2 / bins = {(z ** 2).mean():.2f} over {big.sum()} bins; late side simulated / expected = {ratio_late:.4f} (sigma {sigma_late:.4f})')
    assert (z ** 2).mean() < 1.6, (z ** 2).mean()
    assert abs(ratio_late - 1.0) < 4 * sigma_late + 0.002, (ratio_late, sigma_late)


@pytest.mark.parametrize('bin_ns', (0.5, 0.25))
def test_leading_edge_of_the_digits_against_the_bin_width(bin_ns):
    """A sharp edge (sigma 0.7 ns) on 0.5 and 0.25 ns bins, 1.5M pulses of continuous PE times through the real digitiser. Prints the running excess of simulated over expected digits at the edge and its 1-sigma band.
    The 0.5 ns bin is a multiple of the TDC step and has the bin's spread removed from the kernels: the dip was -769 digits (-4.2 sigma) without that, -4745 before the TDC phase was put in the kernels, -342 (-1.9 sigma)
    now. The 0.25 ns bin is not a multiple of the 0.1 ns TDC step, so its bin edges are off the TDC grid, which the kernels do not handle (the dip stays about -2300 digits, with or without the bin spread)."""
    rng = np.random.default_rng(11)
    mu, centre, sigma, length, n_pulses = 0.4, 50.0, 0.7, 300.0, 1_500_000
    n_bins = int(round(length / bin_ns))
    edges = np.arange(n_bins + 1) * bin_ns
    rate = np.full(n_bins, DARK_KHZ * 1e-6 * bin_ns) + mu * np.diff(ndtr((edges - centre) / sigma))
    kernels = digit_response_kernels(MODEL, bin_ns, _pe_cut(mu), n_draws=3_000_000)
    d, c, f, w = digit_expectation(jnp.asarray(rate[None], jnp.float32), jnp.asarray(kernels.time, jnp.float32), jnp.asarray(kernels.charge_time, jnp.float32),
                                   jnp.asarray(kernels.charge_pmf, jnp.float32), kernels.centre, int(round(200.0 / bin_ns)), int(round(560.0 / bin_ns)))
    expected = n_pulses * np.asarray(d[0], float)
    _, t, q = _simulate_ns([(mu, centre, sigma)], length, n_pulses, rng)
    simulated = np.histogram(t, bins=edges)[0].astype(float)
    cum, band = np.cumsum(simulated - expected), np.sqrt(np.cumsum(expected))
    k = int(np.argmin(cum / band))
    mean_shift = ((simulated * (edges[:-1] + bin_ns / 2)).sum() / simulated.sum() - (expected * (edges[:-1] + bin_ns / 2)).sum() / expected.sum())
    print(f'bin {bin_ns} ns: running excess min {cum.min():.0f} digits ({cum.min() / band[np.argmin(cum)]:+.1f} sigma) at t = {edges[k]:.2f} ns; band there {band[k]:.0f}; '
          f'mean time simulated - expected {mean_shift:+.4f} ns; digits {simulated.sum():.0f} / {expected.sum():.0f}')
    assert abs(mean_shift) < 0.03, mean_shift                                  # 0.059 ns before the TDC phase was put in the kernels, 0.009 ns now
    limit = {0.5: 0.0012, 0.25: 0.006}[bin_ns]                                  # of the digits: -0.17% without the bin spread removed at 0.5 ns, -0.075% with it; -0.5% at 0.25 ns
    assert cum.min() > -limit * simulated.sum(), cum.min()


def test_removing_the_bin_spread_takes_one_bins_variance_and_keeps_the_sum_and_the_mean():
    """A Gaussian binned at its centres has the variance sigma^2 + bin^2 / 12 (Sheppard); the kernels' correction takes that spread out, to second order, and leaves the sum and the mean."""
    for h, s in ((0.5, 1.0), (0.5, 0.8), (1.0, 1.5)):
        x = np.arange(-80, 81) * h
        p = ndtr((x + h / 2 - 0.3) / s) - ndtr((x - h / 2 - 0.3) / s)
        q = _remove_bin_spread(p[None])[0]

        def variance(w):
            return (w * x ** 2).sum() / w.sum() - ((w * x).sum() / w.sum()) ** 2

        assert abs(variance(p) - (s ** 2 + h ** 2 / 12)) < 1e-4
        assert abs(variance(q) - s ** 2) < 1e-4, (variance(q), s ** 2)
        assert abs(q.sum() - 1.0) < 1e-6 and abs((q * x).sum() - 0.3) < 1e-6
        assert q.min() >= 0.0
