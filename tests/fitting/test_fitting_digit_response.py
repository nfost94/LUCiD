"""``DigitResponse``: the gain inside the digitiser's recorded charge, profiled per sensor.

The forward returns the expected windows per number of photoelectrons, ``W[..., m-1]``, and the charge is
``sum_m W_m c_m(g)`` with ``c_m(g)`` not proportional to ``g``. The profile must find each sensor's ``g``
from the data summed over configurations, where a single linear factor cannot; the forward must carry
the windows' trailing axis through its draws and average; the jackknife must accept it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lucid.fitting import CalibrationForward, DigitResponse, jackknife_profile, profile_gains

M = 4
LN_G = np.linspace(-0.45, 0.45, 31)


def _table():
    # A toy c_m(g) with the HK digitiser's shape: one photoelectron ~ g^1.25, many ~ g (column 0 unused).
    g = np.exp(LN_G)[:, None]
    m = np.arange(M + 1)[None, :]
    return np.where(m == 0, 0.0, 0.9 * m * g ** (1.0 + 0.25 / np.maximum(m, 1)))


def _windows(rng, S=3, NS=200):
    # Per configuration a different mix of window sizes per sensor: dim (mostly one photoelectron) to bright.
    occ = rng.uniform(0.01, 3.0, (S, NS, 1))
    m = np.arange(1, M + 1)[None, None, :]
    w = occ**m / np.cumprod(np.arange(1, M + 1))[None, None, :] * np.exp(-occ)
    return jnp.asarray(w * 1000.0, dtype=jnp.float32)


def test_profile_recovers_the_gains_a_linear_factor_cannot():
    rng = np.random.default_rng(0)
    resp = DigitResponse(LN_G, _table())
    W = _windows(rng)
    g = np.exp(rng.normal(0.0, 0.05, W.shape[1]))
    g /= np.exp(np.mean(np.log(g)))  # the log gauge, as the truth
    data = resp.model_charge(W, jnp.asarray(g, jnp.float32), None)  # (S, NS)
    g_hat = np.asarray(resp.profile_gains(W, data.sum(0), None, gauge="log"))
    np.testing.assert_allclose(g_hat, g, rtol=2e-4)
    model = resp.model_charge(W, jnp.asarray(g_hat), None)
    np.testing.assert_allclose(np.asarray(model), np.asarray(data), rtol=1e-3)
    # The linear profile on the gain-1 charge fits the sums but not every configuration: its residual is larger.
    q1 = resp.model_charge(W, jnp.ones(W.shape[1]), None)
    k = profile_gains(q1.sum(0) + 1e-12, data.sum(0), gauge="log")
    lin = np.abs(np.asarray(k[None, :] * q1 - data) / np.asarray(data)).max()
    dig = np.abs(np.asarray(model - data) / np.asarray(data)).max()
    assert lin > 10 * dig


def test_c_of_is_linear_in_ln_g_and_held_beyond_the_table():
    resp = DigitResponse(LN_G, _table())
    tab = _table()
    np.testing.assert_allclose(np.asarray(resp.c_of(jnp.asarray([1.0]))[0]), tab[15], rtol=1e-6)
    mid = 0.5 * (LN_G[3] + LN_G[4])
    np.testing.assert_allclose(
        np.asarray(resp.c_of(jnp.asarray([np.exp(mid)]))[0]), 0.5 * (tab[3] + tab[4]), rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(resp.c_of(jnp.asarray([10.0]))[0]), tab[-1], rtol=1e-5)


def test_forward_and_jackknife_carry_the_windows_axis():
    NS = 7

    def predict(theta, src, wl, gains, key):
        # Windows that move with theta and the source, plus key noise: (NS, M).
        base = theta[0] * src * (1.0 + jnp.arange(NS))[:, None] * jnp.arange(1, M + 1)[None, :]
        return base * (1.0 + 0.01 * jax.random.normal(key, (NS, M)))

    fwd = CalibrationForward(None, [1.0, 2.0, 3.0], None, NS, predict=predict)
    theta = jnp.asarray([1.5])
    one = fwd(
        theta, jax.random.fold_in(jax.random.PRNGKey(100), 0), jnp.ones(NS)
    )  # draw 0 of the key
    assert one.shape == (3, NS, M)
    draws = fwd.draws(theta, jax.random.PRNGKey(100), jnp.ones(NS), 4)
    assert draws.shape == (4, 3, NS, M)
    np.testing.assert_allclose(
        np.asarray(fwd.average(theta, jax.random.PRNGKey(100), jnp.ones(NS), 4)),
        np.asarray(draws.mean(0)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(draws[0]), np.asarray(one), rtol=0, atol=0)
    resp = DigitResponse(LN_G, _table())
    data = resp.model_charge(draws.mean(0), jnp.ones(NS), None)
    k, model = jackknife_profile(draws, data.sum(0), "log", resp, None)
    assert k.shape == (NS,) and model.shape == (3, NS)
    np.testing.assert_allclose(np.asarray(model.sum(0)), np.asarray(data.sum(0)), rtol=2e-3)
