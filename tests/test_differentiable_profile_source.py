"""DifferentiableProfileSource: directions drawn from a frozen proposal, the profile carried in the ray weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.sources import differentiable_profile_source, profile_intensity, PROFILE_FORMS, N_PROFILE_PARAMS

DIRECTION = jnp.array([1.0, 2.0, -3.0]) / jnp.sqrt(14.0)
POSITION = jnp.array([0.5, -1.0, 2.0])
# The physical diffuser model at a main-group template (diffuser_template.ipynb).
PHYSICAL = (1.3, 67.0, 14.7, float(np.log(0.0133)), 23.8, 77.9, 1.64, 2.2, 0.553)
# Representative shapes: a collimator-like and a diffuser-like composite, a super-Gaussian, SK-IV's beam,
# and the physical model.
CASES = [
    ('composite', (2.676, 7.469, 2.705, 0.0578), 3.5),
    ('composite', (38.58, 1.216, 44.78, 2.378), 50.0),
    ('super_gaussian', (36.65, 1.646), 60.0),
    ('sk_beam', (3300.0,), 16.0),
    ('physical', PHYSICAL, 90.0),
]


def _angles(directions):
    return np.degrees(np.arccos(np.clip(np.asarray(directions) @ np.asarray(DIRECTION), -1, 1)))


@pytest.mark.parametrize('profile,params,cut', CASES)
def test_contract_cap_and_normalisation(profile, params, cut):
    src = differentiable_profile_source(POSITION, DIRECTION, params, cut, profile=profile,
                                        intensity=2.5e6)
    d, o, w = src(4096, jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(o, np.tile(POSITION, (4096, 1)))
    np.testing.assert_allclose(float(jnp.sum(w)), 2.5e6, rtol=1e-5)
    assert _angles(d).max() <= cut + 1e-3


@pytest.mark.parametrize('profile,params,cut', CASES)
def test_weighted_rays_reproduce_the_profile(profile, params, cut):
    # Weighted intensity per solid angle in angular bins vs the analytic profile.
    src = differentiable_profile_source(POSITION, DIRECTION, params, cut, profile=profile)
    d, _, w = src(400_000, jax.random.PRNGKey(1))
    a, w = _angles(d), np.asarray(w)
    edges = np.linspace(0, cut, 26)
    h, _ = np.histogram(a, edges, weights=w)
    sa = np.cos(np.radians(edges[:-1])) - np.cos(np.radians(edges[1:]))
    centres = 0.5 * (edges[1:] + edges[:-1])
    pp = jnp.zeros(N_PROFILE_PARAMS).at[:len(params)].set(jnp.asarray(params))
    expect = np.asarray(profile_intensity(jnp.asarray(centres), pp, PROFILE_FORMS.index(profile)))
    measured, expect = h / sa, expect
    measured, expect = measured / measured.max(), expect / expect.max()
    bright = expect > 0.05
    np.testing.assert_allclose(measured[bright], expect[bright], rtol=0.06, atol=0.01)


def test_directions_do_not_depend_on_params():
    # Replacing params on a built source (as a fit does) leaves every ray where it was.
    a = differentiable_profile_source(POSITION, DIRECTION, (38.0, 1.2, 44.0, 2.4), 50.0)
    b = a._replace(params=a.params.at[:4].set(jnp.array([20.0, 3.0, 30.0, 1.0])))
    key = jax.random.PRNGKey(3)
    np.testing.assert_array_equal(a(1024, key)[0], b(1024, key)[0])


def _physical(theta_t):
    return jnp.zeros(N_PROFILE_PARAMS).at[:9].set(jnp.asarray(PHYSICAL, dtype=jnp.float32)).at[1].set(theta_t)


@pytest.mark.parametrize('dtheta', [-4.0, 4.0])
def test_profile_reproduced_away_from_the_proposal(dtheta):
    # Built at the template, theta_t then moved by about the units' spread: the importance weights still
    # reproduce the moved profile.
    src = differentiable_profile_source(POSITION, DIRECTION, PHYSICAL, 90.0, profile='physical')
    src = src._replace(params=_physical(PHYSICAL[1] + dtheta))
    d, _, w = src(400_000, jax.random.PRNGKey(7))
    a, w = _angles(d), np.asarray(w)
    edges = np.linspace(0, 90, 46)
    h, _ = np.histogram(a, edges, weights=w)
    sa = np.cos(np.radians(edges[:-1])) - np.cos(np.radians(edges[1:]))
    centres = 0.5 * (edges[1:] + edges[:-1])
    expect = np.asarray(profile_intensity(jnp.asarray(centres), src.params, PROFILE_FORMS.index('physical')))
    measured, expect = h / sa, expect
    measured, expect = measured / measured.max(), expect / expect.max()
    bright = expect > 0.01
    np.testing.assert_allclose(measured[bright], expect[bright], rtol=0.05, atol=0.005)


def test_effective_sample_size():
    # (sum w)^2 / (n sum w^2): the share of rays that count. Near the proposal almost all; with theta_t
    # moved by the units' spread still most, where drawing directions uniformly over the cap gives ~1/4.
    src = differentiable_profile_source(POSITION, DIRECTION, PHYSICAL, 90.0, profile='physical')
    ess = lambda w: float(jnp.sum(w) ** 2 / (w.size * jnp.sum(w ** 2)))
    key = jax.random.PRNGKey(9)
    assert ess(src(100_000, key)[2]) > 0.95
    assert ess(src._replace(params=_physical(PHYSICAL[1] + 4.0))(100_000, key)[2]) > 0.8
    assert ess(src._replace(params=_physical(PHYSICAL[1] - 4.0))(100_000, key)[2]) > 0.8


def test_gradient_through_weights_matches_finite_differences():
    # A detector-like observable: weighted light inside a 20 deg cone of an off-axis direction.
    target = jnp.array([0.2, 0.9, -0.4]); target = target / jnp.linalg.norm(target)
    key = jax.random.PRNGKey(5)
    p0 = jnp.array([38.58, 1.216, 44.78, 2.378])
    built = differentiable_profile_source(POSITION, DIRECTION, tuple(float(v) for v in p0), 50.0)

    def observable(params):
        d, _, w = built._replace(params=built.params.at[:4].set(params))(20_000, key)
        inside = jax.nn.sigmoid((d @ target - jnp.cos(jnp.radians(20.0))) * 2000.0)
        return jnp.sum(w * inside)

    g = jax.grad(observable)(p0)
    for i, h in enumerate((0.05, 0.01, 0.05, 0.02)):
        e = jnp.zeros(4).at[i].set(h)
        fd = (observable(p0 + e) - observable(p0 - e)) / (2 * h)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=2e-2, atol=1e-3 * abs(float(fd)) + 1e-6)
    assert np.all(np.isfinite(np.asarray(g)))


def _physical_numpy(alpha_deg, t, n_w=1.34, n_g=1.50, da=0.05):
    """The notebook's baseline (profile_table_map.ipynb in the HKUKLI project) in numpy, with its variable kernel."""
    sig = lambda z: 0.5 * (1.0 + np.tanh(0.5 * z))
    def fresnel_T(ci, n1, n2):
        s2t = (n1 / n2) ** 2 * (1 - ci ** 2)
        ct = np.sqrt(np.clip(1 - s2t, 0, 1))
        rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct + 1e-12)) ** 2
        rp = ((n2 * ci - n1 * ct) / (n2 * ci + n1 * ct + 1e-12)) ** 2
        return np.where(s2t < 1, 1 - 0.5 * (rs + rp), 0.0)
    m, th_t, w_t, logB, lam_t, a_c, s_c, d_par, sig_g = t
    a = np.arange(0.0, 90.0 + da / 2, da)
    sa = n_w * np.sin(np.radians(a))
    ct = np.sqrt(np.clip(1 - sa ** 2, 1e-9, 1))
    c_glass = np.sqrt(np.clip(1 - (sa / n_g) ** 2, 0, 1))
    T = np.where(sa < 1, fresnel_T(ct, 1.0, n_g) * fresnel_T(c_glass, n_g, n_w), 0.0)
    jac = np.where(sa < 1, n_w ** 2 * np.cos(np.radians(a)) / ct, 0.0)
    theta = np.degrees(np.arcsin(np.clip(sa, 0, 1)))
    cth = np.cos(np.radians(theta))
    y = (1 + cth) / 2 * cth ** m * sig((th_t - theta) / w_t) * T * jac
    a_max = np.degrees(np.arcsin(1 / n_w))
    y = y + (np.exp(logB) * sig((a - a_max) / 0.3) * np.exp(-(a - a_max).clip(0) / lam_t) * sig((a_c - a) / s_c))
    h = int(np.ceil((d_par + 4 * sig_g) / da)) + 1
    x = np.arange(-h, h + 1) * da
    k = np.sqrt(np.clip(1 - (x / max(d_par, 1e-6)) ** 2, 0, None)) if d_par > da else (x == 0).astype(float)
    if sig_g > da:
        k = np.convolve(k, np.exp(-0.5 * (x / sig_g) ** 2), mode='same')
    yy = np.concatenate([y[h:0:-1], y, np.full(h, y[-1])])
    return np.interp(alpha_deg, a, np.convolve(yy, k / k.sum(), mode='valid'))


@pytest.mark.parametrize('t', [PHYSICAL, (1.9, 80.6, 4.9, float(np.log(0.0136)), 20.5, 79.3, 0.48, 1.8, 0.04),
                               (0.7, 54.0, 12.0, float(np.log(0.014)), 21.0, 78.7, 1.7, 0.0, 0.0)])
def test_physical_matches_the_notebook_model(t):
    # The fixed-support kernel agrees with the notebook's variable one to the Gaussian's far tails.
    a = np.linspace(0.0, 89.0, 400)
    pp = jnp.zeros(N_PROFILE_PARAMS).at[:9].set(jnp.asarray(t, dtype=jnp.float32))
    got = np.asarray(profile_intensity(jnp.asarray(a, dtype=jnp.float32), pp, PROFILE_FORMS.index('physical')))
    want = _physical_numpy(a, t)
    np.testing.assert_allclose(got / want[0], want / want[0], rtol=2e-3, atol=2e-6)


def test_physical_gradient_in_theta_t_matches_finite_differences():
    a = jnp.linspace(0.0, 85.0, 200)
    w = jnp.sin(jnp.radians(a)) * jnp.linspace(1.0, 2.0, 200)
    pid = PROFILE_FORMS.index('physical')
    def obs(th_t):
        p = jnp.zeros(N_PROFILE_PARAMS).at[:9].set(jnp.asarray(PHYSICAL, dtype=jnp.float32)).at[1].set(th_t)
        prof = profile_intensity(a, p, pid)
        return jnp.sum(w * prof) / jnp.sum(prof)
    g = float(jax.grad(obs)(67.0))
    fd = float((obs(67.5) - obs(66.5)) / 1.0)
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=2e-2)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        differentiable_profile_source(POSITION, DIRECTION, (1.0,), 10.0, profile='gaussian')
