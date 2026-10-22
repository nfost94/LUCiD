"""DifferentiableProfileSource with uniform-cap sampling: the profile carried in the ray weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.sources import differentiable_profile_source, profile_intensity, PROFILE_FORMS

DIRECTION = jnp.array([1.0, 2.0, -3.0]) / jnp.sqrt(14.0)
POSITION = jnp.array([0.5, -1.0, 2.0])
# Representative shapes: a collimator-like and a diffuser-like composite, a super-Gaussian, SK-IV's beam.
CASES = [
    ('composite', (2.676, 7.469, 2.705, 0.0578), 3.5),
    ('composite', (38.58, 1.216, 44.78, 2.378), 50.0),
    ('super_gaussian', (36.65, 1.646), 60.0),
    ('sk_beam', (3300.0,), 16.0),
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
    pp = jnp.zeros(4).at[:len(params)].set(jnp.asarray(params))
    expect = np.asarray(profile_intensity(jnp.asarray(centres), pp, PROFILE_FORMS.index(profile)))
    measured, expect = h / sa, expect
    measured, expect = measured / measured.max(), expect / expect.max()
    bright = expect > 0.05
    np.testing.assert_allclose(measured[bright], expect[bright], rtol=0.06, atol=0.01)


def test_directions_do_not_depend_on_params():
    a = differentiable_profile_source(POSITION, DIRECTION, (38.0, 1.2, 44.0, 2.4), 50.0)
    b = differentiable_profile_source(POSITION, DIRECTION, (20.0, 3.0, 30.0, 1.0), 50.0)
    key = jax.random.PRNGKey(3)
    np.testing.assert_array_equal(a(1024, key)[0], b(1024, key)[0])


def test_gradient_through_weights_matches_finite_differences():
    # A detector-like observable: weighted light inside a 20 deg cone of an off-axis direction.
    target = jnp.array([0.2, 0.9, -0.4]); target = target / jnp.linalg.norm(target)
    key = jax.random.PRNGKey(5)

    def observable(params):
        src = differentiable_profile_source(POSITION, DIRECTION, (1.0, 1.0, 1.0, 1.0), 50.0)
        src = src._replace(params=params)
        d, _, w = src(20_000, key)
        inside = jax.nn.sigmoid((d @ target - jnp.cos(jnp.radians(20.0))) * 2000.0)
        return jnp.sum(w * inside)

    p0 = jnp.array([38.58, 1.216, 44.78, 2.378])
    g = jax.grad(observable)(p0)
    for i, h in enumerate((0.05, 0.01, 0.05, 0.02)):
        e = jnp.zeros(4).at[i].set(h)
        fd = (observable(p0 + e) - observable(p0 - e)) / (2 * h)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=2e-2, atol=1e-3 * abs(float(fd)) + 1e-6)
    assert np.all(np.isfinite(np.asarray(g)))


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        differentiable_profile_source(POSITION, DIRECTION, (1.0,), 10.0, profile='gaussian')
