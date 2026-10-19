"""A sensor is the cap of a sphere standing on the wall: aperture radius r, height h.

`cap_silhouette_rows` is the closed form of the cap's silhouette along a line at incidence theta;
it must agree with the 3D test it replaces (the line meets the sphere and the top of its chord is
in front of the wall). `precompute_cap_lookup` convolves that silhouette with a Gaussian, which
conserves its area. At h == r the cap is the hemisphere and the hard step is the free sphere's.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.overlap import (cap_silhouette_rows, cap_sphere, create_overlap_prob,
                           precompute_cap_lookup)

R_AP = 0.254                                # 20-inch aperture radius, metres


def _hits_cap(r, h, theta, x, v):
    """Brute force: does the line at offset (x, v) from the sphere centre's projection meet the cap?"""
    radius, depth = cap_sphere(r, h)
    s, c = np.sin(theta), np.cos(theta)
    # axis z, line direction (s, 0, c), in-plane basis e_u = (-c, 0, s) (projected axis), e_v = y
    z_closest = -depth + x * s                  # z of the line's closest point to the centre
    half_chord = np.sqrt(np.maximum(radius ** 2 - x ** 2 - v ** 2, 0.0))
    return (x ** 2 + v ** 2 <= radius ** 2) & (z_closest + half_chord * c >= 0.0)


@pytest.mark.parametrize('h', [0.18, R_AP])
@pytest.mark.parametrize('theta_deg', [0, 20, 45, 70, 85, 90])
def test_silhouette_rows_match_the_3d_cap(h, theta_deg):
    theta = np.deg2rad(theta_deg)
    rng = np.random.default_rng(theta_deg)
    x = rng.uniform(-0.3, 0.4, 200_000)
    v = rng.uniform(-0.3, 0.3, 200_000)
    lo, hi = cap_silhouette_rows(R_AP, h, theta, v)
    rows = (np.abs(v) <= R_AP) & (x >= lo) & (x <= hi)
    assert np.array_equal(rows, _hits_cap(R_AP, h, theta, x, v))


@pytest.mark.parametrize('theta_deg', [0, 30, 60, 89])
def test_hemisphere_silhouette_is_half_disc_half_ellipse(theta_deg):
    theta = np.deg2rad(theta_deg)
    v = np.linspace(-R_AP, R_AP, 20001)
    lo, hi = cap_silhouette_rows(R_AP, R_AP, theta, v)
    area = np.trapezoid(hi - lo, v) if hasattr(np, 'trapezoid') else np.trapz(hi - lo, v)
    assert area == pytest.approx(np.pi * R_AP ** 2 * (1 + np.cos(theta)) / 2, rel=1e-4)


def test_cap_sphere_at_h_equal_r_is_exactly_the_wall_centred_sphere():
    radius, depth = cap_sphere(R_AP, R_AP)
    assert radius == R_AP and depth == 0.0
    radius, depth = cap_sphere(R_AP, 0.18)          # WCSim BoxandLine20inchHQE
    assert radius == pytest.approx(0.269211, abs=1e-6)
    assert depth == pytest.approx(0.089211, abs=1e-6)


@pytest.mark.parametrize('h', [0.18, R_AP])
def test_smeared_table_conserves_the_silhouette_area(h):
    theta, x0, step, table = precompute_cap_lookup(R_AP, h, 0.2 * R_AP)
    v = np.linspace(-R_AP, R_AP, 20001)
    for k in range(0, len(theta), 9):
        lo, hi = cap_silhouette_rows(R_AP, h, theta[k], v)
        area = np.sum(np.maximum(hi - lo, 0.0)) * (v[1] - v[0])
        # v >= 0 is stored; the v = 0 row is counted once
        total = (2 * table[k].sum() - table[k][:, 0].sum()) * step ** 2
        assert total == pytest.approx(area, rel=1e-4)


def test_hard_step_at_h_equal_r_is_the_free_spheres():
    d = jnp.linspace(0.0, 2 * R_AP, 1001)
    free = create_overlap_prob(None, R_AP)(d)
    cap = create_overlap_prob(None, R_AP, R_AP)(d, None, None, None)
    assert np.array_equal(np.asarray(free), np.asarray(cap))


@pytest.mark.parametrize('h', [0.0, -0.1, R_AP * 1.01])
def test_cap_height_outside_the_model_is_refused(h):
    with pytest.raises(ValueError):
        create_overlap_prob(None, R_AP, h)
