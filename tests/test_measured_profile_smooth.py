"""The diffuser's measured source carries its polar profile between the scan's degrees by a smooth interpolation (smooth_interp_u), not by straight lines."""

import numpy as np
import pytest

from lucid.sources.load_HKLI_profile import build_profile_cdf, smooth_interp_u, N_FINE_U, POLAR_FLOOR

ALPHA = np.arange(0.0, 91.0)                      # the scan's whole degrees, deg
U_NODES = np.cos(np.radians(ALPHA))[::-1]         # increasing
THETA = np.arange(0.0, 181.0)                     # the goniometer angle, deg; alpha = |theta - 90|
PHI = np.arange(0.0, 360.0, 2.0)                  # the device's roll, deg


def _profile(alpha):
    """A diffuser-like profile: a flat core, an edge about 3 deg wide at 48 deg, a dim floor."""
    return 1.0 / (1.0 + np.exp((np.asarray(alpha) - 48.0) / 3.0)) + 0.01


def test_smooth_interp_returns_the_nodes_and_never_overshoots():
    v = _profile(ALPHA)[::-1]
    np.testing.assert_allclose(smooth_interp_u(U_NODES, v, U_NODES), v, rtol=1e-9)
    out = smooth_interp_u(U_NODES, v, np.linspace(0.0, 1.0, 20_001))
    assert np.all(np.diff(out) >= -1e-12)                                   # the profile falls with the angle: rises with u, nowhere a wiggle
    assert out.min() >= v.min() * (1 - 1e-9) and out.max() <= v.max() * (1 + 1e-9)


def test_smooth_interp_survives_noise_below_the_floor():
    # The scans fall to the noise floor, even below zero, beyond about 83 deg: no NaN, nothing above the data's largest value or below the floor.
    rng = np.random.default_rng(0)
    v = _profile(ALPHA)
    v[84:] = rng.normal(0.0, 1e-4, v[84:].size)
    out = smooth_interp_u(U_NODES, v[::-1], np.linspace(0.0, 1.0, 5001))
    assert np.all(np.isfinite(out))
    assert out.min() >= POLAR_FLOOR * v.max() * (1 - 1e-9) and out.max() <= v.max() * (1 + 1e-9)


def test_smooth_interp_is_closer_to_a_smooth_profile_than_straight_lines():
    # Half way between the degrees, against the smooth profile the scan was sampled from: the edge falls by a factor of about 1.4 a degree, where a chord of v is too high.
    mid = ALPHA[:-1] + 0.5
    u_mid = np.cos(np.radians(mid))
    truth = _profile(mid)
    smooth = smooth_interp_u(U_NODES, _profile(ALPHA)[::-1], u_mid)
    straight = np.interp(u_mid, U_NODES, _profile(ALPHA)[::-1])
    assert np.max(np.abs(smooth / truth - 1.0)) < np.max(np.abs(straight / truth - 1.0))


def test_the_measured_diffusers_polar_density_is_the_smooth_profile(tmp_path):
    # The marginal CDF's density, bin by bin, against smooth_interp_u through the roll-averaged scan; the roll structure (a dipole here) does not change it.
    grid = _profile(np.abs(THETA - 90.0))[None, :] * (1.0 + 0.5 * np.cos(np.radians(PHI)))[:, None]
    np.savez(tmp_path / 'synthetic.npz', axis1_vals=THETA, axis2_vals=PHI, grid=grid, device_type='diffuser')
    fine_u, cdf, fine_phi, cond, device = build_profile_cdf(tmp_path, 'synthetic')
    assert device == 'diffuser' and len(fine_u) == N_FINE_U and len(cdf) == N_FINE_U
    density = np.diff(cdf) / np.diff(fine_u)
    expect = smooth_interp_u(U_NODES, _profile(ALPHA)[::-1], 0.5 * (fine_u[1:] + fine_u[:-1]))
    np.testing.assert_allclose(density / density.max(), expect / expect.max(), rtol=5e-3, atol=1e-6)
