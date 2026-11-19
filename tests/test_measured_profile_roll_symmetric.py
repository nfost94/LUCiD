"""measured_profile_source(roll_symmetric=True): the azimuth drawn uniformly, the polar angle still from the scan's marginal."""

import jax
import numpy as np
import pytest

from lucid.sources import measured_profile_source

THETA = np.arange(0.0, 181.0)                     # the goniometer angle, deg; alpha = |theta - 90|
PHI = np.arange(0.0, 360.0, 2.0)                  # the device's roll, deg
BASE = np.exp(-(np.abs(THETA - 90.0) / 30.0) ** 2) + 0.01
MODULATION = 0.5                                  # a dipole in the roll: I(alpha, phi) = base(alpha) (1 + 0.5 cos phi)


@pytest.fixture
def scan_dir(tmp_path):
    grid = BASE[None, :] * (1.0 + MODULATION * np.cos(np.radians(PHI)))[:, None]
    np.savez(tmp_path / 'synthetic.npz', axis1_vals=THETA, axis2_vals=PHI, grid=grid, device_type='diffuser')
    return tmp_path


def _rays(scan_dir, roll_symmetric):
    src = measured_profile_source(np.zeros(3), np.array([1.0, 0.0, 0.0]), 'synthetic', scan_dir, intensity=1.0,
                                  roll_symmetric=roll_symmetric)
    d, _, _ = src(200_000, jax.random.PRNGKey(0))
    return np.asarray(d)


def test_the_rolls_structure_is_kept_by_default(scan_dir):
    d = _rays(scan_dir, False)
    phi = np.arctan2(d[:, 2], d[:, 1])
    assert np.mean(np.cos(phi)) == pytest.approx(MODULATION / 2.0, abs=0.02)      # E[cos phi] under 1 + m cos phi is m / 2


def test_roll_symmetric_draws_the_azimuth_uniformly(scan_dir):
    d = _rays(scan_dir, True)
    phi = np.arctan2(d[:, 2], d[:, 1])
    assert abs(np.mean(np.cos(phi))) < 0.02
    assert abs(np.mean(np.sin(phi))) < 0.02


def test_roll_symmetric_keeps_the_polar_marginal(scan_dir):
    # The marginal CDF is the mean over the roll sweeps either way, and the polar draw uses the same key.
    np.testing.assert_allclose(_rays(scan_dir, True)[:, 0], _rays(scan_dir, False)[:, 0], atol=1e-5)


def test_roll_symmetric_is_for_diffusers(tmp_path):
    theta, phi = np.arange(0.0, 31.0), np.arange(-30.0, 31.0, 5.0)
    np.savez(tmp_path / 'col.npz', axis1_vals=theta, axis2_vals=phi, grid=np.ones((len(phi), len(theta))), device_type='collimator')
    with pytest.raises(ValueError):
        measured_profile_source(np.zeros(3), np.array([1.0, 0.0, 0.0]), 'col', tmp_path, roll_symmetric=True)
