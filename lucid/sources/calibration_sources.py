"""Calibration source ray generators (isotropic, laser, etc.).

Moved from lucid/generate.py and lucid/detector_params.py during Phase 2.2 refactor.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple
from functools import partial
from lucid.utils import normalize, generate_orthonormal_basis, jax_rotate_vector
from lucid.sources.load_HKLI_profile import build_profile_cdf, build_fit_cdf


@partial(jax.jit, static_argnums=(2,))
def get_isotropic_rays(source_position, source_intensity, Nphot, key):
    """
    Fibonacci spiral using scan-based recurrence to avoid float32 precision issues.
    """
    # Block configuration
    block_size = 1000
    n_full_blocks = Nphot // block_size
    remainder = Nphot % block_size

    golden_ratio_inverse = jnp.float32(0.6180339887498949)
    block_increment = (block_size * golden_ratio_inverse) % 1.0

    # Compute block offsets using scan
    def scan_fn(carry, x):
        next_offset = (carry + block_increment) % 1.0
        return next_offset, carry

    _, block_offsets = jax.lax.scan(scan_fn, 0.0, None, length=n_full_blocks + (1 if remainder else 0))

    # Generate all points
    all_indices = jnp.arange(Nphot, dtype=jnp.float32)
    block_idx = all_indices // block_size
    local_idx = all_indices % block_size

    # Compute theta using block offsets
    local_t = (local_idx * golden_ratio_inverse) % 1.0
    block_offset = block_offsets[block_idx.astype(jnp.int32)]
    t = (local_t + block_offset) % 1.0
    theta = 2.0 * jnp.pi * t

    # Compute z and r
    z = 1.0 - 2.0 * (all_indices + 0.5) / Nphot
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))

    # Convert to Cartesian
    x = jnp.cos(theta) * r
    y = jnp.sin(theta) * r

    ray_vectors = jnp.stack([x, y, z], axis=1)
    ray_origins = jnp.tile(source_position, (Nphot, 1))
    photon_weights = jnp.ones(Nphot) * (source_intensity / Nphot)

    return ray_vectors, ray_origins, photon_weights

@partial(jax.jit, static_argnums=(2,))
def get_isotropic_rays_random(source_position, source_intensity, Nphot, key):
    """
    Generate photons isotropically from a point source using random sampling.
    Uses 3 normal samples + normalization for uniform sphere distribution.
    """
    v = random.normal(key, (Nphot, 3))
    ray_vectors = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    ray_origins = jnp.tile(source_position, (Nphot, 1))
    photon_weights = jnp.ones(Nphot) * (source_intensity / Nphot)

    return ray_vectors, ray_origins, photon_weights


@partial(jax.jit, static_argnums=(3,))
def generate_laser_photons(fiber_position, fiber_direction, source_intensity, n_photons, key, n_water=1.33, fiber_NA=0.22):
    """
    Generate laser photons from a fiber tip with realistic angular distribution.
    Photons are distributed WITHIN the cone, not just on its surface.

    Parameters
    ----------
    fiber_position : jnp.ndarray
        3D position of fiber tip (where photons originate)
    fiber_direction : jnp.ndarray
        3D unit vector of fiber pointing direction
    source_intensity : float
        Total intensity of the laser source
    n_photons : int
        Number of photons to generate
    key : jax.random.PRNGKey
        Random key for JAX
    n_water : float, optional
        Refractive index of water, default 1.33
    fiber_NA : float
        Numerical aperture of the fiber, default 0.22

    Returns
    -------
    ray_vectors : jnp.ndarray
        Array of shape (n_photons, 3) containing photon direction vectors
    ray_origins : jnp.ndarray
        Array of shape (n_photons, 3) containing photon origins (all at fiber tip)
    photon_weights : jnp.ndarray
        Array of shape (n_photons,) containing photon weights (uniform)
    """
    # Normalize fiber direction
    fiber_direction = normalize(fiber_direction)

    # Calculate maximum emission angle in water from numerical aperture
    theta_max = jnp.arcsin(fiber_NA / n_water)

    # Split keys for different random samples
    key1, key2 = jax.random.split(key)

    # Sample angles uniformly in solid angle (not uniform in theta!)
    # This gives the correct sin(theta) weighting for angles WITHIN the cone
    u = jax.random.uniform(key1, (n_photons,))
    theta = jnp.arcsin(jnp.sqrt(u) * jnp.sin(theta_max))

    # Sample azimuthal angles uniformly around fiber axis
    phi = jax.random.uniform(key2, (n_photons,)) * 2 * jnp.pi

    # Generate directions in local fiber coordinate system
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    local_x = sin_theta * jnp.cos(phi)
    local_y = sin_theta * jnp.sin(phi)
    local_z = cos_theta  # Along fiber axis

    # Build orthonormal basis with fiber_direction as the z-axis
    basis = generate_orthonormal_basis(fiber_direction)

    # Stack local directions
    local_directions = jnp.stack([local_x, local_y, local_z], axis=1)

    # Transform to global coordinates
    ray_vectors = jnp.einsum('ij,kj->ki', basis, local_directions)

    # All photons originate from the fiber tip position
    ray_origins = jnp.tile(fiber_position[None, :], (n_photons, 1))

    # Uniform weights
    photon_weights = source_intensity * jnp.ones(n_photons) / n_photons

    return ray_vectors, ray_origins, photon_weights


def setup_calibration_generator(source_type='isotropic'):
    """
    Factory function that returns a configured calibration photon generator.

    Parameters
    ----------
    source_type : str
        Type of calibration source: 'isotropic', 'isotropic_random', or 'laser'

    Returns
    -------
    callable
        Generator function with signature:
        (source_origin, source_intensity, Nphot, key) -> (directions, origins, weights)
    """

    if source_type == 'isotropic':
        def generator(source_origin, source_intensity, Nphot, key):
            return get_isotropic_rays(source_origin, source_intensity, Nphot, key)
        return generator

    elif source_type == 'isotropic_random':
        def generator(source_origin, source_intensity, Nphot, key):
            return get_isotropic_rays_random(source_origin, source_intensity, Nphot, key)
        return generator

    elif source_type == 'laser':
        def generator(source_origin, source_intensity, Nphot, key):
            direction = jnp.array([0., 0., -1.])
            return generate_laser_photons(
                source_origin, direction, source_intensity, Nphot, key, fiber_NA=0.22
            )
        return generator

    else:
        raise ValueError(f"Unknown source_type: {source_type}. Available: 'isotropic', 'isotropic_random', 'laser'")


def generate_random_direction(key):
    """
    Generate a random direction uniformly distributed on a unit sphere.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX

    Returns
    -------
    jnp.ndarray
        Normalized 3D vector representing a random direction
    """
    key, subkey = jax.random.split(key)
    # Generate random points on a sphere using the Marsaglia method
    while True:
        # Generate two random numbers between -1 and 1
        u1, u2 = jax.random.uniform(subkey, shape=(2,), minval=-1.0, maxval=1.0)
        s = u1**2 + u2**2
        # Reject if s is outside the unit circle
        if s < 1.0:
            break
        key, subkey = jax.random.split(key)

    # Convert to Cartesian coordinates
    x = 2 * u1 * jnp.sqrt(1 - s)
    y = 2 * u2 * jnp.sqrt(1 - s)
    z = 1 - 2 * s

    # Return normalized vector
    return normalize(jnp.array([x, y, z]))

def generate_random_vertex(key):
    """
    Generate a random vertex within the volume [-1,1]^3.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX

    Returns
    -------
    jnp.ndarray
        3D point within the volume [-1,1]^3
    """
    return jax.random.uniform(key, shape=(3,), minval=-0.1, maxval=0.1)


# ---------------------------------------------------------------------------
# Calibration source types (callable NamedTuples)
# Moved from lucid/detector_params.py during Phase 2.2 refactor.
# ---------------------------------------------------------------------------

class IsotropicSource(NamedTuple):
    """Isotropic point source -- callable JAX pytree.

    Usage: ``source(n_photons, key)`` or ``source(n_photons, key, n_water)``.

    The optional ``wavelength`` field (nm) is used by the simulator in
    wavelength mode to look up per-photon scatter/absorption from the medium.
    """
    position: jnp.ndarray     # (3,)
    intensity: jnp.ndarray    # scalar
    wavelength: object = None # scalar nm, or None for broadband

    def __call__(self, n_photons, key, n_water=1.33):
        return get_isotropic_rays(self.position, self.intensity, n_photons, key)


class IsotropicSourceRandom(NamedTuple):
    """Isotropic point source with random (key-dependent) emission.

    Same fields and call signature as :class:`IsotropicSource`, but emits via
    :func:`get_isotropic_rays_random`, so different PRNG keys give different photon sets.

    ``IsotropicSource`` emits on a Fibonacci lattice that ignores its key (directions depend
    on ``n_photons`` alone). That suits a low-variance forward model but not truth data meant
    to be an independent realisation: the direct-light component of the residual would be
    zero by construction rather than by fit quality. Use this for truth generation; keep the
    lattice for the forward model.
    """
    position: jnp.ndarray     # (3,)
    intensity: jnp.ndarray    # scalar
    wavelength: object = None # scalar nm, or None for broadband

    def __call__(self, n_photons, key, n_water=1.33):
        return get_isotropic_rays_random(self.position, self.intensity, n_photons, key)


class LaserSource(NamedTuple):
    """Laser fibre source -- callable JAX pytree.

    Usage: ``source(n_photons, key)`` or ``source(n_photons, key, n_water)``.

    The optional ``wavelength`` field (nm) is used by the simulator in
    wavelength mode to look up per-photon scatter/absorption from the medium.
    """
    position: jnp.ndarray     # (3,)
    intensity: jnp.ndarray    # scalar
    direction: jnp.ndarray    # (3,), default [0, 0, -1]
    fiber_NA: jnp.ndarray     # scalar, default 0.22
    wavelength: object = None # scalar nm, or None for broadband

    def __call__(self, n_photons, key, n_water=1.33):
        return generate_laser_photons(
            self.position, self.direction, self.intensity,
            n_photons, key, n_water, self.fiber_NA,
        )


# --- Factory helpers with sensible defaults ---

def isotropic_source(position, intensity=1_000_000, wavelength=None):
    """Create an IsotropicSource with default intensity.

    Parameters
    ----------
    wavelength : float or None
        Source wavelength in nm. When set and wavelength_mode=True in the
        simulator, per-photon scatter/absorption are looked up from the
        medium at this wavelength.
    """
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None
    return IsotropicSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        wavelength=wl,
    )


def isotropic_source_random(position, intensity=1_000_000, wavelength=None):
    """Create an :class:`IsotropicSourceRandom` (isotropic, key-dependent emission).

    Same arguments as :func:`isotropic_source`. Use it for truth data that must be an
    independent realisation; :func:`isotropic_source` emits on a deterministic lattice and
    ignores its key (see :class:`IsotropicSourceRandom`).
    """
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None
    return IsotropicSourceRandom(
        position=jnp.asarray(position, dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        wavelength=wl,
    )


def laser_source(position, intensity=1_000_000, direction=None, fiber_NA=0.22,
                 wavelength=None):
    """Create a LaserSource with default direction (downward) and NA.

    Parameters
    ----------
    wavelength : float or None
        Laser wavelength in nm (e.g. 405.0). When set and wavelength_mode=True
        in the simulator, per-photon scatter/absorption are looked up from the
        medium at this wavelength.
    """
    if direction is None:
        direction = [0.0, 0.0, -1.0]
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None
    return LaserSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        direction=jnp.asarray(direction, dtype=jnp.float32),
        fiber_NA=jnp.asarray(float(fiber_NA), dtype=jnp.float32),
        wavelength=wl,
    )


# ---------------------------------------------------------------------------
# Measured-profile source, ported from HKUKLI/measured_profile_source_job.py.
# ---------------------------------------------------------------------------

def _axis_angle_from_x_axis(direction):
    direction = normalize(direction)
    u_ref = jnp.array([1.0, 0.0, 0.0])
    angle = jnp.arccos(jnp.clip(jnp.dot(u_ref, direction), -1.0, 1.0))
    axis = jnp.cross(u_ref, direction)
    axis_norm = jnp.linalg.norm(axis)
    axis = jnp.where(axis_norm > 1e-8, axis / jnp.where(axis_norm > 1e-8, axis_norm, 1.0),
                      jnp.array([0.0, 1.0, 0.0]))
    return axis, angle


def _sample_u_phi(key, fine_u, marginal_cdf, fine_phi, cond, n_photons):
    key_u, key_phi = random.split(key)
    q1 = random.uniform(key_u, (n_photons,))
    q2 = random.uniform(key_phi, (n_photons,))
    u_s = jnp.interp(q1, marginal_cdf, fine_u)
    row_idx = jnp.clip(jnp.searchsorted(fine_u, u_s), 0, fine_u.shape[0] - 1)
    cond_rows = cond[row_idx]
    phi_s = jax.vmap(lambda row, q: jnp.interp(q, row, fine_phi))(cond_rows, q2)
    return u_s, phi_s


@partial(jax.jit, static_argnums=(9,))
def sample_measured_profile_rays(position, rotation_axis, rotation_angle,
                                  fine_u, marginal_cdf, fine_phi, cond,
                                  is_collimator, intensity, n_photons, key):
    u_s, phi_s = _sample_u_phi(key, fine_u, marginal_cdf, fine_phi, cond, n_photons)
    polar_s = jnp.arccos(jnp.clip(u_s, -1.0, 1.0))
    phi_rad = jnp.radians(phi_s)

    d_collimator = jnp.stack([
        jnp.sin(polar_s) * jnp.cos(phi_rad),
        jnp.sin(polar_s) * jnp.sin(phi_rad),
        jnp.cos(polar_s),
    ], axis=-1)
    d_diffuser = jnp.stack([
        jnp.cos(polar_s),
        jnp.sin(polar_s) * jnp.cos(phi_rad),
        jnp.sin(polar_s) * jnp.sin(phi_rad),
    ], axis=-1)
    d_native = jnp.where(is_collimator, d_collimator, d_diffuser)

    ray_vectors = jax.vmap(lambda v: jax_rotate_vector(v, rotation_axis, rotation_angle))(d_native)
    ray_origins = jnp.tile(position[None, :], (n_photons, 1))
    photon_weights = jnp.full((n_photons,), intensity / n_photons)
    return ray_vectors, ray_origins, photon_weights


class MeasuredProfileSource(NamedTuple):
    """Calibration source drawn from a real goniometer profile scan -- callable JAX pytree.

    Usage: ``source(n_photons, key)`` or ``source(n_photons, key, n_water)``.
    Built via `measured_profile_source`; see HKUKLI/injector_profile_comparison.ipynb
    and HKUKLI/measured_profile_source_job.py for derivation/validation.
    """
    position: jnp.ndarray
    rotation_axis: jnp.ndarray
    rotation_angle: jnp.ndarray
    fine_u: jnp.ndarray
    marginal_cdf: jnp.ndarray
    fine_phi: jnp.ndarray
    cond: jnp.ndarray
    is_collimator: jnp.ndarray
    intensity: jnp.ndarray = jnp.float32(1.0)
    wavelength: object = None

    def __call__(self, n_photons, key, n_water=1.33):
        return sample_measured_profile_rays(
            self.position, self.rotation_axis, self.rotation_angle,
            self.fine_u, self.marginal_cdf, self.fine_phi, self.cond,
            self.is_collimator, self.intensity, n_photons, key,
        )


def measured_profile_source(position, direction, sample_id, common_profiles_dir,
                             intensity=1_000_000, wavelength=None):
    """Build a MeasuredProfileSource from a common_profiles/<sample_id>.npz scan."""
    fine_u, marginal_cdf, fine_phi, cond, device_type = build_profile_cdf(common_profiles_dir, sample_id)

    rotation_axis, rotation_angle = _axis_angle_from_x_axis(jnp.asarray(direction, dtype=jnp.float32))
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None

    return MeasuredProfileSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        rotation_axis=rotation_axis,
        rotation_angle=rotation_angle,
        fine_u=jnp.asarray(fine_u, dtype=jnp.float32),
        marginal_cdf=jnp.asarray(marginal_cdf, dtype=jnp.float32),
        fine_phi=jnp.asarray(fine_phi, dtype=jnp.float32),
        cond=jnp.asarray(cond, dtype=jnp.float32),
        is_collimator=jnp.asarray(device_type != 'diffuser'),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        wavelength=wl,
    )


def fitted_profile_source(position, direction, params, alpha_max,
                           intensity=1_000_000, wavelength=None):
    """Build a MeasuredProfileSource from an already-fitted composite(w,p,a0,s) model
    instead of a measured grid -- phi-uniform, so is_collimator=False (the pole-embedding
    convention alpha already assumes). params/alpha_max come from
    HKUKLI/injector_profile_comparison.ipynb's `fit_common_profile` (PyROOT/Minuit2 --
    run offline, not here).
    """
    fine_u, marginal_cdf, fine_phi, cond = build_fit_cdf(params, alpha_max)

    rotation_axis, rotation_angle = _axis_angle_from_x_axis(jnp.asarray(direction, dtype=jnp.float32))
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None

    return MeasuredProfileSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        rotation_axis=rotation_axis,
        rotation_angle=rotation_angle,
        fine_u=jnp.asarray(fine_u, dtype=jnp.float32),
        marginal_cdf=jnp.asarray(marginal_cdf, dtype=jnp.float32),
        fine_phi=jnp.asarray(fine_phi, dtype=jnp.float32),
        cond=jnp.asarray(cond, dtype=jnp.float32),
        is_collimator=jnp.asarray(False),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        wavelength=wl,
    )
