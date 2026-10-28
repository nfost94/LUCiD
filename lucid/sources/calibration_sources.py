"""Calibration source ray generators (isotropic, laser, etc.).

Moved from lucid/generate.py and lucid/detector_params.py during Phase 2.2 refactor.
"""

import jax
import jax.numpy as jnp
import numpy as np
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


def gaussian_beam_theta(key, beam_width_deg, shape=()):
    """Sample polar angles from an empirical Gaussian calibration beam.

    Twelve uniform variates provide the Gaussian approximation before the
    empirical angular transform::

        beam_width_rad = beam_width_deg * pi / 180
        rnd = (sum(12 uniforms) - 6) * beam_width_rad
        ang = sqrt(rnd**2 / 3610**2)
        ang = sqrt(ang)
        theta = atan(ang)

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX.
    beam_width_deg : float or jnp.ndarray
        Empirical beam-width input in degrees.
    shape : tuple, optional
        Shape of the returned angle array.
    """
    beam_width_rad = jnp.asarray(beam_width_deg, dtype=jnp.float32) * jnp.pi / 180.0
    uniforms = random.uniform(key, shape=shape + (12,))
    rnd = (jnp.sum(uniforms, axis=-1) - 6.0) * beam_width_rad
    ang = jnp.sqrt((rnd * rnd) / (3610.0 ** 2))
    ang = jnp.sqrt(ang)
    return jnp.arctan(ang)


@partial(jax.jit, static_argnums=(3,))
def generate_gaussian_laser_photons(
        fiber_position, fiber_direction, source_intensity, n_photons, key,
        beam_width_deg=0.5):
    """Generate laser photons with an empirical Gaussian angular profile.

    The return value follows LUCiD's calibration-source contract:
    ``(directions, origins, weights)``. Photon wavelength remains metadata on
    :class:`GaussianLaserSource`, as it is for the existing laser source.
    """
    fiber_direction = normalize(fiber_direction)
    theta_key, phi_key = random.split(key)

    theta = gaussian_beam_theta(theta_key, beam_width_deg, shape=(n_photons,))
    phi = random.uniform(phi_key, (n_photons,)) * 2.0 * jnp.pi

    sin_theta = jnp.sin(theta)
    local_directions = jnp.stack([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        jnp.cos(theta),
    ], axis=1)

    basis = generate_orthonormal_basis(fiber_direction)
    ray_vectors = jnp.einsum('ij,kj->ki', basis, local_directions)
    ray_origins = jnp.tile(fiber_position[None, :], (n_photons, 1))
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


class GaussianLaserSource(NamedTuple):
    """Laser source using an empirical Gaussian angular profile."""
    position: jnp.ndarray
    intensity: jnp.ndarray
    direction: jnp.ndarray
    beam_width_deg: jnp.ndarray
    wavelength: object = None

    def __call__(self, n_photons, key, n_water=1.33):
        del n_water  # Kept for compatibility with the calibration-source API.
        return generate_gaussian_laser_photons(
            self.position, self.direction, self.intensity,
            n_photons, key, self.beam_width_deg,
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
# Measured-profile sources: a goniometer scan, or a fitted composite, as inverse-CDF tables.
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
    Built via `measured_profile_source` (a scan) or `fitted_profile_source` (a fitted composite).
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
    convention alpha already assumes).
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


# ---------------------------------------------------------------------------
# Differentiable profile source: the profile lives in the ray weights, not the directions.
# ---------------------------------------------------------------------------
# Deposits are linear in the ray weights, so d(charge)/d(profile params) needs no gradient through
# the ray geometry. The directions are drawn from a proposal: the profile at the parameters the source
# was built with, frozen. The weights carry the ratio of the current profile to that proposal, so near
# the proposal every ray counts about equally, and fitting the parameters never moves a ray. The cap must
# contain all the emitted light: anything beyond alpha_cut is dropped.

PROFILE_FORMS = ('composite', 'super_gaussian', 'sk_beam', 'physical')   # index = DifferentiableProfileSource.profile_id
N_PROFILE_PARAMS = 9                     # params is padded to this length for the shorter forms


def _power(x, k):
    """x**k for x >= 0 with a finite derivative in k at x = 0 (0**k with k traced is NaN-prone)."""
    return jnp.exp(k * jnp.log(jnp.maximum(x, 1e-30)))


def _profile_composite(alpha_deg, params):
    # Super-Gaussian core x Fermi edge: exp(-(a/w)^(2p)) / (1 + exp((a - a0)/s)); params (w, p, a0, s).
    w, p, a0, s = params[0], params[1], params[2], params[3]
    return jnp.exp(-_power(alpha_deg / w, 2 * p)) * jax.nn.sigmoid((a0 - alpha_deg) / s)


def _profile_super_gaussian(alpha_deg, params):
    # exp(-(a/w)^(2p)); params (w, p, -, -).
    w, p = params[0], params[1]
    return jnp.exp(-_power(alpha_deg / w, 2 * p))


def _profile_sk_beam(alpha_deg, params):
    # SKDetSim RNGAUSYOO beam (gaussian_beam_theta): tan^2(alpha) = |sigma Z| / 3610 with
    # sigma = sigbm * pi/180, giving I ~ sec^3(alpha) exp(-tan^4(alpha) / (2 (sigma/3610)^2));
    # params (sigbm, -, -, -). SK-IV injector 1 uses sigbm = 3300.
    t = jnp.radians(alpha_deg)
    s = jnp.radians(params[0]) / 3610.0
    return jnp.exp(-jnp.tan(t) ** 4 / (2 * s * s)) / jnp.cos(t) ** 3


# The physical diffuser model (profile_table_map.ipynb in the HKUKLI project, its baseline): a PTFE dome in a dry
# torch behind a flat window, in water. Direct light: the dome's projected area (1 + cos t)/2 x cos^m t x a
# logistic torch eclipse (theta_t, w_t) at internal angle t = asin(n_w sin a), times the Fresnel transmission
# of the window's two faces and the solid-angle factor n_w^2 cos a / cos t; it all exits within asin(1/n_w).
# Beyond that, a tail (B, lam_t, a_c, s_c). Then the scan's bench blur: a semicircle of half-width d_par
# convolved with a Gaussian sig_g, on a fixed grid with the profile mirrored at a = 0 (the kernel's support
# is fixed at +-_PHYS_KERNEL deg so shapes stay static; it truncates only if d_par + 4 sig_g exceeds it).
_PHYS_N_WATER, _PHYS_N_GLASS = 1.34, 1.50
_PHYS_DA = 0.05
_PHYS_GRID = np.arange(0.0, 90.0 + _PHYS_DA / 2, _PHYS_DA)
_PHYS_KERNEL = 15.0
_PHYS_KX = np.arange(-int(round(_PHYS_KERNEL / _PHYS_DA)), int(round(_PHYS_KERNEL / _PHYS_DA)) + 1) * _PHYS_DA


def _fresnel_T(ci, n1, n2):
    """Unpolarised Fresnel transmittance n1 -> n2 at incidence cosine ci (0 beyond TIR)."""
    s2t = (n1 / n2) ** 2 * (1 - ci ** 2)
    ct = jnp.sqrt(jnp.clip(1 - s2t, 0, 1))
    rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct + 1e-12)) ** 2
    rp = ((n2 * ci - n1 * ct) / (n2 * ci + n1 * ct + 1e-12)) ** 2
    return jnp.where(s2t < 1, 1 - 0.5 * (rs + rp), 0.0)


def _profile_physical(alpha_deg, params):
    # params (m, theta_t, w_t, log B, lam_t, a_c, s_c, d_par, sig_g), angles in deg.
    m, th_t, w_t, logB, lam_t, a_c, s_c, d_par, sig_g = (params[i] for i in range(9))
    n_w, n_g = _PHYS_N_WATER, _PHYS_N_GLASS
    a = jnp.asarray(_PHYS_GRID)
    sa = n_w * jnp.sin(jnp.radians(a))
    inside = sa < 1
    ct = jnp.sqrt(jnp.clip(1 - sa ** 2, 1e-9, 1))
    c_glass = jnp.sqrt(jnp.clip(1 - (sa / n_g) ** 2, 0, 1))
    T = jnp.where(inside, _fresnel_T(ct, 1.0, n_g) * _fresnel_T(c_glass, n_g, n_w), 0.0)
    jac = jnp.where(inside, n_w ** 2 * jnp.cos(jnp.radians(a)) / ct, 0.0)
    theta = jnp.degrees(jnp.arcsin(jnp.clip(sa, 0, 1)))
    cth = jnp.cos(jnp.radians(theta))
    direct = (1 + cth) / 2 * _power(cth, m) * jax.nn.sigmoid((th_t - theta) / w_t) * T * jac
    a_max = np.degrees(np.arcsin(1 / n_w))
    tail = (jnp.exp(logB) * jax.nn.sigmoid((a - a_max) / 0.3) * jnp.exp(-jnp.clip(a - a_max, 0) / lam_t)
            * jax.nn.sigmoid((a_c - a) / s_c))
    y = direct + tail
    x = jnp.asarray(_PHYS_KX)
    semi = jnp.where(d_par > _PHYS_DA, jnp.sqrt(jnp.clip(1 - (x / jnp.maximum(d_par, 1e-6)) ** 2, 0, None)),
                     (x == 0).astype(y.dtype))
    gauss = jnp.exp(-0.5 * (x / jnp.maximum(sig_g, 1e-6)) ** 2)
    k = jnp.where(sig_g > _PHYS_DA, jnp.convolve(semi, gauss, mode='same'), semi)
    k = k / jnp.sum(k)
    h = len(_PHYS_KX) // 2
    yy = jnp.concatenate([y[h:0:-1], y, jnp.full(h, y[-1])])
    return jnp.interp(alpha_deg, a, jnp.convolve(yy, k, mode='valid'))


def profile_intensity(alpha_deg, params, profile_id):
    """Relative intensity per solid angle at off-axis angle ``alpha_deg`` for profile ``profile_id``
    (an index into PROFILE_FORMS). ``params`` is a length-N_PROFILE_PARAMS array, padded for the
    shorter forms."""
    return jax.lax.switch(profile_id, (_profile_composite, _profile_super_gaussian,
                                       _profile_sk_beam, _profile_physical), alpha_deg, params)


_PROPOSAL_BINS = 2048     # equal-angle bins over the cap for the proposal's CDF
_PROPOSAL_UNIFORM = 0.02  # share of the proposal spread uniformly over the cap, so every angle stays reachable


@partial(jax.jit, static_argnums=(8,))
def sample_differentiable_profile_rays(position, rotation_axis, rotation_angle, params, proposal,
                                       profile_id, alpha_cut_deg, intensity, n_photons, key):
    key_u, key_phi = random.split(key)
    # The proposal: the profile at `proposal` per solid angle, binned in alpha, mixed with a uniform share.
    # Each bin is uniform in u = cos(alpha) inside, so its density per solid angle is mass / du.
    proposal = jax.lax.stop_gradient(proposal)
    edges = jnp.linspace(0.0, alpha_cut_deg, _PROPOSAL_BINS + 1)
    u_edges = jnp.cos(jnp.radians(edges))
    du = u_edges[:-1] - u_edges[1:]
    p_mid = profile_intensity(0.5 * (edges[1:] + edges[:-1]), proposal, profile_id) * du
    mass = (1 - _PROPOSAL_UNIFORM) * p_mid / jnp.sum(p_mid) + _PROPOSAL_UNIFORM * du / jnp.sum(du)
    cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])

    # Stratified: one draw per equal slice of the proposal's CDF.
    q = (jnp.arange(n_photons) + random.uniform(key_u, (n_photons,))) / n_photons * cdf[-1]
    b = jnp.clip(jnp.searchsorted(cdf, q, side='right') - 1, 0, _PROPOSAL_BINS - 1)
    u = u_edges[b] - (q - cdf[b]) / mass[b] * du[b]
    phi = random.uniform(key_phi, (n_photons,)) * 2.0 * jnp.pi
    polar = jnp.arccos(jnp.clip(u, -1.0, 1.0))

    # Injector axis along local +x, rotated onto the direction (as the measured diffuser source).
    d_native = jnp.stack([jnp.cos(polar), jnp.sin(polar) * jnp.cos(phi),
                          jnp.sin(polar) * jnp.sin(phi)], axis=-1)
    ray_vectors = jax.vmap(lambda v: jax_rotate_vector(v, rotation_axis, rotation_angle))(d_native)
    ray_origins = jnp.tile(position[None, :], (n_photons, 1))

    # Each ray weighted by the profile over the density it was drawn from reproduces the profile;
    # normalising over the batch makes the weights sum to `intensity` exactly.
    rel = profile_intensity(jnp.degrees(polar), params, profile_id) / (mass[b] / du[b])
    photon_weights = intensity * rel / jnp.sum(rel)
    return ray_vectors, ray_origins, photon_weights


class DifferentiableProfileSource(NamedTuple):
    """Calibration source whose angular profile is differentiable in its parameters.

    Usage: ``source(n_photons, key)``. Directions are drawn over the cap ``alpha <= alpha_cut`` from
    the profile at ``proposal`` (the parameters the source was built with); the profile
    (``PROFILE_FORMS[profile_id]`` at ``params``) lives in the ray weights, as its ratio to the
    proposal. Gradients flow to ``params`` (and ``intensity``), not to the directions, so replacing
    ``params`` never moves a ray. Built via :func:`differentiable_profile_source`.
    """
    position: jnp.ndarray
    rotation_axis: jnp.ndarray
    rotation_angle: jnp.ndarray
    params: jnp.ndarray          # (N_PROFILE_PARAMS,), padded for the shorter forms
    proposal: jnp.ndarray        # (N_PROFILE_PARAMS,), the parameters the directions are drawn at
    profile_id: jnp.ndarray      # int, index into PROFILE_FORMS
    alpha_cut: jnp.ndarray       # deg, half-angle of the sampled cap
    intensity: jnp.ndarray = jnp.float32(1.0)
    wavelength: object = None

    def __call__(self, n_photons, key, n_water=1.33):
        return sample_differentiable_profile_rays(
            self.position, self.rotation_axis, self.rotation_angle, self.params, self.proposal,
            self.profile_id, self.alpha_cut, self.intensity, n_photons, key,
        )


def differentiable_profile_source(position, direction, params, alpha_cut, profile='composite',
                                  intensity=1_000_000, wavelength=None):
    """Build a :class:`DifferentiableProfileSource`.

    Parameters
    ----------
    params : sequence
        Profile parameters, angles in degrees: ``composite`` (w, p, a0, s);
        ``super_gaussian`` (w, p); ``sk_beam`` (sigbm,); ``physical`` (m, theta_t, w_t, log B, lam_t,
        a_c, s_c, d_par, sig_g). Also the proposal the directions are drawn from: build the source at
        the profile a fit will stay near.
    alpha_cut : float
        Half-angle (deg) of the sampled cap; must contain all the emitted light.
    profile : str
        One of PROFILE_FORMS.
    """
    if profile not in PROFILE_FORMS:
        raise ValueError(f'profile must be one of {PROFILE_FORMS}, got {profile!r}')
    p = jnp.zeros(N_PROFILE_PARAMS, dtype=jnp.float32).at[:len(params)].set(jnp.asarray(params, dtype=jnp.float32))
    rotation_axis, rotation_angle = _axis_angle_from_x_axis(jnp.asarray(direction, dtype=jnp.float32))
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None
    return DifferentiableProfileSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        rotation_axis=rotation_axis,
        rotation_angle=rotation_angle,
        params=p,
        proposal=p,
        profile_id=jnp.asarray(PROFILE_FORMS.index(profile), dtype=jnp.int32),
        alpha_cut=jnp.asarray(float(alpha_cut), dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        wavelength=wl,
    )


def gaussian_laser_source(position, intensity=1_000_000, direction=None,
                          beam_width_deg=0.5, wavelength=None):
    """Create a laser source with an empirical Gaussian angular profile.

    Parameters
    ----------
    beam_width_deg : float
        Empirical angular-profile width in degrees.
    wavelength : float or None
        Laser wavelength in nm.
    """
    if direction is None:
        direction = [0.0, 0.0, -1.0]
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None
    return GaussianLaserSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        direction=jnp.asarray(direction, dtype=jnp.float32),
        beam_width_deg=jnp.asarray(float(beam_width_deg), dtype=jnp.float32),
        wavelength=wl,
    )
