"""Wavelength-dependent physics for LUCiD.

Submodules:
    medium        — MediumProperties; load_medium (a material's curves) and medium_on_grid
                    (sampled on a wavelength grid), which the simulator setup calls, and
                    make_medium, their composition; QECurve, load_qe_curve
    optical_model — evaluate_optical_model: per-photon λ → optical-property seam
    spectrum      — Cherenkov wavelength sampling
    scattering    — Rayleigh and Mie/HG phase function samplers
"""

from lucid.wavelength.medium import (
    MediumProperties,
    group_velocity_m_per_ns,
    load_qe_curve,
    make_medium,
    sk4_water_group_refractive_index,
    sk4_water_refractive_index,
    sk4_water_refractive_index_derivative,
)
from lucid.wavelength.optical_model import OpticalArrays, evaluate_optical_model
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths

# Default wavelength (nm) for padding and fallback — near Cherenkov peak, mid-grid.
DEFAULT_WAVELENGTH_NM = 400.0

__all__ = [
    "DEFAULT_WAVELENGTH_NM",
    "MediumProperties",
    "group_velocity_m_per_ns",
    "sk4_water_group_refractive_index",
    "sk4_water_refractive_index",
    "sk4_water_refractive_index_derivative",
    "load_qe_curve",
    "make_medium",
    "OpticalArrays",
    "evaluate_optical_model",
    "sample_cherenkov_wavelengths",
]
