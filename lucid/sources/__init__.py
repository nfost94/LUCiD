"""Photon sources and event input.

The SIREN track emitters (``siren_rays``, exported here with the calibration and shotgun
sources), the scintillation photons (``scintillation_photons``), the cascade source
(``cascade``), the PhotonSim ROOT reader (``root_reader``), the data-event helpers
(``placement``, ``data_event``, ``interaction``) and the per-event key derivation
(``seed_utils``).
"""

__all__ = [
    "make_cherenkov_surrogate_fn",
    "make_scintillation_surrogate_fn",
    "latin_hypercube_2d",
    "predict_t0",
    "IsotropicSource",
    "IsotropicSourceRandom",
    "LaserSource",
    "GaussianLaserSource",
    "isotropic_source",
    "isotropic_source_random",
    "laser_source",
    "gaussian_laser_source",
    "get_isotropic_rays",
    "generate_laser_photons",
    "gaussian_beam_theta",
    "generate_gaussian_laser_photons",
    "MeasuredProfileSource",
    "measured_profile_source",
    "fitted_profile_source",
    "sample_measured_profile_rays",
    "DifferentiableProfileSource",
    "differentiable_profile_source",
    "sample_differentiable_profile_rays",
    "profile_intensity",
    "PROFILE_FORMS",
    "N_PROFILE_PARAMS",
    "TABLE_ANGLES_DEG",
    "set_table_nodes",
    "get_table_nodes",
    "load_injectors",
    "injector_position_direction",
    "injector_sources",
    "ShotgunSource",
    "shotgun_source",
    "stack_shotgun_sources",
]

from lucid.sources.calibration_sources import (
    N_PROFILE_PARAMS,
    PROFILE_FORMS,
    TABLE_ANGLES_DEG,
    DifferentiableProfileSource,
    GaussianLaserSource,
    IsotropicSource,
    IsotropicSourceRandom,
    LaserSource,
    MeasuredProfileSource,
    differentiable_profile_source,
    fitted_profile_source,
    gaussian_beam_theta,
    gaussian_laser_source,
    generate_gaussian_laser_photons,
    generate_laser_photons,
    get_isotropic_rays,
    get_table_nodes,
    isotropic_source,
    isotropic_source_random,
    laser_source,
    measured_profile_source,
    profile_intensity,
    sample_differentiable_profile_rays,
    sample_measured_profile_rays,
    set_table_nodes,
)
from lucid.sources.hk_injectors import (
    injector_position_direction,
    injector_sources,
    load_injectors,
)
from lucid.sources.shotgun_source import (
    ShotgunSource,
    shotgun_source,
    stack_shotgun_sources,
)
from lucid.sources.siren_rays import (
    latin_hypercube_2d,
    make_cherenkov_surrogate_fn,
    make_scintillation_surrogate_fn,
    predict_t0,
)
