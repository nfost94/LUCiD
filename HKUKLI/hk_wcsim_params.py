"""HK optics at the laser wavelengths, read from LUCiD's HK_WCSIM physics config.

The HK_WCSIM detector description (config/HK_WCSIM_physics_config.json) is aligned with the
WCSim HK production: water from materials/water_wcsim.json (WCSim's tables x abwff
1.30 / rayff 0.75, no Mie), QE from pmt/HK_QE_wcsim.json (WCSim's net detection per photon
reaching the PMT), glass/cathode reflectivity 0.32 and blacksheet 0.1125. Everything here is
evaluated from those files, so a calibration fit's truth and the WCSim-aligned config cannot drift apart.

WCSim's blacksheet reflectivity is 0.1125 up to 395 nm but 0.145 / 0.1375 at 440 / 500 nm;
the config, like the multi-wavelength fit (CalibrationParams), uses one value at every
wavelength.
"""
import json
import os

import jax.numpy as jnp

from lucid.detector_params import DetectorParams
from lucid.simulation.trigger import trigger_from_block
from lucid.wavelength.medium import load_qe_curve, make_medium

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
PHYSICS_CONFIG = os.path.join(_CONFIG_DIR, 'HK_WCSIM_physics_config.json')
with open(PHYSICS_CONFIG) as _f:
    _PHYS = json.load(_f)

_MEDIUM = os.path.join(_CONFIG_DIR, _PHYS['medium_model'])
_QE = load_qe_curve(os.path.join(_CONFIG_DIR, _PHYS['qe_curve']))

# The reflection the fit holds or fits, as the config sets it: rates and specular fractions.
REFLECTION = {k: float(_PHYS[k]) for k in
              ('wall_reflection_rate', 'wall_fspec', 'sensor_reflection_rate', 'sensor_fspec')}

# The deprecated single-wavelength scripts pass this to setup_event_simulator.
REFLECTION_MODEL = 'scalar'

# The readout, as production digitizes and triggers: the digitizer block, and the trigger's
# readout window (the time over which dark noise accumulates in an event).
DIGITIZER = _PHYS['digitizer']
_TRIGGER = trigger_from_block(_PHYS['trigger'])
READOUT_NS = float(_TRIGGER.pad_before_ns + _TRIGGER.pad_after_ns)

LASER_WAVELENGTHS = (337, 355, 375, 395, 440, 500)


def optics_at(wavelength_nm):
    """Scattering and absorption lengths (m) and QE at one wavelength, from the config's files."""
    m = make_medium('water', wavelength_grid=jnp.array([float(wavelength_nm)]),
                    medium_model_path=_MEDIUM)
    return dict(scatter_length=float(1.0 / m.scatter_coeff[0]),
                absorption_length=float(1.0 / m.absorption_coeff[0]),
                qe=float(_QE(float(wavelength_nm))))


WCSIM_BY_WAVELENGTH = {w: optics_at(w) for w in LASER_WAVELENGTHS}

_m395 = make_medium('water', wavelength_grid=jnp.array([395.0]), medium_model_path=_MEDIUM)
WCSIM_395NM = dict(
    **WCSIM_BY_WAVELENGTH[395],
    mie_scatter_length=float(1.0 / _m395.mie_scatter_coeff[0]),
    g=float(_m395.mie_asymmetry),
    wall_reflection_rate=REFLECTION['wall_reflection_rate'],
    sensor_reflection_rate=REFLECTION['sensor_reflection_rate'],
)


def wcsim_395nm_params(num_sensors, **overrides):
    return DetectorParams.from_flat(qe_corrections=jnp.ones(num_sensors),
                                    **{**WCSIM_395NM, **overrides})
