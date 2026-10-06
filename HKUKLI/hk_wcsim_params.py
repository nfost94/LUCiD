"""Best-guess WCSim optical parameters at 395 nm as LUCiD DetectorParams.

Interpolated linearly in photon energy from WCSim/src/WCSimConstructMaterials.cc with
macros/tuning_parameters_hkfd.mac (rayff 0.75, abwff 1.30, bsrff 2.50, rgcff 0.32), Mie on at
mieff 1 with g from the commented-out MIE_water_const (0.4), QE from config/pmt/HK_QE.json.
"""
import jax.numpy as jnp

from lucid.detector_params import DetectorParams

# WCSim's blacksheet is groundfrontpainted (pure Lambertian) and its glass/cathode is polished
# (pure specular), which is exactly LUCiD's 'scalar' reflection model.
REFLECTION_MODEL = 'scalar'

WCSIM_395NM = dict(
    scatter_length=124.3,
    absorption_length=599.3,
    mie_scatter_length=4879.,
    g=0.4,
    wall_reflection_rate=0.1125,
    sensor_reflection_rate=0.32,
    qe=0.331,
)


# Per laser wavelength (nm), same tables and tuning, linear in photon energy as Geant4 does.
# Blacksheet reflectivity is 0.1125 up to 395 nm but 0.145 / 0.1375 at 440 / 500 nm; the
# multi-wavelength fit (CalibrationParams) shares one reflection model across wavelengths.
WCSIM_BY_WAVELENGTH = {
    337: dict(scatter_length=56.9, absorption_length=830.7, qe=0.3145),
    355: dict(scatter_length=73.8, absorption_length=853.4, qe=0.3324),
    375: dict(scatter_length=96.6, absorption_length=768.1, qe=0.3337),
    395: dict(scatter_length=124.3, absorption_length=599.3, qe=0.3311),
    440: dict(scatter_length=208.1, absorption_length=247.4, qe=0.2804),
    500: dict(scatter_length=378.6, absorption_length=62.8, qe=0.1818),
}


def wcsim_395nm_params(num_sensors, **overrides):
    return DetectorParams.from_flat(qe_corrections=jnp.ones(num_sensors),
                                    **{**WCSIM_395NM, **overrides})
