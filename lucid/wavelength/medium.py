"""Medium optical properties (material physics, NOT detector hardware).

MediumProperties holds the physical properties of the detection medium
(water, ice): refractive index, wavelength-dependent scattering and
absorption coefficients. QE is a detector property and lives elsewhere.
"""

import json
import os
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class MediumProperties(NamedTuple):
    """Optical properties of the detection medium.

    Scalar fields are always populated.  Wavelength-dependent arrays are
    ``None`` until :func:`medium_on_grid` evaluates ``model`` on a grid.
    """

    material: str
    refractive_index: float
    speed_of_light: float  # m/ns in this medium

    # Wavelength-dependent arrays (None = monochromatic mode)
    wavelength_grid: Optional[jnp.ndarray] = None  # (N,) nm
    scatter_coeff: Optional[jnp.ndarray] = None  # (N,) symmetric Rayleigh 1/m
    absorption_coeff: Optional[jnp.ndarray] = None  # (N,) 1/m
    refractive_index_curve: Optional[jnp.ndarray] = None  # (N,) n(lambda)
    mie_scatter_coeff: Optional[jnp.ndarray] = None  # (N,) asymmetric Mie 1/m
    mie_asymmetry: Optional[float] = None  # HG g parameter

    # --- Emission dispatch (static, non-differentiable; the multi-material merge) ---
    # Differentiable scintillation values (S/kB/C/tau) live on DetectorParams; these are
    # the setup-time knobs the simulator reads to pick Cherenkov vs scintillation surrogates.
    emission_processes: tuple = ("cherenkov",)  # ("cherenkov",) | ("cherenkov","scintillation")
    scintillation_lambda_min: float = 340.0  # Moyal sampling window (nm)
    scintillation_lambda_max: float = 550.0
    cherenkov_fraction: float = 0.5  # n_photons split when both processes run
    moyal_loc: Optional[np.float32] = None  # Moyal emission-spectrum shape (nm);
    moyal_scale: Optional[np.float32] = None  # scintillating media only

    model: Optional[dict] = None  # the parsed material JSON

    # Dispersion is setup-time/static; the curve is a diagnostic sample of the
    # analytic model used by ``group_velocity_m_per_ns``.
    dispersion_model: str = "constant"
    group_refractive_index_curve: Optional[jnp.ndarray] = None


# ---------------------------------------------------------------------------
# Refractive-index dispersion and group velocity
# ---------------------------------------------------------------------------

# SK-like water dispersion at 13 C and 1.01325 bar. The phase index n
# determines the group index n_g = n - lambda * dn/dlambda used for optical
# time-of-flight updates.
_SK4_TEMP_C = 13.0
_SK4_PRESSURE_BAR = 1.01325
_SK4_TEMP_REF_C = 19.993
_SK4_PRESSURE_REF_BAR = 1.01325
_SK4_WLA_UM2 = 0.018085
_SK4_A = (5.743534e-3, 1.769238, -2.797222e-2, 8.715348e-3, -1.413942e-3)
_SK4_B = (
    -8.454823e-5,
    -2.787742e-5,
    2.608176e-6,
    -2.050671e-6,
    1.019989e-6,
    -2.611919e-6,
    8.194989e-9,
    -8.10707e-9,
    4.877274e-8,
)
_SK4_C = (8.419632e-6, 1.941681e-5, -7.762524e-8, 4.371257e-8, 7.089664e-9, -2.240384e-8)


def sk4_water_refractive_index(wavelength_nm):
    """SK-like phase refractive index for water at 13 C and 1.01325 bar.

    Uses only JAX operations, so wavelength derivatives work in forward
    and reverse mode.
    """
    wl = jnp.asarray(wavelength_nm) / 1000.0  # nm -> micrometres
    a1, a2, a3, a4, a5 = _SK4_A
    b1, b2, b3, b4, b5, b6, b7, b8, b9 = _SK4_B
    c1, c2, c3, c4, c5, c6 = _SK4_C
    dt = _SK4_TEMP_C - _SK4_TEMP_REF_C
    dp = _SK4_PRESSURE_BAR - _SK4_PRESSURE_REF_BAR
    wl2 = wl * wl
    wl4 = wl2 * wl2
    wl6 = wl4 * wl2
    base = jnp.sqrt(a1 / (wl2 - _SK4_WLA_UM2) + a2 + a3 * wl2 + a4 * wl4 + a5 * wl6)
    return (
        base
        + (b1 + b2 * wl2 + b3 * wl4) * dt
        + (b4 + b5 * wl2 + b6 * wl4) * dt**2
        + (b7 + b8 * wl2 + b9 * wl4) * dt**3
        + (c1 + c2 * wl2 + (c3 + c4 * wl2) * _SK4_TEMP_C) * dp
        + (c5 + c6 * wl2) * dp**2
    )


def sk4_water_refractive_index_derivative(wavelength_nm):
    """Analytic derivative of the SK-like water phase index, ``dn/dlambda`` per nm."""
    wl = jnp.asarray(wavelength_nm) / 1000.0
    a1, _, a3, a4, a5 = _SK4_A
    _, b2, b3, _, b5, b6, _, b8, b9 = _SK4_B
    _, c2, _, c4, _, c6 = _SK4_C
    dt = _SK4_TEMP_C - _SK4_TEMP_REF_C
    dp = _SK4_PRESSURE_BAR - _SK4_PRESSURE_REF_BAR
    wl2 = wl * wl
    wl3 = wl2 * wl
    wl4 = wl2 * wl2
    wl5 = wl4 * wl
    root = jnp.sqrt(
        _SK4_A[0] / (wl2 - _SK4_WLA_UM2)
        + _SK4_A[1]
        + _SK4_A[2] * wl2
        + _SK4_A[3] * wl4
        + _SK4_A[4] * wl4 * wl2
    )
    derivative_per_um = (
        (
            -2.0 * a1 * wl / (wl2 - _SK4_WLA_UM2) ** 2
            + 2.0 * a3 * wl
            + 4.0 * a4 * wl3
            + 6.0 * a5 * wl5
        )
        / (2.0 * root)
        + (2.0 * b2 * wl + 4.0 * b3 * wl3) * dt
        + (2.0 * b5 * wl + 4.0 * b6 * wl3) * dt**2
        + (2.0 * b8 * wl + 4.0 * b9 * wl3) * dt**3
        + (2.0 * c2 * wl + 2.0 * c4 * wl * _SK4_TEMP_C) * dp
        + 2.0 * c6 * wl * dp**2
    )
    return derivative_per_um / 1000.0


def sk4_water_group_refractive_index(wavelength_nm):
    """SK-like water group index: ``n_g = n - lambda * dn/dlambda``."""
    wavelength_nm = jnp.asarray(wavelength_nm)
    return sk4_water_refractive_index(
        wavelength_nm
    ) - wavelength_nm * sk4_water_refractive_index_derivative(wavelength_nm)


def group_velocity_m_per_ns(wavelength_nm, medium):
    """Return differentiable group velocity for each wavelength."""
    wavelength_nm = jnp.asarray(wavelength_nm)
    if medium.dispersion_model == "sk4_water":
        c_vacuum = medium.speed_of_light * medium.refractive_index
        return c_vacuum / sk4_water_group_refractive_index(wavelength_nm)
    if medium.dispersion_model == "constant":
        return jnp.full_like(wavelength_nm, medium.speed_of_light)
    raise ValueError(f"Unknown dispersion model {medium.dispersion_model!r}")


# ---------------------------------------------------------------------------
# QE curve loading
# ---------------------------------------------------------------------------


class QECurve:
    """A PMT QE curve, read once from a JSON file with ``wavelengths_nm`` and ``qe_percent``.

    Calling it gives the QE fraction (0 to 1) at the given wavelengths, linear between knots and
    0 outside the tabulated range. ``bounds`` is the (min, max) knot and ``peak`` the largest QE
    (the curve's maximum, since it interpolates linearly).
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        wavelengths = np.array(data["wavelengths_nm"])
        qe_pct = np.array(data["qe_percent"])
        self.bounds = (float(wavelengths.min()), float(wavelengths.max()))
        self.peak = float(np.max(data["qe_percent"])) / 100.0

        sort_idx = np.argsort(wavelengths)
        wl_knots = jnp.array(wavelengths[sort_idx])
        qe_knots = jnp.array(qe_pct[sort_idx] / 100.0)  # percent -> fraction
        wl_min = float(wl_knots[0])
        wl_max = float(wl_knots[-1])

        @jax.jit
        def get_qe(wavelength):
            qe = jnp.interp(wavelength, wl_knots, qe_knots)
            in_range = (wavelength >= wl_min) & (wavelength <= wl_max)
            return jnp.where(in_range, qe, 0.0)

        self._get_qe = get_qe

    def __call__(self, wavelength):
        return self._get_qe(wavelength)


def load_qe_curve(json_path) -> QECurve:
    """Read a PMT QE curve JSON once; see :class:`QECurve`."""
    return QECurve(json_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Default material lookup dir: config/materials/<material>.json (multi-material merge).
_MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config", "materials")


def _scintillation_kwargs_from(data: dict) -> dict:
    """Extract the STATIC scintillation knobs from a material JSON.

    The differentiable values (light yield S/kB/C, timing tau) live on DetectorParams and are
    sourced via ``load_physics_config``; this returns what the simulator needs at setup to pick
    surrogates, the Moyal window and the Moyal shape. A material is non-scintillating if it lacks
    a ``"scintillation"`` block (→ Cherenkov-only defaults).
    """
    scint = data.get("scintillation")
    if scint is None:
        return {"emission_processes": ("cherenkov",)}
    procs = tuple(scint.get("emission_processes", ("cherenkov", "scintillation")))
    spec = scint.get("spectrum", {})
    split = scint.get("photon_split", {"cherenkov_fraction": 0.5})
    if "moyal_loc" not in spec or "moyal_scale" not in spec:
        raise ValueError(
            f"material {data['material']!r} enables scintillation but has no "
            "scintillation.spectrum.moyal_loc / moyal_scale: the surrogate cannot build its "
            "λ inverse-CDF sampler."
        )
    return {
        "emission_processes": procs,
        "scintillation_lambda_min": float(spec.get("lambda_min", 340.0)),
        "scintillation_lambda_max": float(spec.get("lambda_max", 550.0)),
        "cherenkov_fraction": float(split["cherenkov_fraction"]),
        "moyal_loc": np.float32(spec["moyal_loc"]),
        "moyal_scale": np.float32(spec["moyal_scale"]),
    }


_HC_EV_NM = 1239.84198  # h c in eV nm


def _tabulated_coeff(section: dict, wl: jnp.ndarray) -> jnp.ndarray:
    """1 / length on ``wl`` from a section's ``wavelengths_nm`` / ``length_m`` table.

    The length is interpolated linearly in photon energy and held at the end values outside
    the table, as Geant4 evaluates a material property, so a table taken from a Geant4 material
    (WCSim's water) reproduces it.
    """
    e_tab = _HC_EV_NM / np.asarray(section["wavelengths_nm"], dtype=np.float64)
    order = np.argsort(e_tab)
    length = jnp.interp(
        _HC_EV_NM / wl,
        jnp.asarray(e_tab[order], dtype=jnp.float32),
        jnp.asarray(np.asarray(section["length_m"], dtype=np.float64)[order], dtype=jnp.float32),
    )
    return 1.0 / length


def load_medium(material: str = "water", medium_model_path: str = None) -> MediumProperties:
    """Read a material JSON once: the scalar :class:`MediumProperties`, carrying the parsed
    file as ``model`` for :func:`medium_on_grid`.

    ``medium_model_path`` is the file; without it, ``config/materials/<material>.json``. The
    record's ``material`` is the file's own ``material`` key. The file's optional
    ``dispersion.model`` (``"constant"`` or ``"sk4_water"``) sets the group velocity.
    """
    if medium_model_path is None:
        medium_model_path = os.path.join(_MATERIALS_DIR, f"{material}.json")
        if not os.path.exists(medium_model_path):
            raise ValueError(
                f"No material config at {medium_model_path!r}. Add a "
                f"config/materials/{material}.json or pass medium_model_path."
            )
    with open(medium_model_path) as f:
        data = json.load(f)
    n = data["refractive_index"]
    c_vac = data["speed_of_light_vacuum_m_per_ns"]
    return MediumProperties(
        material=data["material"],
        refractive_index=n,
        speed_of_light=c_vac / n,
        model=data,
        dispersion_model=data.get("dispersion", {}).get("model", "constant"),
        **_scintillation_kwargs_from(data),
    )


def medium_on_grid(medium: MediumProperties, wavelength_grid) -> MediumProperties:
    """``medium`` with its wavelength-dependent arrays evaluated on ``wavelength_grid`` (nm)."""
    data = medium.model
    n = data["refractive_index"]
    wl = jnp.asarray(wavelength_grid, dtype=jnp.float32)
    dispersion_model = medium.dispersion_model

    # Each of the three coefficients is either tabulated (``wavelengths_nm`` / ``length_m``,
    # see `_tabulated_coeff`) or given by the SK calibration paper's parametrisation.

    # Symmetric (Rayleigh) scattering coefficient
    sc = data["scattering"]["symmetric"]
    if "length_m" in sc:
        alpha_sym = _tabulated_coeff(sc, wl)
    else:
        P4, P5 = sc["P4"], sc["P5"]
        alpha_sym = P4 / (wl**4) * (1.0 + P5 / (wl**2))

    # Asymmetric (Mie) scattering coefficient
    ac = data["scattering"]["asymmetric"]
    if "length_m" in ac:
        alpha_asym = _tabulated_coeff(ac, wl)
    else:
        P6, P7, P8 = ac["P6"], ac["P7"], ac["P8"]
        alpha_asym = P6 * (1.0 + P7 / (wl**4) * (wl - P8) ** 2)
    g = ac["default_asymmetry_g"]

    # Absorption coefficient
    ab = data["absorption"]
    if "length_m" in ab:
        alpha_abs = _tabulated_coeff(ab, wl)
    else:
        P0, P1, P2, P3 = ab["P0"], ab["P1"], ab["P2"], ab["P3"]
        pf_wl = jnp.array(ab["pope_fry_wavelengths_nm"], dtype=jnp.float32)
        pf_abs = jnp.array(ab["pope_fry_absorption_m_inv"], dtype=jnp.float32)

        # Eq.(15) power-law fit (blue) spliced onto the Pope&Fry data (red) at 464 nm.
        # The two pieces differ by ~12% at the seam, so a hard switch leaves a kink; blend
        # them with a C2 smootherstep over a +-12 nm window.
        C_powerlaw = P0 * P2 * (wl / 500.0) ** P3
        C_popefry = jnp.interp(wl, pf_wl, pf_abs)
        _bw = 12.0
        _t = jnp.clip((wl - (464.0 - _bw)) / (2.0 * _bw), 0.0, 1.0)
        _s = _t * _t * _t * (_t * (_t * 6.0 - 15.0) + 10.0)  # smootherstep (C2)
        C = (1.0 - _s) * C_powerlaw + _s * C_popefry
        alpha_abs = P0 * P1 / (wl**4) + C

    if dispersion_model == "sk4_water":
        refractive_index_curve = sk4_water_refractive_index(wl)
        group_refractive_index_curve = sk4_water_group_refractive_index(wl)
    elif dispersion_model == "constant":
        refractive_index_curve = jnp.full_like(wl, n)
        group_refractive_index_curve = jnp.full_like(wl, n)
    else:
        raise ValueError(f"Unknown dispersion model {dispersion_model!r}")

    return medium._replace(
        wavelength_grid=wl,
        scatter_coeff=alpha_sym,
        absorption_coeff=alpha_abs,
        refractive_index_curve=refractive_index_curve,
        mie_scatter_coeff=alpha_asym,
        mie_asymmetry=g,
        group_refractive_index_curve=group_refractive_index_curve,
    )


def make_medium(
    material: str = "water",
    wavelength_grid: Optional[jnp.ndarray] = None,
    medium_model_path: str = None,
) -> MediumProperties:
    """:func:`load_medium`, then :func:`medium_on_grid` when a ``wavelength_grid`` is given."""
    medium = load_medium(material, medium_model_path)
    return medium if wavelength_grid is None else medium_on_grid(medium, wavelength_grid)
