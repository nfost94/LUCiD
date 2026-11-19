"""The HK PMT's reflection, detection and loss are the same in the expected-value and the sampled transport.

A photon at the tabulated PMT has three fates: reflected with R(theta) (specular), detected with D(theta) = QE x the detection table's relative efficiency, or lost in the PMT with 1 - R - D.
The expected-value step lets weight R continue and deposits weight D from the same photon; the sampled step reflects with probability R, and the deposit converts at D / (1 - R) (simulator.py),
so that R + (1 - R) x D / (1 - R) = R + D. Here a pencil beam meets the tip of the HK 50 cm PMT's cap (the shipped WCSim-derived R and detection tables, the `hk50cm` shape, transparent water, a
single step) at 0, 30, 60, 80 and 88 degrees, and both modes' detected charge is compared with the tables' D. The reflected light's share, R, is what test_hk_pmt_patch's K = 2 comparison
of the two modes checks.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from tests.helpers.paths import REPO as ROOT  # noqa: E402

GEOM = ROOT / "config" / "SK_like_geom_config.json"
TABLE = ROOT / "config" / "pmt" / "hk50cm_wcsim_optics.txt.gz"
DETECTION = ROOT / "config" / "pmt" / "hk50cm_wcsim_detection_400nm.npz"
QE_FILE = ROOT / "config" / "pmt" / "HK_QE_wcsim.json"
N = 200_000
QE = 0.2
REFLECTION_WAVELENGTH = 400.0
ANGLES_DEG = (0.0, 30.0, 60.0, 80.0, 88.0)
APEX = np.array(
    [0.82, 0.0, 0.0]
)  # the cap's tip: the wall at x = 1 m, less the cap's 0.18 m height


class _GivenDetector:
    """Stands in for the config's detector class: ``from_config`` returns the detector built here."""

    def __init__(self, det):
        self.det = det

    def from_config(self, geometry, config_dir):
        return self.det


@pytest.fixture(scope="module")
def hk_pmt(tmp_path_factory):
    """One HK PMT on the barrel of a 1 m cylinder, with the shipped optical table and detection response."""
    from lucid.geometry import detector_geometry
    from lucid.geometry.cylinder import Cylinder
    from lucid.simulation.pmt_detection import load_pmt_detection_response
    from lucid.simulation.reflection import load_pmt_optical_table

    path = tmp_path_factory.mktemp("hk_pmt") / "one_pmt.npz"
    np.savez(
        path,
        positions_mm=np.array([[1000.0, 0.0, 0.0]]),
        directions=np.array([[-1.0, 0.0, 0.0]]),
        surfaces=np.array(["barrel"]),
        pmt_id=np.array([1]),
        radius=np.asarray(1.0),
        height=np.asarray(2.0),
        sensor_radius=np.asarray(0.254),
    )
    det = Cylinder.from_pmt_file(path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(detector_geometry, "get_detector_class", lambda _: _GivenDetector(det))
        yield det, load_pmt_detection_response(DETECTION), load_pmt_optical_table()


def _params(det, sensor_scale=1.0):
    """Water made transparent and absorbing walls, so the PMT encounter is the only thing that acts; the sensor scale 1 is the table's own reflectance."""
    from lucid.detector_params import DetectorParams

    return DetectorParams.from_flat(
        scatter_length=1e9,
        mie_scatter_length=1e9,
        absorption_length=1e9,
        wall_reflection_rate=0.0,
        sensor_reflection_rate=sensor_scale,
        qe=QE,
        qe_corrections=jnp.ones(len(det.all_points)),
    )


def _simulator(response, **setup):
    from lucid.simulation import setup_event_simulator

    return setup_event_simulator(
        str(GEOM),
        N,
        temperature=None,
        K=1,
        source="light_source",
        wavelength_mode=False,
        sensor_shape="hk50cm",
        max_candidates_per_ray=1,
        reflection_model="tabulated_lambertian",
        reflection_wavelength=REFLECTION_WAVELENGTH,
        pmt_detection_model=response,
        **setup,
    )


@pytest.fixture(scope="module")
def sims(hk_pmt):
    """(expected-value, sampled) simulators, built once: each is compiled on its first call."""
    _, response, _ = hk_pmt
    return (_simulator(response), _simulator(response, transport="sampled", detection="sampled"))


def _truth(hk_pmt, angle):
    """The local incidence cosine and the tables' R and D at it, as the photon step and the detection response evaluate them."""
    from lucid.simulation.reflection import tabulated_pmt_reflectance

    _, response, table = hk_pmt
    cos = float(np.cos(np.deg2rad(angle)))
    r = float(
        tabulated_pmt_reflectance(
            jnp.asarray(cos), jnp.asarray(REFLECTION_WAVELENGTH), table.reflectance
        )
    )
    return (
        cos,
        min(r, 0.999),
        QE * float(response.factor(cos)),
    )  # the reflection step clips R at 0.999


def _encounter(sim, hk_pmt, angle, key=3, sensor_scale=1.0):
    """A pencil beam of N photons meeting the cap's tip at `angle` from its axis. Returns the detected charge per photon."""
    from lucid.sources import laser_source

    det = hk_pmt[0]
    theta = np.deg2rad(angle)
    u = np.array([-np.cos(theta), np.sin(theta), 0.0])  # from the tip back along the beam
    source = laser_source(
        position=list(APEX + 0.4 * u), direction=list(-u), intensity=N, fiber_NA=1e-6
    )
    out = sim(source, _params(det, sensor_scale), jax.random.PRNGKey(key))
    return float(jnp.sum(out.charge)) / N


@pytest.mark.parametrize("angle", ANGLES_DEG)
def test_expected_value_step_gives_the_tables_D(sims, hk_pmt, angle):
    """D is the table's to 0.5%: the expected-value deposit is 0.2% below it at 30 degrees and under and 0.2% above it at 80 and 88 (test_table_of_the_encounter prints it)."""
    cos, r, d = _truth(hk_pmt, angle)
    np.testing.assert_allclose(_encounter(sims[0], hk_pmt, angle), d, rtol=5e-3)
    assert r + d <= 1.0 + 1e-6


@pytest.mark.parametrize("angle", ANGLES_DEG)
def test_sampled_step_gives_the_same_D(sims, hk_pmt, angle):
    """The sampled deposit converts at D / (1 - R), so after the reflection the detected fraction is D."""
    cos, r, d = _truth(hk_pmt, angle)
    detected = _encounter(sims[1], hk_pmt, angle)
    assert abs(detected - d) < 5 * np.sqrt(d * (1 - d) / N), (angle, "detected", detected, d)


@pytest.mark.parametrize("scale", (1.0, 0.1, 1e-6, 0.0))
@pytest.mark.parametrize("angle", (0.0, 60.0))
def test_the_sensor_scale_leaves_the_detection(sims, hk_pmt, angle, scale):
    """Scaling the PMT's reflectance (the job's SENSOR_R), down to 0, leaves detection at D in both modes, because the tabulated model detects independently of R."""
    cos, r, d = _truth(hk_pmt, angle)
    np.testing.assert_allclose(_encounter(sims[0], hk_pmt, angle, sensor_scale=scale), d, rtol=5e-3)
    detected = _encounter(sims[1], hk_pmt, angle, sensor_scale=scale)
    assert abs(detected - d) < 5 * np.sqrt(d * (1 - d) / N) + 5.0 / N, (
        angle,
        scale,
        "detected",
        detected,
        d,
    )


def test_the_conversion_to_the_sampled_detection_never_exceeds_one(hk_pmt):
    """The sampled deposit converts at D / (1 - R) (capped at 1): for every incidence angle in the shipped tables, with the HK QE at its largest, the cap does not bind."""
    from lucid.simulation.reflection import tabulated_pmt_reflectance

    _, response, table = hk_pmt
    qe_max = max(json.load(open(QE_FILE))["qe_percent"]) / 100.0
    cos = np.linspace(0.0, 1.0, 2001)
    r = np.minimum(
        np.asarray(
            tabulated_pmt_reflectance(
                jnp.asarray(cos), jnp.asarray(REFLECTION_WAVELENGTH), table.reflectance
            )
        ),
        0.999,
    )
    d = qe_max * np.asarray(response.factor(cos))
    assert np.all(d / (1.0 - r) <= 1.0), np.max(d / (1.0 - r))


def test_table_of_the_encounter(sims, hk_pmt):
    """Not an assertion: the numbers, printed with -s. Per angle, the tables' R and D, the expected-value mode's D, and the sampled mode's averaged over 20 keys (so about 4M photons) with its
    standard error."""
    keys = range(20)
    print(
        f"\n{'angle':>6s}{'cos':>9s}{'R table':>9s}{'D table':>9s}{'D exp':>9s}{'D exp/table':>12s}{'D sampled (se)':>18s}{'sampled/exp':>12s}"
    )
    for angle in ANGLES_DEG:
        cos, r_tab, d_tab = _truth(hk_pmt, angle)
        d_exp = _encounter(sims[0], hk_pmt, angle)
        runs = np.array([_encounter(sims[1], hk_pmt, angle, key=k) for k in keys])
        d_m, d_se = runs.mean(), runs.std(ddof=1) / np.sqrt(len(keys))
        print(
            f"{angle:>6.0f}{cos:>9.4f}{r_tab:>9.4f}{d_tab:>9.4f}{d_exp:>9.4f}{d_exp / d_tab:>12.4f}{f'{d_m:.4f} ({d_se:.4f})':>18s}{d_m / d_exp:>12.4f}"
        )
