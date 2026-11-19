"""Many HK PMTs: the expected-value and the sampled transport deposit the same charge on each PMT of a patch.

A patch of the real HK barrel PMTs (the shipped geometry, the `hk50cm` shape, the same candidate grid as the timing job) is lit by a wide beam from 2.4 m in front of it, through transparent water, with
absorbing walls and the shipped WCSim-derived PMT tables. The beam meets the PMTs at every angle from normal to grazing, and rays can pass within reach of several neighbouring PMTs. K = 1 is the first
encounter only (candidate routing and first-hit ordering); K = 2 adds the light the PMTs reflect. Per PMT, the sampled counts must match the expected-value charge within their combined counting error.
`test_table_of_the_patch` prints the numbers (run with -s): the sampled / expected ratio by the PMT's incidence angle, and the worst PMTs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from tests.helpers.paths import REPO as ROOT  # noqa: E402

GEOM = ROOT / "config" / "SK_like_geom_config.json"
HK_GEOMETRY = ROOT / "config" / "hk_geometry.npz"
TABLE = ROOT / "config" / "pmt" / "hk50cm_wcsim_optics.txt.gz"
DETECTION = ROOT / "config" / "pmt" / "hk50cm_wcsim_detection_400nm.npz"
GRID = dict(n_cap=200, n_angular=300, n_height=200)  # the timing job's candidate grid
N = 1_000_000
N_KEYS = 2  # calls per mode in the assertion test
N_KEYS_TABLE = 20  # and in the printed table: about 20M photons per mode and K, for a 0.1% total
QE = 0.2
SOURCE = np.array(
    [30.0, 0.0, 0.0]
)  # 2.43 m in front of the wall at x = 32.43 m, which the beam points at
NA = 0.9  # a half-angle of 64 degrees: incidence angles on the patch from normal to grazing
KS = (1, 2)


class _GivenDetector:
    """Stands in for the config's detector class: ``from_config`` returns the detector built here."""

    def __init__(self, det):
        self.det = det

    def from_config(self, geometry, config_dir):
        return self.det


@pytest.fixture(scope="module")
def patch(tmp_path_factory):
    """A patch of the real HK barrel (|z| < 2.5 m, within 0.25 rad of the beam's azimuth) as the whole detector, with the shipped tables."""
    from lucid.geometry import detector_geometry
    from lucid.geometry.cylinder import Cylinder
    from lucid.simulation.pmt_detection import load_pmt_detection_response

    full = np.load(HK_GEOMETRY, allow_pickle=True)
    pos, direction, surface = full["positions_mm"], full["directions"], full["surfaces"]
    keep = (
        (surface == "barrel")
        & (np.abs(pos[:, 2]) < 2500.0)
        & (np.abs(np.arctan2(pos[:, 1], pos[:, 0])) < 0.25)
    )
    assert 20 < keep.sum() < 400, keep.sum()
    path = tmp_path_factory.mktemp("hk_patch") / "patch.npz"
    np.savez(
        path,
        positions_mm=pos[keep],
        directions=direction[keep],
        surfaces=surface[keep],
        pmt_id=np.arange(keep.sum()) + 1,
        radius=full["radius"],
        height=full["height"],
        sensor_radius=full["sensor_radius"],
    )
    det = Cylinder.from_pmt_file(path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(detector_geometry, "get_detector_class", lambda _: _GivenDetector(det))
        yield dict(
            det=det,
            response=load_pmt_detection_response(DETECTION),
            positions=pos[keep] / 1000.0,
            normals=direction[keep],
        )


def _params(det):
    """Transparent water and absorbing walls: only the PMTs act. Sensor scale 1 is the table's own reflectance."""
    from lucid.detector_params import DetectorParams

    return DetectorParams.from_flat(
        scatter_length=1e9,
        mie_scatter_length=1e9,
        absorption_length=1e9,
        wall_reflection_rate=0.0,
        sensor_reflection_rate=1.0,
        qe=QE,
        qe_corrections=jnp.ones(len(det.all_points)),
    )


def _simulator(response, k, **setup):
    from lucid.simulation import setup_event_simulator

    return setup_event_simulator(
        str(GEOM),
        N,
        temperature=None,
        K=k,
        source="light_source",
        wavelength_mode=False,
        sensor_shape="hk50cm",
        reflection_model="tabulated_lambertian",
        reflection_wavelength=400.0,
        pmt_detection_model=response,
        **GRID,
        **setup,
    )


@pytest.fixture(scope="module")
def sims(patch):
    """{K: (expected-value, sampled)} simulators, each compiled on its first call."""
    return {
        k: (
            _simulator(patch["response"], k),
            _simulator(patch["response"], k, transport="sampled", detection="sampled"),
        )
        for k in KS
    }


def _charges(sim, patch, keys):
    """Per-PMT charge, summed over `keys` calls (one call for the expected-value mode, whose rays are the same each time)."""
    from lucid.sources import laser_source

    source = laser_source(
        position=list(SOURCE), direction=[1.0, 0.0, 0.0], intensity=N, fiber_NA=NA
    )
    params = _params(patch["det"])
    return sum(
        np.asarray(sim(source, params, jax.random.PRNGKey(k))[0], dtype=np.float64) for k in keys
    )


def _compare(sims, patch, k, n_keys=N_KEYS):
    """Per-PMT expected charge and sampled counts over the same total photons (`n_keys` calls of N each, different rays), and the counting error of their difference."""
    expected = _charges(sims[k][0], patch, range(n_keys))
    sampled = _charges(sims[k][1], patch, range(100, 100 + n_keys))
    sigma = np.sqrt(
        np.maximum(expected, 1.0) * (1.0 + QE)
    )  # Poisson counts of the sampled mode and the rays' own scatter in the expectation
    return expected, sampled, sigma


@pytest.mark.parametrize("k", KS)
def test_per_pmt_charge_agrees_between_the_modes(sims, patch, k):
    expected, sampled, sigma = _compare(sims, patch, k)
    assert expected.sum() > 0.01 * N * N_KEYS * QE, "the beam should light the patch"
    total_sigma = np.sqrt((sigma**2).sum())
    assert abs(sampled.sum() - expected.sum()) < 5 * total_sigma, (
        sampled.sum(),
        expected.sum(),
        total_sigma,
    )
    lit = expected >= 50
    z = (sampled - expected)[lit] / sigma[lit]
    assert np.abs(z).max() < 5.5, (np.abs(z).max(), np.argmax(np.abs(z)))
    assert 0.6 < (z**2).mean() < 1.6, (z**2).mean()


def test_table_of_the_patch(sims, patch):
    """Not an assertion: the sampled / expected ratio by the beam's incidence angle on each PMT, and the worst PMTs, per K. Run with -s."""
    d = patch["positions"] - SOURCE
    cos_inc = np.abs(np.sum(d * patch["normals"], axis=1)) / np.linalg.norm(d, axis=1)
    angle = np.degrees(np.arccos(np.clip(cos_inc, 0.0, 1.0)))
    bins = [0, 20, 40, 60, 75, 90]
    for k in KS:
        expected, sampled, sigma = _compare(sims, patch, k, N_KEYS_TABLE)
        print(
            f"\nK = {k}: {len(expected)} PMTs, total expected {expected.sum():.6g}, sampled {sampled.sum():.6g}, ratio {sampled.sum() / expected.sum():.4f} "
            f"(sigma {np.sqrt((sigma**2).sum()) / expected.sum():.4f})"
        )
        print(
            f"{'incidence (deg)':>16s}{'PMTs':>6s}{'expected':>12s}{'sampled':>12s}{'ratio':>8s}{'sigma':>8s}{'z':>7s}"
        )
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (angle >= lo) & (angle < hi) & (expected > 0)
            if not m.any():
                continue
            e, s, sg = expected[m].sum(), sampled[m].sum(), np.sqrt((sigma[m] ** 2).sum())
            print(
                f"{f'[{lo}, {hi})':>16s}{m.sum():>6d}{e:>12.6g}{s:>12.6g}{s / e:>8.4f}{sg / e:>8.4f}{(s - e) / sg:>+7.1f}"
            )
        z = (sampled - expected) / sigma
        worst = np.argsort(-np.abs(z))[:5]
        print(
            "worst PMTs (index, incidence deg, expected, sampled, z):",
            [
                (
                    int(i),
                    round(float(angle[i]), 1),
                    round(float(expected[i]), 1),
                    round(float(sampled[i]), 1),
                    round(float(z[i]), 1),
                )
                for i in worst
            ],
        )
