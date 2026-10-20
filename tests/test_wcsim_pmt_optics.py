"""WCSim's default PMT photocathode as a tabulated PMT, and the flat Lambertian (WCSim) wall.

`scripts/make_wcsim_pmt_optics.py` tabulates WCSim's model 0 (outer water->glass Fresnel; at the
glass/cathode surface 1 - rgcff absorbed, rgcff Fresnel into index 1.0) as R, A, T per
polarisation. The shipped table must load through the tabulated models' loader and show WCSim's
behaviour; `tabulated_lambertian` puts it on the PMTs beside a flat Lambertian wall.
"""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.simulation.reflection import (
    INDEPENDENT_SENSOR_DETECTION_MODELS, TabulatedScaledReflection, get_reflection_model,
    load_pmt_optical_table, tabulated_lambertian_reflection, tabulated_pmt_optical_outcome,
    tabulated_pmt_reflectance,
)
from lucid.simulation.optics import compute_reflection_direction

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'config' / 'pmt' / 'hk50cm_wcsim_optics.txt.gz'
DETECTION = ROOT / 'config' / 'pmt' / 'hk50cm_wcsim_detection_400nm.npz'
WCSIM_MATERIALS = Path('/user/nfost94/HKSim/WCSim_alma9/WCSim/src/WCSimConstructMaterials.cc')

_spec = importlib.util.spec_from_file_location('make_wcsim_pmt_optics',
                                               ROOT / 'scripts' / 'make_wcsim_pmt_optics.py')
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _fresnel_unpol(n1, n2, ci):
    s2 = (n1 / n2) ** 2 * (1 - ci * ci)
    if s2 >= 1:
        return 1.0
    ct = np.sqrt(1 - s2)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return 0.5 * (rs * rs + rp * rp)


class TestModelZero:
    N_W, N_G, RGCFF = 1.344, 1.531, 0.32

    def _unpol(self, theta_deg):
        R, A, T = gen.pmt_optics(np.cos(np.deg2rad(theta_deg)), self.N_W, self.N_G, 1.0, self.RGCFF)
        return [0.5 * (x['s'] + x['p']) for x in (R, A, T)]

    def test_head_on_is_the_two_surfaces_in_closed_form(self):
        R1 = ((self.N_G - self.N_W) / (self.N_G + self.N_W)) ** 2
        RF = ((self.N_G - 1.0) / (self.N_G + 1.0)) ** 2
        r = self.RGCFF * RF
        R, A, T = self._unpol(0.0)
        npt.assert_allclose(R, R1 + (1 - R1) ** 2 * r / (1 - R1 * r), rtol=1e-12)
        npt.assert_allclose(A, (1 - R1) * (1 - self.RGCFF) / (1 - R1 * r), rtol=1e-12)

    def test_probabilities_add_up_and_nothing_is_lost_past_total_internal_reflection(self):
        edge = np.rad2deg(np.arcsin(1.0 / self.N_W))
        for theta in (0.0, 20.0, 40.0, edge - 0.5, edge + 0.5, 60.0, 85.0):
            R, A, T = self._unpol(theta)
            npt.assert_allclose(R + A + T, 1.0, atol=1e-12)
            if theta > edge:
                npt.assert_allclose(T, 0.0, atol=1e-12)
                ci = np.cos(np.deg2rad(theta))
                R1 = _fresnel_unpol(self.N_W, self.N_G, ci)
                assert R > self.RGCFF and A < 1 - self.RGCFF   # the rgcff share all comes back out
                assert R1 < R < R1 + self.RGCFF

    def test_absorption_is_flat_until_grazing(self):
        A0 = self._unpol(0.0)[1]
        for theta in (20.0, 40.0, 55.0):
            npt.assert_allclose(self._unpol(theta)[1] / A0, 1.0, atol=0.01)


@pytest.mark.skipif(not WCSIM_MATERIALS.exists(), reason='WCSim source not on this machine')
def test_qe_is_wcsims_head_on_detection_per_arriving_photon():
    """HK_QE_wcsim.json = A(0) x WCSim's detection per absorbed photon, A(0) the head-on
    absorbed fraction (front glass included), as the PMT table has it."""
    import json
    qe = json.loads((ROOT / 'config' / 'pmt' / 'HK_QE_wcsim.json').read_text())
    wl, per_absorbed = gen.wcsim_detection_per_absorbed(
        WCSIM_MATERIALS.parent,
        Path('/user/nfost94/HKSim/Current_Analysis/Watchmal/run_macros/tuning_parameters.mac'))
    npt.assert_allclose(qe['wavelengths_nm'], wl)
    npt.assert_allclose(np.array(qe['qe_percent']) / 100,
                        np.array(qe['_head_on_absorbed']) * per_absorbed, atol=1e-6)
    i400 = qe['wavelengths_nm'].index(400.0)
    head_on = tabulated_pmt_optical_outcome(jnp.asarray(1.0), jnp.asarray(400.0),
                                            load_pmt_optical_table(TABLE)).absorptance
    npt.assert_allclose(qe['_head_on_absorbed'][i400], head_on, atol=2e-4)


@pytest.mark.skipif(not WCSIM_MATERIALS.exists(), reason='WCSim source not on this machine')
def test_wcsim_indices_are_its_active_tables():
    energy, n_water, n_glass, n_cathode = gen.wcsim_indices(WCSIM_MATERIALS)
    assert energy.size == 60 and n_cathode == 1.0
    assert n_glass.min() == pytest.approx(1.511) and n_glass.max() == pytest.approx(1.6328)   # BK7, not 1.6
    assert n_water.min() == pytest.approx(1.32885) and n_water.max() == pytest.approx(1.42535)


class TestShippedTable:
    def test_loads_and_shows_wcsim_behaviour(self):
        t = load_pmt_optical_table(TABLE)
        assert t.reflectance.shape == (108, 200)
        lam = jnp.asarray(400.0)
        head_on = tabulated_pmt_optical_outcome(jnp.asarray(1.0), lam, t)
        assert 0.015 < float(head_on.reflectance) < 0.022
        past = tabulated_pmt_optical_outcome(jnp.cos(jnp.deg2rad(55.0)), lam, t)
        npt.assert_allclose(past.transmittance, 0.0, atol=1e-6)
        assert 0.32 < float(past.reflectance) < 0.34

    def test_detection_response_is_the_tables_absorptance(self):
        from lucid.simulation.pmt_detection import load_pmt_detection_response
        d = load_pmt_detection_response(DETECTION)
        t = load_pmt_optical_table(TABLE)
        lam = jnp.asarray(400.0)
        a0 = tabulated_pmt_optical_outcome(jnp.cos(jnp.deg2rad(0.225)), lam, t).absorptance
        for theta in (10.0, 50.0, 75.0):
            mu = jnp.cos(jnp.deg2rad(theta))
            a = tabulated_pmt_optical_outcome(mu, lam, t).absorptance
            npt.assert_allclose(d.factor(mu), a / a0, atol=2e-3)


class TestTabulatedLambertian:
    def _params(self, wall_scale=0.1125, sensor_scale=1.0):
        t = load_pmt_optical_table(TABLE)
        return TabulatedScaledReflection(
            pmt_reflectance_table=t.reflectance, wall_scale=jnp.asarray(wall_scale),
            sensor_scale=jnp.asarray(sensor_scale), pmt_reflectance_s=t.reflectance_s,
            pmt_reflectance_p=t.reflectance_p)

    NORMAL = jnp.array([0.0, 0.0, 1.0])

    def _incident(self, theta_deg):
        th = np.deg2rad(theta_deg)
        return jnp.array([np.sin(th), 0.0, np.cos(th)])

    def test_registered_with_the_tabulated_models(self):
        fn, _ = get_reflection_model('tabulated_lambertian')
        assert fn is tabulated_lambertian_reflection
        assert 'tabulated_lambertian' in INDEPENDENT_SENSOR_DETECTION_MODELS

    def test_wall_is_flat_and_lambertian(self):
        rp = self._params()
        for theta in (0.0, 45.0, 85.0):
            d = self._incident(theta)
            specular = compute_reflection_direction(d, self.NORMAL)
            dirs = []
            for i in range(200):
                refl, refl_dir, lr = tabulated_lambertian_reflection(
                    d, self.NORMAL, False, rp, jnp.asarray(400.0), jax.random.PRNGKey(i))
                npt.assert_allclose(refl, 0.1125, rtol=1e-6)
                npt.assert_allclose(lr, 0.0)
                dirs.append(np.asarray(refl_dir))
            dirs = np.array(dirs)
            assert np.all(dirs @ np.asarray(self.NORMAL) < 0)          # back into the water
            # cosine-weighted about the inward normal: mean cos = 2/3, whatever the incidence
            npt.assert_allclose(np.mean(-dirs @ np.asarray(self.NORMAL)), 2 / 3, atol=0.05)
            assert not np.allclose(dirs, np.asarray(specular), atol=1e-3)

    def test_pmt_is_the_scaled_table_and_specular(self):
        lam = jnp.asarray(400.0)
        for scale in (1.0, 1.3):
            rp = self._params(sensor_scale=scale)
            for theta in (0.0, 30.0, 60.0):
                d = self._incident(theta)
                refl, refl_dir, _ = tabulated_lambertian_reflection(
                    d, self.NORMAL, True, rp, lam, jax.random.PRNGKey(0))
                table = tabulated_pmt_reflectance(jnp.cos(jnp.deg2rad(theta)), lam,
                                                  rp.pmt_reflectance_table)
                npt.assert_allclose(refl, scale * table, rtol=1e-5)
                npt.assert_allclose(refl_dir, compute_reflection_direction(d, self.NORMAL), atol=1e-6)

    def test_scales_enter_linearly(self):
        lam = jnp.asarray(400.0)
        d = self._incident(30.0)

        def refl(wall_scale, sensor_scale, sensor):
            return tabulated_lambertian_reflection(
                d, self.NORMAL, sensor, self._params(wall_scale, sensor_scale), lam,
                jax.random.PRNGKey(0))[0]
        gw = jax.grad(refl, argnums=0)(0.1125, 1.0, False)
        gs = jax.grad(refl, argnums=1)(0.1125, 1.0, True)
        npt.assert_allclose(gw, 1.0, rtol=1e-6)
        npt.assert_allclose(gs, refl(0.1125, 1.0, True), rtol=1e-5)
