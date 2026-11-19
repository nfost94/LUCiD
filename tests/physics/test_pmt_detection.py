import json

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.pmt_detection import (
    load_pmt_detection_response,
    make_tabulated_pmt_detection_response,
)


def test_tabulated_response_interpolates_and_allows_enhancement():
    response = make_tabulated_pmt_detection_response(
        [0.0, 0.5, 1.0], [0.2, 1.4, 1.0], wavelength_nm=405.5
    )

    factors = response.factor(jnp.array([-1.0, 0.25, 0.5, 2.0]))
    np.testing.assert_allclose(factors, [0.2, 0.8, 1.4, 1.0])


def test_tabulated_response_supports_forward_and_reverse_autodiff():
    response = make_tabulated_pmt_detection_response(
        [0.0, 0.5, 1.0], [0.2, 1.4, 1.0], wavelength_nm=405.5
    )

    primal, tangent = jax.jvp(response.factor, (jnp.asarray(0.25),), (jnp.asarray(1.0),))
    reverse = jax.grad(response.factor)(jnp.asarray(0.25))
    np.testing.assert_allclose(primal, 0.8)
    np.testing.assert_allclose(tangent, 2.4, rtol=1e-6)
    np.testing.assert_allclose(reverse, 2.4, rtol=1e-6)


def test_response_artifact_round_trip(tmp_path):
    path = tmp_path / "pmt_response.npz"
    np.savez_compressed(
        path,
        incidence_cosine=np.array([0.0, 0.5, 1.0]),
        relative_efficiency=np.array([0.2, 1.4, 1.0]),
        wavelength_nm=np.asarray(405.5),
        metadata_json=np.asarray(json.dumps({"source": "unit test"})),
    )

    response = load_pmt_detection_response(path)
    assert response.wavelength_nm == 405.5
    assert response.metadata == {"source": "unit test"}
    np.testing.assert_allclose(response.factor(0.25), 0.8)


def test_sk4_polarized_response_uses_legacy_swapped_sp_labels(tmp_path):
    path = tmp_path / "polarized_response.npz"
    np.savez_compressed(
        path,
        incidence_cosine=np.array([0.0, 1.0]),
        relative_efficiency=np.array([1.0, 1.0]),
        absorptance_s=np.array([0.2, 0.5]),
        absorptance_p=np.array([0.8, 0.5]),
        wavelength_nm=np.asarray(337.5),
    )
    response = load_pmt_detection_response(path)
    direction = jnp.array([[0.0, 0.0, -1.0]])
    normal = jnp.array([[[0.0, 1.0, 0.0]]])
    # The physical s axis is +x; the SK-like convention uses the table's p column.
    physical_s = jnp.array([[1.0, 0.0, 0.0]])
    physical_p = jnp.array([[0.0, 1.0, 0.0]])
    s_factor = response.polarized_factor(jnp.array([[0.0]]), direction, physical_s, normal)
    p_factor = response.polarized_factor(jnp.array([[0.0]]), direction, physical_p, normal)
    # Normal-incidence average is 0.5, so the relative p/s columns are 1.6/0.4.
    np.testing.assert_allclose(s_factor, 1.6, rtol=1e-6)
    np.testing.assert_allclose(p_factor, 0.4, rtol=1e-6)
