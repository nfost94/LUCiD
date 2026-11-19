"""DifferentiableProfileSource: directions drawn from a frozen proposal, the profile carried in the ray weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.sources import differentiable_profile_source, profile_intensity, PROFILE_FORMS, N_PROFILE_PARAMS, TABLE_ANGLES_DEG, set_table_nodes, get_table_nodes

DIRECTION = jnp.array([1.0, 2.0, -3.0]) / jnp.sqrt(14.0)
POSITION = jnp.array([0.5, -1.0, 2.0])
# A collimator with a halo (a power law from the edge outwards and the edge's skirt, as the collimator scans are fitted): (w, p, a0, s, hp, n, h1, lam).
HALO = (2.8, 6.0, 2.65, 0.05, 1.3e-3, 1.44, 1.5e-2, 0.14)
# A toy table for the 'table' form: a strictly decreasing profile, 1 on axis, at the form's nodes, as the log-drops between them.
TABLE_V = (np.exp(-(TABLE_ANGLES_DEG.astype(np.float64) / 35.0) ** 2) + 0.01) / 1.01
TABLE = tuple(float(l) for l in np.log(-np.log(TABLE_V[1:] / TABLE_V[:-1])))
# Representative shapes: a collimator-like and a diffuser-like composite, a super-Gaussian, SK-IV's beam, and a table; the collimator with its halo
# goes out to the hemisphere (the contract only: its core is too narrow for the coarse bins of the reproduction test).
REPRO_CASES = [
    ('composite', (2.676, 7.469, 2.705, 0.0578), 3.5),
    ('composite', (38.58, 1.216, 44.78, 2.378), 50.0),
    ('super_gaussian', (36.65, 1.646), 60.0),
    ('sk_beam', (3300.0,), 16.0),
    ('table', TABLE, 90.0),
    ('table_cubic', TABLE, 90.0),
]
CASES = REPRO_CASES + [('composite', HALO, 90.0)]


def _angles(directions):
    return np.degrees(np.arccos(np.clip(np.asarray(directions) @ np.asarray(DIRECTION), -1, 1)))


def _shifted(src, dl):
    """The table source with every log-drop moved by dl (the drops scale by exp(dl)): a profile away from the proposal it was built at."""
    return src._replace(params=src.params.at[:len(TABLE)].add(dl))


def _binned(src, edges, key, n=400_000):
    """The weighted intensity per solid angle of n rays in angular bins, normalised to its maximum, and the profile at the bin centres likewise."""
    d, _, w = src(n, key)
    h, _ = np.histogram(_angles(d), edges, weights=np.asarray(w))
    sa = np.cos(np.radians(edges[:-1])) - np.cos(np.radians(edges[1:]))
    centres = 0.5 * (edges[1:] + edges[:-1])
    expect = np.asarray(profile_intensity(jnp.asarray(centres), src.params, src.profile_id))
    measured = h / sa
    return measured / measured.max(), expect / expect.max()


@pytest.mark.parametrize('profile,params,cut', CASES)
def test_contract_cap_and_normalisation(profile, params, cut):
    src = differentiable_profile_source(POSITION, DIRECTION, params, cut, profile=profile,
                                        intensity=2.5e6)
    d, o, w = src(4096, jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(o, np.tile(POSITION, (4096, 1)))
    np.testing.assert_allclose(float(jnp.sum(w)), 2.5e6, rtol=1e-5)
    assert _angles(d).max() <= cut + 1e-3


@pytest.mark.parametrize('profile,params,cut', REPRO_CASES)
def test_weighted_rays_reproduce_the_profile(profile, params, cut):
    # Weighted intensity per solid angle in angular bins vs the analytic profile.
    src = differentiable_profile_source(POSITION, DIRECTION, params, cut, profile=profile)
    measured, expect = _binned(src, np.linspace(0, cut, 26), jax.random.PRNGKey(1))
    bright = expect > 0.05
    np.testing.assert_allclose(measured[bright], expect[bright], rtol=0.06, atol=0.01)


def test_directions_do_not_depend_on_params():
    # Replacing params on a built source (as a fit does) leaves every ray where it was.
    a = differentiable_profile_source(POSITION, DIRECTION, (38.0, 1.2, 44.0, 2.4), 50.0, profile='composite')
    b = a._replace(params=a.params.at[:4].set(jnp.array([20.0, 3.0, 30.0, 1.0])))
    key = jax.random.PRNGKey(3)
    np.testing.assert_array_equal(a(1024, key)[0], b(1024, key)[0])


@pytest.mark.parametrize('dl', [-0.2, 0.2])
def test_table_reproduced_away_from_the_proposal(dl):
    # Built at the table, every drop then moved by about a fifth in log: the importance weights still reproduce the moved profile.
    src = _shifted(differentiable_profile_source(POSITION, DIRECTION, TABLE, 90.0), dl)
    measured, expect = _binned(src, np.linspace(0, 90.0, 46), jax.random.PRNGKey(7))
    bright = expect > 0.01
    np.testing.assert_allclose(measured[bright], expect[bright], rtol=0.05, atol=0.005)


def test_effective_sample_size():
    # (sum w)^2 / (n sum w^2): the share of rays that count. Near the proposal almost all; with every drop moved by a fifth in log still most,
    # where drawing directions uniformly over the cap gives ~1/4.
    src = differentiable_profile_source(POSITION, DIRECTION, TABLE, 90.0)
    ess = lambda w: float(jnp.sum(w) ** 2 / (w.size * jnp.sum(w ** 2)))
    key = jax.random.PRNGKey(9)
    assert ess(src(100_000, key)[2]) > 0.95
    assert ess(_shifted(src, 0.2)(100_000, key)[2]) > 0.8
    assert ess(_shifted(src, -0.2)(100_000, key)[2]) > 0.8


@pytest.mark.parametrize('profile,params,idx,steps,cut', [
    ('composite', (38.58, 1.216, 44.78, 2.378), (0, 1, 2, 3), (0.05, 0.01, 0.05, 0.02), 50.0),
    ('table', TABLE, (2, 5, 8, 12), (0.02, 0.02, 0.02, 0.02), 90.0),
    ('table_cubic', TABLE, (2, 5, 8, 12), (0.02, 0.02, 0.02, 0.02), 90.0),
])
def test_gradient_through_weights_matches_finite_differences(profile, params, idx, steps, cut):
    # A detector-like observable: weighted light inside a 20 deg cone of an off-axis direction, as a function of four of the parameters.
    target = jnp.array([0.2, 0.9, -0.4]); target = target / jnp.linalg.norm(target)
    key = jax.random.PRNGKey(5)
    idx = jnp.array(idx)
    built = differentiable_profile_source(POSITION, DIRECTION, params, cut, profile=profile)
    p0 = built.params[idx]

    def observable(p):
        d, _, w = built._replace(params=built.params.at[idx].set(p))(20_000, key)
        inside = jax.nn.sigmoid((d @ target - jnp.cos(jnp.radians(20.0))) * 2000.0)
        return jnp.sum(w * inside)

    g = jax.grad(observable)(p0)
    for i, h in enumerate(steps):
        e = jnp.zeros(len(idx)).at[i].set(h)
        fd = (observable(p0 + e) - observable(p0 - e)) / (2 * h)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=2e-2, atol=1e-3 * abs(float(fd)) + 1e-6)
    assert np.all(np.isfinite(np.asarray(g)))


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        differentiable_profile_source(POSITION, DIRECTION, (1.0,), 10.0, profile='gaussian')


def test_composite_halo_level_is_quoted_at_4_deg_and_the_zero_padding_has_none():
    cid = PROFILE_FORMS.index('composite')
    pp = jnp.zeros(N_PROFILE_PARAMS).at[:len(HALO)].set(jnp.asarray(HALO))
    got = float(profile_intensity(jnp.asarray([10.0]), pp, cid)[0])           # far outside the core and its skirt: the power law alone
    np.testing.assert_allclose(got, HALO[4] * (10.0 / 4.0) ** -HALO[5], rtol=1e-4)
    core_only = jnp.zeros(N_PROFILE_PARAMS).at[:4].set(jnp.asarray(HALO[:4]))
    assert float(profile_intensity(jnp.asarray([10.0]), core_only, cid)[0]) < 1e-12


def test_composite_halo_is_reproduced_out_to_the_cap():
    # Past the edge's skirt the weighted rays carry the power law to 90 deg: each bin's share of the halo's flux against the profile's, on a fine grid.
    src = differentiable_profile_source(POSITION, DIRECTION, HALO, 90.0, profile='composite')
    d, _, w = src(1_000_000, jax.random.PRNGKey(2))
    edges = np.geomspace(5.0, 90.0, 8)
    h, _ = np.histogram(_angles(d), edges, weights=np.asarray(w))
    expect = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        al = np.linspace(lo, hi, 4001)
        mid = jnp.asarray(0.5 * (al[1:] + al[:-1]))
        expect.append(np.sum(np.asarray(profile_intensity(mid, src.params, src.profile_id)) * -np.diff(np.cos(np.radians(al)))))
    np.testing.assert_allclose(h / h.sum(), np.array(expect) / np.sum(expect), rtol=0.15)


def test_composite_jacobian_is_finite_in_every_parameter():
    # Forward mode through all eight parameters, the halo's included, with and without a halo: the zero padding has hp = n = h1 = lam = 0.
    a = jnp.linspace(0.0, 89.0, 200)
    cid = PROFILE_FORMS.index('composite')
    for p in (jnp.zeros(N_PROFILE_PARAMS).at[:4].set(jnp.asarray(HALO[:4])), jnp.zeros(N_PROFILE_PARAMS).at[:len(HALO)].set(jnp.asarray(HALO))):
        J = jax.jacfwd(lambda q: profile_intensity(a, q, cid))(p)
        assert np.all(np.isfinite(np.asarray(J)))


def test_table_profile_returns_its_nodes_and_is_linear_in_cos_alpha_between_them():
    pid = PROFILE_FORMS.index('table')
    pp = jnp.asarray(TABLE, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(profile_intensity(jnp.asarray(TABLE_ANGLES_DEG), pp, pid)), TABLE_V, rtol=2e-4)
    i = 20                                                  # halfway in cos(alpha) between two nodes
    mid = np.degrees(np.arccos(0.5 * (np.cos(np.radians(TABLE_ANGLES_DEG[i])) + np.cos(np.radians(TABLE_ANGLES_DEG[i + 1])))))
    got = float(profile_intensity(jnp.asarray([mid], dtype=jnp.float32), pp, pid)[0])
    np.testing.assert_allclose(got, 0.5 * (TABLE_V[i] + TABLE_V[i + 1]), rtol=2e-4)


def test_table_profile_is_one_on_axis_and_strictly_decreasing_for_any_parameters():
    pid = PROFILE_FORMS.index('table')
    a = jnp.asarray(TABLE_ANGLES_DEG)
    for seed in range(3):
        p = jnp.asarray(np.random.default_rng(seed).normal(-3.0, 3.0, N_PROFILE_PARAMS), dtype=jnp.float32)
        v = np.asarray(profile_intensity(a, p, pid))
        assert v[0] == pytest.approx(1.0)
        assert np.all(np.diff(v) <= 0) and np.all(v >= 0)


def test_table_profile_jacobian_is_the_chain_rule_of_the_log_drops():
    # At a node i, d v_i / d l_j = -v_i exp(l_j) for the gaps j before it, and 0 for the others.
    pid = PROFILE_FORMS.index('table')
    i = 20
    f = lambda p: profile_intensity(jnp.asarray([TABLE_ANGLES_DEG[i]]), p, pid)[0]
    J = np.asarray(jax.grad(f)(jnp.asarray(TABLE, dtype=jnp.float32)))
    expect = np.zeros(len(TABLE))
    expect[:i] = -TABLE_V[i] * np.exp(np.asarray(TABLE)[:i])
    np.testing.assert_allclose(J, expect, rtol=5e-3, atol=1e-6)


def test_table_cubic_returns_its_nodes_and_follows_the_curvature_the_straight_lines_cut():
    cid, tid = PROFILE_FORMS.index('table_cubic'), PROFILE_FORMS.index('table')
    pp = jnp.asarray(TABLE, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(profile_intensity(jnp.asarray(TABLE_ANGLES_DEG), pp, cid)), TABLE_V, rtol=2e-4)
    # Between nodes against the smooth profile the toy table was sampled from: ln v is nearly linear in cos(alpha) there, which the cubic carries and a chord of v does not.
    a = np.linspace(0.5, 45.0, 400)
    truth = (np.exp(-(a / 35.0) ** 2) + 0.01) / 1.01
    err = lambda pid: np.max(np.abs(np.asarray(profile_intensity(jnp.asarray(a, dtype=jnp.float32), pp, pid)) / truth - 1.0))
    assert err(cid) < err(tid)


def test_table_cubic_is_one_on_axis_and_never_rises_for_any_parameters():
    # Monotone for any parameters: no overshoot between the nodes, positive, 1 on axis (to the float32 rounding of the cubic).
    pid = PROFILE_FORMS.index('table_cubic')
    a = jnp.linspace(0.0, 90.0, 2001)
    for seed in range(5):
        p = jnp.asarray(np.random.default_rng(seed).normal(-3.0, 1.5, N_PROFILE_PARAMS), dtype=jnp.float32)
        v = np.asarray(profile_intensity(a, p, pid))
        assert v[0] == pytest.approx(1.0)
        assert np.all(v >= 0)
        assert np.all(np.diff(v) <= 1e-5 * v[:-1] + 1e-30)


def test_table_cubic_gradient_is_finite_for_flat_tables_and_matches_finite_differences():
    # Flat secants (drops that underflow, or are far below the threshold) are the division the guard is for: value, forward and reverse derivatives stay finite.
    pid = PROFILE_FORMS.index('table_cubic')
    n_gaps = len(get_table_nodes()) - 1
    a = jnp.linspace(0.0, 89.9, 300)
    flat = jnp.zeros(N_PROFILE_PARAMS).at[:n_gaps].set(-200.0)
    mixed = jnp.zeros(N_PROFILE_PARAMS).at[:n_gaps].set(jnp.asarray(TABLE)).at[3:9].set(-200.0).at[12:15].set(-60.0)
    np.testing.assert_allclose(np.asarray(profile_intensity(a, flat, pid)), 1.0, atol=1e-6)
    for p in (flat, mixed, jnp.asarray(TABLE, dtype=jnp.float32)):
        assert np.all(np.isfinite(np.asarray(profile_intensity(a, p, pid))))
        assert np.all(np.isfinite(np.asarray(jax.jacfwd(lambda q: profile_intensity(a, q, pid))(p))))
        assert np.all(np.isfinite(np.asarray(jax.grad(lambda q: jnp.sum(profile_intensity(a, q, pid)))(p))))
    # Against a central difference, away from the threshold, for gaps before the angles that see them (the toy table's tail is too dim for a float32 difference).
    angles = jnp.asarray([3.0, 10.0, 22.0, 30.0, 47.0, 61.0, 83.0])
    f = lambda q: jnp.sum(profile_intensity(angles, q, pid))
    p0 = jnp.zeros(N_PROFILE_PARAMS).at[:n_gaps].set(jnp.asarray(TABLE))
    g = jax.grad(f)(p0)
    for i in (1, 3, 5, 8, 10, 12):
        e = jnp.zeros(N_PROFILE_PARAMS).at[i].set(0.02)
        np.testing.assert_allclose(float(g[i]), float((f(p0 + e) - f(p0 - e)) / 0.04), rtol=2e-2, atol=1e-6)


def test_table_cubic_works_at_other_registered_nodes():
    default = get_table_nodes()
    pid = PROFILE_FORMS.index('table_cubic')
    nodes = np.array([0.0, 10.0, 30.0, 50.0, 70.0, 90.0])
    v = np.array([1.0, 0.8, 0.3, 0.1, 0.01, 0.001])
    l = np.log(-np.log(v[1:] / v[:-1]))
    try:
        set_table_nodes(nodes)
        got = np.asarray(profile_intensity(jnp.asarray(nodes, dtype=jnp.float32), jnp.asarray(l, dtype=jnp.float32), pid))
        np.testing.assert_allclose(got, v, rtol=2e-4)
        between = np.asarray(profile_intensity(jnp.linspace(0.0, 90.0, 181), jnp.asarray(l, dtype=jnp.float32), pid))
        assert np.all(between <= 1.0 + 1e-6) and np.all(between >= v[-1] * (1 - 1e-5)) and np.all(np.diff(between) <= 1e-5 * between[:-1])
    finally:
        set_table_nodes(default)
    np.testing.assert_allclose(get_table_nodes(), default)


def test_set_table_nodes_registers_other_nodes_and_can_be_restored():
    default = get_table_nodes()
    pid = PROFILE_FORMS.index('table')
    nodes = np.array([0.0, 10.0, 30.0, 50.0, 70.0, 90.0])
    v = np.array([1.0, 0.8, 0.3, 0.1, 0.01, 0.001])
    l = np.log(-np.log(v[1:] / v[:-1]))
    try:
        set_table_nodes(nodes)
        np.testing.assert_allclose(get_table_nodes(), nodes)
        got = np.asarray(profile_intensity(jnp.asarray(nodes, dtype=jnp.float32), jnp.asarray(l, dtype=jnp.float32), pid))
        np.testing.assert_allclose(got, v, rtol=2e-4)
        for bad in ([0.0, 90.0], [1.0, 40.0, 90.0], [0.0, 50.0, 40.0, 90.0], np.linspace(0.0, 90.0, N_PROFILE_PARAMS + 3)):
            with pytest.raises(ValueError):
                set_table_nodes(bad)
        np.testing.assert_allclose(get_table_nodes(), nodes)                    # a refused set leaves the registered nodes alone
    finally:
        set_table_nodes(default)
    np.testing.assert_allclose(get_table_nodes(), default)
