"""Gradient check for the HK calibration forward: does the fit's Jacobian match the forward?

At truth, for each parameter and diffuser, compares the forward-mode directional derivative of
the per-PMT charge (jax.jvp through the simulator, as CalibrationJacobian's jacfwd computes it)
with a central finite difference of the forward taken on the same keys. Both are averaged over
N_KEYS keys, so each estimates the derivative of the EXPECTED charge; per key they need not agree
(the simulator samples discrete branches), but their means must, within the key-to-key scatter.

Run for two setups with the same truth: 'new' (HK_WCSIM cap PMTs + leg bound, as the closure job now
runs) and 'old' (HK hemisphere PMTs, no leg bound, as the October closures ran). Absorption and
QE are controls. Sums are over the PMTs inside / outside each diffuser's 40 deg cone.
"""
import os
import time


def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import CalibrationParams
    from hk_injector_sources import injector_sources, load_injectors
    from hk_wcsim_params import REFLECTION, WCSIM_BY_WAVELENGTH

    # ---- configuration (edit here) ----
    SETUPS = {'cap, no leg bound': ('../config/HK_WCSIM_geom_config.json', False),
              'hemisphere, leg bound': ('../config/HK_geom_config.json', True)}
    WL = 395
    INJ_IDX = ['5', '28']                     # a barrel and the top diffuser
    INTENSITY = 1e8
    N_RAYS, K = 1_000_000, 12
    GRID = dict(n_cap=200, n_angular=300, n_height=200)
    N_KEYS = 16
    H = 0.1                                   # finite-difference step in the fit's log basis
    CONE_DEG = 40.0

    params = CalibrationParams(n_wavelengths=1, basis='rlogit')
    R = REFLECTION
    theta = jnp.asarray(params.theta_from_physical(
        [WCSIM_BY_WAVELENGTH[WL]], R['wall_reflection_rate'], R['wall_fspec'],
        R['sensor_reflection_rate'], R['sensor_fspec']), dtype=jnp.float32)
    # rlogit layout, one wavelength: [scatter, absorption, qe, wall_R, wall_fspec, sensor_R, sensor_fspec]
    CHECK = {'sensor_R': 5, 'wall_R': 3, 'absorption': 1, 'qe': 2}

    sources = injector_sources('diffuser', 'WarwickDA02', idx=INJ_IDX, intensity=INTENSITY)
    entries = [e for e in load_injectors('diffuser') if e['idx'] in set(INJ_IDX)]
    keys = [jax.random.PRNGKey(70_000 + 17 * k) for k in range(N_KEYS)]

    for setup, (geom, leg) in SETUPS.items():
        t0 = time.perf_counter()
        det = generate_detector(geom)
        pmt = np.asarray(det.all_points)
        ns = len(pmt)
        sim = setup_event_simulator(geom, N_RAYS, temperature=None, K=K, is_calibration=True,
                                    hit_mode='aggregated', wavelength_mode=False,
                                    reflection_model='scalar_mix', deposit_leg_bound=leg, **GRID)
        print(f'\n===== {setup}: {os.path.basename(geom)}, leg bound {leg} =====', flush=True)
        print(f'{"param":11s}{"source":>7s}{"region":>8s}{"charge":>12s}{"jvp mean":>13s}'
              f'{"FD mean":>13s}{"jvp/FD":>8s}{"pull":>7s}')
        for s, (src, e) in enumerate(zip(sources, entries)):
            v = pmt - np.asarray(e['position'], float) / 100.0
            ax = np.asarray(e['direction'], float); ax /= np.linalg.norm(ax)
            cone = (v @ ax) / np.linalg.norm(v, axis=1) >= np.cos(np.deg2rad(CONE_DEG))
            masks = {'cone': jnp.asarray(cone), 'outside': jnp.asarray(~cone)}

            def charge(th, key):
                return sim(src, params.to_dp(th, 0, jnp.ones(ns)), key)[0]
            charge_j = jax.jit(charge)
            jvp_j = jax.jit(lambda th, key, tan: jax.jvp(lambda t: charge(t, key), (th,), (tan,)))

            q0 = np.mean([np.asarray(charge_j(theta, k)) for k in keys], axis=0)
            for name, i in CHECK.items():
                tan = jnp.zeros_like(theta).at[i].set(1.0)
                step = jnp.zeros_like(theta).at[i].set(H)
                per_key = {r: ([], []) for r in masks}
                for k in keys:
                    _, d_jvp = jvp_j(theta, k, tan)
                    d_fd = (charge_j(theta + step, k) - charge_j(theta - step, k)) / (2 * H)
                    for r, m in masks.items():
                        per_key[r][0].append(float(jnp.sum(jnp.where(m, d_jvp, 0.0))))
                        per_key[r][1].append(float(jnp.sum(jnp.where(m, d_fd, 0.0))))
                for r, (a, b) in per_key.items():
                    a, b = np.array(a), np.array(b)
                    d = a - b
                    pull = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-30)
                    print(f'{name:11s}{"dif" + e["idx"]:>7s}{r:>8s}{q0[np.asarray(masks[r])].sum():12.4g}'
                          f'{a.mean():13.4g}{b.mean():13.4g}{a.mean() / (b.mean() + 1e-30):8.3f}'
                          f'{pull:7.1f}', flush=True)
        print(f'[time] {setup}: {time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
