"""Pairwise loss landscapes around truth for the HK multi-wavelength closure, on GPU.

For each pair of fit parameters: the fit's own loss on a 2-D grid of scale factors relative to
truth, every other parameter at truth and the per-PMT gains profiled as the fit profiles them.
The loss is the closure job's cross chi^2, sum r_A * r_B over two independent forward draws: its
expectation is sum E[r]^2 with no Monte-Carlo variance term. Every grid point uses the same two
keys, so the forward's noise is common to the whole landscape and its gradient is smooth.

Reads a closure results file from hk_multiwavelength_job.py (truth, truth gains, data, fit and
its path) and rebuilds that run's forward model; writes the grids for hk_landscape.ipynb.
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--results', required=True, help='closure results .npz to map around')
    ap.add_argument('--out', default='results/hk_landscape.npz')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t_start = time.perf_counter()

    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import CalibrationParams, CalibrationForward
    from lucid.fitting.calib import neyman_residual, profile_gains
    from hk_injector_sources import injector_sources

    # ---- configuration (edit here) ----
    # The forward must be the closure run's: these match hk_multiwavelength_job.py, and the rest
    # (wavelengths, sources, intensity, truth, gains, data) comes from the results file.
    GEOM = '../config/HK_WAND_geom_config.json'
    K, GRID = 12, dict(n_cap=200, n_angular=300, n_height=200)
    # Fewer rays than the fit: the shared keys keep the landscape smooth, and the cross chi^2 is
    # unbiased at any ray count.
    N_RAYS = 250_000
    N_GRID = 15
    PAIRS = [('scatter_length_395', 'absorption_length_395'),
             ('absorption_length_395', 'qe_395'),
             ('absorption_length_395', 'wall_R'),
             ('absorption_length_395', 'sensor_R'),
             ('wall_R', 'sensor_R'),
             ('qe_395', 'sensor_R')]
    SPAN = {'wall_R': (0.5, 1.5), 'sensor_R': (0.5, 1.5)}   # scale ranges; others DEFAULT_SPAN
    DEFAULT_SPAN = (0.8, 1.2)
    KEYS = (80_000, 85_000)            # the two forward draws, shared by every grid point

    r = np.load(args.results)
    names = list(r['names'])
    truth = np.asarray(r['truth'], dtype=float)
    wls = [int(w) for w in r['wavelengths']]
    labels = [str(s) for s in r['source_labels']]
    W = len(wls)
    sources = injector_sources('diffuser', 'WarwickDA02', idx=[s[3:] for s in labels],
                               intensity=float(r['intensity']))
    NS = int(r['truth_k'].shape[0])
    data = jnp.asarray(np.asarray(r['truth_charge']).reshape(W * len(sources), NS))
    frac = float(r['frac']) if 'frac' in r.files else 0.0
    q_floor = 0.01 * float(jnp.mean(data))             # the closure job's floor
    data_sum = data.sum(0)

    sim = setup_event_simulator(GEOM, N_RAYS, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,
                                reflection_model='scalar_mix', deposit_leg_bound=True, **GRID)
    params = CalibrationParams(n_wavelengths=W, basis='rlogit')
    NG = 3 * W
    refl = dict(zip(names[NG:], truth[NG:]))
    theta_true = np.asarray(params.theta_from_physical(
        [dict(scatter_length=truth[3 * w], absorption_length=truth[3 * w + 1], qe=truth[3 * w + 2])
         for w in range(W)],
        refl['wall_R'], refl['wall_fspec'], refl['sensor_R'], refl['sensor_fspec']), dtype=float)
    # Every parameter mapped here is log-basis in the fit vector (CalibrationParams 'rlogit':
    # lengths, QE and reflection rates), so a scale s is a shift log(s) at the parameter's index.
    index = {n: i for i, n in enumerate(names)}
    for a, b in PAIRS:
        for n in (a, b):
            if n not in index or n.endswith('_fspec'):
                raise ValueError(f'{n!r} is not a log-basis fit parameter of this run')

    fwd = CalibrationForward(sim, sources, params, NS)
    ones = jnp.ones(NS)

    def resid(theta, key_base):
        mu = fwd.average(jnp.asarray(theta, dtype=jnp.float32), key_base, ones, 1)
        k = profile_gains(mu.sum(0) + 1e-12, data_sum, gauge='linear')
        return neyman_residual(k[None, :] * mu, data, q_floor, frac)

    def loss(theta):
        return float(jnp.sum(resid(theta, KEYS[0]) * resid(theta, KEYS[1])))

    t = time.perf_counter()
    loss_truth = loss(theta_true)
    print(f'loss at truth {loss_truth:.6g} ({time.perf_counter() - t:.1f} s per point)', flush=True)

    scales, losses = [], []
    for a, b in PAIRS:
        t = time.perf_counter()
        sa = np.linspace(*SPAN.get(a, DEFAULT_SPAN), N_GRID)
        sb = np.linspace(*SPAN.get(b, DEFAULT_SPAN), N_GRID)
        grid = np.empty((N_GRID, N_GRID))              # [i_b, i_a]: rows are y, columns x
        for i, s_b in enumerate(sb):
            for j, s_a in enumerate(sa):
                th = theta_true.copy()
                th[index[a]] += np.log(s_a)
                th[index[b]] += np.log(s_b)
                grid[i, j] = loss(th)
        scales.append((sa, sb))
        losses.append(grid)
        print(f'{a} x {b}: {time.perf_counter() - t:.0f} s', flush=True)

    out = dict(pairs=np.array(PAIRS), scale_x=np.array([s[0] for s in scales]),
               scale_y=np.array([s[1] for s in scales]), loss=np.array(losses),
               loss_truth=loss_truth, names=np.array(names), truth=truth,
               n_rays=N_RAYS, keys=np.array(KEYS), results=args.results)
    for key in ('fit', 'history', 'sigma', 'fisher_names'):
        if key in r.files:
            out[key] = r[key]
    np.savez(args.out, **out)
    print(f'saved -> {args.out}  [time] total {time.perf_counter() - t_start:.0f} s')


if __name__ == '__main__':
    main()
