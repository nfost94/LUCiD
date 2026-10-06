"""Run the calibration-gradients pipeline (loss landscape, Hessian/Fisher/CRB,
before/after fit) on GPU as a standalone process, on HK geometry.

Replicates tutorials/calibration_gradients.ipynb but on HK_geom_config.json (with
a grid ~2x finer, since HK is ~2x SK_like's linear size) and runs fully inside its
own process so the CUDA context / JAX preallocated GPU memory pool is released on
exit, instead of being pinned to the notebook's kernel for the rest of the session.
"""
import argparse
import os
import time
from contextlib import contextmanager

TIMINGS = {}


@contextmanager
def timed(label):
    t0 = time.perf_counter()
    print(f'[time] {label} ...', flush=True)
    yield
    TIMINGS[label] = time.perf_counter() - t0
    print(f'[time] {label}: {TIMINGS[label]:.1f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='calibration_gradients_results.npz')
    ap.add_argument('--geom', default='../config/HK_geom_config.json')
    ap.add_argument('--seed', type=int, default=0,
                    help='selects the truth key (PRNGKey(1 + seed)), the fit key streams and the start perturbation')
    ap.add_argument('--steps', type=int, default=150, help='Gauss-Newton fit steps')
    ap.add_argument('--frac', type=float, default=0.03,
                    help='fractional uncertainty in the fit/CRB variance, Q + (frac*Q)^2; 0 = pure Poisson')
    ap.add_argument('--sweeps', choices=['none', '1d', 'all'], default='1d',
                    help='loss-landscape sweeps to run: none, 1D only, or 1D + 2D')
    ap.add_argument('--sweep2d-points', type=int, default=15, help='2D sweep grid points per axis')
    ap.add_argument('--fixed-key', action='store_true',
                    help='deterministic closure: every forward (fit, Jacobian, CRB) reuses the truth key')
    ap.add_argument('--refresh', type=int, default=15, help='fit Jacobian refresh cadence (steps)')
    ap.add_argument('--max-step', type=float, default=0.08, help='fit per-step clip in log-parameter space')
    ap.add_argument('--lam', type=float, default=0.01, help='fit Gauss-Newton damping')
    ap.add_argument('--jacobian-draws', type=int, default=2,
                    help='random draws averaged into each fit Jacobian refresh')
    ap.add_argument('--scenario', choices=['hk', 'tutorial'], default='hk',
                    help='hk: WCSim-395nm optics + injector sources chosen in `named`; '
                         'tutorial: tutorials/calibration_gradients optics + 3 lasers & isotropic, on --geom')
    ap.add_argument('--optics', choices=['wcsim', 'tutorial'], default=None,
                    help="override the scenario's truth optics (and reflection model) independently of its sources")
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t_start = time.perf_counter()

    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.losses import WC_smooth_loss
    from lucid.fitting import build_calibration_problem, fit, crb
    from lucid.fitting.schur_gn import SourceModel
    from dataclasses import replace
    from tqdm import tqdm
    from lucid.gradient_analysis import SweepParam
    from lucid.gradient_analysis.sweep import (set_param_value, get_grad_component,
                                               numerical_gradient, find_zero_crossing)
    from lucid.sources import laser_source, isotropic_source
    from lucid.detector_params import DetectorParams
    from hk_injector_sources import injector_sources, load_injectors, injector_position_direction
    from hk_wcsim_params import wcsim_395nm_params, REFLECTION_MODEL

    print(jax.devices())          # should list CudaDevice, not CpuDevice
    print(jax.default_backend())  # should print 'gpu'

    with timed('setup: detector, sources, simulator'):
        det = generate_detector(args.geom); NS = len(det.all_points); pts = jnp.asarray(det.all_points)

        dp = wcsim_395nm_params(NS)

        # Real HK injector positions/directions (UKLIinfo/LightInjectorsDetails.json): at three barrel
        # (5, 14, 23) and the top (28) / bottom (32) cap positions, a diffuser (WarwickDA02) and a
        # collimator (WarwickCol_C01_repeat). Each list comes back in survey-file order.
        INJ_IDX = ['5', '14', '23', '28', '32']
        INJ_IDX = [str(i) for i in range(28)] + ['28', '32']

        INTENSITY_diffuser = 1e7
        INTENSITY_collimator = 1e6
        INTENSITY_laser = 1e6

        diffusers = injector_sources('diffuser', 'WarwickDA02', idx=INJ_IDX, intensity=INTENSITY_diffuser)
        collimators = injector_sources('collimator', 'WarwickCol_C01_repeat', idx=INJ_IDX, intensity=INTENSITY_collimator)

        def labels(kind, tag):  # same survey-file order injector_sources returns
            return [f'{tag}{e["idx"]}' for e in load_injectors(kind) if e['idx'] in set(INJ_IDX)]

        # The tutorial's laser_source (NA 0.22) at each collimator's position and direction.
        laser_idx = [lab[3:] for lab in labels('collimator', 'col')]
        lasers = [laser_source(position=p, direction=d, intensity=INTENSITY_laser)
                  for p, d in (injector_position_direction('collimator', i) for i in laser_idx)]

        # Choose the source set here; labels stay paired with their sources.
        # Collimators instead of lasers: list(zip(labels('collimator', 'col'), collimators))
        
        #named = (list(zip([f'las{i}' for i in laser_idx], lasers))
        #         + list(zip(labels('diffuser', 'dif'), diffusers)))
        named = list(zip(labels('diffuser', 'dif'), diffusers))
        refl_model = REFLECTION_MODEL

        # tutorials/calibration_gradients.ipynb on this geometry: its toy optics and the default
        # reflection model; its sources are lasers down / up / in from the barrel wall + a flasher.
        optics = args.optics or ('tutorial' if args.scenario == 'tutorial' else 'wcsim')
        if optics == 'tutorial':
            dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                          wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                          absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
            refl_model = 'scalar_mix'
        if args.scenario == 'tutorial':
            top, bot, R = det.H / 2 - .1, -det.H / 2 + .1, det.r
            named = [('las_top', laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6)),
                     ('las_bot', laser_source(position=[0, 0, bot], direction=[0, 0, 1], intensity=1e6)),
                     ('las_wall', laser_source(position=[R - .1, 0, 0], direction=[-1, 0, 0], intensity=1e6)),
                     ('iso', isotropic_source(position=[0, 0, 0], intensity=1e6))]

        source_labels = [n for n, _ in named]
        sources = [s for _, s in named]
        print(f'scenario {args.scenario}, optics {optics}: sources {source_labels}, reflection model {refl_model}')

        GRID = dict(n_cap=200, n_angular=300, n_height=200)  # finer grid: HK is ~2x SK_like's linear size
        sim = setup_event_simulator(args.geom, 1_000_000, temperature=None, K=8, is_calibration=True,
                                    hit_mode='aggregated', wavelength_mode=False,
                                    reflection_model=refl_model, **GRID)
        FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length', 'wall_reflection_rate', 'sensor_reflection_rate', 'qe']

    TRUTH_KEY = jax.random.PRNGKey(1 + args.seed)
    with timed(f'build_calibration_problem: truth charge for {len(sources)} sources (incl. JIT)'):
        prob = build_calibration_problem(sim, sources, dp, FIELDS, key=TRUTH_KEY)
    print(f'{NS} PMTs | {len(sources)} calibration sources | {len(FIELDS)} global parameters')

    if args.fixed_key:
        # Same forward as build_calibration_problem's, but ignoring the per-step keys it is handed:
        # at the true parameters the model then reproduces truth_charge exactly (zero residual).
        def fixed_key_forward(src):
            def forward(theta_log, ek, pk):
                return sim(src, prob['unravel'](theta_log, 1.0), TRUTH_KEY)[0]
            return forward
        prob['source_models'] = [SourceModel(fixed_key_forward(s), eps=1e-8) for s in sources]
        print('fixed-key closure: every forward reuses the truth key (no Monte-Carlo noise between data and model)')

    # ---- 1. Gradient: loss landscape (optional, --sweeps) ----
    # Sweeps record every source separately; the saved `_losses`/`_gradients` (and losses2d etc.)
    # are their mean, the `_src_` arrays are per source in `source_labels` order.
    save_sweeps = {}
    if args.sweeps != 'none':
        with timed(f'sweep setup: truth events for {len(sources)} sources + loss/grad JIT warm-up'):
            sim_truth = setup_event_simulator(args.geom, 1_000_000, temperature=None, K=8, is_calibration=True,
                          hit_mode='aggregated', wavelength_mode=False, default_detector_params=dp,
                          reflection_model=refl_model, **GRID)
            true_data = [jax.block_until_ready(jax.lax.stop_gradient(sim_truth(s, jax.random.PRNGKey(9))))
                         for s in sources]

            @jax.jit
            def source_loss_and_grad(p, s, td):
                def L(x):
                    return WC_smooth_loss(pts, *td, *sim(s, x, jax.random.PRNGKey(3)),
                                          lambda_poisson=1.0, lambda_time=0.0, tau=0.5)
                return jax.value_and_grad(L)(p)

            def per_source(p):
                return [source_loss_and_grad(p, s, td) for s, td in zip(sources, true_data)]
            jax.block_until_ready(per_source(dp))

        def sweep_1d_per_source(sp):
            sp = sp.resolve(dp)
            vals = np.asarray(sp.values)
            L = np.full((len(sources), len(vals)), np.nan); G = np.full_like(L, np.nan)
            for i, v in enumerate(tqdm(vals, desc=sp.name)):
                for s, (l, g) in enumerate(per_source(set_param_value(dp, sp, float(v)))):
                    L[s, i], G[s, i] = float(l), get_grad_component(g, sp)
            return sp, vals, L, G

        def sweep_2d_per_source(px, py, num_points):
            px = replace(px.resolve(dp), num_points=num_points)
            py = replace(py.resolve(dp), num_points=num_points)
            xv, yv = np.asarray(px.values), np.asarray(py.values)
            L = np.full((len(sources), len(xv), len(yv)), np.nan); GX = L.copy(); GY = L.copy()
            for i, vx in enumerate(tqdm(xv, desc=f'{px.name} x {py.name}')):
                for j, vy in enumerate(yv):
                    p = set_param_value(set_param_value(dp, px, float(vx)), py, float(vy))
                    for s, (l, g) in enumerate(per_source(p)):
                        L[s, i, j] = float(l)
                        GX[s, i, j], GY[s, i, j] = get_grad_component(g, px), get_grad_component(g, py)
            return px, py, xv, yv, L, GX, GY

        sweeps = [SweepParam('Scatter Length', 'scatter_length', half_width=30., unit='m', min_val=0.001, grad_scale=100.),
                  SweepParam('Absorption Length', 'absorption_length', half_width=200., unit='m', min_val=25., grad_scale=100.),
                  SweepParam('Wall Reflection', 'wall_reflection_rate', half_width=0.1, min_val=0., max_val=1., grad_scale=0.1),
                  SweepParam('QE', 'qe', half_width=0.05, min_val=0.001, grad_scale=0.1)]
        with timed('1D sweeps'):
            res1d = [sweep_1d_per_source(sp) for sp in sweeps]
        save_sweeps['sweep1d_order'] = np.array([sp.field for sp, *_ in res1d])
        for sp, vals, L, G in res1d:
            key = sp.field
            save_sweeps[f'{key}_values'] = vals
            save_sweeps[f'{key}_losses'] = L.mean(0)
            save_sweeps[f'{key}_gradients'] = G.mean(0)
            save_sweeps[f'{key}_numerical_grads'] = numerical_gradient(L.mean(0), vals)
            save_sweeps[f'{key}_center'] = np.float64(sp.center)
            save_sweeps[f'{key}_label'] = sp.label
            save_sweeps[f'{key}_src_losses'] = L
            save_sweeps[f'{key}_src_gradients'] = G

        print('\nper-source sweep minimum (AD-gradient zero crossing) as offset from truth; nan = none in range')
        print(f'{"source":10s}' + ''.join(f'{sp.field:>24s}' for sp, *_ in res1d))
        for s, lab in list(enumerate(source_labels)) + [(None, 'mean')]:
            cells = ''
            for sp, vals, L, G in res1d:
                zc = find_zero_crossing(vals, G.mean(0) if s is None else G[s])
                cells += f'{zc / sp.center - 1:+24.1%}'
            print(f'{lab:10s}{cells}')

        # ---- 2D landscape: absorption length x QE (a parameter degeneracy) ----
        if args.sweeps == 'all':
            with timed(f'2D sweep ({args.sweep2d_points}x{args.sweep2d_points})'):
                px2, py2, x2d, y2d, L2, GX2, GY2 = sweep_2d_per_source(
                    sweeps[1], sweeps[3], num_points=args.sweep2d_points)
            save_sweeps.update(
                x2d_values=x2d, y2d_values=y2d,
                losses2d=L2.mean(0), gradx2d=GX2.mean(0), grady2d=GY2.mean(0),
                losses2d_src=L2, gradx2d_src=GX2, grady2d_src=GY2,
                x2d_label=px2.label, y2d_label=py2.label,
                x2d_name=px2.name, y2d_name=py2.name,
                x2d_center=px2.center, y2d_center=py2.center,
                x2d_grad_scale=px2.grad_scale, y2d_grad_scale=py2.grad_scale)

    # ---- 2. Hessian / Fisher / CRB ----
    with timed('CRB at truth'):
        c = jax.block_until_ready(crb(prob['source_models'], prob['theta_true'], NS, frac=args.frac))
    F, cov, sigma = c['fisher'], c['cov'], c['sigma']

    # ---- 3. Before / after the fit ----
    start = prob['theta_true'] + np.random.default_rng(args.seed).uniform(-.15, .15, prob['theta_true'].shape)
    with timed(f'fit ({args.steps} steps)'):
        res = fit(prob['source_models'], prob['truth_charge'], start, NS, steps=args.steps,
                  refresh=args.refresh, max_step=args.max_step, lam=args.lam,
                  jacobian_draws=args.jacobian_draws, frac=args.frac, seed=args.seed)
    print(f'[time] fit: {TIMINGS[f"fit ({args.steps} steps)"] / args.steps:.2f} s per step on average')
    truth = np.exp(prob['theta_true'])

    print(f'{"param":22s}{"truth":>10s}{"start":>10s}{"fit":>10s}{"err":>8s}{"CRB":>8s}')
    for i, f in enumerate(FIELDS):
        e = res['theta'][i] / truth[i] - 1
        print(f'{f:22s}{truth[i]:10.3f}{np.exp(start[i]):10.3f}{res["theta"][i]:10.3f}{e:+7.1%}{sigma[i]:8.1%}')

    # before/after Hessian check: is the Fisher stable across the basin?
    with timed('CRB at start'):
        c0 = jax.block_until_ready(crb(prob['source_models'], start, NS, frac=args.frac))

    total = time.perf_counter() - t_start
    print(f'\n[time] summary (total {total:.1f} s)')
    for label, dt in TIMINGS.items():
        print(f'  {label:58s} {dt:8.1f} s  {dt / total:6.1%}')

    np.savez(args.out,
             fields=np.array(FIELDS), ns=NS, source_labels=np.array(source_labels),
             fisher=np.asarray(F), cov=np.asarray(cov), sigma=np.asarray(sigma),
             start=np.exp(start), fit_theta=np.asarray(res['theta']), truth=truth,
             # fit() returns history in log space with the start as row 0
             history=np.exp(np.asarray(res['history'])[1:]), sigma_at_start=np.asarray(c0['sigma']),
             # profiled per-PMT gains at the fit (closure truth is 1) and each source's truth charge
             gains=np.asarray(res['k']), truth_charge=np.stack([np.asarray(q) for q in prob['truth_charge']]),
             frac=args.frac,
             **save_sweeps)
    print(f'saved results -> {args.out}')


if __name__ == '__main__':
    main()
