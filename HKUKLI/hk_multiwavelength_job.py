"""Multi-wavelength HK closure calibration, on GPU as a standalone process.

The paper's calibration recipe (analysis/paper/utils/calib_run.py: CalibrationParams,
closure_data, calibrate) on HK geometry, with the HK laser wavelengths, WCSim optics and real
injector diffusers. Scattering, absorption and QE are free per wavelength; wall / PMT reflection
(rate and specular fraction) and the per-PMT gains are shared across wavelengths. Mie is off.
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='hk_multiwavelength_results.npz')
    ap.add_argument('--seed', type=int, default=0, help='fit start and fit key streams')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t_start = time.perf_counter()

    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import CalibrationParams, CalibrationForward, calibrate, closure_data
    from hk_injector_sources import injector_sources, load_injectors
    from hk_wcsim_params import WCSIM_BY_WAVELENGTH

    # ---- configuration (edit here) ----
    GEOM = '../config/HK_geom_config.json'
    WAVELENGTHS = [337, 355, 375, 395, 440, 500]       # HK laser system
    INJ_IDX = ['5', '14', '23', '28', '32']             # diffusers: 3 barrel + top + bottom
    INTENSITY = 1e8                                      # photons per source per wavelength
    N_RAYS, K = 1_000_000, 12
    GRID = dict(n_cap=200, n_angular=300, n_height=200)
    # WCSim reflection, shared across wavelengths: blacksheet 0.045 x bsrff 2.5 (diffuse) and
    # glass/cathode rgcff 0.32 (specular). Fractions sit just inside (0, 1): the fit uses logit(f).
    WALL_R, WALL_FSPEC, SENSOR_R, SENSOR_FSPEC = 0.1125, 0.01, 0.32, 0.99
    FIT_SPECULAR_FRACTIONS = False                       # False: hold f_w, f_s at truth
    QE_SPREAD = 0.05                                     # truth per-PMT gain spread (paper value)
    # paper recipe (calib_run.run_seed defaults)
    STEPS, REFRESH, JACOBIAN_DRAWS, TRUTH_DRAWS, POLYAK = 200, 20, 8, 8, 50
    # Forward draws averaged per step: the profiled gains are nonlinear in the forward, so its MC
    # noise biases the fixed point ~1/N_FORWARD (CalibrationForward.average). Paper: 1.
    N_FORWARD = 4
    # Jacobian refresh schedule: every REFRESH_EARLY steps for the first REFRESH_SWITCH fraction of
    # the run (while parameters move fast), then every REFRESH. REFRESH_EARLY = REFRESH: constant.
    REFRESH_EARLY, REFRESH_SWITCH = 5, 0.2
    MAX_STEP, LAM, MU = 0.5, 0.01, 0.1
    # Fractional variance (frac*Q)^2 added to the Neyman weight: caps each bright PMT's pull at
    # ~1/FRAC^2 Poisson-equivalent counts (near-injector hot spots). 0: the published estimator.
    FRAC = 0.0
    PERT, PERT_SCATTER, PERT_REFL = 0.4, 2.0, 1.2       # start offsets, log / logit space
    EVAL_DRAWS = 16                                      # chi2(fit) vs chi2(truth) check, 0: skip
    FISHER_DRAWS = 8                                     # Fisher/correlations at truth, 0: skip
    FIT = True                                           # False: Fisher only (fast layout compare)

    print(jax.devices())
    det = generate_detector(GEOM); NS = len(det.all_points)
    sources = injector_sources('diffuser', 'WarwickDA02', idx=INJ_IDX, intensity=INTENSITY)
    source_labels = [f'dif{e["idx"]}' for e in load_injectors('diffuser') if e['idx'] in set(INJ_IDX)]

    sim = setup_event_simulator(GEOM, N_RAYS, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,
                                reflection_model='scalar_mix', **GRID)

    W = len(WAVELENGTHS)
    params = CalibrationParams(n_wavelengths=W, basis='rlogit')
    NG, P = params.NG, params.P
    truths = [WCSIM_BY_WAVELENGTH[w] for w in WAVELENGTHS]
    theta_true = params.theta_from_physical(truths, WALL_R, WALL_FSPEC, SENSOR_R, SENSOR_FSPEC)

    names = [f'{q}_{w}' for w in WAVELENGTHS for q in ('scatter_length', 'absorption_length', 'qe')]
    names += ['wall_R', 'wall_fspec', 'sensor_R', 'sensor_fspec']

    def physical(theta):
        theta = np.asarray(theta, dtype=float)
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        return np.concatenate([np.exp(theta[:NG]),
                               [np.exp(theta[NG]), sig(theta[NG + 1]),
                                np.exp(theta[NG + 2]), sig(theta[NG + 3])]])

    truth = np.array([t[q] for t in truths for q in ('scatter_length', 'absorption_length', 'qe')]
                     + [WALL_R, WALL_FSPEC, SENSOR_R, SENSOR_FSPEC])

    tk = np.exp(np.random.default_rng(12345).normal(0.0, QE_SPREAD, NS))
    truth_k = tk / np.exp(np.mean(np.log(tk)))

    print(f'{NS} PMTs | {len(sources)} sources x {W} wavelengths | {P} parameters', flush=True)
    fwd = CalibrationForward(sim, sources, params, NS)
    t = time.perf_counter()
    data = jax.block_until_ready(
        closure_data(fwd, theta_true, jnp.asarray(truth_k), n_draws=TRUTH_DRAWS))
    print(f'[time] truth ({TRUTH_DRAWS} draws): {time.perf_counter() - t:.0f} s', flush=True)

    rng = np.random.default_rng(4000 + args.seed)
    start = theta_true + rng.uniform(-PERT, PERT, P)
    for w in range(W):
        start[3 * w] = theta_true[3 * w] + rng.uniform(-PERT_SCATTER, PERT_SCATTER)
    for j in range(NG, P):
        start[j] = theta_true[j] + rng.uniform(-PERT_REFL, PERT_REFL)
    fix = () if FIT_SPECULAR_FRACTIONS else (NG + 1, NG + 3)
    for j in fix:
        start[j] = theta_true[j]
    free = np.array([j not in fix for j in range(P)])
    q_floor = 0.01 * float(jnp.mean(data))               # calibrate's default (q_floor_frac 0.01)
    out = dict(names=np.array(names), truth=truth, free=free, wavelengths=np.array(WAVELENGTHS),
               source_labels=np.array(source_labels), truth_k=truth_k, intensity=INTENSITY,
               n_rays=N_RAYS, seed=args.seed, frac=FRAC,
               n_forward=N_FORWARD,
               truth_charge=np.asarray(data).reshape(W, len(sources), NS))   # (wl, source, PMT)

    # Fisher at truth with the per-PMT gains marginalised (Schur complement, gauge mean log k = 0).
    # r = (kM - Q)/sqrt(var), var = neyman_variance(Q, floor, FRAC), is unit-variance, so F = J^T J
    # with J the fit's own Jacobian (weighted by the same var); d r / d log k_p ~ Q / sqrt(var) at
    # truth. Free params are all log-basis, so sigma is fractional. Raw engine bound: x sqrt(12)
    # for the honest CRB.
    from lucid.fitting.calib import CalibrationJacobian, neyman_variance
    if FISHER_DRAWS:
        from lucid.fitting.schur_gn import make_constrained_schur
        t = time.perf_counter()
        var = neyman_variance(data, q_floor, FRAC)
        jac = CalibrationJacobian(sim, sources, params, NS, key0=8_000_000)
        J = np.asarray(jac(jnp.asarray(theta_true, dtype=jnp.float32), jnp.log(jnp.asarray(truth_k)),
                           0, var, q_floor, draws=range(FISHER_DRAWS)), dtype=np.float64)
        Q = np.asarray(data, dtype=np.float64)
        a = Q / np.sqrt(np.asarray(var, dtype=np.float64))               # (G, NS)
        Htt = np.einsum('gnp,gnq->pq', J, J)
        Htk = np.einsum('gnp,gn->pn', J, a)
        Minv = make_constrained_schur((a ** 2).sum(0) + 1e-12)
        F = (Htt - Htk @ Minv(Htk.T))[np.ix_(free, free)]
        cov = np.linalg.inv(F)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(sig, sig)
        fn = list(np.array(names)[free])
        print(f'\n[time] Fisher ({FISHER_DRAWS} Jacobian draws): {time.perf_counter() - t:.0f} s')
        print(f'{"param":24s}{"sigma":>9s}  most correlated with')
        for i, n in enumerate(fn):
            c = corr[i].copy(); c[i] = 0; j = np.abs(c).argmax()
            print(f'{n:24s}{sig[i]:9.2%}  {fn[j]} ({c[j]:+.2f})')
        print(f'\n{"corr with":24s}' + ''.join(f'{w:>8d}' for w in WAVELENGTHS))
        for r in ('wall_R', 'sensor_R'):
            if r in fn:
                print(f'abs_len x {r:14s}' + ''.join(
                    f'{corr[fn.index(f"absorption_length_{w}"), fn.index(r)]:+8.2f}' for w in WAVELENGTHS))
        if 'wall_R' in fn and 'sensor_R' in fn:
            print(f'wall_R x sensor_R: {corr[fn.index("wall_R"), fn.index("sensor_R")]:+.2f}')
        out.update(fisher=F, cov=cov, sigma=sig, corr=corr, fisher_names=np.array(fn))

    if not FIT:
        np.savez(args.out, **out)
        print(f'saved Fisher -> {args.out}')
        return

    t_fit = time.perf_counter()

    def progress(step, theta, g, H, loss):
        if step % 25 and step != STEPS - 1:
            return
        err = np.abs(physical(theta) / truth - 1)[free]
        print(f'  step {step:4d}  chi2 {float(loss):.5g}  worst {err.max():.1%} '
              f'({np.array(names)[free][err.argmax()]})  {time.perf_counter() - t_fit:.0f} s', flush=True)

    res = calibrate(sim, sources, params, data, start, steps=STEPS, max_step=MAX_STEP,
                    refresh=REFRESH_EARLY, refresh_final=REFRESH, refresh_switch=REFRESH_SWITCH,
                    jacobian_draws=JACOBIAN_DRAWS, n_forward_draws=N_FORWARD,
                    gauge='linear', seed=args.seed,
                    readout='polyak', polyak=POLYAK, lam=LAM, mu=MU, fix=fix, forward=fwd,
                    frac=FRAC, on_step=progress)
    print(f'[time] fit ({STEPS} steps): {time.perf_counter() - t_fit:.0f} s', flush=True)

    fit = physical(res['theta'])
    k_hat = np.asarray(res['gains'])
    print(f'\n{"param":24s}{"truth":>10s}{"start":>10s}{"fit":>10s}{"err":>9s}')
    for n, tr, st, ft, fr in zip(names, truth, physical(start), fit, free):
        print(f'{n:24s}{tr:10.4g}{st:10.4g}{ft:10.4g}{ft / tr - 1:+9.1%}' + ('' if fr else '  (fixed)'))
    print(f'per-PMT gain residual RMS: {np.std(k_hat / truth_k - 1):.2%}')

    # Bias or convergence? The fit's own objective (same q_floor, gauge, Neyman weight) at truth
    # and at the fit. sum r^2 from one forward draw also counts the forward's MC variance, which
    # moves with theta (the gains rescale each ray's weight) and swamps the fit-quality difference.
    # So use the cross product of two INDEPENDENT draws, sum r_A * r_B: its expectation is
    # sum E[r]^2 with no variance term. The same key pair at both points cancels most of the noise
    # in the difference. Delta < 0 (fit lower): the objective's minimum is displaced from truth.
    # Delta > 0: truth is better and the optimiser hasn't got there.
    from lucid.fitting.calib import profile_gains, neyman_residual
    ones, data_sum = jnp.ones(NS), data.sum(0)

    def resid(theta, key_base):
        mu = fwd.average(jnp.asarray(theta, dtype=jnp.float32), key_base, ones, 1)
        k = profile_gains(mu.sum(0) + 1e-12, data_sum, gauge='linear')
        return neyman_residual(k[None, :] * mu, data, q_floor, FRAC)

    def chi2(theta, ka, kb):
        return float(jnp.sum(resid(theta, ka) * resid(theta, kb)))

    chi2_truth, chi2_fit = np.zeros(EVAL_DRAWS), np.zeros(EVAL_DRAWS)
    if EVAL_DRAWS:
        t = time.perf_counter()
        for d in range(EVAL_DRAWS):
            ka, kb = 80_000 + 10_000 * d, 85_000 + 10_000 * d
            chi2_truth[d], chi2_fit[d] = chi2(theta_true, ka, kb), chi2(res['theta'], ka, kb)
        delta = chi2_fit - chi2_truth
        print(f'\nchi2 (cross) truth {chi2_truth.mean():.6g}  fit {chi2_fit.mean():.6g}  '
              f'(over {data.size} bins, {EVAL_DRAWS} paired draw pairs)')
        print(f'delta = fit - truth: {delta.mean():+.1f} +- {delta.std(ddof=1) / np.sqrt(EVAL_DRAWS):.1f}'
              f'  ({(delta < 0).sum()}/{EVAL_DRAWS} draws favour the fit)')
        print(f'[time] chi2 check: {time.perf_counter() - t:.0f} s')
    print(f'[time] total: {time.perf_counter() - t_start:.0f} s')

    out.update(start=physical(start), fit=fit,
               history=np.stack([physical(t) for t in res['history'][1:]]),
               loss=np.asarray(res['loss']), gains=k_hat, steps=STEPS,
               chi2_truth=chi2_truth, chi2_fit=chi2_fit)
    np.savez(args.out, **out)
    print(f'saved results -> {args.out}')


if __name__ == '__main__':
    main()
