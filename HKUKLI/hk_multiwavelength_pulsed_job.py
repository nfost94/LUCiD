"""Multi-wavelength HK closure calibration on data-like data, on GPU as a standalone process.

hk_multiwavelength_job.py with the data recorded as the detector would: each diffuser is pulsed
N_PULSES times at PULSE_PHOTONS photons per pulse (~1% occupancy in its 40 deg cone), and every
pulse is digitized (the HK_WCSIM config's digitizer: SPE spectrum, WCSim threshold curve, charge
precision) with dark noise over the full event readout, then summed per PMT. The fit models the
same readout (lucid.fitting.PulseResponse) and weights by the model's variance of the recorded
charge (IRLS, refreshed between stages; the data's measured pulse-to-pulse variance is recorded but
not used as a weight). Scattering, absorption and QE are free per wavelength; the wall reflection rate and the
per-PMT gains are shared across wavelengths. PMT reflection is held at its known value.
Mie is off.

Reflection is the tabulated model with WCSim's surfaces ('tabulated_lambertian'), as in
hk_multiwavelength_job.py: the PMTs are WCSim's default photocathode tabulated
(config/pmt/hk50cm_wcsim_optics.txt.gz), its reflectance R(theta) times sensor_R (held at 1),
specular, with detection from its absorptance A(theta)/A(0); the wall is WCSim's blacksheet, a flat
reflectance wall_R with Lambertian reflection.
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='results/hk_multiwavelength_pulsed_results.npz')
    ap.add_argument('--seed', type=int, default=0, help='fit start and fit key streams')
    ap.add_argument('--resume', default=None,
                    help="start from this results .npz's fit instead of the perturbed start. A "
                         "continuation only if the truth settings (TRUTH_DRAWS, N_RAYS, INJ_IDX, "
                         "optics, QE, PULSE_PHOTONS, N_PULSES) are unchanged; use a new --seed. STEPS = extra steps")
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t_start = time.perf_counter()

    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import (CalibrationParams, CalibrationForward, PulseResponse, calibrate,
                               pulsed_closure_data)
    from lucid.simulation.digitizer import resolve_model_config
    from hk_injector_sources import injector_sources, load_injectors
    from hk_wcsim_params import DIGITIZER, READOUT_NS, REFLECTION, WCSIM_BY_WAVELENGTH

    # ---- configuration (edit here) ----
    # The HK WCSim setup (HK_WCSIM configs): hk50cm PMTs (the config's sensor_shape), whose hard-mode deposit is exact
    # only with the leg bound on (as in production); optics truth from HK_WCSIM_physics_config.json.
    GEOM = '../config/HK_WCSIM_geom_config.json'
    # WCSim's PMT, tabulated. The tabulated models find their table through this variable; the
    # reflection is evaluated at one wavelength for every laser slice (model 0 varies with
    # wavelength only through the water and glass dispersion).
    PMT_OPTICS = '../config/pmt/hk50cm_wcsim_optics.txt.gz'
    PMT_DETECTION = '../config/pmt/hk50cm_wcsim_detection_400nm.npz'
    REFLECTION_WAVELENGTH = 400.0
    WAVELENGTHS = [337]  # , 355, 375, 395, 440, 500]  # HK laser system
    INJ_IDX = ['5', '14', '23', '28', '32']             # diffusers: 3 barrel + top + bottom
    # Photons per diffuser pulse for ~1% occupancy of the PMTs inside the diffuser's 40 deg
    # half-angle cone (HK_WCSIM setup, digitizer threshold included; within ~30% of every diffuser).
    # The per-pulse mean sets the digitizer's regime and N_PULSES the statistics: the light per
    # source and wavelength is N_PULSES x PULSE_PHOTONS[w].
    PULSE_PHOTONS = {337: 2500, 355: 2000, 375: 1500, 395: 1500, 440: 2000, 500: 6000}
    N_PULSES = 2000
    N_RAYS, K = 1_000_000, 12
    GRID = dict(n_cap=200, n_angular=300, n_height=200)
    # The two reflection scales, shared across wavelengths: wall_R is the blacksheet's flat
    # reflectance (0.045 x bsrff 2.5, as the HK_WCSIM config sets it); sensor_R scales the WCSim PMT
    # table, so its truth is 1. The tabulated model fixes both reflection directions, so the
    # specular fractions do nothing and are held (the fit's logit(f) needs them inside (0, 1)).
    WALL_R, WALL_FSPEC = REFLECTION['wall_reflection_rate'], REFLECTION['wall_fspec']
    SENSOR_R, SENSOR_FSPEC = 1.0, REFLECTION['sensor_fspec']
    QE_SPREAD = 0.05                                     # truth per-PMT gain spread (paper value)
    # paper recipe (calib_run.run_seed defaults)
    STEPS, REFRESH, JACOBIAN_DRAWS, TRUTH_DRAWS, POLYAK = 200, 20, 8, 100, 50

    # Forward draws per step. The profiled gains are nonlinear in the forward, so its MC noise
    # biases the fixed point ~1/N_FORWARD; with N_FORWARD >= 2 the fit jackknifes the profiled
    # model over the draws (lucid.fitting.calib.jackknife_profile), cancelling that 1/N term, and
    # 2 already beats plain profiling of 8. Paper: 1 (no jackknife).
    N_FORWARD = 4
    # Jacobian refresh schedule: every REFRESH_EARLY steps for the first REFRESH_SWITCH fraction of
    # the run (while parameters move fast), then every REFRESH. REFRESH_EARLY = REFRESH: constant.
    REFRESH_EARLY, REFRESH_SWITCH = 5, 0.2
    # Step = LR x (A^-1 g), A = H + LAM·diag(H) + MU·median(diag H)·I, in the preconditioned units
    # set after the Fisher (scale S), so MU damps every parameter by the same relative amount.
    # MU is load-bearing on the soft directions (MU ~ 0 lets noise walk them). MAX_STEP clips the
    # scaled step (a parameter's physical cap is MAX_STEP * S_p).
    MAX_STEP, LAM, MU, LR = 0.5, 0.01, 0.1, 1.0      # paper recipe
    # Weights from the model, not the data (iteratively reweighted least squares), as
    # hk_multiwavelength_job.py. Digitized charge is not Poisson in photoelectrons, so the weight is
    # the model's VARIANCE of the summed charge, N_PULSES x Var_pulse(x) at the model's expected
    # photoelectrons per pulse x (the digitizer's threshold, SPE spectrum and dark noise included,
    # lucid.simulation.digitizer.pulse_charge_response): the quasi-likelihood weight. It is
    # computed at the current theta from IRLS_DRAWS forward draws on keys independent of the fit's,
    # held fixed for a stage of STEPS / IRLS_STAGES steps and recomputed between stages.
    IRLS_STAGES, IRLS_DRAWS = 4, 16
    IRLS_KEY0 = 3_000_000                                # away from the truth's and fit's keys
    PERT, PERT_SCATTER, PERT_REFL = 0.4, 2.0, 1.2       # start offsets, log / logit space
    EVAL_DRAWS = 16                                      # chi2(fit) vs chi2(truth) check, 0: skip
    FISHER_DRAWS = 8                                     # Fisher/correlations at truth, 0: skip
    FIT = True                                           # False: Fisher only (fast layout compare)

    print(jax.devices())
    os.environ['LUCID_PMT_REFLECTANCE_TABLE'] = os.path.abspath(PMT_OPTICS)
    det = generate_detector(GEOM); NS = len(det.all_points)
    # The forward's normalisation is the sources' nominal photon count; it cancels in pulse_scale.
    sources = injector_sources('diffuser', 'WarwickDA02', idx=INJ_IDX)
    source_labels = [f'dif{e["idx"]}' for e in load_injectors('diffuser') if e['idx'] in set(INJ_IDX)]

    sim = setup_event_simulator(GEOM, N_RAYS, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,
                                reflection_model='tabulated_lambertian',
                                pmt_detection_model=PMT_DETECTION,
                                reflection_wavelength=REFLECTION_WAVELENGTH,
                                deposit_leg_bound=True, **GRID)

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
    truth_k = tk / np.exp(np.mean(np.log(tk)))   # geometric mean 1, the fit's gauge='log'

    print(f'{NS} PMTs | {len(sources)} sources x {W} wavelengths | {P} parameters', flush=True)
    fwd = CalibrationForward(sim, sources, params, NS)
    # The data are what the detector records: N_PULSES pulses per source and wavelength, each
    # digitized with dark noise over the full event readout, summed. The fit models the same
    # readout (PulseResponse), with the HK_WCSIM config's digitizer.
    digitizer = resolve_model_config(DIGITIZER)
    dark_mu = digitizer['dark_rate_khz'] * READOUT_NS * 1e-6      # dark pulses per PMT per pulse
    # pulse_scale: expected photoelectrons per pulse per unit of forward charge, i.e. photons per
    # pulse over the photons the forward's sources emit.
    forward_photons = float(sources[0].intensity)
    resp = PulseResponse(digitizer, N_PULSES, [PULSE_PHOTONS[w] / forward_photons for w in WAVELENGTHS],
                         dark_mu=dark_mu)
    t = time.perf_counter()
    data, data_var = pulsed_closure_data(fwd, theta_true, resp, jnp.asarray(truth_k),
                                         n_draws=TRUTH_DRAWS)
    print(f'[time] truth ({TRUTH_DRAWS} draws, {N_PULSES} pulses digitized, '
          f'dark {dark_mu:.4f}/PMT/pulse): {time.perf_counter() - t:.0f} s', flush=True)

    rng = np.random.default_rng(4000 + args.seed)
    start = theta_true + rng.uniform(-PERT, PERT, P)
    for w in range(W):
        start[3 * w] = theta_true[3 * w] + rng.uniform(-PERT_SCATTER, PERT_SCATTER)
    for j in range(NG, P):
        start[j] = theta_true[j] + rng.uniform(-PERT_REFL, PERT_REFL)
    if args.resume:
        f = np.load(args.resume)['fit']                  # physical values, ordered as `names`
        start = np.asarray(params.theta_from_physical(
            [dict(scatter_length=f[3 * w], absorption_length=f[3 * w + 1], qe=f[3 * w + 2])
             for w in range(W)], *f[NG:NG + 4]), dtype=float)
        print(f'resuming from {args.resume}', flush=True)
    fix = (NG + 1, NG + 2, NG + 3)                       # wall_fspec, sensor_R, sensor_fspec
    for j in fix:
        start[j] = theta_true[j]
    free = np.array([j not in fix for j in range(P)])
    q_floor = 0.01 * float(jnp.mean(data))               # calibrate's default (q_floor_frac 0.01)

    from lucid.fitting.calib import jackknife_profile, neyman_residual
    from lucid.simulation.digitizer import pulse_charge_response
    ones, data_sum = jnp.ones(NS), data.sum(0)
    wl_rows = (jnp.arange(data.shape[0]) // len(sources))[:, None]
    # Var_pulse(x), tabulated once as PulseResponse tabulates the mean: the dark-only variance at
    # x = 0 plus x times a ratio on a log grid, held above the grid.
    x_grid = np.geomspace(1e-6, 300.0, 600)
    _, v_tab, _ = pulse_charge_response(resp.model, np.concatenate([[0.0], x_grid]), dark_mu=dark_mu)
    var_dark = float(v_tab[0])
    var_ratio = jnp.asarray((v_tab[1:] - v_tab[0]) / x_grid, dtype=jnp.float32)
    log_x_grid = jnp.asarray(np.log(x_grid), dtype=jnp.float32)

    def model_variance(theta, key_base):
        """The IRLS weight and the expected charge at theta: N_PULSES x Var_pulse(x) and
        N_PULSES x (g(x) + pedestal), x the expected photoelectrons per pulse, gains profiled
        (through the response) on the mean of IRLS_DRAWS forward draws on independent keys."""
        mu = fwd.average(jnp.asarray(theta, dtype=jnp.float32), key_base, ones, IRLS_DRAWS)
        k = resp.profile_gains(mu, data_sum, wl_rows, gauge='log')
        x = resp.pulse_scale[wl_rows] * k[None, :] * mu
        v = var_dark + x * jnp.interp(jnp.log(jnp.maximum(x, x_grid[0])), log_x_grid, var_ratio)
        return N_PULSES * v, resp.charge(k[None, :] * mu, wl_rows)

    out = dict(names=np.array(names), truth=truth, free=free, wavelengths=np.array(WAVELENGTHS),
               source_labels=np.array(source_labels), truth_k=truth_k,
               photons=np.array([N_PULSES * PULSE_PHOTONS[w] for w in WAVELENGTHS], dtype=float),
               n_rays=N_RAYS, seed=args.seed,
               n_forward=N_FORWARD, n_pulses=N_PULSES, dark_mu=dark_mu,
               pulse_photons=np.array([PULSE_PHOTONS[w] for w in WAVELENGTHS]),
               truth_charge=np.asarray(data).reshape(W, len(sources), NS),   # (wl, source, PMT)
               truth_charge_var=np.asarray(data_var).reshape(W, len(sources), NS))

    # Fisher at truth with the per-PMT gains marginalised (Schur complement, gauge mean log k = 0).
    # r = (model - Q)/sqrt(var), var = the model's variance at truth (the IRLS weight), is
    # unit-variance, so F = J^T J with J the fit's own Jacobian (weighted by the same var);
    # d r / d log k_p ~ (expected charge - pedestal) / sqrt(var) at truth (exact where the response
    # is linear, as it nearly is). Free params are all log-basis, so sigma is fractional. Raw engine
    # bound: x sqrt(12) for the honest CRB.
    from lucid.fitting.calib import CalibrationJacobian, neyman_variance
    scale = None                                         # set from the Fisher block's Jacobian
    if FISHER_DRAWS:
        from lucid.fitting.schur_gn import make_constrained_schur
        t = time.perf_counter()
        var_model, q_model = model_variance(theta_true, IRLS_KEY0 - 1_000_000)
        var = neyman_variance(data, q_floor, var_model)
        jac = CalibrationJacobian(sim, sources, params, NS, key0=8_000_000, response=resp)
        J = np.asarray(jac(jnp.asarray(theta_true, dtype=jnp.float32), jnp.log(jnp.asarray(truth_k)),
                           0, var, q_floor, draws=range(FISHER_DRAWS)), dtype=np.float64)
        Q = np.clip(np.asarray(q_model, dtype=np.float64) - N_PULSES * resp.pedestal, 0.0, None)
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

        # Preconditioning, as hk_multiwavelength_job.py: the step is taken in units where every
        # free parameter's own curvature (J^T J)_pp is the same, theta_p = S_p u_p with
        # S_p ~ 1/sqrt((J^T J)_pp), so MU damps every parameter by the same relative amount. S is
        # normalised to geometric mean 1; calibrate clips the SCALED step, so each parameter's
        # physical cap is MAX_STEP * S_p (printed).
        s = 1.0 / np.sqrt(np.diag(Htt)[free])
        s /= np.exp(np.mean(np.log(s)))
        scale = np.ones(P)
        scale[free] = s
        print(f'\n{"param":24s}{"scale S":>10s}{"step cap":>10s}')
        for n, sp in zip(fn, s):
            print(f'{n:24s}{sp:10.3g}{MAX_STEP * sp:10.3g}')
        out.update(scale=scale)

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

    # IRLS: weights (the model's variance) from the model at the stage's starting theta, fixed
    # for the stage.
    stage_steps = STEPS // IRLS_STAGES
    theta, histories, losses = np.asarray(start, dtype=float), [], []
    for stage in range(IRLS_STAGES):
        var, _ = model_variance(theta, IRLS_KEY0 + 100_003 * stage)
        last = stage == IRLS_STAGES - 1
        print(f'  IRLS stage {stage + 1}/{IRLS_STAGES}: weights from the model at the stage start',
              flush=True)
        res = calibrate(sim, sources, params, data, theta, steps=stage_steps, max_step=MAX_STEP,
                        refresh=REFRESH_EARLY, refresh_final=REFRESH, refresh_switch=REFRESH_SWITCH,
                        jacobian_draws=JACOBIAN_DRAWS, n_forward_draws=N_FORWARD,
                        gauge='log', seed=args.seed + 1000 * stage,
                        readout='polyak' if last else 'final', polyak=POLYAK if last else 0,
                        lam=LAM, mu=MU, lr=LR, fix=fix, forward=fwd,
                        scale=scale, on_step=progress, response=resp, data_var=var)
        theta = np.asarray(res['theta'], dtype=float)
        histories += list(res['history'][1:])
        losses += list(np.asarray(res['loss']))
    res['history'] = [start] + histories
    res['loss'] = np.asarray(losses)
    print(f'[time] fit ({STEPS} steps, {IRLS_STAGES} IRLS stages): '
          f'{time.perf_counter() - t_fit:.0f} s', flush=True)

    fit = physical(res['theta'])
    k_hat = np.asarray(res['gains'])
    print(f'\n{"param":24s}{"truth":>10s}{"start":>10s}{"fit":>10s}{"err":>9s}')
    for n, tr, st, ft, fr in zip(names, truth, physical(start), fit, free):
        print(f'{n:24s}{tr:10.4g}{st:10.4g}{ft:10.4g}{ft / tr - 1:+9.1%}' + ('' if fr else '  (fixed)'))
    print(f'per-PMT gain residual RMS: {np.std(k_hat / truth_k - 1):.2%}')

    # Bias or convergence? The fit's own objective (same q_floor, gauge, and the last IRLS stage's
    # model variance, the same at both points) at truth and at the fit. sum r^2 from one forward
    # draw also counts the forward's MC variance, which moves with theta (the gains rescale each
    # ray's weight) and swamps the fit-quality difference. So use the cross product of two
    # INDEPENDENT draws, sum r_A * r_B: its expectation is sum E[r]^2 with no variance term. The
    # same key pair at both points cancels most of the noise in the difference. Delta < 0 (fit
    # lower): the objective's minimum is displaced from truth. Delta > 0: truth is better and the
    # optimiser hasn't got there.
    def resid(theta, key_base):
        # The fit's own model: N_FORWARD draws, jackknifed when there are two or more.
        th = jnp.asarray(theta, dtype=jnp.float32)
        if N_FORWARD < 2:
            mu = fwd.average(th, key_base, ones, 1)
            k = resp.profile_gains(mu, data_sum, wl_rows, gauge='log')
            model = resp.charge(k[None, :] * mu, wl_rows)
        else:
            _, model = jackknife_profile(fwd.draws(th, key_base, ones, N_FORWARD), data_sum,
                                         gauge='log', response=resp, wl=wl_rows)
        return neyman_residual(model, data, q_floor, var)

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
