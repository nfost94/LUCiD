"""Run the HK calibration pipeline on GPU as a standalone process.

Invoked from the notebook via subprocess/`!` so the CUDA context (and JAX's
preallocated GPU memory pool) is released when this process exits, instead of
being held by the long-lived notebook kernel.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='calibration_results.npz')
    ap.add_argument('--geom', default='../config/HK_geom_config.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import build_calibration_problem, fit, crb
    from hk_injector_sources import injector_sources
    from hk_wcsim_params import wcsim_395nm_params, REFLECTION_MODEL

    print(jax.devices())          # should list CudaDevice, not CpuDevice
    print(jax.default_backend())  # should print 'gpu'

    det = generate_detector(args.geom); NS = len(det.all_points)
    print(f'{NS} PMTs | r={det.r:.1f} m | H={det.H:.1f} m')

    # "just some" -- these 8 idx spread across all 7 barrel heights plus one topcap and
    # one endcap entry; positions/directions are the real HK injector survey (cm -> m).
    # All reuse the same loaded WarwickDA02 profile. Change idx to pick different/more/
    # fewer positions, or pass idx=None for every ID diffuser.
    sources = injector_sources('diffuser', 'WarwickDA02',
                               idx=['0', '5', '10', '15', '20', '25', '30', '35'])
    print(f'{len(sources)} calibration sources (measured diffuser profile, WarwickDA02)')

    dp = wcsim_395nm_params(NS)

    sim = setup_event_simulator(args.geom, 1_000_000, temperature=None, K=8, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,
                                reflection_model=REFLECTION_MODEL,
                                n_cap=200, n_angular=300, n_height=200)  # finer grid: HK is ~2x SK_like's linear size

    FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
              'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
    prob = build_calibration_problem(sim, sources, dp, FIELDS, key=jax.random.PRNGKey(1))
    print(f'{NS} PMTs | {len(sources)} calibration sources | {len(FIELDS)} global parameters')

    sigma = crb(prob['source_models'], prob['theta_true'], NS)['sigma']
    print('CRB (fractional 1sigma) per parameter:')
    for f, s in zip(FIELDS, sigma):
        print(f'  {f:22s} {s:6.2%}')

    start = prob['theta0'] + np.random.default_rng(args.seed).uniform(-.15, .15, prob['theta0'].shape)
    res = fit(prob['source_models'], prob['truth_charge'], start, NS, steps=150, refresh=15, jacobian_draws=2)

    truth = np.exp(prob['theta0'])
    print(f'{"param":22s}{"truth":>9s}{"start":>9s}{"fit":>9s}{"err":>8s}{"CRB":>7s}')
    for i, f in enumerate(FIELDS):
        e = res['theta'][i] / truth[i] - 1
        print(f'{f:22s}{truth[i]:9.3f}{np.exp(start[i]):9.3f}{res["theta"][i]:9.3f}{e:+7.1%}{sigma[i]:7.1%}')

    np.savez(args.out,
             fields=np.array(FIELDS),
             ns=NS, r=det.r, h=det.H,
             sigma=np.asarray(sigma),
             truth=truth,
             start=np.exp(start),
             fit_theta=np.asarray(res['theta']),
             history=np.asarray(res['history']))
    print(f'saved results -> {args.out}')


if __name__ == '__main__':
    main()
