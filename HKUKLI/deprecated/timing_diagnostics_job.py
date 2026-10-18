"""Per-deposit timing diagnostics: where do LUCiD's ray-step arrival times break?

Physics is switched on one process at a time (geometry only, wall reflection, PMT reflection,
Rayleigh, then the full WCSim 395 nm set) for a few sources, with the deposit bounded to the
travelled leg and not (`deposit_leg_bound`). Every deposit is saved with its PMT, time, weight,
propagation step and ray, so timing_diagnostics.ipynb can test each against an exact answer:
direct light (step 0, nothing on) must arrive at |PMT - source| / v, and NOTHING may arrive
earlier than that straight-line bound.

Caveat: the per_photon hit mode drops deposits with t <= 0 (make_hits_likelihood's valid mask),
so a negative-time pathology would be invisible here, not merely rare.
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='timing_diagnostics.npz')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t_start = time.perf_counter()

    import sys; sys.path.append('..')
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.sources import isotropic_source
    from lucid.wavelength.medium import make_medium
    from hk_injector_sources import injector_sources, injector_position_direction
    from hk_wcsim_params import REFLECTION_MODEL, wcsim_395nm_params

    # ---- configuration (edit here) ----
    GEOM = '../config/HK_geom_config.json'
    N_RAYS, K = 100_000, 12
    INTENSITY = 1e6
    GRID = dict(n_cap=200, n_angular=300, n_height=200)
    OFF = 1e9                                        # length (m) that switches a process off
    NONE = dict(scatter_length=OFF, absorption_length=OFF, mie_scatter_length=OFF,
                wall_reflection_rate=0.0, sensor_reflection_rate=0.0)
    # (label, overrides on the WCSim 395 nm params). qe = 1 so weights are expected photons.
    CONFIGS = [
        ('geometry', NONE),
        ('wall_refl', {**NONE, 'wall_reflection_rate': 0.5}),
        ('pmt_refl', {**NONE, 'sensor_reflection_rate': 0.5}),
        ('rayleigh', {**NONE, 'scatter_length': 50.0}),
        ('wcsim395', {}),
    ]
    LEG_BOUND = [False, True]
    DIF_IDX = '14'                                   # barrel diffuser, z = 0
    dif_pos, dif_dir = injector_position_direction('diffuser', DIF_IDX)
    SOURCES = [
        ('iso_centre', [0.0, 0.0, 0.0], isotropic_source([0.0, 0.0, 0.0], intensity=INTENSITY)),
        ('iso_offset', [20.0, 0.0, 15.0], isotropic_source([20.0, 0.0, 15.0], intensity=INTENSITY)),
        (f'dif{DIF_IDX}', dif_pos,
         injector_sources('diffuser', 'WarwickDA02', idx=[DIF_IDX], intensity=INTENSITY)[0]),
    ]

    print(jax.devices())
    det = generate_detector(GEOM); NS = len(det.all_points)
    v = float(make_medium('water').speed_of_light)
    out = dict(pmt_pos=np.asarray(det.all_points), sensor_radius=float(det.sensor_radius),
               speed_of_light=v, K=K, n_rays=N_RAYS,
               configs=np.array([c for c, _ in CONFIGS]), sources=np.array([s for s, _, _ in SOURCES]),
               source_pos=np.array([p for _, p, _ in SOURCES], float), leg_bound=np.array(LEG_BOUND),
               dif_dir=np.asarray(dif_dir, float))
    print(f'{NS} PMTs | v = {v:.4f} m/ns | {N_RAYS:,} rays, K = {K}', flush=True)

    for lb in LEG_BOUND:
        sim = setup_event_simulator(GEOM, N_RAYS, temperature=None, K=K, is_calibration=True,
                                    hit_mode='per_photon', wavelength_mode=False,
                                    reflection_model=REFLECTION_MODEL, deposit_leg_bound=lb,
                                    **GRID)
        for cfg, over in CONFIGS:
            dp = wcsim_395nm_params(NS, **{**over, 'qe': 1.0})
            for sname, _, src in SOURCES:
                t = time.perf_counter()
                log_w, times, idx, charge = jax.block_until_ready(sim(src, dp, jax.random.PRNGKey(0)))
                log_w, times, idx = np.asarray(log_w), np.asarray(times), np.asarray(idx)
                valid = log_w > -1e9
                pos = np.nonzero(valid)[0]
                per_step = log_w.size // K                   # flat layout is (step, candidate, ray)
                key = f'{cfg}__{sname}__lb{int(lb)}'
                out[key + '__w'] = np.exp(log_w[valid]).astype(np.float32)
                out[key + '__t'] = times[valid].astype(np.float32)
                out[key + '__pmt'] = idx[valid].astype(np.int32)
                out[key + '__step'] = (pos // per_step).astype(np.int8)
                out[key + '__ray'] = (pos % N_RAYS).astype(np.int32)
                out[key + '__charge'] = np.asarray(charge, np.float32)
                print(f'  {key:34s} {valid.sum():9,d} deposits  charge {float(charge.sum()):.4g}  '
                      f'{time.perf_counter() - t:.1f} s', flush=True)

    np.savez(args.out, **out)
    print(f'saved -> {args.out}  [total {time.perf_counter() - t_start:.0f} s]')


if __name__ == '__main__':
    main()
