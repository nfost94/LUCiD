"""Run the wavelength-dependent calibration sweep on GPU as a standalone process,
on HK geometry.

Replicates good_notebooks/calibration/wavelength_calibration.ipynb: optimizes all 4
detector parameters (scatter_length, wall_reflection_rate, sensor_reflection_rate,
absorption_length) at several laser wavelengths, with "true" scatter/absorption taken
from the water medium physics at each wavelength (wavelength_mode=False, so they're
free optimization parameters -- see wavelength_calibration_findings.md for why this is
a harder, more degenerate problem than wavelength_mode=True).

Runs fully inside its own process so the CUDA context / JAX preallocated GPU memory
pool is released on exit, instead of being pinned to the notebook's kernel for the
rest of the session.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='wavelength_calibration_results.npz')
    ap.add_argument('--geom', default='../config/HK_geom_config.json')
    ap.add_argument('--qe', default='../config/pmt/HK_QE.json')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import sys; sys.path.append('..')
    import time
    import jax, jax.numpy as jnp, numpy as np
    import optax
    from tqdm import tqdm
    from lucid.geometry import generate_detector
    from lucid.losses import WC_smooth_loss
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import (
        DetectorParams, laser_source,
        normalize_params, denormalize_params, default_bounds,
        make_optimization_mask,
    )
    from lucid.wavelength.medium import make_medium, load_qe_curve

    print(jax.devices())          # should list CudaDevice, not CpuDevice
    print(jax.default_backend())  # should print 'gpu'

    GRID_KW = dict(n_cap=300, n_angular=500, n_height=300)  # ~2x SK_like tutorial's grid: HK is ~2x the linear size

    detector = generate_detector(args.geom)
    detector_points = jnp.array(detector.all_points)
    NUM_SENSORS = len(detector_points)

    wl_grid = jnp.linspace(280, 650, 371)
    medium = make_medium('water', wavelength_grid=wl_grid)
    qe_fn = load_qe_curve(args.qe)

    WAVELENGTHS = [350, 375, 405, 450, 500]

    NPHOT = 500_000
    K = 10
    NPHOT_TRUE = 5_000_000
    K_TRUE = 12
    ADAM_LR = 0.05
    ADAM_ITERS = 500
    WARMUP_FRAC = 0.4
    INIT_FRAC_DIFF = 0.5
    SOURCE_INTENSITY = 100_000_000

    print(f'Detector: {NUM_SENSORS} sensors')
    print(f'Optimizer: Nphot={NPHOT}, K={K}, {ADAM_ITERS} iters')

    print(f'{"wl (nm)":>8s}  {"L_scat (m)":>10s}  {"L_abs (m)":>10s}  {"QE":>6s}')
    print('-' * 40)
    true_params_per_wl = {}
    for wl in WAVELENGTHS:
        sc = float(jnp.interp(float(wl), wl_grid, medium.scatter_coeff))
        ac = float(jnp.interp(float(wl), wl_grid, medium.absorption_coeff))
        L_s, L_a = 1.0 / sc, 1.0 / ac
        qe_val = float(qe_fn(float(wl)))
        print(f'{wl:8d}  {L_s:10.1f}  {L_a:10.1f}  {qe_val:6.3f}')
        true_params_per_wl[wl] = {
            'scatter_length': L_s,
            'absorption_length': L_a,
            'wall_reflection_rate': 0.2,
            'sensor_reflection_rate': 0.2,
            'qe': qe_val,
        }

    def run_calibration_at_wavelength(wl, true_dict, seed):
        """Run 4-param calibration at a specific wavelength."""
        # DetectorParams is nested (scattering/absorption/reflection/response/per_pmt/
        # scintillation sub-tuples) -- from_flat builds it from flat leaf kwargs, and
        # leaves are read back via their sub-tuple (true_dp.scattering.scatter_length, etc).
        true_dp = DetectorParams.from_flat(
            scatter_length=true_dict['scatter_length'],
            wall_reflection_rate=true_dict['wall_reflection_rate'],
            sensor_reflection_rate=true_dict['sensor_reflection_rate'],
            absorption_length=true_dict['absorption_length'],
            qe=true_dict['qe'],
            qe_corrections=jnp.ones(NUM_SENSORS),
            num_sensors=NUM_SENSORS)

        source = laser_source(
            position=[0.0, 0.0, detector.H / 2 - 0.1],
            intensity=SOURCE_INTENSITY)

        # wavelength_mode=False so we can optimize scatter/absorption as scalars
        sim_true = setup_event_simulator(
            args.geom, NPHOT_TRUE, temperature=None, K=K_TRUE,
            is_calibration=True, hit_mode='aggregated', default_detector_params=true_dp,
            wavelength_mode=False, **GRID_KW)
        sim_opt = setup_event_simulator(
            args.geom, NPHOT, temperature=None, K=K,
            is_calibration=True, hit_mode='aggregated', wavelength_mode=False, **GRID_KW)

        key = jax.random.PRNGKey(seed)
        ks, kd, ki = jax.random.split(key, 3)

        true_data = jax.lax.stop_gradient(sim_true(source, kd))
        print(f'  True data charge_sum={float(jnp.sum(true_data[0])):.0f}')

        true_scatter = float(true_dp.scattering.scatter_length)
        true_wall = float(true_dp.reflection.wall_reflection_rate)
        true_sensor = float(true_dp.reflection.sensor_reflection_rate)
        true_absorption = float(true_dp.absorption.absorption_length)

        mults = jax.random.uniform(ki, (4,), minval=1 - INIT_FRAC_DIFF, maxval=1 + INIT_FRAC_DIFF)
        init = DetectorParams.from_flat(
            scatter_length=jnp.clip(true_scatter * mults[0], 5.0, max(true_scatter * 2, 100.0)),
            wall_reflection_rate=jnp.clip(true_wall * mults[1], 0.05, 0.5),
            sensor_reflection_rate=jnp.clip(true_sensor * mults[2], 0.05, 0.4),
            absorption_length=jnp.clip(true_absorption * mults[3], 10.0, max(true_absorption * 2, 500.0)),
            qe=true_dp.response.qe,
            qe_corrections=true_dp.per_pmt.qe_corrections,
            num_sensors=NUM_SENSORS)

        bmin, bmax = default_bounds(NUM_SENSORS)
        bmin = bmin._replace(
            scattering=bmin.scattering._replace(scatter_length=jnp.array(5.)),
            reflection=bmin.reflection._replace(wall_reflection_rate=jnp.array(0.05),
                                                sensor_reflection_rate=jnp.array(0.05)),
            absorption=bmin.absorption._replace(absorption_length=jnp.array(10.)))
        bmax = bmax._replace(
            scattering=bmax.scattering._replace(scatter_length=jnp.array(max(true_scatter * 3, 200.))),
            absorption=bmax.absorption._replace(absorption_length=jnp.array(max(true_absorption * 3, 1000.))))

        TRAINABLE = {'scatter_length', 'wall_reflection_rate',
                     'sensor_reflection_rate', 'absorption_length'}

        @jax.jit
        def step_fn(p):
            return jax.value_and_grad(lambda p: WC_smooth_loss(
                detector_points, *true_data,
                *sim_opt(source, denormalize_params(p, bmin, bmax), ks),
                lambda_poisson=1.0, lambda_time=0.0, tau=2.0))(p)

        params = normalize_params(init, bmin, bmax)
        mask = make_optimization_mask(params, TRAINABLE)
        labels = jax.tree.map(lambda m: 'train' if m else 'freeze', mask)
        warmup_steps = int(WARMUP_FRAC * ADAM_ITERS)
        sched = optax.warmup_constant_schedule(init_value=0., peak_value=ADAM_LR,
                                               warmup_steps=warmup_steps)
        opt = optax.multi_transform({
            'train': optax.adam(learning_rate=sched, b1=0.95, b2=0.99),
            'freeze': optax.set_to_zero(),
        }, labels)
        opt_state = opt.init(params)

        _ = step_fn(params); jax.block_until_ready(_)  # warmup JIT

        losses = np.zeros(ADAM_ITERS)
        hist = {k: np.zeros(ADAM_ITERS) for k in
                ('scatter_length', 'wall_reflection_rate', 'sensor_reflection_rate', 'absorption_length')}
        t0 = time.time()

        for i in tqdm(range(ADAM_ITERS), desc=f'{wl}nm', leave=False):
            loss, grads = step_fn(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = jax.tree.map(lambda p: jnp.clip(p, 0.01, 0.99), params)

            dp = denormalize_params(params, bmin, bmax)
            losses[i] = float(loss)
            hist['scatter_length'][i] = float(dp.scattering.scatter_length)
            hist['wall_reflection_rate'][i] = float(dp.reflection.wall_reflection_rate)
            hist['sensor_reflection_rate'][i] = float(dp.reflection.sensor_reflection_rate)
            hist['absorption_length'][i] = float(dp.absorption.absorption_length)

        elapsed = time.time() - t0
        final = denormalize_params(params, bmin, bmax)

        return {
            'init': {
                'scatter_length': float(init.scattering.scatter_length),
                'wall_reflection_rate': float(init.reflection.wall_reflection_rate),
                'sensor_reflection_rate': float(init.reflection.sensor_reflection_rate),
                'absorption_length': float(init.absorption.absorption_length),
            },
            'final': {
                'scatter_length': float(final.scattering.scatter_length),
                'wall_reflection_rate': float(final.reflection.wall_reflection_rate),
                'sensor_reflection_rate': float(final.reflection.sensor_reflection_rate),
                'absorption_length': float(final.absorption.absorption_length),
            },
            'losses': losses,
            'hist': hist,
            'runtime': elapsed,
        }

    save = {'wavelengths': np.array(WAVELENGTHS)}
    for wl in WAVELENGTHS:
        print(f'\n--- {wl} nm ---')
        t = true_params_per_wl[wl]
        print(f'  True: scatter={t["scatter_length"]:.1f}m, absorption={t["absorption_length"]:.1f}m')
        r = run_calibration_at_wavelength(wl, t, seed=args.seed)
        f = r['final']
        print(f'  Final: scatter={f["scatter_length"]:.1f} (true={t["scatter_length"]:.1f})  '
              f'absorption={f["absorption_length"]:.1f} (true={t["absorption_length"]:.1f})')
        print(f'  Final: wall_refl={f["wall_reflection_rate"]:.3f} (true=0.200)  '
              f'sensor_refl={f["sensor_reflection_rate"]:.3f} (true=0.200)')
        print(f'  Time: {r["runtime"]:.1f}s')

        p = f'wl{wl}_'
        for field in ('scatter_length', 'wall_reflection_rate', 'sensor_reflection_rate', 'absorption_length'):
            save[p + 'true_' + field] = t[field]
            save[p + 'init_' + field] = r['init'][field]
            save[p + 'final_' + field] = r['final'][field]
            save[p + 'hist_' + field] = r['hist'][field]
        save[p + 'true_qe'] = t['qe']
        save[p + 'losses'] = r['losses']
        save[p + 'runtime'] = r['runtime']

    np.savez(args.out, **save)
    print(f'saved results -> {args.out}')


if __name__ == '__main__':
    main()
