"""Simulate diffuser/collimator/laser events (real HK injector positions) on GPU as a
standalone process, and save both the raw (charge, time) arrays AND rendered PNG displays.

Mirrors the other HKUKLI *_job.py scripts for the GPU work itself (CUDA context / JAX
preallocated GPU memory pool released on exit instead of held by the notebook kernel) --
but the PLOTTING also happens here, not in the notebook. lucid.visualization imports
lucid.utils (where the jax-based helpers live), so even just importing the plotting
function in the notebook would cascade-import jax into that long-lived kernel and grab a
GPU device on first import -- sticky for the kernel's lifetime regardless of env vars set
afterward. The notebook only loads the saved PNGs; no lucid import there at all.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='1', help='CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--out', default='event_display_results.npz')
    ap.add_argument('--geom', default='../config/HK_geom_config.json')
    ap.add_argument('--n-photons', type=int, default=1_000_000)
    ap.add_argument('--diffuser-idx', default='28', help="injector 'idx' for the diffuser source")
    ap.add_argument('--collimator-idx', default='28', help="injector 'idx' for the collimator source")
    ap.add_argument('--intensity', type=float, default=1e6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--across-cone', type=float, default=5.0,
                    help='far-side PMTs: those within this angle (deg) of the injector pointing axis, '
                         'as seen from the injector')
    ap.add_argument('--across-factor', type=float, default=1.0,
                    help='colour-scale max = this x the brightest far-side PMT')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import sys; sys.path.append('..')
    import matplotlib
    matplotlib.use('Agg')  # headless: this runs in a subprocess with no display
    import jax, jax.numpy as jnp, numpy as np
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams
    from lucid.sources import laser_source, fitted_profile_source
    from lucid.utils import jax_rotate_vector
    from lucid.visualization import create_detector_display
    from hk_injector_sources import injector_sources, injector_position_direction

    print(jax.devices())          # should list CudaDevice, not CpuDevice
    print(jax.default_backend())  # should print 'gpu'

    det = generate_detector(args.geom); NS = len(det.all_points)
    print(f'{NS} PMTs | r={det.r:.1f} m | H={det.H:.1f} m')

    dp = DetectorParams.from_flat(scatter_length=131.8, mie_scatter_length=3000., g=0.9,
         wall_reflection_rate=.1125, sensor_reflection_rate=.32, absorption_length=553.6, qe=0.32,wall_fspec=0.0,
         qe_corrections=jnp.ones(NS))

    src_diffuser = injector_sources('diffuser', 'WarwickDA02',
                                    idx=[args.diffuser_idx], intensity=args.intensity)[0]
    src_collimator = injector_sources('collimator', 'WarwickCol_C01_repeat',
                                      idx=[args.collimator_idx], intensity=args.intensity)[0]

    # Idealized laser at the SAME position/direction as the collimator -- same propagation/
    # reflection code path, so if it shows the same diffuse wash, that's a propagation
    # effect, not something specific to the measured-profile collimator's own sampling.
    collimator_pos, collimator_dir = injector_position_direction('collimator', args.collimator_idx)
    diffuser_pos, diffuser_dir = injector_position_direction('diffuser', args.diffuser_idx)
    src_laser = laser_source(position=collimator_pos, direction=collimator_dir, intensity=args.intensity)

    # Fit-based versions of the SAME two profiles, at the SAME positions/directions as the
    # grid-based ones above: fitted_profile_source builds its CDF tables from the fitted
    # composite(w,p,a0,s) analytic function (build_fit_cdf) -- a completely different
    # table-construction code path from measured_profile_source's grid/bilinear-lookup-on-
    # real-scan-data (build_profile_cdf) -- but both funnel into the same
    # sample_measured_profile_rays/rotation code afterward. Params from
    # fitted_profile_source_results.npz (fit_common_profile, run earlier).
    collimator_fit_params = (2.74664291, 5.52616698, 2.66646386, 0.05983868)
    collimator_fit_alpha_max = 4.18865004859075
    diffuser_fit_params = (38.92248043, 1.15089979, 44.37117314, 2.5984676)
    diffuser_fit_alpha_max = 88.875
    src_collimator_fit = fitted_profile_source(
        collimator_pos, collimator_dir, collimator_fit_params, collimator_fit_alpha_max,
        intensity=args.intensity)
    src_diffuser_fit = fitted_profile_source(
        diffuser_pos, diffuser_dir, diffuser_fit_params, diffuser_fit_alpha_max,
        intensity=args.intensity)

    # Synthetic (hand-picked, NOT fitted to any real data) forward-biased composite, at the
    # SAME spot as the collimator/laser above: a wide-w/high-p super-Gaussian core stays
    # near-flat out past a0, and a hard (small-s) Fermi edge cuts it off at a0=9.5deg --
    # approximating the laser's ~9.5deg uniform-density hard-cutoff cone
    # (arcsin(0.22/1.33)), but still going through MeasuredProfileSource's real code path
    # (fitted_profile_source -> sample_measured_profile_rays -> the same rotation). If this
    # behaves like the laser (clean), the anomaly is shape-dependent (peaked vs flat-top),
    # not code-path-dependent; if it still shows the anomaly, that's a much stronger case
    # for something in the shared MeasuredProfileSource code itself.
    synthetic_forward_params = (15.0, 6.0, 9.5, 0.3)
    synthetic_forward_alpha_max = 15.0
    src_synthetic_beam = fitted_profile_source(
        collimator_pos, collimator_dir, synthetic_forward_params, synthetic_forward_alpha_max,
        intensity=args.intensity)

    # ---- Diagnostic 1: rotation ALONE, nothing else. Pull rotation_axis/rotation_angle
    # straight off the built source object (MeasuredProfileSource doesn't store the
    # original `direction` it was built from -- only the derived axis/angle -- so this is
    # the only way to check the rotation step in isolation from sampling). Rotate the
    # native-frame reference (1,0,0) through it and compare to the intended direction.
    def rotation_only_check(label, src, intended_direction):
        dir_unit = jnp.asarray(intended_direction, dtype=jnp.float32)
        dir_unit = dir_unit / jnp.linalg.norm(dir_unit)
        rotated_x = jax_rotate_vector(jnp.array([1.0, 0.0, 0.0]), src.rotation_axis, src.rotation_angle)
        angle_deg = float(jnp.degrees(jnp.arccos(jnp.clip(jnp.dot(rotated_x, dir_unit), -1.0, 1.0))))
        print(f'{label} rotation-only check: rotate(1,0,0) -> {np.asarray(rotated_x)} vs '
              f'intended direction {np.asarray(dir_unit)} | angle apart: {angle_deg:.4f} deg')

    rotation_only_check('collimator', src_collimator, collimator_dir)
    rotation_only_check('diffuser', src_diffuser, diffuser_dir)
    rotation_only_check('collimator_fit', src_collimator_fit, collimator_dir)
    rotation_only_check('diffuser_fit', src_diffuser_fit, diffuser_dir)
    rotation_only_check('synthetic_beam', src_synthetic_beam, collimator_dir)

    # ---- Diagnostic 1b: raw array dump, same key, collimator vs laser -- dtype/shape/
    # NaN/Inf and a few actual values, since statistics (min/max/mean) can hide a
    # structural mismatch (dtype, NaN, degenerate weight) that summary numbers wouldn't
    # show up in.
    def raw_dump(label, src):
        rv, ro, pw = src(5, jax.random.PRNGKey(2))
        rv, ro, pw = np.asarray(rv), np.asarray(ro), np.asarray(pw)
        print(f'{label} raw dump:')
        print(f'  ray_vectors  dtype={rv.dtype} shape={rv.shape} '
              f'nan={np.isnan(rv).sum()} inf={np.isinf(rv).sum()}\n{rv}')
        print(f'  ray_origins  dtype={ro.dtype} shape={ro.shape} '
              f'nan={np.isnan(ro).sum()} inf={np.isinf(ro).sum()}\n{ro}')
        print(f'  photon_weights dtype={pw.dtype} shape={pw.shape} '
              f'nan={np.isnan(pw).sum()} inf={np.isinf(pw).sum()} values={pw}')

    raw_dump('collimator', src_collimator)
    raw_dump('laser', src_laser)
    raw_dump('collimator_fit', src_collimator_fit)

    # ---- Diagnostic 2: full sampled ray_vectors vs intended direction (sampling +
    # rotation combined). The raw collimator scan only covers ~87-93deg theta / +-3deg
    # phi, and jnp.interp clips rather than extrapolates, so this should stay within a
    # few degrees -- if it doesn't (and diagnostic 1 above is clean), the SAMPLING step
    # (not the rotation) is where the extra spread is coming from.
    def angle_report(label, src, intended_direction):
        dir_unit = jnp.asarray(intended_direction, dtype=jnp.float32)
        dir_unit = dir_unit / jnp.linalg.norm(dir_unit)
        ray_vectors, _, _ = src(200_000, jax.random.PRNGKey(1))
        cos_angle = jnp.clip(ray_vectors @ dir_unit, -1.0, 1.0)
        angle_deg = np.asarray(jnp.degrees(jnp.arccos(cos_angle)))
        n_backward = int((angle_deg > 90).sum())
        print(f'{label} source angle-to-direction: min={angle_deg.min():.2f} '
              f'max={angle_deg.max():.2f} mean={angle_deg.mean():.2f} '
              f'99.9pct={np.percentile(angle_deg, 99.9):.2f} '
              f'(deg) | {n_backward} of {angle_deg.size} rays > 90deg')

    angle_report('collimator', src_collimator, collimator_dir)
    angle_report('diffuser', src_diffuser, diffuser_dir)
    angle_report('laser', src_laser, collimator_dir)
    angle_report('collimator_fit', src_collimator_fit, collimator_dir)
    angle_report('diffuser_fit', src_diffuser_fit, diffuser_dir)
    angle_report('synthetic_beam', src_synthetic_beam, collimator_dir)

    sim = setup_event_simulator(args.geom, args.n_photons, temperature=None, K=8, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,deposit_leg_bound=True,
                                n_cap=200, n_angular=300, n_height=200)  # finer grid: HK is ~2x SK_like's linear size

    key = jax.random.PRNGKey(args.seed)
    charge_diffuser, time_diffuser = sim(src_diffuser, dp, key)
    charge_collimator, time_collimator = sim(src_collimator, dp, key)
    charge_laser, time_laser = sim(src_laser, dp, key)
    charge_collimator_fit, time_collimator_fit = sim(src_collimator_fit, dp, key)
    charge_diffuser_fit, time_diffuser_fit = sim(src_diffuser_fit, dp, key)
    charge_synthetic_beam, time_synthetic_beam = sim(src_synthetic_beam, dp, key)

    results = [('diffuser', charge_diffuser, time_diffuser),
               ('collimator', charge_collimator, time_collimator),
               ('laser', charge_laser, time_laser),
               ('collimator_fit', charge_collimator_fit, time_collimator_fit),
               ('diffuser_fit', charge_diffuser_fit, time_diffuser_fit),
               ('synthetic_beam', charge_synthetic_beam, time_synthetic_beam)]

    # Plot HERE, in this subprocess, and save PNGs -- not in the notebook. Importing
    # lucid.visualization there would cascade-import jax (it pulls lucid.utils, where the
    # jax-based helpers live) into the long-lived notebook kernel, defeating the whole
    # point of running the GPU work in this isolated subprocess: JAX grabs/inits a device
    # on first import anywhere, and that's sticky for the kernel's lifetime regardless of
    # env vars set afterward. The notebook only ever loads these PNGs -- no lucid import.
    # Save BOTH a linear and a log-scale PNG per source, so the notebook can toggle between
    # them per call instead of committing to one scale for the whole run.
    display_fn = create_detector_display(args.geom, sparse=False)
    # Colour-scale max tied to each source's direct light: the display's default 1-99th
    # percentile clip hides narrow-beam spots, and the global max is set by the few
    # near-field backscatter PMTs next to the injector. Only PMTs above vmax are clipped.
    pmt_pos = np.asarray(det.all_points)

    def across_vmax(charge, pos, direction):
        v = pmt_pos - np.asarray(pos)
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        ang = np.degrees(np.arccos(np.clip(v @ d / np.linalg.norm(v, axis=1), -1, 1)))
        return args.across_factor * float(charge[ang < args.across_cone].max())

    for label, charge, time in results:
        charge = np.asarray(charge)
        pos, direction = (diffuser_pos, diffuser_dir) if label.startswith('diffuser') \
            else (collimator_pos, collimator_dir)
        vmax = float(charge.max())  #across_vmax(charge, pos, direction)
        n_hit = int((charge > 0).sum())
        n_over = int((charge > vmax).sum())
        print(f'{label}: {n_hit} PMTs hit ({n_hit / NS:.1%}), total charge {charge.sum():.3g}, '
              f'vmax {vmax:.3g} ({n_over} PMTs above)')
        display_fn(charge, np.asarray(time), file_name=f'{label}.png', log_scale=False,
                   vmin=0.0, vmax=vmax,
                   colorbar_label=f'{label} -- Photoelectron Count')
        display_fn(charge, np.asarray(time), file_name=f'{label}_log.png', log_scale=True,
                   vmin=0.1, vmax=vmax,
                   colorbar_label=f'{label} -- Photoelectron Count (log)')

    np.savez(args.out,
             geom=args.geom,
             diffuser_idx=args.diffuser_idx, collimator_idx=args.collimator_idx,
             charge_diffuser=np.asarray(charge_diffuser), time_diffuser=np.asarray(time_diffuser),
             charge_collimator=np.asarray(charge_collimator), time_collimator=np.asarray(time_collimator),
             charge_laser=np.asarray(charge_laser), time_laser=np.asarray(time_laser),
             charge_collimator_fit=np.asarray(charge_collimator_fit),
             time_collimator_fit=np.asarray(time_collimator_fit),
             charge_diffuser_fit=np.asarray(charge_diffuser_fit),
             time_diffuser_fit=np.asarray(time_diffuser_fit),
             charge_synthetic_beam=np.asarray(charge_synthetic_beam),
             time_synthetic_beam=np.asarray(time_synthetic_beam))
    print(f'saved results -> {args.out}')


if __name__ == '__main__':
    main()
