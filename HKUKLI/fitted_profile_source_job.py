"""Run the fit-based measured-profile source (lucid.sources.fitted_profile_source)
end-to-end -- fit composite(w,p,a0,s) by least squares (scipy), build the source, sample, and
cross-check against the already-validated grid-based measured_profile_source and the raw
measured data.

Mirrors measured_profile_source_job.py's pattern: this script does all the compute
(including the one-time profile fit) and saves results to an .npz; the notebook only loads
that file and plots.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpus', type=int,
                     default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
                     help='CPU core/thread count to restrict numpy BLAS and JAX:CPU to')
    ap.add_argument('--out', default='fitted_profile_source_results.npz')
    ap.add_argument('--n-validate', type=int, default=2_000_000)
    ap.add_argument('--n-rotation', type=int, default=500_000)
    ap.add_argument('--n-scan', type=int, default=5_000_000)
    args = ap.parse_args()

    n = str(args.cpus)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = n
    os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={args.cpus}'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'

    from pathlib import Path

    import numpy as np
    import jax
    from jax import random as jrandom
    from scipy.optimize import curve_fit

    print('jax devices:', jax.devices(), '| backend:', jax.default_backend(), '| cpus:', args.cpus)

    COMMON_DIR = Path('common_profiles')

    sys.path.append('..')
    from lucid.sources.load_HKLI_profile import dedupe_wraparound_phi, composite_profile
    from lucid.sources.calibration_sources import measured_profile_source, fitted_profile_source

    # ---- fit-path-specific data reduction (injector_profile_comparison.ipynb cell 14): the
    # diffuser's genuine duplicate of (theta0+d, roll=r) is (theta0-d, roll=(r+180) mod 360),
    # not (theta0-d, roll=r) -- a real +180deg pairing, needed here because fit_common_profile
    # assumes uniform-in-phi. Different (and more careful) than
    # load_HKLI_profile.fold_theta_diffuser's same-roll fold, which the grid-CDF sampler uses
    # because it keeps the full phi axis instead of averaging it out.
    def fold_theta(theta_vals, phi_vals, grid, theta0=90.0):
        phi_vals, grid = dedupe_wraparound_phi(phi_vals, grid)
        delta = np.round(theta_vals - theta0, 6)
        phi_vals = np.round(phi_vals, 6)
        pos, neg, zero = delta > 0, delta < 0, delta == 0

        alpha_pos = delta[pos]
        alpha_neg = -delta[neg]
        assert np.array_equal(np.sort(alpha_pos), np.sort(alpha_neg)), \
            'raw theta grid must be symmetric about theta0 for this pairing to be exact'
        order_pos, order_neg = np.argsort(alpha_pos), np.argsort(alpha_neg)
        alpha_vals = alpha_pos[order_pos]

        grid_pos, grid_neg = grid[:, pos][:, order_pos], grid[:, neg][:, order_neg]
        phi_pos_target = np.round((-phi_vals) % 360.0, 6)
        phi_neg_target = np.round((180.0 - phi_vals) % 360.0, 6)
        row_pos = np.searchsorted(phi_vals, phi_pos_target)
        row_neg = np.searchsorted(phi_vals, phi_neg_target)
        assert set(row_pos) == set(row_neg) == set(range(phi_vals.size)), \
            'phi grid must be a regular 0-360 sweep for the +180 deg pairing to land exactly'

        out_pos = np.empty_like(grid_pos); out_pos[row_pos] = grid_pos
        out_neg = np.empty_like(grid_neg); out_neg[row_neg] = grid_neg
        out = 0.5 * (out_pos + out_neg)

        if zero.any():
            alpha_vals = np.concatenate([[0.0], alpha_vals])
            out = np.concatenate([grid[:, zero][:, :1], out], axis=1)
        return alpha_vals, phi_vals, out

    def load_common_profile(sample_id):
        d = np.load(COMMON_DIR / f'{sample_id}.npz', allow_pickle=True)
        theta_vals, phi_vals, grid, device_type = d['axis1_vals'], d['axis2_vals'], d['grid'], str(d['device_type'])
        if device_type == 'diffuser':
            axis1_vals, phi_vals, grid = fold_theta(theta_vals, phi_vals, grid, theta0=90.0)
        else:
            axis1_vals = theta_vals - 90.0
        return axis1_vals, phi_vals, grid, device_type

    def collimator_alpha_grid(axis1_vals, axis2_vals):
        theta_raw = 90.0 + axis1_vals[None, :]
        phi_raw = axis2_vals[:, None]
        return np.degrees(np.arccos(np.clip(
            np.sin(np.radians(theta_raw)) * np.cos(np.radians(phi_raw)), -1, 1)))

    def bin_profile(alpha, inten, n_bins, amax=None):
        alpha = np.asarray(alpha); inten = np.asarray(inten, dtype=float)
        amax = alpha.max() if amax is None else amax
        bins = np.linspace(0, amax, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        count, _ = np.histogram(alpha, bins=bins)
        s, _ = np.histogram(alpha, bins=bins, weights=inten)
        s2, _ = np.histogram(alpha, bins=bins, weights=inten ** 2)
        with np.errstate(invalid='ignore'):
            mean = np.divide(s, count, out=np.full_like(s, np.nan, dtype=float), where=count > 0)
            var = np.divide(s2, count, out=np.full_like(s, np.nan, dtype=float), where=count > 0) - mean ** 2
            sem = np.sqrt(np.maximum(var, 0) / np.maximum(count, 1))
        floor = np.nanmax(mean) * 1e-3 if np.any(count > 0) else 1e-6
        sem = np.maximum(sem, floor)
        return centers, mean, sem, count

    def least_squares_fit(x, y, model, param_names, init, limits):
        # Unweighted least squares with box limits (what the old TGraph/Minuit2 fit did: the
        # graph carried no errors).
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        lo, hi = (np.array(b, dtype=float) for b in zip(*limits))
        p0 = np.clip(np.asarray(init, dtype=float), lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        values, cov = curve_fit(lambda xx, *p: model(xx, *p), x, y, p0=p0, bounds=(lo, hi),
                                maxfev=20000)
        rss = float(np.sum((model(x, *values) - y) ** 2))
        return dict(names=param_names, values=list(values),
                    errors=list(np.sqrt(np.clip(np.diag(cov), 0, None))),
                    rss=rss, ndf=len(x) - len(values))

    def fit_common_profile(sample_id, n_bins=40):
        axis1_vals, axis2_vals, grid, device_type = load_common_profile(sample_id)
        grid = grid / grid.max()

        if device_type == 'collimator':
            alpha = collimator_alpha_grid(axis1_vals, axis2_vals).ravel()
            inten = grid.ravel()
        else:
            alpha = axis1_vals
            inten = grid.mean(axis=0)

        amax = float(alpha.max())
        centers, mean, sem, count = bin_profile(alpha, inten, n_bins, amax=amax)
        mask = ~np.isnan(mean)
        c, peak = centers[mask], np.nanmax(mean[mask])
        m = mean[mask] / peak

        half_idx = np.argmin(np.abs(m - 0.5))
        w_guess = max(float(c[half_idx]), 0.5)
        below = c[m < 0.05]
        edge_guess = float(below[0]) if below.size else amax * 0.9

        fit = least_squares_fit(
            c, m, composite_profile, ['w', 'p', 'a0', 's'],
            init=[w_guess, 1.0, edge_guess, max(edge_guess * 0.1, 0.5)],
            limits=[(0.1, amax), (0.2, 10), (max(edge_guess * 0.3, 0.5), amax), (0.05, amax)])

        return dict(sample_id=sample_id, device_type=device_type, alpha_max=amax,
                    alpha=c, intensity=m, params=fit['values'], errors=fit['errors'], rss=fit['rss'])

    def bin_edges_from_centers(vals):
        edges = np.empty(vals.size + 1)
        edges[1:-1] = 0.5 * (vals[1:] + vals[:-1])
        edges[0] = vals[0] - (edges[1] - vals[0])
        edges[-1] = vals[-1] + (vals[-1] - edges[-2])
        return edges

    def axis_angle_from_reference_np(direction):
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        u_ref = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.clip(np.dot(u_ref, direction), -1.0, 1.0))
        axis = np.cross(u_ref, direction)
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1e-8 else np.array([0.0, 1.0, 0.0])
        return axis, angle

    # Same shared direction as measured_profile_source_job.py, so the two notebooks' plots
    # sit on the same shared frame and are directly comparable.
    SHARED_DIRECTION = (0.3, 0.9, 0.4)
    shared_dir_norm = np.asarray(SHARED_DIRECTION) / np.linalg.norm(SHARED_DIRECTION)
    boresight_theta = np.degrees(np.arccos(np.clip(shared_dir_norm[2], -1, 1)))
    boresight_phi = np.degrees(np.arctan2(shared_dir_norm[1], shared_dir_norm[0])) % 360.0

    # ---- fit + build + sample + validate ----
    results = {}
    sample_ids = ['WarwickDA02', 'WarwickCol_C01_repeat']
    rotation_targets = {'WarwickDA02': (0.0, 1.0, 0.0), 'WarwickCol_C01_repeat': (0.0, 0.0, 1.0)}
    results['shared_boresight_theta'] = boresight_theta
    results['shared_boresight_phi'] = boresight_phi

    for sample_id in sample_ids:
        fit = fit_common_profile(sample_id)
        print(f"{sample_id} ({fit['device_type']}): "
              f"w={fit['params'][0]:.2f} p={fit['params'][1]:.2f} "
              f"a0={fit['params'][2]:.2f} s={fit['params'][3]:.2f}  RSS={fit['rss']:.3e}")

        # Fit-based sample: alpha from the true off-boresight angle (arccos of the x-component,
        # since direction=(1,0,0) makes the boresight the native x-axis for both embeddings --
        # same trick the rotation check below uses), phi from arctan2(z, y) (the pole-embedding
        # azimuth fitted_profile_source's is_collimator=False assumes).
        source_fit = fitted_profile_source((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), fit['params'], fit['alpha_max'])
        ray_fit, _, _ = source_fit(args.n_validate, jrandom.PRNGKey(0))
        ray_fit = np.asarray(ray_fit)
        alpha_fit = np.degrees(np.arccos(np.clip(ray_fit[:, 0], -1, 1)))
        phi_fit = np.degrees(np.arctan2(ray_fit[:, 2], ray_fit[:, 1])) % 360.0

        # Profile-based (already-validated grid sampler) sample, same true-alpha extraction.
        source_p = measured_profile_source((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sample_id, COMMON_DIR)
        ray_p, _, _ = source_p(args.n_validate, jrandom.PRNGKey(0))
        alpha_p = np.degrees(np.arccos(np.clip(np.asarray(ray_p)[:, 0], -1, 1)))

        bin_width = max(fit['alpha_max'] / 60, 0.05)
        bins = np.arange(0, fit['alpha_max'] + bin_width, bin_width)
        centers = 0.5 * (bins[1:] + bins[:-1])
        jac = np.maximum(np.sin(np.radians(centers)), 1e-6)

        def hist_density(alpha_s):
            counts, _ = np.histogram(alpha_s, bins=bins)
            dens = counts / jac
            return dens / dens.max()

        results[f'{sample_id}__device_type'] = fit['device_type']
        results[f'{sample_id}__fit_params'] = fit['params']
        results[f'{sample_id}__measured_alpha'] = fit['alpha']
        results[f'{sample_id}__measured_intensity'] = fit['intensity']
        results[f'{sample_id}__centers'] = centers
        results[f'{sample_id}__profile_density'] = hist_density(alpha_p)
        results[f'{sample_id}__fit_density'] = hist_density(alpha_fit)
        results[f'{sample_id}__phi_fit'] = phi_fit

        # Rotation-shape-preservation, same pattern as measured_profile_source_job.py: the
        # same PRNG key rotated vs un-rotated should give identical angle-from-own-axis.
        direction = np.asarray(rotation_targets[sample_id], dtype=float)
        direction /= np.linalg.norm(direction)
        source0 = fitted_profile_source((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), fit['params'], fit['alpha_max'])
        ray0, _, _ = source0(args.n_rotation, jrandom.PRNGKey(1))
        angle_from_boresight = np.degrees(np.arccos(np.clip(np.asarray(ray0)[:, 0], -1, 1)))
        source_r = fitted_profile_source((0.0, 0.0, 0.0), direction, fit['params'], fit['alpha_max'])
        ray_r, _, _ = source_r(args.n_rotation, jrandom.PRNGKey(1))
        angle_from_direction = np.degrees(np.arccos(np.clip(np.asarray(ray_r) @ direction, -1, 1)))

        results[f'{sample_id}__angle_from_boresight'] = angle_from_boresight
        results[f'{sample_id}__angle_from_direction'] = angle_from_direction

        # ---- shared-frame comparison scan, same pattern as measured_profile_source_job.py's
        # Stage 6, but "native" is the analytic composite fit evaluated at each shared-frame
        # grid point's true alpha (angle from SHARED_DIRECTION) -- no griddata reprojection
        # needed, since (unlike the measured grid) the fit is a smooth function we can just
        # evaluate anywhere directly, and it's phi-uniform by construction so there's no
        # native (theta_raw, phi_raw) grid to reproject in the first place.
        if fit['device_type'] == 'diffuser':
            scan_theta_vals = np.linspace(0.0, 180.0, 181)
            scan_phi_vals = np.linspace(0.0, 180.0, 181)
        else:
            half_width = 15.0
            scan_theta_vals = np.linspace(boresight_theta - half_width, boresight_theta + half_width, 301)
            scan_phi_vals = np.linspace(boresight_phi - half_width, boresight_phi + half_width, 301)
        THETA_GRID, PHI_GRID = np.meshgrid(scan_theta_vals, scan_phi_vals)

        th, ph = np.radians(THETA_GRID), np.radians(PHI_GRID)
        d_shared = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
        alpha_native = np.degrees(np.arccos(np.clip(d_shared @ shared_dir_norm, -1, 1)))
        native_reproj = composite_profile(alpha_native, *fit['params'])

        source_scan = fitted_profile_source((0.0, 0.0, 0.0), SHARED_DIRECTION, fit['params'], fit['alpha_max'])
        ray_scan, _, _ = source_scan(args.n_scan, jrandom.PRNGKey(2))
        ray_scan = np.asarray(ray_scan)
        theta_scan = np.degrees(np.arccos(np.clip(ray_scan[:, 2], -1, 1)))
        phi_scan = np.degrees(np.arctan2(ray_scan[:, 1], ray_scan[:, 0])) % 360.0

        theta_edges = bin_edges_from_centers(scan_theta_vals)
        phi_edges = bin_edges_from_centers(scan_phi_vals)
        counts_2d, _, _ = np.histogram2d(phi_scan, theta_scan, bins=[phi_edges, theta_edges])
        jac_theta = np.cos(np.radians(theta_edges[:-1])) - np.cos(np.radians(theta_edges[1:]))
        scan_intensity = counts_2d / np.maximum(jac_theta, 1e-6)[None, :]

        results[f'{sample_id}__shared_theta_vals'] = scan_theta_vals
        results[f'{sample_id}__shared_phi_vals'] = scan_phi_vals
        results[f'{sample_id}__shared_scan_intensity'] = scan_intensity
        results[f'{sample_id}__shared_native_intensity'] = native_reproj

    np.savez(args.out, sample_ids=np.array(sample_ids), **results)
    print('saved results ->', args.out)


if __name__ == '__main__':
    main()
