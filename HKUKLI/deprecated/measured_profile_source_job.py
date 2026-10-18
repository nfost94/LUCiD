"""Run the measured-profile JAX source pipeline (load -> build CDF -> sample -> validate)
as a standalone process, restricted to a fixed number of CPU cores/threads and kept off the
GPU entirely -- on a shared HPC node, unrestricted BLAS/XLA thread pools try to grab every
core on the physical node, not just the ones actually allocated to this job.

Mirrors calibration_gradients.ipynb's pattern: this script does all the compute and saves
results to an .npz; the notebook only loads that file and plots.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpus', type=int,
                     default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
                     help='CPU core/thread count to restrict numpy BLAS and JAX:CPU to')
    ap.add_argument('--out', default='measured_profile_source_results.npz')
    ap.add_argument('--n-validate', type=int, default=2_000_000)
    ap.add_argument('--n-rotation', type=int, default=500_000)
    ap.add_argument('--n-scan', type=int, default=5_000_000)
    args = ap.parse_args()

    # All thread/device env vars must be set before numpy/jax are imported.
    n = str(args.cpus)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = n
    os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={args.cpus}'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # no GPU at all for this job

    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import random as jrandom
    from scipy.interpolate import griddata

    print('jax devices:', jax.devices(), '| backend:', jax.default_backend(), '| cpus:', args.cpus)

    COMMON_DIR = Path('common_profiles')

    # ---- Stages 1-5 now live in lucid.sources -- test that code directly here. ----
    sys.path.append('..')
    from lucid.sources.load_HKLI_profile import load_raw_grid as _load_raw_grid, fold_theta_diffuser
    from lucid.sources.calibration_sources import measured_profile_source as _measured_profile_source

    def load_raw_grid(sample_id):
        return _load_raw_grid(COMMON_DIR, sample_id)

    def bin_edges_from_centers(vals):
        edges = np.empty(vals.size + 1)
        edges[1:-1] = 0.5 * (vals[1:] + vals[:-1])
        edges[0] = vals[0] - (edges[1] - vals[0])
        edges[-1] = vals[-1] + (vals[-1] - edges[-2])
        return edges

    def measured_profile_source(position, direction, sample_id, intensity=1.0):
        source = _measured_profile_source(position, direction, sample_id, COMMON_DIR, intensity=intensity)
        device_type = 'collimator' if bool(source.is_collimator) else 'diffuser'
        return source, device_type

    # ---- Stage 6: shared-frame (lab-equatorial) comparison scan ----
    # Point BOTH sources' boresight at a common direction in one shared spherical frame
    # (theta = polar angle from z, phi = azimuth in the x-y plane from x) -- away from either
    # coordinate pole. Both devices' profiles (out to ~50 deg radius) then live well clear of
    # the sin(theta)->0 singularity that broke the old per-device virtual scan at its own pole
    # (theta=0 for the collimator's lab frame, alpha=0 for the diffuser's boresight-centered
    # frame).
    #
    # Deliberately NOT an axis-aligned direction like (0,1,0): that reaches the reference
    # (1,0,0) via an exact 90-degree rotation about a single coordinate axis, which lands the
    # diffuser's own psi=90/270 meridian exactly on the shared frame's x=0 plane -- precisely
    # where atan2(y,x) is most sensitive to floating-point noise in x. A generic direction
    # avoids that alignment.
    SHARED_DIRECTION = (0.3, 0.9, 0.4)

    def rodrigues_rotate_np(d, axis, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cross = np.cross(axis, d)
        dot = np.sum(axis * d, axis=-1, keepdims=True)
        return d * cos_a + cross * sin_a + axis * dot * (1 - cos_a)

    def axis_angle_from_reference_np(direction):
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        u_ref = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.clip(np.dot(u_ref, direction), -1.0, 1.0))
        axis = np.cross(u_ref, direction)
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1e-8 else np.array([0.0, 1.0, 0.0])
        return axis, angle

    def native_direction_grid(device_type, axis1_vals, axis2_vals):
        """axis1/axis2 mesh -> native-frame direction vectors, same embedding as the JAX
        source's d_collimator/d_diffuser above. Shape (n_axis2, n_axis1, 3), matching
        grid[axis2_idx, axis1_idx]."""
        A1, A2 = np.meshgrid(axis1_vals, axis2_vals)
        polar, phi = np.radians(A1), np.radians(A2)
        if device_type == 'diffuser':
            return np.stack([np.cos(polar), np.sin(polar) * np.cos(phi), np.sin(polar) * np.sin(phi)], axis=-1)
        return np.stack([np.sin(polar) * np.cos(phi), np.sin(polar) * np.sin(phi), np.cos(polar)], axis=-1)

    def reproject_native_grid(device_type, axis1_vals, axis2_vals, grid, axis, angle, theta_grid, phi_grid):
        """Rotate every native grid point's direction into the shared frame and resample
        its measured intensity onto the shared (theta,phi) mesh -- puts the native profile
        on the same axes as the JAX virtual scan for a direct, apples-to-apples comparison
        (and lets the two device types be compared to each other directly, too)."""
        d = native_direction_grid(device_type, axis1_vals, axis2_vals).reshape(-1, 3)
        d_rot = rodrigues_rotate_np(d, axis, angle)
        theta_pts = np.degrees(np.arccos(np.clip(d_rot[:, 2], -1.0, 1.0)))
        phi_pts = np.degrees(np.arctan2(d_rot[:, 1], d_rot[:, 0])) % 360.0
        return griddata((theta_pts, phi_pts), grid.reshape(-1), (theta_grid, phi_grid), method='linear')

    # ---- run the two validation checks from the notebook, save arrays for plotting ----
    results = {}
    sample_ids = ['WarwickDA02', 'WarwickCol_C01_repeat']
    rotation_targets = {'WarwickDA02': (0.0, 1.0, 0.0), 'WarwickCol_C01_repeat': (0.0, 0.0, 1.0)}

    shared_dir_norm = np.asarray(SHARED_DIRECTION) / np.linalg.norm(SHARED_DIRECTION)
    boresight_theta = np.degrees(np.arccos(np.clip(shared_dir_norm[2], -1, 1)))
    boresight_phi = np.degrees(np.arctan2(shared_dir_norm[1], shared_dir_norm[0])) % 360.0
    results['shared_boresight_theta'] = boresight_theta
    results['shared_boresight_phi'] = boresight_phi

    for sample_id in sample_ids:
        source, device_type = measured_profile_source((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sample_id)
        ray_vectors, _, _ = source(args.n_validate, jrandom.PRNGKey(0))
        ray_vectors = np.asarray(ray_vectors)

        theta_vals, phi_vals, grid, _ = load_raw_grid(sample_id)
        if device_type == 'diffuser':
            alpha_vals, grid_folded = fold_theta_diffuser(theta_vals, grid)

        # Collimator: boresight sits on theta_raw/phi_raw's equator -- recover theta_raw from
        # the z-component and compare to grid.mean(axis=0) with the cos(delta_theta) Jacobian.
        # Diffuser: boresight is the pole of the (alpha, psi) embedding -- recover alpha
        # directly from the x-component and compare to the *folded* profile with the
        # sin(alpha) Jacobian (the correct solid-angle-around-boresight weighting, now that
        # alpha/psi are a genuine polar/azimuth pair around that pole).
        if device_type == 'diffuser':
            axis1_vals = alpha_vals
            s1 = np.degrees(np.arccos(np.clip(ray_vectors[:, 0], -1, 1)))
            jac_fn = lambda deg: np.maximum(np.sin(np.radians(deg)), 1e-6)
            measured_1d = grid_folded.mean(axis=0)
        else:
            axis1_vals = theta_vals - 90.0
            s1 = np.degrees(np.arccos(np.clip(ray_vectors[:, 2], -1, 1))) - 90.0
            jac_fn = lambda deg: np.maximum(np.cos(np.radians(deg)), 1e-6)
            measured_1d = grid.mean(axis=0)

        bin_width = np.median(np.diff(axis1_vals))
        bins = np.arange(axis1_vals.min(), axis1_vals.max() + bin_width, bin_width)
        counts_s, edges = np.histogram(s1, bins=bins)
        centers = 0.5 * (edges[1:] + edges[:-1])
        jac = jac_fn(centers)
        sampled_intensity = counts_s / jac
        sampled_intensity /= sampled_intensity.max()
        meas_interp = np.interp(centers, axis1_vals, measured_1d)
        meas_interp /= meas_interp.max()

        results[f'{sample_id}__device_type'] = device_type
        results[f'{sample_id}__centers'] = centers
        results[f'{sample_id}__sampled_intensity'] = sampled_intensity
        results[f'{sample_id}__meas_interp'] = meas_interp

        direction = np.asarray(rotation_targets[sample_id], dtype=float)
        direction /= np.linalg.norm(direction)
        source0, _ = measured_profile_source((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sample_id)
        ray0, _, _ = source0(args.n_rotation, jrandom.PRNGKey(1))
        angle_from_boresight = np.degrees(np.arccos(np.clip(np.asarray(ray0)[:, 0], -1, 1)))
        source_r, _ = measured_profile_source((0.0, 0.0, 0.0), direction, sample_id)
        ray_r, _, _ = source_r(args.n_rotation, jrandom.PRNGKey(1))
        angle_from_direction = np.degrees(np.arccos(np.clip(np.asarray(ray_r) @ direction, -1, 1)))

        results[f'{sample_id}__angle_from_boresight'] = angle_from_boresight
        results[f'{sample_id}__angle_from_direction'] = angle_from_direction
        results[f'{sample_id}__rotation_target'] = direction

        # ---- shared-frame comparison scan: rotate BOTH sources to point at theta=90,
        # phi=90 in one common frame and reconstruct (theta,phi) with the SAME spherical
        # formula for both device types -- no more per-device branching at reconstruction
        # time, and both profiles land near the equator (far from either pole), so the
        # sin(theta) solid-angle Jacobian never needs the near-zero clip-floor hack that
        # broke the old diffuser-at-its-own-pole version.
        if device_type == 'diffuser':
            native_axis1_vals, native_axis2_vals, native_grid_ref = alpha_vals, phi_vals, grid_folded
        else:
            native_axis1_vals, native_axis2_vals, native_grid_ref = theta_vals, phi_vals, grid

        shared_axis, shared_angle = axis_angle_from_reference_np(SHARED_DIRECTION)
        # The collimator's beam is only a few degrees across -- a 1-degree-resolution grid
        # spanning the full 0-180 range (fine for the diffuser's ~90-degree-wide profile)
        # leaves only a handful of bins across it. Zoom the query grid to a window around the
        # shared boresight with much finer bins instead, for the collimator specifically.
        if device_type == 'diffuser':
            scan_theta_vals = np.linspace(0.0, 180.0, 181)
            scan_phi_vals = np.linspace(0.0, 180.0, 181)
        else:
            half_width = 15.0
            scan_theta_vals = np.linspace(boresight_theta - half_width, boresight_theta + half_width, 301)
            scan_phi_vals = np.linspace(boresight_phi - half_width, boresight_phi + half_width, 301)
        THETA_GRID, PHI_GRID = np.meshgrid(scan_theta_vals, scan_phi_vals)

        native_reproj = reproject_native_grid(device_type, native_axis1_vals, native_axis2_vals,
                                               native_grid_ref, shared_axis, shared_angle,
                                               THETA_GRID, PHI_GRID)

        source_scan, _ = measured_profile_source((0.0, 0.0, 0.0), SHARED_DIRECTION, sample_id)
        ray_scan, _, _ = source_scan(args.n_scan, jrandom.PRNGKey(2))
        ray_scan = np.asarray(ray_scan)
        theta_scan = np.degrees(np.arccos(np.clip(ray_scan[:, 2], -1, 1)))
        phi_scan = np.degrees(np.arctan2(ray_scan[:, 1], ray_scan[:, 0])) % 360.0

        theta_edges = bin_edges_from_centers(scan_theta_vals)
        phi_edges = bin_edges_from_centers(scan_phi_vals)
        counts_2d, _, _ = np.histogram2d(phi_scan, theta_scan, bins=[phi_edges, theta_edges])
        # Exact per-bin solid angle (cos(lo)-cos(hi), integrated over the bin) rather than
        # a differential sin(center)*width approximation -- stays correct even for a bin
        # that straddles a pole, though here both profiles sit safely near the equator so
        # it barely matters either way.
        jac_theta = np.cos(np.radians(theta_edges[:-1])) - np.cos(np.radians(theta_edges[1:]))
        scan_intensity = counts_2d / np.maximum(jac_theta, 1e-6)[None, :]

        results[f'{sample_id}__shared_theta_vals'] = scan_theta_vals
        results[f'{sample_id}__shared_phi_vals'] = scan_phi_vals
        results[f'{sample_id}__shared_scan_intensity'] = scan_intensity
        results[f'{sample_id}__shared_native_intensity'] = native_reproj

        print(f'{sample_id} ({device_type}): validated + rotation-checked, '
              f'max |sample-measured| = {np.nanmax(np.abs(sampled_intensity - meas_interp)):.3f}')

    np.savez(args.out, sample_ids=np.array(sample_ids), **results)
    print('saved results ->', args.out)


if __name__ == '__main__':
    main()
