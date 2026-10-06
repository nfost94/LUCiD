"""Load HKUKLI goniometer calibration-source scans and build their 2D
inverse-CDF sampling tables.

Ported from HKUKLI/injector_profile_comparison.ipynb / HKUKLI/measured_profile_source_job.py.
Pure numpy, one-time setup -- consumed by lucid.sources.calibration_sources.measured_profile_source.
"""

import numpy as np
from pathlib import Path


def dedupe_wraparound_phi(phi_vals, grid):
    phi_mod = np.round(phi_vals % 360.0, 6)
    uniq = np.array(sorted(set(phi_mod)))
    if uniq.size == phi_vals.size:
        return phi_vals, grid
    idx = np.searchsorted(uniq, phi_mod)
    sums = np.zeros((uniq.size, grid.shape[1]))
    counts = np.zeros((uniq.size, 1))
    np.add.at(sums, idx, grid)
    np.add.at(counts, idx, 1)
    return uniq, sums / counts


def load_raw_grid(common_profiles_dir, sample_id):
    d = np.load(Path(common_profiles_dir) / f'{sample_id}.npz', allow_pickle=True)
    theta_vals, phi_vals, grid = d['axis1_vals'], d['axis2_vals'], d['grid']
    device_type = str(d['device_type'])
    phi_vals, grid = dedupe_wraparound_phi(phi_vals, grid)
    return theta_vals, phi_vals, grid, device_type


def fold_theta_diffuser(theta_vals, grid):
    # phi_raw is the device's own roll, not a spatial axis paired with theta_raw
    # -- fold theta_raw and its mirror (180-theta_raw) into alpha=|theta_raw-90|.
    alpha_vals = np.round(np.abs(theta_vals - 90.0), 6)
    uniq_alpha = np.array(sorted(set(alpha_vals)))
    idx = np.searchsorted(uniq_alpha, alpha_vals)
    folded = np.zeros((grid.shape[0], uniq_alpha.size))
    counts = np.zeros(uniq_alpha.size)
    for j, k in enumerate(idx):
        folded[:, k] += grid[:, j]
        counts[k] += 1
    folded /= counts[None, :]
    return uniq_alpha, folded


def bilinear_lookup(axis1_vals, axis2_vals, grid, x, y):
    x = np.clip(x, axis1_vals[0], axis1_vals[-1])
    y = np.clip(y, axis2_vals[0], axis2_vals[-1])
    i1 = np.clip(np.searchsorted(axis1_vals, x, side='right') - 1, 0, axis1_vals.size - 2)
    i2 = np.clip(np.searchsorted(axis2_vals, y, side='right') - 1, 0, axis2_vals.size - 2)
    x0, x1 = axis1_vals[i1], axis1_vals[i1 + 1]
    y0, y1 = axis2_vals[i2], axis2_vals[i2 + 1]
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    g00, g10 = grid[i2, i1], grid[i2, i1 + 1]
    g01, g11 = grid[i2 + 1, i1], grid[i2 + 1, i1 + 1]
    top = g00 * (1 - tx) + g10 * tx
    bot = g01 * (1 - tx) + g11 * tx
    return top * (1 - ty) + bot * ty


def build_2d_cdf_u(u_vals, phi_vals, grid_u, n_fine_u=400, n_fine_phi=200, phi_range=None):
    fine_u = np.linspace(u_vals.min(), u_vals.max(), n_fine_u)
    if phi_range is None:
        fine_phi = np.linspace(phi_vals.min(), phi_vals.max(), n_fine_phi)
    else:
        # phi_range is the true non-overlapping 360deg span; closing point at
        # phi_range[1] is appended below (density = phi_range[0] column again).
        # n_fine_phi - 1 here so the table always has n_fine_phi points: every
        # profile source then has the same shapes and can be stacked/batched.
        fine_phi = np.linspace(phi_range[0], phi_range[1], n_fine_phi - 1, endpoint=False)
    UU, PP = np.meshgrid(fine_u, fine_phi)
    dense = bilinear_lookup(u_vals, phi_vals, grid_u, UU.ravel(), PP.ravel()).reshape(UU.shape)
    if phi_range is not None:
        dense = np.concatenate([dense, dense[:1, :]], axis=0)
        fine_phi = np.concatenate([fine_phi, [phi_range[1]]])
    marginal = np.trapz(dense, fine_phi, axis=0)
    marginal_cdf = np.concatenate([[0.0], np.cumsum(0.5 * (marginal[1:] + marginal[:-1]) * np.diff(fine_u))])
    marginal_cdf = marginal_cdf / marginal_cdf[-1]
    cum = np.cumsum(0.5 * (dense[1:, :] + dense[:-1, :]) * np.diff(fine_phi)[:, None], axis=0)
    cond = np.concatenate([np.zeros((1, n_fine_u)), cum], axis=0)
    col_tot = cond[-1, :]
    col_tot_safe = np.where(col_tot > 0, col_tot, 1.0)
    cond = cond / col_tot_safe[None, :]
    flat_fallback = np.linspace(0.0, 1.0, fine_phi.shape[0])[:, None]
    cond = np.where(col_tot[None, :] > 0, cond, flat_fallback)
    return fine_u, marginal_cdf, fine_phi, cond.T


def composite_profile(alpha_deg, w, p, a0, s):
    # Diffuse super-Gaussian core (w, p) times a Fermi-style soft cutoff edge at a0
    # (softness s) -- see HKUKLI/injector_profile_comparison.ipynb cells 8/16 for the fit.
    core = np.exp(-(np.asarray(alpha_deg) / max(w, 1e-6)) ** (2 * p))
    edge = 1.0 / (1.0 + np.exp((np.asarray(alpha_deg) - a0) / max(s, 1e-6)))
    return core * edge


def build_fit_cdf(params, alpha_max, n_fine_u=400, n_fine_phi=200):
    """(fine_u, marginal_cdf, fine_phi, cond) tables for a fitted composite(w,p,a0,s)
    model, same shape as build_profile_cdf -- phi-uniform (rank-1 in u), so cond comes out
    exactly flat in phi with no special-casing. alpha is the true off-boresight polar angle
    from a real pole (for both device types, once fit_common_profile has recentered the
    collimator's real 2D scan into it), so u=cos(alpha) is already the flat measure -- same
    "no jacobian" property build_2d_cdf_u relies on, just built directly here since the
    density has no phi structure to integrate out.
    """
    fine_u = np.linspace(np.cos(np.radians(alpha_max)), 1.0, n_fine_u)
    alpha_deg = np.degrees(np.arccos(np.clip(fine_u, -1.0, 1.0)))
    density_u = np.clip(composite_profile(alpha_deg, *params), 0, None)
    marginal_cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density_u[1:] + density_u[:-1]) * np.diff(fine_u))])
    marginal_cdf = marginal_cdf / marginal_cdf[-1]
    fine_phi = np.linspace(0.0, 360.0, n_fine_phi)
    cond = np.tile(np.linspace(0.0, 1.0, n_fine_phi), (n_fine_u, 1))
    return fine_u, marginal_cdf, fine_phi, cond


def build_profile_cdf(common_profiles_dir, sample_id):
    """Load common_profiles/<sample_id>.npz and build its 2D inverse-CDF
    tables in (cos(polar), phi).

    Returns (fine_u, marginal_cdf, fine_phi, cond, device_type).
    """
    theta_vals, phi_vals, grid, device_type = load_raw_grid(common_profiles_dir, sample_id)

    phi_range = None
    if device_type == 'diffuser':
        polar_vals, grid_for_cdf = fold_theta_diffuser(theta_vals, grid)
        # Pad one wraparound ghost point each side so bilinear lookup interpolates
        # across the seam; phi_range itself stays the true 360deg span.
        phi_range = (phi_vals[0], phi_vals[0] + 360.0)
        phi_vals = np.concatenate([[phi_vals[-1] - 360.0], phi_vals, [phi_vals[0] + 360.0]])
        grid_for_cdf = np.concatenate([grid_for_cdf[-1:], grid_for_cdf, grid_for_cdf[:1]], axis=0)
    else:
        polar_vals, grid_for_cdf = theta_vals, grid

    u_vals = np.cos(np.radians(polar_vals))
    order = np.argsort(u_vals)
    u_vals, grid_u = u_vals[order], grid_for_cdf[:, order]
    fine_u, marginal_cdf, fine_phi, cond = build_2d_cdf_u(u_vals, phi_vals, grid_u, phi_range=phi_range)
    return fine_u, marginal_cdf, fine_phi, cond, device_type
