#!/usr/bin/env python3
"""Tabulate WCSim's default PMT photocathode as a LUCiD PMT optical table.

WCSim's default photocathode (PMTCathodePara model 0, as the HK production runs it) meets a photon
at two surfaces:

* outer, water -> glass: no optical surface, plain Fresnel between the two materials' RINDEX
  (water RINDEX1, glass RINDEX_glass, a BK7 dispersion; WCSimConstructMaterials.cc);
* inner, GlassCathodeSurface (dielectric_dielectric, polished, REFLECTIVITY = rgcff, EFFICIENCY 1,
  RINDEX_cathode 1.0): 1 - rgcff is absorbed and detected at any angle; rgcff goes to Fresnel
  from the glass into index 1.0, totally reflected beyond n_water sin(theta) = 1 and otherwise
  mostly transmitted into the PMT interior, whose Air absorbs it (it is never detected).

Summed over bounces between the two surfaces (r = rgcff R_F, t = rgcff (1 - R_F)), per
polarisation, a photon arriving from the water at incidence theta is

    reflected  R = R1 + (1 - R1)^2 r / (1 - R1 r)
    absorbed   A = (1 - R1)(1 - rgcff) / (1 - R1 r)
    lost       T = (1 - R1) t / (1 - R1 r) = 1 - R - A

(planar, incoherent; the 4 mm glass's absorption, ~0.3% a pass, is neglected). The indices are
read from the WCSim source and interpolated linearly in photon energy, as Geant4 evaluates a
material property, and rgcff from the production's tuning macro.

Writes, on the SK-like table grid lucid.simulation.reflection.load_pmt_optical_table reads
(wavelength centres 337.5 + i nm, i < 108; angle centres 0.225 + 0.45 j degrees, j < 200):

* ``<out>/hk50cm_wcsim_optics.txt.gz``: columns wavelength, angle, Rs, Rp, Ts, Tp, As, Ap;
* ``<out>/hk50cm_wcsim_detection_<nm>nm.npz``: the absorptance at one wavelength as a relative
  response A(theta)/A(0) for lucid.simulation.pmt_detection (its first angle bin is A(0), as in
  scripts/convert_sk_pmt_absorptance.py).

The s and p columns are the physical polarisations. Unpolarised use averages them; the SK-like
polarised path swaps the two (SK-IV's convention), so polarised use needs that handled first.
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np

HC_EV_NM = 1239.84198
N_WAVELENGTHS, N_ANGLES = 108, 200
WAVELENGTH_CENTRES = 337.5 + np.arange(N_WAVELENGTHS)
ANGLE_CENTRES = 0.225 + 0.45 * np.arange(N_ANGLES)


def _strip_comments(source):
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
    return re.sub(r'//[^\n]*', '', source)


def _array(source, name):
    """The values of the (uncommented) C array ``name``, units such as ``*GeV`` dropped."""
    block = re.findall(rf'G4double\s+{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}}', source, flags=re.S)
    if len(block) != 1:
        raise ValueError(f'expected one active {name} array, found {len(block)}')
    return np.array([float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block[0])])


def wcsim_indices(materials_cc):
    """(energy_eV, n_water, n_glass, n_cathode) from WCSimConstructMaterials.cc."""
    source = _strip_comments(Path(materials_cc).read_text())
    energy = _array(source, 'ENERGY_water') * 1e9          # written in GeV
    n_water = _array(source, 'RINDEX1')
    n_glass = _array(source, 'RINDEX_glass')
    n_cathode = _array(source, 'RINDEX_cathode')
    if not (energy.size == n_water.size == n_glass.size == 60):
        raise ValueError('WCSim water/glass tables are not on the 60-point ENERGY_water grid')
    if not np.allclose(n_cathode, n_cathode[0]):
        raise ValueError('RINDEX_cathode is not flat')
    return energy, n_water, n_glass, float(n_cathode[0])


def tuning_value(tuning_mac, name):
    match = re.search(rf'^/WCSim/tuning/{name}\s+([-+0-9.eE]+)', Path(tuning_mac).read_text(), re.M)
    if match is None:
        raise ValueError(f'{name} not set in {tuning_mac}')
    return float(match.group(1))


def _fresnel(n1, c1, n2, c2):
    """Fresnel reflectances (s, p) for real indices, cosines c1 (incident) and c2 (transmitted)."""
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return rs * rs, rp * rp


def pmt_optics(cos_theta, n_water, n_glass, n_cathode, rgcff):
    """(R, A, T), each a dict {'s', 'p'}, at water-side incidence cosine(s) ``cos_theta``."""
    sin2 = 1.0 - cos_theta ** 2
    c_glass = np.sqrt(1.0 - (n_water / n_glass) ** 2 * sin2)
    R1 = dict(zip('sp', _fresnel(n_water, cos_theta, n_glass, c_glass)))
    s2t = (n_water / n_cathode) ** 2 * sin2                   # Snell: n_glass sin(theta_g) = n_water sin(theta)
    tir = s2t >= 1.0
    c_cath = np.sqrt(np.where(tir, 0.0, 1.0 - s2t))
    RF = dict(zip('sp', _fresnel(n_glass, c_glass, n_cathode, c_cath)))
    R, A, T = {}, {}, {}
    for pol in 'sp':
        rf = np.where(tir, 1.0, RF[pol])
        r = rgcff * rf
        d = 1.0 - R1[pol] * r
        R[pol] = R1[pol] + (1.0 - R1[pol]) ** 2 * r / d
        A[pol] = (1.0 - R1[pol]) * (1.0 - rgcff) / d
        T[pol] = 1.0 - R[pol] - A[pol]
    return R, A, T


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--wcsim-materials', type=Path,
                    default=Path('/user/nfost94/HKSim/WCSim_alma9/WCSim/src/WCSimConstructMaterials.cc'))
    ap.add_argument('--tuning-mac', type=Path,
                    default=Path('/user/nfost94/HKSim/Current_Analysis/Watchmal/run_macros/tuning_parameters.mac'))
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'config' / 'pmt')
    ap.add_argument('--detection-wavelength', type=float, default=400.0)
    args = ap.parse_args()

    energy, n_water_tab, n_glass_tab, n_cathode = wcsim_indices(args.wcsim_materials)
    rgcff = tuning_value(args.tuning_mac, 'rgcff')
    order = np.argsort(energy)
    e_wl = HC_EV_NM / WAVELENGTH_CENTRES
    n_water = np.interp(e_wl, energy[order], n_water_tab[order])[:, None]
    n_glass = np.interp(e_wl, energy[order], n_glass_tab[order])[:, None]
    cos_theta = np.cos(np.deg2rad(ANGLE_CENTRES))[None, :]
    R, A, T = pmt_optics(cos_theta, n_water, n_glass, n_cathode, rgcff)

    wl = np.broadcast_to(WAVELENGTH_CENTRES[:, None], R['s'].shape)
    ang = np.broadcast_to(ANGLE_CENTRES[None, :], R['s'].shape)
    rows = np.stack([wl, ang, R['s'], R['p'], T['s'], T['p'], A['s'], A['p']], axis=-1).reshape(-1, 8)
    args.out.mkdir(parents=True, exist_ok=True)
    table_path = args.out / 'hk50cm_wcsim_optics.txt.gz'
    with gzip.open(table_path, 'wt') as f:
        f.write('# WCSim default photocathode (scripts/make_wcsim_pmt_optics.py): '
                f'rgcff {rgcff}, cathode index {n_cathode}, WCSim water RINDEX1 and glass RINDEX_glass\n')
        f.write('# wavelength_nm angle_deg Rs Rp Ts Tp As Ap\n')
        np.savetxt(f, rows, fmt='%.7g')

    # Detection response at one wavelength, from the same table block the reflection lookup uses.
    iw = int(np.clip(np.floor(args.detection_wavelength - 337.0), 0, N_WAVELENGTHS - 1))
    a_s, a_p = A['s'][iw], A['p'][iw]
    a = 0.5 * (a_s + a_p)
    mu = np.cos(np.deg2rad(ANGLE_CENTRES))
    asc = np.argsort(mu)
    metadata = {
        'model': 'WCSim default photocathode (PMTCathodePara model 0), unpolarised absorptance ratio',
        'source': str(args.wcsim_materials), 'tuning_mac': str(args.tuning_mac), 'rgcff': rgcff,
        'cathode_index': n_cathode, 'table_wavelength_nm': float(WAVELENGTH_CENTRES[iw]),
        'n_water': float(n_water[iw, 0]), 'n_glass': float(n_glass[iw, 0]),
        'normal_incidence_absorptance': float(a[0]),
        'normalization': 'unpolarised absorptance / first angle-bin absorptance',
    }
    det_path = args.out / f'hk50cm_wcsim_detection_{args.detection_wavelength:.0f}nm.npz'
    np.savez_compressed(
        det_path, incidence_cosine=mu[asc], relative_efficiency=(a / a[0])[asc],
        angle_deg=ANGLE_CENTRES[asc], absorptance=a[asc], absorptance_s=a_s[asc],
        absorptance_p=a_p[asc], wavelength_nm=np.asarray(WAVELENGTH_CENTRES[iw]),
        metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps(metadata, indent=2))
    print(table_path, det_path)


if __name__ == '__main__':
    main()
