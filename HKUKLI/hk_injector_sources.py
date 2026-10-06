"""Build measured-profile calibration sources from the real HK injector survey
(UKLIinfo/LightInjectorsDetails.json). Shared by run_calibration.py and
calibration_gradients_job.py so both build sources the same way.
"""
import json
from pathlib import Path

from lucid.sources import measured_profile_source

INJECTORS_PATH = 'UKLIinfo/LightInjectorsDetails.json'
COMMON_DIR = Path('common_profiles')


def load_injectors(injector_type, detector='ID', injectors_path=INJECTORS_PATH):
    """All survey entries of one type ('diffuser' or 'collimator') for one detector
    region ('ID' or 'OD')."""
    injectors = json.load(open(injectors_path))['injectors']
    return [e for e in injectors if e['type'] == injector_type and e['detector'] == detector]


def _true_position(e):
    """cm -> m. No 'offset' correction: per WCSimLIGen.cc (WCSim/src/WCSimLIGen.cc), the
    injector's own 'position' is used as-is for the emission vertex; 'offset' there shifts
    each individual PHOTON's origin forward along that photon's own sampled/rotated
    direction (a self-shadowing-avoidance trick for the injector housing geometry, applied
    after direction sampling), not a static correction to the injector's mounted position
    along its fixed axis -- what we were doing here."""
    return [c / 100.0 for c in e['position']]


def injector_position_direction(injector_type, idx, detector='ID', injectors_path=INJECTORS_PATH):
    """(position_m, direction) for exactly one injector entry (offset-applied), e.g. to
    build a laser_source/isotropic_source at the same spot as a measured-profile one."""
    entries = load_injectors(injector_type, detector=detector, injectors_path=injectors_path)
    matches = [e for e in entries if e['idx'] == str(idx)]
    if not matches:
        raise ValueError(f"no {injector_type}/{detector} entry with idx={idx!r}")
    e = matches[0]
    return _true_position(e), e['direction']


def injector_sources(injector_type, sample_id, idx=None, *, detector='ID',
                     common_profiles_dir=COMMON_DIR, intensity=1e6,
                     injectors_path=INJECTORS_PATH):
    """measured_profile_source(...) instances at real injector positions, all built from
    the one loaded `sample_id` profile (e.g. 'WarwickDA02' for diffuser,
    'WarwickCol_C01_repeat' for collimator).

    idx : injector 'idx' values (from the JSON) to select, e.g. ['0', '28', '32'] --
        None selects every entry of this type/detector.
    """
    entries = load_injectors(injector_type, detector=detector, injectors_path=injectors_path)
    if idx is not None:
        wanted = set(str(i) for i in idx)
        entries = [e for e in entries if e['idx'] in wanted]

    return [
        measured_profile_source(
            position=_true_position(e),
            direction=e['direction'],
            sample_id=sample_id,
            common_profiles_dir=common_profiles_dir,
            intensity=intensity,
        )
        for e in entries
    ]
