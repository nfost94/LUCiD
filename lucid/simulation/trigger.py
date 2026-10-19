"""Sliding-window readout trigger for the LUCiD data pipeline.

After hit-making (:mod:`lucid.simulation.digitizer`) the detector free-runs with
physics light *and* dark noise. A real DAQ doesn't record everything — it issues
a trigger when the detector-wide hit rate exceeds a threshold and reads out only
the coincident window(s). This module reproduces that: it slides a coincidence
window of width ``window_ns`` over the (detector-wide) digit times, opens a gate
when the multiplicity crosses **up** through ``n_thr``, closes it when the
multiplicity drops back **below**, expands each gate by ``±pad_ns``, and merges
overlaps. Only in-gate digits are kept; the gate list is stored as truth.

This is the SK-like hit-sum / N200 trigger, applied to the digit stream. The
threshold sits well above the dark floor (μ = rate·n_sensors·W): for 4.2 kHz on
~11k PMTs at W=200 ns, μ≈9.2, and because a trigger examines ~1/W ≈ 5·10⁶
windows/s the operating point is ~μ+7σ (n_thr≈30) to keep the accidental rate
below ~0.1 Hz — not the naive 3σ.

:class:`HKTrigger` is the alternative for HK: WCSim's NDigits trigger as the HK
production runs it (fixed readout gates around a trigger time; see its docstring).
A physics-config ``"trigger"`` block with ``"model": "hk"`` selects it
(:func:`trigger_from_block`).

Pure NumPy; operates on the per-event digit time list, host-side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriggerConfig:
    """Sliding-window trigger parameters.

    ``window_ns``     — coincidence window W (SK-like ~200 ns; distinct from the
                        per-sensor integration window of the digitizer).
    ``n_thr``         — detector-wide hit multiplicity to fire (in-W hit count).
    ``pad_before_ns`` — gate expansion before the up-crossing (pre-trigger).
    ``pad_after_ns``  — gate expansion after the down-crossing (post-trigger).
    """
    window_ns: float = 200.0
    n_thr: int = 30
    pad_before_ns: float = 30.0
    pad_after_ns: float = 30.0

    name = "sliding_window"

    def find_gates(self, times: np.ndarray) -> np.ndarray:
        return find_trigger_gates(times, self)

    @classmethod
    def from_block(cls, block):
        """Build from a physics-config ``"trigger"`` block (dict) or None.

        ``None`` disables the trigger (caller treats it as off). A dict may set
        any of ``window_ns``/``n_thr``/``pad_before_ns``/``pad_after_ns`` (also
        accepts a symmetric ``pad_ns`` shorthand).
        """
        if block is None:
            return None
        b = dict(block)
        if "pad_ns" in b:
            b.setdefault("pad_before_ns", b["pad_ns"])
            b.setdefault("pad_after_ns", b["pad_ns"])
            b.pop("pad_ns")
        allowed = {"window_ns", "n_thr", "pad_before_ns", "pad_after_ns"}
        return cls(**{k: v for k, v in b.items() if k in allowed})


def find_trigger_gates(times: np.ndarray, cfg: TriggerConfig) -> np.ndarray:
    """Return the trigger gates ``[[t_start, t_end], ...]`` for one event.

    The sliding count is the trailing-window multiplicity ``m(t) = #hits in
    (t-W, t]`` — it steps +1 at each hit time and -1 at ``hit+W``. A gate spans
    from the up-crossing of ``n_thr`` (minus pad) to the following down-crossing
    (plus pad); overlapping/adjacent gates are merged. Returns an empty ``(0, 2)``
    array when nothing triggers.

    ``pad_before_ns`` is the pre-trigger window: a burst rising slower than it
    can lose its leading edge (the count only clears ``n_thr`` partway up).
    Physical Cherenkov bursts rise in ~10 ns, well inside the pad, so in practice
    the gate brackets the whole burst (measured physics efficiency ~99.9%).
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    n = t.size
    if n == 0:
        return np.empty((0, 2))
    ev_t = np.concatenate([t, t + cfg.window_ns])                 # +1 at hit, -1 at hit+W
    ev_d = np.concatenate([np.ones(n, np.int64), -np.ones(n, np.int64)])
    order = np.lexsort((-ev_d, ev_t))                             # time asc; +1 before -1 at ties
    ev_t, ev_d = ev_t[order], ev_d[order]
    count = np.cumsum(ev_d)
    above = count >= cfg.n_thr
    prev = np.concatenate([[False], above[:-1]])
    starts = ev_t[above & ~prev] - cfg.pad_before_ns              # up-crossings
    ends = ev_t[~above & prev] + cfg.pad_after_ns                 # down-crossings
    gates = np.column_stack([starts, ends])
    if gates.shape[0] == 0:
        return np.empty((0, 2))
    gates = gates[np.argsort(gates[:, 0])]
    merged: list[list[float]] = []
    for s, e in gates:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([float(s), float(e)])
    return np.asarray(merged)


# WCSimWCTriggerBase::AlgNDigits steps its search window, and rounds the trigger time down, by this.
_WCSIM_STEP_NS = 5.0


@dataclass
class HKTrigger:
    """HK trigger: WCSim's NDigits trigger (``WCSimWCTriggerBase::AlgNDigits``,
    ``src/WCSimWCTrigger.cc:240-367``) with the HK production's settings as defaults
    (``run_macros/particle_iso_e.mac``: ``/DAQ/TriggerNDigits/Threshold 25``,
    ``AdjustForNoise true``, ``Window 200 ns``, ``PreTriggerWindow -400 ns``,
    ``PostTriggerWindow +950 ns``).

    * It fires when a ``window_ns`` window first holds more than the threshold's
      digits, the threshold raised for dark noise by round(rate x window x PMTs)
      (``AdjustNDigitsThresholdForNoise``, ``:155-174``). WCSim uses the
      ``/DarkRate/SetDarkRate`` value there, before the dark-rate conversion factor
      the digitizer's ``dark_rate_khz`` includes: 25 + round(4.2 kHz x 200 ns x
      19746) = 42 for HK.
    * The trigger time is the time of the digit that makes it fire, rounded down to
      WCSim's 5 ns search step, and the gate read out is
      ``[t_trig - pad_before_ns, t_trig + pad_after_ns]``, started no earlier than
      the previous gate's end so no digit is read twice (``FillDigitsCollection``).
    * The search resumes at the gate's end counting only later digits, so late light
      (a Michel electron) makes a trigger of its own.

    ``n_thr`` (digits in a window that fire) is the adjusted threshold + 1, so the
    attributes the pipeline records read as they do for :class:`TriggerConfig`.
    """
    threshold: int = 25
    window_ns: float = 200.0
    pad_before_ns: float = 400.0
    pad_after_ns: float = 950.0
    noise_dark_rate_khz: float = 4.2     # 0: no adjustment for noise
    n_pmts: int = 19746                  # HK ID 20" PMTs (geofile Type 1)

    name = "hk"

    @property
    def adjusted_threshold(self) -> int:
        mean_dark = self.noise_dark_rate_khz * 1e3 * self.window_ns * 1e-9 * self.n_pmts
        return self.threshold + int(np.floor(mean_dark + 0.5))     # C++ round()

    @property
    def n_thr(self) -> int:
        return self.adjusted_threshold + 1

    def find_gates(self, times: np.ndarray) -> np.ndarray:
        return find_hk_trigger_gates(times, self)

    @classmethod
    def from_block(cls, block):
        allowed = {"threshold", "window_ns", "pad_before_ns", "pad_after_ns",
                   "noise_dark_rate_khz", "n_pmts"}
        return cls(**{k: v for k, v in dict(block).items() if k in allowed})


def find_hk_trigger_gates(times: np.ndarray, cfg: HKTrigger) -> np.ndarray:
    """Return the :class:`HKTrigger` gates ``[[t_start, t_end], ...]`` for one event.

    The first digit to complete ``n_thr`` digits within ``window_ns`` (the
    ``n_thr``-th of them) is the one WCSim's stepped search fires on: its trigger
    time is the ``n_thr``-th digit at or after the first window start that holds
    ``n_thr`` digits. Returns an empty ``(0, 2)`` array when nothing triggers.
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    n = cfg.n_thr
    gates = []
    start = 0                       # first digit the search counts
    prev_end = -np.inf
    while t.size - start >= n:
        tt = t[start:]
        fires = np.flatnonzero(tt[n - 1:] - tt[:tt.size - n + 1] <= cfg.window_ns)
        if fires.size == 0:
            break
        t_trig = np.floor(tt[fires[0] + n - 1] / _WCSIM_STEP_NS) * _WCSIM_STEP_NS
        end = t_trig + cfg.pad_after_ns
        gates.append([max(t_trig - cfg.pad_before_ns, prev_end), end])
        prev_end = end
        start += int(np.searchsorted(tt, end, side="left"))
    return np.asarray(gates, dtype=np.float64).reshape(-1, 2)


def trigger_from_block(block):
    """The trigger a physics-config ``"trigger"`` block describes: :class:`HKTrigger` for
    ``"model": "hk"``, otherwise :class:`TriggerConfig`; ``None`` disables the trigger."""
    if block is not None and dict(block).get("model") == "hk":
        return HKTrigger.from_block(block)
    return TriggerConfig.from_block(block)


def assign_windows(times: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Per-hit trigger-window index (``0..n_gates-1``); ``-1`` if out of every gate.

    Gates are disjoint (post-merge), so each hit maps to at most one window.
    """
    times = np.asarray(times, dtype=np.float64)
    win = np.full(times.shape[0], -1, dtype=np.int64)
    if gates.shape[0] == 0 or times.size == 0:
        return win
    idx = np.searchsorted(gates[:, 0], times, side="right") - 1   # candidate gate
    ok = idx >= 0
    inside = np.zeros(times.shape[0], dtype=bool)
    inside[ok] = times[ok] <= gates[idx[ok], 1]
    win[inside] = idx[inside]
    return win


def hits_in_gates(times: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: True for hits inside any gate."""
    return assign_windows(times, gates) >= 0


def apply_trigger(sensor_digits: dict, hits_sparse: dict, seg_hits: dict, cfg):
    """Apply the readout trigger to one event's digitizer output.

    Keeps only in-gate digits, **canonically re-sorts** them by
    ``(window, sensor_idx, T)`` (so each window is a contiguous digit slice and
    a PMT's hits within a window are time-ordered), remaps ``digit_idx`` in the
    ``hits_sparse`` / ``seg_hits`` decompositions to the new digit rows, and
    builds the ``per_window`` CSR (``window_start``, ``window_end``,
    ``digit_offsets``).

    Returns ``(sensor_digits, hits_sparse, seg_hits, per_window)`` or ``None``
    when the event produced no trigger (the caller drops the event).
    """
    T = np.asarray(sensor_digits["T"], dtype=np.float64)
    n_dig = T.shape[0]
    gates = cfg.find_gates(T)                          # TriggerConfig or HKTrigger
    if gates.shape[0] == 0:
        return None
    win = assign_windows(T, gates)                    # per-digit window, -1 if out
    kept = np.flatnonzero(win >= 0)
    if kept.size == 0:
        return None

    # Canonical order of kept digits: window, then sensor, then time.
    si = np.asarray(sensor_digits["sensor_idx"])[kept]
    order = np.lexsort((T[kept], si, win[kept]))
    new_of_kept = kept[order]                          # old digit index in new order

    new_sd = {k: np.asarray(sensor_digits[k])[new_of_kept] for k in ("sensor_idx", "PE", "T")}

    # old digit_idx -> new row (or -1 if the digit was dropped)
    remap = np.full(n_dig, -1, dtype=np.int64)
    remap[new_of_kept] = np.arange(new_of_kept.size)

    def _remap(d):
        nd = remap[np.asarray(d["digit_idx"])]
        m = nd >= 0
        out = {k: np.asarray(v)[m] for k, v in d.items()}
        out["digit_idx"] = nd[m].astype(np.int32)
        return out

    new_hits = _remap(hits_sparse)
    new_seg = _remap(seg_hits) if seg_hits is not None else None

    # per_window CSR: window per new digit is non-decreasing after the sort.
    win_sorted = win[new_of_kept]
    counts = np.bincount(win_sorted, minlength=gates.shape[0])
    digit_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    per_window = {
        "window_start": gates[:, 0].astype(np.float64),
        "window_end": gates[:, 1].astype(np.float64),
        "digit_offsets": digit_offsets,
    }
    return new_sd, new_hits, new_seg, per_window
