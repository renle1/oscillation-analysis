# -*- coding: utf-8 -*-
"""
Streaming-only oscillation detector (single-channel first).

What this module does:
- Accepts stream-style samples with `(device, channel, t, v)`.
- Keeps per-channel ring-buffer state.
- Evaluates a sliding window every update tick.
- Emits `risk_on` / `risk_off` intervals with evidence + hysteresis logic.

What this module does NOT do anymore:
- Batch case processing
- Multiprocessing workers
- External CSV sorting / summary pipelines
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import time
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import config_paths_no_first as cfg


# Signal scoring constants (kept from existing logic)
WIN_SEC = 8.0
BASE_SEC = 6.0
RMS_WIN_SEC = 0.25
TAIL_FRAC = 0.30
MIN_SIGN_CHANGES = 3
DAMPED_RATIO_CUT = 0.70
DAMPED_SCORE_PENALTY_FLOOR = 0.35
EPS_AMP = 1e-6
SCORE_CUT = 1e-6

# Streaming defaults
# Default device label used when source metadata does not provide one.
DEFAULT_DEVICE = str(getattr(cfg, "STREAM_DEVICE", "WMU-SAMPLE"))
# Default update tick interval (seconds) for batch-to-stream replay.
DEFAULT_UPDATE_SEC = float(getattr(cfg, "STREAM_UPDATE_SEC", 2.0))
# Replay mode flag: if True, sleep between ticks to mimic real time.
DEFAULT_REALTIME_SLEEP = bool(getattr(cfg, "STREAM_REALTIME_SLEEP", False))
# Per-tick verbose log flag.
DEFAULT_PRINT_TICK = bool(getattr(cfg, "STREAM_PRINT_TICK", True))

# Finite-state machine phases.
# No active oscillation risk.
PHASE_OFF = "OFF"
# ON candidate stage before confirmation.
PHASE_ON_CANDIDATE = "ON_CANDIDATE"
# Confirmed ON (risk active).
PHASE_ON_CONFIRMED = "ON_CONFIRMED"
# OFF candidate stage while risk is still active.
PHASE_OFF_CANDIDATE = "OFF_CANDIDATE"
# Confirmed OFF transition; normalized back to OFF in state update.
PHASE_OFF_CONFIRMED = "OFF_CONFIRMED"


SampleTuple = tuple[str, str, float, float]
UpdateBatch = list[SampleTuple]
UpdateBatchList = list[UpdateBatch]


@dataclass
class ChannelStreamState:
    """Per-channel stream state for evidence + interval tracking."""

    ring: deque[tuple[float, float]] = field(default_factory=deque)
    ring_time_sorted: bool = True
    last_risk_on: bool = False
    phase: str = PHASE_OFF
    evidence: float = 0.0
    active_start_t: float | None = None
    active_start_update_idx: int | None = None
    off_candidate_start_t: float | None = None
    off_candidate_start_update_idx: int | None = None
    on_event_emitted: bool = False
    damped_streak: int = 0
    on_candidate_streak: int = 0
    on_short_votes: deque[int] = field(default_factory=deque)                    # recent ON short votes (0/1)
    on_soft_votes: deque[int] = field(default_factory=deque)                     # CAL_ON soft-support votes (0/1)
    long_score_hist: deque[tuple[float, float]] = field(default_factory=deque)  # (t_end, log10(score))
    long_off_recent_hist: deque[tuple[float, float]] = field(default_factory=deque)  # recent (t_end, log10(score))
    long_baseline_hist: deque[float] = field(default_factory=deque)              # OFF-state baseline log scores
    long_off_votes: deque[int] = field(default_factory=deque)                     # recent OFF candidate votes (0/1)
    warmup_on_votes: deque[int] = field(default_factory=deque)                    # pre-long ON candidate votes (0/1)
    warmup_on_start_t: float | None = None
    warmup_on_start_update_idx: int | None = None
    last_quality: dict[str, float] = field(default_factory=dict)                  # latest confidence/periodicity metrics
    last_quality_t_end: float = float("nan")                                       # last t_end where periodicity quality was freshly computed
    last_rms_decay: float = float("nan")                                           # last RMS-decay value
    last_rms_decay_r2: float = float("nan")                                        # last RMS-decay fit R^2
    last_rms_decay_n: int = 0                                                      # number of RMS windows used in last estimate
    last_rms_decay_t_end: float = float("nan")                                     # last t_end where RMS-decay was freshly computed
    last_rms_decay_event: float = float("nan")                                     # last event-aware RMS-decay value
    last_rms_decay_event_r2: float = float("nan")                                  # last event-aware RMS-decay fit R^2
    last_rms_decay_event_n: int = 0                                                # number of RMS windows used in last event-aware estimate
    last_rms_decay_event_win_sec: float = float("nan")                             # effective event-aware window length
    last_rms_decay_event_t_end: float = float("nan")                               # last t_end where event-aware RMS-decay was computed
    conf_raw_hist: deque[float] = field(default_factory=deque)                    # raw confidence history for quantile calibration
    periodicity_collapse_streak: int = 0
    long_ready_streak: int = 0
    prev_score_log: float = float("nan")
    prev_e_t: float = float("nan")
    on_support_ema: float = 0.0


@dataclass
class GateFlags:
    """Gate decisions grouped for FSM and tick logging."""

    short_trigger: bool
    short_no_long_ready: bool
    accel_ok: bool
    on_conf_ok: bool
    warmup_on_confirmed: bool
    long_ready: bool
    on_long_gate_ok: bool
    off_vote_core: bool
    long_off_confirmed: bool
    force_off_now: bool
    rms_decay_on_ok: bool
    rms_decay_off_hint: bool
    on_entry_ready: bool


@dataclass
class WindowSlice:
    """Prepared short-window arrays and derived timing values for one tick."""

    tw0: np.ndarray
    vw0: np.ndarray
    tw: np.ndarray
    vw: np.ndarray
    dt: float
    t0: float
    t1: float


@dataclass
class LongStatsSnapshot:
    """Intermediate long-window stats computed from current tick and history."""

    base_med: float
    base_scale: float
    long_ratio_on: float
    long_zmax: float
    long_n: int
    long_ratio_off: float
    long_ratio_off_recent: float
    long_off_n_recent: int


@dataclass
class OffPathSnapshot:
    """OFF-path gate artifacts produced from current tick context."""

    collapse_ok: bool
    off_vote_core: bool
    long_off_confirmed: bool
    force_off_now: bool


@dataclass
class StreamConfig:
    """Runtime stream execution controls."""

    update_sec: float = DEFAULT_UPDATE_SEC
    window_sec: float = WIN_SEC
    realtime_sleep: bool = DEFAULT_REALTIME_SLEEP
    print_tick: bool = DEFAULT_PRINT_TICK
    min_interval_sec_for_alert: float = 8.0
    stitch_gap_sec: float = 6.0
    emit_legacy_rms_aliases: bool = False


@dataclass
class ThresholdConfig:
    """Short-stage score/evidence thresholds and transition guards."""

    risk_cut: float | None = None
    risk_cut_on: float | None = None
    risk_cut_off: float | None = None
    short_dynamic_cut_enabled: bool = True
    short_dynamic_min_baseline: int = 12
    short_dynamic_on_z: float = 3.0
    short_dynamic_off_z: float = 2.0
    short_dynamic_on_max_mult: float = 2.0
    short_dynamic_off_max_mult: float = 1.5
    short_damped_cut_relax: float = 0.85
    rms_decay_gate_enabled: bool = True
    rms_decay_window_sec: float = 20.0
    rms_decay_rms_win_sec: float = 1.0
    rms_decay_step_sec: float = 0.25
    rms_decay_min_windows: int = 8
    rms_decay_on_max: float = 0.08
    rms_decay_off_min: float = 0.03
    rms_decay_cache_enabled: bool = True
    rms_decay_cache_refresh_sec: float = 4.0
    rms_decay_event_enabled: bool = True
    rms_decay_event_max_window_sec: float = 60.0
    rms_decay_event_min_window_sec: float = 12.0
    rms_decay_event_rms_win_sec: float = 1.0
    rms_decay_event_step_sec: float = 0.25
    rms_decay_event_min_windows: int = 8
    evidence_alpha: float = 0.8
    evidence_clip: float = 1.0
    theta_on: float = 0.45
    theta_off: float = 0.15
    on_consecutive_required: int = 2
    on_short_votes_window: int = 3
    on_confirm_min_sec: float = 1.0
    on_accel_score_log_min: float = 0.10
    on_accel_evidence_min: float = 0.05
    on_require_accel_for_candidate: bool = False
    damped_force_off_streak: int = 2
    off_hold_down_sec: float = 4.0
    off_confirm_min_sec: float = 2.0
    off_periodicity_collapse_conf_raw_max: float = 0.45
    off_periodicity_collapse_streak_required: int = 1
    excluded_damped_evidence_penalty: float = 0.15
    excluded_damped_hard_penalty_streak: int = 4
    force_off_require_long_not_on: bool = True
    force_off_long_on_ratio: float | None = None


@dataclass
class LongConfig:
    """Long-window activity, baseline, and warmup gating parameters."""

    long_window_sec: float = 120.0
    long_min_points: int = 20
    long_min_baseline: int = 40
    long_baseline_max: int = 600
    long_z_on: float = 2.5
    long_on_ratio: float = 0.20
    warmup_long_enabled: bool = True
    warmup_on_min_points: int = 10
    warmup_min_baseline: int = 10
    warmup_on_z: float = 8.0
    warmup_on_ratio: float = 0.57
    # Deprecated: raw-confidence warmup hard gate rollback is intentionally maintained.
    warmup_on_conf_raw_min: float = 0.70
    warmup_on_votes_required: int = 2
    warmup_on_votes_window: int = 3
    warmup_on_confirm_min_sec: float = 6.0
    warmup_handoff_grace_ticks: int = 2
    warmup_cancel_on_excluded_damped: bool = True
    long_z_off: float = 1.0
    long_off_ratio: float = 0.20
    long_off_recent_window_sec: float = 30.0
    long_off_votes_required: int = 2
    long_off_votes_window: int = 5
    long_off_recent_min_points: int = 5
    baseline_include_quiet_on: bool = False
    baseline_quiet_on_max_sec: float = 45.0


@dataclass
class PeriodicityConfig:
    """Periodicity-quality and confidence calibration parameters."""

    acf_min_points: int = 24
    acf_min_period_sec: float = 4.0
    acf_max_period_sec: float = 40.0
    confidence_on_min: float = 0.45
    confidence_off_max: float = 0.35
    confidence_use_calibration: bool = False
    confidence_dual_gate_when_cal: bool = True
    confidence_raw_on_min_when_cal: float = 0.30
    confidence_raw_off_max_when_cal: float = 0.35
    confidence_cal_min_points: int = 40
    confidence_cal_hist_max: int = 600
    confidence_cal_off_only: bool = True
    cal_on_soft_mode: bool = True
    cal_on_noise_scale_low: float = 0.08
    cal_on_noise_scale_high: float = 0.30
    cal_on_conf_adapt_gain: float = 0.15
    cal_on_conf_floor: float = 0.35
    cal_on_conf_ceil: float = 0.75
    cal_on_support_score_w: float = 0.35
    cal_on_support_score_acf_bonus: float = 0.08
    cal_on_support_long_w: float = 0.25
    cal_on_support_conf_w: float = 0.30
    cal_on_support_accel_w: float = 0.10
    cal_on_support_enter_min: float = 0.52
    cal_on_support_confirm_min: float = 0.56
    cal_on_support_hold_min: float = 0.45
    cal_on_support_ema_alpha: float = 0.65
    cal_on_confirm_votes_required: int = 2
    cal_on_confirm_votes_window: int = 3
    confidence_w_acf: float = 0.34
    confidence_w_spec: float = 0.33
    confidence_w_env: float = 0.33
    confidence_w_fft: float = 0.10
    freq_band_low_hz: float = 0.15
    freq_band_high_hz: float = 20.0
    freq_linear_detrend: bool = True
    freq_ar1_whiten: bool = False
    periodicity_quality_cache_enabled: bool = True
    periodicity_quality_cache_refresh_sec: float = 6.0


@dataclass
class DetectorConfig:
    """Top-level grouped configuration for the streaming detector."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    long: LongConfig = field(default_factory=LongConfig)
    periodicity: PeriodicityConfig = field(default_factory=PeriodicityConfig)


def _detector_config_override_target_map() -> dict[str, tuple[str, str]]:
    """Map flat override names to `(section, field)` in `DetectorConfig`."""

    cfg = DetectorConfig()
    sections = {
        "stream": cfg.stream,
        "threshold": cfg.threshold,
        "long": cfg.long,
        "periodicity": cfg.periodicity,
    }
    out: dict[str, tuple[str, str]] = {}
    for sec_name, sec_obj in sections.items():
        for f in fields(sec_obj):
            if f.name in out:
                raise RuntimeError(f"duplicate config field name: {f.name}")
            out[f.name] = (sec_name, f.name)
    return out


_CFG_OVERRIDE_TARGETS = _detector_config_override_target_map()


def _apply_detector_overrides(cfg: DetectorConfig, overrides: dict[str, object]) -> DetectorConfig:
    """Apply flat keyword overrides into a deep-copied detector config."""

    out = copy.deepcopy(cfg)
    unknown: list[str] = []
    for k, v in overrides.items():
        target = _CFG_OVERRIDE_TARGETS.get(str(k))
        if target is None:
            unknown.append(str(k))
            continue
        sec_name, field_name = target
        sec_obj = getattr(out, sec_name)
        setattr(sec_obj, field_name, v)
    if unknown:
        raise TypeError(f"Unknown config override(s): {', '.join(sorted(unknown))}")
    return out


PRESET_SAFE = "safe"
PRESET_BALANCED = "balanced"
PRESET_SENSITIVE = "sensitive"
PRESET_CHOICES = (PRESET_SAFE, PRESET_BALANCED, PRESET_SENSITIVE)


def preset_safe() -> DetectorConfig:
    """Return conservative production-safe defaults."""

    # Baseline operations profile: conservative and stable defaults.
    return DetectorConfig()


def preset_balanced() -> DetectorConfig:
    """Return medium-sensitivity defaults with calibrated confidence."""

    # Moderate sensitivity with stronger CAL_ON soft confirmation to suppress sample FP.
    cfg = DetectorConfig()
    cfg.periodicity.confidence_use_calibration = True
    cfg.periodicity.cal_on_support_enter_min = 0.56
    cfg.periodicity.cal_on_support_confirm_min = 0.60
    cfg.periodicity.cal_on_support_hold_min = 0.48
    cfg.periodicity.cal_on_conf_adapt_gain = 0.20
    cfg.periodicity.cal_on_conf_floor = 0.40
    cfg.periodicity.confidence_raw_on_min_when_cal = 0.33
    return cfg


def preset_sensitive() -> DetectorConfig:
    """Return recall-oriented defaults for replay/research usage."""

    # Replay/research profile: favor recall, accept higher FP risk.
    cfg = DetectorConfig()
    cfg.periodicity.confidence_use_calibration = True
    cfg.periodicity.cal_on_support_enter_min = 0.48
    cfg.periodicity.cal_on_support_confirm_min = 0.52
    cfg.periodicity.cal_on_support_hold_min = 0.40
    cfg.threshold.on_short_votes_window = 2
    cfg.threshold.on_consecutive_required = 1
    cfg.threshold.on_confirm_min_sec = 0.5
    return cfg


def make_preset_config(name: str) -> DetectorConfig:
    """Build preset configuration by preset name."""

    n = str(name).strip().lower()
    if n == PRESET_SAFE:
        return preset_safe()
    if n == PRESET_BALANCED:
        return preset_balanced()
    if n == PRESET_SENSITIVE:
        return preset_sensitive()
    raise ValueError(f"Unknown preset: {name}")


class StreamListReceiver:
    """Simple receiver for `on_sample(device, channel, t, v)` stream interface."""

    def __init__(self) -> None:
        self.samples: list[SampleTuple] = []

    def on_sample(self, device: str, channel: str, t: float, v: float) -> None:
        self.samples.append((str(device), str(channel), float(t), float(v)))


def _safe_float(v, default: float = float("-inf")) -> float:
    """Convert value to finite float, otherwise return `default`."""

    try:
        fv = float(v)
    except Exception:
        return default
    if not np.isfinite(fv):
        return default
    return fv


def to_float_np(s: pd.Series) -> np.ndarray:
    """Convert pandas Series to float numpy array with NaN on parse errors."""

    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def infer_time_col(df: pd.DataFrame) -> str:
    """Pick a time-like column, prioritizing numeric-convertible candidates."""
    cols = [str(c) for c in df.columns]
    if not cols:
        raise ValueError("empty dataframe has no columns")

    def _is_time_like(name: str) -> bool:
        n = str(name).strip().lower()
        return (
            ("time" in n)
            or (n in {"t", "ts", "t_s", "sec", "secs", "second", "seconds", "elapsed", "elapsed_s"})
            or n.endswith("_s")
        )

    candidates = [c for c in cols if _is_time_like(c)] or cols

    best_col = candidates[0]
    best_cnt = -1
    for c in candidates:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        cnt = int(np.count_nonzero(np.isfinite(arr)))
        if cnt > best_cnt:
            best_cnt = cnt
            best_col = c
    if best_cnt > 0:
        return best_col

    for c in cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if int(np.count_nonzero(np.isfinite(arr))) > 0:
            return c
    return cols[0]


def duration_over_mask(t: np.ndarray, mask: np.ndarray) -> float:
    """Sum positive time deltas where the left-edge mask is True."""

    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    return float(np.sum(dt[mask[:-1]]))


def tail_sign_changes(x_tail: np.ndarray, eps: float) -> int:
    """Count sign changes in tail signal after epsilon dead-band filtering."""

    xt = np.asarray(x_tail, dtype=float)
    xt = xt[np.isfinite(xt)]
    if xt.size < 10:
        return 0
    xt = xt[np.abs(xt) >= eps]
    if xt.size < 10:
        return 0
    sgn = np.sign(xt)
    return int(np.sum(sgn[1:] * sgn[:-1] < 0))


def score_one_channel_equiv(tw: np.ndarray, vw: np.ndarray, dt: float, t0: float, t1: float):
    """
    Score one channel in the given window.

    Returns:
    - score, A_tail, D_tail, reason
    """
    tb1 = tw[0] + float(BASE_SEC)
    mb = tw <= tb1
    if np.count_nonzero(mb) < 5:
        mb = np.arange(tw.size) < max(5, int(0.25 * tw.size))
    base = float(np.nanmedian(vw[mb]))
    x = vw - base

    k = int(max(3, round(float(BASE_SEC) / float(dt))))
    if (k % 2) == 0:
        k += 1
    trend = pd.Series(x).rolling(window=k, center=True, min_periods=1).median().to_numpy(dtype=float)
    residual = x - trend

    tail_start_t = float(t1) - float(TAIL_FRAC) * float(t1 - t0)
    mtail = tw >= tail_start_t

    sc = tail_sign_changes(residual[mtail], eps=float(EPS_AMP))
    if sc < int(MIN_SIGN_CHANGES):
        return 0.0, np.nan, 0.0, "no_osc_sign_change"

    win_n = int(max(1, round(float(RMS_WIN_SEC) / float(dt))))
    rms_mean = _rolling_mean_trailing_np(
        residual * residual,
        window=int(win_n),
        min_periods=int(win_n),
    )
    rms = np.sqrt(rms_mean)

    head_end_t = float(t0) + (1.0 - float(TAIL_FRAC)) * float(t1 - t0)
    mhead = tw <= head_end_t
    rms_head = rms[mhead]
    rms_tail = rms[mtail]

    A_head = float(np.nanquantile(rms_head, 0.80))
    A_tail = float(np.nanquantile(rms_tail, 0.80))

    ratio = float(A_tail / (A_head + float(EPS_AMP)))
    is_damped = bool(np.isfinite(ratio) and (ratio <= float(DAMPED_RATIO_CUT)))

    thr = float(np.nanquantile(rms_tail, 0.70))
    mask_hi = np.isfinite(rms) & (rms >= thr) & mtail
    D_tail = duration_over_mask(tw, mask_hi)

    if (not np.isfinite(A_tail)) or (A_tail <= 0.0) or (D_tail <= 0.0):
        return 0.0, (A_tail if np.isfinite(A_tail) else np.nan), float(D_tail), "weak_tail"

    score = (A_tail ** 2.0) * (D_tail ** 3.0)
    if not np.isfinite(score):
        score = 0.0
    if is_damped:
        damped_scale = float(np.clip(
            float(ratio) / max(float(DAMPED_RATIO_CUT), float(EPS_AMP)),
            float(DAMPED_SCORE_PENALTY_FLOOR),
            1.0,
        ))
        score = float(score) * float(damped_scale)
        return float(score), float(A_tail), float(D_tail), "excluded_damped"
    return float(score), float(A_tail), float(D_tail), "ok"


def _estimate_dt_from_timestamps(tw: np.ndarray) -> float:
    """Estimate sampling interval from positive finite timestamp deltas."""

    dt = np.diff(np.asarray(tw, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return float(np.median(dt))


def _trim_ring_by_time(ring: deque[tuple[float, float]], keep_from_t: float) -> None:
    """Trim stream ring buffer to keep samples newer than `keep_from_t`."""

    while ring and (float(ring[0][0]) < float(keep_from_t)):
        ring.popleft()


def _trim_score_hist_by_time(hist: deque[tuple[float, float]], keep_from_t: float) -> None:
    """Trim score history to retain points newer than `keep_from_t`."""

    while hist and (float(hist[0][0]) < float(keep_from_t)):
        hist.popleft()


def _rolling_mean_trailing_np(
    x: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> np.ndarray:
    """
    NumPy trailing rolling mean that matches pandas semantics:
    - right-aligned trailing window
    - NaN-aware mean
    - min_periods counts finite values
    """
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    out = np.full(n, np.nan, dtype=float)
    if n <= 0:
        return out
    w = int(window)
    mp = int(min_periods)
    if w <= 0:
        raise ValueError("window must be >= 1")
    if mp <= 0:
        raise ValueError("min_periods must be >= 1")

    finite = np.isfinite(arr)
    vals = np.where(finite, arr, 0.0)
    csum = np.cumsum(vals, dtype=float)
    ccnt = np.cumsum(finite.astype(np.int64), dtype=np.int64)

    idx = np.arange(n, dtype=np.int64)
    j0 = np.maximum(np.int64(0), idx - np.int64(w) + np.int64(1))
    prev = j0 - np.int64(1)
    has_prev = (prev >= 0)

    sum_prev = np.where(has_prev, csum[prev], 0.0)
    cnt_prev = np.where(has_prev, ccnt[prev], np.int64(0))
    sum_win = csum - sum_prev
    cnt_win = ccnt - cnt_prev

    ok = cnt_win >= np.int64(mp)
    out[ok] = sum_win[ok] / cnt_win[ok].astype(float)
    return out


def _local_rms_decay_from_signal(
    tw: np.ndarray,
    vw: np.ndarray,
    *,
    trailing_sec: float,
    rms_win_sec: float,
    step_sec: float,
    min_windows: int,
) -> tuple[float, float, int]:
    """
    Estimate local RMS decay from trailing signal.

    Returns:
    - rms_decay: -slope of log(RMS) vs time (1/sec)
    - r2: linear-fit R^2 in log domain
    - n_windows: number of RMS windows used
    """
    t = np.asarray(tw, dtype=float)
    v = np.asarray(vw, dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    t = t[m]
    v = v[m]
    if t.size < 8:
        return float("nan"), float("nan"), 0
    t_end = float(t[-1])
    t_start = float(t_end - float(trailing_sec))
    m_trail = (t >= t_start) & (t <= t_end)
    t = t[m_trail]
    v = v[m_trail]
    if t.size < 8:
        return float("nan"), float("nan"), 0

    v = v - float(np.nanmedian(v))
    dt = _estimate_dt_from_timestamps(t)
    if (not np.isfinite(dt)) or (dt <= 0.0):
        return float("nan"), float("nan"), 0

    win_n = int(max(4, round(float(rms_win_sec) / float(dt))))
    step_n = int(max(1, round(float(step_sec) / float(dt))))
    if t.size < win_n:
        return float("nan"), float("nan"), 0

    starts = np.arange(0, int(t.size - win_n + 1), int(step_n), dtype=np.int64)
    if starts.size <= 0:
        return float("nan"), float("nan"), 0

    v2 = np.asarray(v * v, dtype=float)
    csum = np.cumsum(v2, dtype=float)
    csum_prev = np.concatenate(([0.0], csum[:-win_n]))
    win_sum_all = csum[win_n - 1 :] - csum_prev
    win_sum = win_sum_all[starts]
    rms = np.sqrt(np.maximum(win_sum / float(win_n), 1e-18))
    tc = t[starts + (win_n // 2)]

    ok = np.isfinite(tc) & np.isfinite(rms) & (rms > 0.0)
    tc = tc[ok]
    rms = rms[ok]
    n_win = int(tc.size)
    if n_win < int(min_windows):
        return float("nan"), float("nan"), n_win
    if float(np.ptp(tc)) <= 0.0:
        return float("nan"), float("nan"), n_win

    y = np.log(np.maximum(rms, 1e-18))
    slope, intercept = np.polyfit(tc, y, 1)
    pred = slope * tc + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = (1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else float("nan")
    return float(-slope), float(r2), n_win


def _event_rms_decay_from_signal(
    tw: np.ndarray,
    vw: np.ndarray,
    *,
    event_start_t: float,
    t_end: float,
    max_window_sec: float = 60.0,
    min_window_sec: float = 12.0,
    rms_win_sec: float = 1.0,
    step_sec: float = 0.25,
    min_windows: int = 8,
) -> tuple[float, float, int, float]:
    """
    Estimate event-aware RMS decay over adaptive window.

    Window definition:
    - t_ws = max(event_start_t, t_end - max_window_sec)
    - returns NaN when effective window is too short

    Returns:
    - rms_decay_event: -slope of log(RMS) vs time (1/sec)
    - r2: linear-fit R^2 in log domain
    - n_windows: number of RMS windows used
    - win_sec: effective window length (t_end - t_ws)
    """
    if (not np.isfinite(event_start_t)) or (not np.isfinite(t_end)):
        return float("nan"), float("nan"), 0, float("nan")
    if (not np.isfinite(max_window_sec)) or (float(max_window_sec) <= 0.0):
        return float("nan"), float("nan"), 0, float("nan")
    if (not np.isfinite(min_window_sec)) or (float(min_window_sec) <= 0.0):
        return float("nan"), float("nan"), 0, float("nan")

    t = np.asarray(tw, dtype=float)
    v = np.asarray(vw, dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    t = t[m]
    v = v[m]
    if t.size < 8:
        return float("nan"), float("nan"), 0, float("nan")

    t_ws = float(max(float(event_start_t), float(t_end) - float(max_window_sec)))
    win_sec = float(max(0.0, float(t_end) - float(t_ws)))
    if float(win_sec) < float(min_window_sec):
        return float("nan"), float("nan"), 0, float(win_sec)

    m_win = (t >= float(t_ws)) & (t <= float(t_end))
    t = t[m_win]
    v = v[m_win]
    if t.size < 8:
        return float("nan"), float("nan"), 0, float(win_sec)

    v = v - float(np.nanmedian(v))
    dt = _estimate_dt_from_timestamps(t)
    if (not np.isfinite(dt)) or (dt <= 0.0):
        return float("nan"), float("nan"), 0, float(win_sec)

    win_n = int(max(4, round(float(rms_win_sec) / float(dt))))
    step_n = int(max(1, round(float(step_sec) / float(dt))))
    if t.size < win_n:
        return float("nan"), float("nan"), 0, float(win_sec)

    starts = np.arange(0, int(t.size - win_n + 1), int(step_n), dtype=np.int64)
    if starts.size <= 0:
        return float("nan"), float("nan"), 0, float(win_sec)

    v2 = np.asarray(v * v, dtype=float)
    csum = np.cumsum(v2, dtype=float)
    csum_prev = np.concatenate(([0.0], csum[:-win_n]))
    win_sum_all = csum[win_n - 1 :] - csum_prev
    win_sum = win_sum_all[starts]
    rms = np.sqrt(np.maximum(win_sum / float(win_n), 1e-18))
    tc = t[starts + (win_n // 2)]

    ok = np.isfinite(tc) & np.isfinite(rms) & (rms > 0.0)
    tc = tc[ok]
    rms = rms[ok]
    n_win = int(tc.size)
    if n_win < int(min_windows):
        return float("nan"), float("nan"), n_win, float(win_sec)
    if float(np.ptp(tc)) <= 0.0:
        return float("nan"), float("nan"), n_win, float(win_sec)

    y = np.log(np.maximum(rms, 1e-18))
    slope, intercept = np.polyfit(tc, y, 1)
    pred = slope * tc + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = (1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else float("nan")
    return float(-slope), float(r2), n_win, float(win_sec)


def _is_risk_active_phase(phase: str) -> bool:
    """Return True for phases that are considered risk-active."""

    return str(phase) in {PHASE_ON_CONFIRMED, PHASE_OFF_CANDIDATE}


def _robust_center_scale(vals: Sequence[float]) -> tuple[float, float]:
    """Return robust `(median, 1.4826*MAD)` for finite input values."""

    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    scale = max(1e-9, 1.4826 * mad)
    return med, scale


def _dynamic_score_cut_from_log_baseline(
    *,
    base_med: float,
    base_scale: float,
    z_thr: float,
    fallback_cut: float,
) -> float:
    """Derive score cutoff from robust log-score baseline statistics."""

    if (not np.isfinite(base_med)) or (not np.isfinite(base_scale)) or (float(base_scale) <= 0.0):
        return float(fallback_cut)
    if (not np.isfinite(z_thr)):
        return float(fallback_cut)
    s_log_cut = float(base_med) + float(z_thr) * float(base_scale)
    if not np.isfinite(s_log_cut):
        return float(fallback_cut)
    s_log_cut = float(np.clip(s_log_cut, -18.0, 18.0))
    dyn_cut = float(10.0 ** s_log_cut)
    if (not np.isfinite(dyn_cut)) or (dyn_cut <= 0.0):
        return float(fallback_cut)
    return float(dyn_cut)


def _empirical_cdf_rank(vals: Sequence[float], x: float) -> float:
    """Return mid-rank empirical CDF percentile for `x` in `vals`."""

    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0 or (not np.isfinite(x)):
        return float("nan")
    arr.sort()
    lo = int(np.searchsorted(arr, x, side="left"))
    hi = int(np.searchsorted(arr, x, side="right"))
    # Mid-rank percentile to reduce ties sensitivity.
    p = (float(lo + hi) * 0.5) / float(arr.size)
    return float(np.clip(p, 0.0, 1.0))


def _calibrate_confidence_quantile(
    conf_raw: float,
    *,
    conf_hist: deque[float],
    min_points: int,
) -> float:
    """Calibrate raw confidence into empirical quantile using history."""

    if (not np.isfinite(conf_raw)) or int(min_points) <= 0:
        return float("nan")
    if len(conf_hist) < int(min_points):
        return float("nan")
    return _empirical_cdf_rank(conf_hist, float(conf_raw))


def _long_activity_ratio(
    long_hist: deque[tuple[float, float]],
    *,
    center: float,
    scale: float,
    z_thr: float,
) -> tuple[float, float, int]:
    """Compute z-threshold exceedance ratio over long-window log-score history."""

    if (not np.isfinite(center)) or (not np.isfinite(scale)) or (scale <= 0):
        return 0.0, float("-inf"), 0
    if not long_hist:
        return 0.0, float("-inf"), 0
    vals = np.asarray([x[1] for x in long_hist], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, float("-inf"), 0
    z = (vals - float(center)) / float(scale)
    ratio = float(np.mean(z >= float(z_thr)))
    zmax = float(np.max(z))
    return ratio, zmax, int(vals.size)


def _acf_peak_periodicity(
    hist: Sequence[tuple[float, float]],
    *,
    min_points: int,
    min_period_sec: float,
    max_period_sec: float,
) -> tuple[float, float, int, int]:
    """
    Estimate periodicity from detrended long-window signal using normalized ACF.

    Returns:
    - acf_peak (max ACF at non-zero lag in period range)
    - acf_period_sec (period estimate from peak lag)
    - acf_lag_steps (peak lag index)
    - n_used (number of samples used)
    """
    arr = np.asarray(list(hist), dtype=float)
    if arr.ndim != 2 or arr.shape[0] < int(min_points):
        return float("nan"), float("nan"), -1, int(arr.shape[0] if arr.ndim == 2 else 0)

    t = arr[:, 0]
    y = arr[:, 1]
    mt = np.isfinite(t) & np.isfinite(y)
    t = t[mt]
    y = y[mt]
    n = int(y.size)
    if n < int(min_points):
        return float("nan"), float("nan"), -1, n

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan"), float("nan"), -1, n
    dt_med = float(np.median(dt))
    if (not np.isfinite(dt_med)) or (dt_med <= 0.0):
        return float("nan"), float("nan"), -1, n

    min_lag = max(1, int(np.ceil(float(min_period_sec) / dt_med)))
    max_lag = int(np.floor(float(max_period_sec) / dt_med))
    max_lag = min(max_lag, n - 1)
    if max_lag < min_lag:
        return float("nan"), float("nan"), -1, n

    idx = np.arange(n, dtype=float)
    if n >= 3:
        p = np.polyfit(idx, y, deg=1)
        trend = p[0] * idx + p[1]
        yd = y - trend
    else:
        yd = y - float(np.nanmedian(y))

    yd = yd - float(np.nanmean(yd))
    var0 = float(np.dot(yd, yd))
    if (not np.isfinite(var0)) or (var0 <= 1e-12):
        return float("nan"), float("nan"), -1, n

    acf_full = np.correlate(yd, yd, mode="full")
    acf = acf_full[n - 1:] / var0
    seg = acf[min_lag:max_lag + 1]
    if seg.size <= 0:
        return float("nan"), float("nan"), -1, n

    rel = int(np.argmax(seg))
    lag = int(min_lag + rel)
    peak = float(seg[rel])
    period_sec = float(lag * dt_med)
    if not np.isfinite(peak):
        return float("nan"), float("nan"), -1, n
    return peak, period_sec, lag, n


def _clip01(x: float) -> float:
    """Clip finite scalar to [0, 1], preserving NaN for non-finite values."""

    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(float(x), 0.0, 1.0))


def _normalize01(x: float, lo: float, hi: float) -> float:
    """Normalize scalar to [0, 1] given bounds, return NaN if invalid."""

    if (not np.isfinite(x)) or (not np.isfinite(lo)) or (not np.isfinite(hi)) or (float(hi) <= float(lo)):
        return float("nan")
    return _clip01((float(x) - float(lo)) / (float(hi) - float(lo)))


def _preprocess_for_periodicity(
    tw: np.ndarray,
    vw: np.ndarray,
    *,
    band_low_hz: float,
    band_high_hz: float,
    linear_detrend: bool,
    ar1_whiten: bool,
) -> tuple[np.ndarray, float]:
    """Detrend/whiten/band-limit signal and return processed signal with `dt`."""

    t = np.asarray(tw, dtype=float)
    y = np.asarray(vw, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 8:
        return np.asarray([], dtype=float), float("nan")

    dt = _estimate_dt_from_timestamps(t)
    if (not np.isfinite(dt)) or (float(dt) <= 0.0):
        return np.asarray([], dtype=float), float("nan")

    y = y - float(np.nanmean(y))

    if bool(linear_detrend) and y.size >= 3:
        tr = t - float(t[0])
        p = np.polyfit(tr, y, deg=1)
        y = y - (p[0] * tr + p[1])

    if bool(ar1_whiten) and y.size >= 3:
        x0 = y[:-1]
        x1 = y[1:]
        den = float(np.dot(x0, x0))
        if den > 1e-12:
            phi = float(np.dot(x0, x1) / den)
            phi = float(np.clip(phi, -0.99, 0.99))
            yw = np.empty_like(y)
            yw[0] = y[0]
            yw[1:] = y[1:] - phi * y[:-1]
            y = yw

    nyq = 0.5 / float(dt)
    flo = max(0.0, float(band_low_hz))
    fhi = min(float(band_high_hz), nyq * 0.98)
    if (not np.isfinite(flo)) or (not np.isfinite(fhi)) or (fhi <= flo):
        return np.asarray([], dtype=float), float(dt)

    n = int(y.size)
    freqs = np.fft.rfftfreq(n, d=float(dt))
    yf = np.fft.rfft(y)
    mask = (freqs >= flo) & (freqs <= fhi)
    if int(np.count_nonzero(mask)) <= 0:
        return np.asarray([], dtype=float), float(dt)
    yf = np.where(mask, yf, 0.0)
    y_bp = np.fft.irfft(yf, n=n)
    y_bp = y_bp - float(np.nanmean(y_bp))
    return np.asarray(y_bp, dtype=float), float(dt)


def _welch_peak_and_concentration(
    y: np.ndarray,
    *,
    dt: float,
    f_low_hz: float,
    f_high_hz: float,
) -> tuple[float, float, int]:
    """Estimate dominant Welch frequency and peak concentration in band."""

    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = int(yy.size)
    if n < 16 or (not np.isfinite(dt)) or (float(dt) <= 0.0):
        return float("nan"), float("nan"), 0

    nperseg = min(256, n)
    if nperseg < 16:
        return float("nan"), float("nan"), 0
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    win_pow = float(np.sum(window * window))
    if win_pow <= 1e-12:
        return float("nan"), float("nan"), 0

    pxx_acc = None
    nseg = 0
    for s in range(0, n - nperseg + 1, step):
        seg = yy[s:s + nperseg]
        seg = seg - float(np.mean(seg))
        spec = np.fft.rfft(seg * window)
        pxx = (np.abs(spec) ** 2.0) / win_pow
        if pxx_acc is None:
            pxx_acc = pxx
        else:
            pxx_acc = pxx_acc + pxx
        nseg += 1
    if (pxx_acc is None) or (nseg <= 0):
        return float("nan"), float("nan"), 0

    pxx = pxx_acc / float(nseg)
    freqs = np.fft.rfftfreq(nperseg, d=float(dt))
    fb = np.isfinite(freqs) & (freqs >= float(f_low_hz)) & (freqs <= float(f_high_hz))
    if int(np.count_nonzero(fb)) <= 0:
        return float("nan"), float("nan"), int(nseg)

    pb = pxx[fb]
    fbv = freqs[fb]
    if pb.size == 0 or (not np.any(np.isfinite(pb))):
        return float("nan"), float("nan"), int(nseg)

    idx = int(np.nanargmax(pb))
    f_peak = float(fbv[idx])
    total = float(np.nansum(pb))
    if (not np.isfinite(total)) or (total <= 0.0):
        return f_peak, float("nan"), int(nseg)

    i0 = max(0, idx - 1)
    i1 = min(pb.size - 1, idx + 1)
    peak_band = float(np.nansum(pb[i0:i1 + 1]))
    c_spec = _clip01(peak_band / total)
    return f_peak, c_spec, int(nseg)


def _zero_cross_frequency(y: np.ndarray, *, dt: float) -> tuple[float, int]:
    """Estimate frequency from median half-period of zero crossings."""

    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = int(yy.size)
    if n < 8 or (not np.isfinite(dt)) or (float(dt) <= 0.0):
        return float("nan"), 0

    crosses: list[float] = []
    for i in range(n - 1):
        y0 = float(yy[i])
        y1 = float(yy[i + 1])
        if (y0 == 0.0) and (y1 == 0.0):
            continue
        if (y0 <= 0.0 and y1 > 0.0) or (y0 >= 0.0 and y1 < 0.0):
            den = (y1 - y0)
            frac = (0.5 if abs(den) <= 1e-12 else (-y0 / den))
            frac = float(np.clip(frac, 0.0, 1.0))
            crosses.append((float(i) + frac) * float(dt))

    if len(crosses) < 4:
        return float("nan"), len(crosses)

    hc = np.diff(np.asarray(crosses, dtype=float))
    hc = hc[np.isfinite(hc) & (hc > 0.0)]
    if hc.size < 3:
        return float("nan"), len(crosses)

    half_period = float(np.median(hc))
    if (not np.isfinite(half_period)) or (half_period <= 0.0):
        return float("nan"), len(crosses)
    f_zc = 1.0 / (2.0 * half_period)
    return float(f_zc), len(crosses)


def _acf_peak_from_signal(
    y: np.ndarray,
    *,
    dt: float,
    min_points: int,
    min_period_sec: float,
    max_period_sec: float,
) -> tuple[float, float, int, int]:
    """Find ACF peak and period within lag bounds on a preprocessed signal."""

    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = int(yy.size)
    if n < int(min_points) or (not np.isfinite(dt)) or (float(dt) <= 0.0):
        return float("nan"), float("nan"), -1, n

    min_lag = max(1, int(np.ceil(float(min_period_sec) / float(dt))))
    max_lag = min(int(np.floor(float(max_period_sec) / float(dt))), n - 1)
    if max_lag < min_lag:
        return float("nan"), float("nan"), -1, n

    y0 = yy - float(np.mean(yy))
    var0 = float(np.dot(y0, y0))
    if (not np.isfinite(var0)) or (var0 <= 1e-12):
        return float("nan"), float("nan"), -1, n

    acf_full = np.correlate(y0, y0, mode="full")
    acf = acf_full[n - 1:] / var0
    seg = acf[min_lag:max_lag + 1]
    if seg.size <= 0:
        return float("nan"), float("nan"), -1, n

    ridx = int(np.argmax(seg))
    lag = int(min_lag + ridx)
    peak = float(seg[ridx])
    period_sec = float(lag * float(dt))
    if not np.isfinite(peak):
        return float("nan"), float("nan"), -1, n
    return peak, period_sec, lag, n


def _envelope_stability_score(y: np.ndarray, *, dt: float) -> float:
    """Score envelope stability using robust variation of smoothed magnitude."""

    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    if yy.size < 8 or (not np.isfinite(dt)) or (float(dt) <= 0.0):
        return float("nan")

    env = np.abs(yy)
    k = int(max(3, round(0.25 / float(dt))))
    sm = _rolling_mean_trailing_np(
        env,
        window=int(k),
        min_periods=int(max(2, k // 2)),
    )
    sm = sm[np.isfinite(sm)]
    if sm.size < 4:
        return float("nan")
    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med)))
    cv = float(mad / (med + 1e-9))
    return _clip01(1.0 / (1.0 + 4.0 * cv))


def _fft_peak_and_concentration(
    y: np.ndarray,
    *,
    dt: float,
    f_low_hz: float,
    f_high_hz: float,
) -> tuple[float, float]:
    """Estimate FFT-peak frequency and local peak energy concentration."""

    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = int(yy.size)
    if n < 16 or (not np.isfinite(dt)) or (float(dt) <= 0.0):
        return float("nan"), float("nan")

    y0 = yy - float(np.mean(yy))
    window = np.hanning(n)
    win_pow = float(np.sum(window * window))
    if win_pow <= 1e-12:
        return float("nan"), float("nan")

    spec = np.fft.rfft(y0 * window)
    pxx = (np.abs(spec) ** 2.0) / win_pow
    freqs = np.fft.rfftfreq(n, d=float(dt))
    fb = np.isfinite(freqs) & (freqs >= float(f_low_hz)) & (freqs <= float(f_high_hz))
    if int(np.count_nonzero(fb)) <= 0:
        return float("nan"), float("nan")

    pb = pxx[fb]
    fbv = freqs[fb]
    if pb.size == 0 or (not np.any(np.isfinite(pb))):
        return float("nan"), float("nan")

    idx = int(np.nanargmax(pb))
    f_peak = float(fbv[idx])
    total = float(np.nansum(pb))
    if (not np.isfinite(total)) or (total <= 0.0):
        return f_peak, float("nan")

    i0 = max(0, idx - 1)
    i1 = min(pb.size - 1, idx + 1)
    peak_band = float(np.nansum(pb[i0:i1 + 1]))
    c_fft = _clip01(peak_band / total)
    return f_peak, c_fft


def _compute_periodicity_quality(
    tw: np.ndarray,
    vw: np.ndarray,
    *,
    acf_min_points: int,
    acf_min_period_sec: float,
    acf_max_period_sec: float,
    band_low_hz: float,
    band_high_hz: float,
    linear_detrend: bool,
    ar1_whiten: bool,
    w_acf: float,
    w_spec: float,
    w_env: float,
    w_fft: float,
) -> dict[str, float]:
    """Compute periodicity quality components and fused confidence metrics."""

    yb, dt = _preprocess_for_periodicity(
        tw,
        vw,
        band_low_hz=float(band_low_hz),
        band_high_hz=float(band_high_hz),
        linear_detrend=bool(linear_detrend),
        ar1_whiten=bool(ar1_whiten),
    )
    if yb.size <= 0 or (not np.isfinite(dt)):
        return {
            "confidence": float("nan"),
            "c_acf": float("nan"),
            "c_spec": float("nan"),
            "c_env": float("nan"),
            "c_fft": float("nan"),
            "c_freq_agree": float("nan"),
            "f_welch": float("nan"),
            "f_zc": float("nan"),
            "f_fft": float("nan"),
            "acf_peak": float("nan"),
            "acf_period_sec": float("nan"),
            "acf_lag_steps": -1.0,
            "acf_n": 0.0,
        }

    acf_peak, acf_period_sec, acf_lag_steps, acf_n = _acf_peak_from_signal(
        yb,
        dt=float(dt),
        min_points=int(acf_min_points),
        min_period_sec=float(acf_min_period_sec),
        max_period_sec=float(acf_max_period_sec),
    )
    c_acf = _normalize01(acf_peak, 0.10, 0.60) if np.isfinite(acf_peak) else float("nan")

    f_welch, c_spec, _ = _welch_peak_and_concentration(
        yb,
        dt=float(dt),
        f_low_hz=float(band_low_hz),
        f_high_hz=float(band_high_hz),
    )
    f_fft, c_fft = _fft_peak_and_concentration(
        yb,
        dt=float(dt),
        f_low_hz=float(band_low_hz),
        f_high_hz=float(band_high_hz),
    )
    f_zc, _ = _zero_cross_frequency(yb, dt=float(dt))
    c_env = _envelope_stability_score(yb, dt=float(dt))

    if np.isfinite(f_welch) and np.isfinite(f_zc):
        f_ref = max(0.5, abs(float(f_welch)), abs(float(f_zc)))
        c_freq_agree = _clip01(np.exp(-abs(float(f_welch) - float(f_zc)) / f_ref))
    else:
        c_freq_agree = float("nan")

    parts = {
        "c_acf": c_acf,
        "c_spec": c_spec,
        "c_env": c_env,
        "c_fft": c_fft,
    }
    weights = {
        "c_acf": max(0.0, float(w_acf)),
        "c_spec": max(0.0, float(w_spec)),
        "c_env": max(0.0, float(w_env)),
        "c_fft": max(0.0, float(w_fft)),
    }
    den = 0.0
    num = 0.0
    for k, w in weights.items():
        v = parts[k]
        if np.isfinite(v) and (w > 0.0):
            num += w * float(v)
            den += w
    confidence_base = (float(num / den) if den > 0.0 else float("nan"))
    confidence_raw = confidence_base
    if np.isfinite(confidence_base) and np.isfinite(c_freq_agree):
        confidence_raw = float(confidence_base * (0.5 + 0.5 * float(c_freq_agree)))

    return {
        "confidence": float(confidence_raw) if np.isfinite(confidence_raw) else float("nan"),
        "confidence_raw": float(confidence_raw) if np.isfinite(confidence_raw) else float("nan"),
        "c_acf": float(c_acf) if np.isfinite(c_acf) else float("nan"),
        "c_spec": float(c_spec) if np.isfinite(c_spec) else float("nan"),
        "c_env": float(c_env) if np.isfinite(c_env) else float("nan"),
        "c_fft": float(c_fft) if np.isfinite(c_fft) else float("nan"),
        "c_freq_agree": float(c_freq_agree) if np.isfinite(c_freq_agree) else float("nan"),
        "f_welch": float(f_welch) if np.isfinite(f_welch) else float("nan"),
        "f_zc": float(f_zc) if np.isfinite(f_zc) else float("nan"),
        "f_fft": float(f_fft) if np.isfinite(f_fft) else float("nan"),
        "acf_peak": float(acf_peak) if np.isfinite(acf_peak) else float("nan"),
        "acf_period_sec": float(acf_period_sec) if np.isfinite(acf_period_sec) else float("nan"),
        "acf_lag_steps": float(acf_lag_steps),
        "acf_n": float(acf_n),
    }


def _pick_channel_columns(
    vdf: pd.DataFrame,
    *,
    tcol: str,
    target_channels: Sequence[str] | None,
    max_channels: int | None,
) -> list[str]:
    """Select numeric channel columns with metadata-aware prioritization."""

    raw_cols = [str(c) for c in vdf.columns if str(c) != str(tcol)]
    if not raw_cols:
        return []

    numeric_meta: dict[str, tuple[int, float, bool, bool]] = {}
    for c in raw_cols:
        arr = to_float_np(vdf[c])
        fm = np.isfinite(arr)
        cnt = int(np.count_nonzero(fm))
        if cnt <= 0:
            continue
        std = float(np.nanstd(arr[fm])) if cnt > 0 else 0.0
        name = c.strip().lower()
        is_voltage_like = ("volt" in name) or ("v_pu" in name) or ("voltage" in name) or (name == "v")
        is_quality_like = ("quality" in name) or (name in {"q", "flag", "status"})
        numeric_meta[c] = (cnt, std, is_voltage_like, is_quality_like)

    if not numeric_meta:
        return []

    if target_channels is not None:
        out = []
        for c in target_channels:
            cs = str(c).strip()
            if cs and cs in numeric_meta:
                out.append(cs)
    else:
        out = list(numeric_meta.keys())
        non_quality = [c for c in out if (not numeric_meta[c][3])]
        # Prefer non-metadata numeric channels by default when auto-selecting.
        if non_quality:
            out = non_quality
        out.sort(
            key=lambda c: (
                1 if numeric_meta[c][2] else 0,            # voltage-like first
                1 if (not numeric_meta[c][3]) else 0,      # non-quality first
                1 if numeric_meta[c][1] > 0 else 0,        # non-flat first
                numeric_meta[c][1],                        # larger std first
                numeric_meta[c][0],                        # more valid points first
            ),
            reverse=True,
        )

    if max_channels is not None:
        mch = int(max_channels)
        if mch <= 0:
            return []
        out = out[:mch]
    return out


def build_update_batches_from_voltage_csv(
    vcsv: str,
    *,
    device: str = DEFAULT_DEVICE,
    update_sec: float = DEFAULT_UPDATE_SEC,
    target_channels: Sequence[str] | None = None,
    max_channels: int | None = 1,
) -> UpdateBatchList:
    """
    Convert CSV into update batches.

    Returns:
    - list of updates; each update is a list of `(device, channel, t, v)`.
    """
    if (not np.isfinite(update_sec)) or (float(update_sec) <= 0.0):
        raise ValueError("update_sec must be positive")

    vdf = pd.read_csv(vcsv, encoding="utf-8-sig")
    if vdf.shape[0] == 0 or vdf.shape[1] < 2:
        return []

    tcol = infer_time_col(vdf)
    channel_cols = _pick_channel_columns(
        vdf,
        tcol=tcol,
        target_channels=target_channels,
        max_channels=max_channels,
    )
    if not channel_cols:
        return []

    t_arr = to_float_np(vdf[tcol])
    m_t = np.isfinite(t_arr)
    if int(np.count_nonzero(m_t)) == 0:
        return []

    rows = vdf.loc[m_t, channel_cols].copy()
    rows.insert(0, "__t__", t_arr[m_t])
    rows = rows.sort_values("__t__", kind="mergesort")

    t_start = float(rows["__t__"].iloc[0])
    if rows.empty:
        return []

    t_last = float(rows["__t__"].iloc[-1])
    eff_duration = max(1e-12, max(0.0, t_last - t_start) + 1e-12)
    n_updates = int(max(1, np.ceil(eff_duration / float(update_sec))))
    batches: UpdateBatchList = [[] for _ in range(n_updates)]
    t_rows = rows["__t__"].to_numpy(dtype=float, copy=False)
    idx_rows = ((t_rows - float(t_start)) / float(update_sec)).astype(np.int64, copy=False)
    idx_valid = (idx_rows >= 0) & (idx_rows < int(n_updates))
    if not np.any(idx_valid):
        return batches

    vals = rows[channel_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
    dev_s = str(device)
    valid_rows = np.flatnonzero(idx_valid)
    ch_names = [str(c) for c in channel_cols]
    for r in valid_rows:
        idx = int(idx_rows[r])
        t = float(t_rows[r])
        rowv = vals[r]
        for j, ch_s in enumerate(ch_names):
            fv = float(rowv[j])
            if not np.isfinite(fv):
                continue
            batches[idx].append((dev_s, ch_s, t, fv))
    return batches


def ingest_batches_to_list(
    batches: UpdateBatchList,
    *,
    receiver: StreamListReceiver | None = None,
) -> list[SampleTuple]:
    """Flatten update batches into sample tuples via stream receiver API."""

    recv = receiver if receiver is not None else StreamListReceiver()
    for batch in batches:
        for device, channel, t, v in batch:
            recv.on_sample(device, channel, t, v)
    return recv.samples


def _finite_or_nan(v: float) -> float:
    """Convert finite numeric values to float, otherwise return np.nan."""

    return float(v) if np.isfinite(v) else np.nan


LEGACY_RMS_ALIAS_MAP = {
    "rms_decay": "rms_decay_local",
    "rms_decay_r2": "rms_decay_local_r2",
    "rms_decay_n": "rms_decay_local_n",
    "rms_decay_on_ok": "rms_decay_local_on_ok",
    "rms_decay_off_hint": "rms_decay_local_off_hint",
}

# Legacy alias cleanup plan:
# 1) Default: emit canonical (`rms_decay_local*`) keys only.
# 2) Compatibility mode: emit legacy aliases only when explicitly requested.
# 3) Final: remove legacy alias compatibility path after consumer migration.


def _apply_legacy_rms_aliases(payload: dict[str, object]) -> None:
    """Populate legacy RMS alias keys from canonical `rms_decay_local*` fields."""

    for legacy_key, canonical_key in LEGACY_RMS_ALIAS_MAP.items():
        payload[legacy_key] = payload.get(canonical_key, np.nan)


def _set_quality_rms_metrics(
    quality: dict[str, object],
    *,
    rms_decay_local: float,
    rms_decay_local_r2: float,
    rms_decay_local_n: int,
    rms_decay_local_on_ok: bool | int,
    rms_decay_local_off_hint: bool | int,
    rms_decay_event: float,
    rms_decay_event_r2: float,
    rms_decay_event_n: int,
    rms_decay_event_win_sec: float,
) -> None:
    """Write canonical RMS metrics into quality payload."""

    quality["rms_decay_local"] = float(rms_decay_local) if np.isfinite(rms_decay_local) else float("nan")
    quality["rms_decay_local_r2"] = float(rms_decay_local_r2) if np.isfinite(rms_decay_local_r2) else float("nan")
    quality["rms_decay_local_n"] = int(rms_decay_local_n)
    quality["rms_decay_local_on_ok"] = int(rms_decay_local_on_ok)
    quality["rms_decay_local_off_hint"] = int(rms_decay_local_off_hint)
    quality["rms_decay_event"] = float(rms_decay_event) if np.isfinite(rms_decay_event) else float("nan")
    quality["rms_decay_event_r2"] = float(rms_decay_event_r2) if np.isfinite(rms_decay_event_r2) else float("nan")
    quality["rms_decay_event_n"] = int(rms_decay_event_n)
    quality["rms_decay_event_win_sec"] = float(rms_decay_event_win_sec) if np.isfinite(rms_decay_event_win_sec) else float("nan")


def _extract_quality_rms_metrics(quality: dict[str, object]) -> dict[str, float | int]:
    """Read RMS metrics from quality payload with canonical-first fallback."""

    rms_decay_local = float(quality.get("rms_decay_local", quality.get("rms_decay", np.nan)))
    rms_decay_local_r2 = float(quality.get("rms_decay_local_r2", quality.get("rms_decay_r2", np.nan)))
    rms_decay_local_n = int(quality.get("rms_decay_local_n", quality.get("rms_decay_n", 0)))
    rms_decay_local_on_ok = int(quality.get("rms_decay_local_on_ok", quality.get("rms_decay_on_ok", 0)))
    rms_decay_local_off_hint = int(quality.get("rms_decay_local_off_hint", quality.get("rms_decay_off_hint", 0)))
    return {
        "rms_decay_local": float(rms_decay_local),
        "rms_decay_local_r2": float(rms_decay_local_r2),
        "rms_decay_local_n": int(rms_decay_local_n),
        "rms_decay_local_on_ok": int(rms_decay_local_on_ok),
        "rms_decay_local_off_hint": int(rms_decay_local_off_hint),
        "rms_decay": float(rms_decay_local),
        "rms_decay_r2": float(rms_decay_local_r2),
        "rms_decay_n": int(rms_decay_local_n),
        "rms_decay_on_ok": int(rms_decay_local_on_ok),
        "rms_decay_off_hint": int(rms_decay_local_off_hint),
        "rms_decay_event": float(quality.get("rms_decay_event", np.nan)),
        "rms_decay_event_r2": float(quality.get("rms_decay_event_r2", np.nan)),
        "rms_decay_event_n": int(quality.get("rms_decay_event_n", 0)),
        "rms_decay_event_win_sec": float(quality.get("rms_decay_event_win_sec", np.nan)),
    }


def _warn_deprecated_warmup_params(*, warmup_on_conf_raw_min: float) -> None:
    """Warn only when deprecated warmup raw-confidence param is explicitly changed."""

    default_val = float(LongConfig().warmup_on_conf_raw_min)
    if np.isfinite(warmup_on_conf_raw_min) and (not np.isclose(float(warmup_on_conf_raw_min), default_val)):
        print(
            "[DEPRECATED] warmup_on_conf_raw_min is ignored in warmup_core "
            "(raw-confidence rollback remains active)."
        )


def _build_risk_event_metrics(
    *,
    score: float,
    confidence: float,
    confidence_raw: float,
    confidence_cal: float,
    cal_on_conf_thr: float,
    on_support: float,
    on_support_ema: float,
    on_soft_votes_sum: int,
    on_soft_votes_n: int,
    c_acf: float,
    c_spec: float,
    c_env: float,
    c_fft: float,
    c_freq_agree: float,
    f_welch: float,
    f_zc: float,
    f_fft: float,
    A_tail: float,
    D_tail: float,
    rms_decay: float,
    rms_decay_r2: float,
    rms_decay_n: int,
    rms_decay_on_ok: bool | int,
    rms_decay_off_hint: bool | int,
    rms_decay_event: float = float("nan"),
    rms_decay_event_r2: float = float("nan"),
    rms_decay_event_n: int = 0,
    rms_decay_event_win_sec: float = float("nan"),
    reason: str,
    transition_reason: str,
    evidence: float,
    long_ratio_on: float,
    long_ratio_off: float,
    long_ratio_off_recent: float,
    long_off_n_recent: int,
    long_off_votes_sum: int,
    long_off_votes_n: int,
    acf_peak: float,
    acf_period_sec: float,
    acf_lag_steps: int,
    acf_n: int,
    long_zmax: float,
    long_n: int,
    baseline_n: int,
    emit_legacy_rms_aliases: bool = False,
) -> dict:
    """Build shared risk-event metrics payload."""

    payload = {
        "score": _finite_or_nan(score),
        "risk_score": _finite_or_nan(score),
        "confidence": _finite_or_nan(confidence),
        "confidence_raw": _finite_or_nan(confidence_raw),
        "confidence_cal": _finite_or_nan(confidence_cal),
        "cal_on_conf_thr": _finite_or_nan(cal_on_conf_thr),
        "on_support": _finite_or_nan(on_support),
        "on_support_ema": _finite_or_nan(on_support_ema),
        "on_soft_votes_sum": int(on_soft_votes_sum),
        "on_soft_votes_n": int(on_soft_votes_n),
        "c_acf": _finite_or_nan(c_acf),
        "c_spec": _finite_or_nan(c_spec),
        "c_env": _finite_or_nan(c_env),
        "c_fft": _finite_or_nan(c_fft),
        "c_freq_agree": _finite_or_nan(c_freq_agree),
        "f_welch": _finite_or_nan(f_welch),
        "f_zc": _finite_or_nan(f_zc),
        "f_fft": _finite_or_nan(f_fft),
        "A_tail": _finite_or_nan(A_tail),
        "D_tail": _finite_or_nan(D_tail),
        "rms_decay_local": _finite_or_nan(rms_decay),
        "rms_decay_local_r2": _finite_or_nan(rms_decay_r2),
        "rms_decay_local_n": int(rms_decay_n),
        "rms_decay_local_on_ok": int(rms_decay_on_ok),
        "rms_decay_local_off_hint": int(rms_decay_off_hint),
        "rms_decay_event": _finite_or_nan(rms_decay_event),
        "rms_decay_event_r2": _finite_or_nan(rms_decay_event_r2),
        "rms_decay_event_n": int(rms_decay_event_n),
        "rms_decay_event_win_sec": _finite_or_nan(rms_decay_event_win_sec),
        "reason": str(reason),
        "transition_reason": str(transition_reason),
        "evidence": float(evidence),
        "long_ratio_on": _finite_or_nan(long_ratio_on),
        "long_ratio_off": _finite_or_nan(long_ratio_off),
        "long_ratio_off_recent": _finite_or_nan(long_ratio_off_recent),
        "long_off_n_recent": int(long_off_n_recent),
        "long_off_votes_sum": int(long_off_votes_sum),
        "long_off_votes_n": int(long_off_votes_n),
        "acf_peak": _finite_or_nan(acf_peak),
        "acf_period_sec": _finite_or_nan(acf_period_sec),
        "acf_lag_steps": int(acf_lag_steps),
        "acf_n": int(acf_n),
        "long_zmax": _finite_or_nan(long_zmax),
        "long_n": int(long_n),
        "baseline_n": int(baseline_n),
    }
    if bool(emit_legacy_rms_aliases):
        _apply_legacy_rms_aliases(payload)
    return payload


def _emit_or_stitch_interval(
    *,
    key: tuple[str, str],
    start_t: float,
    start_update_idx: int,
    end_t: float,
    end_update_idx: int,
    end_reason: str,
    status: str,
    events: list[dict],
    last_interval_by_key: dict[tuple[str, str], dict],
    next_interval_id: int,
    stitch_gap_sec: float,
) -> tuple[dict, int]:
    """Emit a new final interval event or stitch into a nearby prior one."""

    duration_sec = float(end_t - start_t)
    prev = last_interval_by_key.get(key)
    if (
        prev is not None
        and prev.get("status") in {"closed", "open"}
        and np.isfinite(prev.get("end_t", np.nan))
        and (float(start_t) - float(prev["end_t"]) <= float(stitch_gap_sec))
    ):
        prev["end_t"] = float(end_t)
        prev["end_update_idx"] = int(end_update_idx)
        prev["duration_sec"] = float(prev["end_t"] - float(prev["start_t"]))
        prev["status"] = str(status)
        prev["end_reason"] = str(end_reason)
        prev["stitch_count"] = int(prev.get("stitch_count", 0)) + 1
        print(
            f"[STITCH] dev={key[0]} | ch={key[1]} | interval_id={prev.get('interval_id')} | "
            f"new_end={float(end_t):.3f} | duration={float(prev['duration_sec']):.3f}s | reason={end_reason}"
        )
        return prev, int(next_interval_id)

    interval_ev = {
        "event": "interval_final",
        "interval_id": int(next_interval_id),
        "device": str(key[0]),
        "channel": str(key[1]),
        "start_t": float(start_t),
        "start_update_idx": int(start_update_idx),
        "end_t": float(end_t),
        "end_update_idx": int(end_update_idx),
        "duration_sec": float(duration_sec),
        "status": str(status),
        "end_reason": str(end_reason),
        "stitch_count": 0,
    }
    events.append(interval_ev)
    last_interval_by_key[key] = interval_ev
    return interval_ev, int(next_interval_id) + 1


def _extract_window_slice(
    st: ChannelStreamState,
    *,
    max_keep_sec: float,
    window_sec: float,
) -> WindowSlice | None:
    """Prepare current short window arrays/timing from one channel ring state."""

    if not bool(st.ring_time_sorted):
        st.ring = deque(sorted(st.ring, key=lambda x: float(x[0])))
        st.ring_time_sorted = True
    if not st.ring:
        return None

    t_now = float(st.ring[-1][0])
    _trim_ring_by_time(st.ring, keep_from_t=(t_now - float(max_keep_sec)))

    tw0 = np.fromiter((float(x[0]) for x in st.ring), dtype=float)
    vw0 = np.fromiter((float(x[1]) for x in st.ring), dtype=float)
    m = np.isfinite(tw0) & np.isfinite(vw0) & (tw0 >= (t_now - float(window_sec))) & (tw0 <= t_now)
    if np.count_nonzero(m) < 5:
        return None

    tw = tw0[m]
    vw = vw0[m]
    if tw.size < 5:
        return None

    dt = _estimate_dt_from_timestamps(tw)
    if (not np.isfinite(dt)) or (dt <= 0):
        return None

    t0 = float(tw[0])
    t1 = float(tw[-1])
    if (t1 - t0) <= 0:
        return None

    return WindowSlice(
        tw0=tw0,
        vw0=vw0,
        tw=tw,
        vw=vw,
        dt=float(dt),
        t0=float(t0),
        t1=float(t1),
    )


def _extract_features(
    st: ChannelStreamState,
    *,
    max_keep_sec: float,
    window_sec: float,
) -> dict[str, object] | None:
    """Extract base short-window features from ring buffer state."""

    window_slice = _extract_window_slice(
        st,
        max_keep_sec=float(max_keep_sec),
        window_sec=float(window_sec),
    )
    if window_slice is None:
        return None

    score, A_tail, D_tail, reason = score_one_channel_equiv(
        window_slice.tw,
        window_slice.vw,
        float(window_slice.dt),
        float(window_slice.t0),
        float(window_slice.t1),
    )
    return {
        "tw0": window_slice.tw0,
        "vw0": window_slice.vw0,
        "tw": window_slice.tw,
        "vw": window_slice.vw,
        "dt": float(window_slice.dt),
        "t0": float(window_slice.t0),
        "t1": float(window_slice.t1),
        "score": float(score),
        "A_tail": float(A_tail),
        "D_tail": float(D_tail),
        "reason": str(reason),
        "score_reason_ok": bool(reason in {"ok", "excluded_damped"}),
    }


def _update_baseline_and_long_stats(
    st: ChannelStreamState,
    *,
    t1: float,
    s_log: float,
    risk_prev: bool,
    reason: str,
    short_high: bool,
    score: float,
    cut_off_cmp: float,
    baseline_include_quiet_on: bool,
    baseline_quiet_on_max_sec: float,
    long_baseline_max: int,
    long_window_sec: float,
    long_off_recent_window_sec: float,
    long_z_on: float,
    long_z_off: float,
) -> LongStatsSnapshot:
    """Update baseline/long histories and compute long-window ratio stats."""

    st.long_score_hist.append((float(t1), float(s_log)))
    _trim_score_hist_by_time(st.long_score_hist, keep_from_t=(float(t1) - float(long_window_sec)))
    st.long_off_recent_hist.append((float(t1), float(s_log)))
    _trim_score_hist_by_time(st.long_off_recent_hist, keep_from_t=(float(t1) - float(long_off_recent_window_sec)))

    on_age_sec = (
        float(t1 - float(st.active_start_t))
        if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
        else float("inf")
    )
    quiet_on_for_baseline = (
        bool(baseline_include_quiet_on)
        and bool(risk_prev)
        and (str(reason) == "ok")
        and (not bool(short_high))
        and np.isfinite(score)
        and (float(score) < float(cut_off_cmp))
        and (float(on_age_sec) <= float(baseline_quiet_on_max_sec))
    )
    if ((not bool(risk_prev)) and (str(reason) == "ok") and (not bool(short_high))) or quiet_on_for_baseline:
        st.long_baseline_hist.append(float(s_log))
        while len(st.long_baseline_hist) > int(long_baseline_max):
            st.long_baseline_hist.popleft()

    base_med, base_scale = _robust_center_scale(st.long_baseline_hist)
    long_ratio_on, long_zmax, long_n = _long_activity_ratio(
        st.long_score_hist,
        center=base_med,
        scale=base_scale,
        z_thr=float(long_z_on),
    )
    long_ratio_off, _, _ = _long_activity_ratio(
        st.long_score_hist,
        center=base_med,
        scale=base_scale,
        z_thr=float(long_z_off),
    )
    long_ratio_off_recent, _, long_off_n_recent = _long_activity_ratio(
        st.long_off_recent_hist,
        center=base_med,
        scale=base_scale,
        z_thr=float(long_z_off),
    )
    return LongStatsSnapshot(
        base_med=float(base_med),
        base_scale=float(base_scale),
        long_ratio_on=float(long_ratio_on),
        long_zmax=float(long_zmax),
        long_n=int(long_n),
        long_ratio_off=float(long_ratio_off),
        long_ratio_off_recent=float(long_ratio_off_recent),
        long_off_n_recent=int(long_off_n_recent),
    )


def _compute_quality_and_support(
    st: ChannelStreamState,
    *,
    tw: np.ndarray,
    vw: np.ndarray,
    t1: float,
    score: float,
    score_reason_ok: bool,
    cut_off_cmp: float,
    cut_on_eff: float,
    cut_off_eff: float,
    s_log: float,
    reason: str,
    risk_prev: bool,
    short_high: bool,
    short_trigger: bool,
    long_ratio_on: float,
    long_n: int,
    base_med: float,
    base_scale: float,
    baseline_n: int,
    long_min_points: int,
    long_min_baseline: int,
    long_on_ratio: float,
    warmup_long_enabled: bool,
    warmup_on_min_points: int,
    warmup_min_baseline: int,
    warmup_on_z: float,
    warmup_on_ratio: float,
    long_zmax: float,
    confidence_use_calibration: bool,
    confidence_cal_min_points: int,
    confidence_cal_hist_max: int,
    confidence_cal_off_only: bool,
    acf_min_points: int,
    acf_min_period_sec: float,
    acf_max_period_sec: float,
    freq_band_low_hz: float,
    freq_band_high_hz: float,
    freq_linear_detrend: bool,
    freq_ar1_whiten: bool,
    confidence_w_acf: float,
    confidence_w_spec: float,
    confidence_w_env: float,
    confidence_w_fft: float,
    update_sec: float,
    periodicity_quality_cache_enabled: bool,
    periodicity_quality_cache_refresh_sec: float,
    rms_decay_local: float,
    rms_decay_local_r2: float,
    rms_decay_local_n: int,
    rms_decay_local_on_ok: bool,
    rms_decay_local_off_hint: bool,
    rms_decay_event: float,
    rms_decay_event_r2: float,
    rms_decay_event_n: int,
    rms_decay_event_win_sec: float,
    evidence_alpha: float,
    evidence_clip: float,
    excluded_damped_evidence_penalty: float,
    excluded_damped_hard_penalty_streak: int,
    on_require_accel_for_candidate: bool,
    on_accel_score_log_min: float,
    on_accel_evidence_min: float,
    confidence_on_min: float,
    cal_on_soft_mode: bool,
    confidence_raw_on_min_when_cal: float,
    cal_on_noise_scale_low: float,
    cal_on_noise_scale_high: float,
    cal_on_conf_adapt_gain: float,
    cal_on_conf_floor: float,
    cal_on_conf_ceil: float,
    cal_on_support_score_w: float,
    cal_on_support_score_acf_bonus: float,
    cal_on_support_long_w: float,
    cal_on_support_conf_w: float,
    cal_on_support_accel_w: float,
    cal_on_support_enter_min: float,
    cal_on_confirm_votes_window: int,
    cal_on_confirm_votes_required: int,
    cal_on_support_ema_alpha: float,
    phase_now: str,
) -> dict[str, object]:
    """Compute periodicity quality, confidence/support, and evidence derivatives."""

    warmup_conf_gate_possible = bool(
        bool(warmup_long_enabled)
        and (str(phase_now) in {PHASE_OFF, PHASE_ON_CANDIDATE})
        and (int(long_n) >= int(warmup_on_min_points))
        and (int(baseline_n) >= int(warmup_min_baseline))
        and np.isfinite(long_zmax)
        and (float(long_zmax) >= float(warmup_on_z))
        and np.isfinite(long_ratio_on)
        and (float(long_ratio_on) >= float(warmup_on_ratio))
    )
    quality_need_context = bool(
        bool(confidence_use_calibration)
        or bool(risk_prev)
        or bool(short_high)
        or bool(short_trigger)
        or bool(warmup_conf_gate_possible)
    )
    quality_low_floor = float(max(float(cut_off_cmp) * 0.25, 1e-18))
    quality_weak_tick = bool(
        (not bool(score_reason_ok))
        or (not np.isfinite(score))
        or (float(score) < float(quality_low_floor))
    )
    periodicity_idle_refresh_sec = max(float(update_sec), float(periodicity_quality_cache_refresh_sec))
    can_skip_quality_heavy = bool(
        (not bool(quality_need_context))
        and bool(quality_weak_tick)
        and bool(st.last_quality)
    )
    can_reuse_quality = bool(
        bool(periodicity_quality_cache_enabled)
        and (not bool(confidence_use_calibration))
        and (not bool(risk_prev))
        and (str(phase_now) == PHASE_OFF)
        and (not bool(short_high))
        and (not bool(short_trigger))
        and (not bool(warmup_conf_gate_possible))
        and bool(st.last_quality)
        and np.isfinite(st.last_quality_t_end)
        and ((float(t1) - float(st.last_quality_t_end)) < float(periodicity_idle_refresh_sec))
    )
    if can_skip_quality_heavy:
        quality = dict(st.last_quality)
    elif can_reuse_quality:
        quality = dict(st.last_quality)
    else:
        quality = _compute_periodicity_quality(
            tw,
            vw,
            acf_min_points=int(acf_min_points),
            acf_min_period_sec=float(acf_min_period_sec),
            acf_max_period_sec=float(acf_max_period_sec),
            band_low_hz=float(freq_band_low_hz),
            band_high_hz=float(freq_band_high_hz),
            linear_detrend=bool(freq_linear_detrend),
            ar1_whiten=bool(freq_ar1_whiten),
            w_acf=float(confidence_w_acf),
            w_spec=float(confidence_w_spec),
            w_env=float(confidence_w_env),
            w_fft=float(confidence_w_fft),
        )
        st.last_quality_t_end = float(t1)
    confidence_raw = float(quality.get("confidence_raw", quality.get("confidence", np.nan)))
    confidence_cal = _calibrate_confidence_quantile(
        confidence_raw,
        conf_hist=st.conf_raw_hist,
        min_points=int(confidence_cal_min_points),
    )
    confidence_used = (
        float(confidence_cal)
        if (bool(confidence_use_calibration) and np.isfinite(confidence_cal))
        else (float(confidence_raw) if np.isfinite(confidence_raw) else float("nan"))
    )
    quality["confidence_raw"] = float(confidence_raw) if np.isfinite(confidence_raw) else float("nan")
    quality["confidence_cal"] = float(confidence_cal) if np.isfinite(confidence_cal) else float("nan")
    quality["confidence_used"] = float(confidence_used) if np.isfinite(confidence_used) else float("nan")
    quality["confidence"] = float(confidence_used) if np.isfinite(confidence_used) else float("nan")
    _set_quality_rms_metrics(
        quality,
        rms_decay_local=float(rms_decay_local),
        rms_decay_local_r2=float(rms_decay_local_r2),
        rms_decay_local_n=int(rms_decay_local_n),
        rms_decay_local_on_ok=bool(rms_decay_local_on_ok),
        rms_decay_local_off_hint=bool(rms_decay_local_off_hint),
        rms_decay_event=float(rms_decay_event),
        rms_decay_event_r2=float(rms_decay_event_r2),
        rms_decay_event_n=int(rms_decay_event_n),
        rms_decay_event_win_sec=float(rms_decay_event_win_sec),
    )
    rms_metrics = _extract_quality_rms_metrics(quality)
    if np.isfinite(confidence_raw) and ((not bool(confidence_cal_off_only)) or (not bool(risk_prev))):
        st.conf_raw_hist.append(float(confidence_raw))
        while len(st.conf_raw_hist) > int(confidence_cal_hist_max):
            st.conf_raw_hist.popleft()
    st.last_quality = quality

    acf_peak = float(quality.get("acf_peak", np.nan))
    acf_period_sec = float(quality.get("acf_period_sec", np.nan))
    acf_lag_steps = int(quality.get("acf_lag_steps", -1))
    acf_n = int(quality.get("acf_n", 0))
    confidence = float(quality.get("confidence", np.nan))
    confidence_raw = float(quality.get("confidence_raw", np.nan))
    confidence_cal = float(quality.get("confidence_cal", np.nan))
    c_acf = float(quality.get("c_acf", np.nan))
    c_spec = float(quality.get("c_spec", np.nan))
    c_env = float(quality.get("c_env", np.nan))
    c_fft = float(quality.get("c_fft", np.nan))
    c_freq_agree = float(quality.get("c_freq_agree", np.nan))
    f_welch = float(quality.get("f_welch", np.nan))
    f_zc = float(quality.get("f_zc", np.nan))
    f_fft = float(quality.get("f_fft", np.nan))

    long_ready = (
        (int(long_n) >= int(long_min_points))
        and (int(baseline_n) >= int(long_min_baseline))
        and np.isfinite(base_med)
        and np.isfinite(base_scale)
    )
    if long_ready:
        st.long_ready_streak += 1
    else:
        st.long_ready_streak = 0
    warmup_mode = bool(warmup_long_enabled) and (not bool(long_ready))

    baseline_cut = float(cut_off_eff if risk_prev else cut_on_eff)
    score_safe = float(score) if (np.isfinite(score) and float(score) > 0.0) else (baseline_cut * 1e-12)
    ratio = max(score_safe / baseline_cut, 1e-12)
    e_t = float(np.clip(np.log10(ratio), -float(evidence_clip), float(evidence_clip)))
    if str(reason) == "excluded_damped":
        damped_penalty = float(excluded_damped_evidence_penalty)
        if int(st.damped_streak) >= int(excluded_damped_hard_penalty_streak):
            damped_penalty = float(evidence_clip)
        damped_penalty = float(np.clip(damped_penalty, 0.0, float(evidence_clip)))
        e_t = -float(damped_penalty)
    elif str(reason) != "ok":
        e_t = min(e_t, 0.0)

    delta_s_log = (
        float(s_log - float(st.prev_score_log))
        if np.isfinite(st.prev_score_log)
        else float("nan")
    )
    delta_e = (
        float(e_t - float(st.prev_e_t))
        if np.isfinite(st.prev_e_t)
        else float("nan")
    )
    accel_ok = (
        (not bool(on_require_accel_for_candidate))
        or (np.isfinite(delta_s_log) and (float(delta_s_log) >= float(on_accel_score_log_min)))
        or (np.isfinite(delta_e) and (float(delta_e) >= float(on_accel_evidence_min)))
    )
    st.evidence = float(evidence_alpha) * float(st.evidence) + (1.0 - float(evidence_alpha)) * e_t
    st.prev_score_log = float(s_log)
    st.prev_e_t = float(e_t)

    conf_now = float(confidence) if np.isfinite(confidence) else 0.0
    raw_now = float(confidence_raw) if np.isfinite(confidence_raw) else float("nan")
    use_cal_gate = bool(confidence_use_calibration) and np.isfinite(confidence_cal)
    cal_on_active = bool(cal_on_soft_mode) and bool(use_cal_gate)

    cal_noise_norm = _normalize01(float(base_scale), float(cal_on_noise_scale_low), float(cal_on_noise_scale_high))
    if not np.isfinite(cal_noise_norm):
        cal_noise_norm = 0.0
    cal_on_conf_thr = float(np.clip(
        float(confidence_on_min) + float(cal_on_conf_adapt_gain) * float(cal_noise_norm),
        float(cal_on_conf_floor),
        float(cal_on_conf_ceil),
    ))

    score_support_base = _clip01((float(e_t) + float(evidence_clip)) / (2.0 * float(evidence_clip)))
    acf_support_bonus = (
        _clip01(float(c_acf)) * max(0.0, float(cal_on_support_score_acf_bonus))
        if np.isfinite(c_acf)
        else 0.0
    )
    score_support = _clip01(float(score_support_base) + float(acf_support_bonus))
    long_support = (
        _clip01(float(long_ratio_on) / max(float(long_on_ratio), 1e-9))
        if bool(long_ready)
        else (
            _clip01(float(long_ratio_on) / max(float(warmup_on_ratio), 1e-9))
            if bool(warmup_mode) else 0.0
        )
    )
    conf_support = _clip01(float(conf_now))
    accel_support_s = (
        _clip01(float(delta_s_log) / max(abs(float(on_accel_score_log_min)), 1e-6))
        if np.isfinite(delta_s_log) else float("nan")
    )
    accel_support_e = (
        _clip01(float(delta_e) / max(abs(float(on_accel_evidence_min)), 1e-6))
        if np.isfinite(delta_e) else float("nan")
    )
    accel_parts = [float(x) for x in (accel_support_s, accel_support_e) if np.isfinite(x)]
    accel_support = (max(accel_parts) if accel_parts else float("nan"))
    if not np.isfinite(accel_support):
        accel_support = (1.0 if bool(accel_ok) else 0.0)
    cal_support_weights = {
        "score": max(0.0, float(cal_on_support_score_w)),
        "long": max(0.0, float(cal_on_support_long_w)),
        "conf": max(0.0, float(cal_on_support_conf_w)),
        "accel": max(0.0, float(cal_on_support_accel_w)),
    }
    cal_support_parts = {
        "score": score_support,
        "long": long_support,
        "conf": conf_support,
        "accel": accel_support,
    }
    cal_support_den = 0.0
    cal_support_num = 0.0
    for k, w in cal_support_weights.items():
        v = cal_support_parts[k]
        if np.isfinite(v) and (w > 0.0):
            cal_support_num += float(w) * float(v)
            cal_support_den += float(w)
    on_support = (float(cal_support_num / cal_support_den) if cal_support_den > 0.0 else float("nan"))
    if np.isfinite(on_support):
        st.on_support_ema = (
            float(cal_on_support_ema_alpha) * float(st.on_support_ema)
            + (1.0 - float(cal_on_support_ema_alpha)) * float(on_support)
        )
    elif not np.isfinite(st.on_support_ema):
        st.on_support_ema = 0.0
    on_support_ema = float(st.on_support_ema)
    if bool(cal_on_active) and (str(phase_now) in {PHASE_OFF, PHASE_ON_CANDIDATE}):
        soft_vote = bool(
            np.isfinite(on_support)
            and (float(on_support) >= float(cal_on_support_enter_min))
            and (float(conf_now) >= float(cal_on_conf_thr))
        )
        st.on_soft_votes.append(1 if soft_vote else 0)
        while len(st.on_soft_votes) > int(cal_on_confirm_votes_window):
            st.on_soft_votes.popleft()
    else:
        st.on_soft_votes.clear()
    on_soft_vote_sum = int(sum(st.on_soft_votes))
    on_soft_confirmed = bool(
        (len(st.on_soft_votes) >= int(cal_on_confirm_votes_window))
        and (int(on_soft_vote_sum) >= int(cal_on_confirm_votes_required))
    )

    return {
        "acf_peak": float(acf_peak),
        "acf_period_sec": float(acf_period_sec),
        "acf_lag_steps": int(acf_lag_steps),
        "acf_n": int(acf_n),
        "confidence": float(confidence),
        "confidence_raw": float(confidence_raw),
        "confidence_cal": float(confidence_cal),
        "c_acf": float(c_acf),
        "c_spec": float(c_spec),
        "c_env": float(c_env),
        "c_fft": float(c_fft),
        "c_freq_agree": float(c_freq_agree),
        "f_welch": float(f_welch),
        "f_zc": float(f_zc),
        "f_fft": float(f_fft),
        "rms_decay": float(rms_metrics["rms_decay"]),
        "rms_decay_r2": float(rms_metrics["rms_decay_r2"]),
        "rms_decay_n": int(rms_metrics["rms_decay_n"]),
        "rms_decay_on_ok": int(rms_metrics["rms_decay_on_ok"]),
        "rms_decay_off_hint": int(rms_metrics["rms_decay_off_hint"]),
        "rms_decay_event": float(rms_metrics["rms_decay_event"]),
        "rms_decay_event_r2": float(rms_metrics["rms_decay_event_r2"]),
        "rms_decay_event_n": int(rms_metrics["rms_decay_event_n"]),
        "rms_decay_event_win_sec": float(rms_metrics["rms_decay_event_win_sec"]),
        "long_ready": bool(long_ready),
        "warmup_mode": bool(warmup_mode),
        "e_t": float(e_t),
        "delta_s_log": float(delta_s_log),
        "delta_e": float(delta_e),
        "accel_ok": bool(accel_ok),
        "conf_now": float(conf_now),
        "raw_now": float(raw_now),
        "use_cal_gate": bool(use_cal_gate),
        "cal_on_active": bool(cal_on_active),
        "cal_on_conf_thr": float(cal_on_conf_thr),
        "on_support": float(on_support),
        "on_support_ema": float(on_support_ema),
        "on_soft_vote_sum": int(on_soft_vote_sum),
        "on_soft_confirmed": bool(on_soft_confirmed),
    }


def _compute_off_path(
    st: ChannelStreamState,
    *,
    phase_now: str,
    short_release: bool,
    conf_now: float,
    raw_now: float,
    use_cal_gate: bool,
    confidence_off_max: float,
    off_periodicity_collapse_conf_raw_max: float,
    confidence_raw_off_max_when_cal: float,
    off_periodicity_collapse_streak_required: int,
    long_ready: bool,
    long_off_n_recent: int,
    long_off_recent_min_points: int,
    long_ratio_off_recent: float,
    long_off_ratio: float,
    rms_decay_local_off_hint: bool,
    long_off_votes_window: int,
    long_off_votes_required: int,
    damped_force_off_streak: int,
    theta_off: float,
    force_off_long_on_ratio: float | None,
    force_off_require_long_not_on: bool,
    long_ratio_on: float,
    long_on_ratio: float,
) -> OffPathSnapshot:
    """Compute OFF/collapse votes and forced-OFF guard outputs."""

    off_aux_low_conf = bool(
        (np.isfinite(conf_now) and (float(conf_now) <= float(confidence_off_max)))
        or (np.isfinite(raw_now) and (float(raw_now) <= float(off_periodicity_collapse_conf_raw_max)))
        or (
            bool(use_cal_gate)
            and np.isfinite(raw_now)
            and (float(raw_now) <= float(confidence_raw_off_max_when_cal))
        )
    )
    collapse_tick = bool(
        bool(short_release)
        and bool(off_aux_low_conf)
    )
    if collapse_tick:
        st.periodicity_collapse_streak += 1
    else:
        st.periodicity_collapse_streak = 0
    collapse_ok = bool(st.periodicity_collapse_streak >= int(off_periodicity_collapse_streak_required))

    off_vote_core = bool(
        bool(short_release)
        and bool(long_ready)
        and (int(long_off_n_recent) >= int(long_off_recent_min_points))
        and (float(long_ratio_off_recent) <= float(long_off_ratio))
        and (bool(collapse_ok) or bool(rms_decay_local_off_hint))
    )
    if str(phase_now) in {PHASE_ON_CONFIRMED, PHASE_OFF_CANDIDATE}:
        st.long_off_votes.append(1 if off_vote_core else 0)
        while len(st.long_off_votes) > int(long_off_votes_window):
            st.long_off_votes.popleft()
    else:
        st.long_off_votes.clear()
        st.off_candidate_start_t = None
        st.off_candidate_start_update_idx = None
    long_off_vote_sum = int(sum(st.long_off_votes))
    long_off_confirmed = bool(
        (len(st.long_off_votes) >= int(long_off_votes_window))
        and (int(long_off_vote_sum) >= int(long_off_votes_required))
    )

    force_off_guard_ratio = float(long_on_ratio if force_off_long_on_ratio is None else force_off_long_on_ratio)
    force_off_long_ok = ((not bool(long_ready)) or (float(long_ratio_on) < float(force_off_guard_ratio)))
    force_off_now = bool(
        (int(st.damped_streak) >= int(damped_force_off_streak))
        and bool(short_release)
        and (float(st.evidence) <= float(theta_off))
        and ((not bool(force_off_require_long_not_on)) or force_off_long_ok)
    )
    return OffPathSnapshot(
        collapse_ok=bool(collapse_ok),
        off_vote_core=bool(off_vote_core),
        long_off_confirmed=bool(long_off_confirmed),
        force_off_now=bool(force_off_now),
    )


def _evaluate_gates(
    *,
    conf_now: float,
    raw_now: float,
    use_cal_gate: bool,
    cal_on_active: bool,
    on_support_ema: float,
    cal_on_conf_thr: float,
    confidence_on_min: float,
    confidence_dual_gate_when_cal: bool,
    confidence_raw_on_min_when_cal: float,
    cal_on_support_enter_min: float,
    long_ready: bool,
    long_ratio_on: float,
    long_on_ratio: float,
    warmup_long_enabled: bool,
    long_ready_streak: int,
    warmup_handoff_grace_ticks: int,
    warmup_mode: bool,
    warmup_on_confirmed: bool,
    short_trigger: bool,
    baseline_n: int,
    short_dynamic_min_baseline: int,
    accel_ok: bool,
    rms_decay_local_on_ok: bool,
    rms_decay_local_off_hint: bool,
    off_vote_core: bool,
    long_off_confirmed: bool,
    force_off_now: bool,
) -> tuple[GateFlags, int]:
    """Evaluate ON/OFF entry and transition gates for current tick."""

    on_conf_ok = bool(conf_now >= float(confidence_on_min))
    if bool(use_cal_gate) and bool(confidence_dual_gate_when_cal):
        on_conf_ok = bool(
            on_conf_ok
            and np.isfinite(raw_now)
            and (float(raw_now) >= float(confidence_raw_on_min_when_cal))
        )
    if bool(cal_on_active):
        raw_guard_ok = (
            (not bool(confidence_dual_gate_when_cal))
            or (np.isfinite(raw_now) and (float(raw_now) >= float(confidence_raw_on_min_when_cal)))
        )
        on_conf_ok = bool(
            np.isfinite(conf_now)
            and (float(conf_now) >= float(cal_on_conf_thr))
            and np.isfinite(on_support_ema)
            and (float(on_support_ema) >= float(cal_on_support_enter_min))
            and raw_guard_ok
        )

    long_on_core = bool(bool(long_ready) and (float(long_ratio_on) >= float(long_on_ratio)))
    warmup_handoff_active = bool(
        bool(warmup_long_enabled)
        and bool(long_ready)
        and (int(long_ready_streak) <= int(warmup_handoff_grace_ticks))
    )
    if bool(warmup_mode):
        on_long_gate_ok = bool(warmup_on_confirmed)
    elif warmup_handoff_active:
        on_long_gate_ok = bool(long_on_core and warmup_on_confirmed)
    else:
        on_long_gate_ok = bool(long_on_core)

    short_no_long_ready = bool(
        bool(short_trigger)
        and (int(baseline_n) >= int(short_dynamic_min_baseline))
    )
    on_entry_ready = bool(
        bool(accel_ok)
        and bool(on_conf_ok)
        and (bool(on_long_gate_ok) or bool(short_no_long_ready))
        and bool(rms_decay_local_on_ok)
    )
    gate_flags = GateFlags(
        short_trigger=bool(short_trigger),
        short_no_long_ready=bool(short_no_long_ready),
        accel_ok=bool(accel_ok),
        on_conf_ok=bool(on_conf_ok),
        warmup_on_confirmed=bool(warmup_on_confirmed),
        long_ready=bool(long_ready),
        on_long_gate_ok=bool(on_long_gate_ok),
        off_vote_core=bool(off_vote_core),
        long_off_confirmed=bool(long_off_confirmed),
        force_off_now=bool(force_off_now),
        rms_decay_on_ok=bool(rms_decay_local_on_ok),
        rms_decay_off_hint=bool(rms_decay_local_off_hint),
        on_entry_ready=bool(on_entry_ready),
    )
    on_entry_vote_sum = (
        int(gate_flags.short_trigger)
        + int(gate_flags.on_long_gate_ok)
        + int(gate_flags.on_conf_ok)
        + int(gate_flags.rms_decay_on_ok)
    )
    return gate_flags, int(on_entry_vote_sum)


def _step_fsm(
    st: ChannelStreamState,
    *,
    phase_now: str,
    t1: float,
    upd_idx: int,
    reason: str,
    score_reason_ok: bool,
    score: float,
    cut_off_cmp: float,
    cal_on_active: bool,
    on_support_ema: float,
    cal_on_support_hold_min: float,
    on_confirm_min_sec: float,
    on_soft_confirmed: bool,
    cal_on_support_confirm_min: float,
    off_hold_down_sec: float,
    short_high: bool,
    collapse_ok: bool,
    on_consecutive_required: int,
    off_confirm_min_sec: float,
    gate_flags: GateFlags,
) -> tuple[str, str]:
    """Advance phase machine using evaluated gates and current feature values."""

    transition_reason = str(reason)
    if phase_now == PHASE_OFF:
        if gate_flags.on_entry_ready:
            phase_now = PHASE_ON_CANDIDATE
            transition_reason = "off_to_on_candidate"
            st.active_start_t = float(t1)
            st.active_start_update_idx = int(upd_idx)
            st.on_event_emitted = False
    elif phase_now == PHASE_ON_CANDIDATE:
        cand_start_t = float(st.active_start_t) if st.active_start_t is not None else float(t1)
        if st.active_start_t is None:
            st.active_start_t = float(t1)
            st.active_start_update_idx = int(upd_idx)
            cand_start_t = float(t1)
        cand_age = float(t1 - cand_start_t)
        keep_candidate = bool(
            gate_flags.short_trigger
            or (score_reason_ok and np.isfinite(score) and (float(score) >= float(cut_off_cmp)))
        )
        if cal_on_active:
            keep_candidate = bool(
                keep_candidate
                or (
                    np.isfinite(on_support_ema)
                    and (float(on_support_ema) >= float(cal_on_support_hold_min))
                )
            )
        if not keep_candidate:
            phase_now = PHASE_OFF
            transition_reason = "on_candidate_revert_to_off"
            st.active_start_t = None
            st.active_start_update_idx = None
            st.on_short_votes.clear()
            st.on_soft_votes.clear()
            st.warmup_on_votes.clear()
            st.warmup_on_start_t = None
            st.warmup_on_start_update_idx = None
            st.on_support_ema = 0.0
            st.on_candidate_streak = 0
            st.on_event_emitted = False
        elif float(cand_age) >= float(on_confirm_min_sec):
            if cal_on_active:
                stage2_ok = bool(
                    on_soft_confirmed
                    and np.isfinite(on_support_ema)
                    and (float(on_support_ema) >= float(cal_on_support_confirm_min))
                )
                if stage2_ok:
                    phase_now = PHASE_ON_CONFIRMED
                    transition_reason = "on_candidate_to_on_confirmed_soft"
                    st.on_soft_votes.clear()
                else:
                    transition_reason = "on_candidate_wait_soft_confirm"
            else:
                phase_now = PHASE_ON_CONFIRMED
                transition_reason = "on_candidate_to_on_confirmed"
    elif phase_now == PHASE_ON_CONFIRMED:
        on_age = (
            float(t1 - float(st.active_start_t))
            if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
            else float("inf")
        )
        st.on_soft_votes.clear()
        if gate_flags.force_off_now:
            phase_now = PHASE_OFF_CONFIRMED
            transition_reason = "forced_off_damped_with_low_evidence"
        elif (float(on_age) >= float(off_hold_down_sec)) and gate_flags.off_vote_core:
            phase_now = PHASE_OFF_CANDIDATE
            transition_reason = "on_confirmed_to_off_candidate"
            if st.off_candidate_start_t is None:
                st.off_candidate_start_t = float(t1)
                st.off_candidate_start_update_idx = int(upd_idx)
    elif phase_now == PHASE_OFF_CANDIDATE:
        if gate_flags.force_off_now:
            phase_now = PHASE_OFF_CONFIRMED
            transition_reason = "forced_off_damped_with_low_evidence"
        else:
            if st.off_candidate_start_t is None:
                st.off_candidate_start_t = float(t1)
                st.off_candidate_start_update_idx = int(upd_idx)
            off_cand_age = float(t1 - float(st.off_candidate_start_t))
            recover_to_on = bool(
                short_high
                and (int(sum(st.on_short_votes)) >= int(on_consecutive_required))
                and (not bool(collapse_ok))
            )
            if recover_to_on:
                phase_now = PHASE_ON_CONFIRMED
                transition_reason = "off_candidate_revert_to_on_confirmed"
                st.off_candidate_start_t = None
                st.off_candidate_start_update_idx = None
                st.long_off_votes.clear()
            elif gate_flags.long_off_confirmed and (float(off_cand_age) >= float(off_confirm_min_sec)):
                phase_now = PHASE_OFF_CONFIRMED
                transition_reason = "off_candidate_to_off_confirmed"
    return str(phase_now), str(transition_reason)


def _emit_events(
    *,
    events: list[dict],
    on_event: Callable[[dict], None] | None,
    st: ChannelStreamState,
    key: tuple[str, str],
    upd_idx: int,
    t1: float,
    risk_now: bool,
    transition_reason: str,
    min_interval_sec_for_alert: float,
    raw_risk_interval_count: int,
    suppressed_interval_count: int,
    last_interval_by_key: dict[tuple[str, str], dict],
    next_interval_id: int,
    stitch_gap_sec: float,
    metrics_kwargs: dict[str, object],
) -> tuple[int, int, int]:
    """Emit risk transition events and interval records for one tick."""

    if risk_now and (not st.last_risk_on):
        if st.active_start_t is None:
            st.active_start_t = float(t1)
            st.active_start_update_idx = int(upd_idx)
        st.on_event_emitted = False

    if risk_now and (st.active_start_t is not None) and (not st.on_event_emitted):
        on_duration = float(t1 - float(st.active_start_t))
        if on_duration >= float(min_interval_sec_for_alert):
            ev = {
                "event": "risk_on",
                "update_idx": int(upd_idx),
                "device": str(key[0]),
                "channel": str(key[1]),
                "t_end": float(t1),
                "start_t": float(st.active_start_t),
                "start_update_idx": int(st.active_start_update_idx) if st.active_start_update_idx is not None else int(upd_idx),
                **_build_risk_event_metrics(**metrics_kwargs),
            }
            events.append(ev)
            print(
                f"[ALERT] RISK_ON | upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                f"start={float(st.active_start_t):.3f} | t_end={t1:.3f} | score={float(metrics_kwargs.get('score', np.nan)):.3e} | "
                f"S={float(metrics_kwargs.get('evidence', st.evidence)):+.3f} | duration={on_duration:.3f}s"
            )
            if on_event is not None:
                on_event(ev)
            st.on_event_emitted = True
        return int(raw_risk_interval_count), int(suppressed_interval_count), int(next_interval_id)

    if (not risk_now) and st.last_risk_on:
        start_t = float(st.active_start_t) if st.active_start_t is not None else float("nan")
        start_idx = int(st.active_start_update_idx) if st.active_start_update_idx is not None else -1
        duration_sec = float(t1 - start_t) if np.isfinite(start_t) else float("nan")
        raw_risk_interval_count += 1
        if np.isfinite(duration_sec) and (duration_sec < float(min_interval_sec_for_alert)):
            suppressed_interval_count += 1
            print(
                f"[SUPPRESS] short_interval | dev={key[0]} | ch={key[1]} | "
                f"start={start_t:.3f}(upd={start_idx:03d}) | end={t1:.3f}(upd={upd_idx:03d}) | "
                f"duration={duration_sec:.3f}s < min={float(min_interval_sec_for_alert):.3f}s"
            )
        else:
            if (not st.on_event_emitted) and np.isfinite(start_t):
                delayed_on_metrics = dict(metrics_kwargs)
                delayed_on_metrics["transition_reason"] = "delayed_emit_before_off"
                on_ev = {
                    "event": "risk_on",
                    "update_idx": int(start_idx if start_idx >= 0 else upd_idx),
                    "device": str(key[0]),
                    "channel": str(key[1]),
                    "t_end": float(start_t),
                    "start_t": start_t,
                    "start_update_idx": int(start_idx),
                    **_build_risk_event_metrics(**delayed_on_metrics),
                }
                events.append(on_ev)
                if on_event is not None:
                    on_event(on_ev)

            off_metrics = dict(metrics_kwargs)
            off_metrics["transition_reason"] = str(transition_reason)
            ev = {
                "event": "risk_off",
                "update_idx": int(upd_idx),
                "device": str(key[0]),
                "channel": str(key[1]),
                "t_end": float(t1),
                "start_t": start_t,
                "start_update_idx": start_idx,
                "end_t": float(t1),
                "end_update_idx": int(upd_idx),
                "duration_sec": duration_sec,
                **_build_risk_event_metrics(**off_metrics),
                "end_reason": str(transition_reason),
            }
            events.append(ev)
            print(
                f"[ALERT] RISK_OFF | upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                f"t_end={t1:.3f} | score={float(metrics_kwargs.get('score', np.nan)):.3e} | "
                f"S={float(metrics_kwargs.get('evidence', st.evidence)):+.3f} | reason={metrics_kwargs.get('reason', '')}"
            )
            if np.isfinite(start_t):
                print(
                    f"[INTERVAL] dev={key[0]} | ch={key[1]} | start={start_t:.3f}(upd={start_idx:03d}) | "
                    f"end={t1:.3f}(upd={upd_idx:03d}) | duration={duration_sec:.3f}s"
                )
                _, next_interval_id = _emit_or_stitch_interval(
                    key=key,
                    start_t=float(start_t),
                    start_update_idx=int(start_idx),
                    end_t=float(t1),
                    end_update_idx=int(upd_idx),
                    end_reason=str(transition_reason),
                    status="closed",
                    events=events,
                    last_interval_by_key=last_interval_by_key,
                    next_interval_id=int(next_interval_id),
                    stitch_gap_sec=float(stitch_gap_sec),
                )
            if on_event is not None:
                on_event(ev)
        st.active_start_t = None
        st.active_start_update_idx = None
        st.on_candidate_streak = 0
        st.on_short_votes.clear()
        st.on_soft_votes.clear()
        st.warmup_on_votes.clear()
        st.warmup_on_start_t = None
        st.warmup_on_start_update_idx = None
        st.on_support_ema = 0.0
        st.long_off_votes.clear()
        st.off_candidate_start_t = None
        st.off_candidate_start_update_idx = None
        st.on_event_emitted = False
        st.phase = PHASE_OFF

    return int(raw_risk_interval_count), int(suppressed_interval_count), int(next_interval_id)


def _validate_detector_runtime_args(
    *,
    window_sec: float,
    long_window_sec: float,
    cut_off: float,
    cut_on: float,
    short_dynamic_min_baseline: int,
    short_dynamic_on_z: float,
    short_dynamic_off_z: float,
    short_dynamic_on_max_mult: float,
    short_dynamic_off_max_mult: float,
    short_damped_cut_relax: float,
    rms_decay_window_sec: float,
    rms_decay_rms_win_sec: float,
    rms_decay_step_sec: float,
    rms_decay_min_windows: int,
    rms_decay_on_max: float,
    rms_decay_off_min: float,
    rms_decay_cache_refresh_sec: float,
    rms_decay_event_max_window_sec: float,
    rms_decay_event_min_window_sec: float,
    rms_decay_event_rms_win_sec: float,
    rms_decay_event_step_sec: float,
    rms_decay_event_min_windows: int,
    evidence_alpha: float,
    evidence_clip: float,
    theta_on: float,
    theta_off: float,
    on_consecutive_required: int,
    on_short_votes_window: int,
    on_confirm_min_sec: float,
    on_accel_score_log_min: float,
    on_accel_evidence_min: float,
    damped_force_off_streak: int,
    off_hold_down_sec: float,
    off_confirm_min_sec: float,
    off_periodicity_collapse_conf_raw_max: float,
    off_periodicity_collapse_streak_required: int,
    excluded_damped_evidence_penalty: float,
    excluded_damped_hard_penalty_streak: int,
    long_min_points: int,
    long_min_baseline: int,
    long_baseline_max: int,
    long_off_recent_window_sec: float,
    long_off_votes_required: int,
    long_off_votes_window: int,
    long_off_recent_min_points: int,
    acf_min_points: int,
    acf_min_period_sec: float,
    acf_max_period_sec: float,
    confidence_on_min: float,
    confidence_off_max: float,
    confidence_w_acf: float,
    confidence_w_spec: float,
    confidence_w_env: float,
    confidence_w_fft: float,
    confidence_cal_min_points: int,
    confidence_cal_hist_max: int,
    confidence_raw_on_min_when_cal: float,
    confidence_raw_off_max_when_cal: float,
    cal_on_noise_scale_low: float,
    cal_on_noise_scale_high: float,
    cal_on_conf_adapt_gain: float,
    cal_on_conf_floor: float,
    cal_on_conf_ceil: float,
    cal_on_support_score_w: float,
    cal_on_support_score_acf_bonus: float,
    cal_on_support_long_w: float,
    cal_on_support_conf_w: float,
    cal_on_support_accel_w: float,
    cal_on_support_enter_min: float,
    cal_on_support_confirm_min: float,
    cal_on_support_hold_min: float,
    cal_on_support_ema_alpha: float,
    cal_on_confirm_votes_required: int,
    cal_on_confirm_votes_window: int,
    freq_band_low_hz: float,
    freq_band_high_hz: float,
    periodicity_quality_cache_refresh_sec: float,
    long_z_on: float,
    long_z_off: float,
    long_on_ratio: float,
    long_off_ratio: float,
    warmup_on_min_points: int,
    warmup_min_baseline: int,
    warmup_on_z: float,
    warmup_on_ratio: float,
    warmup_on_votes_required: int,
    warmup_on_votes_window: int,
    warmup_on_confirm_min_sec: float,
    warmup_handoff_grace_ticks: int,
    force_off_long_on_ratio: float | None,
    baseline_include_quiet_on: bool,
    baseline_quiet_on_max_sec: float,
    min_interval_sec_for_alert: float,
    stitch_gap_sec: float,
    **_ignored: object,
) -> None:
    """Validate runtime parameters used by the streaming detector loop."""

    if (not np.isfinite(window_sec)) or (float(window_sec) <= 0):
        raise ValueError("window_sec must be positive")
    if (not np.isfinite(long_window_sec)) or (float(long_window_sec) <= 0):
        raise ValueError("long_window_sec must be positive")
    if (not np.isfinite(cut_off)) or (float(cut_off) <= 0):
        raise ValueError("risk_cut_off must be positive")
    if (not np.isfinite(cut_on)) or (float(cut_on) <= 0):
        raise ValueError("risk_cut_on must be positive")
    if float(cut_on) < float(cut_off):
        raise ValueError("risk_cut_on must be >= risk_cut_off")
    if int(short_dynamic_min_baseline) <= 0:
        raise ValueError("short_dynamic_min_baseline must be >= 1")
    if (not np.isfinite(short_dynamic_on_z)) or (float(short_dynamic_on_z) < 0.0):
        raise ValueError("short_dynamic_on_z must be >= 0")
    if (not np.isfinite(short_dynamic_off_z)) or (float(short_dynamic_off_z) < 0.0):
        raise ValueError("short_dynamic_off_z must be >= 0")
    if float(short_dynamic_off_z) > float(short_dynamic_on_z):
        raise ValueError("short_dynamic_off_z must be <= short_dynamic_on_z")
    if (not np.isfinite(short_dynamic_on_max_mult)) or (float(short_dynamic_on_max_mult) < 1.0):
        raise ValueError("short_dynamic_on_max_mult must be >= 1")
    if (not np.isfinite(short_dynamic_off_max_mult)) or (float(short_dynamic_off_max_mult) < 1.0):
        raise ValueError("short_dynamic_off_max_mult must be >= 1")
    if float(short_dynamic_off_max_mult) > float(short_dynamic_on_max_mult):
        raise ValueError("short_dynamic_off_max_mult must be <= short_dynamic_on_max_mult")
    if (not np.isfinite(short_damped_cut_relax)) or (float(short_damped_cut_relax) <= 0.0) or (float(short_damped_cut_relax) > 1.0):
        raise ValueError("short_damped_cut_relax must be in (0, 1]")
    if (not np.isfinite(rms_decay_window_sec)) or (float(rms_decay_window_sec) <= 0.0):
        raise ValueError("rms_decay_window_sec must be positive")
    if (not np.isfinite(rms_decay_rms_win_sec)) or (float(rms_decay_rms_win_sec) <= 0.0):
        raise ValueError("rms_decay_rms_win_sec must be positive")
    if (not np.isfinite(rms_decay_step_sec)) or (float(rms_decay_step_sec) <= 0.0):
        raise ValueError("rms_decay_step_sec must be positive")
    if int(rms_decay_min_windows) <= 0:
        raise ValueError("rms_decay_min_windows must be >= 1")
    if (not np.isfinite(rms_decay_on_max)):
        raise ValueError("rms_decay_on_max must be finite")
    if (not np.isfinite(rms_decay_off_min)):
        raise ValueError("rms_decay_off_min must be finite")
    if float(rms_decay_off_min) > float(rms_decay_on_max):
        raise ValueError("rms_decay_off_min must be <= rms_decay_on_max")
    if (not np.isfinite(rms_decay_cache_refresh_sec)) or (float(rms_decay_cache_refresh_sec) <= 0.0):
        raise ValueError("rms_decay_cache_refresh_sec must be positive")
    if (not np.isfinite(rms_decay_event_max_window_sec)) or (float(rms_decay_event_max_window_sec) <= 0.0):
        raise ValueError("rms_decay_event_max_window_sec must be positive")
    if (not np.isfinite(rms_decay_event_min_window_sec)) or (float(rms_decay_event_min_window_sec) <= 0.0):
        raise ValueError("rms_decay_event_min_window_sec must be positive")
    if float(rms_decay_event_max_window_sec) < float(rms_decay_event_min_window_sec):
        raise ValueError("rms_decay_event_max_window_sec must be >= rms_decay_event_min_window_sec")
    if (not np.isfinite(rms_decay_event_rms_win_sec)) or (float(rms_decay_event_rms_win_sec) <= 0.0):
        raise ValueError("rms_decay_event_rms_win_sec must be positive")
    if (not np.isfinite(rms_decay_event_step_sec)) or (float(rms_decay_event_step_sec) <= 0.0):
        raise ValueError("rms_decay_event_step_sec must be positive")
    if int(rms_decay_event_min_windows) <= 0:
        raise ValueError("rms_decay_event_min_windows must be >= 1")
    if (not np.isfinite(evidence_alpha)) or not (0.0 < float(evidence_alpha) < 1.0):
        raise ValueError("evidence_alpha must be in (0, 1)")
    if (not np.isfinite(evidence_clip)) or (float(evidence_clip) <= 0):
        raise ValueError("evidence_clip must be positive")
    if (not np.isfinite(theta_on)) or (not np.isfinite(theta_off)):
        raise ValueError("theta_on/theta_off must be finite")
    if float(theta_off) >= float(theta_on):
        raise ValueError("theta_off must be smaller than theta_on")
    if int(on_consecutive_required) <= 0:
        raise ValueError("on_consecutive_required must be >= 1")
    if int(on_short_votes_window) <= 0:
        raise ValueError("on_short_votes_window must be >= 1")
    if int(on_consecutive_required) > int(on_short_votes_window):
        raise ValueError("on_consecutive_required must be <= on_short_votes_window")
    if (not np.isfinite(on_confirm_min_sec)) or (float(on_confirm_min_sec) < 0.0):
        raise ValueError("on_confirm_min_sec must be >= 0")
    if (not np.isfinite(on_accel_score_log_min)):
        raise ValueError("on_accel_score_log_min must be finite")
    if (not np.isfinite(on_accel_evidence_min)):
        raise ValueError("on_accel_evidence_min must be finite")
    if int(damped_force_off_streak) <= 0:
        raise ValueError("damped_force_off_streak must be >= 1")
    if (not np.isfinite(off_hold_down_sec)) or (float(off_hold_down_sec) < 0.0):
        raise ValueError("off_hold_down_sec must be >= 0")
    if (not np.isfinite(off_confirm_min_sec)) or (float(off_confirm_min_sec) < 0.0):
        raise ValueError("off_confirm_min_sec must be >= 0")
    if (not np.isfinite(off_periodicity_collapse_conf_raw_max)) or (float(off_periodicity_collapse_conf_raw_max) < 0.0) or (float(off_periodicity_collapse_conf_raw_max) > 1.0):
        raise ValueError("off_periodicity_collapse_conf_raw_max must be in [0, 1]")
    if int(off_periodicity_collapse_streak_required) <= 0:
        raise ValueError("off_periodicity_collapse_streak_required must be >= 1")
    if (not np.isfinite(excluded_damped_evidence_penalty)) or (float(excluded_damped_evidence_penalty) < 0.0):
        raise ValueError("excluded_damped_evidence_penalty must be >= 0")
    if int(excluded_damped_hard_penalty_streak) <= 0:
        raise ValueError("excluded_damped_hard_penalty_streak must be >= 1")
    if int(long_min_points) <= 0:
        raise ValueError("long_min_points must be >= 1")
    if int(long_min_baseline) <= 0:
        raise ValueError("long_min_baseline must be >= 1")
    if int(long_baseline_max) <= 0:
        raise ValueError("long_baseline_max must be >= 1")
    if (not np.isfinite(long_off_recent_window_sec)) or (float(long_off_recent_window_sec) <= 0.0):
        raise ValueError("long_off_recent_window_sec must be positive")
    if int(long_off_votes_required) <= 0:
        raise ValueError("long_off_votes_required must be >= 1")
    if int(long_off_votes_window) <= 0:
        raise ValueError("long_off_votes_window must be >= 1")
    if int(long_off_votes_required) > int(long_off_votes_window):
        raise ValueError("long_off_votes_required must be <= long_off_votes_window")
    if int(long_off_recent_min_points) <= 0:
        raise ValueError("long_off_recent_min_points must be >= 1")
    if int(acf_min_points) <= 0:
        raise ValueError("acf_min_points must be >= 1")
    if (not np.isfinite(acf_min_period_sec)) or (float(acf_min_period_sec) <= 0.0):
        raise ValueError("acf_min_period_sec must be positive")
    if (not np.isfinite(acf_max_period_sec)) or (float(acf_max_period_sec) <= 0.0):
        raise ValueError("acf_max_period_sec must be positive")
    if float(acf_max_period_sec) <= float(acf_min_period_sec):
        raise ValueError("acf_max_period_sec must be larger than acf_min_period_sec")
    if (not np.isfinite(confidence_on_min)) or (float(confidence_on_min) < 0.0) or (float(confidence_on_min) > 1.0):
        raise ValueError("confidence_on_min must be in [0, 1]")
    if (not np.isfinite(confidence_off_max)) or (float(confidence_off_max) < 0.0) or (float(confidence_off_max) > 1.0):
        raise ValueError("confidence_off_max must be in [0, 1]")
    if float(confidence_off_max) > float(confidence_on_min):
        raise ValueError("confidence_off_max should be <= confidence_on_min")
    if (not np.isfinite(confidence_w_acf)) or (float(confidence_w_acf) < 0.0):
        raise ValueError("confidence_w_acf must be >= 0")
    if (not np.isfinite(confidence_w_spec)) or (float(confidence_w_spec) < 0.0):
        raise ValueError("confidence_w_spec must be >= 0")
    if (not np.isfinite(confidence_w_env)) or (float(confidence_w_env) < 0.0):
        raise ValueError("confidence_w_env must be >= 0")
    if (not np.isfinite(confidence_w_fft)) or (float(confidence_w_fft) < 0.0):
        raise ValueError("confidence_w_fft must be >= 0")
    if (float(confidence_w_acf) + float(confidence_w_spec) + float(confidence_w_env) + float(confidence_w_fft)) <= 0.0:
        raise ValueError("sum of confidence weights must be > 0")
    if int(confidence_cal_min_points) <= 0:
        raise ValueError("confidence_cal_min_points must be >= 1")
    if int(confidence_cal_hist_max) <= 0:
        raise ValueError("confidence_cal_hist_max must be >= 1")
    if (not np.isfinite(confidence_raw_on_min_when_cal)) or (float(confidence_raw_on_min_when_cal) < 0.0) or (float(confidence_raw_on_min_when_cal) > 1.0):
        raise ValueError("confidence_raw_on_min_when_cal must be in [0, 1]")
    if (not np.isfinite(confidence_raw_off_max_when_cal)) or (float(confidence_raw_off_max_when_cal) < 0.0) or (float(confidence_raw_off_max_when_cal) > 1.0):
        raise ValueError("confidence_raw_off_max_when_cal must be in [0, 1]")
    if (not np.isfinite(cal_on_noise_scale_low)) or (not np.isfinite(cal_on_noise_scale_high)):
        raise ValueError("cal_on_noise_scale_low/high must be finite")
    if float(cal_on_noise_scale_high) <= float(cal_on_noise_scale_low):
        raise ValueError("cal_on_noise_scale_high must be larger than cal_on_noise_scale_low")
    if (not np.isfinite(cal_on_conf_adapt_gain)) or (float(cal_on_conf_adapt_gain) < 0.0):
        raise ValueError("cal_on_conf_adapt_gain must be >= 0")
    if (not np.isfinite(cal_on_conf_floor)) or (float(cal_on_conf_floor) < 0.0) or (float(cal_on_conf_floor) > 1.0):
        raise ValueError("cal_on_conf_floor must be in [0, 1]")
    if (not np.isfinite(cal_on_conf_ceil)) or (float(cal_on_conf_ceil) < 0.0) or (float(cal_on_conf_ceil) > 1.0):
        raise ValueError("cal_on_conf_ceil must be in [0, 1]")
    if float(cal_on_conf_floor) > float(cal_on_conf_ceil):
        raise ValueError("cal_on_conf_floor must be <= cal_on_conf_ceil")
    if (not np.isfinite(cal_on_support_score_w)) or (float(cal_on_support_score_w) < 0.0):
        raise ValueError("cal_on_support_score_w must be >= 0")
    if (not np.isfinite(cal_on_support_score_acf_bonus)) or (float(cal_on_support_score_acf_bonus) < 0.0) or (float(cal_on_support_score_acf_bonus) > 1.0):
        raise ValueError("cal_on_support_score_acf_bonus must be in [0, 1]")
    if (not np.isfinite(cal_on_support_long_w)) or (float(cal_on_support_long_w) < 0.0):
        raise ValueError("cal_on_support_long_w must be >= 0")
    if (not np.isfinite(cal_on_support_conf_w)) or (float(cal_on_support_conf_w) < 0.0):
        raise ValueError("cal_on_support_conf_w must be >= 0")
    if (not np.isfinite(cal_on_support_accel_w)) or (float(cal_on_support_accel_w) < 0.0):
        raise ValueError("cal_on_support_accel_w must be >= 0")
    if (float(cal_on_support_score_w) + float(cal_on_support_long_w) + float(cal_on_support_conf_w) + float(cal_on_support_accel_w)) <= 0.0:
        raise ValueError("sum of CAL_ON support weights must be > 0")
    if (not np.isfinite(cal_on_support_enter_min)) or (float(cal_on_support_enter_min) < 0.0) or (float(cal_on_support_enter_min) > 1.0):
        raise ValueError("cal_on_support_enter_min must be in [0, 1]")
    if (not np.isfinite(cal_on_support_confirm_min)) or (float(cal_on_support_confirm_min) < 0.0) or (float(cal_on_support_confirm_min) > 1.0):
        raise ValueError("cal_on_support_confirm_min must be in [0, 1]")
    if (not np.isfinite(cal_on_support_hold_min)) or (float(cal_on_support_hold_min) < 0.0) or (float(cal_on_support_hold_min) > 1.0):
        raise ValueError("cal_on_support_hold_min must be in [0, 1]")
    if float(cal_on_support_confirm_min) < float(cal_on_support_enter_min):
        raise ValueError("cal_on_support_confirm_min should be >= cal_on_support_enter_min")
    if (not np.isfinite(cal_on_support_ema_alpha)) or (float(cal_on_support_ema_alpha) < 0.0) or (float(cal_on_support_ema_alpha) >= 1.0):
        raise ValueError("cal_on_support_ema_alpha must be in [0, 1)")
    if int(cal_on_confirm_votes_required) <= 0:
        raise ValueError("cal_on_confirm_votes_required must be >= 1")
    if int(cal_on_confirm_votes_window) <= 0:
        raise ValueError("cal_on_confirm_votes_window must be >= 1")
    if int(cal_on_confirm_votes_required) > int(cal_on_confirm_votes_window):
        raise ValueError("cal_on_confirm_votes_required must be <= cal_on_confirm_votes_window")
    if (not np.isfinite(freq_band_low_hz)) or (float(freq_band_low_hz) <= 0.0):
        raise ValueError("freq_band_low_hz must be positive")
    if (not np.isfinite(freq_band_high_hz)) or (float(freq_band_high_hz) <= 0.0):
        raise ValueError("freq_band_high_hz must be positive")
    if float(freq_band_high_hz) <= float(freq_band_low_hz):
        raise ValueError("freq_band_high_hz must be larger than freq_band_low_hz")
    if (not np.isfinite(periodicity_quality_cache_refresh_sec)) or (float(periodicity_quality_cache_refresh_sec) <= 0.0):
        raise ValueError("periodicity_quality_cache_refresh_sec must be positive")
    if (not np.isfinite(long_z_on)) or (not np.isfinite(long_z_off)):
        raise ValueError("long_z_on/long_z_off must be finite")
    if (not np.isfinite(long_on_ratio)) or (not np.isfinite(long_off_ratio)):
        raise ValueError("long_on_ratio/long_off_ratio must be finite")
    if (float(long_on_ratio) <= 0.0) or (float(long_on_ratio) > 1.0):
        raise ValueError("long_on_ratio must be in (0, 1]")
    if int(warmup_on_min_points) <= 0:
        raise ValueError("warmup_on_min_points must be >= 1")
    if int(warmup_min_baseline) <= 0:
        raise ValueError("warmup_min_baseline must be >= 1")
    if (not np.isfinite(warmup_on_z)):
        raise ValueError("warmup_on_z must be finite")
    if (not np.isfinite(warmup_on_ratio)) or (float(warmup_on_ratio) <= 0.0) or (float(warmup_on_ratio) > 1.0):
        raise ValueError("warmup_on_ratio must be in (0, 1]")
    if int(warmup_on_votes_required) <= 0:
        raise ValueError("warmup_on_votes_required must be >= 1")
    if int(warmup_on_votes_window) <= 0:
        raise ValueError("warmup_on_votes_window must be >= 1")
    if int(warmup_on_votes_required) > int(warmup_on_votes_window):
        raise ValueError("warmup_on_votes_required must be <= warmup_on_votes_window")
    if (not np.isfinite(warmup_on_confirm_min_sec)) or (float(warmup_on_confirm_min_sec) < 0.0):
        raise ValueError("warmup_on_confirm_min_sec must be >= 0")
    if int(warmup_handoff_grace_ticks) < 0:
        raise ValueError("warmup_handoff_grace_ticks must be >= 0")
    if (float(long_off_ratio) < 0.0) or (float(long_off_ratio) > 1.0):
        raise ValueError("long_off_ratio must be in [0, 1]")
    if float(long_z_off) > float(long_z_on):
        raise ValueError("long_z_off should be <= long_z_on")
    if force_off_long_on_ratio is not None:
        if (not np.isfinite(force_off_long_on_ratio)) or (float(force_off_long_on_ratio) < 0.0) or (float(force_off_long_on_ratio) > 1.0):
            raise ValueError("force_off_long_on_ratio must be in [0, 1]")
    if bool(baseline_include_quiet_on):
        if (not np.isfinite(baseline_quiet_on_max_sec)) or (float(baseline_quiet_on_max_sec) <= 0.0):
            raise ValueError("baseline_quiet_on_max_sec must be positive when baseline_include_quiet_on is enabled")
    if (not np.isfinite(min_interval_sec_for_alert)) or (float(min_interval_sec_for_alert) < 0.0):
        raise ValueError("min_interval_sec_for_alert must be >= 0")
    if (not np.isfinite(stitch_gap_sec)) or (float(stitch_gap_sec) < 0.0):
        raise ValueError("stitch_gap_sec must be >= 0")


def _run_streaming_alert_demo_one_channel_flat_impl(
    vcsv: str,
    *,
    device: str = DEFAULT_DEVICE,
    target_channel: str | None = None,
    update_sec: float = DEFAULT_UPDATE_SEC,
    window_sec: float = WIN_SEC,
    risk_cut: float | None = None,
    risk_cut_on: float | None = None,
    risk_cut_off: float | None = None,
    short_dynamic_cut_enabled: bool = True,
    short_dynamic_min_baseline: int = 12,
    short_dynamic_on_z: float = 3.0,
    short_dynamic_off_z: float = 2.0,
    short_dynamic_on_max_mult: float = 2.0,
    short_dynamic_off_max_mult: float = 1.5,
    short_damped_cut_relax: float = 0.85,
    rms_decay_gate_enabled: bool = True,
    rms_decay_window_sec: float = 20.0,
    rms_decay_rms_win_sec: float = 1.0,
    rms_decay_step_sec: float = 0.25,
    rms_decay_min_windows: int = 8,
    rms_decay_on_max: float = 0.08,
    rms_decay_off_min: float = 0.03,
    rms_decay_cache_enabled: bool = True,
    rms_decay_cache_refresh_sec: float = 4.0,
    rms_decay_event_enabled: bool = True,
    rms_decay_event_max_window_sec: float = 60.0,
    rms_decay_event_min_window_sec: float = 12.0,
    rms_decay_event_rms_win_sec: float = 1.0,
    rms_decay_event_step_sec: float = 0.25,
    rms_decay_event_min_windows: int = 8,
    realtime_sleep: bool = DEFAULT_REALTIME_SLEEP,
    print_tick: bool = DEFAULT_PRINT_TICK,
    on_event: Callable[[dict], None] | None = None,
    evidence_alpha: float = 0.8,
    evidence_clip: float = 1.0,
    theta_on: float = 0.45,
    theta_off: float = 0.15,
    on_consecutive_required: int = 2,
    on_short_votes_window: int = 3,
    on_confirm_min_sec: float = 1.0,
    on_accel_score_log_min: float = 0.10,
    on_accel_evidence_min: float = 0.05,
    on_require_accel_for_candidate: bool = False,
    damped_force_off_streak: int = 2,
    off_hold_down_sec: float = 4.0,
    off_confirm_min_sec: float = 2.0,
    off_periodicity_collapse_conf_raw_max: float = 0.45,
    off_periodicity_collapse_streak_required: int = 1,
    excluded_damped_evidence_penalty: float = 0.15,
    excluded_damped_hard_penalty_streak: int = 4,
    force_off_require_long_not_on: bool = True,
    force_off_long_on_ratio: float | None = None,
    long_window_sec: float = 120.0,
    long_min_points: int = 20,
    long_min_baseline: int = 40,
    long_baseline_max: int = 600,
    long_z_on: float = 2.5,
    long_on_ratio: float = 0.20,
    warmup_long_enabled: bool = True,
    warmup_on_min_points: int = 10,
    warmup_min_baseline: int = 10,
    warmup_on_z: float = 8.0,
    warmup_on_ratio: float = 0.57,
    warmup_on_votes_required: int = 2,
    warmup_on_votes_window: int = 3,
    warmup_on_confirm_min_sec: float = 6.0,
    warmup_handoff_grace_ticks: int = 2,
    warmup_cancel_on_excluded_damped: bool = True,
    long_z_off: float = 1.0,
    long_off_ratio: float = 0.20,
    long_off_recent_window_sec: float = 30.0,
    long_off_votes_required: int = 2,
    long_off_votes_window: int = 5,
    long_off_recent_min_points: int = 5,
    acf_min_points: int = 24,
    acf_min_period_sec: float = 4.0,
    acf_max_period_sec: float = 40.0,
    confidence_on_min: float = 0.45,
    confidence_off_max: float = 0.35,
    confidence_use_calibration: bool = False,
    confidence_dual_gate_when_cal: bool = True,
    confidence_raw_on_min_when_cal: float = 0.30,
    confidence_raw_off_max_when_cal: float = 0.35,
    confidence_cal_min_points: int = 40,
    confidence_cal_hist_max: int = 600,
    confidence_cal_off_only: bool = True,
    cal_on_soft_mode: bool = True,
    cal_on_noise_scale_low: float = 0.08,
    cal_on_noise_scale_high: float = 0.30,
    cal_on_conf_adapt_gain: float = 0.15,
    cal_on_conf_floor: float = 0.35,
    cal_on_conf_ceil: float = 0.75,
    cal_on_support_score_w: float = 0.35,
    cal_on_support_score_acf_bonus: float = 0.08,
    cal_on_support_long_w: float = 0.25,
    cal_on_support_conf_w: float = 0.30,
    cal_on_support_accel_w: float = 0.10,
    cal_on_support_enter_min: float = 0.52,
    cal_on_support_confirm_min: float = 0.56,
    cal_on_support_hold_min: float = 0.45,
    cal_on_support_ema_alpha: float = 0.65,
    cal_on_confirm_votes_required: int = 2,
    cal_on_confirm_votes_window: int = 3,
    confidence_w_acf: float = 0.34,
    confidence_w_spec: float = 0.33,
    confidence_w_env: float = 0.33,
    confidence_w_fft: float = 0.10,
    freq_band_low_hz: float = 0.15,
    freq_band_high_hz: float = 20.0,
    freq_linear_detrend: bool = True,
    freq_ar1_whiten: bool = False,
    periodicity_quality_cache_enabled: bool = True,
    periodicity_quality_cache_refresh_sec: float = 6.0,
    baseline_include_quiet_on: bool = False,
    baseline_quiet_on_max_sec: float = 45.0,
    min_interval_sec_for_alert: float = 8.0,
    stitch_gap_sec: float = 6.0,
    emit_legacy_rms_aliases: bool = False,
) -> list[dict]:
    """
    Run 2-second-tick style streaming evaluation on one channel.

    State logic:
    - 5-state machine:
      OFF -> ON_CANDIDATE -> ON_CONFIRMED -> OFF_CANDIDATE -> OFF_CONFIRMED
    - ON path uses short k-of-n + acceleration + short dwell
    - OFF path uses hold-down + periodicity-collapse persistence + m-of-n + dwell
    - final ON/OFF still references long-window robust statistics (median/MAD z-score ratio)
    - periodicity quality is used as confidence (not pass/fail gate)
    - optional confidence quantile calibration maps conf_raw -> conf_cal for threshold portability
    - with calibration, optional dual-gate can require both calibrated and raw confidence
    - CAL_ON supports adaptive/noise-aware confidence gating + soft support scoring + 2-stage confirm
    - short-stage RMS-decay estimate is used as ON gate / OFF hint
    - event-aware RMS-decay is recorded as analysis feature only (not used in transitions)
    - forced OFF from repeated `excluded_damped` is guarded by long activity state
    """
    base_cut = float(SCORE_CUT if risk_cut is None else risk_cut)
    cut_off = float(base_cut if risk_cut_off is None else risk_cut_off)
    cut_on = float((cut_off * 3.0) if risk_cut_on is None else risk_cut_on)

    _validate_detector_runtime_args(**locals())

    selected = [target_channel] if (target_channel and str(target_channel).strip()) else None
    batches = build_update_batches_from_voltage_csv(
        vcsv,
        device=device,
        update_sec=float(update_sec),
        target_channels=selected,
        max_channels=1,
    )
    if not batches:
        print("[STREAM] no batches were created")
        return []

    states: dict[tuple[str, str], ChannelStreamState] = {}
    events: list[dict] = []
    raw_risk_interval_count = 0
    suppressed_interval_count = 0
    last_interval_by_key: dict[tuple[str, str], dict] = {}
    next_interval_id = 1
    max_keep_sec = max(
        float(window_sec) * 2.0,
        float(window_sec) + float(BASE_SEC) + float(RMS_WIN_SEC),
        float(rms_decay_window_sec) + float(rms_decay_rms_win_sec) + float(rms_decay_step_sec),
        float(rms_decay_event_max_window_sec) + float(rms_decay_event_rms_win_sec) + float(rms_decay_event_step_sec),
    )

    for upd_idx, batch in enumerate(batches, start=1):
        if not batch:
            if realtime_sleep:
                time.sleep(float(update_sec))
            continue

        touched_keys = set()
        for dev, ch, t, v in batch:
            key = (str(dev), str(ch))
            st = states.get(key)
            if st is None:
                st = ChannelStreamState()
                states[key] = st
            t_f = float(t)
            v_f = float(v)
            if st.ring and (t_f < float(st.ring[-1][0])):
                st.ring_time_sorted = False
            st.ring.append((t_f, v_f))
            touched_keys.add(key)

        for key in touched_keys:
            st = states[key]
            feature_base = _extract_features(
                st,
                max_keep_sec=float(max_keep_sec),
                window_sec=float(window_sec),
            )
            if feature_base is None:
                continue
            tw0 = feature_base["tw0"]
            vw0 = feature_base["vw0"]
            tw = feature_base["tw"]
            vw = feature_base["vw"]
            dt = float(feature_base["dt"])
            t0 = float(feature_base["t0"])
            t1 = float(feature_base["t1"])
            score = float(feature_base["score"])
            A_tail = float(feature_base["A_tail"])
            D_tail = float(feature_base["D_tail"])
            reason = str(feature_base["reason"])
            score_reason_ok = bool(feature_base["score_reason_ok"])

            if reason == "excluded_damped":
                st.damped_streak += 1
            else:
                st.damped_streak = 0

            pre_base_med, pre_base_scale = _robust_center_scale(st.long_baseline_hist)
            cut_on_eff = float(cut_on)
            cut_off_eff = float(cut_off)
            if bool(short_dynamic_cut_enabled) and (len(st.long_baseline_hist) >= int(short_dynamic_min_baseline)):
                cut_on_dyn = _dynamic_score_cut_from_log_baseline(
                    base_med=pre_base_med,
                    base_scale=pre_base_scale,
                    z_thr=float(short_dynamic_on_z),
                    fallback_cut=float(cut_on),
                )
                cut_off_dyn = _dynamic_score_cut_from_log_baseline(
                    base_med=pre_base_med,
                    base_scale=pre_base_scale,
                    z_thr=float(short_dynamic_off_z),
                    fallback_cut=float(cut_off),
                )
                cut_on_eff = float(np.clip(
                    float(cut_on_dyn),
                    float(cut_on),
                    float(cut_on) * float(short_dynamic_on_max_mult),
                ))
                cut_off_eff = float(np.clip(
                    float(cut_off_dyn),
                    float(cut_off),
                    float(cut_off) * float(short_dynamic_off_max_mult),
                ))
            if float(cut_off_eff) > float(cut_on_eff):
                cut_off_eff = float(cut_on_eff)
            damped_cut_relax = float(short_damped_cut_relax if (reason == "excluded_damped") else 1.0)
            cut_on_cmp = float(max(1e-18, float(cut_on_eff) * float(damped_cut_relax)))
            cut_off_cmp = float(max(1e-18, float(cut_off_eff) * float(damped_cut_relax)))
            risk_prev = bool(_is_risk_active_phase(st.phase))

            short_high = score_reason_ok and np.isfinite(score) and (float(score) >= float(cut_on_cmp))
            st.on_short_votes.append(1 if short_high else 0)
            while len(st.on_short_votes) > int(on_short_votes_window):
                st.on_short_votes.popleft()
            st.on_candidate_streak = int(sum(st.on_short_votes))

            score_log_safe = float(score) if (np.isfinite(score) and float(score) > 0.0) else 1e-18
            s_log = float(np.log10(score_log_safe))
            short_trigger = bool(
                (len(st.on_short_votes) >= int(on_short_votes_window))
                and (int(sum(st.on_short_votes)) >= int(on_consecutive_required))
            )
            rms_transition_active = bool(
                risk_prev
                or short_high
                or short_trigger
                or (str(st.phase) != PHASE_OFF)
            )
            rms_decay_idle_refresh_sec = max(float(update_sec), float(rms_decay_cache_refresh_sec))
            can_reuse_rms = bool(
                bool(rms_decay_cache_enabled)
                and (not rms_transition_active)
                and np.isfinite(st.last_rms_decay_t_end)
                and ((float(t1) - float(st.last_rms_decay_t_end)) < float(rms_decay_idle_refresh_sec))
            )
            if can_reuse_rms:
                rms_decay_local = float(st.last_rms_decay)
                rms_decay_local_r2 = float(st.last_rms_decay_r2)
                rms_decay_local_n = int(st.last_rms_decay_n)
            else:
                rms_decay_local, rms_decay_local_r2, rms_decay_local_n = _local_rms_decay_from_signal(
                    tw0,
                    vw0,
                    trailing_sec=float(rms_decay_window_sec),
                    rms_win_sec=float(rms_decay_rms_win_sec),
                    step_sec=float(rms_decay_step_sec),
                    min_windows=int(rms_decay_min_windows),
                )
                st.last_rms_decay = float(rms_decay_local) if np.isfinite(rms_decay_local) else float("nan")
                st.last_rms_decay_r2 = float(rms_decay_local_r2) if np.isfinite(rms_decay_local_r2) else float("nan")
                st.last_rms_decay_n = int(rms_decay_local_n)
                st.last_rms_decay_t_end = float(t1)
            rms_decay_local_on_ok = bool(
                (not bool(rms_decay_gate_enabled))
                or (not np.isfinite(rms_decay_local))
                or (float(rms_decay_local) <= float(rms_decay_on_max))
            )
            rms_decay_local_off_hint = bool(
                bool(rms_decay_gate_enabled)
                and np.isfinite(rms_decay_local)
                and (float(rms_decay_local) >= float(rms_decay_off_min))
            )
            rms_decay_event = float("nan")
            rms_decay_event_r2 = float("nan")
            rms_decay_event_n = 0
            rms_decay_event_win_sec = float("nan")
            if (
                bool(rms_decay_event_enabled)
                and (str(st.phase) in {PHASE_ON_CONFIRMED, PHASE_OFF_CANDIDATE})
                and (st.active_start_t is not None)
                and np.isfinite(st.active_start_t)
            ):
                event_age_sec = float(t1 - float(st.active_start_t))
                if float(event_age_sec) >= float(rms_decay_event_min_window_sec):
                    rms_decay_event, rms_decay_event_r2, rms_decay_event_n, rms_decay_event_win_sec = _event_rms_decay_from_signal(
                        tw0,
                        vw0,
                        event_start_t=float(st.active_start_t),
                        t_end=float(t1),
                        max_window_sec=float(rms_decay_event_max_window_sec),
                        min_window_sec=float(rms_decay_event_min_window_sec),
                        rms_win_sec=float(rms_decay_event_rms_win_sec),
                        step_sec=float(rms_decay_event_step_sec),
                        min_windows=int(rms_decay_event_min_windows),
                    )
            st.last_rms_decay_event = float(rms_decay_event) if np.isfinite(rms_decay_event) else float("nan")
            st.last_rms_decay_event_r2 = float(rms_decay_event_r2) if np.isfinite(rms_decay_event_r2) else float("nan")
            st.last_rms_decay_event_n = int(rms_decay_event_n)
            st.last_rms_decay_event_win_sec = float(rms_decay_event_win_sec) if np.isfinite(rms_decay_event_win_sec) else float("nan")
            st.last_rms_decay_event_t_end = float(t1)
            short_release = (not score_reason_ok) or (not np.isfinite(score)) or (float(score) < float(cut_off_cmp))
            long_stats = _update_baseline_and_long_stats(
                st,
                t1=float(t1),
                s_log=float(s_log),
                risk_prev=bool(risk_prev),
                reason=str(reason),
                short_high=bool(short_high),
                score=float(score),
                cut_off_cmp=float(cut_off_cmp),
                baseline_include_quiet_on=bool(baseline_include_quiet_on),
                baseline_quiet_on_max_sec=float(baseline_quiet_on_max_sec),
                long_baseline_max=int(long_baseline_max),
                long_window_sec=float(long_window_sec),
                long_off_recent_window_sec=float(long_off_recent_window_sec),
                long_z_on=float(long_z_on),
                long_z_off=float(long_z_off),
            )
            base_med = float(long_stats.base_med)
            base_scale = float(long_stats.base_scale)
            long_ratio_on = float(long_stats.long_ratio_on)
            long_zmax = float(long_stats.long_zmax)
            long_n = int(long_stats.long_n)
            long_ratio_off = float(long_stats.long_ratio_off)
            long_ratio_off_recent = float(long_stats.long_ratio_off_recent)
            long_off_n_recent = int(long_stats.long_off_n_recent)
            phase_now = str(st.phase)

            quality_snapshot = _compute_quality_and_support(
                st,
                tw=tw,
                vw=vw,
                t1=float(t1),
                score=float(score),
                score_reason_ok=bool(score_reason_ok),
                cut_off_cmp=float(cut_off_cmp),
                cut_on_eff=float(cut_on_eff),
                cut_off_eff=float(cut_off_eff),
                s_log=float(s_log),
                reason=str(reason),
                risk_prev=bool(risk_prev),
                short_high=bool(short_high),
                short_trigger=bool(short_trigger),
                long_ratio_on=float(long_ratio_on),
                long_n=int(long_n),
                base_med=float(base_med),
                base_scale=float(base_scale),
                baseline_n=int(len(st.long_baseline_hist)),
                long_min_points=int(long_min_points),
                long_min_baseline=int(long_min_baseline),
                long_on_ratio=float(long_on_ratio),
                warmup_long_enabled=bool(warmup_long_enabled),
                warmup_on_min_points=int(warmup_on_min_points),
                warmup_min_baseline=int(warmup_min_baseline),
                warmup_on_z=float(warmup_on_z),
                warmup_on_ratio=float(warmup_on_ratio),
                long_zmax=float(long_zmax),
                confidence_use_calibration=bool(confidence_use_calibration),
                confidence_cal_min_points=int(confidence_cal_min_points),
                confidence_cal_hist_max=int(confidence_cal_hist_max),
                confidence_cal_off_only=bool(confidence_cal_off_only),
                acf_min_points=int(acf_min_points),
                acf_min_period_sec=float(acf_min_period_sec),
                acf_max_period_sec=float(acf_max_period_sec),
                freq_band_low_hz=float(freq_band_low_hz),
                freq_band_high_hz=float(freq_band_high_hz),
                freq_linear_detrend=bool(freq_linear_detrend),
                freq_ar1_whiten=bool(freq_ar1_whiten),
                confidence_w_acf=float(confidence_w_acf),
                confidence_w_spec=float(confidence_w_spec),
                confidence_w_env=float(confidence_w_env),
                confidence_w_fft=float(confidence_w_fft),
                update_sec=float(update_sec),
                periodicity_quality_cache_enabled=bool(periodicity_quality_cache_enabled),
                periodicity_quality_cache_refresh_sec=float(periodicity_quality_cache_refresh_sec),
                rms_decay_local=float(rms_decay_local),
                rms_decay_local_r2=float(rms_decay_local_r2),
                rms_decay_local_n=int(rms_decay_local_n),
                rms_decay_local_on_ok=bool(rms_decay_local_on_ok),
                rms_decay_local_off_hint=bool(rms_decay_local_off_hint),
                rms_decay_event=float(rms_decay_event),
                rms_decay_event_r2=float(rms_decay_event_r2),
                rms_decay_event_n=int(rms_decay_event_n),
                rms_decay_event_win_sec=float(rms_decay_event_win_sec),
                evidence_alpha=float(evidence_alpha),
                evidence_clip=float(evidence_clip),
                excluded_damped_evidence_penalty=float(excluded_damped_evidence_penalty),
                excluded_damped_hard_penalty_streak=int(excluded_damped_hard_penalty_streak),
                on_require_accel_for_candidate=bool(on_require_accel_for_candidate),
                on_accel_score_log_min=float(on_accel_score_log_min),
                on_accel_evidence_min=float(on_accel_evidence_min),
                confidence_on_min=float(confidence_on_min),
                cal_on_soft_mode=bool(cal_on_soft_mode),
                confidence_raw_on_min_when_cal=float(confidence_raw_on_min_when_cal),
                cal_on_noise_scale_low=float(cal_on_noise_scale_low),
                cal_on_noise_scale_high=float(cal_on_noise_scale_high),
                cal_on_conf_adapt_gain=float(cal_on_conf_adapt_gain),
                cal_on_conf_floor=float(cal_on_conf_floor),
                cal_on_conf_ceil=float(cal_on_conf_ceil),
                cal_on_support_score_w=float(cal_on_support_score_w),
                cal_on_support_score_acf_bonus=float(cal_on_support_score_acf_bonus),
                cal_on_support_long_w=float(cal_on_support_long_w),
                cal_on_support_conf_w=float(cal_on_support_conf_w),
                cal_on_support_accel_w=float(cal_on_support_accel_w),
                cal_on_support_enter_min=float(cal_on_support_enter_min),
                cal_on_confirm_votes_window=int(cal_on_confirm_votes_window),
                cal_on_confirm_votes_required=int(cal_on_confirm_votes_required),
                cal_on_support_ema_alpha=float(cal_on_support_ema_alpha),
                phase_now=str(phase_now),
            )
            acf_peak = float(quality_snapshot["acf_peak"])
            acf_period_sec = float(quality_snapshot["acf_period_sec"])
            acf_lag_steps = int(quality_snapshot["acf_lag_steps"])
            acf_n = int(quality_snapshot["acf_n"])
            confidence = float(quality_snapshot["confidence"])
            confidence_raw = float(quality_snapshot["confidence_raw"])
            confidence_cal = float(quality_snapshot["confidence_cal"])
            c_acf = float(quality_snapshot["c_acf"])
            c_spec = float(quality_snapshot["c_spec"])
            c_env = float(quality_snapshot["c_env"])
            c_fft = float(quality_snapshot["c_fft"])
            c_freq_agree = float(quality_snapshot["c_freq_agree"])
            f_welch = float(quality_snapshot["f_welch"])
            f_zc = float(quality_snapshot["f_zc"])
            f_fft = float(quality_snapshot["f_fft"])
            rms_decay = float(quality_snapshot["rms_decay"])
            rms_decay_r2 = float(quality_snapshot["rms_decay_r2"])
            rms_decay_n = int(quality_snapshot["rms_decay_n"])
            rms_decay_on_ok = int(quality_snapshot["rms_decay_on_ok"])
            rms_decay_off_hint = int(quality_snapshot["rms_decay_off_hint"])
            rms_decay_event = float(quality_snapshot["rms_decay_event"])
            rms_decay_event_r2 = float(quality_snapshot["rms_decay_event_r2"])
            rms_decay_event_n = int(quality_snapshot["rms_decay_event_n"])
            rms_decay_event_win_sec = float(quality_snapshot["rms_decay_event_win_sec"])
            long_ready = bool(quality_snapshot["long_ready"])
            warmup_mode = bool(quality_snapshot["warmup_mode"])
            e_t = float(quality_snapshot["e_t"])
            delta_s_log = float(quality_snapshot["delta_s_log"])
            delta_e = float(quality_snapshot["delta_e"])
            accel_ok = bool(quality_snapshot["accel_ok"])
            conf_now = float(quality_snapshot["conf_now"])
            raw_now = float(quality_snapshot["raw_now"])
            use_cal_gate = bool(quality_snapshot["use_cal_gate"])
            cal_on_active = bool(quality_snapshot["cal_on_active"])
            cal_on_conf_thr = float(quality_snapshot["cal_on_conf_thr"])
            on_support = float(quality_snapshot["on_support"])
            on_support_ema = float(quality_snapshot["on_support_ema"])
            on_soft_vote_sum = int(quality_snapshot["on_soft_vote_sum"])
            on_soft_confirmed = bool(quality_snapshot["on_soft_confirmed"])

            warmup_core = bool(
                bool(warmup_mode)
                and (int(long_n) >= int(warmup_on_min_points))
                and (len(st.long_baseline_hist) >= int(warmup_min_baseline))
                and np.isfinite(long_zmax)
                and (float(long_zmax) >= float(warmup_on_z))
                and np.isfinite(long_ratio_on)
                and (float(long_ratio_on) >= float(warmup_on_ratio))
            )
            # Rollback kept intentionally: warmup raw-confidence hard gate is not reconnected.
            # Deprecated param: warmup_on_conf_raw_min
            if (
                bool(warmup_cancel_on_excluded_damped)
                and (reason == "excluded_damped")
                and ((not np.isfinite(score)) or (float(score) < float(cut_off_cmp)))
            ):
                warmup_core = False
            if phase_now in {PHASE_OFF, PHASE_ON_CANDIDATE}:
                st.warmup_on_votes.append(1 if warmup_core else 0)
                while len(st.warmup_on_votes) > int(warmup_on_votes_window):
                    st.warmup_on_votes.popleft()
                if warmup_core:
                    if st.warmup_on_start_t is None:
                        st.warmup_on_start_t = float(t1)
                        st.warmup_on_start_update_idx = int(upd_idx)
                else:
                    st.warmup_on_start_t = None
                    st.warmup_on_start_update_idx = None
            else:
                st.warmup_on_votes.clear()
                st.warmup_on_start_t = None
                st.warmup_on_start_update_idx = None
            warmup_vote_sum = int(sum(st.warmup_on_votes))
            warmup_votes_ready = bool(
                (len(st.warmup_on_votes) >= int(warmup_on_votes_window))
                and (int(warmup_vote_sum) >= int(warmup_on_votes_required))
            )
            warmup_age_sec = (
                float(t1 - float(st.warmup_on_start_t))
                if (st.warmup_on_start_t is not None) and np.isfinite(st.warmup_on_start_t)
                else 0.0
            )
            warmup_on_confirmed = bool(
                warmup_votes_ready
                and (float(warmup_age_sec) >= float(warmup_on_confirm_min_sec))
            )

            off_snapshot = _compute_off_path(
                st,
                phase_now=str(phase_now),
                short_release=bool(short_release),
                conf_now=float(conf_now),
                raw_now=float(raw_now),
                use_cal_gate=bool(use_cal_gate),
                confidence_off_max=float(confidence_off_max),
                off_periodicity_collapse_conf_raw_max=float(off_periodicity_collapse_conf_raw_max),
                confidence_raw_off_max_when_cal=float(confidence_raw_off_max_when_cal),
                off_periodicity_collapse_streak_required=int(off_periodicity_collapse_streak_required),
                long_ready=bool(long_ready),
                long_off_n_recent=int(long_off_n_recent),
                long_off_recent_min_points=int(long_off_recent_min_points),
                long_ratio_off_recent=float(long_ratio_off_recent),
                long_off_ratio=float(long_off_ratio),
                rms_decay_local_off_hint=bool(rms_decay_local_off_hint),
                long_off_votes_window=int(long_off_votes_window),
                long_off_votes_required=int(long_off_votes_required),
                damped_force_off_streak=int(damped_force_off_streak),
                theta_off=float(theta_off),
                force_off_long_on_ratio=force_off_long_on_ratio,
                force_off_require_long_not_on=bool(force_off_require_long_not_on),
                long_ratio_on=float(long_ratio_on),
                long_on_ratio=float(long_on_ratio),
            )
            collapse_ok = bool(off_snapshot.collapse_ok)
            off_vote_core = bool(off_snapshot.off_vote_core)
            long_off_confirmed = bool(off_snapshot.long_off_confirmed)
            force_off_now = bool(off_snapshot.force_off_now)
            transition_reason = str(reason)
            gate_flags, on_entry_vote_sum = _evaluate_gates(
                conf_now=float(conf_now),
                raw_now=float(raw_now),
                use_cal_gate=bool(use_cal_gate),
                cal_on_active=bool(cal_on_active),
                on_support_ema=float(on_support_ema),
                cal_on_conf_thr=float(cal_on_conf_thr),
                confidence_on_min=float(confidence_on_min),
                confidence_dual_gate_when_cal=bool(confidence_dual_gate_when_cal),
                confidence_raw_on_min_when_cal=float(confidence_raw_on_min_when_cal),
                cal_on_support_enter_min=float(cal_on_support_enter_min),
                long_ready=bool(long_ready),
                long_ratio_on=float(long_ratio_on),
                long_on_ratio=float(long_on_ratio),
                warmup_long_enabled=bool(warmup_long_enabled),
                long_ready_streak=int(st.long_ready_streak),
                warmup_handoff_grace_ticks=int(warmup_handoff_grace_ticks),
                warmup_mode=bool(warmup_mode),
                warmup_on_confirmed=bool(warmup_on_confirmed),
                short_trigger=bool(short_trigger),
                baseline_n=int(len(st.long_baseline_hist)),
                short_dynamic_min_baseline=int(short_dynamic_min_baseline),
                accel_ok=bool(accel_ok),
                rms_decay_local_on_ok=bool(rms_decay_local_on_ok),
                rms_decay_local_off_hint=bool(rms_decay_local_off_hint),
                off_vote_core=bool(off_vote_core),
                long_off_confirmed=bool(long_off_confirmed),
                force_off_now=bool(force_off_now),
            )
            long_on_core = bool(long_ready and (float(long_ratio_on) >= float(long_on_ratio)))
            warmup_handoff_active = bool(
                bool(warmup_long_enabled)
                and bool(long_ready)
                and (int(st.long_ready_streak) <= int(warmup_handoff_grace_ticks))
            )

            phase_now, transition_reason = _step_fsm(
                st,
                phase_now=str(phase_now),
                t1=float(t1),
                upd_idx=int(upd_idx),
                reason=str(reason),
                score_reason_ok=bool(score_reason_ok),
                score=float(score),
                cut_off_cmp=float(cut_off_cmp),
                cal_on_active=bool(cal_on_active),
                on_support_ema=float(on_support_ema),
                cal_on_support_hold_min=float(cal_on_support_hold_min),
                on_confirm_min_sec=float(on_confirm_min_sec),
                on_soft_confirmed=bool(on_soft_confirmed),
                cal_on_support_confirm_min=float(cal_on_support_confirm_min),
                off_hold_down_sec=float(off_hold_down_sec),
                short_high=bool(short_high),
                collapse_ok=bool(collapse_ok),
                on_consecutive_required=int(on_consecutive_required),
                off_confirm_min_sec=float(off_confirm_min_sec),
                gate_flags=gate_flags,
            )

            risk_now = bool(_is_risk_active_phase(phase_now))

            if print_tick:
                off_vote_str = f"{int(sum(st.long_off_votes))}/{len(st.long_off_votes)}"
                warmup_vote_str = f"{warmup_vote_sum}/{len(st.warmup_on_votes)}"
                long_gate_mode = ("W" if warmup_mode else ("H" if warmup_handoff_active else "L"))
                print(
                    f"[TICK] upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                    f"t_end={t1:.3f} | score={float(score):.3e} | e={e_t:+.3f} | S={float(st.evidence):+.3f} | "
                    f"phase={phase_now} | on_votes={st.on_candidate_streak}/{len(st.on_short_votes)} | "
                    f"damped_streak={st.damped_streak} | coll_streak={st.periodicity_collapse_streak} | "
                    f"Lon={long_ratio_on:.2f} | Loff={long_ratio_off:.2f} | LoffR={long_ratio_off_recent:.2f} | "
                    f"Conf={confidence:.2f}[raw={confidence_raw:.2f},cal={confidence_cal:.2f}]"
                    f"(acf={c_acf:.2f},spec={c_spec:.2f},env={c_env:.2f},fft={c_fft:.2f},xv={c_freq_agree:.2f}) | "
                    f"f={f_welch:.2f}/{f_zc:.2f}/{f_fft:.2f}Hz | ACF={acf_peak:.2f}@{acf_period_sec:.1f}s | "
                    f"RMSd={rms_decay:+.3f}[n={int(rms_decay_n)},r2={rms_decay_r2:.2f}] | "
                    f"CALthr={cal_on_conf_thr:.2f} | Soft={on_support:.2f}/{on_support_ema:.2f} | "
                    f"SV={on_soft_vote_sum}/{len(st.on_soft_votes)} | Wv={warmup_vote_str}({int(gate_flags.warmup_on_confirmed)}) | "
                    f"CUT[on={cut_on_cmp:.2e},off={cut_off_cmp:.2e}] | "
                    f"Lg={long_gate_mode}:{int(gate_flags.on_long_gate_ok)} | Ev={int(on_entry_vote_sum)}/4 | Lv={off_vote_str} | "
                    f"G[st={int(gate_flags.short_trigger)},sn={int(gate_flags.short_no_long_ready)},ac={int(gate_flags.accel_ok)},cf={int(gate_flags.on_conf_ok)},wr={int(gate_flags.warmup_on_confirmed)},"
                    f"lr={int(gate_flags.long_ready)},ov={int(gate_flags.off_vote_core)},oc={int(gate_flags.long_off_confirmed)},fo={int(gate_flags.force_off_now)},rg={int(gate_flags.rms_decay_on_ok)},ro={int(gate_flags.rms_decay_off_hint)}] | "
                    f"dS={delta_s_log:+.2f} | dE={delta_e:+.2f} | Lzmax={long_zmax:.2f} | "
                    f"Ln={long_n} | Bn={len(st.long_baseline_hist)} | reason={reason} | risk={'ON' if risk_now else 'OFF'}"
                )

            metrics_kwargs = {
                "score": float(score),
                "confidence": float(confidence),
                "confidence_raw": float(confidence_raw),
                "confidence_cal": float(confidence_cal),
                "cal_on_conf_thr": float(cal_on_conf_thr),
                "on_support": float(on_support),
                "on_support_ema": float(on_support_ema),
                "on_soft_votes_sum": int(on_soft_vote_sum),
                "on_soft_votes_n": int(len(st.on_soft_votes)),
                "c_acf": float(c_acf),
                "c_spec": float(c_spec),
                "c_env": float(c_env),
                "c_fft": float(c_fft),
                "c_freq_agree": float(c_freq_agree),
                "f_welch": float(f_welch),
                "f_zc": float(f_zc),
                "f_fft": float(f_fft),
                "A_tail": float(A_tail),
                "D_tail": float(D_tail),
                "rms_decay": float(rms_decay),
                "rms_decay_r2": float(rms_decay_r2),
                "rms_decay_n": int(rms_decay_n),
                "rms_decay_on_ok": int(rms_decay_on_ok),
                "rms_decay_off_hint": int(rms_decay_off_hint),
                "rms_decay_event": float(rms_decay_event),
                "rms_decay_event_r2": float(rms_decay_event_r2),
                "rms_decay_event_n": int(rms_decay_event_n),
                "rms_decay_event_win_sec": float(rms_decay_event_win_sec),
                "reason": str(reason),
                "transition_reason": str(transition_reason),
                "evidence": float(st.evidence),
                "long_ratio_on": float(long_ratio_on),
                "long_ratio_off": float(long_ratio_off),
                "long_ratio_off_recent": float(long_ratio_off_recent),
                "long_off_n_recent": int(long_off_n_recent),
                "long_off_votes_sum": int(sum(st.long_off_votes)),
                "long_off_votes_n": int(len(st.long_off_votes)),
                "acf_peak": float(acf_peak),
                "acf_period_sec": float(acf_period_sec),
                "acf_lag_steps": int(acf_lag_steps),
                "acf_n": int(acf_n),
                "long_zmax": float(long_zmax),
                "long_n": int(long_n),
                "baseline_n": int(len(st.long_baseline_hist)),
                "emit_legacy_rms_aliases": bool(emit_legacy_rms_aliases),
            }
            raw_risk_interval_count, suppressed_interval_count, next_interval_id = _emit_events(
                events=events,
                on_event=on_event,
                st=st,
                key=key,
                upd_idx=int(upd_idx),
                t1=float(t1),
                risk_now=bool(risk_now),
                transition_reason=str(transition_reason),
                min_interval_sec_for_alert=float(min_interval_sec_for_alert),
                raw_risk_interval_count=int(raw_risk_interval_count),
                suppressed_interval_count=int(suppressed_interval_count),
                last_interval_by_key=last_interval_by_key,
                next_interval_id=int(next_interval_id),
                stitch_gap_sec=float(stitch_gap_sec),
                metrics_kwargs=metrics_kwargs,
            )

            if str(phase_now) == PHASE_OFF_CONFIRMED:
                st.phase = PHASE_OFF
            elif (not risk_now) and str(phase_now) == PHASE_OFF:
                st.phase = PHASE_OFF
            else:
                st.phase = str(phase_now)
            st.last_risk_on = bool(risk_now)

        if realtime_sleep:
            time.sleep(float(update_sec))

    for key, st in states.items():
        if (not st.last_risk_on) or (not st.ring):
            continue
        end_t = float(st.ring[-1][0])
        start_t = float(st.active_start_t) if st.active_start_t is not None else float("nan")
        start_idx = int(st.active_start_update_idx) if st.active_start_update_idx is not None else -1
        duration_sec = float(end_t - start_t) if np.isfinite(start_t) else float("nan")
        raw_risk_interval_count += 1
        q = st.last_quality if isinstance(st.last_quality, dict) else {}
        acf_peak = float(q.get("acf_peak", np.nan))
        acf_period_sec = float(q.get("acf_period_sec", np.nan))
        acf_lag_steps = int(q.get("acf_lag_steps", -1))
        acf_n = int(q.get("acf_n", 0))
        confidence = float(q.get("confidence", np.nan))
        confidence_raw = float(q.get("confidence_raw", np.nan))
        confidence_cal = float(q.get("confidence_cal", np.nan))
        c_acf = float(q.get("c_acf", np.nan))
        c_spec = float(q.get("c_spec", np.nan))
        c_env = float(q.get("c_env", np.nan))
        c_fft = float(q.get("c_fft", np.nan))
        c_freq_agree = float(q.get("c_freq_agree", np.nan))
        f_welch = float(q.get("f_welch", np.nan))
        f_zc = float(q.get("f_zc", np.nan))
        f_fft = float(q.get("f_fft", np.nan))
        # Canonical-first read with legacy fallback (phase-out target after consumer migration).
        rms_metrics = _extract_quality_rms_metrics(q)
        rms_decay_local = float(rms_metrics["rms_decay_local"])
        rms_decay_local_r2 = float(rms_metrics["rms_decay_local_r2"])
        rms_decay_local_n = int(rms_metrics["rms_decay_local_n"])
        rms_decay_local_on_ok = int(rms_metrics["rms_decay_local_on_ok"])
        rms_decay_local_off_hint = int(rms_metrics["rms_decay_local_off_hint"])
        rms_decay_event = float(rms_metrics["rms_decay_event"])
        rms_decay_event_r2 = float(rms_metrics["rms_decay_event_r2"])
        rms_decay_event_n = int(rms_metrics["rms_decay_event_n"])
        rms_decay_event_win_sec = float(rms_metrics["rms_decay_event_win_sec"])
        rms_decay = float(rms_metrics["rms_decay"])
        rms_decay_r2 = float(rms_metrics["rms_decay_r2"])
        rms_decay_n = int(rms_metrics["rms_decay_n"])
        rms_decay_on_ok = int(rms_metrics["rms_decay_on_ok"])
        rms_decay_off_hint = int(rms_metrics["rms_decay_off_hint"])
        if np.isfinite(duration_sec) and (duration_sec < float(min_interval_sec_for_alert)):
            suppressed_interval_count += 1
            print(
                f"[SUPPRESS] short_open_interval | dev={key[0]} | ch={key[1]} | "
                f"start={start_t:.3f}(upd={start_idx:03d}) | end={end_t:.3f}(stream_end) | "
                f"duration={duration_sec:.3f}s < min={float(min_interval_sec_for_alert):.3f}s"
            )
            continue
        if not st.on_event_emitted and np.isfinite(start_t):
            on_ev = {
                "event": "risk_on",
                "update_idx": int(start_idx if start_idx >= 0 else len(batches)),
                "device": str(key[0]),
                "channel": str(key[1]),
                "t_end": float(start_t),
                "start_t": start_t,
                "start_update_idx": start_idx,
                **_build_risk_event_metrics(
                    score=float("nan"),
                    confidence=float(confidence),
                    confidence_raw=float(confidence_raw),
                    confidence_cal=float(confidence_cal),
                    cal_on_conf_thr=float("nan"),
                    on_support=float("nan"),
                    on_support_ema=float(st.on_support_ema),
                    on_soft_votes_sum=int(sum(st.on_soft_votes)),
                    on_soft_votes_n=int(len(st.on_soft_votes)),
                    c_acf=float(c_acf),
                    c_spec=float(c_spec),
                    c_env=float(c_env),
                    c_fft=float(c_fft),
                    c_freq_agree=float(c_freq_agree),
                    f_welch=float(f_welch),
                    f_zc=float(f_zc),
                    f_fft=float(f_fft),
                    A_tail=float("nan"),
                    D_tail=float("nan"),
                    rms_decay=float(rms_decay),
                    rms_decay_r2=float(rms_decay_r2),
                    rms_decay_n=int(rms_decay_n),
                    rms_decay_on_ok=int(rms_decay_on_ok),
                    rms_decay_off_hint=int(rms_decay_off_hint),
                    rms_decay_event=float(rms_decay_event),
                    rms_decay_event_r2=float(rms_decay_event_r2),
                    rms_decay_event_n=int(rms_decay_event_n),
                    rms_decay_event_win_sec=float(rms_decay_event_win_sec),
                    reason="stream_end_delayed_emit",
                    transition_reason="delayed_emit_before_open_close",
                    evidence=float(st.evidence),
                    long_ratio_on=float("nan"),
                    long_ratio_off=float("nan"),
                    long_ratio_off_recent=float("nan"),
                    long_off_n_recent=0,
                    long_off_votes_sum=int(sum(st.long_off_votes)),
                    long_off_votes_n=int(len(st.long_off_votes)),
                    acf_peak=float(acf_peak),
                    acf_period_sec=float(acf_period_sec),
                    acf_lag_steps=int(acf_lag_steps),
                    acf_n=int(acf_n),
                    long_zmax=float("nan"),
                    long_n=int(len(st.long_score_hist)),
                    baseline_n=int(len(st.long_baseline_hist)),
                    emit_legacy_rms_aliases=bool(emit_legacy_rms_aliases),
                ),
            }
            events.append(on_ev)
            if on_event is not None:
                on_event(on_ev)
        ev = {
            "event": "risk_interval_open",
            "device": str(key[0]),
            "channel": str(key[1]),
            "start_t": start_t,
            "start_update_idx": start_idx,
            "end_t": float(end_t),
            "end_update_idx": int(len(batches)),
            "duration_sec": duration_sec,
            **_build_risk_event_metrics(
                score=float("nan"),
                confidence=float(confidence),
                confidence_raw=float(confidence_raw),
                confidence_cal=float(confidence_cal),
                cal_on_conf_thr=float("nan"),
                on_support=float("nan"),
                on_support_ema=float(st.on_support_ema),
                on_soft_votes_sum=int(sum(st.on_soft_votes)),
                on_soft_votes_n=int(len(st.on_soft_votes)),
                c_acf=float(c_acf),
                c_spec=float(c_spec),
                c_env=float(c_env),
                c_fft=float(c_fft),
                c_freq_agree=float(c_freq_agree),
                f_welch=float(f_welch),
                f_zc=float(f_zc),
                f_fft=float(f_fft),
                A_tail=float("nan"),
                D_tail=float("nan"),
                rms_decay=float(rms_decay),
                rms_decay_r2=float(rms_decay_r2),
                rms_decay_n=int(rms_decay_n),
                rms_decay_on_ok=int(rms_decay_on_ok),
                rms_decay_off_hint=int(rms_decay_off_hint),
                rms_decay_event=float(rms_decay_event),
                rms_decay_event_r2=float(rms_decay_event_r2),
                rms_decay_event_n=int(rms_decay_event_n),
                rms_decay_event_win_sec=float(rms_decay_event_win_sec),
                reason="stream_end",
                transition_reason="stream_end_open",
                evidence=float(st.evidence),
                long_ratio_on=float("nan"),
                long_ratio_off=float("nan"),
                long_ratio_off_recent=float("nan"),
                long_off_n_recent=0,
                long_off_votes_sum=int(sum(st.long_off_votes)),
                long_off_votes_n=int(len(st.long_off_votes)),
                acf_peak=float(acf_peak),
                acf_period_sec=float(acf_period_sec),
                acf_lag_steps=int(acf_lag_steps),
                acf_n=int(acf_n),
                long_zmax=float("nan"),
                long_n=int(len(st.long_score_hist)),
                baseline_n=int(len(st.long_baseline_hist)),
                emit_legacy_rms_aliases=bool(emit_legacy_rms_aliases),
            ),
            "end_reason": "stream_end_open",
        }
        events.append(ev)
        print(
            f"[INTERVAL_OPEN] dev={key[0]} | ch={key[1]} | start={start_t:.3f}(upd={start_idx:03d}) | "
            f"end={end_t:.3f}(stream_end) | duration={duration_sec:.3f}s"
        )
        if np.isfinite(start_t):
            _, next_interval_id = _emit_or_stitch_interval(
                key=key,
                start_t=float(start_t),
                start_update_idx=int(start_idx),
                end_t=float(end_t),
                end_update_idx=int(len(batches)),
                end_reason="stream_end_open",
                status="open",
                events=events,
                last_interval_by_key=last_interval_by_key,
                next_interval_id=int(next_interval_id),
                stitch_gap_sec=float(stitch_gap_sec),
            )
        if on_event is not None:
            on_event(ev)

    raw_risk_interval_exists = int(raw_risk_interval_count > 0)
    suppressed_event_exists = int(suppressed_interval_count > 0)
    print(
        f"[STREAM_DONE] updates={len(batches)} | events={len(events)} | "
        f"raw_intervals={raw_risk_interval_count} | suppressed_intervals={suppressed_interval_count} | "
        f"raw_risk_interval_exists={raw_risk_interval_exists} | suppressed_event_exists={suppressed_event_exists}"
    )
    return events


def _run_streaming_alert_demo_one_channel_impl(
    vcsv: str,
    *,
    cfg: DetectorConfig | None = None,
    device: str = DEFAULT_DEVICE,
    target_channel: str | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Internal config-first impl boundary for streaming detector execution."""

    cfg_eff = copy.deepcopy(cfg) if cfg is not None else DetectorConfig()
    s = cfg_eff.stream
    th = cfg_eff.threshold
    lg = cfg_eff.long
    pq = cfg_eff.periodicity

    _warn_deprecated_warmup_params(warmup_on_conf_raw_min=float(lg.warmup_on_conf_raw_min))

    return _run_streaming_alert_demo_one_channel_flat_impl(
        vcsv,
        device=str(device),
        target_channel=target_channel,
        update_sec=float(s.update_sec),
        window_sec=float(s.window_sec),
        risk_cut=th.risk_cut,
        risk_cut_on=th.risk_cut_on,
        risk_cut_off=th.risk_cut_off,
        short_dynamic_cut_enabled=bool(th.short_dynamic_cut_enabled),
        short_dynamic_min_baseline=int(th.short_dynamic_min_baseline),
        short_dynamic_on_z=float(th.short_dynamic_on_z),
        short_dynamic_off_z=float(th.short_dynamic_off_z),
        short_dynamic_on_max_mult=float(th.short_dynamic_on_max_mult),
        short_dynamic_off_max_mult=float(th.short_dynamic_off_max_mult),
        short_damped_cut_relax=float(th.short_damped_cut_relax),
        rms_decay_gate_enabled=bool(th.rms_decay_gate_enabled),
        rms_decay_window_sec=float(th.rms_decay_window_sec),
        rms_decay_rms_win_sec=float(th.rms_decay_rms_win_sec),
        rms_decay_step_sec=float(th.rms_decay_step_sec),
        rms_decay_min_windows=int(th.rms_decay_min_windows),
        rms_decay_on_max=float(th.rms_decay_on_max),
        rms_decay_off_min=float(th.rms_decay_off_min),
        rms_decay_cache_enabled=bool(th.rms_decay_cache_enabled),
        rms_decay_cache_refresh_sec=float(th.rms_decay_cache_refresh_sec),
        rms_decay_event_enabled=bool(th.rms_decay_event_enabled),
        rms_decay_event_max_window_sec=float(th.rms_decay_event_max_window_sec),
        rms_decay_event_min_window_sec=float(th.rms_decay_event_min_window_sec),
        rms_decay_event_rms_win_sec=float(th.rms_decay_event_rms_win_sec),
        rms_decay_event_step_sec=float(th.rms_decay_event_step_sec),
        rms_decay_event_min_windows=int(th.rms_decay_event_min_windows),
        realtime_sleep=bool(s.realtime_sleep),
        print_tick=bool(s.print_tick),
        on_event=on_event,
        evidence_alpha=float(th.evidence_alpha),
        evidence_clip=float(th.evidence_clip),
        theta_on=float(th.theta_on),
        theta_off=float(th.theta_off),
        on_consecutive_required=int(th.on_consecutive_required),
        on_short_votes_window=int(th.on_short_votes_window),
        on_confirm_min_sec=float(th.on_confirm_min_sec),
        on_accel_score_log_min=float(th.on_accel_score_log_min),
        on_accel_evidence_min=float(th.on_accel_evidence_min),
        on_require_accel_for_candidate=bool(th.on_require_accel_for_candidate),
        damped_force_off_streak=int(th.damped_force_off_streak),
        off_hold_down_sec=float(th.off_hold_down_sec),
        off_confirm_min_sec=float(th.off_confirm_min_sec),
        off_periodicity_collapse_conf_raw_max=float(th.off_periodicity_collapse_conf_raw_max),
        off_periodicity_collapse_streak_required=int(th.off_periodicity_collapse_streak_required),
        excluded_damped_evidence_penalty=float(th.excluded_damped_evidence_penalty),
        excluded_damped_hard_penalty_streak=int(th.excluded_damped_hard_penalty_streak),
        force_off_require_long_not_on=bool(th.force_off_require_long_not_on),
        force_off_long_on_ratio=(None if th.force_off_long_on_ratio is None else float(th.force_off_long_on_ratio)),
        long_window_sec=float(lg.long_window_sec),
        long_min_points=int(lg.long_min_points),
        long_min_baseline=int(lg.long_min_baseline),
        long_baseline_max=int(lg.long_baseline_max),
        long_z_on=float(lg.long_z_on),
        long_on_ratio=float(lg.long_on_ratio),
        warmup_long_enabled=bool(lg.warmup_long_enabled),
        warmup_on_min_points=int(lg.warmup_on_min_points),
        warmup_min_baseline=int(lg.warmup_min_baseline),
        warmup_on_z=float(lg.warmup_on_z),
        warmup_on_ratio=float(lg.warmup_on_ratio),
        warmup_on_votes_required=int(lg.warmup_on_votes_required),
        warmup_on_votes_window=int(lg.warmup_on_votes_window),
        warmup_on_confirm_min_sec=float(lg.warmup_on_confirm_min_sec),
        warmup_handoff_grace_ticks=int(lg.warmup_handoff_grace_ticks),
        warmup_cancel_on_excluded_damped=bool(lg.warmup_cancel_on_excluded_damped),
        long_z_off=float(lg.long_z_off),
        long_off_ratio=float(lg.long_off_ratio),
        long_off_recent_window_sec=float(lg.long_off_recent_window_sec),
        long_off_votes_required=int(lg.long_off_votes_required),
        long_off_votes_window=int(lg.long_off_votes_window),
        long_off_recent_min_points=int(lg.long_off_recent_min_points),
        acf_min_points=int(pq.acf_min_points),
        acf_min_period_sec=float(pq.acf_min_period_sec),
        acf_max_period_sec=float(pq.acf_max_period_sec),
        confidence_on_min=float(pq.confidence_on_min),
        confidence_off_max=float(pq.confidence_off_max),
        confidence_use_calibration=bool(pq.confidence_use_calibration),
        confidence_dual_gate_when_cal=bool(pq.confidence_dual_gate_when_cal),
        confidence_raw_on_min_when_cal=float(pq.confidence_raw_on_min_when_cal),
        confidence_raw_off_max_when_cal=float(pq.confidence_raw_off_max_when_cal),
        confidence_cal_min_points=int(pq.confidence_cal_min_points),
        confidence_cal_hist_max=int(pq.confidence_cal_hist_max),
        confidence_cal_off_only=bool(pq.confidence_cal_off_only),
        cal_on_soft_mode=bool(pq.cal_on_soft_mode),
        cal_on_noise_scale_low=float(pq.cal_on_noise_scale_low),
        cal_on_noise_scale_high=float(pq.cal_on_noise_scale_high),
        cal_on_conf_adapt_gain=float(pq.cal_on_conf_adapt_gain),
        cal_on_conf_floor=float(pq.cal_on_conf_floor),
        cal_on_conf_ceil=float(pq.cal_on_conf_ceil),
        cal_on_support_score_w=float(pq.cal_on_support_score_w),
        cal_on_support_score_acf_bonus=float(pq.cal_on_support_score_acf_bonus),
        cal_on_support_long_w=float(pq.cal_on_support_long_w),
        cal_on_support_conf_w=float(pq.cal_on_support_conf_w),
        cal_on_support_accel_w=float(pq.cal_on_support_accel_w),
        cal_on_support_enter_min=float(pq.cal_on_support_enter_min),
        cal_on_support_confirm_min=float(pq.cal_on_support_confirm_min),
        cal_on_support_hold_min=float(pq.cal_on_support_hold_min),
        cal_on_support_ema_alpha=float(pq.cal_on_support_ema_alpha),
        cal_on_confirm_votes_required=int(pq.cal_on_confirm_votes_required),
        cal_on_confirm_votes_window=int(pq.cal_on_confirm_votes_window),
        confidence_w_acf=float(pq.confidence_w_acf),
        confidence_w_spec=float(pq.confidence_w_spec),
        confidence_w_env=float(pq.confidence_w_env),
        confidence_w_fft=float(pq.confidence_w_fft),
        freq_band_low_hz=float(pq.freq_band_low_hz),
        freq_band_high_hz=float(pq.freq_band_high_hz),
        freq_linear_detrend=bool(pq.freq_linear_detrend),
        freq_ar1_whiten=bool(pq.freq_ar1_whiten),
        periodicity_quality_cache_enabled=bool(pq.periodicity_quality_cache_enabled),
        periodicity_quality_cache_refresh_sec=float(pq.periodicity_quality_cache_refresh_sec),
        baseline_include_quiet_on=bool(lg.baseline_include_quiet_on),
        baseline_quiet_on_max_sec=float(lg.baseline_quiet_on_max_sec),
        min_interval_sec_for_alert=float(s.min_interval_sec_for_alert),
        stitch_gap_sec=float(s.stitch_gap_sec),
        emit_legacy_rms_aliases=bool(s.emit_legacy_rms_aliases),
    )


def run_streaming_alert_demo_one_channel(
    vcsv: str,
    *,
    cfg: DetectorConfig | None = None,
    device: str = DEFAULT_DEVICE,
    target_channel: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    **overrides,
) -> list[dict]:
    """
    Public runner with compact config-first interface.

    Backward compatibility:
    - legacy kwargs (e.g., `risk_cut`, `long_window_sec`, `confidence_use_calibration`)
      are accepted via `**overrides` and mapped into `DetectorConfig`.
    """
    cfg_eff = copy.deepcopy(cfg) if cfg is not None else DetectorConfig()
    if overrides:
        cfg_eff = _apply_detector_overrides(cfg_eff, overrides)
    return _run_streaming_alert_demo_one_channel_impl(
        vcsv,
        cfg=cfg_eff,
        device=str(device),
        target_channel=target_channel,
        on_event=on_event,
    )


def _guess_default_stream_csv() -> str | None:
    """Pick a default stream CSV path so `python module.py` runs immediately."""
    candidates: list[str] = []

    for key in ("STREAM_INPUT_CSV", "SAMPLE_STREAM_CSV", "STREAM_SAMPLE_CSV"):
        v = getattr(cfg, key, None)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    env_csv = os.environ.get("STREAM_INPUT_CSV", "").strip()
    if env_csv:
        candidates.append(env_csv)

    sample_dir = r"C:\Users\pspo\Desktop\pspo\DB\sample_data"
    candidates.extend(
        [
            os.path.join(sample_dir, "wmu_voltage_random_1min_256Hz.csv"),
            os.path.join(sample_dir, "wmu_voltage_mock_5min_256Hz.csv"),
        ]
    )

    csv_dir = getattr(cfg, "CSV_DIR", None)
    if isinstance(csv_dir, str) and csv_dir.strip() and os.path.isdir(csv_dir):
        for p in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
            candidates.append(p)

    seen = set()
    for p in candidates:
        ap = os.path.abspath(str(p))
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            return ap
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for the streaming detector runner."""

    p = argparse.ArgumentParser(description="Streaming oscillation detector runner (minimal CLI)")
    p.add_argument("--csv", default="", help="Input CSV path. If omitted, auto-discovery is used.")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="Device id label")
    p.add_argument("--channel", default="", help="Target channel name (optional)")
    p.add_argument("--preset", choices=PRESET_CHOICES, default=PRESET_SAFE, help="Detector preset profile")
    p.add_argument("--update-sec", type=float, default=DEFAULT_UPDATE_SEC, help="Stream update period in seconds")
    p.add_argument("--window-sec", type=float, default=WIN_SEC, help="Sliding evaluation window in seconds")
    p.add_argument("--realtime-sleep", action="store_true", help="Sleep each update_sec during replay")
    p.add_argument("--no-print-tick", action="store_true", help="Disable per-tick logs")
    p.add_argument("--emit-legacy-rms-aliases", action="store_true", help="Include legacy rms_decay* alias keys in emitted event payloads")
    p.add_argument("--min-interval-sec-for-alert", type=float, default=None, help="Minimum ON duration to emit risk_on/risk_off interval events")
    p.add_argument("--risk-cut", type=float, default=None, help="Base score cutoff override")
    p.add_argument("--no-rms-decay-gate", action="store_true", help="Disable RMS-decay gate/hint integration")
    p.add_argument("--rms-decay-window-sec", type=float, default=None, help="Trailing window length for RMS-decay estimate")
    p.add_argument("--rms-decay-on-max", type=float, default=None, help="ON entry allowed maximum RMS decay")
    p.add_argument("--rms-decay-off-min", type=float, default=None, help="OFF hint minimum RMS decay")
    p.add_argument("--confidence-use-calibration", action="store_true", help="Use quantile-calibrated confidence for ON/OFF gating")
    p.add_argument("--confidence-w-acf", type=float, default=None, help="Override ACF weight in confidence fusion")
    p.add_argument("--cal-on-support-score-acf-bonus", type=float, default=None, help="Additive ACF bonus for CAL_ON score-support term (0~1)")
    p.add_argument("--no-cal-on-soft-mode", action="store_true", help="Disable adaptive/soft CAL_ON gating and use fixed calibrated confidence gate")
    return p


def main() -> None:
    """CLI entrypoint: build config, run detector, and print summary."""

    args = _build_arg_parser().parse_args()

    csv_path = str(args.csv).strip() if str(args.csv).strip() else _guess_default_stream_csv()
    if not csv_path or (not os.path.isfile(csv_path)):
        raise FileNotFoundError(
            "No stream CSV found. Provide --csv or set config/env STREAM_INPUT_CSV."
        )

    print(f"[MAIN] csv={os.path.abspath(csv_path)}")
    print(f"[MAIN] device={args.device} | update_sec={args.update_sec} | preset={args.preset}")

    cfg = make_preset_config(str(args.preset))
    cfg.stream.update_sec = float(args.update_sec)
    cfg.stream.window_sec = float(args.window_sec)
    cfg.stream.realtime_sleep = bool(args.realtime_sleep)
    cfg.stream.print_tick = (not bool(args.no_print_tick))
    cfg.stream.emit_legacy_rms_aliases = bool(args.emit_legacy_rms_aliases)
    if args.min_interval_sec_for_alert is not None:
        cfg.stream.min_interval_sec_for_alert = float(args.min_interval_sec_for_alert)
    if args.risk_cut is not None:
        cfg.threshold.risk_cut = float(args.risk_cut)
    if bool(args.no_rms_decay_gate):
        cfg.threshold.rms_decay_gate_enabled = False
    if args.rms_decay_window_sec is not None:
        cfg.threshold.rms_decay_window_sec = float(args.rms_decay_window_sec)
    if args.rms_decay_on_max is not None:
        cfg.threshold.rms_decay_on_max = float(args.rms_decay_on_max)
    if args.rms_decay_off_min is not None:
        cfg.threshold.rms_decay_off_min = float(args.rms_decay_off_min)
    if bool(args.confidence_use_calibration):
        cfg.periodicity.confidence_use_calibration = True
    if args.confidence_w_acf is not None:
        cfg.periodicity.confidence_w_acf = float(args.confidence_w_acf)
    if args.cal_on_support_score_acf_bonus is not None:
        cfg.periodicity.cal_on_support_score_acf_bonus = float(args.cal_on_support_score_acf_bonus)
    if bool(args.no_cal_on_soft_mode):
        cfg.periodicity.cal_on_soft_mode = False

    events = run_streaming_alert_demo_one_channel(
        csv_path,
        cfg=cfg,
        device=str(args.device),
        target_channel=(str(args.channel).strip() if str(args.channel).strip() else None),
    )
    print(f"[MAIN] done | events={len(events)}")


if __name__ == "__main__":
    main()

