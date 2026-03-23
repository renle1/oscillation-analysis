"""Signal/math core functions for modular streaming detector."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd

from .osc_state_modul import PHASE_OFF_CANDIDATE, PHASE_ON_CONFIRMED

# Signal scoring constants (aligned with OSC_streaming.py)
BASE_SEC = 6.0
RMS_WIN_SEC = 0.25
TAIL_FRAC = 0.30
MIN_SIGN_CHANGES = 3
DAMPED_RATIO_CUT = 0.70
DAMPED_SCORE_PENALTY_FLOOR = 0.35
EPS_AMP = 1e-6

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


def _clean_time_value_pairs(
    timestamps: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite `(t, v)` arrays sorted by time with duplicate timestamps removed."""

    t = np.asarray(timestamps, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if t.size != v.size:
        raise ValueError("timestamps/values size mismatch")
    if t.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    m = np.isfinite(t) & np.isfinite(v)
    if int(np.count_nonzero(m)) <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    t = t[m]
    v = v[m]
    if t.size <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]
    if t.size <= 1:
        return t.astype(float, copy=False), v.astype(float, copy=False)

    dt = np.diff(t)
    keep = np.concatenate(([True], dt > 1e-12))
    return t[keep].astype(float, copy=False), v[keep].astype(float, copy=False)


def _prepare_uniform_signal_for_band_energy(
    timestamps: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    min_samples: int,
    dt_cv_max: float,
    detrend_linear: bool,
) -> tuple[np.ndarray, float]:
    """Prepare finite near-uniform signal for FFT band-energy computation."""

    t, v = _clean_time_value_pairs(timestamps, values)
    if t.size < int(max(2, min_samples)):
        return np.zeros(0, dtype=float), float("nan")

    dt_pos = np.diff(t)
    dt_pos = dt_pos[np.isfinite(dt_pos) & (dt_pos > 0.0)]
    if dt_pos.size <= 0:
        return np.zeros(0, dtype=float), float("nan")
    dt_med = float(np.median(dt_pos))
    if (not np.isfinite(dt_med)) or (dt_med <= 0.0):
        return np.zeros(0, dtype=float), float("nan")

    dt_cv = float(np.std(dt_pos) / max(dt_med, 1e-12)) if dt_pos.size > 1 else 0.0
    if np.isfinite(dt_cv) and (dt_cv > float(max(0.0, dt_cv_max))):
        n_uniform = int(max(2, np.floor((float(t[-1]) - float(t[0])) / dt_med) + 1))
        t_uni = float(t[0]) + (np.arange(n_uniform, dtype=float) * dt_med)
        t_uni = t_uni[t_uni <= (float(t[-1]) + (0.5 * dt_med))]
        if t_uni.size < int(max(2, min_samples)):
            return np.zeros(0, dtype=float), float("nan")
        v = np.interp(t_uni, t, v)
        t = t_uni

    if t.size < int(max(2, min_samples)):
        return np.zeros(0, dtype=float), float("nan")

    x = np.asarray(v, dtype=float)
    x = x - float(np.nanmedian(x))
    if bool(detrend_linear) and (x.size >= 3) and (float(np.ptp(t)) > 0.0):
        p = np.polyfit(t, x, deg=1)
        x = x - (float(p[0]) * t + float(p[1]))
    else:
        x = x - float(np.nanmean(x))
    return x.astype(float, copy=False), float(dt_med)


def _band_rms_from_rfft(freqs_hz: np.ndarray, spectrum: np.ndarray, n_samples: int, freq_low_hz: float, freq_high_hz: float) -> float:
    """Compute time-domain RMS-equivalent energy in one FFT frequency band."""

    f_lo = float(freq_low_hz)
    f_hi = float(freq_high_hz)
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (f_hi <= f_lo):
        return float("nan")
    if int(n_samples) <= 1:
        return float("nan")
    if freqs_hz.size != spectrum.size:
        return float("nan")

    m = np.isfinite(freqs_hz) & (freqs_hz >= f_lo) & (freqs_hz <= f_hi)
    if int(np.count_nonzero(m)) <= 0:
        return float("nan")

    mag2 = np.abs(spectrum) ** 2.0
    weights = np.full(spectrum.shape, 2.0, dtype=float)
    if weights.size > 0:
        weights[0] = 1.0
    if (int(n_samples) % 2) == 0 and (weights.size > 1):
        weights[-1] = 1.0

    power = float(np.sum(weights[m] * mag2[m]) / float(int(n_samples) * int(n_samples)))
    if (not np.isfinite(power)) or (power < 0.0):
        return float("nan")
    return float(np.sqrt(max(0.0, power)))


def compute_band_rms_energies(
    timestamps: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    bands: Sequence[tuple[str, float, float]],
    min_samples: int = 16,
    dt_cv_max: float = 0.25,
    detrend_linear: bool = True,
) -> dict[str, float]:
    """Compute RMS-energy for multiple frequency bands from one signal window."""

    out: dict[str, float] = {}
    for b in bands:
        if isinstance(b, tuple) and len(b) >= 1:
            out[str(b[0])] = float("nan")
    if not bands:
        return out

    x, dt = _prepare_uniform_signal_for_band_energy(
        timestamps,
        values,
        min_samples=int(max(2, min_samples)),
        dt_cv_max=float(max(0.0, dt_cv_max)),
        detrend_linear=bool(detrend_linear),
    )
    if (x.size < int(max(2, min_samples))) or (not np.isfinite(dt)) or (dt <= 0.0):
        return out

    n = int(x.size)
    spectrum = np.fft.rfft(x.astype(float, copy=False))
    freqs_hz = np.fft.rfftfreq(n, d=float(dt))
    for b in bands:
        if (not isinstance(b, tuple)) or (len(b) < 3):
            continue
        name = str(b[0])
        out[name] = _band_rms_from_rfft(
            freqs_hz=freqs_hz,
            spectrum=spectrum,
            n_samples=int(n),
            freq_low_hz=float(b[1]),
            freq_high_hz=float(b[2]),
        )
    return out


def compute_band_rms_energy(
    timestamps: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    freq_low_hz: float,
    freq_high_hz: float,
    min_samples: int = 16,
    dt_cv_max: float = 0.25,
    detrend_linear: bool = True,
) -> float:
    """Compute RMS-energy for one frequency band from one signal window."""

    out = compute_band_rms_energies(
        timestamps,
        values,
        bands=(("band", float(freq_low_hz), float(freq_high_hz)),),
        min_samples=int(max(2, min_samples)),
        dt_cv_max=float(max(0.0, dt_cv_max)),
        detrend_linear=bool(detrend_linear),
    )
    return float(out.get("band", float("nan")))


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



__all__ = [
    "_safe_float",
    "to_float_np",
    "infer_time_col",
    "_clean_time_value_pairs",
    "_prepare_uniform_signal_for_band_energy",
    "_band_rms_from_rfft",
    "compute_band_rms_energies",
    "compute_band_rms_energy",
    "duration_over_mask",
    "tail_sign_changes",
    "score_one_channel_equiv",
    "_estimate_dt_from_timestamps",
    "_trim_ring_by_time",
    "_trim_score_hist_by_time",
    "_rolling_mean_trailing_np",
    "_local_rms_decay_from_signal",
    "_event_rms_decay_from_signal",
    "_robust_center_scale",
    "_dynamic_score_cut_from_log_baseline",
    "_empirical_cdf_rank",
    "_calibrate_confidence_quantile",
    "_long_activity_ratio",
    "_acf_peak_periodicity",
    "_clip01",
    "_normalize01",
    "_preprocess_for_periodicity",
    "_welch_peak_and_concentration",
    "_zero_cross_frequency",
    "_acf_peak_from_signal",
    "_envelope_stability_score",
    "_fft_peak_and_concentration",
    "_compute_periodicity_quality",
]
