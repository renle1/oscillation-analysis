"""Matrix-Pencil post analysis functions for modular streaming detector."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Callable, Sequence

import numpy as np

from .osc_core_postprep_modul import apply_modal_postprep, default_postprep_summary
from .osc_core_signal_modul import (
    _estimate_dt_from_timestamps,
    _local_rms_decay_from_signal,
    _trim_ring_by_time,
)
from .osc_state_modul import MP_RUNTIME_MODE_LIVE, MP_RUNTIME_MODE_REPLAY, ChannelStreamState


def _damping_ratio_from(freq_hz: float, damping_per_sec: float) -> float:
    """Convert mode `(freq_hz, damping_per_sec)` into damping ratio."""

    f = float(freq_hz)
    d = float(damping_per_sec)
    if (not np.isfinite(f)) or (f <= 0.0) or (not np.isfinite(d)) or (d < 0.0):
        return float("nan")
    omega = float(2.0 * np.pi * f)
    denom = float(np.sqrt((d * d) + (omega * omega)))
    if (not np.isfinite(denom)) or (denom <= 0.0):
        return float("nan")
    zeta = float(d / denom)
    return float(zeta) if np.isfinite(zeta) else float("nan")


def _build_mp_interval_record_base(interval_ev: dict) -> dict[str, object]:
    """Build common interval metadata for post-analysis records."""

    return {
        "event": "interval_analysis_mp",
        "analysis_type": "matrix_pencil_post",
        "interval_id": int(interval_ev.get("interval_id", -1)),
        "device": str(interval_ev.get("device", "")),
        "channel": str(interval_ev.get("channel", "")),
        "start_t": float(interval_ev.get("start_t", np.nan)),
        "start_update_idx": int(interval_ev.get("start_update_idx", -1)),
        "end_t": float(interval_ev.get("end_t", np.nan)),
        "end_update_idx": int(interval_ev.get("end_update_idx", -1)),
        "duration_sec": float(interval_ev.get("duration_sec", np.nan)),
        "interval_status": str(interval_ev.get("status", "")),
        "interval_end_reason": str(interval_ev.get("end_reason", "")),
        "interval_stitch_count": int(interval_ev.get("stitch_count", 0)),
    }


def _merge_mp_record_sections(*sections: dict[str, object]) -> dict[str, object]:
    """Merge semantic sections into one flat MP payload dict."""

    merged: dict[str, object] = {}
    for section in sections:
        merged.update(section)
    return merged


def _apply_primary_mode_aliases(rec: dict[str, object]) -> dict[str, object]:
    """Expose explicit primary-mode aliases while preserving legacy MP keys."""

    dominant_freq_hz = float(rec.get("mp_dominant_freq_hz", np.nan))
    dominant_signed_rate = float(
        rec.get(
            "mp_primary_mode_signed_rate_per_sec",
            rec.get("mp_primary_mode_rate_per_sec", rec.get("mp_dominant_damping_per_sec", np.nan)),
        )
    )
    rec["mp_primary_mode_freq_hz"] = float(dominant_freq_hz)
    rec["mp_primary_mode_signed_rate_per_sec"] = float(dominant_signed_rate)
    rec["mp_primary_mode_rate_per_sec"] = float(dominant_signed_rate)
    return rec


def _get_primary_mode_signed_rate_per_sec(src: dict[str, object]) -> float:
    """Read preferred primary-mode signed-rate alias with legacy fallback."""

    return float(
        src.get(
            "mp_primary_mode_signed_rate_per_sec",
            src.get("mp_primary_mode_rate_per_sec", src.get("mp_dominant_damping_per_sec", np.nan)),
        )
    )


def _select_mp_analysis_window(
    *,
    start_t: float,
    end_t: float,
    duration_sec: float,
    rms_event_win_sec: float,
    mp_onset_skip_sec: float,
    mp_onset_window_sec: float,
    mp_onset_min_window_sec: float,
    mp_fallback_use_rms_event_window: bool,
    mp_fallback_default_window_sec: float,
) -> tuple[str, float, float]:
    """Pick onset-first analysis window, then fallback windows."""

    onset_start = float(start_t + max(0.0, float(mp_onset_skip_sec)))
    onset_end = float(min(end_t, onset_start + max(0.0, float(mp_onset_window_sec))))
    if np.isfinite(onset_start) and np.isfinite(onset_end) and ((onset_end - onset_start) >= float(mp_onset_min_window_sec)):
        return "onset", float(onset_start), float(onset_end)

    if bool(mp_fallback_use_rms_event_window) and np.isfinite(rms_event_win_sec) and (float(rms_event_win_sec) > 0.0):
        fb_sec = float(min(float(duration_sec), float(rms_event_win_sec)))
        fb_start = float(max(start_t, end_t - fb_sec))
        if (end_t - fb_start) >= float(mp_onset_min_window_sec):
            return "rms_event_fallback", float(fb_start), float(end_t)

    fb_tail_sec = float(min(float(duration_sec), max(0.0, float(mp_fallback_default_window_sec))))
    fb_tail_start = float(max(start_t, end_t - fb_tail_sec))
    if (end_t - fb_tail_start) >= float(mp_onset_min_window_sec):
        return "tail_fallback", float(fb_tail_start), float(end_t)

    return "none", float("nan"), float("nan")


def _adaptive_profile_from_freq(
    freq_hz: float,
    *,
    low_band_max_hz: float,
    high_band_min_hz: float,
) -> str:
    """Map dominant frequency to adaptive profile: high / mid / low."""

    f = float(freq_hz)
    if np.isfinite(f):
        if f >= float(high_band_min_hz):
            return "high"
        if f < float(low_band_max_hz):
            return "low"
    return "mid"


def _window_sec_for_profile(
    profile: str,
    *,
    high_sec: float,
    mid_sec: float,
    low_sec: float,
) -> float:
    """Return profile-specific window seconds with finite fallback."""

    p = str(profile).strip().lower()
    if p == "high":
        out = float(high_sec)
    elif p == "low":
        out = float(low_sec)
    else:
        out = float(mid_sec)
    if (not np.isfinite(out)) or (out <= 0.0):
        out = float(mid_sec)
    if (not np.isfinite(out)) or (out <= 0.0):
        out = 12.0
    return float(out)


def _select_tail_analysis_window(
    *,
    start_t: float,
    end_t: float,
    duration_sec: float,
    tail_window_sec: float,
    min_window_sec: float,
) -> tuple[str, float, float]:
    """Select trailing window from event tail for damping analysis."""

    tail_sec = float(min(float(duration_sec), max(0.0, float(tail_window_sec))))
    w0 = float(max(float(start_t), float(end_t) - tail_sec))
    w1 = float(end_t)
    if np.isfinite(w0) and np.isfinite(w1) and ((w1 - w0) >= float(min_window_sec)):
        return "tail", float(w0), float(w1)
    return "none", float("nan"), float("nan")


def _clean_mp_order_candidates(mp_model_order: int, mp_order_candidates: Sequence[int] | None) -> list[int]:
    """Normalize rank-candidate sequence while preserving user order."""

    raw: list[int] = []
    if mp_order_candidates is not None:
        for x in mp_order_candidates:
            try:
                raw.append(int(x))
            except Exception:
                continue
    if not raw:
        raw = [int(mp_model_order)]

    out: list[int] = []
    seen: set[int] = set()
    for x in raw:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _uniform_time_axis(t0: float, t1: float, dt: float) -> np.ndarray:
    """Build inclusive uniform axis from `t0` to `t1` at step `dt`."""

    n_uniform = int(max(2, np.floor((float(t1) - float(t0)) / float(dt)) + 1))
    t_uni = float(t0) + (np.arange(n_uniform, dtype=float) * float(dt))
    return t_uni[t_uni <= (float(t1) + 0.5 * float(dt))]


def _downsample_mp_window_if_needed(
    t_arr: np.ndarray,
    v_arr: np.ndarray,
    *,
    enabled: bool,
    target_fs_hz: float,
    lpf_cutoff_hz: float,
    lpf_order: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Optionally apply LPF+uniform resampling before MP factorization."""

    t_raw = np.asarray(t_arr, dtype=float).reshape(-1)
    v_raw = np.asarray(v_arr, dtype=float).reshape(-1)
    try:
        order_eff = int(max(1, int(lpf_order)))
    except Exception:
        order_eff = 1

    summary: dict[str, object] = {
        "mp_downsample_enabled": bool(enabled),
        "mp_downsample_applied": 0,
        "mp_downsample_reason": "disabled",
        "mp_downsample_source_fs_hz": float("nan"),
        "mp_downsample_target_fs_hz": float(target_fs_hz) if np.isfinite(float(target_fs_hz)) else float("nan"),
        "mp_downsample_lpf_cutoff_hz": float(lpf_cutoff_hz) if np.isfinite(float(lpf_cutoff_hz)) else float("nan"),
        "mp_downsample_lpf_order": int(order_eff),
        "mp_downsample_n_samples_in": int(t_raw.size),
        "mp_downsample_n_samples_out": int(t_raw.size),
    }
    if not bool(enabled):
        return t_raw, v_raw, summary

    if t_raw.size != v_raw.size:
        summary["mp_downsample_reason"] = "shape_mismatch"
        return t_raw, v_raw, summary

    m = np.isfinite(t_raw) & np.isfinite(v_raw)
    t = t_raw[m]
    v = v_raw[m]
    if t.size < int(max(8, min_samples)):
        summary.update(
            {
                "mp_downsample_reason": "too_few_samples",
                "mp_downsample_n_samples_in": int(t.size),
                "mp_downsample_n_samples_out": int(t.size),
            }
        )
        return t_raw, v_raw, summary

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]
    keep = np.r_[True, np.diff(t) > 0.0]
    t = t[keep]
    v = v[keep]
    if t.size < int(max(8, min_samples)):
        summary.update(
            {
                "mp_downsample_reason": "too_few_unique_samples",
                "mp_downsample_n_samples_in": int(t.size),
                "mp_downsample_n_samples_out": int(t.size),
            }
        )
        return t_raw, v_raw, summary

    dt_pos = np.diff(t)
    dt_pos = dt_pos[np.isfinite(dt_pos) & (dt_pos > 0.0)]
    if dt_pos.size < 2:
        summary.update(
            {
                "mp_downsample_reason": "invalid_dt",
                "mp_downsample_n_samples_in": int(t.size),
                "mp_downsample_n_samples_out": int(t.size),
            }
        )
        return t_raw, v_raw, summary

    dt_src = float(np.median(dt_pos))
    if (not np.isfinite(dt_src)) or (dt_src <= 0.0):
        summary.update(
            {
                "mp_downsample_reason": "invalid_dt",
                "mp_downsample_n_samples_in": int(t.size),
                "mp_downsample_n_samples_out": int(t.size),
            }
        )
        return t_raw, v_raw, summary

    source_fs_hz = float(1.0 / dt_src)
    summary["mp_downsample_source_fs_hz"] = float(source_fs_hz)
    summary["mp_downsample_n_samples_in"] = int(t.size)

    target_fs = float(target_fs_hz)
    if (not np.isfinite(target_fs)) or (target_fs <= 0.0):
        summary["mp_downsample_reason"] = "invalid_target_fs"
        return t_raw, v_raw, summary
    if source_fs_hz <= (target_fs * 1.01):
        summary["mp_downsample_reason"] = "source_fs_not_higher_than_target"
        return t_raw, v_raw, summary

    t_uniform = _uniform_time_axis(float(t[0]), float(t[-1]), float(dt_src))
    if t_uniform.size < int(max(8, min_samples)):
        summary["mp_downsample_reason"] = "too_few_samples_after_uniformize"
        return t_raw, v_raw, summary
    x_uniform = np.interp(t_uniform, t, v)

    nyq_source = 0.5 * float(source_fs_hz)
    nyq_target = 0.5 * float(target_fs)
    cutoff_eff = float(min(float(lpf_cutoff_hz), 0.95 * nyq_source, 0.95 * nyq_target))
    if (not np.isfinite(cutoff_eff)) or (cutoff_eff <= 0.0):
        summary["mp_downsample_reason"] = "invalid_cutoff"
        return t_raw, v_raw, summary
    summary["mp_downsample_lpf_cutoff_hz"] = float(cutoff_eff)

    taps_n = int(max(15, (8 * int(order_eff)) + 1))
    if (taps_n % 2) == 0:
        taps_n += 1
    max_taps = int(max(3, (2 * int(x_uniform.size)) - 1))
    taps_n = int(min(taps_n, max_taps))
    if (taps_n % 2) == 0:
        taps_n = int(max(3, taps_n - 1))
    if taps_n < 3:
        summary["mp_downsample_reason"] = "too_few_samples_for_lpf"
        return t_raw, v_raw, summary

    n_idx = np.arange(taps_n, dtype=float) - (0.5 * float(taps_n - 1))
    cutoff_norm = float(cutoff_eff / max(source_fs_hz, 1e-12))
    kernel = (2.0 * cutoff_norm) * np.sinc((2.0 * cutoff_norm) * n_idx)
    kernel = kernel * np.hamming(taps_n)
    ksum = float(np.sum(kernel))
    if (not np.isfinite(ksum)) or (abs(ksum) <= 1e-12):
        summary["mp_downsample_reason"] = "invalid_lpf_kernel"
        return t_raw, v_raw, summary
    kernel = kernel / ksum

    pad = int(taps_n // 2)
    pad_mode = "reflect" if (x_uniform.size > 1) and (pad <= (x_uniform.size - 1)) else "edge"
    x_pad = np.pad(x_uniform, (pad, pad), mode=pad_mode)
    x_lpf = np.convolve(x_pad, kernel, mode="valid")
    if x_lpf.size != x_uniform.size:
        summary["mp_downsample_reason"] = "lpf_size_mismatch"
        return t_raw, v_raw, summary

    dt_target = float(1.0 / target_fs)
    t_down = _uniform_time_axis(float(t_uniform[0]), float(t_uniform[-1]), float(dt_target))
    if t_down.size < int(max(8, min_samples)):
        summary["mp_downsample_reason"] = "too_few_samples_after_downsample"
        return t_raw, v_raw, summary
    v_down = np.interp(t_down, t_uniform, x_lpf)

    summary.update(
        {
            "mp_downsample_applied": 1,
            "mp_downsample_reason": "ok",
            "mp_downsample_target_fs_hz": float(target_fs),
            "mp_downsample_n_samples_out": int(t_down.size),
        }
    )
    return t_down, v_down, summary


def _prepare_mp_window_factorization(
    t_win: np.ndarray,
    v_win: np.ndarray,
    *,
    mp_min_samples: int,
    mp_dt_cv_max: float,
    mp_signal_std_min: float,
    mp_singular_ratio_min: float,
) -> dict[str, object]:
    """Prepare MP window arrays and one-time SVD factors shared by rank candidates."""

    t_arr = np.asarray(t_win, dtype=float).reshape(-1)
    v_arr = np.asarray(v_win, dtype=float).reshape(-1)
    m = np.isfinite(t_arr) & np.isfinite(v_arr)
    t_arr = t_arr[m]
    v_arr = v_arr[m]
    if t_arr.size < int(mp_min_samples):
        return {"mp_status": "skipped", "mp_reason": "too_few_samples"}

    order = np.argsort(t_arr, kind="mergesort")
    t_arr = t_arr[order]
    v_arr = v_arr[order]
    dt_all = np.diff(t_arr)
    dt_pos = dt_all[np.isfinite(dt_all) & (dt_all > 0.0)]
    if dt_pos.size < 2:
        return {"mp_status": "skipped", "mp_reason": "invalid_dt"}

    dt = float(np.median(dt_pos))
    if (not np.isfinite(dt)) or (dt <= 0.0):
        return {"mp_status": "skipped", "mp_reason": "invalid_dt"}
    dt_cv = float(np.std(dt_pos) / max(dt, 1e-12))

    if np.isfinite(dt_cv) and (dt_cv > float(mp_dt_cv_max)):
        n_uniform = int(max(2, np.floor((float(t_arr[-1]) - float(t_arr[0])) / dt) + 1))
        t_uni = float(t_arr[0]) + (np.arange(n_uniform, dtype=float) * dt)
        t_uni = t_uni[t_uni <= (float(t_arr[-1]) + 0.5 * dt)]
        if t_uni.size < int(mp_min_samples):
            return {
                "mp_status": "skipped",
                "mp_reason": "too_few_samples_after_resample",
                "mp_dt_sec": float(dt),
                "mp_dt_cv": float(dt_cv),
            }
        v_uni = np.interp(t_uni, t_arr, v_arr)
        t_arr = t_uni
        v_arr = v_uni

    x = np.asarray(v_arr, dtype=float)
    x = x - float(np.mean(x))
    signal_std = float(np.std(x))
    if (not np.isfinite(signal_std)) or (signal_std < float(mp_signal_std_min)):
        return {
            "mp_status": "skipped",
            "mp_reason": "low_signal_std",
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }

    n = int(x.size)
    if n < max(int(mp_min_samples), 8):
        return {
            "mp_status": "skipped",
            "mp_reason": "too_few_samples",
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }

    l = int(max(4, n // 2))
    if (n - l + 1) < 3:
        return {
            "mp_status": "skipped",
            "mp_reason": "window_too_short_for_hankel",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }

    try:
        hankel = np.lib.stride_tricks.sliding_window_view(x, window_shape=l)
        x0 = hankel[:-1].T
        x1 = hankel[1:].T
        u, s, vh = np.linalg.svd(x0, full_matrices=False)
    except Exception:
        return {
            "mp_status": "failed",
            "mp_reason": "svd_failed",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }

    if s.size == 0:
        return {
            "mp_status": "failed",
            "mp_reason": "empty_svd",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }
    s0 = float(s[0])
    if (not np.isfinite(s0)) or (s0 <= 0.0):
        return {
            "mp_status": "failed",
            "mp_reason": "invalid_singular_values",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }

    s_rel = np.asarray(s, dtype=float) / s0
    rank_floor = int(np.count_nonzero(s_rel >= float(max(0.0, mp_singular_ratio_min))))
    rank_floor = int(max(1, rank_floor))
    rank_max = int(min(rank_floor, int(s.size)))
    if rank_max < 1:
        return {
            "mp_status": "failed",
            "mp_reason": "invalid_rank_floor",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
        }

    return {
        "mp_status": "prepared",
        "mp_reason": "ok",
        "x": x,
        "n": int(n),
        "dt": float(dt),
        "dt_cv": float(dt_cv),
        "signal_std": float(signal_std),
        "x1": x1,
        "u": u,
        "s": s,
        "vh": vh,
        "s0": float(s0),
        "rank_floor": int(rank_floor),
        "rank_max": int(rank_max),
    }


def _run_matrix_pencil_rank_candidate(
    prepared: dict[str, object],
    rank_used: int,
    *,
    mp_max_modes: int,
    mp_freq_low_hz: float,
    mp_freq_high_hz: float,
) -> dict[str, object]:
    """Evaluate one rank candidate from precomputed MP factorization."""

    x = np.asarray(prepared.get("x"), dtype=float)
    n = int(prepared.get("n", x.size))
    dt = float(prepared.get("dt", np.nan))
    dt_cv = float(prepared.get("dt_cv", np.nan))
    signal_std = float(prepared.get("signal_std", np.nan))
    x1 = np.asarray(prepared.get("x1"))
    u = np.asarray(prepared.get("u"))
    s = np.asarray(prepared.get("s"), dtype=float)
    vh = np.asarray(prepared.get("vh"))
    s0 = float(prepared.get("s0", np.nan))
    rank_max = int(prepared.get("rank_max", s.size))
    rank_used_eff = int(min(max(1, int(rank_used)), max(1, rank_max), int(s.size)))

    ur = u[:, :rank_used_eff]
    sr = s[:rank_used_eff]
    vr = vh[:rank_used_eff, :].T
    try:
        a_tilde = (ur.T @ x1 @ vr) @ np.diag(1.0 / np.maximum(sr, 1e-12))
        z = np.linalg.eigvals(a_tilde)
    except Exception:
        return {
            "mp_status": "failed",
            "mp_reason": "eig_failed",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
            "mp_rank_used": int(rank_used_eff),
            "mp_singular_ratio": float(sr[-1] / s0) if (sr.size > 0) and np.isfinite(s0) and (s0 > 0.0) else float("nan"),
        }

    if z.size == 0:
        return {
            "mp_status": "skipped",
            "mp_reason": "no_eigenvalues",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
            "mp_rank_used": int(rank_used_eff),
            "mp_singular_ratio": float(sr[-1] / s0) if (sr.size > 0) and np.isfinite(s0) and (s0 > 0.0) else float("nan"),
        }

    z_keep: list[complex] = []
    mode_meta: list[tuple[float, float, float]] = []
    for zi in z:
        mag = float(np.abs(zi))
        if (not np.isfinite(mag)) or (mag <= 1e-12) or (mag > 1.2):
            continue
        freq_hz = float(np.abs(np.angle(zi)) / (2.0 * np.pi * dt))
        if (not np.isfinite(freq_hz)) or (freq_hz < float(mp_freq_low_hz)) or (freq_hz > float(mp_freq_high_hz)):
            continue
        damping = float(-np.log(max(mag, 1e-12)) / dt)
        if not np.isfinite(damping):
            continue
        z_keep.append(complex(zi))
        mode_meta.append((freq_hz, damping, mag))

    if not z_keep:
        return {
            "mp_status": "skipped",
            "mp_reason": "no_modes_in_band",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
            "mp_rank_used": int(rank_used_eff),
            "mp_singular_ratio": float(sr[-1] / s0),
        }

    try:
        n_idx = np.arange(n, dtype=float)
        z_mat = np.power(np.asarray(z_keep, dtype=np.complex128)[None, :], n_idx[:, None])
        coef, *_ = np.linalg.lstsq(z_mat, x.astype(np.complex128), rcond=None)
        x_hat = np.real(z_mat @ coef)
    except Exception:
        return {
            "mp_status": "failed",
            "mp_reason": "amplitude_fit_failed",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
            "mp_rank_used": int(rank_used_eff),
            "mp_singular_ratio": float(sr[-1] / s0),
        }

    sst = float(np.sum((x - float(np.mean(x))) ** 2))
    sse = float(np.sum((x - x_hat) ** 2))
    fit_r2 = float(1.0 - (sse / sst)) if sst > 1e-12 else float("nan")

    modes: list[dict[str, float | int]] = []
    amps = np.abs(coef)
    for i, (freq_hz, damping, mag) in enumerate(mode_meta):
        amp = float(amps[i]) if i < amps.size else float("nan")
        damp_ratio = _damping_ratio_from(float(freq_hz), float(damping))
        modes.append(
            {
                "rank": int(i + 1),
                "freq_hz": float(freq_hz),
                "damping_per_sec": float(damping),
                "damping_ratio": float(damp_ratio),
                "amplitude": float(amp),
                "pole_mag": float(mag),
            }
        )
    modes = sorted(modes, key=lambda m_: float(m_.get("amplitude", np.nan)), reverse=True)
    modes = modes[: max(1, int(mp_max_modes))]
    if not modes:
        return {
            "mp_status": "skipped",
            "mp_reason": "no_modes_after_sort",
            "mp_n_samples": int(n),
            "mp_dt_sec": float(dt),
            "mp_dt_cv": float(dt_cv),
            "mp_signal_std": float(signal_std),
            "mp_rank_used": int(rank_used_eff),
            "mp_singular_ratio": float(sr[-1] / s0),
            "mp_fit_r2": float(fit_r2),
            "mp_modes": [],
            "mp_mode_count": 0,
        }

    dom = modes[0]
    return {
        "mp_status": "ok",
        "mp_reason": "ok",
        "mp_n_samples": int(n),
        "mp_dt_sec": float(dt),
        "mp_dt_cv": float(dt_cv),
        "mp_signal_std": float(signal_std),
        "mp_rank_used": int(rank_used_eff),
        "mp_singular_ratio": float(sr[-1] / s0),
        "mp_fit_r2": float(fit_r2),
        "mp_mode_count": int(len(modes)),
        "mp_dominant_freq_hz": float(dom.get("freq_hz", np.nan)),
        "mp_dominant_damping_per_sec": float(dom.get("damping_per_sec", np.nan)),
        "mp_dominant_damping_ratio": float(dom.get("damping_ratio", np.nan)),
        "mp_dominant_amplitude": float(dom.get("amplitude", np.nan)),
        "mp_modes": modes,
    }


def _is_mp_attempt_plausible(a: dict[str, object], prev: dict[str, object] | None) -> bool:
    """Conservative plausibility check to avoid over-ordering unstable modes."""

    freq = float(a.get("mp_dominant_freq_hz", np.nan))
    damping = _get_primary_mode_signed_rate_per_sec(a)
    damp_ratio = float(a.get("mp_dominant_damping_ratio", np.nan))
    amp = float(a.get("mp_dominant_amplitude", np.nan))
    if (not np.isfinite(freq)) or (freq <= 0.0):
        return False
    if (not np.isfinite(damping)) or (damping < 0.0):
        return False
    if np.isfinite(damp_ratio) and ((damp_ratio < 0.0) or (damp_ratio > 1.0)):
        return False
    if (not np.isfinite(amp)) or (amp <= 0.0):
        return False

    modes = a.get("mp_modes")
    if isinstance(modes, list) and modes:
        amp_sum = float(
            np.sum(
                [
                    float(m.get("amplitude", 0.0))
                    for m in modes
                    if isinstance(m, dict) and np.isfinite(float(m.get("amplitude", np.nan)))
                ]
            )
        )
        if np.isfinite(amp_sum) and (amp_sum > 0.0):
            dom_share = float(amp / amp_sum)
            if dom_share < 0.10:
                return False

    if prev is not None:
        prev_freq = float(prev.get("mp_dominant_freq_hz", np.nan))
        if np.isfinite(prev_freq) and (prev_freq > 0.0):
            rel_jump = abs(freq - prev_freq) / max(prev_freq, 1e-9)
            if rel_jump > 0.20:
                return False
    return True


def _run_matrix_pencil_on_window(
    t_win: np.ndarray,
    v_win: np.ndarray,
    *,
    mp_model_order: int,
    mp_max_modes: int,
    mp_freq_low_hz: float,
    mp_freq_high_hz: float,
    mp_min_samples: int,
    mp_dt_cv_max: float,
    mp_signal_std_min: float,
    mp_singular_ratio_min: float,
    mp_order_selection_enabled: bool = False,
    mp_order_candidates: Sequence[int] | None = None,
) -> dict[str, object]:
    """Estimate local modal components on one window using Matrix Pencil."""

    prepared = _prepare_mp_window_factorization(
        t_win,
        v_win,
        mp_min_samples=int(mp_min_samples),
        mp_dt_cv_max=float(mp_dt_cv_max),
        mp_signal_std_min=float(mp_signal_std_min),
        mp_singular_ratio_min=float(mp_singular_ratio_min),
    )
    prepared_status = str(prepared.get("mp_status", "failed"))
    if prepared_status != "prepared":
        out = dict(prepared)
        out.update(
            {
                "mp_order_selection_enabled": bool(mp_order_selection_enabled),
                "mp_order_candidates_used": [],
                "mp_attempt_count": 0,
                "mp_best_attempt_idx": -1,
                "mp_best_rank": -1,
                "mp_order_stable": 0,
                "mp_order_select_reason": "prepare_failed",
                "mp_attempts": [],
            }
        )
        return out

    rank_max = int(prepared.get("rank_max", 1))
    if rank_max < 1:
        return {
            "mp_status": "failed",
            "mp_reason": "invalid_rank_floor",
            "mp_n_samples": int(prepared.get("n", 0)),
            "mp_dt_sec": float(prepared.get("dt", np.nan)),
            "mp_dt_cv": float(prepared.get("dt_cv", np.nan)),
            "mp_signal_std": float(prepared.get("signal_std", np.nan)),
            "mp_order_selection_enabled": bool(mp_order_selection_enabled),
            "mp_order_candidates_used": [],
            "mp_attempt_count": 0,
            "mp_best_attempt_idx": -1,
            "mp_best_rank": -1,
            "mp_order_stable": 0,
            "mp_order_select_reason": "invalid_rank_floor",
            "mp_attempts": [],
        }

    if bool(mp_order_selection_enabled):
        candidates_raw = _clean_mp_order_candidates(int(mp_model_order), mp_order_candidates)
    else:
        candidates_raw = [int(mp_model_order)]
    candidates_eff: list[int] = []
    seen: set[int] = set()
    for c in candidates_raw:
        cc = int(min(max(1, int(c)), int(rank_max)))
        if cc in seen:
            continue
        seen.add(cc)
        candidates_eff.append(cc)
    if not candidates_eff:
        candidates_eff = [int(min(max(1, int(mp_model_order)), int(rank_max)))]

    attempts: list[dict[str, object]] = []
    for idx, rank_try in enumerate(candidates_eff):
        one = _run_matrix_pencil_rank_candidate(
            prepared,
            int(rank_try),
            mp_max_modes=int(mp_max_modes),
            mp_freq_low_hz=float(mp_freq_low_hz),
            mp_freq_high_hz=float(mp_freq_high_hz),
        )
        one["mp_attempt_idx"] = int(idx)
        attempts.append(one)

    if not attempts:
        return {
            "mp_status": "skipped",
            "mp_reason": "no_attempt_executed",
            "mp_n_samples": int(prepared.get("n", 0)),
            "mp_dt_sec": float(prepared.get("dt", np.nan)),
            "mp_dt_cv": float(prepared.get("dt_cv", np.nan)),
            "mp_signal_std": float(prepared.get("signal_std", np.nan)),
            "mp_order_selection_enabled": bool(mp_order_selection_enabled),
            "mp_order_candidates_used": [int(x) for x in candidates_eff],
            "mp_attempt_count": 0,
            "mp_best_attempt_idx": -1,
            "mp_best_rank": -1,
            "mp_order_stable": 0,
            "mp_order_select_reason": "no_attempt_executed",
            "mp_attempts": [],
        }

    valid_attempts: list[dict[str, object]] = []
    for a in attempts:
        if str(a.get("mp_status", "")) != "ok":
            continue
        if int(a.get("mp_mode_count", 0)) <= 0:
            continue
        fit_r2 = float(a.get("mp_fit_r2", np.nan))
        freq = float(a.get("mp_dominant_freq_hz", np.nan))
        damping = _get_primary_mode_signed_rate_per_sec(a)
        if not np.isfinite(fit_r2):
            continue
        if (not np.isfinite(freq)) or (freq < float(mp_freq_low_hz)) or (freq > float(mp_freq_high_hz)):
            continue
        if (not np.isfinite(damping)) or (damping < 0.0):
            continue
        valid_attempts.append(a)

    best: dict[str, object] | None = None
    best_idx = -1
    order_stable = 0
    select_reason = "fallback"
    fit_margin = 0.01

    if valid_attempts:
        best_fit = max(float(a.get("mp_fit_r2", np.nan)) for a in valid_attempts)
        near_best = [a for a in valid_attempts if float(a.get("mp_fit_r2", np.nan)) >= (best_fit - fit_margin)]
        near_best.sort(key=lambda a: int(a.get("mp_rank_used", 10**9)))

        stable_near_best: list[dict[str, object]] = []
        for a in near_best:
            rank_now = int(a.get("mp_rank_used", -1))
            prev: dict[str, object] | None = None
            for b in valid_attempts:
                rank_b = int(b.get("mp_rank_used", -1))
                if rank_b >= rank_now:
                    continue
                if (prev is None) or (rank_b > int(prev.get("mp_rank_used", -1))):
                    prev = b
            if _is_mp_attempt_plausible(a, prev):
                stable_near_best.append(a)

        if stable_near_best:
            stable_near_best.sort(key=lambda a: int(a.get("mp_rank_used", 10**9)))
            best = stable_near_best[0]
            order_stable = 1
            select_reason = "lowest_stable_within_fit_margin"
        elif near_best:
            best = near_best[0]
            order_stable = 0
            select_reason = "lowest_within_fit_margin_unstable"
        else:
            best = max(valid_attempts, key=lambda a: float(a.get("mp_fit_r2", float("-inf"))))
            order_stable = 0
            select_reason = "best_fit_only"
        best_idx = int(best.get("mp_attempt_idx", -1))
    else:
        def _fallback_key(a: dict[str, object]) -> tuple[int, int, float]:
            fit_r2 = float(a.get("mp_fit_r2", np.nan))
            return (
                int(a.get("mp_mode_count", 0)),
                int(a.get("mp_rank_used", 0)),
                float(fit_r2) if np.isfinite(fit_r2) else float("-inf"),
            )

        best = max(attempts, key=_fallback_key)
        best_idx = int(best.get("mp_attempt_idx", -1))
        order_stable = 0
        select_reason = "no_valid_ok_attempt"

    attempt_summary = [
        {
            "idx": int(a.get("mp_attempt_idx", -1)),
            "rank": int(a.get("mp_rank_used", -1)),
            "status": str(a.get("mp_status", "")),
            "reason": str(a.get("mp_reason", "")),
            "mode_count": int(a.get("mp_mode_count", 0)),
            "fit_r2": float(a.get("mp_fit_r2", np.nan)),
            "dominant_freq_hz": float(a.get("mp_dominant_freq_hz", np.nan)),
            "dominant_damping_per_sec": float(_get_primary_mode_signed_rate_per_sec(a)),
            "dominant_amplitude": float(a.get("mp_dominant_amplitude", np.nan)),
            "singular_ratio": float(a.get("mp_singular_ratio", np.nan)),
        }
        for a in attempts
    ]

    if best is None:
        return {
            "mp_status": "skipped",
            "mp_reason": "no_attempt_executed",
            "mp_n_samples": int(prepared.get("n", 0)),
            "mp_dt_sec": float(prepared.get("dt", np.nan)),
            "mp_dt_cv": float(prepared.get("dt_cv", np.nan)),
            "mp_signal_std": float(prepared.get("signal_std", np.nan)),
            "mp_order_selection_enabled": bool(mp_order_selection_enabled),
            "mp_order_candidates_used": [int(x) for x in candidates_eff],
            "mp_attempt_count": int(len(attempts)),
            "mp_best_attempt_idx": -1,
            "mp_best_rank": -1,
            "mp_order_stable": 0,
            "mp_order_select_reason": "no_attempt_executed",
            "mp_attempts": attempt_summary,
        }

    out = dict(best)
    out.update(
        {
            "mp_n_samples": int(prepared.get("n", out.get("mp_n_samples", 0))),
            "mp_dt_sec": float(prepared.get("dt", out.get("mp_dt_sec", np.nan))),
            "mp_dt_cv": float(prepared.get("dt_cv", out.get("mp_dt_cv", np.nan))),
            "mp_signal_std": float(prepared.get("signal_std", out.get("mp_signal_std", np.nan))),
            "mp_order_selection_enabled": bool(mp_order_selection_enabled),
            "mp_order_candidates_used": [int(x) for x in candidates_eff],
            "mp_attempt_count": int(len(attempts)),
            "mp_best_attempt_idx": int(best_idx),
            "mp_best_rank": int(out.get("mp_rank_used", -1)),
            "mp_order_stable": int(order_stable),
            "mp_order_select_reason": str(select_reason),
            "mp_attempts": attempt_summary,
        }
    )
    if "mp_modes" not in out:
        out["mp_modes"] = []
    if "mp_mode_count" not in out:
        out["mp_mode_count"] = int(len(out["mp_modes"])) if isinstance(out.get("mp_modes"), list) else 0
    return out


class _IntervalMPPostRuntime:
    """Non-blocking interval post-analysis runtime for Matrix Pencil."""

    def __init__(
        self,
        *,
        mp_enabled: bool,
        mp_min_interval_sec: float,
        mp_onset_skip_sec: float,
        mp_onset_window_sec: float,
        mp_onset_min_window_sec: float,
        mp_fallback_use_rms_event_window: bool,
        mp_fallback_default_window_sec: float,
        mp_model_order: int,
        mp_max_modes: int,
        mp_freq_low_hz: float,
        mp_freq_high_hz: float,
        mp_min_samples: int,
        mp_dt_cv_max: float,
        mp_signal_std_min: float,
        mp_singular_ratio_min: float,
        mp_downsample_enabled: bool,
        mp_target_fs_hz: float,
        mp_downsample_lpf_cutoff_hz: float,
        mp_downsample_lpf_order: int,
        mp_order_selection_enabled: bool,
        mp_order_candidates: Sequence[int] | None,
        modal_preprocess_enabled: bool,
        modal_preprocess_method: str,
        modal_preprocess_hampel_half_window: int,
        modal_preprocess_hampel_nsigma: float,
        modal_preprocess_max_repair_points: int,
        modal_preprocess_max_repair_fraction: float,
        modal_preprocess_repair_mode: str,
        modal_preprocess_keep_raw_summary: bool,
        mp_update_cadence_sec: float,
        mp_adaptive_low_band_max_hz: float,
        mp_adaptive_high_band_min_hz: float,
        mp_freq_window_high_sec: float,
        mp_freq_window_mid_sec: float,
        mp_freq_window_low_sec: float,
        mp_decay_window_high_sec: float,
        mp_decay_window_mid_sec: float,
        mp_decay_window_low_sec: float,
        mp_decay_rms_win_sec: float,
        mp_decay_step_sec: float,
        mp_decay_min_windows: int,
        mp_decay_min_r2: float,
        mp_decay_freq_jump_rel_max: float,
        mp_async_enabled: bool,
        mp_queue_maxsize: int,
        mp_finalize_wait_sec: float,
        mp_runtime_mode: str = MP_RUNTIME_MODE_LIVE,
        mp_preview_async_enabled: bool | None = None,
    ) -> None:
        self.enabled = bool(mp_enabled)
        self.mp_min_interval_sec = float(mp_min_interval_sec)
        self.mp_onset_skip_sec = float(mp_onset_skip_sec)
        self.mp_onset_window_sec = float(mp_onset_window_sec)
        self.mp_onset_min_window_sec = float(mp_onset_min_window_sec)
        self.mp_fallback_use_rms_event_window = bool(mp_fallback_use_rms_event_window)
        self.mp_fallback_default_window_sec = float(mp_fallback_default_window_sec)
        self.mp_model_order = int(max(1, mp_model_order))
        self.mp_max_modes = int(max(1, mp_max_modes))
        self.mp_freq_low_hz = float(mp_freq_low_hz)
        self.mp_freq_high_hz = float(mp_freq_high_hz)
        self.mp_min_samples = int(max(8, mp_min_samples))
        self.mp_dt_cv_max = float(max(0.0, mp_dt_cv_max))
        self.mp_signal_std_min = float(max(0.0, mp_signal_std_min))
        self.mp_singular_ratio_min = float(max(0.0, mp_singular_ratio_min))
        self.mp_downsample_enabled = bool(mp_downsample_enabled)
        self.mp_target_fs_hz = float(max(1e-9, mp_target_fs_hz))
        self.mp_downsample_lpf_cutoff_hz = float(max(1e-9, mp_downsample_lpf_cutoff_hz))
        self.mp_downsample_lpf_order = int(max(1, mp_downsample_lpf_order))
        self.mp_order_selection_enabled = bool(mp_order_selection_enabled)
        self.mp_order_candidates = tuple(int(x) for x in _clean_mp_order_candidates(int(self.mp_model_order), mp_order_candidates))
        self.modal_preprocess_enabled = bool(modal_preprocess_enabled)
        self.modal_preprocess_method = str(modal_preprocess_method).strip().lower()
        self.modal_preprocess_hampel_half_window = int(max(1, modal_preprocess_hampel_half_window))
        self.modal_preprocess_hampel_nsigma = float(max(0.0, modal_preprocess_hampel_nsigma))
        self.modal_preprocess_max_repair_points = int(max(0, modal_preprocess_max_repair_points))
        self.modal_preprocess_max_repair_fraction = float(max(0.0, modal_preprocess_max_repair_fraction))
        self.modal_preprocess_repair_mode = str(modal_preprocess_repair_mode).strip().lower()
        self.modal_preprocess_keep_raw_summary = bool(modal_preprocess_keep_raw_summary)
        self.mp_update_cadence_sec = float(max(0.0, mp_update_cadence_sec))
        self.mp_adaptive_low_band_max_hz = float(max(1e-9, mp_adaptive_low_band_max_hz))
        self.mp_adaptive_high_band_min_hz = float(max(self.mp_adaptive_low_band_max_hz, mp_adaptive_high_band_min_hz))
        self.mp_freq_window_high_sec = float(max(1.0, mp_freq_window_high_sec))
        self.mp_freq_window_mid_sec = float(max(1.0, mp_freq_window_mid_sec))
        self.mp_freq_window_low_sec = float(max(1.0, mp_freq_window_low_sec))
        self.mp_decay_window_high_sec = float(max(1.0, mp_decay_window_high_sec))
        self.mp_decay_window_mid_sec = float(max(1.0, mp_decay_window_mid_sec))
        self.mp_decay_window_low_sec = float(max(1.0, mp_decay_window_low_sec))
        self.mp_decay_rms_win_sec = float(max(1e-3, mp_decay_rms_win_sec))
        self.mp_decay_step_sec = float(max(1e-3, mp_decay_step_sec))
        self.mp_decay_min_windows = int(max(2, mp_decay_min_windows))
        self.mp_decay_min_r2 = float(max(0.0, min(1.0, mp_decay_min_r2)))
        self.mp_decay_freq_jump_rel_max = float(max(0.0, mp_decay_freq_jump_rel_max))
        self.mp_async_enabled = bool(mp_async_enabled)
        self.mp_finalize_wait_sec = float(max(0.0, mp_finalize_wait_sec))
        mode = str(mp_runtime_mode).strip().lower()
        if mode not in (MP_RUNTIME_MODE_LIVE, MP_RUNTIME_MODE_REPLAY):
            mode = MP_RUNTIME_MODE_LIVE
        self.mp_runtime_mode = str(mode)
        self.mp_preview_async_enabled = (
            bool(mp_preview_async_enabled)
            if (mp_preview_async_enabled is not None)
            else bool(self.mp_runtime_mode == MP_RUNTIME_MODE_LIVE)
        )

        self._active_capture_lock = threading.Lock()
        self._active_capture_by_key: dict[tuple[str, str], dict[str, object]] = {}
        self._next_interval_seq_by_key: dict[tuple[str, str], int] = {}
        self._interval_capture_by_interval_id: dict[int, list[tuple[float, float]]] = {}
        self._records_by_interval_id: dict[int, dict[str, object]] = {}
        self._latest_seq_by_interval_id: dict[int, int] = {}
        self._latest_interval_snapshot_by_interval_id: dict[
            int,
            tuple[
                dict,
                float,
                tuple[tuple[float, float], ...],
            ],
        ] = {}
        self._submit_seq = 0
        self._record_lock = threading.Lock()

        self._queue: queue.Queue[
            tuple[
                int,
                dict,
                float,
                tuple[tuple[float, float], ...],
            ]
        ] | None = None
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        if self.enabled and self.mp_async_enabled:
            qsize = int(max(1, mp_queue_maxsize))
            self._queue = queue.Queue(maxsize=qsize)
            self._stop_event = threading.Event()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

        self._preview_lock = threading.Lock()
        self._preview_latest_job_by_scope: dict[tuple[str, str, int], dict[str, object]] = {}
        self._preview_enqueued_scopes: set[tuple[str, str, int]] = set()
        self._preview_queue: queue.Queue[tuple[str, str, int]] | None = None
        self._preview_stop_event: threading.Event | None = None
        self._preview_worker: threading.Thread | None = None
        if self.enabled and self.mp_preview_async_enabled and (self.mp_update_cadence_sec > 0.0):
            self._preview_queue = queue.Queue()
            self._preview_stop_event = threading.Event()
            self._preview_worker = threading.Thread(target=self._preview_worker_loop, daemon=True)
            self._preview_worker.start()

    def _new_capture_state(self, key: tuple[str, str], start_t: float) -> dict[str, object]:
        seq = int(self._next_interval_seq_by_key.get(key, 0)) + 1
        self._next_interval_seq_by_key[key] = int(seq)
        return {
            "start_t": float(start_t),
            "samples": [],
            "live_modal_snapshot": None,
            "live_freq_hint_hz": float("nan"),
            "last_preview_submit_t": float("nan"),
            "last_preview_submit_sample_count": 0,
            "preview_seq": 0,
            "last_preview_apply_seq": 0,
            "interval_seq": int(seq),
            "lock": threading.Lock(),
        }

    def _reset_capture_state_locked(self, cap: dict[str, object], *, key: tuple[str, str], start_t: float) -> int:
        seq = int(self._next_interval_seq_by_key.get(key, int(cap.get("interval_seq", 0)))) + 1
        self._next_interval_seq_by_key[key] = int(seq)
        cap["start_t"] = float(start_t)
        cap["samples"] = []
        cap["live_modal_snapshot"] = None
        cap["live_freq_hint_hz"] = float("nan")
        cap["last_preview_submit_t"] = float("nan")
        cap["last_preview_submit_sample_count"] = 0
        cap["preview_seq"] = 0
        cap["last_preview_apply_seq"] = 0
        cap["interval_seq"] = int(seq)
        return int(seq)

    def sync_interval_capture(
        self,
        *,
        key: tuple[str, str],
        active_start_t: float,
        is_interval_active: bool,
        tick_samples: Sequence[tuple[float, float]],
    ) -> None:
        """Capture interval samples and submit preview snapshots only (no in-thread MP compute)."""

        if not self.enabled:
            return
        if (not bool(is_interval_active)) or (not np.isfinite(active_start_t)):
            with self._active_capture_lock:
                cap = self._active_capture_by_key.pop(key, None)
            if isinstance(cap, dict):
                lock = cap.get("lock")
                if (lock is None) or (not hasattr(lock, "acquire")):
                    lock = threading.Lock()
                    cap["lock"] = lock
                with lock:
                    interval_seq = int(cap.get("interval_seq", -1))
                if int(interval_seq) >= 0:
                    self._invalidate_preview_scope(key=key, interval_seq=int(interval_seq))
            return

        with self._active_capture_lock:
            cap = self._active_capture_by_key.get(key)
            if cap is None:
                cap = self._new_capture_state(key=key, start_t=float(active_start_t))
                self._active_capture_by_key[key] = cap

        lock = cap.get("lock")
        if (lock is None) or (not hasattr(lock, "acquire")):
            lock = threading.Lock()
            cap["lock"] = lock

        invalidate_interval_seq = -1
        preview_job: dict[str, object] | None = None
        with lock:
            start_t = float(cap.get("start_t", np.nan))
            if not np.isfinite(start_t):
                start_t = float(active_start_t)
                cap["start_t"] = float(start_t)
            elif float(active_start_t) > (float(start_t) + 1e-6):
                invalidate_interval_seq = int(cap.get("interval_seq", -1))
                self._reset_capture_state_locked(cap, key=key, start_t=float(active_start_t))
                start_t = float(active_start_t)
            elif float(active_start_t) < (float(start_t) - 1e-6):
                cap["start_t"] = float(active_start_t)
                start_t = float(active_start_t)

            samples = cap.get("samples")
            if not isinstance(samples, list):
                samples = []
                cap["samples"] = samples
            for t, v in tick_samples:
                t_f = float(t)
                v_f = float(v)
                if np.isfinite(t_f) and np.isfinite(v_f) and (t_f >= float(start_t)):
                    samples.append((t_f, v_f))
            preview_job = self._build_preview_job_locked(key=key, cap=cap)

        if int(invalidate_interval_seq) >= 0:
            self._invalidate_preview_scope(key=key, interval_seq=int(invalidate_interval_seq))
        if isinstance(preview_job, dict):
            self._submit_preview_snapshot(preview_job)

    def _build_preview_job_locked(self, *, key: tuple[str, str], cap: dict[str, object]) -> dict[str, object] | None:
        if (not bool(self.mp_preview_async_enabled)) or (float(self.mp_update_cadence_sec) <= 0.0):
            return None
        samples = cap.get("samples")
        if not isinstance(samples, list):
            return None
        n_samples = int(len(samples))
        if n_samples < int(self.mp_min_samples):
            return None
        start_t = float(cap.get("start_t", np.nan))
        t_last = float(samples[-1][0]) if samples else float("nan")
        if (not np.isfinite(start_t)) or (not np.isfinite(t_last)) or (float(t_last) <= float(start_t)):
            return None
        readiness_min_sec = float(max(0.0, float(self.mp_onset_skip_sec)) + max(0.0, float(self.mp_onset_min_window_sec)))
        if float(t_last - start_t) < float(readiness_min_sec):
            return None
        last_submit_t = float(cap.get("last_preview_submit_t", np.nan))
        if np.isfinite(last_submit_t) and ((float(t_last) - float(last_submit_t)) < float(self.mp_update_cadence_sec)):
            return None
        last_submit_n = int(cap.get("last_preview_submit_sample_count", 0))
        min_new_samples = int(max(8, int(self.mp_min_samples) // 4))
        if int(last_submit_n) > 0 and ((n_samples - int(last_submit_n)) < int(min_new_samples)):
            return None

        preview_seq = int(cap.get("preview_seq", 0)) + 1
        cap["preview_seq"] = int(preview_seq)
        cap["last_preview_submit_t"] = float(t_last)
        cap["last_preview_submit_sample_count"] = int(n_samples)
        prev_hint_hz = float(cap.get("live_freq_hint_hz", np.nan))
        snap_obj = cap.get("live_modal_snapshot")
        if isinstance(snap_obj, dict):
            try:
                snap_hint = float(snap_obj.get("freq_hz", np.nan))
            except Exception:
                snap_hint = float("nan")
            if np.isfinite(snap_hint) and (snap_hint > 0.0):
                prev_hint_hz = float(snap_hint)
        sample_snapshot = tuple((float(t), float(v)) for t, v in samples)
        return {
            "key": (str(key[0]), str(key[1])),
            "interval_seq": int(cap.get("interval_seq", -1)),
            "preview_seq": int(preview_seq),
            "start_t": float(start_t),
            "t_end": float(t_last),
            "prev_hint_hz": float(prev_hint_hz),
            "samples": sample_snapshot,
        }

    def _submit_preview_snapshot(self, job: dict[str, object]) -> None:
        if self._preview_queue is None:
            return
        key = job.get("key")
        if (not isinstance(key, tuple)) or (len(key) != 2):
            return
        scope = (str(key[0]), str(key[1]), int(job.get("interval_seq", -1)))
        with self._preview_lock:
            self._preview_latest_job_by_scope[scope] = dict(job)
            if scope in self._preview_enqueued_scopes:
                return
            self._preview_enqueued_scopes.add(scope)
            self._preview_queue.put_nowait(scope)

    def _invalidate_preview_scope(self, *, key: tuple[str, str], interval_seq: int) -> None:
        scope = (str(key[0]), str(key[1]), int(interval_seq))
        with self._preview_lock:
            self._preview_latest_job_by_scope.pop(scope, None)
            self._preview_enqueued_scopes.discard(scope)

    def _preview_worker_loop(self) -> None:
        assert self._preview_queue is not None
        while True:
            if (self._preview_stop_event is not None) and self._preview_stop_event.is_set():
                break
            try:
                scope = self._preview_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._preview_lock:
                    self._preview_enqueued_scopes.discard(scope)
                    job = self._preview_latest_job_by_scope.pop(scope, None)
                if not isinstance(job, dict):
                    continue
                try:
                    snapshot = self._estimate_live_modal_snapshot(
                        start_t=float(job.get("start_t", np.nan)),
                        t_end=float(job.get("t_end", np.nan)),
                        samples=job.get("samples", ()),
                        prev_hint_hz=float(job.get("prev_hint_hz", np.nan)),
                    )
                except Exception:
                    snapshot = None
                self._apply_preview_snapshot(job=job, snapshot=snapshot)
            finally:
                self._preview_queue.task_done()

    def _apply_preview_snapshot(self, *, job: dict[str, object], snapshot: dict[str, object] | None) -> None:
        if not isinstance(snapshot, dict):
            return
        status = str(snapshot.get("status", "")).strip().lower()
        hint_hz = float(snapshot.get("freq_hz", np.nan))
        if (status != "ok") or (not np.isfinite(hint_hz)) or (float(hint_hz) <= 0.0):
            return
        key = job.get("key")
        if (not isinstance(key, tuple)) or (len(key) != 2):
            return
        interval_seq = int(job.get("interval_seq", -1))
        preview_seq = int(job.get("preview_seq", -1))
        if int(interval_seq) < 0 or int(preview_seq) < 0:
            return
        with self._active_capture_lock:
            cap = self._active_capture_by_key.get((str(key[0]), str(key[1])))
        if not isinstance(cap, dict):
            return
        lock = cap.get("lock")
        if (lock is None) or (not hasattr(lock, "acquire")):
            return
        with lock:
            if int(cap.get("interval_seq", -1)) != int(interval_seq):
                return
            if int(cap.get("preview_seq", -1)) != int(preview_seq):
                return
            if int(preview_seq) <= int(cap.get("last_preview_apply_seq", -1)):
                return
            cap["live_modal_snapshot"] = dict(snapshot)
            cap["live_freq_hint_hz"] = float(hint_hz)
            cap["last_preview_apply_seq"] = int(preview_seq)

    def _estimate_live_modal_snapshot(
        self,
        *,
        start_t: float,
        t_end: float,
        samples: Sequence[tuple[float, float]],
        prev_hint_hz: float,
    ) -> dict[str, object] | None:
        """Estimate local modal snapshot on active interval bounded preview window."""

        if (not np.isfinite(start_t)) or (not np.isfinite(t_end)) or (float(t_end) <= float(start_t)):
            return None
        if not samples:
            return None
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        t_src = arr[:, 0]
        v_src = arr[:, 1]
        duration_sec = float(t_end - start_t)
        prof = _adaptive_profile_from_freq(
            float(prev_hint_hz),
            low_band_max_hz=float(self.mp_adaptive_low_band_max_hz),
            high_band_min_hz=float(self.mp_adaptive_high_band_min_hz),
        )
        w_sec = _window_sec_for_profile(
            str(prof),
            high_sec=float(self.mp_freq_window_high_sec),
            mid_sec=float(self.mp_freq_window_mid_sec),
            low_sec=float(self.mp_freq_window_low_sec),
        )
        _src, w0, w1 = _select_mp_analysis_window(
            start_t=float(start_t),
            end_t=float(t_end),
            duration_sec=float(duration_sec),
            rms_event_win_sec=float("nan"),
            mp_onset_skip_sec=float(self.mp_onset_skip_sec),
            mp_onset_window_sec=float(w_sec),
            mp_onset_min_window_sec=float(self.mp_onset_min_window_sec),
            mp_fallback_use_rms_event_window=False,
            mp_fallback_default_window_sec=float(w_sec),
        )
        if (not np.isfinite(w0)) or (not np.isfinite(w1)) or (float(w1) <= float(w0)):
            return None
        m = np.isfinite(t_src) & np.isfinite(v_src) & (t_src >= float(w0)) & (t_src <= float(w1))
        if int(np.count_nonzero(m)) < int(self.mp_min_samples):
            return None
        fit = _run_matrix_pencil_on_window(
            t_src[m],
            v_src[m],
            mp_model_order=int(self.mp_model_order),
            mp_max_modes=int(self.mp_max_modes),
            mp_freq_low_hz=float(self.mp_freq_low_hz),
            mp_freq_high_hz=float(self.mp_freq_high_hz),
            mp_min_samples=int(self.mp_min_samples),
            mp_dt_cv_max=float(self.mp_dt_cv_max),
            mp_signal_std_min=float(self.mp_signal_std_min),
            mp_singular_ratio_min=float(self.mp_singular_ratio_min),
            mp_order_selection_enabled=bool(self.mp_order_selection_enabled),
            mp_order_candidates=self.mp_order_candidates,
        )
        status = str(fit.get("mp_status", "")).strip().lower()
        if status != "ok":
            return None
        f0 = float(fit.get("mp_dominant_freq_hz", np.nan))
        if (not np.isfinite(f0)) or (float(f0) <= 0.0):
            return None
        dps = float(_get_primary_mode_signed_rate_per_sec(fit))
        damping_ratio = _damping_ratio_from(float(f0), float(dps))
        modes_raw = fit.get("mp_modes", [])
        attempts_raw = fit.get("mp_attempts", [])
        cands_raw = fit.get("mp_order_candidates_used", self.mp_order_candidates)
        modes = [dict(m) for m in modes_raw] if isinstance(modes_raw, list) else []
        attempts = [dict(a) for a in attempts_raw] if isinstance(attempts_raw, list) else []
        order_candidates_used = (
            [int(x) for x in cands_raw]
            if isinstance(cands_raw, (list, tuple))
            else [int(x) for x in self.mp_order_candidates]
        )
        return {
            "status": str(status),
            "reason": str(fit.get("mp_reason", "ok")),
            "freq_hz": float(f0),
            "damping_per_sec": float(dps),
            "damping_ratio": float(damping_ratio),
            "fit_r2": float(fit.get("mp_fit_r2", np.nan)),
            "mode_count": int(fit.get("mp_mode_count", len(modes))),
            "window_source": str(_src),
            "window_start_t": float(w0),
            "window_end_t": float(w1),
            "window_sec_target": float(w_sec),
            "freq_profile": str(prof),
            "n_samples": int(fit.get("mp_n_samples", int(np.count_nonzero(m)))),
            "signal_std": float(fit.get("mp_signal_std", np.nan)),
            "rank_used": int(fit.get("mp_rank_used", -1)),
            "dominant_amplitude": float(fit.get("mp_dominant_amplitude", np.nan)),
            "order_selection_enabled": bool(fit.get("mp_order_selection_enabled", self.mp_order_selection_enabled)),
            "order_candidates_used": order_candidates_used,
            "attempt_count": int(fit.get("mp_attempt_count", len(attempts))),
            "best_attempt_idx": int(fit.get("mp_best_attempt_idx", -1)),
            "best_rank": int(fit.get("mp_best_rank", -1)),
            "order_stable": int(fit.get("mp_order_stable", 0)),
            "order_select_reason": str(fit.get("mp_order_select_reason", "not_reported")),
            "attempts": attempts,
            "modes": modes,
        }

    @staticmethod
    def _sorted_sample_pairs(samples: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Sort and lightly deduplicate sample pairs by `(t, v)`."""

        if not samples:
            return []
        out = sorted(
            (
                (float(t), float(v))
                for t, v in samples
                if np.isfinite(float(t)) and np.isfinite(float(v))
            ),
            key=lambda x: x[0],
        )
        if not out:
            return []
        dedup: list[tuple[float, float]] = [out[0]]
        for t, v in out[1:]:
            pt, pv = dedup[-1]
            if (abs(float(t) - float(pt)) <= 1e-12) and (abs(float(v) - float(pv)) <= 1e-12):
                continue
            dedup.append((float(t), float(v)))
        return dedup

    def submit_interval(
        self,
        interval_ev: dict,
        rms_event_win_sec: float = float("nan"),
    ) -> None:
        """Submit finalized interval for post-analysis without blocking detector path."""

        if not self.enabled:
            return
        interval_copy = dict(interval_ev)
        interval_id = int(interval_copy.get("interval_id", -1))
        key = (str(interval_copy.get("device", "")), str(interval_copy.get("channel", "")))

        with self._active_capture_lock:
            seg_capture = self._active_capture_by_key.pop(key, None)

        live_hint = float("nan")
        live_snapshot: dict[str, object] | None = None
        interval_seq = -1
        seg_samples: list[tuple[float, float]] = []
        if isinstance(seg_capture, dict):
            lock = seg_capture.get("lock")
            if (lock is None) or (not hasattr(lock, "acquire")):
                lock = threading.Lock()
                seg_capture["lock"] = lock
            with lock:
                live_hint = float(seg_capture.get("live_freq_hint_hz", np.nan))
                snap_obj = seg_capture.get("live_modal_snapshot")
                if isinstance(snap_obj, dict):
                    live_snapshot = dict(snap_obj)
                interval_seq = int(seg_capture.get("interval_seq", -1))
                samples_obj = seg_capture.get("samples", [])
                if isinstance(samples_obj, list):
                    seg_samples = [(float(t), float(v)) for t, v in samples_obj]
        if isinstance(live_snapshot, dict):
            snap_hint = float(live_snapshot.get("freq_hz", np.nan))
            if np.isfinite(snap_hint) and (snap_hint > 0.0):
                live_hint = float(snap_hint)
        if np.isfinite(live_hint) and (live_hint > 0.0):
            interval_copy["mp_live_freq_hint_hz"] = float(live_hint)
        if int(interval_seq) >= 0:
            self._invalidate_preview_scope(key=key, interval_seq=int(interval_seq))

        if str(self.mp_runtime_mode) == MP_RUNTIME_MODE_LIVE:
            rec = self._build_live_record_from_snapshot(
                interval_ev=interval_copy,
                rms_event_win_sec=float(rms_event_win_sec),
                live_snapshot=live_snapshot,
            )
            with self._record_lock:
                self._submit_seq += 1
                seq = int(self._submit_seq)
                self._latest_seq_by_interval_id[interval_id] = seq
                self._latest_interval_snapshot_by_interval_id[interval_id] = (
                    interval_copy,
                    float(rms_event_win_sec),
                    tuple(),
                )
            self._store_record(int(seq), int(interval_id), rec)
            self._interval_capture_by_interval_id.pop(interval_id, None)
            return

        merged_samples = list(self._interval_capture_by_interval_id.get(interval_id, []))
        if seg_samples:
            merged_samples.extend((float(t), float(v)) for t, v in seg_samples)
        merged_samples = self._sorted_sample_pairs(merged_samples)
        self._interval_capture_by_interval_id[interval_id] = merged_samples

        merged_samples_ro = tuple(merged_samples)
        with self._record_lock:
            self._submit_seq += 1
            seq = int(self._submit_seq)
            self._latest_seq_by_interval_id[interval_id] = seq
            self._latest_interval_snapshot_by_interval_id[interval_id] = (
                interval_copy,
                float(rms_event_win_sec),
                merged_samples_ro,
            )
        job = (int(seq), interval_copy, float(rms_event_win_sec), merged_samples_ro)

        if self._queue is not None:
            try:
                self._queue.put_nowait(job)
                return
            except queue.Full:
                self._queue.put(job)
                return

        try:
            rec = self._analyze_interval(
                interval_ev=interval_copy,
                rms_event_win_sec=float(rms_event_win_sec),
                samples=merged_samples_ro,
            )
        except Exception as exc:
            rec = self._build_skip_record(
                interval_ev=interval_copy,
                reason=f"analysis_exception:{type(exc).__name__}",
                rms_event_win_sec=float(rms_event_win_sec),
                window_source="none",
                window_start_t=float("nan"),
                window_end_t=float("nan"),
            )
        self._store_record(int(seq), int(interval_id), rec)

    def finalize(self) -> list[dict[str, object]]:
        """Stop worker and return interval-keyed post-analysis records."""

        if not self.enabled:
            return []

        if self._preview_stop_event is not None:
            self._preview_stop_event.set()
        if self._preview_worker is not None and self._preview_worker.is_alive():
            self._preview_worker.join(timeout=0.1)
        if self._preview_queue is not None:
            while True:
                try:
                    _scope = self._preview_queue.get_nowait()
                except queue.Empty:
                    break
                self._preview_queue.task_done()
        with self._preview_lock:
            self._preview_latest_job_by_scope.clear()
            self._preview_enqueued_scopes.clear()

        if self._stop_event is not None:
            if self._queue is not None:
                wait_deadline = time.time() + float(self.mp_finalize_wait_sec)
                while (time.time() < wait_deadline) and (not self._queue.empty()):
                    time.sleep(0.01)
            self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=float(self.mp_finalize_wait_sec))

        if self._queue is not None:
            while True:
                try:
                    seq, interval_ev, rms_event_win_sec, _samples = self._queue.get_nowait()
                except queue.Empty:
                    break
                interval_id = int(interval_ev.get("interval_id", -1))
                rec = self._build_skip_record(
                    interval_ev=interval_ev,
                    reason="worker_not_finished",
                    rms_event_win_sec=float(rms_event_win_sec),
                    window_source="none",
                    window_start_t=float("nan"),
                    window_end_t=float("nan"),
                )
                self._store_record(int(seq), int(interval_id), rec)
                self._queue.task_done()

        with self._record_lock:
            missing_ids = [
                int(iid)
                for iid in self._latest_seq_by_interval_id.keys()
                if int(iid) not in self._records_by_interval_id
            ]
            for iid in missing_ids:
                snap, rms_event_win_sec, _samples = self._latest_interval_snapshot_by_interval_id.get(
                    int(iid),
                    ({}, float("nan"), tuple()),
                )
                self._records_by_interval_id[int(iid)] = self._build_skip_record(
                    interval_ev=dict(snap),
                    reason="pending_timeout",
                    rms_event_win_sec=float(rms_event_win_sec),
                    window_source="none",
                    window_start_t=float("nan"),
                    window_end_t=float("nan"),
                )

        with self._record_lock:
            out = list(self._records_by_interval_id.values())
        out.sort(key=lambda e: (int(e.get("interval_id", -1)), float(e.get("end_t", np.nan))))
        with self._active_capture_lock:
            self._active_capture_by_key.clear()
            self._next_interval_seq_by_key.clear()
        self._interval_capture_by_interval_id.clear()
        return out

    def _worker_loop(self) -> None:
        """Consume finalized interval jobs in background."""

        assert self._queue is not None
        while True:
            if (self._stop_event is not None) and self._stop_event.is_set():
                break
            try:
                seq, interval_ev, rms_event_win_sec, samples = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            interval_id = int(interval_ev.get("interval_id", -1))
            try:
                rec = self._analyze_interval(
                    interval_ev=interval_ev,
                    rms_event_win_sec=float(rms_event_win_sec),
                    samples=samples,
                )
            except Exception as exc:
                rec = self._build_skip_record(
                    interval_ev=interval_ev,
                    reason=f"worker_exception:{type(exc).__name__}",
                    rms_event_win_sec=float(rms_event_win_sec),
                    window_source="none",
                    window_start_t=float("nan"),
                    window_end_t=float("nan"),
                )
            self._store_record(int(seq), int(interval_id), rec)
            self._queue.task_done()

    def _store_record(self, seq: int, interval_id: int, rec: dict[str, object]) -> None:
        """Store latest record per interval_id (stitch-safe overwrite)."""

        with self._record_lock:
            latest = int(self._latest_seq_by_interval_id.get(int(interval_id), -1))
            if int(seq) < latest:
                return
            self._records_by_interval_id[int(interval_id)] = rec

    def _build_skip_record(
        self,
        *,
        interval_ev: dict,
        reason: str,
        rms_event_win_sec: float,
        window_source: str,
        window_start_t: float,
        window_end_t: float,
    ) -> dict[str, object]:
        """Build skipped/failed post-analysis record with interval metadata."""

        rec = _build_mp_interval_record_base(interval_ev)
        rec.update(
            default_postprep_summary(
                enabled=bool(self.modal_preprocess_enabled),
                method=str(self.modal_preprocess_method),
                repair_mode=str(self.modal_preprocess_repair_mode),
                keep_raw_summary=bool(self.modal_preprocess_keep_raw_summary),
            )
        )
        analysis_section = {
            "mp_status": "skipped",
            "mp_reason": str(reason),
        }
        window_section = {
            "mp_window_source": str(window_source),
            "mp_window_start_t": float(window_start_t),
            "mp_window_end_t": float(window_end_t),
            "mp_window_duration_sec": (
                float(window_end_t - window_start_t)
                if np.isfinite(window_start_t) and np.isfinite(window_end_t)
                else float("nan")
            ),
            "mp_rms_event_win_sec": float(rms_event_win_sec),
        }
        provenance_section = {
            "mp_update_cadence_sec": float(self.mp_update_cadence_sec),
            "mp_live_freq_hint_hz": float(interval_ev.get("mp_live_freq_hint_hz", np.nan)),
        }
        profile_section = {
            "mp_freq_profile": "mid",
            "mp_freq_window_sec_target": float(self.mp_freq_window_mid_sec),
            "mp_decay_profile": "mid",
            "mp_decay_window_sec_target": float(self.mp_decay_window_mid_sec),
        }
        decay_section = {
            "mp_decay_window_source": "none",
            "mp_decay_window_start_t": float("nan"),
            "mp_decay_window_end_t": float("nan"),
            "mp_decay_window_duration_sec": float("nan"),
            "mp_decay_qualified": 0,
            "mp_decay_qualified_reason": "not_executed",
            "mp_decay_fit_status": "not_executed",
            "mp_decay_fit_reason": "not_executed",
            "mp_decay_fit_freq_hz": float("nan"),
            "mp_decay_freq_jump_rel": float("nan"),
            "mp_decay_rms_slope_1_per_sec": float("nan"),
            "mp_decay_rms_r2": float("nan"),
            "mp_decay_rms_n_windows": 0,
            "mp_effective_decay_rate_1_per_sec": float("nan"),
            "mp_half_life_sec": float("nan"),
            "mp_freq_window_damping_per_sec": float("nan"),
            "mp_freq_window_damping_ratio": float("nan"),
        }
        result_quality_section = {
            "mp_n_samples": 0,
            "mp_fit_r2": float("nan"),
            "mp_mode_count": 0,
            "mp_dominant_freq_hz": float("nan"),
            "mp_dominant_damping_per_sec": float("nan"),
            "mp_dominant_damping_ratio": float("nan"),
            "mp_dominant_amplitude": float("nan"),
            "mp_signal_std": float("nan"),
        }
        selection_section = {
            "mp_rank_used": -1,
            "mp_order_selection_enabled": bool(self.mp_order_selection_enabled),
            "mp_order_candidates_used": [int(x) for x in self.mp_order_candidates],
            "mp_attempt_count": 0,
            "mp_best_attempt_idx": -1,
            "mp_best_rank": -1,
            "mp_order_stable": 0,
            "mp_order_select_reason": "not_executed",
            "mp_attempts": [],
            "mp_modes": [],
        }
        downsample_section = {
            "mp_downsample_enabled": bool(self.mp_downsample_enabled),
            "mp_downsample_applied": 0,
            "mp_downsample_reason": "disabled" if (not bool(self.mp_downsample_enabled)) else "not_applied",
            "mp_downsample_source_fs_hz": float("nan"),
            "mp_downsample_target_fs_hz": float(self.mp_target_fs_hz),
            "mp_downsample_lpf_cutoff_hz": float(self.mp_downsample_lpf_cutoff_hz),
            "mp_downsample_lpf_order": int(self.mp_downsample_lpf_order),
            "mp_downsample_n_samples_in": 0,
            "mp_downsample_n_samples_out": 0,
        }
        rec.update(
            _merge_mp_record_sections(
                analysis_section,
                window_section,
                provenance_section,
                profile_section,
                decay_section,
                result_quality_section,
                selection_section,
                downsample_section,
            )
        )
        return _apply_primary_mode_aliases(rec)

    def _build_live_record_from_snapshot(
        self,
        *,
        interval_ev: dict,
        rms_event_win_sec: float,
        live_snapshot: dict[str, object] | None,
    ) -> dict[str, object]:
        """Build interval record directly from latest valid live modal snapshot."""

        if not isinstance(live_snapshot, dict):
            return self._build_skip_record(
                interval_ev=interval_ev,
                reason="no_live_snapshot",
                rms_event_win_sec=float(rms_event_win_sec),
                window_source="live_preview",
                window_start_t=float("nan"),
                window_end_t=float("nan"),
            )

        freq_hz = float(live_snapshot.get("freq_hz", np.nan))
        damping_per_sec = float(live_snapshot.get("damping_per_sec", np.nan))
        damping_ratio = float(live_snapshot.get("damping_ratio", np.nan))
        if not np.isfinite(damping_ratio):
            damping_ratio = _damping_ratio_from(float(freq_hz), float(damping_per_sec))
        window_start_t = float(live_snapshot.get("window_start_t", np.nan))
        window_end_t = float(live_snapshot.get("window_end_t", np.nan))
        window_source = str(live_snapshot.get("window_source", "live_preview"))
        if (not np.isfinite(window_start_t)) or (not np.isfinite(window_end_t)) or (float(window_end_t) <= float(window_start_t)):
            return self._build_skip_record(
                interval_ev=interval_ev,
                reason="invalid_live_snapshot_window",
                rms_event_win_sec=float(rms_event_win_sec),
                window_source=str(window_source),
                window_start_t=float(window_start_t),
                window_end_t=float(window_end_t),
            )

        if (not np.isfinite(freq_hz)) or (float(freq_hz) <= 0.0):
            return self._build_skip_record(
                interval_ev=interval_ev,
                reason="invalid_live_snapshot_freq",
                rms_event_win_sec=float(rms_event_win_sec),
                window_source=str(window_source),
                window_start_t=float(window_start_t),
                window_end_t=float(window_end_t),
            )

        rec = self._build_skip_record(
            interval_ev=interval_ev,
            reason="live_preview_base",
            rms_event_win_sec=float(rms_event_win_sec),
            window_source=str(window_source),
            window_start_t=float(window_start_t),
            window_end_t=float(window_end_t),
        )
        modes_raw = live_snapshot.get("modes", [])
        attempts_raw = live_snapshot.get("attempts", [])
        cands_raw = live_snapshot.get("order_candidates_used", self.mp_order_candidates)
        modes = [dict(m) for m in modes_raw] if isinstance(modes_raw, list) else []
        attempts = [dict(a) for a in attempts_raw] if isinstance(attempts_raw, list) else []
        order_candidates_used = (
            [int(x) for x in cands_raw]
            if isinstance(cands_raw, (list, tuple))
            else [int(x) for x in self.mp_order_candidates]
        )
        status = str(live_snapshot.get("status", "ok")).strip().lower() or "ok"
        reason = str(live_snapshot.get("reason", "ok"))
        freq_profile = str(live_snapshot.get("freq_profile", "mid"))
        provenance_section = {
            "mp_runtime_source": "live_preview_snapshot",
        }
        analysis_section = {
            "mp_status": str(status),
            "mp_reason": str(reason),
        }
        window_section = {
            "mp_window_source": str(window_source),
            "mp_window_start_t": float(window_start_t),
            "mp_window_end_t": float(window_end_t),
            "mp_window_duration_sec": float(window_end_t - window_start_t),
            "mp_rms_event_win_sec": float(rms_event_win_sec),
        }
        profile_section = {
            "mp_freq_profile": str(freq_profile),
            "mp_freq_window_sec_target": float(live_snapshot.get("window_sec_target", self.mp_freq_window_mid_sec)),
            "mp_decay_profile": str(freq_profile),
            "mp_decay_window_sec_target": float("nan"),
        }
        decay_section = {
            "mp_decay_window_source": "not_executed",
            "mp_decay_window_start_t": float("nan"),
            "mp_decay_window_end_t": float("nan"),
            "mp_decay_window_duration_sec": float("nan"),
            "mp_decay_qualified": 0,
            "mp_decay_qualified_reason": "live_freq_window_only",
            "mp_decay_fit_status": "not_executed",
            "mp_decay_fit_reason": "not_executed",
            "mp_decay_fit_freq_hz": float("nan"),
            "mp_decay_freq_jump_rel": float("nan"),
            "mp_decay_rms_slope_1_per_sec": float("nan"),
            "mp_decay_rms_r2": float("nan"),
            "mp_decay_rms_n_windows": 0,
            "mp_effective_decay_rate_1_per_sec": float("nan"),
            "mp_half_life_sec": float("nan"),
            "mp_freq_window_damping_per_sec": float(damping_per_sec),
            "mp_freq_window_damping_ratio": float(damping_ratio),
        }
        result_quality_section = {
            "mp_n_samples": int(live_snapshot.get("n_samples", 0)),
            "mp_fit_r2": float(live_snapshot.get("fit_r2", np.nan)),
            "mp_mode_count": int(live_snapshot.get("mode_count", len(modes))),
            "mp_dominant_freq_hz": float(freq_hz),
            "mp_dominant_damping_per_sec": float(damping_per_sec),
            "mp_dominant_damping_ratio": float(damping_ratio),
            "mp_dominant_amplitude": float(live_snapshot.get("dominant_amplitude", np.nan)),
            "mp_signal_std": float(live_snapshot.get("signal_std", np.nan)),
        }
        selection_section = {
            "mp_rank_used": int(live_snapshot.get("rank_used", -1)),
            "mp_order_selection_enabled": bool(
                live_snapshot.get("order_selection_enabled", self.mp_order_selection_enabled)
            ),
            "mp_order_candidates_used": order_candidates_used,
            "mp_attempt_count": int(live_snapshot.get("attempt_count", len(attempts))),
            "mp_best_attempt_idx": int(live_snapshot.get("best_attempt_idx", -1)),
            "mp_best_rank": int(live_snapshot.get("best_rank", -1)),
            "mp_order_stable": int(live_snapshot.get("order_stable", 0)),
            "mp_order_select_reason": str(live_snapshot.get("order_select_reason", "not_reported")),
            "mp_attempts": attempts,
            "mp_modes": modes,
        }
        downsample_section = {
            "mp_downsample_applied": 0,
            "mp_downsample_reason": "preview_not_applied" if bool(self.mp_downsample_enabled) else "disabled",
            "mp_downsample_n_samples_in": int(live_snapshot.get("n_samples", 0)),
            "mp_downsample_n_samples_out": int(live_snapshot.get("n_samples", 0)),
        }
        rec.update(
            _merge_mp_record_sections(
                provenance_section,
                analysis_section,
                window_section,
                profile_section,
                decay_section,
                result_quality_section,
                selection_section,
                downsample_section,
            )
        )
        return _apply_primary_mode_aliases(rec)

    def _fit_window_full(
        self,
        *,
        t_src: np.ndarray,
        v_src: np.ndarray,
        w0: float,
        w1: float,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """Run postprep + optional downsample + MP on one bounded time window."""

        m = np.isfinite(t_src) & np.isfinite(v_src) & (t_src >= float(w0)) & (t_src <= float(w1))
        t_win = t_src[m]
        v_win = v_src[m]
        v_win_clean, _edit_mask, postprep_summary = apply_modal_postprep(
            v_win,
            enabled=bool(self.modal_preprocess_enabled),
            method=str(self.modal_preprocess_method),
            hampel_half_window=int(self.modal_preprocess_hampel_half_window),
            hampel_nsigma=float(self.modal_preprocess_hampel_nsigma),
            max_repair_points=int(self.modal_preprocess_max_repair_points),
            max_repair_fraction=float(self.modal_preprocess_max_repair_fraction),
            repair_mode=str(self.modal_preprocess_repair_mode),
            keep_raw_summary=bool(self.modal_preprocess_keep_raw_summary),
        )
        t_mp, v_mp, downsample_summary = _downsample_mp_window_if_needed(
            t_win,
            v_win_clean,
            enabled=bool(self.mp_downsample_enabled),
            target_fs_hz=float(self.mp_target_fs_hz),
            lpf_cutoff_hz=float(self.mp_downsample_lpf_cutoff_hz),
            lpf_order=int(self.mp_downsample_lpf_order),
            min_samples=int(self.mp_min_samples),
        )
        fit = _run_matrix_pencil_on_window(
            t_mp,
            v_mp,
            mp_model_order=int(self.mp_model_order),
            mp_max_modes=int(self.mp_max_modes),
            mp_freq_low_hz=float(self.mp_freq_low_hz),
            mp_freq_high_hz=float(self.mp_freq_high_hz),
            mp_min_samples=int(self.mp_min_samples),
            mp_dt_cv_max=float(self.mp_dt_cv_max),
            mp_signal_std_min=float(self.mp_signal_std_min),
            mp_singular_ratio_min=float(self.mp_singular_ratio_min),
            mp_order_selection_enabled=bool(self.mp_order_selection_enabled),
            mp_order_candidates=self.mp_order_candidates,
        )
        return fit, postprep_summary, downsample_summary

    def _analyze_interval(
        self,
        *,
        interval_ev: dict,
        rms_event_win_sec: float,
        samples: Sequence[tuple[float, float]],
    ) -> dict[str, object]:
        """Run interval-local MP analysis from captured raw input reslice."""

        rec = _build_mp_interval_record_base(interval_ev)
        rec.update(
            default_postprep_summary(
                enabled=bool(self.modal_preprocess_enabled),
                method=str(self.modal_preprocess_method),
                repair_mode=str(self.modal_preprocess_repair_mode),
                keep_raw_summary=bool(self.modal_preprocess_keep_raw_summary),
            )
        )
        window_section = {
            "mp_window_source": "none",
            "mp_window_start_t": float("nan"),
            "mp_window_end_t": float("nan"),
            "mp_window_duration_sec": float("nan"),
            "mp_rms_event_win_sec": float(rms_event_win_sec),
        }
        provenance_section = {
            "mp_update_cadence_sec": float(self.mp_update_cadence_sec),
            "mp_live_freq_hint_hz": float(interval_ev.get("mp_live_freq_hint_hz", np.nan)),
        }
        profile_section = {
            "mp_freq_profile": "mid",
            "mp_freq_window_sec_target": float(self.mp_freq_window_mid_sec),
            "mp_decay_profile": "mid",
            "mp_decay_window_sec_target": float(self.mp_decay_window_mid_sec),
        }
        decay_section = {
            "mp_decay_window_source": "none",
            "mp_decay_window_start_t": float("nan"),
            "mp_decay_window_end_t": float("nan"),
            "mp_decay_window_duration_sec": float("nan"),
            "mp_decay_qualified": 0,
            "mp_decay_qualified_reason": "not_executed",
            "mp_decay_fit_status": "not_executed",
            "mp_decay_fit_reason": "not_executed",
            "mp_decay_fit_freq_hz": float("nan"),
            "mp_decay_freq_jump_rel": float("nan"),
            "mp_decay_rms_slope_1_per_sec": float("nan"),
            "mp_decay_rms_r2": float("nan"),
            "mp_decay_rms_n_windows": 0,
            "mp_effective_decay_rate_1_per_sec": float("nan"),
            "mp_half_life_sec": float("nan"),
            "mp_freq_window_damping_per_sec": float("nan"),
            "mp_freq_window_damping_ratio": float("nan"),
        }
        result_quality_section = {
            "mp_n_samples": 0,
            "mp_fit_r2": float("nan"),
            "mp_mode_count": 0,
            "mp_dominant_freq_hz": float("nan"),
            "mp_dominant_damping_per_sec": float("nan"),
            "mp_dominant_damping_ratio": float("nan"),
            "mp_dominant_amplitude": float("nan"),
            "mp_signal_std": float("nan"),
        }
        selection_section = {
            "mp_rank_used": -1,
            "mp_order_selection_enabled": bool(self.mp_order_selection_enabled),
            "mp_order_candidates_used": [int(x) for x in self.mp_order_candidates],
            "mp_attempt_count": 0,
            "mp_best_attempt_idx": -1,
            "mp_best_rank": -1,
            "mp_order_stable": 0,
            "mp_order_select_reason": "not_executed",
            "mp_attempts": [],
            "mp_modes": [],
        }
        downsample_section = {
            "mp_downsample_enabled": bool(self.mp_downsample_enabled),
            "mp_downsample_applied": 0,
            "mp_downsample_reason": "disabled" if (not bool(self.mp_downsample_enabled)) else "not_applied",
            "mp_downsample_source_fs_hz": float("nan"),
            "mp_downsample_target_fs_hz": float(self.mp_target_fs_hz),
            "mp_downsample_lpf_cutoff_hz": float(self.mp_downsample_lpf_cutoff_hz),
            "mp_downsample_lpf_order": int(self.mp_downsample_lpf_order),
            "mp_downsample_n_samples_in": 0,
            "mp_downsample_n_samples_out": 0,
        }
        rec.update(
            _merge_mp_record_sections(
                window_section,
                provenance_section,
                profile_section,
                decay_section,
                result_quality_section,
                selection_section,
                downsample_section,
            )
        )
        start_t = float(interval_ev.get("start_t", np.nan))
        end_t = float(interval_ev.get("end_t", np.nan))
        duration_sec = float(interval_ev.get("duration_sec", np.nan))
        if (not np.isfinite(start_t)) or (not np.isfinite(end_t)) or (float(end_t) <= float(start_t)):
            rec.update({"mp_status": "skipped", "mp_reason": "invalid_interval"})
            return _apply_primary_mode_aliases(rec)
        if (not np.isfinite(duration_sec)) or (float(duration_sec) < float(self.mp_min_interval_sec)):
            rec.update({"mp_status": "skipped", "mp_reason": "interval_too_short"})
            return _apply_primary_mode_aliases(rec)

        if not samples:
            rec.update({"mp_status": "skipped", "mp_reason": "missing_interval_capture"})
            return _apply_primary_mode_aliases(rec)
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            rec.update({"mp_status": "skipped", "mp_reason": "invalid_raw_samples"})
            return _apply_primary_mode_aliases(rec)
        t_src = arr[:, 0]
        v_src = arr[:, 1]

        live_hint_hz = float(rec.get("mp_live_freq_hint_hz", np.nan))
        provisional_freq_hz = float(live_hint_hz)
        if (not np.isfinite(provisional_freq_hz)) or (provisional_freq_hz <= 0.0):
            _p_src, pw0, pw1 = _select_mp_analysis_window(
                start_t=float(start_t),
                end_t=float(end_t),
                duration_sec=float(duration_sec),
                rms_event_win_sec=float(rms_event_win_sec),
                mp_onset_skip_sec=float(self.mp_onset_skip_sec),
                mp_onset_window_sec=float(self.mp_onset_window_sec),
                mp_onset_min_window_sec=float(self.mp_onset_min_window_sec),
                mp_fallback_use_rms_event_window=bool(self.mp_fallback_use_rms_event_window),
                mp_fallback_default_window_sec=float(self.mp_fallback_default_window_sec),
            )
            if np.isfinite(pw0) and np.isfinite(pw1) and (float(pw1) > float(pw0)):
                m0 = np.isfinite(t_src) & np.isfinite(v_src) & (t_src >= float(pw0)) & (t_src <= float(pw1))
                if int(np.count_nonzero(m0)) >= int(self.mp_min_samples):
                    p_fit = _run_matrix_pencil_on_window(
                        t_src[m0],
                        v_src[m0],
                        mp_model_order=int(self.mp_model_order),
                        mp_max_modes=int(self.mp_max_modes),
                        mp_freq_low_hz=float(self.mp_freq_low_hz),
                        mp_freq_high_hz=float(self.mp_freq_high_hz),
                        mp_min_samples=int(self.mp_min_samples),
                        mp_dt_cv_max=float(self.mp_dt_cv_max),
                        mp_signal_std_min=float(self.mp_signal_std_min),
                        mp_singular_ratio_min=float(self.mp_singular_ratio_min),
                        mp_order_selection_enabled=bool(self.mp_order_selection_enabled),
                        mp_order_candidates=self.mp_order_candidates,
                    )
                    if str(p_fit.get("mp_status", "")) == "ok":
                        f0 = float(p_fit.get("mp_dominant_freq_hz", np.nan))
                        if np.isfinite(f0) and (f0 > 0.0):
                            provisional_freq_hz = float(f0)

        freq_profile = _adaptive_profile_from_freq(
            float(provisional_freq_hz),
            low_band_max_hz=float(self.mp_adaptive_low_band_max_hz),
            high_band_min_hz=float(self.mp_adaptive_high_band_min_hz),
        )
        freq_window_sec = _window_sec_for_profile(
            str(freq_profile),
            high_sec=float(self.mp_freq_window_high_sec),
            mid_sec=float(self.mp_freq_window_mid_sec),
            low_sec=float(self.mp_freq_window_low_sec),
        )
        rec["mp_freq_profile"] = str(freq_profile)
        rec["mp_freq_window_sec_target"] = float(freq_window_sec)

        window_source, w0, w1 = _select_mp_analysis_window(
            start_t=float(start_t),
            end_t=float(end_t),
            duration_sec=float(duration_sec),
            rms_event_win_sec=float(rms_event_win_sec),
            mp_onset_skip_sec=float(self.mp_onset_skip_sec),
            mp_onset_window_sec=float(freq_window_sec),
            mp_onset_min_window_sec=float(self.mp_onset_min_window_sec),
            mp_fallback_use_rms_event_window=bool(self.mp_fallback_use_rms_event_window),
            mp_fallback_default_window_sec=float(freq_window_sec),
        )
        if (not np.isfinite(w0)) or (not np.isfinite(w1)) or (float(w1) <= float(w0)):
            rec.update({"mp_status": "skipped", "mp_reason": "no_valid_window", "mp_window_source": str(window_source)})
            return _apply_primary_mode_aliases(rec)

        fit, postprep_summary, downsample_summary = self._fit_window_full(
            t_src=t_src,
            v_src=v_src,
            w0=float(w0),
            w1=float(w1),
        )
        rec.update(
            {
                "mp_status": str(fit.get("mp_status", "failed")),
                "mp_reason": str(fit.get("mp_reason", "unknown")),
                "mp_window_source": str(window_source),
                "mp_window_start_t": float(w0),
                "mp_window_end_t": float(w1),
                "mp_window_duration_sec": float(w1 - w0),
                "mp_rms_event_win_sec": float(rms_event_win_sec),
            }
        )
        rec.update(fit)
        rec.update(postprep_summary)
        rec.update(downsample_summary)

        dominant_freq_hz = float(rec.get("mp_dominant_freq_hz", np.nan))
        if ((not np.isfinite(dominant_freq_hz)) or (dominant_freq_hz <= 0.0)) and np.isfinite(provisional_freq_hz):
            dominant_freq_hz = float(provisional_freq_hz)
            rec["mp_dominant_freq_hz"] = float(dominant_freq_hz)
        rec["mp_freq_window_damping_per_sec"] = float(_get_primary_mode_signed_rate_per_sec(rec))
        rec["mp_freq_window_damping_ratio"] = _damping_ratio_from(
            float(dominant_freq_hz),
            float(rec.get("mp_freq_window_damping_per_sec", np.nan)),
        )

        decay_profile = _adaptive_profile_from_freq(
            float(dominant_freq_hz),
            low_band_max_hz=float(self.mp_adaptive_low_band_max_hz),
            high_band_min_hz=float(self.mp_adaptive_high_band_min_hz),
        )
        decay_window_sec = _window_sec_for_profile(
            str(decay_profile),
            high_sec=float(self.mp_decay_window_high_sec),
            mid_sec=float(self.mp_decay_window_mid_sec),
            low_sec=float(self.mp_decay_window_low_sec),
        )
        rec["mp_decay_profile"] = str(decay_profile)
        rec["mp_decay_window_sec_target"] = float(decay_window_sec)
        decay_source, dw0, dw1 = _select_tail_analysis_window(
            start_t=float(start_t),
            end_t=float(end_t),
            duration_sec=float(duration_sec),
            tail_window_sec=float(decay_window_sec),
            min_window_sec=float(self.mp_onset_min_window_sec),
        )
        rec["mp_decay_window_source"] = str(decay_source)
        rec["mp_decay_window_start_t"] = float(dw0)
        rec["mp_decay_window_end_t"] = float(dw1)
        rec["mp_decay_window_duration_sec"] = (
            float(dw1 - dw0) if np.isfinite(dw0) and np.isfinite(dw1) and (float(dw1) > float(dw0)) else float("nan")
        )

        rec["mp_dominant_damping_per_sec"] = float("nan")
        rec["mp_dominant_damping_ratio"] = float("nan")
        rec["mp_effective_decay_rate_1_per_sec"] = float("nan")
        rec["mp_half_life_sec"] = float("nan")

        if str(rec.get("mp_status", "")) != "ok":
            rec["mp_decay_qualified_reason"] = "freq_fit_not_ok"
        elif (not np.isfinite(dw0)) or (not np.isfinite(dw1)) or (float(dw1) <= float(dw0)):
            rec["mp_decay_qualified_reason"] = "no_valid_decay_window"
        else:
            decay_fit, _dec_postprep, _dec_downsample = self._fit_window_full(
                t_src=t_src,
                v_src=v_src,
                w0=float(dw0),
                w1=float(dw1),
            )
            rec["mp_decay_fit_status"] = str(decay_fit.get("mp_status", "failed"))
            rec["mp_decay_fit_reason"] = str(decay_fit.get("mp_reason", "unknown"))
            decay_freq_hz = float(decay_fit.get("mp_dominant_freq_hz", np.nan))
            rec["mp_decay_fit_freq_hz"] = float(decay_freq_hz)

            m_decay = np.isfinite(t_src) & np.isfinite(v_src) & (t_src >= float(dw0)) & (t_src <= float(dw1))
            t_decay = t_src[m_decay]
            v_decay = v_src[m_decay]
            rms_decay, rms_decay_r2, rms_decay_n = _local_rms_decay_from_signal(
                t_decay,
                v_decay,
                trailing_sec=float(max(0.0, float(dw1 - dw0))),
                rms_win_sec=float(self.mp_decay_rms_win_sec),
                step_sec=float(self.mp_decay_step_sec),
                min_windows=int(self.mp_decay_min_windows),
            )
            rec["mp_decay_rms_slope_1_per_sec"] = float(rms_decay)
            rec["mp_decay_rms_r2"] = float(rms_decay_r2)
            rec["mp_decay_rms_n_windows"] = int(rms_decay_n)

            freq_jump_rel = float("nan")
            if np.isfinite(dominant_freq_hz) and (dominant_freq_hz > 0.0) and np.isfinite(decay_freq_hz) and (decay_freq_hz > 0.0):
                freq_jump_rel = abs(float(decay_freq_hz) - float(dominant_freq_hz)) / max(float(dominant_freq_hz), 1e-9)
            rec["mp_decay_freq_jump_rel"] = float(freq_jump_rel)

            decay_dir_ok = np.isfinite(rms_decay) and (float(rms_decay) > 0.0)
            decay_r2_ok = np.isfinite(rms_decay_r2) and (float(rms_decay_r2) >= float(self.mp_decay_min_r2))
            decay_freq_ok = np.isfinite(freq_jump_rel) and (float(freq_jump_rel) <= float(self.mp_decay_freq_jump_rel_max))
            decay_fit_ok = str(decay_fit.get("mp_status", "")) == "ok"

            if decay_fit_ok and decay_dir_ok and decay_r2_ok and decay_freq_ok:
                eff_decay = float(rms_decay)
                rec["mp_decay_qualified"] = 1
                rec["mp_decay_qualified_reason"] = "ok"
                rec["mp_effective_decay_rate_1_per_sec"] = float(eff_decay)
                rec["mp_half_life_sec"] = (float(np.log(2.0) / eff_decay) if (eff_decay > 0.0) else float("nan"))
                rec["mp_dominant_damping_per_sec"] = float(eff_decay)
                rec["mp_dominant_damping_ratio"] = _damping_ratio_from(float(dominant_freq_hz), float(eff_decay))
            else:
                reason_parts: list[str] = []
                if not decay_fit_ok:
                    reason_parts.append("decay_mp_fail")
                if not decay_dir_ok:
                    reason_parts.append("decay_direction_fail")
                if not decay_r2_ok:
                    reason_parts.append("decay_r2_fail")
                if not decay_freq_ok:
                    reason_parts.append("decay_freq_jump_fail")
                rec["mp_decay_qualified_reason"] = "+".join(reason_parts) if reason_parts else "decay_not_qualified"
        if "mp_modes" not in rec:
            rec["mp_modes"] = []
        if "mp_mode_count" not in rec:
            rec["mp_mode_count"] = int(len(rec["mp_modes"])) if isinstance(rec.get("mp_modes"), list) else 0
        if "mp_fit_r2" not in rec:
            rec["mp_fit_r2"] = float("nan")
        if "mp_dominant_freq_hz" not in rec:
            rec["mp_dominant_freq_hz"] = float("nan")
        if "mp_dominant_damping_per_sec" not in rec:
            rec["mp_dominant_damping_per_sec"] = float("nan")
        if "mp_dominant_damping_ratio" not in rec:
            rec["mp_dominant_damping_ratio"] = _damping_ratio_from(
                float(rec.get("mp_dominant_freq_hz", np.nan)),
                float(_get_primary_mode_signed_rate_per_sec(rec)),
            )
        if "mp_dominant_amplitude" not in rec:
            rec["mp_dominant_amplitude"] = float("nan")
        if "mp_signal_std" not in rec:
            rec["mp_signal_std"] = float("nan")
        if "mp_rank_used" not in rec:
            rec["mp_rank_used"] = -1
        if "mp_order_selection_enabled" not in rec:
            rec["mp_order_selection_enabled"] = bool(self.mp_order_selection_enabled)
        if "mp_order_candidates_used" not in rec:
            rec["mp_order_candidates_used"] = [int(x) for x in self.mp_order_candidates]
        if "mp_attempt_count" not in rec:
            rec["mp_attempt_count"] = int(len(rec.get("mp_attempts", []))) if isinstance(rec.get("mp_attempts"), list) else 0
        if "mp_best_attempt_idx" not in rec:
            rec["mp_best_attempt_idx"] = -1
        if "mp_best_rank" not in rec:
            rec["mp_best_rank"] = int(rec.get("mp_rank_used", -1))
        if "mp_order_stable" not in rec:
            rec["mp_order_stable"] = 0
        if "mp_order_select_reason" not in rec:
            rec["mp_order_select_reason"] = "not_reported"
        if "mp_attempts" not in rec:
            rec["mp_attempts"] = []
        return _apply_primary_mode_aliases(rec)


def _extract_window_slice(
    st: ChannelStreamState,
    *,
    max_keep_sec: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
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

    return tw0, vw0, tw, vw, float(dt), float(t0), float(t1)



__all__ = [
    "_build_mp_interval_record_base",
    "_select_mp_analysis_window",
    "_run_matrix_pencil_on_window",
    "_IntervalMPPostRuntime",
    "_extract_window_slice",
]
