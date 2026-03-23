"""Prony post-analysis functions for modular streaming detector."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Sequence

import numpy as np

from .osc_core_postprep_modul import apply_modal_postprep, default_postprep_summary
from .osc_core_mp_modul import (
    _select_mp_analysis_window,
)
from .osc_core_signal_modul import _estimate_dt_from_timestamps, _trim_ring_by_time
from .osc_state_modul import ChannelStreamState


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


def _build_prony_interval_record_base(interval_ev: dict) -> dict[str, object]:
    """Build common interval metadata for Prony post-analysis records."""

    return {
        "event": "interval_analysis_prony",
        "analysis_type": "prony_post",
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


def _pick_interval_reference_freq(interval_ev: dict) -> tuple[str, float]:
    """Pick reference periodicity freq for representative-mode selection."""

    f_welch = float(interval_ev.get("f_welch", np.nan))
    if np.isfinite(f_welch) and (f_welch > 0.0):
        return "f_welch", float(f_welch)
    acf_period_sec = float(interval_ev.get("acf_period_sec", np.nan))
    if np.isfinite(acf_period_sec) and (acf_period_sec > 0.0):
        return "acf_period_sec", float(1.0 / max(acf_period_sec, 1e-12))
    return "none", float("nan")


def _build_reference_mode_fields(modes: object, ref_source: str, ref_freq_hz: float) -> dict[str, object]:
    """Build nearest-reference-mode summary fields."""

    out = {
        "prony_ref_source": str(ref_source),
        "prony_ref_freq_hz": float(ref_freq_hz),
        "prony_ref_mode_rank": -1,
        "prony_ref_mode_freq_hz": float("nan"),
        "prony_ref_mode_damping_per_sec": float("nan"),
        "prony_ref_mode_damping_ratio": float("nan"),
        "prony_ref_mode_amplitude": float("nan"),
        "prony_ref_mode_freq_abs_err_hz": float("nan"),
    }
    if (not np.isfinite(ref_freq_hz)) or (ref_freq_hz <= 0.0):
        return out
    if not isinstance(modes, list):
        return out

    best_idx = -1
    best_err = float("inf")
    for i, mode in enumerate(modes):
        if not isinstance(mode, dict):
            continue
        f_mode = float(mode.get("freq_hz", np.nan))
        if (not np.isfinite(f_mode)) or (f_mode <= 0.0):
            continue
        err = abs(float(f_mode) - float(ref_freq_hz))
        if err < best_err:
            best_err = float(err)
            best_idx = int(i)
    if best_idx < 0:
        return out

    mode = modes[best_idx]
    out.update(
        {
            "prony_ref_mode_rank": int(mode.get("rank", best_idx + 1)),
            "prony_ref_mode_freq_hz": float(mode.get("freq_hz", np.nan)),
            "prony_ref_mode_damping_per_sec": float(mode.get("damping_per_sec", np.nan)),
            "prony_ref_mode_damping_ratio": float(mode.get("damping_ratio", np.nan)),
            "prony_ref_mode_amplitude": float(mode.get("amplitude", np.nan)),
            "prony_ref_mode_freq_abs_err_hz": float(best_err),
        }
    )
    return out


def _clean_order_candidates(prony_model_order: int, prony_order_candidates: Sequence[int] | None) -> list[int]:
    """Normalize order-candidate sequence while preserving user order."""

    raw: list[int] = []
    if prony_order_candidates is not None:
        for x in prony_order_candidates:
            try:
                raw.append(int(x))
            except Exception:
                continue
    if not raw:
        raw = [int(prony_model_order)]
    out: list[int] = []
    seen: set[int] = set()
    for x in raw:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _select_conjugate_pairs(
    candidates: Sequence[tuple[complex, float, float, float, float]],
    *,
    imag_eps: float = 1e-8,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-4,
) -> list[tuple[tuple[complex, float, float, float, float], tuple[complex, float, float, float, float], float]]:
    """Greedily match positive-imag roots with negative-imag conjugates."""

    if not candidates:
        return []
    used: set[int] = set()
    pairs: list[tuple[tuple[complex, float, float, float, float], tuple[complex, float, float, float, float], float]] = []
    for i, ci in enumerate(candidates):
        if i in used:
            continue
        ri = complex(ci[0])
        if float(np.imag(ri)) <= float(imag_eps):
            continue
        target = np.conjugate(ri)
        best_j = -1
        best_err = float("inf")
        for j, cj in enumerate(candidates):
            if (j == i) or (j in used):
                continue
            rj = complex(cj[0])
            if float(np.imag(rj)) >= -float(imag_eps):
                continue
            err = abs(rj - target)
            if err < best_err:
                best_err = float(err)
                best_j = int(j)
        if best_j < 0:
            continue
        tol = float(max(abs_tol, rel_tol * max(1.0, float(abs(ri)))))
        if best_err > tol:
            continue
        used.add(int(i))
        used.add(int(best_j))
        pairs.append((candidates[i], candidates[best_j], float(best_err)))
    return pairs


def _run_prony_single_order(
    *,
    x: np.ndarray,
    dt: float,
    n: int,
    order_raw: int,
    order_max: int,
    prony_max_modes: int,
    prony_freq_low_hz: float,
    prony_freq_high_hz: float,
    prony_root_mag_max: float,
) -> dict[str, object]:
    """Run one Prony attempt for one AR order with diagnostic fields."""

    order_used = int(min(max(2, int(order_raw)), int(order_max)))
    out: dict[str, object] = {
        "prony_order_used": int(order_used),
        "prony_model_order_used": int(order_used),
        "prony_roots_total": 0,
        "prony_roots_after_mag": 0,
        "prony_roots_after_freq": 0,
        "prony_roots_after_pair": 0,
    }
    if order_used < 2:
        out.update({"prony_status": "skipped", "prony_reason": "model_order_too_high"})
        return out
    if (n - order_used) <= 1:
        out.update({"prony_status": "skipped", "prony_reason": "window_too_short_for_model"})
        return out

    try:
        y = -x[order_used:]
        x_cols = [x[(order_used - k - 1) : (n - k - 1)] for k in range(order_used)]
        x_mat = np.column_stack(x_cols)
        ar_coef, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    except Exception:
        out.update({"prony_status": "failed", "prony_reason": "ar_fit_failed"})
        return out
    if ar_coef.size != order_used:
        out.update({"prony_status": "failed", "prony_reason": "ar_fit_invalid"})
        return out

    try:
        poly = np.concatenate(([1.0], np.asarray(ar_coef, dtype=float)))
        roots = np.roots(poly)
    except Exception:
        out.update({"prony_status": "failed", "prony_reason": "root_solve_failed"})
        return out
    out["prony_roots_total"] = int(roots.size)
    if roots.size == 0:
        out.update({"prony_status": "skipped", "prony_reason": "no_roots"})
        return out

    roots_mag: list[complex] = []
    for ri in roots:
        mag = float(np.abs(ri))
        if (not np.isfinite(mag)) or (mag <= 1e-12) or (mag > float(prony_root_mag_max)):
            continue
        roots_mag.append(complex(ri))
    out["prony_roots_after_mag"] = int(len(roots_mag))
    if not roots_mag:
        out.update({"prony_status": "skipped", "prony_reason": "no_modes_after_mag_filter"})
        return out

    roots_freq: list[tuple[complex, float, float, float, float]] = []
    for ri in roots_mag:
        s = np.log(complex(ri)) / max(float(dt), 1e-12)
        sigma = float(np.real(s))
        omega = float(np.imag(s))
        if (not np.isfinite(sigma)) or (not np.isfinite(omega)):
            continue
        freq_hz = float(abs(omega) / (2.0 * np.pi))
        if (not np.isfinite(freq_hz)) or (freq_hz < float(prony_freq_low_hz)) or (freq_hz > float(prony_freq_high_hz)):
            continue
        damping = float(-sigma)
        if not np.isfinite(damping):
            continue
        roots_freq.append((complex(ri), float(freq_hz), float(damping), float(abs(ri)), float(np.angle(ri))))
    out["prony_roots_after_freq"] = int(len(roots_freq))
    if not roots_freq:
        out.update({"prony_status": "skipped", "prony_reason": "no_modes_in_band"})
        return out

    pairs = _select_conjugate_pairs(roots_freq)
    out["prony_roots_after_pair"] = int(len(pairs) * 2)
    if not pairs:
        out.update({"prony_status": "skipped", "prony_reason": "no_conjugate_pairs_in_band"})
        return out

    roots_keep: list[complex] = []
    mode_meta: list[tuple[float, float, float, float, int]] = []
    for idx, (pos_root_meta, neg_root_meta, pair_err) in enumerate(pairs, start=1):
        r_pos, freq_hz, damping, mag, phase_rad = pos_root_meta
        r_neg = complex(neg_root_meta[0])
        roots_keep.extend([complex(r_pos), complex(r_neg)])
        mode_meta.append((float(freq_hz), float(damping), float(mag), float(phase_rad), int(idx)))

    try:
        n_idx = np.arange(n, dtype=float)
        z_mat = np.power(np.asarray(roots_keep, dtype=np.complex128)[None, :], n_idx[:, None])
        coef, *_ = np.linalg.lstsq(z_mat, x.astype(np.complex128), rcond=None)
        x_hat = np.real(z_mat @ coef)
    except Exception:
        out.update({"prony_status": "failed", "prony_reason": "amplitude_fit_failed"})
        return out

    sst = float(np.sum((x - float(np.mean(x))) ** 2))
    sse = float(np.sum((x - x_hat) ** 2))
    fit_r2 = float(1.0 - (sse / sst)) if sst > 1e-12 else float("nan")

    modes: list[dict[str, float | int]] = []
    coef_abs = np.abs(np.asarray(coef, dtype=np.complex128))
    for i, (freq_hz, damping, mag, phase_rad, pair_id) in enumerate(mode_meta):
        c0 = float(coef_abs[(2 * i)]) if (2 * i) < coef_abs.size else float("nan")
        c1 = float(coef_abs[(2 * i) + 1]) if ((2 * i) + 1) < coef_abs.size else float("nan")
        amp = float(0.5 * (c0 + c1)) if np.isfinite(c0) and np.isfinite(c1) else float("nan")
        damp_ratio = _damping_ratio_from(float(freq_hz), float(damping))
        modes.append(
            {
                "rank": int(i + 1),
                "pair_id": int(pair_id),
                "freq_hz": float(freq_hz),
                "damping_per_sec": float(damping),
                "damping_ratio": float(damp_ratio),
                "amplitude": float(amp),
                "pole_mag": float(mag),
                "pole_phase_rad": float(phase_rad),
            }
        )
    modes = sorted(modes, key=lambda m_: float(m_.get("amplitude", np.nan)), reverse=True)
    modes = modes[: max(1, int(prony_max_modes))]
    if not modes:
        out.update({"prony_status": "skipped", "prony_reason": "no_modes_after_pairing", "prony_fit_r2": float(fit_r2), "prony_modes": [], "prony_mode_count": 0})
        return out

    dom = modes[0]
    out.update(
        {
            "prony_status": "ok",
            "prony_reason": "ok",
            "prony_fit_r2": float(fit_r2),
            "prony_mode_count": int(len(modes)),
            "prony_dominant_freq_hz": float(dom.get("freq_hz", np.nan)),
            "prony_dominant_damping_per_sec": float(dom.get("damping_per_sec", np.nan)),
            "prony_dominant_damping_ratio": float(dom.get("damping_ratio", np.nan)),
            "prony_dominant_amplitude": float(dom.get("amplitude", np.nan)),
            "prony_modes": modes,
        }
    )
    return out


def _run_prony_on_window(
    t_win: np.ndarray,
    v_win: np.ndarray,
    *,
    prony_model_order: int,
    prony_order_candidates: Sequence[int] | None,
    prony_max_modes: int,
    prony_freq_low_hz: float,
    prony_freq_high_hz: float,
    prony_min_samples: int,
    prony_dt_cv_max: float,
    prony_signal_std_min: float,
    prony_root_mag_max: float,
) -> dict[str, object]:
    """Estimate local modal components on one window using Prony (order sweep)."""

    t_arr = np.asarray(t_win, dtype=float).reshape(-1)
    v_arr = np.asarray(v_win, dtype=float).reshape(-1)
    m = np.isfinite(t_arr) & np.isfinite(v_arr)
    t_arr = t_arr[m]
    v_arr = v_arr[m]
    if t_arr.size < int(prony_min_samples):
        return {"prony_status": "skipped", "prony_reason": "too_few_samples", "prony_best_attempt_idx": -1}

    order = np.argsort(t_arr, kind="mergesort")
    t_arr = t_arr[order]
    v_arr = v_arr[order]
    dt_all = np.diff(t_arr)
    dt_pos = dt_all[np.isfinite(dt_all) & (dt_all > 0.0)]
    if dt_pos.size < 2:
        return {"prony_status": "skipped", "prony_reason": "invalid_dt", "prony_best_attempt_idx": -1}

    dt = float(np.median(dt_pos))
    if (not np.isfinite(dt)) or (dt <= 0.0):
        return {"prony_status": "skipped", "prony_reason": "invalid_dt", "prony_best_attempt_idx": -1}
    dt_cv = float(np.std(dt_pos) / max(dt, 1e-12))

    if np.isfinite(dt_cv) and (dt_cv > float(prony_dt_cv_max)):
        n_uniform = int(max(2, np.floor((float(t_arr[-1]) - float(t_arr[0])) / dt) + 1))
        t_uni = float(t_arr[0]) + (np.arange(n_uniform, dtype=float) * dt)
        t_uni = t_uni[t_uni <= (float(t_arr[-1]) + 0.5 * dt)]
        if t_uni.size < int(prony_min_samples):
            return {
                "prony_status": "skipped",
                "prony_reason": "too_few_samples_after_resample",
                "prony_best_attempt_idx": -1,
                "prony_dt_sec": float(dt),
                "prony_dt_cv": float(dt_cv),
            }
        v_uni = np.interp(t_uni, t_arr, v_arr)
        t_arr = t_uni
        v_arr = v_uni

    x = np.asarray(v_arr, dtype=float)
    x = x - float(np.mean(x))
    signal_std = float(np.std(x))
    if (not np.isfinite(signal_std)) or (signal_std < float(prony_signal_std_min)):
        return {
            "prony_status": "skipped",
            "prony_reason": "low_signal_std",
            "prony_best_attempt_idx": -1,
            "prony_dt_sec": float(dt),
            "prony_dt_cv": float(dt_cv),
            "prony_signal_std": float(signal_std),
        }

    n = int(x.size)
    if n < max(int(prony_min_samples), 8):
        return {
            "prony_status": "skipped",
            "prony_reason": "too_few_samples",
            "prony_best_attempt_idx": -1,
            "prony_dt_sec": float(dt),
            "prony_dt_cv": float(dt_cv),
            "prony_signal_std": float(signal_std),
        }

    order_max = int(max(2, min((n - 1) // 2, n - 2)))
    candidates_raw = _clean_order_candidates(int(prony_model_order), prony_order_candidates)
    candidates_eff: list[int] = []
    seen: set[int] = set()
    for c in candidates_raw:
        cc = int(min(max(2, int(c)), order_max))
        if cc in seen:
            continue
        seen.add(cc)
        candidates_eff.append(cc)
    if not candidates_eff:
        candidates_eff = [int(min(max(2, int(prony_model_order)), order_max))]

    attempts: list[dict[str, object]] = []
    ok_attempts: list[dict[str, object]] = []
    for idx, ord_try in enumerate(candidates_eff):
        one = _run_prony_single_order(
            x=x,
            dt=float(dt),
            n=int(n),
            order_raw=int(ord_try),
            order_max=int(order_max),
            prony_max_modes=int(prony_max_modes),
            prony_freq_low_hz=float(prony_freq_low_hz),
            prony_freq_high_hz=float(prony_freq_high_hz),
            prony_root_mag_max=float(prony_root_mag_max),
        )
        one["prony_attempt_idx"] = int(idx)
        attempts.append(one)
        if (str(one.get("prony_status", "")) == "ok") and (int(one.get("prony_mode_count", 0)) > 0):
            ok_attempts.append(one)

    best: dict[str, object] | None = None
    best_idx = -1
    if ok_attempts:
        def _ok_key(a: dict[str, object]) -> tuple[int, float, int, float, int]:
            r2 = float(a.get("prony_fit_r2", np.nan))
            amp = float(a.get("prony_dominant_amplitude", np.nan))
            return (
                int(np.isfinite(r2)),
                float(r2) if np.isfinite(r2) else float("-inf"),
                int(np.isfinite(amp)),
                float(amp) if np.isfinite(amp) else float("-inf"),
                int(a.get("prony_mode_count", 0)),
            )
        best = max(ok_attempts, key=_ok_key)
        best_idx = int(best.get("prony_attempt_idx", -1))
    elif attempts:
        def _fail_key(a: dict[str, object]) -> tuple[int, int, int, int]:
            return (
                int(a.get("prony_roots_after_pair", 0)),
                int(a.get("prony_roots_after_freq", 0)),
                int(a.get("prony_roots_after_mag", 0)),
                int(a.get("prony_roots_total", 0)),
            )
        best = max(attempts, key=_fail_key)
        best_idx = int(best.get("prony_attempt_idx", -1))

    attempt_summary = [
        {
            "idx": int(a.get("prony_attempt_idx", -1)),
            "order": int(a.get("prony_order_used", -1)),
            "status": str(a.get("prony_status", "")),
            "reason": str(a.get("prony_reason", "")),
            "roots_after_pair": int(a.get("prony_roots_after_pair", 0)),
            "roots_after_freq": int(a.get("prony_roots_after_freq", 0)),
            "mode_count": int(a.get("prony_mode_count", 0)),
            "fit_r2": float(a.get("prony_fit_r2", np.nan)),
        }
        for a in attempts
    ]

    if best is None:
        return {
            "prony_status": "skipped",
            "prony_reason": "no_attempt_executed",
            "prony_n_samples": int(n),
            "prony_dt_sec": float(dt),
            "prony_dt_cv": float(dt_cv),
            "prony_signal_std": float(signal_std),
            "prony_order_candidates_used": [int(x) for x in candidates_eff],
            "prony_attempt_count": int(len(attempts)),
            "prony_best_attempt_idx": -1,
            "prony_attempts": attempt_summary,
        }

    out = dict(best)
    out.update(
        {
            "prony_n_samples": int(n),
            "prony_dt_sec": float(dt),
            "prony_dt_cv": float(dt_cv),
            "prony_signal_std": float(signal_std),
            "prony_order_candidates_used": [int(x) for x in candidates_eff],
            "prony_attempt_count": int(len(attempts)),
            "prony_best_attempt_idx": int(best_idx),
            "prony_attempts": attempt_summary,
        }
    )
    if "prony_modes" not in out:
        out["prony_modes"] = []
    if "prony_mode_count" not in out:
        out["prony_mode_count"] = int(len(out["prony_modes"])) if isinstance(out.get("prony_modes"), list) else 0
    return out


class _IntervalPronyPostRuntime:
    """Non-blocking interval post-analysis runtime for Prony."""

    def __init__(
        self,
        *,
        prony_enabled: bool,
        prony_min_interval_sec: float,
        prony_onset_skip_sec: float,
        prony_onset_window_sec: float,
        prony_onset_min_window_sec: float,
        prony_fallback_use_rms_event_window: bool,
        prony_fallback_default_window_sec: float,
        prony_model_order: int,
        prony_order_candidates: Sequence[int] | None,
        prony_max_modes: int,
        prony_freq_low_hz: float,
        prony_freq_high_hz: float,
        prony_min_samples: int,
        prony_dt_cv_max: float,
        prony_signal_std_min: float,
        prony_root_mag_max: float,
        modal_preprocess_enabled: bool,
        modal_preprocess_method: str,
        modal_preprocess_hampel_half_window: int,
        modal_preprocess_hampel_nsigma: float,
        modal_preprocess_max_repair_points: int,
        modal_preprocess_max_repair_fraction: float,
        modal_preprocess_repair_mode: str,
        modal_preprocess_keep_raw_summary: bool,
        prony_async_enabled: bool,
        prony_queue_maxsize: int,
        prony_finalize_wait_sec: float,
    ) -> None:
        self.enabled = bool(prony_enabled)
        self.prony_min_interval_sec = float(prony_min_interval_sec)
        self.prony_onset_skip_sec = float(prony_onset_skip_sec)
        self.prony_onset_window_sec = float(prony_onset_window_sec)
        self.prony_onset_min_window_sec = float(prony_onset_min_window_sec)
        self.prony_fallback_use_rms_event_window = bool(prony_fallback_use_rms_event_window)
        self.prony_fallback_default_window_sec = float(prony_fallback_default_window_sec)
        self.prony_model_order = int(max(2, prony_model_order))
        self.prony_order_candidates = tuple(int(x) for x in _clean_order_candidates(int(self.prony_model_order), prony_order_candidates))
        self.prony_max_modes = int(max(1, prony_max_modes))
        self.prony_freq_low_hz = float(prony_freq_low_hz)
        self.prony_freq_high_hz = float(prony_freq_high_hz)
        self.prony_min_samples = int(max(8, prony_min_samples))
        self.prony_dt_cv_max = float(max(0.0, prony_dt_cv_max))
        self.prony_signal_std_min = float(max(0.0, prony_signal_std_min))
        self.prony_root_mag_max = float(max(1e-6, prony_root_mag_max))
        self.modal_preprocess_enabled = bool(modal_preprocess_enabled)
        self.modal_preprocess_method = str(modal_preprocess_method).strip().lower()
        self.modal_preprocess_hampel_half_window = int(max(1, modal_preprocess_hampel_half_window))
        self.modal_preprocess_hampel_nsigma = float(max(0.0, modal_preprocess_hampel_nsigma))
        self.modal_preprocess_max_repair_points = int(max(0, modal_preprocess_max_repair_points))
        self.modal_preprocess_max_repair_fraction = float(max(0.0, modal_preprocess_max_repair_fraction))
        self.modal_preprocess_repair_mode = str(modal_preprocess_repair_mode).strip().lower()
        self.modal_preprocess_keep_raw_summary = bool(modal_preprocess_keep_raw_summary)
        self.prony_async_enabled = bool(prony_async_enabled)
        self.prony_finalize_wait_sec = float(max(0.0, prony_finalize_wait_sec))

        self._active_capture_by_key: dict[tuple[str, str], dict[str, object]] = {}
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
        if self.enabled and self.prony_async_enabled:
            qsize = int(max(1, prony_queue_maxsize))
            self._queue = queue.Queue(maxsize=qsize)
            self._stop_event = threading.Event()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    def sync_interval_capture(
        self,
        *,
        key: tuple[str, str],
        active_start_t: float,
        is_interval_active: bool,
        tick_samples: Sequence[tuple[float, float]],
    ) -> None:
        """Capture only interval-local raw samples (avoid unbounded channel history)."""

        if not self.enabled:
            return
        if bool(is_interval_active) and np.isfinite(active_start_t):
            cap = self._active_capture_by_key.get(key)
            if cap is None:
                cap = {"start_t": float(active_start_t), "samples": []}
                self._active_capture_by_key[key] = cap
            else:
                prev_start = float(cap.get("start_t", np.nan))
                if (not np.isfinite(prev_start)) or (float(active_start_t) < prev_start):
                    cap["start_t"] = float(active_start_t)
            start_t = float(cap.get("start_t", active_start_t))
            samples = cap.get("samples")
            if not isinstance(samples, list):
                samples = []
                cap["samples"] = samples
            for t, v in tick_samples:
                t_f = float(t)
                v_f = float(v)
                if np.isfinite(t_f) and np.isfinite(v_f) and (t_f >= start_t):
                    samples.append((t_f, v_f))
            return

        self._active_capture_by_key.pop(key, None)

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
        seg_capture = self._active_capture_by_key.pop(key, None)
        seg_samples = seg_capture.get("samples", []) if isinstance(seg_capture, dict) else []
        if not isinstance(seg_samples, list):
            seg_samples = []
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
                rec = self._build_skip_record(
                    interval_ev=interval_copy,
                    reason="queue_full",
                    rms_event_win_sec=float(rms_event_win_sec),
                    window_source="none",
                    window_start_t=float("nan"),
                    window_end_t=float("nan"),
                )
                self._store_record(int(seq), int(interval_id), rec)
                return

        rec = self._analyze_interval(
            interval_ev=interval_copy,
            rms_event_win_sec=float(rms_event_win_sec),
            samples=merged_samples_ro,
        )
        self._store_record(int(seq), int(interval_id), rec)

    def finalize(self) -> list[dict[str, object]]:
        """Stop worker and return interval-keyed post-analysis records."""

        if not self.enabled:
            return []

        if self._stop_event is not None:
            if self._queue is not None:
                wait_deadline = time.time() + float(self.prony_finalize_wait_sec)
                while (time.time() < wait_deadline) and (not self._queue.empty()):
                    time.sleep(0.01)
            self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=float(self.prony_finalize_wait_sec))

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
        self._active_capture_by_key.clear()
        self._interval_capture_by_interval_id.clear()
        return out

    def _worker_loop(self) -> None:
        """Consume analysis jobs in background."""

        assert self._queue is not None
        while True:
            if (self._stop_event is not None) and self._stop_event.is_set():
                break
            try:
                seq, interval_ev, rms_event_win_sec, samples = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            interval_id = int(interval_ev.get("interval_id", -1))
            rec = self._analyze_interval(
                interval_ev=interval_ev,
                rms_event_win_sec=float(rms_event_win_sec),
                samples=samples,
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

        rec = _build_prony_interval_record_base(interval_ev)
        rec.update(
            default_postprep_summary(
                enabled=bool(self.modal_preprocess_enabled),
                method=str(self.modal_preprocess_method),
                repair_mode=str(self.modal_preprocess_repair_mode),
                keep_raw_summary=bool(self.modal_preprocess_keep_raw_summary),
            )
        )
        rec.update(
            {
                "prony_status": "skipped",
                "prony_reason": str(reason),
                "prony_window_source": str(window_source),
                "prony_window_start_t": float(window_start_t),
                "prony_window_end_t": float(window_end_t),
                "prony_window_duration_sec": (
                    float(window_end_t - window_start_t)
                    if np.isfinite(window_start_t) and np.isfinite(window_end_t)
                    else float("nan")
                ),
                "prony_rms_event_win_sec": float(rms_event_win_sec),
                "prony_n_samples": 0,
                "prony_order_used": -1,
                "prony_model_order_used": -1,
                "prony_roots_total": 0,
                "prony_roots_after_mag": 0,
                "prony_roots_after_freq": 0,
                "prony_roots_after_pair": 0,
                "prony_best_attempt_idx": -1,
                "prony_order_candidates_used": [int(x) for x in self.prony_order_candidates],
                "prony_attempt_count": 0,
                "prony_attempts": [],
                "prony_fit_r2": float("nan"),
                "prony_mode_count": 0,
                "prony_dominant_freq_hz": float("nan"),
                "prony_dominant_damping_per_sec": float("nan"),
                "prony_dominant_damping_ratio": float("nan"),
                "prony_dominant_amplitude": float("nan"),
                "prony_signal_std": float("nan"),
                "prony_modes": [],
            }
        )
        ref_source, ref_freq_hz = _pick_interval_reference_freq(interval_ev)
        rec.update(_build_reference_mode_fields(rec.get("prony_modes"), ref_source, ref_freq_hz))
        return rec

    def _analyze_interval(
        self,
        *,
        interval_ev: dict,
        rms_event_win_sec: float,
        samples: Sequence[tuple[float, float]],
    ) -> dict[str, object]:
        """Run interval-local Prony analysis from captured raw input reslice."""

        rec = _build_prony_interval_record_base(interval_ev)
        rec.update(
            default_postprep_summary(
                enabled=bool(self.modal_preprocess_enabled),
                method=str(self.modal_preprocess_method),
                repair_mode=str(self.modal_preprocess_repair_mode),
                keep_raw_summary=bool(self.modal_preprocess_keep_raw_summary),
            )
        )
        rec.update(
            {
                "prony_window_source": "none",
                "prony_window_start_t": float("nan"),
                "prony_window_end_t": float("nan"),
                "prony_window_duration_sec": float("nan"),
                "prony_rms_event_win_sec": float(rms_event_win_sec),
            }
        )
        start_t = float(interval_ev.get("start_t", np.nan))
        end_t = float(interval_ev.get("end_t", np.nan))
        duration_sec = float(interval_ev.get("duration_sec", np.nan))
        if (not np.isfinite(start_t)) or (not np.isfinite(end_t)) or (float(end_t) <= float(start_t)):
            rec.update({"prony_status": "skipped", "prony_reason": "invalid_interval"})
            return rec
        if (not np.isfinite(duration_sec)) or (float(duration_sec) < float(self.prony_min_interval_sec)):
            rec.update({"prony_status": "skipped", "prony_reason": "interval_too_short"})
            return rec

        if not samples:
            rec.update({"prony_status": "skipped", "prony_reason": "missing_interval_capture"})
            return rec
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            rec.update({"prony_status": "skipped", "prony_reason": "invalid_raw_samples"})
            return rec
        t_src = arr[:, 0]
        v_src = arr[:, 1]

        window_source, w0, w1 = _select_mp_analysis_window(
            start_t=float(start_t),
            end_t=float(end_t),
            duration_sec=float(duration_sec),
            rms_event_win_sec=float(rms_event_win_sec),
            mp_onset_skip_sec=float(self.prony_onset_skip_sec),
            mp_onset_window_sec=float(self.prony_onset_window_sec),
            mp_onset_min_window_sec=float(self.prony_onset_min_window_sec),
            mp_fallback_use_rms_event_window=bool(self.prony_fallback_use_rms_event_window),
            mp_fallback_default_window_sec=float(self.prony_fallback_default_window_sec),
        )
        if (not np.isfinite(w0)) or (not np.isfinite(w1)) or (float(w1) <= float(w0)):
            rec.update({"prony_status": "skipped", "prony_reason": "no_valid_window", "prony_window_source": str(window_source)})
            return rec

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
        fit = _run_prony_on_window(
            t_win,
            v_win_clean,
            prony_model_order=int(self.prony_model_order),
            prony_order_candidates=self.prony_order_candidates,
            prony_max_modes=int(self.prony_max_modes),
            prony_freq_low_hz=float(self.prony_freq_low_hz),
            prony_freq_high_hz=float(self.prony_freq_high_hz),
            prony_min_samples=int(self.prony_min_samples),
            prony_dt_cv_max=float(self.prony_dt_cv_max),
            prony_signal_std_min=float(self.prony_signal_std_min),
            prony_root_mag_max=float(self.prony_root_mag_max),
        )

        rec.update(
            {
                "prony_status": str(fit.get("prony_status", "failed")),
                "prony_reason": str(fit.get("prony_reason", "unknown")),
                "prony_window_source": str(window_source),
                "prony_window_start_t": float(w0),
                "prony_window_end_t": float(w1),
                "prony_window_duration_sec": float(w1 - w0),
                "prony_rms_event_win_sec": float(rms_event_win_sec),
            }
        )
        rec.update(fit)
        rec.update(postprep_summary)
        if "prony_order_used" not in rec:
            rec["prony_order_used"] = int(rec.get("prony_model_order_used", -1))
        if "prony_model_order_used" not in rec:
            rec["prony_model_order_used"] = int(rec.get("prony_order_used", -1))
        if "prony_roots_total" not in rec:
            rec["prony_roots_total"] = 0
        if "prony_roots_after_mag" not in rec:
            rec["prony_roots_after_mag"] = 0
        if "prony_roots_after_freq" not in rec:
            rec["prony_roots_after_freq"] = 0
        if "prony_roots_after_pair" not in rec:
            rec["prony_roots_after_pair"] = 0
        if "prony_best_attempt_idx" not in rec:
            rec["prony_best_attempt_idx"] = -1
        if "prony_order_candidates_used" not in rec:
            rec["prony_order_candidates_used"] = [int(x) for x in self.prony_order_candidates]
        if "prony_attempt_count" not in rec:
            rec["prony_attempt_count"] = 0
        if "prony_attempts" not in rec:
            rec["prony_attempts"] = []
        if "prony_fit_r2" not in rec:
            rec["prony_fit_r2"] = float("nan")
        if "prony_modes" not in rec:
            rec["prony_modes"] = []
        if "prony_mode_count" not in rec:
            rec["prony_mode_count"] = int(len(rec["prony_modes"])) if isinstance(rec.get("prony_modes"), list) else 0
        if "prony_dominant_freq_hz" not in rec:
            rec["prony_dominant_freq_hz"] = float("nan")
        if "prony_dominant_damping_per_sec" not in rec:
            rec["prony_dominant_damping_per_sec"] = float("nan")
        if "prony_dominant_damping_ratio" not in rec:
            rec["prony_dominant_damping_ratio"] = _damping_ratio_from(
                float(rec.get("prony_dominant_freq_hz", np.nan)),
                float(rec.get("prony_dominant_damping_per_sec", np.nan)),
            )
        if "prony_dominant_amplitude" not in rec:
            rec["prony_dominant_amplitude"] = float("nan")
        if "prony_signal_std" not in rec:
            rec["prony_signal_std"] = float("nan")

        ref_source, ref_freq_hz = _pick_interval_reference_freq(interval_ev)
        rec.update(_build_reference_mode_fields(rec.get("prony_modes"), ref_source, ref_freq_hz))
        return rec


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
    "_build_prony_interval_record_base",
    "_run_prony_on_window",
    "_IntervalPronyPostRuntime",
    "_extract_window_slice",
]
