"""Decision/FSM functions for modular streaming detector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .osc_config_modul import LongConfig, PeriodicityConfig, ThresholdConfig
from .osc_core_mp_modul import _extract_window_slice
from .osc_core_signal_modul import (
    BASE_SEC,
    RMS_WIN_SEC,
    _acf_peak_periodicity,
    _calibrate_confidence_quantile,
    _clip01,
    _compute_periodicity_quality,
    _dynamic_score_cut_from_log_baseline,
    _event_rms_decay_from_signal,
    _is_risk_active_phase,
    _local_rms_decay_from_signal,
    _long_activity_ratio,
    _normalize01,
    _robust_center_scale,
    _trim_score_hist_by_time,
    compute_band_rms_energies,
    score_one_channel_equiv,
)
from .osc_state_modul import (
    ChannelStreamState,
    DecisionContext,
    PHASE_OFF,
    PHASE_OFF_CANDIDATE,
    PHASE_OFF_CONFIRMED,
    PHASE_ON_CANDIDATE,
    PHASE_ON_CONFIRMED,
    QualityCacheSnapshot,
    TickFeatures,
    TickQualityState,
    TickSignalState,
    TickVoteState,
)


@dataclass
class _QualitySupportSnapshot:
    """Per-tick quality/support bundle used to assemble TickFeatures."""

    acf_peak: float
    acf_period_sec: float
    acf_lag_steps: int
    acf_n: int
    confidence: float
    confidence_raw: float
    confidence_cal: float
    c_acf: float
    c_spec: float
    c_env: float
    c_fft: float
    c_freq_agree: float
    f_welch: float
    f_zc: float
    f_fft: float
    rms_decay: float
    rms_decay_r2: float
    rms_decay_n: int
    rms_decay_on_ok: int
    rms_decay_off_hint: int
    rms_decay_event: float
    rms_decay_event_r2: float
    rms_decay_event_n: int
    rms_decay_event_win_sec: float
    long_ready: bool
    warmup_mode: bool
    e_t: float
    delta_s_log: float
    delta_e: float
    accel_ok: bool
    conf_now: float
    raw_now: float
    use_cal_gate: bool
    cal_on_active: bool
    cal_on_conf_thr: float
    on_support: float
    on_support_ema: float
    on_soft_vote_sum: int
    on_soft_confirmed: bool


def _finite_or_nan(v: float) -> float:
    """Convert finite numeric values to float, otherwise return np.nan."""

    return float(v) if np.isfinite(v) else np.nan


def _quality_cache_from_payload(payload: dict[str, float]) -> QualityCacheSnapshot:
    """Convert raw periodicity-quality dict into typed quality cache snapshot."""

    return QualityCacheSnapshot(
        confidence=float(payload.get("confidence", np.nan)),
        confidence_raw=float(payload.get("confidence_raw", np.nan)),
        confidence_cal=float("nan"),
        confidence_used=float(payload.get("confidence", np.nan)),
        c_acf=float(payload.get("c_acf", np.nan)),
        c_spec=float(payload.get("c_spec", np.nan)),
        c_env=float(payload.get("c_env", np.nan)),
        c_fft=float(payload.get("c_fft", np.nan)),
        c_freq_agree=float(payload.get("c_freq_agree", np.nan)),
        f_welch=float(payload.get("f_welch", np.nan)),
        f_zc=float(payload.get("f_zc", np.nan)),
        f_fft=float(payload.get("f_fft", np.nan)),
        acf_peak=float(payload.get("acf_peak", np.nan)),
        acf_period_sec=float(payload.get("acf_period_sec", np.nan)),
        acf_lag_steps=int(payload.get("acf_lag_steps", -1)),
        acf_n=int(payload.get("acf_n", 0)),
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
    status_cb: Callable[[str], None] | None = None,
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
        if status_cb is not None:
            status_cb(
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



def _extract_features(
    st: ChannelStreamState,
    *,
    max_keep_sec: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, str, bool] | None:
    """Extract base short-window features from ring buffer state."""

    window_slice = _extract_window_slice(
        st,
        max_keep_sec=float(max_keep_sec),
        window_sec=float(window_sec),
    )
    if window_slice is None:
        return None
    tw0, vw0, tw, vw, dt, t0, t1 = window_slice

    score, A_tail, D_tail, reason = score_one_channel_equiv(
        tw,
        vw,
        float(dt),
        float(t0),
        float(t1),
    )
    return (
        tw0,
        vw0,
        tw,
        vw,
        float(t1),
        float(score),
        float(A_tail),
        float(D_tail),
        str(reason),
        bool(reason in {"ok", "excluded_damped"}),
    )


def _update_baseline_and_long_stats(
    st: ChannelStreamState,
    *,
    t1: float,
    s_log: float,
    long_metric_value: float,
    use_external_baseline: bool,
    risk_prev: bool,
    phase_now: str,
    reason: str,
    short_high: bool,
    long_baseline_max: int,
    long_window_sec: float,
    long_off_recent_window_sec: float,
    long_z_on: float,
    long_z_off: float,
    baseline_post_off_cooldown_sec: float,
) -> tuple[float, float, float, float, int, float, float, int]:
    """Update baseline/long histories and compute long-window ratio stats."""

    metric_now = float(long_metric_value) if np.isfinite(long_metric_value) else float(s_log)
    st.long_score_hist.append((float(t1), float(metric_now)))
    _trim_score_hist_by_time(st.long_score_hist, keep_from_t=(float(t1) - float(long_window_sec)))
    st.long_off_recent_hist.append((float(t1), float(metric_now)))
    _trim_score_hist_by_time(st.long_off_recent_hist, keep_from_t=(float(t1) - float(long_off_recent_window_sec)))

    event_zone_phase = str(phase_now) in {
        PHASE_ON_CANDIDATE,
        PHASE_ON_CONFIRMED,
        PHASE_OFF_CANDIDATE,
    }
    off_cooldown_active = bool(
        np.isfinite(st.last_off_t)
        and (float(t1) >= float(st.last_off_t))
        and ((float(t1) - float(st.last_off_t)) < float(baseline_post_off_cooldown_sec))
    )
    baseline_update_ok = bool(
        (not bool(risk_prev))
        and (str(reason) == "ok")
        and (not bool(short_high))
        and (not bool(event_zone_phase))
        and (not bool(off_cooldown_active))
    )
    if baseline_update_ok:
        st.long_baseline_hist.append(float(s_log))
        while len(st.long_baseline_hist) > int(long_baseline_max):
            st.long_baseline_hist.popleft()

    if bool(use_external_baseline):
        base_med, base_scale = 0.0, 1.0
    else:
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
    return (
        float(base_med),
        float(base_scale),
        float(long_ratio_on),
        float(long_zmax),
        int(long_n),
        float(long_ratio_off),
        float(long_ratio_off_recent),
        int(long_off_n_recent),
    )


def _pick_dominant_band_energy(
    *,
    band_energy_by_name: dict[str, float],
    baseline_band_defs: tuple[tuple[str, float, float], ...],
) -> tuple[str, float]:
    """Pick dominant finite-energy band; fallback to inter_area/first known band."""

    fallback_band = ""
    for name, _, _ in baseline_band_defs:
        band_name = str(name).strip()
        if (not fallback_band) and band_name:
            fallback_band = band_name
        if band_name.lower() == "inter_area":
            fallback_band = band_name
            break

    best_band = ""
    best_energy = float("nan")
    for name, val in band_energy_by_name.items():
        e = float(val)
        if not np.isfinite(e):
            continue
        if (not best_band) or (not np.isfinite(best_energy)) or (e > float(best_energy)):
            best_band = str(name)
            best_energy = float(e)
    if best_band:
        return str(best_band), float(best_energy)
    return str(fallback_band), float("nan")


def _compute_external_baseline_z(
    *,
    tw: np.ndarray,
    vw: np.ndarray,
    channel_name: str,
    baseline_store: object | None,
    baseline_band_defs: tuple[tuple[str, float, float], ...],
) -> tuple[float, bool]:
    """Compute external baseline z-score from dominant band energy when available."""

    if (baseline_store is None) or (not baseline_band_defs):
        return float("nan"), False
    if not hasattr(baseline_store, "energy_to_z"):
        return float("nan"), False

    band_energy_by_name = compute_band_rms_energies(
        tw,
        vw,
        bands=baseline_band_defs,
        min_samples=16,
        dt_cv_max=0.25,
        detrend_linear=True,
    )
    band_name, band_energy = _pick_dominant_band_energy(
        band_energy_by_name=band_energy_by_name,
        baseline_band_defs=baseline_band_defs,
    )
    if (not str(band_name).strip()) or (not np.isfinite(band_energy)):
        return float("nan"), False

    z, _mean, _std, used_fallback = baseline_store.energy_to_z(  # type: ignore[attr-defined]
        channel=str(channel_name),
        band_name=str(band_name),
        energy_value=float(band_energy),
    )
    if (not np.isfinite(z)) or bool(used_fallback):
        return float("nan"), False
    return float(z), True


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
    long_zmax: float,
    update_sec: float,
    threshold_cfg: ThresholdConfig,
    long_cfg: LongConfig,
    periodicity_cfg: PeriodicityConfig,
    rms_decay_local: float,
    rms_decay_local_r2: float,
    rms_decay_local_n: int,
    rms_decay_local_on_ok: bool,
    rms_decay_local_off_hint: bool,
    rms_decay_event: float,
    rms_decay_event_r2: float,
    rms_decay_event_n: int,
    rms_decay_event_win_sec: float,
    phase_now: str,
) -> _QualitySupportSnapshot:
    """Compute periodicity quality, confidence/support, and evidence derivatives."""
    th = threshold_cfg
    lg = long_cfg
    pq = periodicity_cfg
    warmup_cold_start_allowed = bool(not np.isfinite(st.last_off_t))

    warmup_conf_gate_possible = bool(
        bool(warmup_cold_start_allowed)
        and
        bool(lg.warmup_long_enabled)
        and (str(phase_now) in {PHASE_OFF, PHASE_ON_CANDIDATE})
        and (int(long_n) >= int(lg.warmup_on_min_points))
        and (int(baseline_n) >= int(lg.warmup_min_baseline))
        and np.isfinite(long_zmax)
        and (float(long_zmax) >= float(lg.warmup_on_z))
        and np.isfinite(long_ratio_on)
        and (float(long_ratio_on) >= float(lg.warmup_on_ratio))
    )
    quality_need_context = bool(
        bool(pq.confidence_use_calibration)
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
    periodicity_idle_refresh_sec = max(float(update_sec), float(pq.periodicity_quality_cache_refresh_sec))
    can_skip_quality_heavy = bool(
        (not bool(quality_need_context))
        and bool(quality_weak_tick)
        and (st.last_quality is not None)
    )
    can_reuse_quality = bool(
        bool(pq.periodicity_quality_cache_enabled)
        and (not bool(pq.confidence_use_calibration))
        and (not bool(risk_prev))
        and (str(phase_now) == PHASE_OFF)
        and (not bool(short_high))
        and (not bool(short_trigger))
        and (not bool(warmup_conf_gate_possible))
        and (st.last_quality is not None)
        and np.isfinite(st.last_quality_t_end)
        and ((float(t1) - float(st.last_quality_t_end)) < float(periodicity_idle_refresh_sec))
    )
    if can_skip_quality_heavy or can_reuse_quality:
        quality = st.last_quality if st.last_quality is not None else QualityCacheSnapshot()
    else:
        quality = _quality_cache_from_payload(
            _compute_periodicity_quality(
                tw,
                vw,
                acf_min_points=int(pq.acf_min_points),
                acf_min_period_sec=float(pq.acf_min_period_sec),
                acf_max_period_sec=float(pq.acf_max_period_sec),
                band_low_hz=float(pq.freq_band_low_hz),
                band_high_hz=float(pq.freq_band_high_hz),
                linear_detrend=bool(pq.freq_linear_detrend),
                ar1_whiten=bool(pq.freq_ar1_whiten),
                w_acf=float(pq.confidence_w_acf),
                w_spec=float(pq.confidence_w_spec),
                w_env=float(pq.confidence_w_env),
                w_fft=float(pq.confidence_w_fft),
            )
        )
        st.last_quality_t_end = float(t1)
    confidence_raw = (
        float(quality.confidence_raw)
        if np.isfinite(quality.confidence_raw)
        else float(quality.confidence)
    )
    confidence_cal = _calibrate_confidence_quantile(
        confidence_raw,
        conf_hist=st.conf_raw_hist,
        min_points=int(pq.confidence_cal_min_points),
    )
    confidence_used = (
        float(confidence_cal)
        if (bool(pq.confidence_use_calibration) and np.isfinite(confidence_cal))
        else (float(confidence_raw) if np.isfinite(confidence_raw) else float("nan"))
    )
    quality.confidence_raw = float(confidence_raw) if np.isfinite(confidence_raw) else float("nan")
    quality.confidence_cal = float(confidence_cal) if np.isfinite(confidence_cal) else float("nan")
    quality.confidence_used = float(confidence_used) if np.isfinite(confidence_used) else float("nan")
    quality.confidence = float(confidence_used) if np.isfinite(confidence_used) else float("nan")
    quality.rms_decay_local = float(rms_decay_local) if np.isfinite(rms_decay_local) else float("nan")
    quality.rms_decay_local_r2 = float(rms_decay_local_r2) if np.isfinite(rms_decay_local_r2) else float("nan")
    quality.rms_decay_local_n = int(rms_decay_local_n)
    quality.rms_decay_local_on_ok = int(rms_decay_local_on_ok)
    quality.rms_decay_local_off_hint = int(rms_decay_local_off_hint)
    quality.rms_decay_event = float(rms_decay_event) if np.isfinite(rms_decay_event) else float("nan")
    quality.rms_decay_event_r2 = float(rms_decay_event_r2) if np.isfinite(rms_decay_event_r2) else float("nan")
    quality.rms_decay_event_n = int(rms_decay_event_n)
    quality.rms_decay_event_win_sec = float(rms_decay_event_win_sec) if np.isfinite(rms_decay_event_win_sec) else float("nan")
    if np.isfinite(confidence_raw) and ((not bool(pq.confidence_cal_off_only)) or (not bool(risk_prev))):
        st.conf_raw_hist.append(float(confidence_raw))
        while len(st.conf_raw_hist) > int(pq.confidence_cal_hist_max):
            st.conf_raw_hist.popleft()
    st.last_quality = quality

    acf_peak = float(quality.acf_peak)
    acf_period_sec = float(quality.acf_period_sec)
    acf_lag_steps = int(quality.acf_lag_steps)
    acf_n = int(quality.acf_n)
    confidence = float(quality.confidence)
    confidence_raw = float(quality.confidence_raw)
    confidence_cal = float(quality.confidence_cal)
    c_acf = float(quality.c_acf)
    c_spec = float(quality.c_spec)
    c_env = float(quality.c_env)
    c_fft = float(quality.c_fft)
    c_freq_agree = float(quality.c_freq_agree)
    f_welch = float(quality.f_welch)
    f_zc = float(quality.f_zc)
    f_fft = float(quality.f_fft)

    long_ready = (
        (int(long_n) >= int(lg.long_min_points))
        and (int(baseline_n) >= int(lg.long_min_baseline))
        and np.isfinite(base_med)
        and np.isfinite(base_scale)
    )
    if long_ready:
        st.long_ready_streak += 1
    else:
        st.long_ready_streak = 0
    warmup_mode = bool(lg.warmup_long_enabled) and bool(warmup_cold_start_allowed) and (not bool(long_ready))

    baseline_cut = float(cut_off_eff if risk_prev else cut_on_eff)
    score_safe = float(score) if (np.isfinite(score) and float(score) > 0.0) else (baseline_cut * 1e-12)
    ratio = max(score_safe / baseline_cut, 1e-12)
    e_t = float(np.clip(np.log10(ratio), -float(th.evidence_clip), float(th.evidence_clip)))
    if str(reason) == "excluded_damped":
        damped_penalty = float(th.excluded_damped_evidence_penalty)
        if int(st.damped_streak) >= int(th.excluded_damped_hard_penalty_streak):
            damped_penalty = float(th.evidence_clip)
        damped_penalty = float(np.clip(damped_penalty, 0.0, float(th.evidence_clip)))
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
        (not bool(th.on_require_accel_for_candidate))
        or (np.isfinite(delta_s_log) and (float(delta_s_log) >= float(th.on_accel_score_log_min)))
        or (np.isfinite(delta_e) and (float(delta_e) >= float(th.on_accel_evidence_min)))
    )
    st.evidence = float(th.evidence_alpha) * float(st.evidence) + (1.0 - float(th.evidence_alpha)) * e_t
    st.prev_score_log = float(s_log)
    st.prev_e_t = float(e_t)

    conf_now = float(confidence) if np.isfinite(confidence) else 0.0
    raw_now = float(confidence_raw) if np.isfinite(confidence_raw) else float("nan")
    use_cal_gate = bool(pq.confidence_use_calibration) and np.isfinite(confidence_cal)
    cal_on_active = bool(pq.cal_on_soft_mode) and bool(use_cal_gate)

    cal_noise_norm = _normalize01(float(base_scale), float(pq.cal_on_noise_scale_low), float(pq.cal_on_noise_scale_high))
    if not np.isfinite(cal_noise_norm):
        cal_noise_norm = 0.0
    cal_on_conf_thr = float(np.clip(
        float(pq.confidence_on_min) + float(pq.cal_on_conf_adapt_gain) * float(cal_noise_norm),
        float(pq.cal_on_conf_floor),
        float(pq.cal_on_conf_ceil),
    ))

    # Weighted CAL_ON support blend (weights are configurable runtime knobs).
    score_support_base = _clip01((float(e_t) + float(th.evidence_clip)) / (2.0 * float(th.evidence_clip)))
    score_acf_bonus = (
        float(pq.cal_on_support_score_acf_bonus) * _clip01(float(c_acf))
        if np.isfinite(c_acf)
        else 0.0
    )
    score_support = _clip01(float(score_support_base) + float(score_acf_bonus))
    long_support = (
        _clip01(float(long_ratio_on) / max(float(lg.long_on_ratio), 1e-9))
        if bool(long_ready)
        else (
            _clip01(float(long_ratio_on) / max(float(lg.warmup_on_ratio), 1e-9))
            if bool(warmup_mode) else 0.0
        )
    )
    d_long = (
        float(long_ratio_on - float(st.prev_long_ratio_on))
        if (np.isfinite(long_ratio_on) and np.isfinite(st.prev_long_ratio_on))
        else float("nan")
    )
    off_age_sec = (
        float(t1 - float(st.last_off_t))
        if np.isfinite(st.last_off_t)
        else float("nan")
    )
    reentry_support_ctx = bool(
        bool(pq.reentry_support_suppress_enabled)
        and np.isfinite(st.last_off_t)
        and (str(phase_now) in {PHASE_OFF, PHASE_ON_CANDIDATE})
        and np.isfinite(off_age_sec)
        and (float(off_age_sec) >= 0.0)
        and (float(off_age_sec) < float(pq.reentry_support_tail_window_sec))
    )
    residual_tail_like = bool(
        np.isfinite(long_ratio_on)
        and (float(long_ratio_on) >= float(pq.reentry_support_residual_long_min))
    )
    if bool(reentry_support_ctx) and bool(residual_tail_like):
        growth_vals: list[float] = []
        g_s_min = float(max(0.0, float(pq.reentry_support_growth_score_log_min)))
        g_e_min = float(max(0.0, float(pq.reentry_support_growth_evidence_min)))
        g_l_min = float(max(0.0, float(pq.reentry_support_growth_long_min)))
        if np.isfinite(delta_s_log):
            g_s_hi = float(max(g_s_min * 3.0, g_s_min + 1e-6, 1e-6))
            growth_vals.append(_clip01(_normalize01(float(delta_s_log), g_s_min, g_s_hi)))
        if np.isfinite(delta_e):
            g_e_hi = float(max(g_e_min * 3.0, g_e_min + 1e-6, 1e-6))
            growth_vals.append(_clip01(_normalize01(float(delta_e), g_e_min, g_e_hi)))
        if np.isfinite(d_long):
            g_l_hi = float(max(g_l_min * 3.0, g_l_min + 1e-6, 1e-6))
            growth_vals.append(_clip01(_normalize01(float(d_long), g_l_min, g_l_hi)))
        fresh_growth = (max(growth_vals) if growth_vals else 0.0)
        growth_factor = _clip01(float(fresh_growth))
        long_factor = float(max(float(pq.reentry_support_long_floor), float(growth_factor)))
        score_factor = float(max(
            float(pq.reentry_support_score_floor),
            (0.5 + (0.5 * float(growth_factor))),
        ))
        long_support = _clip01(float(long_support) * float(long_factor))
        score_support = _clip01(float(score_support) * float(score_factor))

    conf_support = _clip01(float(conf_now))
    accel_support_s = (
        _clip01(float(delta_s_log) / max(abs(float(th.on_accel_score_log_min)), 1e-6))
        if np.isfinite(delta_s_log) else float("nan")
    )
    accel_support_e = (
        _clip01(float(delta_e) / max(abs(float(th.on_accel_evidence_min)), 1e-6))
        if np.isfinite(delta_e) else float("nan")
    )
    accel_parts = [float(x) for x in (accel_support_s, accel_support_e) if np.isfinite(x)]
    accel_support = (max(accel_parts) if accel_parts else float("nan"))
    if not np.isfinite(accel_support):
        accel_support = (1.0 if bool(accel_ok) else 0.0)

    weighted_sum = 0.0
    weight_total = 0.0
    support_axes = (
        (float(pq.cal_on_support_score_w), float(score_support)),
        (float(pq.cal_on_support_long_w), float(long_support)),
        (float(pq.cal_on_support_conf_w), float(conf_support)),
        (float(pq.cal_on_support_accel_w), float(accel_support)),
    )
    for w, x in support_axes:
        if (not np.isfinite(w)) or (w <= 0.0) or (not np.isfinite(x)):
            continue
        weighted_sum += float(w) * float(x)
        weight_total += float(w)
    on_support = (float(weighted_sum / weight_total) if weight_total > 0.0 else float("nan"))
    if np.isfinite(on_support):
        st.on_support_ema = (
            float(pq.cal_on_support_ema_alpha) * float(st.on_support_ema)
            + (1.0 - float(pq.cal_on_support_ema_alpha)) * float(on_support)
        )
    elif not np.isfinite(st.on_support_ema):
        st.on_support_ema = 0.0
    on_support_ema = float(st.on_support_ema)
    if bool(cal_on_active) and (str(phase_now) in {PHASE_OFF, PHASE_ON_CANDIDATE}):
        soft_vote = bool(
            np.isfinite(on_support)
            and (float(on_support) >= float(pq.cal_on_support_enter_min))
            and (float(conf_now) >= float(cal_on_conf_thr))
        )
        st.on_soft_votes.append(1 if soft_vote else 0)
        while len(st.on_soft_votes) > int(pq.cal_on_confirm_votes_window):
            st.on_soft_votes.popleft()
    else:
        st.on_soft_votes.clear()
    on_soft_vote_sum = int(st.on_soft_votes.sum)
    on_soft_confirmed = bool(
        (len(st.on_soft_votes) >= int(pq.cal_on_confirm_votes_window))
        and (int(on_soft_vote_sum) >= int(pq.cal_on_confirm_votes_required))
    )
    st.prev_long_ratio_on = (float(long_ratio_on) if np.isfinite(long_ratio_on) else float("nan"))

    return _QualitySupportSnapshot(
        acf_peak=float(acf_peak),
        acf_period_sec=float(acf_period_sec),
        acf_lag_steps=int(acf_lag_steps),
        acf_n=int(acf_n),
        confidence=float(confidence),
        confidence_raw=float(confidence_raw),
        confidence_cal=float(confidence_cal),
        c_acf=float(c_acf),
        c_spec=float(c_spec),
        c_env=float(c_env),
        c_fft=float(c_fft),
        c_freq_agree=float(c_freq_agree),
        f_welch=float(f_welch),
        f_zc=float(f_zc),
        f_fft=float(f_fft),
        rms_decay=float(quality.rms_decay_local),
        rms_decay_r2=float(quality.rms_decay_local_r2),
        rms_decay_n=int(quality.rms_decay_local_n),
        rms_decay_on_ok=int(quality.rms_decay_local_on_ok),
        rms_decay_off_hint=int(quality.rms_decay_local_off_hint),
        rms_decay_event=float(quality.rms_decay_event),
        rms_decay_event_r2=float(quality.rms_decay_event_r2),
        rms_decay_event_n=int(quality.rms_decay_event_n),
        rms_decay_event_win_sec=float(quality.rms_decay_event_win_sec),
        long_ready=bool(long_ready),
        warmup_mode=bool(warmup_mode),
        e_t=float(e_t),
        delta_s_log=float(delta_s_log),
        delta_e=float(delta_e),
        accel_ok=bool(accel_ok),
        conf_now=float(conf_now),
        raw_now=float(raw_now),
        use_cal_gate=bool(use_cal_gate),
        cal_on_active=bool(cal_on_active),
        cal_on_conf_thr=float(cal_on_conf_thr),
        on_support=float(on_support),
        on_support_ema=float(on_support_ema),
        on_soft_vote_sum=int(on_soft_vote_sum),
        on_soft_confirmed=bool(on_soft_confirmed),
    )


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
    rms_decay_event_off_hint: bool,
    long_off_votes_window: int,
    long_off_votes_required: int,
    damped_force_off_streak: int,
    theta_off: float,
    force_off_long_on_ratio: float | None,
    force_off_require_long_not_on: bool,
    long_ratio_on: float,
    long_on_ratio: float,
) -> tuple[bool, bool, bool, bool]:
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

    long_off_decay_drop_gate = float(np.clip(
        float(long_ratio_on) - (0.25 * float(long_off_ratio)),
        0.0,
        1.0,
    ))
    long_off_decay_recent_ok = bool(
        np.isfinite(long_ratio_off_recent)
        and (
            (float(long_ratio_off_recent) <= float(long_off_ratio))
            or (float(long_ratio_off_recent) <= float(long_off_decay_drop_gate))
        )
    )
    off_vote_decay = bool(
        bool(short_release)
        and bool(long_ready)
        and (int(long_off_n_recent) >= int(long_off_recent_min_points))
        and bool(long_off_decay_recent_ok)
        and (bool(rms_decay_local_off_hint) or bool(rms_decay_event_off_hint))
    )
    off_vote_collapse = bool(
        bool(short_release)
        and bool(collapse_ok)
    )
    off_vote_core = bool(off_vote_collapse or off_vote_decay)
    if str(phase_now) in {PHASE_ON_CONFIRMED, PHASE_OFF_CANDIDATE}:
        st.long_off_votes.append(1 if off_vote_core else 0)
        while len(st.long_off_votes) > int(long_off_votes_window):
            st.long_off_votes.popleft()
    else:
        st.long_off_votes.clear()
        st.off_candidate_start_t = None
        st.off_candidate_start_update_idx = None
    long_off_vote_sum = int(st.long_off_votes.sum)
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
    return bool(collapse_ok), bool(off_vote_core), bool(long_off_confirmed), bool(force_off_now)


def compute_tick_features(
    st: ChannelStreamState,
    *,
    channel_name: str,
    max_keep_sec: float,
    window_sec: float,
    upd_idx: int,
    cut_on: float,
    cut_off: float,
    update_sec: float,
    threshold_cfg: ThresholdConfig,
    long_cfg: LongConfig,
    periodicity_cfg: PeriodicityConfig,
    baseline_store: object | None = None,
    baseline_band_defs: tuple[tuple[str, float, float], ...] = (),
) -> TickFeatures | None:
    """Compute all per-tick feature values used by FSM decisions and event payloads."""

    feature_base = _extract_features(
        st,
        max_keep_sec=float(max_keep_sec),
        window_sec=float(window_sec),
    )
    if feature_base is None:
        return None

    tw0, vw0, tw, vw, t1, score, A_tail, D_tail, reason, score_reason_ok = feature_base
    phase_now = str(st.phase)
    th = threshold_cfg
    lg = long_cfg
    pq = periodicity_cfg

    if reason == "excluded_damped":
        st.damped_streak += 1
    else:
        st.damped_streak = 0

    pre_base_med, pre_base_scale = _robust_center_scale(st.long_baseline_hist)
    cut_on_eff = float(cut_on)
    cut_off_eff = float(cut_off)
    if bool(th.short_dynamic_cut_enabled) and (len(st.long_baseline_hist) >= int(th.short_dynamic_min_baseline)):
        cut_on_dyn = _dynamic_score_cut_from_log_baseline(
            base_med=pre_base_med,
            base_scale=pre_base_scale,
            z_thr=float(th.short_dynamic_on_z),
            fallback_cut=float(cut_on),
        )
        cut_off_dyn = _dynamic_score_cut_from_log_baseline(
            base_med=pre_base_med,
            base_scale=pre_base_scale,
            z_thr=float(th.short_dynamic_off_z),
            fallback_cut=float(cut_off),
        )
        cut_on_eff = float(np.clip(
            float(cut_on_dyn),
            float(cut_on),
            float(cut_on) * float(th.short_dynamic_on_max_mult),
        ))
        cut_off_eff = float(np.clip(
            float(cut_off_dyn),
            float(cut_off),
            float(cut_off) * float(th.short_dynamic_off_max_mult),
        ))
    if float(cut_off_eff) > float(cut_on_eff):
        cut_off_eff = float(cut_on_eff)
    damped_cut_relax = float(th.short_damped_cut_relax if (reason == "excluded_damped") else 1.0)
    cut_on_cmp = float(max(1e-18, float(cut_on_eff) * float(damped_cut_relax)))
    cut_off_cmp = float(max(1e-18, float(cut_off_eff) * float(damped_cut_relax)))

    risk_prev = bool(_is_risk_active_phase(st.phase))
    short_high = bool(score_reason_ok and np.isfinite(score) and (float(score) >= float(cut_on_cmp)))
    st.on_short_votes.append(1 if short_high else 0)
    while len(st.on_short_votes) > int(th.on_short_votes_window):
        st.on_short_votes.popleft()
    st.on_candidate_streak = int(st.on_short_votes.sum)
    short_trigger = bool(
        (len(st.on_short_votes) >= int(th.on_short_votes_window))
        and (int(st.on_short_votes.sum) >= int(th.on_consecutive_required))
    )

    score_log_safe = float(score) if (np.isfinite(score) and float(score) > 0.0) else 1e-18
    s_log = float(np.log10(score_log_safe))
    external_energy_z, external_baseline_active = _compute_external_baseline_z(
        tw=tw,
        vw=vw,
        channel_name=str(channel_name),
        baseline_store=baseline_store,
        baseline_band_defs=tuple(baseline_band_defs),
    )
    long_metric_value = float(external_energy_z) if bool(external_baseline_active) else float(s_log)

    rms_transition_active = bool(
        risk_prev
        or short_high
        or short_trigger
        or (str(st.phase) != PHASE_OFF)
    )
    rms_decay_idle_refresh_sec = max(float(update_sec), float(th.rms_decay_cache_refresh_sec))
    can_reuse_rms = bool(
        bool(th.rms_decay_cache_enabled)
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
            trailing_sec=float(th.rms_decay_window_sec),
            rms_win_sec=float(th.rms_decay_rms_win_sec),
            step_sec=float(th.rms_decay_step_sec),
            min_windows=int(th.rms_decay_min_windows),
        )
        st.last_rms_decay = float(rms_decay_local) if np.isfinite(rms_decay_local) else float("nan")
        st.last_rms_decay_r2 = float(rms_decay_local_r2) if np.isfinite(rms_decay_local_r2) else float("nan")
        st.last_rms_decay_n = int(rms_decay_local_n)
        st.last_rms_decay_t_end = float(t1)
    rms_decay_local_on_ok = bool(
        (not bool(th.rms_decay_gate_enabled))
        or (not np.isfinite(rms_decay_local))
        or (float(rms_decay_local) <= float(th.rms_decay_on_max))
    )
    rms_decay_local_off_hint = bool(
        bool(th.rms_decay_gate_enabled)
        and np.isfinite(rms_decay_local)
        and (float(rms_decay_local) >= float(th.rms_decay_off_min))
    )

    rms_decay_event = float("nan")
    rms_decay_event_r2 = float("nan")
    rms_decay_event_n = 0
    rms_decay_event_win_sec = float("nan")
    if (
        bool(th.rms_decay_event_enabled)
        and (str(st.phase) in {PHASE_ON_CONFIRMED, PHASE_OFF_CANDIDATE})
        and (st.active_start_t is not None)
        and np.isfinite(st.active_start_t)
    ):
        event_age_sec = float(t1 - float(st.active_start_t))
        if float(event_age_sec) >= float(th.rms_decay_event_min_window_sec):
            rms_decay_event, rms_decay_event_r2, rms_decay_event_n, rms_decay_event_win_sec = _event_rms_decay_from_signal(
                tw0,
                vw0,
                event_start_t=float(st.active_start_t),
                t_end=float(t1),
                max_window_sec=float(th.rms_decay_event_max_window_sec),
                min_window_sec=float(th.rms_decay_event_min_window_sec),
                rms_win_sec=float(th.rms_decay_event_rms_win_sec),
                step_sec=float(th.rms_decay_event_step_sec),
                min_windows=int(th.rms_decay_event_min_windows),
            )
    st.last_rms_decay_event = float(rms_decay_event) if np.isfinite(rms_decay_event) else float("nan")
    st.last_rms_decay_event_r2 = float(rms_decay_event_r2) if np.isfinite(rms_decay_event_r2) else float("nan")
    st.last_rms_decay_event_n = int(rms_decay_event_n)
    st.last_rms_decay_event_win_sec = float(rms_decay_event_win_sec) if np.isfinite(rms_decay_event_win_sec) else float("nan")
    st.last_rms_decay_event_t_end = float(t1)
    if str(st.phase) not in {PHASE_ON_CONFIRMED, PHASE_OFF_CANDIDATE}:
        st.rms_decay_event_peak = float("nan")
    if np.isfinite(rms_decay_event):
        prev_peak = float(st.rms_decay_event_peak) if np.isfinite(st.rms_decay_event_peak) else float("nan")
        st.rms_decay_event_peak = float(rms_decay_event) if (not np.isfinite(prev_peak)) else float(max(prev_peak, float(rms_decay_event)))
    rms_decay_event_for_hint = float(st.rms_decay_event_peak) if np.isfinite(st.rms_decay_event_peak) else float(rms_decay_event)
    rms_decay_event_off_hint = bool(
        bool(th.rms_decay_event_enabled)
        and np.isfinite(rms_decay_event_for_hint)
        and (float(rms_decay_event_for_hint) >= float(th.rms_decay_event_off_min))
    )

    short_release = (not score_reason_ok) or (not np.isfinite(score)) or (float(score) < float(cut_off_cmp))
    (
        base_med,
        base_scale,
        long_ratio_on,
        long_zmax,
        long_n,
        long_ratio_off,
        long_ratio_off_recent,
        long_off_n_recent,
    ) = _update_baseline_and_long_stats(
        st,
        t1=float(t1),
        s_log=float(s_log),
        long_metric_value=float(long_metric_value),
        use_external_baseline=bool(external_baseline_active),
        risk_prev=bool(risk_prev),
        phase_now=str(phase_now),
        reason=str(reason),
        short_high=bool(short_high),
        long_baseline_max=int(lg.long_baseline_max),
        long_window_sec=float(lg.long_window_sec),
        long_off_recent_window_sec=float(lg.long_off_recent_window_sec),
        long_z_on=float(lg.long_z_on),
        long_z_off=float(lg.long_z_off),
        baseline_post_off_cooldown_sec=float(lg.baseline_post_off_cooldown_sec),
    )
    baseline_n_local = int(len(st.long_baseline_hist))
    baseline_n = int(baseline_n_local)
    if bool(external_baseline_active):
        baseline_n = int(max(
            baseline_n_local,
            int(lg.long_min_baseline),
            int(lg.warmup_min_baseline),
        ))

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
        baseline_n=int(baseline_n),
        long_zmax=float(long_zmax),
        update_sec=float(update_sec),
        threshold_cfg=th,
        long_cfg=lg,
        periodicity_cfg=pq,
        rms_decay_local=float(rms_decay_local),
        rms_decay_local_r2=float(rms_decay_local_r2),
        rms_decay_local_n=int(rms_decay_local_n),
        rms_decay_local_on_ok=bool(rms_decay_local_on_ok),
        rms_decay_local_off_hint=bool(rms_decay_local_off_hint),
        rms_decay_event=float(rms_decay_event),
        rms_decay_event_r2=float(rms_decay_event_r2),
        rms_decay_event_n=int(rms_decay_event_n),
        rms_decay_event_win_sec=float(rms_decay_event_win_sec),
        phase_now=str(phase_now),
    )
    acf_peak = float(quality_snapshot.acf_peak)
    acf_period_sec = float(quality_snapshot.acf_period_sec)
    acf_lag_steps = int(quality_snapshot.acf_lag_steps)
    acf_n = int(quality_snapshot.acf_n)
    confidence = float(quality_snapshot.confidence)
    confidence_raw = float(quality_snapshot.confidence_raw)
    confidence_cal = float(quality_snapshot.confidence_cal)
    c_acf = float(quality_snapshot.c_acf)
    c_spec = float(quality_snapshot.c_spec)
    c_env = float(quality_snapshot.c_env)
    c_fft = float(quality_snapshot.c_fft)
    c_freq_agree = float(quality_snapshot.c_freq_agree)
    f_welch = float(quality_snapshot.f_welch)
    f_zc = float(quality_snapshot.f_zc)
    f_fft = float(quality_snapshot.f_fft)
    rms_decay = float(quality_snapshot.rms_decay)
    rms_decay_r2 = float(quality_snapshot.rms_decay_r2)
    rms_decay_n = int(quality_snapshot.rms_decay_n)
    rms_decay_on_ok = int(quality_snapshot.rms_decay_on_ok)
    rms_decay_off_hint = int(quality_snapshot.rms_decay_off_hint)
    rms_decay_event = float(quality_snapshot.rms_decay_event)
    rms_decay_event_r2 = float(quality_snapshot.rms_decay_event_r2)
    rms_decay_event_n = int(quality_snapshot.rms_decay_event_n)
    rms_decay_event_win_sec = float(quality_snapshot.rms_decay_event_win_sec)
    long_ready = bool(quality_snapshot.long_ready)
    warmup_mode = bool(quality_snapshot.warmup_mode)
    e_t = float(quality_snapshot.e_t)
    delta_s_log = float(quality_snapshot.delta_s_log)
    delta_e = float(quality_snapshot.delta_e)
    accel_ok = bool(quality_snapshot.accel_ok)
    conf_now = float(quality_snapshot.conf_now)
    raw_now = float(quality_snapshot.raw_now)
    use_cal_gate = bool(quality_snapshot.use_cal_gate)
    cal_on_active = bool(quality_snapshot.cal_on_active)
    cal_on_conf_thr = float(quality_snapshot.cal_on_conf_thr)
    on_support = float(quality_snapshot.on_support)
    on_support_ema = float(quality_snapshot.on_support_ema)
    on_soft_vote_sum = int(quality_snapshot.on_soft_vote_sum)
    on_soft_confirmed = bool(quality_snapshot.on_soft_confirmed)

    warmup_core = bool(
        bool(warmup_mode)
        and (int(long_n) >= int(lg.warmup_on_min_points))
        and (int(baseline_n) >= int(lg.warmup_min_baseline))
        and np.isfinite(long_zmax)
        and (float(long_zmax) >= float(lg.warmup_on_z))
        and np.isfinite(long_ratio_on)
        and (float(long_ratio_on) >= float(lg.warmup_on_ratio))
    )
    if (
        bool(lg.warmup_cancel_on_excluded_damped)
        and (reason == "excluded_damped")
        and ((not np.isfinite(score)) or (float(score) < float(cut_off_cmp)))
    ):
        warmup_core = False
    if (phase_now in {PHASE_OFF, PHASE_ON_CANDIDATE}) and bool(warmup_mode):
        st.warmup_on_votes.append(1 if warmup_core else 0)
        while len(st.warmup_on_votes) > int(lg.warmup_on_votes_window):
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
    warmup_vote_sum = int(st.warmup_on_votes.sum)
    warmup_votes_ready = bool(
        (len(st.warmup_on_votes) >= int(lg.warmup_on_votes_window))
        and (int(warmup_vote_sum) >= int(lg.warmup_on_votes_required))
    )
    warmup_age_sec = (
        float(t1 - float(st.warmup_on_start_t))
        if (st.warmup_on_start_t is not None) and np.isfinite(st.warmup_on_start_t)
        else 0.0
    )
    warmup_on_confirmed = bool(
        warmup_votes_ready
        and (float(warmup_age_sec) >= float(lg.warmup_on_confirm_min_sec))
    )

    collapse_ok, off_vote_core, long_off_confirmed, force_off_now = _compute_off_path(
        st,
        phase_now=str(phase_now),
        short_release=bool(short_release),
        conf_now=float(conf_now),
        raw_now=float(raw_now),
        use_cal_gate=bool(use_cal_gate),
        confidence_off_max=float(pq.confidence_off_max),
        off_periodicity_collapse_conf_raw_max=float(th.off_periodicity_collapse_conf_raw_max),
        confidence_raw_off_max_when_cal=float(pq.confidence_raw_off_max_when_cal),
        off_periodicity_collapse_streak_required=int(th.off_periodicity_collapse_streak_required),
        long_ready=bool(long_ready),
        long_off_n_recent=int(long_off_n_recent),
        long_off_recent_min_points=int(lg.long_off_recent_min_points),
        long_ratio_off_recent=float(long_ratio_off_recent),
        long_off_ratio=float(lg.long_off_ratio),
        rms_decay_local_off_hint=bool(rms_decay_local_off_hint),
        rms_decay_event_off_hint=bool(rms_decay_event_off_hint),
        long_off_votes_window=int(lg.long_off_votes_window),
        long_off_votes_required=int(lg.long_off_votes_required),
        damped_force_off_streak=int(th.damped_force_off_streak),
        theta_off=float(th.theta_off),
        force_off_long_on_ratio=th.force_off_long_on_ratio,
        force_off_require_long_not_on=bool(th.force_off_require_long_not_on),
        long_ratio_on=float(long_ratio_on),
        long_on_ratio=float(lg.long_on_ratio),
    )

    return TickFeatures(
        signal=TickSignalState(
            t1=float(t1),
            score=float(score),
            A_tail=float(A_tail),
            D_tail=float(D_tail),
            reason=str(reason),
            score_reason_ok=bool(score_reason_ok),
            cut_on_cmp=float(cut_on_cmp),
            cut_off_cmp=float(cut_off_cmp),
            long_ratio_on=float(long_ratio_on),
            long_ratio_off=float(long_ratio_off),
            long_ratio_off_recent=float(long_ratio_off_recent),
            long_off_n_recent=int(long_off_n_recent),
            long_zmax=float(long_zmax),
            long_n=int(long_n),
            baseline_n=int(baseline_n),
            short_high=bool(short_high),
            short_trigger=bool(short_trigger),
            collapse_ok=bool(collapse_ok),
            off_vote_core=bool(off_vote_core),
            long_off_confirmed=bool(long_off_confirmed),
            force_off_now=bool(force_off_now),
        ),
        quality=TickQualityState(
            acf_peak=float(acf_peak),
            acf_period_sec=float(acf_period_sec),
            acf_lag_steps=int(acf_lag_steps),
            acf_n=int(acf_n),
            confidence=float(confidence),
            confidence_raw=float(confidence_raw),
            confidence_cal=float(confidence_cal),
            c_acf=float(c_acf),
            c_spec=float(c_spec),
            c_env=float(c_env),
            c_fft=float(c_fft),
            c_freq_agree=float(c_freq_agree),
            f_welch=float(f_welch),
            f_zc=float(f_zc),
            f_fft=float(f_fft),
            rms_decay=float(rms_decay),
            rms_decay_r2=float(rms_decay_r2),
            rms_decay_n=int(rms_decay_n),
            rms_decay_on_ok=bool(rms_decay_on_ok),
            rms_decay_off_hint=bool(rms_decay_off_hint),
            rms_decay_event=float(rms_decay_event),
            rms_decay_event_r2=float(rms_decay_event_r2),
            rms_decay_event_n=int(rms_decay_event_n),
            rms_decay_event_win_sec=float(rms_decay_event_win_sec),
            long_ready=bool(long_ready),
            warmup_mode=bool(warmup_mode),
            e_t=float(e_t),
            delta_s_log=float(delta_s_log),
            delta_e=float(delta_e),
            accel_ok=bool(accel_ok),
            conf_now=float(conf_now),
            raw_now=float(raw_now),
            use_cal_gate=bool(use_cal_gate),
            cal_on_active=bool(cal_on_active),
            cal_on_conf_thr=float(cal_on_conf_thr),
            on_support=float(on_support),
            on_support_ema=float(on_support_ema),
        ),
        vote=TickVoteState(
            warmup_vote_sum=int(warmup_vote_sum),
            warmup_on_confirmed=bool(warmup_on_confirmed),
            on_soft_vote_sum=int(on_soft_vote_sum),
            on_soft_confirmed=bool(on_soft_confirmed),
        ),
    )


def build_decision_context(
    *,
    tick: TickFeatures,
    st: ChannelStreamState,
    threshold_cfg: ThresholdConfig,
    long_cfg: LongConfig,
    periodicity_cfg: PeriodicityConfig,
) -> DecisionContext:
    """Evaluate ON-entry using four axes: confidence/support/long-readiness/acceleration."""
    th = threshold_cfg
    lg = long_cfg
    pq = periodicity_cfg
    conf_now = float(tick.conf_now)
    raw_now = float(tick.raw_now)
    use_cal_gate = bool(tick.use_cal_gate)
    cal_on_active = bool(tick.cal_on_active)
    on_support_ema = float(tick.on_support_ema)
    cal_on_conf_thr = float(tick.cal_on_conf_thr)
    phase_now = str(st.phase)
    long_ready = bool(tick.long_ready)
    long_ratio_on = float(tick.long_ratio_on)
    short_trigger = bool(tick.short_trigger)
    warmup_mode = bool(tick.warmup_mode)
    warmup_on_confirmed = bool(tick.warmup_on_confirmed)
    accel_ok = bool(tick.accel_ok)
    delta_s_log = float(tick.delta_s_log)
    delta_e = float(tick.delta_e)
    warmup_cold_start_allowed = bool(not np.isfinite(st.last_off_t))

    # Confidence axis: independent from support axis.
    on_conf_ok = bool(conf_now >= float(pq.confidence_on_min))
    if bool(use_cal_gate) and bool(pq.confidence_dual_gate_when_cal):
        on_conf_ok = bool(
            on_conf_ok
            and np.isfinite(raw_now)
            and (float(raw_now) >= float(pq.confidence_raw_on_min_when_cal))
        )
    # Support axis: driven by support EMA only.
    on_support_ok = bool(np.isfinite(on_support_ema) and (float(on_support_ema) >= float(pq.cal_on_support_enter_min)))
    if bool(cal_on_active):
        raw_guard_ok = (
            (not bool(pq.confidence_dual_gate_when_cal))
            or (np.isfinite(raw_now) and (float(raw_now) >= float(pq.confidence_raw_on_min_when_cal)))
        )
        on_conf_ok = bool(
            np.isfinite(conf_now)
            and (float(conf_now) >= float(cal_on_conf_thr))
            and raw_guard_ok
        )

    long_on_core = bool(bool(long_ready) and (float(long_ratio_on) >= float(lg.long_on_ratio)))
    warmup_handoff_active = bool(
        bool(lg.warmup_long_enabled)
        and bool(warmup_cold_start_allowed)
        and bool(long_ready)
        and (int(st.long_ready_streak) <= int(lg.warmup_handoff_grace_ticks))
    )
    off_age_sec = (
        float(tick.t1 - float(st.last_off_t))
        if np.isfinite(st.last_off_t)
        else float("nan")
    )
    re_on_active = bool(
        np.isfinite(off_age_sec)
        and (float(off_age_sec) >= 0.0)
        and (float(off_age_sec) < float(th.re_on_grace_sec))
    )
    accel_evidence_ok = bool(
        (np.isfinite(delta_s_log) and (float(delta_s_log) >= float(th.on_accel_score_log_min)))
        or (np.isfinite(delta_e) and (float(delta_e) >= float(th.on_accel_evidence_min)))
    )
    re_on_short_ok = bool((not bool(re_on_active)) or (not bool(th.re_on_require_short_trigger)) or bool(short_trigger))
    re_on_accel_ok = bool((not bool(re_on_active)) or (not bool(th.re_on_require_accel)) or bool(accel_evidence_ok))
    post_off_rearm_active = bool(
        np.isfinite(off_age_sec)
        and (float(off_age_sec) >= 0.0)
        and (float(off_age_sec) < float(lg.post_off_rearm_sec))
    )
    if bool(warmup_mode):
        on_long_gate_ok = bool(warmup_on_confirmed)
    elif warmup_handoff_active:
        on_long_gate_ok = bool(long_on_core and warmup_on_confirmed)
    elif post_off_rearm_active:
        on_long_gate_ok = bool(long_on_core and short_trigger)
    else:
        on_long_gate_ok = bool(long_on_core)

    on_entry_ready = bool(
        bool(accel_ok)
        and bool(on_conf_ok)
        and bool(on_support_ok)
        and bool(on_long_gate_ok)
        and bool(re_on_short_ok)
        and bool(re_on_accel_ok)
    )
    on_entry_vote_sum = (
        int(bool(accel_ok))
        + int(bool(on_conf_ok))
        + int(bool(on_support_ok))
        + int(bool(on_long_gate_ok))
    )
    return DecisionContext(
        accel_ok=bool(accel_ok),
        on_conf_ok=bool(on_conf_ok),
        on_support_ok=bool(on_support_ok),
        long_ready=bool(long_ready),
        on_long_gate_ok=bool(on_long_gate_ok),
        accel_evidence_ok=bool(accel_evidence_ok),
        re_on_active=bool(re_on_active),
        re_on_short_ok=bool(re_on_short_ok),
        re_on_accel_ok=bool(re_on_accel_ok),
        on_entry_ready=bool(on_entry_ready),
        on_entry_vote_sum=int(on_entry_vote_sum),
    )


def step_fsm(
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
    re_on_confirm_min_sec: float,
    re_on_require_short_trigger: bool,
    re_on_require_accel: bool,
    re_on_grace_sec: float,
    on_soft_confirmed: bool,
    cal_on_support_confirm_min: float,
    off_hold_down_sec: float,
    short_trigger: bool,
    short_high: bool,
    collapse_ok: bool,
    off_vote_core: bool,
    long_off_confirmed: bool,
    force_off_now: bool,
    on_consecutive_required: int,
    off_confirm_min_sec: float,
    gate_flags: DecisionContext,
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
        off_age_sec = (
            float(t1 - float(st.last_off_t))
            if np.isfinite(st.last_off_t)
            else float("nan")
        )
        re_on_active = bool(
            np.isfinite(off_age_sec)
            and (float(off_age_sec) >= 0.0)
            and (float(off_age_sec) < float(re_on_grace_sec))
        )
        cand_start_t = float(st.active_start_t) if st.active_start_t is not None else float(t1)
        if st.active_start_t is None:
            st.active_start_t = float(t1)
            st.active_start_update_idx = int(upd_idx)
            cand_start_t = float(t1)
        cand_age = float(t1 - cand_start_t)
        effective_on_confirm_min_sec = (
            float(max(float(on_confirm_min_sec), float(re_on_confirm_min_sec)))
            if bool(re_on_active)
            else float(on_confirm_min_sec)
        )
        keep_candidate = bool(
            bool(short_trigger)
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
        if bool(re_on_active) and bool(re_on_require_short_trigger):
            keep_candidate = bool(keep_candidate and bool(short_trigger))
        if bool(re_on_active) and bool(re_on_require_accel):
            keep_candidate = bool(keep_candidate and bool(gate_flags.accel_evidence_ok))
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
        elif float(cand_age) >= float(effective_on_confirm_min_sec):
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
        if bool(force_off_now):
            phase_now = PHASE_OFF_CONFIRMED
            transition_reason = "forced_off_damped_with_low_evidence"
        elif (float(on_age) >= float(off_hold_down_sec)) and bool(off_vote_core):
            phase_now = PHASE_OFF_CANDIDATE
            transition_reason = "on_confirmed_to_off_candidate"
            if st.off_candidate_start_t is None:
                st.off_candidate_start_t = float(t1)
                st.off_candidate_start_update_idx = int(upd_idx)
    elif phase_now == PHASE_OFF_CANDIDATE:
        if bool(force_off_now):
            phase_now = PHASE_OFF_CONFIRMED
            transition_reason = "forced_off_damped_with_low_evidence"
        else:
            if st.off_candidate_start_t is None:
                st.off_candidate_start_t = float(t1)
                st.off_candidate_start_update_idx = int(upd_idx)
            off_cand_age = float(t1 - float(st.off_candidate_start_t))
            recover_to_on = bool(
                short_high
                and (int(st.on_short_votes.sum) >= int(on_consecutive_required))
                and (not bool(collapse_ok))
            )
            if recover_to_on:
                phase_now = PHASE_ON_CONFIRMED
                transition_reason = "off_candidate_revert_to_on_confirmed"
                st.off_candidate_start_t = None
                st.off_candidate_start_update_idx = None
                st.long_off_votes.clear()
            elif bool(long_off_confirmed) and (float(off_cand_age) >= float(off_confirm_min_sec)):
                phase_now = PHASE_OFF_CONFIRMED
                transition_reason = "off_candidate_to_off_confirmed"
    return str(phase_now), str(transition_reason)


def emit_events(
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
    interval_post_submit: Callable[[dict, float], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
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
            if status_cb is not None:
                status_cb(
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
            if status_cb is not None:
                status_cb(
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
            if status_cb is not None:
                status_cb(
                    f"[ALERT] RISK_OFF | upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                    f"t_end={t1:.3f} | score={float(metrics_kwargs.get('score', np.nan)):.3e} | "
                    f"S={float(metrics_kwargs.get('evidence', st.evidence)):+.3f} | reason={metrics_kwargs.get('reason', '')}"
                )
            if np.isfinite(start_t):
                if status_cb is not None:
                    status_cb(
                        f"[INTERVAL] dev={key[0]} | ch={key[1]} | start={start_t:.3f}(upd={start_idx:03d}) | "
                        f"end={t1:.3f}(upd={upd_idx:03d}) | duration={duration_sec:.3f}s"
                    )
                interval_ev, next_interval_id = _emit_or_stitch_interval(
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
                    status_cb=status_cb,
                )
                if interval_post_submit is not None:
                    interval_post_submit(
                        interval_ev,
                        float(metrics_kwargs.get("rms_decay_event_win_sec", np.nan)),
                    )
            if on_event is not None:
                on_event(ev)
        st.last_off_t = float(t1)
        st.active_start_t = None
        st.active_start_update_idx = None
        st.on_candidate_streak = 0
        st.on_short_votes.clear()
        st.on_soft_votes.clear()
        st.warmup_on_votes.clear()
        st.warmup_on_start_t = None
        st.warmup_on_start_update_idx = None
        st.prev_long_ratio_on = float("nan")
        st.on_support_ema = 0.0
        st.long_off_votes.clear()
        # Keep long-baseline store, but reset event-memory histories on OFF confirm.
        st.long_score_hist.clear()
        st.long_off_recent_hist.clear()
        st.long_ready_streak = 0
        st.off_candidate_start_t = None
        st.off_candidate_start_update_idx = None
        st.on_event_emitted = False
        st.phase = PHASE_OFF

    return int(raw_risk_interval_count), int(suppressed_interval_count), int(next_interval_id)



__all__ = [
    "_finite_or_nan",
    "_build_risk_event_metrics",
    "_emit_or_stitch_interval",
    "_extract_features",
    "_update_baseline_and_long_stats",
    "_compute_quality_and_support",
    "_compute_off_path",
    "compute_tick_features",
    "build_decision_context",
    "step_fsm",
    "emit_events",
]
