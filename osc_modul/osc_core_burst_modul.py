"""Sidecar burst detector core (parallel to sustained-risk FSM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .osc_config_modul import BurstConfig
from .osc_state_modul import (
    BURST_PHASE_ACTIVE,
    BURST_PHASE_CANDIDATE,
    BURST_PHASE_OFF,
    BurstChannelState,
    TickFeatures,
)


def _finite_or_nan(v: object) -> float:
    """Convert value to finite float or NaN."""

    try:
        out = float(v)
    except Exception:
        return float("nan")
    return float(out) if np.isfinite(out) else float("nan")


def _clip01(v: object) -> float:
    """Clamp numeric input into [0, 1] with NaN fallback to 0."""

    f = _finite_or_nan(v)
    if not np.isfinite(f):
        return 0.0
    return float(np.clip(float(f), 0.0, 1.0))


@dataclass
class BurstTickFeatures:
    """Per-tick burst feature bundle from shared tick features."""

    t1: float
    score: float
    reason: str
    reason_ok: bool
    short_trigger: bool
    accel_ok: bool
    support: float
    confidence: float
    c_spec: float
    c_fft: float
    c_env: float
    f_welch: float
    f_fft: float
    freq_hz: float
    freq_pair_ok: bool
    freq_range_ok: bool
    freq_consistent: bool
    candidate_core: bool


@dataclass
class BurstDecisionContext:
    """Derived burst decisions for one tick."""

    entry_core: bool
    hold_core: bool
    density_ok: bool
    candidate_vote_sum: int
    candidate_vote_n: int


def _build_burst_metrics(
    *,
    st: BurstChannelState,
    transition_reason: str,
    reason: str,
    fallback_freq_hz: float,
) -> dict[str, object]:
    """Assemble burst event payload metrics from state peaks and current context."""

    st_signal = st.signal
    dominant_freq_hz = _finite_or_nan(st_signal.freq_peak_hz)
    if not np.isfinite(dominant_freq_hz):
        dominant_freq_hz = _finite_or_nan(st_signal.freq_ema_hz)
    if not np.isfinite(dominant_freq_hz):
        dominant_freq_hz = _finite_or_nan(fallback_freq_hz)
    return {
        "dominant_freq_hz": float(dominant_freq_hz),
        "burst_support_peak": _finite_or_nan(st_signal.support_peak),
        "burst_confidence_peak": _finite_or_nan(st_signal.confidence_peak),
        "burst_score_peak": _finite_or_nan(st_signal.score_peak),
        "burst_candidate_true_ticks": int(st_signal.candidate_true_ticks),
        "burst_short_trigger_hits": int(st_signal.short_trigger_hits),
        "burst_accel_hits": int(st_signal.accel_hits),
        "reason": str(reason),
        "transition_reason": str(transition_reason),
    }


def _reset_episode_state(st: BurstChannelState, *, clear_votes: bool) -> None:
    """Reset burst episode memory while keeping long-lived channel state."""

    st_signal = st.signal
    st_votes = st.votes
    st_signal.candidate_start_t = None
    st_signal.candidate_start_update_idx = None
    st_signal.candidate_last_seen_t = float("nan")
    st_signal.active_start_t = None
    st_signal.active_start_update_idx = None
    st_signal.active_last_seen_t = float("nan")
    st_signal.on_event_emitted = False
    st_signal.freq_ema_hz = float("nan")
    st_signal.freq_peak_hz = float("nan")
    st_signal.support_peak = float("nan")
    st_signal.confidence_peak = float("nan")
    st_signal.score_peak = float("nan")
    st_signal.short_trigger_hits = 0
    st_signal.accel_hits = 0
    st_signal.candidate_true_ticks = 0
    if bool(clear_votes):
        st_votes.candidate_votes.clear()


def _update_episode_peaks(st: BurstChannelState, feat: BurstTickFeatures) -> None:
    """Update burst episode maxima and frequency tracking."""

    st_signal = st.signal
    f = _finite_or_nan(feat.freq_hz)
    s = _finite_or_nan(feat.support)
    c = _finite_or_nan(feat.confidence)
    score = _finite_or_nan(feat.score)
    if np.isfinite(f):
        prev = _finite_or_nan(st_signal.freq_ema_hz)
        st_signal.freq_ema_hz = float(f) if (not np.isfinite(prev)) else float((0.7 * prev) + (0.3 * f))
    if np.isfinite(s):
        prev_peak = _finite_or_nan(st_signal.support_peak)
        if (not np.isfinite(prev_peak)) or (float(s) >= float(prev_peak)):
            st_signal.support_peak = float(s)
            if np.isfinite(f):
                st_signal.freq_peak_hz = float(f)
    if np.isfinite(c):
        prev_c = _finite_or_nan(st_signal.confidence_peak)
        st_signal.confidence_peak = float(c) if (not np.isfinite(prev_c)) else float(max(prev_c, c))
    if np.isfinite(score):
        prev_score = _finite_or_nan(st_signal.score_peak)
        st_signal.score_peak = float(score) if (not np.isfinite(prev_score)) else float(max(prev_score, score))
    if bool(feat.short_trigger):
        st_signal.short_trigger_hits += 1
    if bool(feat.accel_ok):
        st_signal.accel_hits += 1
    if bool(feat.candidate_core):
        st_signal.candidate_true_ticks += 1


def _emit_or_stitch_burst_interval(
    *,
    key: tuple[str, str],
    start_t: float,
    start_update_idx: int,
    end_t: float,
    end_update_idx: int,
    end_reason: str,
    status: str,
    events: list[dict[str, object]],
    last_interval_by_key: dict[tuple[str, str], dict[str, object]],
    next_interval_id: int,
    merge_gap_sec: float,
    metrics: dict[str, object],
    status_cb: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], int]:
    """Emit or stitch burst interval final events."""

    prev = last_interval_by_key.get(key)
    if (
        prev is not None
        and prev.get("status") in {"closed", "open"}
        and np.isfinite(_finite_or_nan(prev.get("end_t", np.nan)))
        and (float(start_t) - float(prev.get("end_t", np.nan)) <= float(merge_gap_sec))
    ):
        prev["end_t"] = float(end_t)
        prev["end_update_idx"] = int(end_update_idx)
        prev["duration_sec"] = float(float(prev.get("end_t", np.nan)) - float(prev.get("start_t", np.nan)))
        prev["status"] = str(status)
        prev["end_reason"] = str(end_reason)
        prev["stitch_count"] = int(prev.get("stitch_count", 0)) + 1
        for k, v in metrics.items():
            if k in {
                "dominant_freq_hz",
                "burst_support_peak",
                "burst_confidence_peak",
                "burst_score_peak",
            }:
                old = _finite_or_nan(prev.get(k, np.nan))
                new = _finite_or_nan(v)
                if np.isfinite(new) and ((not np.isfinite(old)) or (float(new) >= float(old))):
                    prev[k] = float(new)
            elif k in {"burst_candidate_true_ticks", "burst_short_trigger_hits", "burst_accel_hits"}:
                prev[k] = int(max(int(prev.get(k, 0)), int(v)))
        if status_cb is not None:
            status_cb(
                f"[BURST_STITCH] dev={key[0]} | ch={key[1]} | burst_interval_id={prev.get('burst_interval_id')} | "
                f"new_end={float(end_t):.3f} | duration={float(prev.get('duration_sec', np.nan)):.3f}s | reason={end_reason}"
            )
        return prev, int(next_interval_id)

    interval_ev: dict[str, object] = {
        "event": "burst_interval_final",
        "burst_interval_id": int(next_interval_id),
        "device": str(key[0]),
        "channel": str(key[1]),
        "start_t": float(start_t),
        "start_update_idx": int(start_update_idx),
        "end_t": float(end_t),
        "end_update_idx": int(end_update_idx),
        "duration_sec": float(float(end_t) - float(start_t)),
        "end_reason": str(end_reason),
        "status": str(status),
        "stitch_count": 0,
        **dict(metrics),
    }
    events.append(interval_ev)
    last_interval_by_key[key] = interval_ev
    return interval_ev, int(next_interval_id + 1)


def build_burst_tick_features(
    *,
    tick: TickFeatures,
    st: BurstChannelState,
    burst_cfg: BurstConfig,
) -> BurstTickFeatures:
    """Build burst features from shared tick features."""

    tick_signal = tick.signal
    tick_quality = tick.quality
    st_signal = st.signal
    c_spec = _clip01(tick_quality.c_spec)
    c_fft = _clip01(tick_quality.c_fft)
    c_env = _clip01(tick_quality.c_env)
    w_spec = float(max(0.0, burst_cfg.burst_conf_w_spec))
    w_fft = float(max(0.0, burst_cfg.burst_conf_w_fft))
    w_env = float(max(0.0, burst_cfg.burst_conf_w_env))
    w_sum = float(w_spec + w_fft + w_env)
    confidence = float("nan")
    if w_sum > 0.0:
        confidence = float(((w_spec * c_spec) + (w_fft * c_fft) + (w_env * c_env)) / w_sum)

    f_welch = _finite_or_nan(tick_quality.f_welch)
    f_fft = _finite_or_nan(tick_quality.f_fft)
    freq_pair_ok = True
    if np.isfinite(f_welch) and np.isfinite(f_fft):
        freq_pair_ok = bool(abs(float(f_welch) - float(f_fft)) <= float(burst_cfg.burst_freq_match_tol_hz))
    freq_hz = float("nan")
    if np.isfinite(f_welch) and np.isfinite(f_fft) and bool(freq_pair_ok):
        freq_hz = float(0.5 * (float(f_welch) + float(f_fft)))
    elif np.isfinite(f_welch):
        freq_hz = float(f_welch)
    elif np.isfinite(f_fft):
        freq_hz = float(f_fft)
    freq_range_ok = bool(
        np.isfinite(freq_hz)
        and (float(freq_hz) >= float(burst_cfg.burst_freq_low_hz))
        and (float(freq_hz) <= float(burst_cfg.burst_freq_high_hz))
    )

    prev_freq = _finite_or_nan(st_signal.freq_ema_hz)
    freq_consistent = True
    if np.isfinite(freq_hz) and np.isfinite(prev_freq):
        freq_consistent = bool(abs(float(freq_hz) - float(prev_freq)) <= float(burst_cfg.burst_freq_stability_tol_hz))

    reason = str(tick_signal.reason)
    reason_ok = bool((reason == "ok") and bool(tick_signal.score_reason_ok))
    support = _finite_or_nan(tick_quality.feature_support_ema)
    if not np.isfinite(support):
        support = _finite_or_nan(tick_quality.feature_support_score)
    score = _finite_or_nan(tick_signal.score)
    short_trigger = bool(tick_signal.short_trigger or tick_signal.short_high)
    accel_ok = bool(tick_quality.gate_onset_acceleration_ok)

    candidate_core = bool(
        bool(freq_pair_ok)
        and bool(freq_range_ok)
        and bool(freq_consistent)
        and np.isfinite(support)
        and (float(support) >= float(burst_cfg.burst_entry_support_min))
        and np.isfinite(confidence)
        and (float(confidence) >= float(burst_cfg.burst_entry_conf_min))
        and ((not bool(burst_cfg.burst_require_short_trigger)) or bool(short_trigger))
        and ((not bool(burst_cfg.burst_require_accel_on_entry)) or bool(accel_ok))
        and ((not bool(burst_cfg.burst_require_reason_ok)) or bool(reason_ok))
    )

    return BurstTickFeatures(
        t1=float(tick_signal.t1),
        score=float(score),
        reason=reason,
        reason_ok=bool(reason_ok),
        short_trigger=bool(short_trigger),
        accel_ok=bool(accel_ok),
        support=float(support),
        confidence=float(confidence),
        c_spec=float(c_spec),
        c_fft=float(c_fft),
        c_env=float(c_env),
        f_welch=float(f_welch),
        f_fft=float(f_fft),
        freq_hz=float(freq_hz),
        freq_pair_ok=bool(freq_pair_ok),
        freq_range_ok=bool(freq_range_ok),
        freq_consistent=bool(freq_consistent),
        candidate_core=bool(candidate_core),
    )


def build_burst_decision_context(
    *,
    st: BurstChannelState,
    feat: BurstTickFeatures,
    burst_cfg: BurstConfig,
) -> BurstDecisionContext:
    """Build entry/hold/density decisions for one burst tick."""

    st_votes = st.votes
    st_votes.candidate_votes.append(1 if bool(feat.candidate_core) else 0)
    while len(st_votes.candidate_votes) > int(burst_cfg.burst_recent_window_ticks):
        st_votes.candidate_votes.popleft()
    vote_sum = int(st_votes.candidate_votes.sum)
    vote_n = int(len(st_votes.candidate_votes))
    density_ok = bool(
        (vote_n >= int(burst_cfg.burst_recent_window_ticks))
        and (vote_sum >= int(burst_cfg.burst_recent_required_ticks))
    )
    hold_core = bool(
        bool(feat.freq_pair_ok)
        and bool(feat.freq_range_ok)
        and bool(feat.freq_consistent)
        and np.isfinite(feat.support)
        and (float(feat.support) >= float(burst_cfg.burst_hold_support_min))
        and np.isfinite(feat.confidence)
        and (float(feat.confidence) >= float(burst_cfg.burst_hold_conf_min))
        and ((not bool(burst_cfg.burst_require_reason_ok)) or bool(feat.reason_ok))
    )
    return BurstDecisionContext(
        entry_core=bool(feat.candidate_core),
        hold_core=bool(hold_core),
        density_ok=bool(density_ok),
        candidate_vote_sum=int(vote_sum),
        candidate_vote_n=int(vote_n),
    )


def step_burst_fsm(
    *,
    st: BurstChannelState,
    feat: BurstTickFeatures,
    decision: BurstDecisionContext,
    upd_idx: int,
    burst_cfg: BurstConfig,
) -> tuple[str, str]:
    """Step simple burst FSM: OFF -> CANDIDATE -> ACTIVE."""

    st_signal = st.signal
    phase_now = str(st_signal.phase)
    transition_reason = str(feat.reason)
    t1 = float(feat.t1)
    if bool(decision.entry_core):
        st_signal.candidate_last_seen_t = float(t1)

    if phase_now == BURST_PHASE_OFF:
        if bool(decision.density_ok):
            _reset_episode_state(st, clear_votes=False)
            st_signal.candidate_start_t = float(t1)
            st_signal.candidate_start_update_idx = int(upd_idx)
            if bool(decision.entry_core):
                st_signal.candidate_last_seen_t = float(t1)
            phase_now = BURST_PHASE_CANDIDATE
            transition_reason = "burst_off_to_candidate"
            _update_episode_peaks(st, feat)
    elif phase_now == BURST_PHASE_CANDIDATE:
        if st_signal.candidate_start_t is None:
            st_signal.candidate_start_t = float(t1)
            st_signal.candidate_start_update_idx = int(upd_idx)
        _update_episode_peaks(st, feat)
        cand_age = float(t1 - float(st_signal.candidate_start_t))
        last_seen_age = (
            float(t1 - float(st_signal.candidate_last_seen_t))
            if np.isfinite(_finite_or_nan(st_signal.candidate_last_seen_t))
            else float("inf")
        )
        if (not bool(decision.density_ok)) and (float(last_seen_age) > float(burst_cfg.burst_hold_gap_sec)):
            phase_now = BURST_PHASE_OFF
            transition_reason = "burst_candidate_drop"
            _reset_episode_state(st, clear_votes=True)
        elif bool(decision.density_ok) and (float(cand_age) >= float(burst_cfg.burst_confirm_min_sec)):
            phase_now = BURST_PHASE_ACTIVE
            transition_reason = "burst_candidate_to_active"
            st_signal.active_start_t = float(st_signal.candidate_start_t) if st_signal.candidate_start_t is not None else float(t1)
            st_signal.active_start_update_idx = (
                int(st_signal.candidate_start_update_idx)
                if st_signal.candidate_start_update_idx is not None
                else int(upd_idx)
            )
            st_signal.active_last_seen_t = float(t1)
            st_signal.on_event_emitted = False
    elif phase_now == BURST_PHASE_ACTIVE:
        _update_episode_peaks(st, feat)
        if bool(decision.hold_core):
            st_signal.active_last_seen_t = float(t1)
        hold_age = (
            float(t1 - float(st_signal.active_last_seen_t))
            if np.isfinite(_finite_or_nan(st_signal.active_last_seen_t))
            else float("inf")
        )
        if float(hold_age) > float(burst_cfg.burst_hold_gap_sec):
            phase_now = BURST_PHASE_OFF
            transition_reason = "burst_active_to_off_gap"
            st_signal.last_off_t = float(t1)
    return str(phase_now), str(transition_reason)


def emit_burst_events(
    *,
    events: list[dict[str, object]],
    on_event: Callable[[dict], None] | None,
    st: BurstChannelState,
    key: tuple[str, str],
    upd_idx: int,
    t1: float,
    phase_now: str,
    transition_reason: str,
    feat: BurstTickFeatures,
    raw_burst_interval_count: int,
    last_burst_interval_by_key: dict[tuple[str, str], dict[str, object]],
    next_burst_interval_id: int,
    merge_gap_sec: float,
    status_cb: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """Emit burst transition events and burst_interval_final records."""

    st_signal = st.signal
    prev_active = bool(str(st_signal.phase) == BURST_PHASE_ACTIVE)
    now_active = bool(str(phase_now) == BURST_PHASE_ACTIVE)

    if now_active and (not prev_active):
        if st_signal.active_start_t is None:
            st_signal.active_start_t = float(t1)
            st_signal.active_start_update_idx = int(upd_idx)
        if not bool(st_signal.on_event_emitted):
            metrics = _build_burst_metrics(
                st=st,
                transition_reason="burst_on",
                reason=str(feat.reason),
                fallback_freq_hz=float(feat.freq_hz),
            )
            ev = {
                "event": "burst_on",
                "update_idx": int(upd_idx),
                "device": str(key[0]),
                "channel": str(key[1]),
                "t_end": float(t1),
                "start_t": float(st_signal.active_start_t),
                "start_update_idx": int(st_signal.active_start_update_idx) if st_signal.active_start_update_idx is not None else int(upd_idx),
                **metrics,
            }
            events.append(ev)
            if status_cb is not None:
                status_cb(
                    f"[BURST] ON | upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                    f"start={float(st_signal.active_start_t):.3f} | t_end={float(t1):.3f} | "
                    f"f={float(metrics.get('dominant_freq_hz', np.nan)):.2f}Hz"
                )
            if on_event is not None:
                on_event(ev)
            st_signal.on_event_emitted = True
        return int(raw_burst_interval_count), int(next_burst_interval_id)

    if (not now_active) and prev_active:
        start_t = float(st_signal.active_start_t) if st_signal.active_start_t is not None else float("nan")
        start_idx = int(st_signal.active_start_update_idx) if st_signal.active_start_update_idx is not None else -1
        duration_sec = float(float(t1) - float(start_t)) if np.isfinite(start_t) else float("nan")
        raw_burst_interval_count += 1
        metrics = _build_burst_metrics(
            st=st,
            transition_reason=str(transition_reason),
            reason=str(feat.reason),
            fallback_freq_hz=float(feat.freq_hz),
        )
        off_ev = {
            "event": "burst_off",
            "update_idx": int(upd_idx),
            "device": str(key[0]),
            "channel": str(key[1]),
            "t_end": float(t1),
            "start_t": float(start_t),
            "start_update_idx": int(start_idx),
            "end_t": float(t1),
            "end_update_idx": int(upd_idx),
            "duration_sec": float(duration_sec),
            "end_reason": str(transition_reason),
            **metrics,
        }
        events.append(off_ev)
        if status_cb is not None:
            status_cb(
                f"[BURST] OFF | upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                f"duration={float(duration_sec):.3f}s | reason={transition_reason}"
            )
        if np.isfinite(start_t):
            interval_ev, next_burst_interval_id = _emit_or_stitch_burst_interval(
                key=key,
                start_t=float(start_t),
                start_update_idx=int(start_idx),
                end_t=float(t1),
                end_update_idx=int(upd_idx),
                end_reason=str(transition_reason),
                status="closed",
                events=events,
                last_interval_by_key=last_burst_interval_by_key,
                next_interval_id=int(next_burst_interval_id),
                merge_gap_sec=float(merge_gap_sec),
                metrics=metrics,
                status_cb=status_cb,
            )
            if status_cb is not None:
                status_cb(
                    f"[BURST_INTERVAL] dev={key[0]} | ch={key[1]} | "
                    f"start={float(start_t):.3f}(upd={int(start_idx):03d}) | end={float(t1):.3f}(upd={int(upd_idx):03d}) | "
                    f"duration={float(duration_sec):.3f}s | burst_interval_id={int(interval_ev.get('burst_interval_id', -1))}"
                )
        if on_event is not None:
            on_event(off_ev)
        st_signal.last_off_t = float(t1)
        _reset_episode_state(st, clear_votes=True)

    return int(raw_burst_interval_count), int(next_burst_interval_id)


def emit_burst_stream_end_open_event(
    *,
    events: list[dict[str, object]],
    on_event: Callable[[dict], None] | None,
    st: BurstChannelState,
    key: tuple[str, str],
    end_t: float,
    end_update_idx: int,
    last_burst_interval_by_key: dict[tuple[str, str], dict[str, object]],
    next_burst_interval_id: int,
    merge_gap_sec: float,
    status_cb: Callable[[str], None] | None = None,
) -> int:
    """Finalize still-open burst interval at stream end."""

    st_signal = st.signal
    if str(st_signal.phase) != BURST_PHASE_ACTIVE:
        return int(next_burst_interval_id)
    start_t = float(st_signal.active_start_t) if st_signal.active_start_t is not None else float("nan")
    start_idx = int(st_signal.active_start_update_idx) if st_signal.active_start_update_idx is not None else -1
    if not np.isfinite(start_t):
        return int(next_burst_interval_id)
    duration_sec = float(float(end_t) - float(start_t))
    metrics = _build_burst_metrics(
        st=st,
        transition_reason="stream_end_open",
        reason="stream_end",
        fallback_freq_hz=float(st_signal.freq_ema_hz),
    )
    if not bool(st_signal.on_event_emitted):
        on_ev = {
            "event": "burst_on",
            "update_idx": int(start_idx if start_idx >= 0 else end_update_idx),
            "device": str(key[0]),
            "channel": str(key[1]),
            "t_end": float(start_t),
            "start_t": float(start_t),
            "start_update_idx": int(start_idx),
            **metrics,
        }
        events.append(on_ev)
        if on_event is not None:
            on_event(on_ev)
        st_signal.on_event_emitted = True
    open_ev = {
        "event": "burst_interval_open",
        "device": str(key[0]),
        "channel": str(key[1]),
        "start_t": float(start_t),
        "start_update_idx": int(start_idx),
        "end_t": float(end_t),
        "end_update_idx": int(end_update_idx),
        "duration_sec": float(duration_sec),
        "end_reason": "stream_end_open",
        **metrics,
    }
    events.append(open_ev)
    interval_ev, next_burst_interval_id = _emit_or_stitch_burst_interval(
        key=key,
        start_t=float(start_t),
        start_update_idx=int(start_idx),
        end_t=float(end_t),
        end_update_idx=int(end_update_idx),
        end_reason="stream_end_open",
        status="open",
        events=events,
        last_interval_by_key=last_burst_interval_by_key,
        next_interval_id=int(next_burst_interval_id),
        merge_gap_sec=float(merge_gap_sec),
        metrics=metrics,
        status_cb=status_cb,
    )
    if status_cb is not None:
        status_cb(
            f"[BURST_OPEN] dev={key[0]} | ch={key[1]} | start={float(start_t):.3f}(upd={int(start_idx):03d}) | "
            f"end={float(end_t):.3f}(stream_end) | duration={float(duration_sec):.3f}s | "
            f"burst_interval_id={int(interval_ev.get('burst_interval_id', -1))}"
        )
    if on_event is not None:
        on_event(open_ev)
    return int(next_burst_interval_id)


__all__ = [
    "BurstTickFeatures",
    "BurstDecisionContext",
    "build_burst_tick_features",
    "build_burst_decision_context",
    "step_burst_fsm",
    "emit_burst_events",
    "emit_burst_stream_end_open_event",
]
