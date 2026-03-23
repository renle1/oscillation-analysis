"""Burst alert policy layer built on top of burst_interval_final events."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .osc_config_modul import BurstPolicyConfig


def _as_float(v: object) -> float:
    """Best-effort float conversion returning NaN on conversion failure."""

    try:
        out = float(v)
    except Exception:
        return float("nan")
    return float(out) if np.isfinite(out) else float("nan")


def _freq_match(f1: float, f2: float, tol_hz: float) -> bool:
    """Frequency match predicate with unknown-frequency permissive handling."""

    if (not np.isfinite(f1)) or (not np.isfinite(f2)):
        return True
    return abs(float(f1) - float(f2)) <= float(max(0.0, tol_hz))


def _intervals_near(
    *,
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
    window_sec: float,
) -> bool:
    """Check temporal overlap or near overlap with tolerance window."""

    w = float(max(0.0, window_sec))
    if (not np.isfinite(a_start)) or (not np.isfinite(a_end)) or (not np.isfinite(b_start)) or (not np.isfinite(b_end)):
        return False
    return (a_start <= (b_end + w)) and (b_start <= (a_end + w))


def _level_name(level_num: int) -> str:
    if int(level_num) >= 2:
        return "investigate"
    return "advisory"


def _find_covering_sustained_alert(
    *,
    burst_ev: dict[str, object],
    sustained_alerts: Sequence[dict[str, object]],
    window_sec: float,
    freq_tol_hz: float,
) -> tuple[str, int]:
    """Return suppression reason + sustained interval id if burst is covered."""

    s0 = _as_float(burst_ev.get("start_t", np.nan))
    e0 = _as_float(burst_ev.get("end_t", np.nan))
    f0 = _as_float(burst_ev.get("dominant_freq_hz", np.nan))
    for sustained_ev in sustained_alerts:
        s1 = _as_float(sustained_ev.get("start_t", np.nan))
        e1 = _as_float(sustained_ev.get("end_t", np.nan))
        if not _intervals_near(
            a_start=float(s0),
            a_end=float(e0),
            b_start=float(s1),
            b_end=float(e1),
            window_sec=float(window_sec),
        ):
            continue
        f1 = _as_float(sustained_ev.get("dominant_freq_hz", np.nan))
        sustained_interval_id = int(sustained_ev.get("interval_id", -1))
        sustained_level_num = int(sustained_ev.get("alert_level_num", 0))
        sustained_modal_reliable = int(sustained_ev.get("modal_reliable", 0))

        if np.isfinite(f0) and np.isfinite(f1):
            if _freq_match(float(f0), float(f1), float(freq_tol_hz)):
                return "covered_by_sustained_same_freq", int(sustained_interval_id)
            # When sustained frequency is not modal-reliable, allow investigate-level
            # fallback coverage even on numeric mismatch.
            if (sustained_level_num >= 2) and (sustained_modal_reliable <= 0):
                return "covered_by_sustained_investigate_unreliable_freq", int(sustained_interval_id)
            continue

        # Frequency can be missing on sustained operator alerts in some pipelines.
        # In that case, allow suppression only when sustained severity is investigate+.
        if sustained_level_num >= 2:
            return "covered_by_sustained_investigate_fallback", int(sustained_interval_id)
    return "", -1


def evaluate_burst_alerts(
    *,
    events: Sequence[dict[str, object]],
    burst_policy_cfg: BurstPolicyConfig,
    status_cb: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Build burst policy alert events from burst interval final records."""

    policy = burst_policy_cfg
    if not bool(policy.burst_policy_enabled):
        return []
    interval_events = [dict(ev) for ev in events if str(ev.get("event", "")) == "burst_interval_final"]
    sustained_alert_events = [
        dict(ev)
        for ev in events
        if (str(ev.get("event", "")) == "operator_alert")
        and (str(ev.get("analysis_type", "")).strip().lower() == "operator_policy")
    ]
    sustained_alert_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for ev in sustained_alert_events:
        key = (str(ev.get("device", "")), str(ev.get("channel", "")))
        sustained_alert_by_key.setdefault(key, []).append(ev)
    for key in sustained_alert_by_key:
        sustained_alert_by_key[key].sort(
            key=lambda ev: (
                float(_as_float(ev.get("start_t", np.nan))),
                int(ev.get("interval_id", -1)),
            )
        )
    if not interval_events:
        return []
    interval_events.sort(
        key=lambda ev: (
            float(_as_float(ev.get("start_t", np.nan))),
            int(ev.get("burst_interval_id", -1)),
        )
    )

    out: list[dict[str, object]] = []
    suppressed_by_burst_id: dict[int, tuple[str, int]] = {}
    suppressed_reason_counts: dict[str, int] = {}
    if bool(policy.burst_suppress_covered_by_sustained):
        for interval_ev in interval_events:
            interval_id = int(interval_ev.get("burst_interval_id", -1))
            dev = str(interval_ev.get("device", ""))
            ch = str(interval_ev.get("channel", ""))
            sustained_list = sustained_alert_by_key.get((dev, ch), [])
            suppress_reason, sustained_interval_id = _find_covering_sustained_alert(
                burst_ev=interval_ev,
                sustained_alerts=sustained_list,
                window_sec=float(policy.burst_suppress_window_sec),
                freq_tol_hz=float(policy.burst_suppress_freq_match_tol_hz),
            )
            if not suppress_reason:
                continue
            suppressed_by_burst_id[int(interval_id)] = (str(suppress_reason), int(sustained_interval_id))
            suppressed_reason_counts[suppress_reason] = int(suppressed_reason_counts.get(suppress_reason, 0)) + 1
            if status_cb is not None:
                status_cb(
                    f"[BURST_ALERT_SUPPRESS] burst_interval_id={interval_id} | sustained_interval_id={sustained_interval_id} | "
                    f"dev={dev} | ch={ch} | reason={suppress_reason}"
                )

    suppressed_count = int(len(suppressed_by_burst_id))
    for idx, interval_ev in enumerate(interval_events):
        interval_id = int(interval_ev.get("burst_interval_id", -1))
        if int(interval_id) in suppressed_by_burst_id:
            continue
        dev = str(interval_ev.get("device", ""))
        ch = str(interval_ev.get("channel", ""))
        s0 = _as_float(interval_ev.get("start_t", np.nan))
        e0 = _as_float(interval_ev.get("end_t", np.nan))
        f0 = _as_float(interval_ev.get("dominant_freq_hz", np.nan))
        duration_sec = _as_float(interval_ev.get("duration_sec", np.nan))

        repeat_count = 1
        for prev in interval_events[:idx]:
            if int(prev.get("burst_interval_id", -1)) in suppressed_by_burst_id:
                continue
            if str(prev.get("channel", "")) != ch:
                continue
            p_start = _as_float(prev.get("start_t", np.nan))
            if (not np.isfinite(p_start)) or (not np.isfinite(s0)):
                continue
            if (float(s0) - float(p_start)) > float(policy.burst_repeat_window_sec):
                continue
            if not _freq_match(float(f0), _as_float(prev.get("dominant_freq_hz", np.nan)), float(policy.burst_policy_freq_match_tol_hz)):
                continue
            repeat_count += 1

        multi_channel_set: set[str] = {ch}
        for other in interval_events:
            if int(other.get("burst_interval_id", -1)) == int(interval_id):
                continue
            if int(other.get("burst_interval_id", -1)) in suppressed_by_burst_id:
                continue
            other_ch = str(other.get("channel", ""))
            if not other_ch:
                continue
            if not _freq_match(float(f0), _as_float(other.get("dominant_freq_hz", np.nan)), float(policy.burst_policy_freq_match_tol_hz)):
                continue
            if not _intervals_near(
                a_start=float(s0),
                a_end=float(e0),
                b_start=float(_as_float(other.get("start_t", np.nan))),
                b_end=float(_as_float(other.get("end_t", np.nan))),
                window_sec=float(policy.burst_multi_channel_window_sec),
            ):
                continue
            multi_channel_set.add(other_ch)

        multi_channel_count = int(len(multi_channel_set))
        reasons: list[str] = []
        level_num = 1
        if repeat_count >= int(policy.burst_investigate_repeat_count):
            reasons.append("repeat_pattern")
            level_num = max(level_num, 2)
        if multi_channel_count >= int(policy.burst_multi_channel_min_count):
            reasons.append("multi_channel_coincident")
            level_num = max(level_num, 2)
        if not reasons:
            reasons.append("single_burst_observed")

        if (level_num == 1) and (not bool(policy.burst_emit_advisory)):
            continue
        if (level_num >= 2) and (not bool(policy.burst_emit_investigate)):
            if bool(policy.burst_emit_advisory):
                level_num = 1
            else:
                continue

        out.append(
            {
                "event": "burst_operator_alert",
                "analysis_type": "burst_policy",
                "alert_level": _level_name(level_num),
                "alert_level_num": int(level_num),
                "device": dev,
                "channel": ch,
                "burst_interval_id": int(interval_id),
                "start_t": float(s0),
                "end_t": float(e0),
                "duration_sec": float(duration_sec),
                "dominant_freq_hz": float(f0),
                "burst_support_peak": float(_as_float(interval_ev.get("burst_support_peak", np.nan))),
                "burst_confidence_peak": float(_as_float(interval_ev.get("burst_confidence_peak", np.nan))),
                "burst_score_peak": float(_as_float(interval_ev.get("burst_score_peak", np.nan))),
                "burst_candidate_true_ticks": int(interval_ev.get("burst_candidate_true_ticks", 0)),
                "burst_short_trigger_hits": int(interval_ev.get("burst_short_trigger_hits", 0)),
                "burst_accel_hits": int(interval_ev.get("burst_accel_hits", 0)),
                "burst_repeat_count": int(repeat_count),
                "burst_repeat_window_sec": float(policy.burst_repeat_window_sec),
                "burst_multi_channel_count": int(multi_channel_count),
                "burst_multi_channel_required": int(policy.burst_multi_channel_min_count),
                "burst_policy_reasons": ",".join(reasons),
            }
        )
    out.sort(
        key=lambda ev: (
            float(_as_float(ev.get("start_t", np.nan))),
            int(ev.get("burst_interval_id", -1)),
        )
    )
    if status_cb is not None and bool(policy.burst_suppress_covered_by_sustained):
        reason_summary = ",".join(
            f"{k}:{suppressed_reason_counts[k]}"
            for k in sorted(suppressed_reason_counts)
        ) or "none"
        status_cb(
            f"[BURST_ALERT_SUPPRESS_SUMMARY] suppressed={suppressed_count}/{len(interval_events)} | "
            f"reasons={reason_summary} | window_sec={float(policy.burst_suppress_window_sec)} | "
            f"freq_tol_hz={float(policy.burst_suppress_freq_match_tol_hz)}"
        )
    return out


__all__ = [
    "evaluate_burst_alerts",
]
