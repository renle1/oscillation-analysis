"""Operator alert policy layer built on top of raw detector events."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .osc_config_modul import (
    BASELINE_MISSING_POLICY_DISABLE,
    BASELINE_MISSING_POLICY_ERROR,
    BASELINE_MISSING_POLICY_WARN_AND_FALLBACK,
    AlertBandConfig,
    AlertPolicyConfig,
    BaselineConfig,
)


def _as_float(v: object) -> float:
    """Best-effort float conversion returning NaN on conversion failure."""

    try:
        out = float(v)
    except Exception:
        return float("nan")
    return float(out) if np.isfinite(out) else float("nan")


def _first_finite(*vals: object) -> float:
    """Return first finite float from candidates, else NaN."""

    for v in vals:
        f = _as_float(v)
        if np.isfinite(f):
            return float(f)
    return float("nan")


def _damping_ratio_from(freq_hz: object, damping_per_sec: object) -> float:
    """Compute damping ratio zeta from sigma(=damping/sec) and omega=2*pi*f."""

    f = _as_float(freq_hz)
    d = _as_float(damping_per_sec)
    if (not np.isfinite(f)) or (f <= 0.0) or (not np.isfinite(d)) or (d < 0.0):
        return float("nan")
    omega = float(2.0 * np.pi * f)
    denom = float(np.sqrt((d * d) + (omega * omega)))
    if (not np.isfinite(denom)) or (denom <= 0.0):
        return float("nan")
    zeta = float(d / denom)
    return float(zeta) if np.isfinite(zeta) else float("nan")


def _normalize_stats(raw: object) -> dict[str, object]:
    """Normalize one baseline stats object into known keys."""

    if not isinstance(raw, dict):
        return {}
    out = {
        "mean_energy": _as_float(raw.get("mean_energy", np.nan)),
        "std_energy": _as_float(raw.get("std_energy", np.nan)),
        "median_energy": _as_float(raw.get("median_energy", np.nan)),
        "mad_energy": _as_float(raw.get("mad_energy", np.nan)),
        "known_mode_freqs": raw.get("known_mode_freqs", []),
        "known_mode_damping_baseline": raw.get("known_mode_damping_baseline", []),
    }
    if not isinstance(out["known_mode_freqs"], list):
        out["known_mode_freqs"] = []
    if not isinstance(out["known_mode_damping_baseline"], list):
        out["known_mode_damping_baseline"] = []
    return out


@dataclass
class BaselineStore:
    """Read-only baseline snapshot for channel+band lookups."""

    cfg: BaselineConfig
    by_channel_band: dict[tuple[str, str], dict[str, object]]
    by_band: dict[str, dict[str, object]]

    @classmethod
    def load(
        cls,
        *,
        baseline_cfg: BaselineConfig,
        status_cb: Callable[[str], None] | None = None,
    ) -> "BaselineStore":
        """Load baseline json if present; otherwise return fallback-only store."""

        status = print if status_cb is None else status_cb
        cfg = BaselineConfig(
            baseline_enabled=bool(baseline_cfg.baseline_enabled),
            baseline_file_path=str(baseline_cfg.baseline_file_path),
            baseline_missing_policy=str(baseline_cfg.baseline_missing_policy),
            baseline_fallback_mean_energy=float(baseline_cfg.baseline_fallback_mean_energy),
            baseline_fallback_std_energy=float(baseline_cfg.baseline_fallback_std_energy),
            baseline_fallback_std_epsilon=float(baseline_cfg.baseline_fallback_std_epsilon),
        )
        cfg.validate()
        path = str(cfg.baseline_file_path).strip()
        by_channel_band: dict[tuple[str, str], dict[str, object]] = {}
        by_band: dict[str, dict[str, object]] = {}

        if (not bool(cfg.baseline_enabled)) or (not path):
            return cls(cfg=cfg, by_channel_band=by_channel_band, by_band=by_band)
        if not os.path.isfile(path):
            policy = str(cfg.baseline_missing_policy).strip().lower()
            if policy == BASELINE_MISSING_POLICY_ERROR:
                raise FileNotFoundError(f"baseline_file_path not found: {path}")
            if policy == BASELINE_MISSING_POLICY_WARN_AND_FALLBACK:
                status(f"[BASELINE] missing file, fallback mode | path={path}")
            return cls(cfg=cfg, by_channel_band=by_channel_band, by_band=by_band)

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            policy = str(cfg.baseline_missing_policy).strip().lower()
            if policy == BASELINE_MISSING_POLICY_ERROR:
                raise RuntimeError(f"baseline load failed: {path}") from exc
            status(f"[BASELINE] load failed, fallback mode | path={path} | err={exc}")
            return cls(cfg=cfg, by_channel_band=by_channel_band, by_band=by_band)

        if isinstance(raw, dict):
            channels = raw.get("channels", {})
            if isinstance(channels, dict):
                for ch_name, ch_payload in channels.items():
                    if not isinstance(ch_payload, dict):
                        continue
                    for band_name, band_stats in ch_payload.items():
                        by_channel_band[(str(ch_name), str(band_name))] = _normalize_stats(band_stats)
            bands = raw.get("bands", {})
            if isinstance(bands, dict):
                for band_name, band_stats in bands.items():
                    by_band[str(band_name)] = _normalize_stats(band_stats)
        status(
            f"[BASELINE] loaded | path={path} | channel_band_stats={len(by_channel_band)} | band_stats={len(by_band)}"
        )
        return cls(cfg=cfg, by_channel_band=by_channel_band, by_band=by_band)

    def get_band_stats(self, channel: str, band_name: str) -> dict[str, object]:
        """Return channel-specific band stats or global-band stats if available."""

        key = (str(channel), str(band_name))
        if key in self.by_channel_band:
            return dict(self.by_channel_band[key])
        if str(band_name) in self.by_band:
            return dict(self.by_band[str(band_name)])
        return {}

    def energy_to_z(
        self,
        *,
        channel: str,
        band_name: str,
        energy_value: float,
    ) -> tuple[float, float, float, bool]:
        """Convert energy value to z-score using loaded baseline or fallback."""

        stats = self.get_band_stats(str(channel), str(band_name))
        mean = _as_float(stats.get("mean_energy", np.nan))
        std = _as_float(stats.get("std_energy", np.nan))
        used_fallback = False

        if not np.isfinite(mean):
            mean = float(self.cfg.baseline_fallback_mean_energy)
            used_fallback = True
        if (not np.isfinite(std)) or (std <= 0.0):
            std = float(self.cfg.baseline_fallback_std_energy)
            used_fallback = True
        std = float(max(float(self.cfg.baseline_fallback_std_epsilon), std))

        if not np.isfinite(energy_value):
            return float("nan"), float(mean), float(std), bool(used_fallback)
        z = float((float(energy_value) - float(mean)) / float(std))
        return (float(z) if np.isfinite(z) else float("nan")), float(mean), float(std), bool(used_fallback)


def _select_band(*, freq_hz: float, band_cfgs: Sequence[AlertBandConfig]) -> AlertBandConfig:
    """Select alert band by dominant frequency, with inter_area fallback."""

    bands = [b for b in band_cfgs if isinstance(b, AlertBandConfig)]
    if not bands:
        raise ValueError("band_cfgs must not be empty")
    if np.isfinite(freq_hz):
        for b in bands:
            if (float(freq_hz) >= float(b.freq_low_hz)) and (float(freq_hz) <= float(b.freq_high_hz)):
                return b
    for b in bands:
        if str(b.band_name).strip().lower() == "inter_area":
            return b
    return bands[0]


def _compute_energy_proxy(
    interval_ev: dict[str, object],
    support_ev: dict[str, object] | None,
    *,
    band_name: str,
) -> tuple[float, str, float, float]:
    """Compute energy proxy and emit source metadata for traceability."""

    interval_raw = _first_finite(
        interval_ev.get("energy_value"),
        interval_ev.get("band_energy"),
        interval_ev.get("rms_energy"),
    )
    support_raw = float("nan")
    if np.isfinite(interval_raw):
        return float(interval_raw), "interval_direct", float(interval_raw), float(support_raw)

    bmap_interval = interval_ev.get("band_energy_by_name")
    if isinstance(bmap_interval, dict):
        e_by_band = _as_float(bmap_interval.get(str(band_name), np.nan))
        if np.isfinite(e_by_band):
            return float(e_by_band), "interval_band_map", float(interval_raw), float(support_raw)

    if isinstance(support_ev, dict):
        support_raw = _first_finite(
            support_ev.get("energy_value"),
            support_ev.get("band_energy"),
            support_ev.get("rms_energy"),
        )
        chosen_band_name = str(band_name).strip()
        support_band_name = str(support_ev.get("band_name", "")).strip()
        support_band_match = (not support_band_name) or (support_band_name == chosen_band_name)
        if support_band_match and np.isfinite(support_raw):
            return float(support_raw), "support_direct", float(interval_raw), float(support_raw)

        bmap_sup = support_ev.get("band_energy_by_name")
        if isinstance(bmap_sup, dict):
            e_by_band = _as_float(bmap_sup.get(str(band_name), np.nan))
            if np.isfinite(e_by_band):
                return float(e_by_band), "support_band_map", float(interval_raw), float(support_raw)

        rms_evt = _as_float(support_ev.get("rms_decay_event", np.nan))
        if np.isfinite(rms_evt):
            return float(abs(rms_evt)), "support_rms_decay_event", float(interval_raw), float(support_raw)

    score = _first_finite(interval_ev.get("risk_score"), interval_ev.get("score"))
    if np.isfinite(score) and (score > 0.0):
        return float(abs(np.log10(max(score, 1e-18)))), "score_fallback", float(interval_raw), float(support_raw)
    return float("nan"), "score_fallback", float(interval_raw), float(support_raw)


def _pick_modal_summary(
    *,
    interval_id: int,
    mp_map: dict[int, dict[str, object]],
    prony_map: dict[int, dict[str, object]],
) -> dict[str, object]:
    """Select best available modal summary between Prony and MP."""

    mp = mp_map.get(int(interval_id))
    pr = prony_map.get(int(interval_id))

    def _build_from(prefix: str, rec: dict[str, object] | None) -> dict[str, object]:
        if not isinstance(rec, dict):
            return {
                "modal_source": "none",
                "modal_status": "none",
                "dominant_freq_hz": float("nan"),
                "dominant_damping_per_sec": float("nan"),
                "dominant_signed_rate_per_sec": float("nan"),
                "dominant_damping_ratio": float("nan"),
                "modal_fit_r2": float("nan"),
                "modal_mode_count": 0,
                "modal_signal_std": float("nan"),
            }
        if prefix == "mp":
            f = _first_finite(
                rec.get("mp_primary_mode_freq_hz", np.nan),
                rec.get("mp_dominant_freq_hz", np.nan),
            )
            signed_rate = _first_finite(
                rec.get("mp_primary_mode_signed_rate_per_sec", np.nan),
                rec.get("mp_primary_mode_rate_per_sec", np.nan),
                rec.get("mp_dominant_damping_per_sec", np.nan),
            )
            dps = _first_finite(
                rec.get("mp_dominant_damping_per_sec", np.nan),
                (abs(float(signed_rate)) if np.isfinite(signed_rate) else np.nan),
            )
            drz = _first_finite(
                rec.get("mp_dominant_damping_ratio", np.nan),
                _damping_ratio_from(f, abs(float(signed_rate)) if np.isfinite(signed_rate) else np.nan),
                _damping_ratio_from(f, dps),
            )
        else:
            f = _first_finite(rec.get(f"{prefix}_dominant_freq_hz", np.nan))
            dps = _first_finite(rec.get(f"{prefix}_dominant_damping_per_sec", np.nan))
            signed_rate = float(dps)
            drz = _first_finite(
                rec.get(f"{prefix}_dominant_damping_ratio", np.nan),
                _damping_ratio_from(f, dps),
            )
        return {
            "modal_source": str(prefix),
            "modal_status": str(rec.get(f"{prefix}_status", "")),
            "dominant_freq_hz": float(f),
            "dominant_damping_per_sec": float(dps),
            "dominant_signed_rate_per_sec": float(signed_rate),
            "dominant_damping_ratio": float(drz),
            "modal_fit_r2": _first_finite(rec.get(f"{prefix}_fit_r2", np.nan)),
            "modal_mode_count": int(rec.get(f"{prefix}_mode_count", 0)),
            "modal_signal_std": _first_finite(rec.get(f"{prefix}_signal_std", np.nan)),
        }

    a = _build_from("prony", pr)
    b = _build_from("mp", mp)
    a_ok = str(a.get("modal_status", "")) == "ok"
    b_ok = str(b.get("modal_status", "")) == "ok"
    if a_ok and b_ok:
        a_r2 = _as_float(a.get("modal_fit_r2", np.nan))
        b_r2 = _as_float(b.get("modal_fit_r2", np.nan))
        if np.isfinite(a_r2) and np.isfinite(b_r2):
            return a if a_r2 >= b_r2 else b
        return a
    if a_ok:
        return a
    if b_ok:
        return b
    return a if a.get("modal_source") != "none" else b


def _find_interval_support_event(
    *,
    interval_ev: dict[str, object],
    risk_events_by_key: dict[tuple[str, str], list[dict[str, object]]],
) -> dict[str, object] | None:
    """Pick best matching risk event for one finalized interval."""

    dev = str(interval_ev.get("device", ""))
    ch = str(interval_ev.get("channel", ""))
    key = (dev, ch)
    candidates = risk_events_by_key.get(key, [])
    if not candidates:
        return None
    s0 = _as_float(interval_ev.get("start_t", np.nan))
    e0 = _as_float(interval_ev.get("end_t", np.nan))
    in_span: list[dict[str, object]] = []
    for ev in candidates:
        s1 = _as_float(ev.get("start_t", np.nan))
        e1 = _as_float(ev.get("end_t", np.nan))
        if (not np.isfinite(s1)) or (not np.isfinite(e1)):
            continue
        if np.isfinite(s0) and np.isfinite(e0) and (s1 >= (s0 - 1e-6)) and (e1 <= (e0 + 1e-6)):
            in_span.append(ev)
    if in_span:
        in_span.sort(key=lambda ev: _as_float(ev.get("end_t", np.nan)))
        return in_span[-1]
    candidates.sort(key=lambda ev: _as_float(ev.get("end_t", np.nan)))
    return candidates[-1]


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


def _freq_match(f1: float, f2: float, tol_hz: float) -> bool:
    """Frequency match predicate with unknown-frequency permissive handling."""

    if (not np.isfinite(f1)) or (not np.isfinite(f2)):
        return True
    return abs(float(f1) - float(f2)) <= float(max(0.0, tol_hz))


def _level_name(level_num: int) -> str:
    if int(level_num) >= 3:
        return "corrective"
    if int(level_num) == 2:
        return "investigate"
    return "advisory"


def evaluate_operator_alerts(
    *,
    events: Sequence[dict[str, object]],
    alert_policy_cfg: AlertPolicyConfig,
    alert_band_cfgs: Sequence[AlertBandConfig],
    baseline_store: BaselineStore,
) -> list[dict[str, object]]:
    """Build operator-facing alert events from raw interval and modal events."""

    policy = alert_policy_cfg
    if not bool(policy.operator_alert_enabled):
        return []
    bands = tuple(b for b in alert_band_cfgs if isinstance(b, AlertBandConfig))
    if not bands:
        return []

    interval_events = [dict(ev) for ev in events if str(ev.get("event", "")) == "interval_final"]
    if not interval_events:
        return []

    risk_events_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    mp_map: dict[int, dict[str, object]] = {}
    prony_map: dict[int, dict[str, object]] = {}
    for ev in events:
        ev_name = str(ev.get("event", ""))
        if ev_name in {"risk_off", "risk_interval_open"}:
            key = (str(ev.get("device", "")), str(ev.get("channel", "")))
            risk_events_by_key.setdefault(key, []).append(dict(ev))
            continue
        if ev_name == "interval_analysis_mp":
            mp_map[int(ev.get("interval_id", -1))] = dict(ev)
            continue
        if ev_name == "interval_analysis_prony":
            prony_map[int(ev.get("interval_id", -1))] = dict(ev)
            continue

    candidates: list[dict[str, object]] = []
    for interval_ev in interval_events:
        interval_id = int(interval_ev.get("interval_id", -1))
        support_ev = _find_interval_support_event(interval_ev=interval_ev, risk_events_by_key=risk_events_by_key)
        modal = _pick_modal_summary(interval_id=interval_id, mp_map=mp_map, prony_map=prony_map)
        freq_hz = _first_finite(
            modal.get("dominant_freq_hz", np.nan),
            (support_ev or {}).get("f_welch", np.nan),
            (support_ev or {}).get("f_fft", np.nan),
            (support_ev or {}).get("f_zc", np.nan),
        )
        band = _select_band(freq_hz=float(freq_hz), band_cfgs=bands)
        energy_value, energy_source, interval_energy_value_raw, support_energy_value_raw = _compute_energy_proxy(
            interval_ev=interval_ev,
            support_ev=support_ev,
            band_name=str(band.band_name),
        )
        energy_z, mean_energy, std_energy, used_baseline_fallback = baseline_store.energy_to_z(
            channel=str(interval_ev.get("channel", "")),
            band_name=str(band.band_name),
            energy_value=float(energy_value),
        )
        duration_sec = _as_float(interval_ev.get("duration_sec", np.nan))
        dominant_signed_rate_per_sec = _as_float(
            modal.get("dominant_signed_rate_per_sec", np.nan)
        )
        dominant_damping_per_sec = _as_float(modal.get("dominant_damping_per_sec", np.nan))
        damping_ratio = _first_finite(
            modal.get("dominant_damping_ratio", np.nan),
            _damping_ratio_from(
                modal.get("dominant_freq_hz", np.nan),
                (
                    abs(float(dominant_signed_rate_per_sec))
                    if np.isfinite(dominant_signed_rate_per_sec)
                    else dominant_damping_per_sec
                ),
            ),
            _damping_ratio_from(
                modal.get("dominant_freq_hz", np.nan),
                dominant_damping_per_sec,
            ),
        )
        fit_r2 = _as_float(modal.get("modal_fit_r2", np.nan))
        mode_count = int(modal.get("modal_mode_count", 0))
        signal_std = _as_float(modal.get("modal_signal_std", np.nan))
        modal_reliable = bool(
            np.isfinite(fit_r2)
            and (fit_r2 >= float(policy.modal_min_fit_r2))
            and (mode_count >= int(policy.modal_min_mode_count))
            and np.isfinite(signal_std)
            and (signal_std >= float(policy.modal_min_signal_std))
        )

        reasons: list[str] = []
        level_num = 0
        if np.isfinite(duration_sec) and (duration_sec >= float(band.persist_sec_level1)):
            level_num = max(level_num, 1)
            reasons.append("persistence_l1")
        if np.isfinite(duration_sec) and (duration_sec >= float(band.persist_sec_level2)):
            level_num = max(level_num, 2)
            reasons.append("persistence_l2")
        if np.isfinite(energy_z) and (energy_z >= float(band.energy_level1_z)):
            level_num = max(level_num, 1)
            reasons.append("energy_l1")
        if np.isfinite(energy_z) and (energy_z >= float(band.energy_level2_z)):
            level_num = max(level_num, 2)
            reasons.append("energy_l2")

        review_required = 0
        if bool(band.enable_damping_gate) and np.isfinite(damping_ratio):
            if float(damping_ratio) <= float(band.damping_ratio_investigate_max):
                if modal_reliable:
                    level_num = max(level_num, 2)
                    reasons.append("low_damping_investigate")
                elif bool(policy.review_required_on_low_damping):
                    level_num = max(level_num, 1)
                    review_required = 1
                    reasons.append("low_damping_review")
            if modal_reliable and (float(damping_ratio) <= float(band.damping_ratio_corrective_max)):
                level_num = max(level_num, 3)
                reasons.append("low_damping_corrective")

        if level_num <= 0:
            continue

        candidates.append(
            {
                "interval_ev": interval_ev,
                "support_ev": support_ev if isinstance(support_ev, dict) else {},
                "band": band,
                "level_num": int(level_num),
                "reasons": reasons,
                "review_required": int(review_required),
                "freq_hz": float(freq_hz),
                "dominant_signed_rate_per_sec": float(dominant_signed_rate_per_sec),
                "dominant_damping_per_sec": float(dominant_damping_per_sec),
                "damping_ratio": float(damping_ratio),
                "modal_reliable": int(modal_reliable),
                "modal_source": str(modal.get("modal_source", "none")),
                "modal_fit_r2": float(fit_r2),
                "modal_mode_count": int(mode_count),
                "modal_signal_std": float(signal_std),
                "energy_value": float(energy_value),
                "energy_source": str(energy_source),
                "interval_energy_value_raw": float(interval_energy_value_raw),
                "support_energy_value_raw": float(support_energy_value_raw),
                "energy_z": float(energy_z),
                "baseline_mean_energy": float(mean_energy),
                "baseline_std_energy": float(std_energy),
                "used_baseline_fallback": int(used_baseline_fallback),
                "duration_sec": float(duration_sec),
            }
        )

    out: list[dict[str, object]] = []
    for cand in candidates:
        interval_ev = cand["interval_ev"]
        band = cand["band"]
        sev = int(cand["level_num"])
        ch_set: set[str] = {str(interval_ev.get("channel", ""))}
        a_start = _as_float(interval_ev.get("start_t", np.nan))
        a_end = _as_float(interval_ev.get("end_t", np.nan))
        for other in candidates:
            other_interval = other["interval_ev"]
            other_band = other["band"]
            if int(other_interval.get("interval_id", -1)) == int(interval_ev.get("interval_id", -1)):
                continue
            if str(other_band.band_name) != str(band.band_name):
                continue
            b_start = _as_float(other_interval.get("start_t", np.nan))
            b_end = _as_float(other_interval.get("end_t", np.nan))
            if not _intervals_near(
                a_start=float(a_start),
                a_end=float(a_end),
                b_start=float(b_start),
                b_end=float(b_end),
                window_sec=float(policy.multi_channel_window_sec),
            ):
                continue
            if not _freq_match(
                float(cand["freq_hz"]),
                float(other.get("freq_hz", np.nan)),
                float(policy.dominant_freq_match_tol_hz),
            ):
                continue
            ch_set.add(str(other_interval.get("channel", "")))

        ch_count = int(len(ch_set))
        wide_area_candidate = int(ch_count >= int(policy.multi_channel_min_count))
        if wide_area_candidate and sev == 1:
            sev = 2
        if bool(policy.require_multi_channel) and (ch_count < int(policy.multi_channel_min_count)) and (sev > 1):
            sev = 1

        if sev == 1 and (not bool(policy.emit_advisory)):
            continue
        if sev == 2 and (not bool(policy.emit_investigate)):
            continue
        if sev >= 3 and (not bool(policy.emit_corrective)):
            if bool(policy.emit_investigate):
                sev = 2
            elif bool(policy.emit_advisory):
                sev = 1
            else:
                continue

        support_ev = cand["support_ev"]
        out.append(
            {
                "event": "operator_alert",
                "analysis_type": "operator_policy",
                "alert_level": _level_name(sev),
                "alert_level_num": int(sev),
                "device": str(interval_ev.get("device", "")),
                "channel": str(interval_ev.get("channel", "")),
                "interval_id": int(interval_ev.get("interval_id", -1)),
                "start_t": float(interval_ev.get("start_t", np.nan)),
                "end_t": float(interval_ev.get("end_t", np.nan)),
                "duration_sec": float(cand["duration_sec"]),
                "band_name": str(band.band_name),
                "band_freq_low_hz": float(band.freq_low_hz),
                "band_freq_high_hz": float(band.freq_high_hz),
                "dominant_freq_hz": float(cand["freq_hz"]),
                "dominant_signed_rate_per_sec": float(cand["dominant_signed_rate_per_sec"]),
                "dominant_damping_per_sec": float(cand["dominant_damping_per_sec"]),
                "dominant_damping_ratio": float(cand["damping_ratio"]),
                "modal_source": str(cand["modal_source"]),
                "modal_reliable": int(cand["modal_reliable"]),
                "modal_fit_r2": float(cand["modal_fit_r2"]),
                "modal_mode_count": int(cand["modal_mode_count"]),
                "modal_signal_std": float(cand["modal_signal_std"]),
                "energy_value": float(cand["energy_value"]),
                "energy_source": str(cand["energy_source"]),
                "interval_energy_value_raw": float(cand["interval_energy_value_raw"]),
                "support_energy_value_raw": float(cand["support_energy_value_raw"]),
                "energy_z": float(cand["energy_z"]),
                "baseline_mean_energy": float(cand["baseline_mean_energy"]),
                "baseline_std_energy": float(cand["baseline_std_energy"]),
                "baseline_used_fallback": int(cand["used_baseline_fallback"]),
                "persistence_sec_level1": float(band.persist_sec_level1),
                "persistence_sec_level2": float(band.persist_sec_level2),
                "energy_level1_z": float(band.energy_level1_z),
                "energy_level2_z": float(band.energy_level2_z),
                "damping_ratio_investigate_max": float(band.damping_ratio_investigate_max),
                "damping_ratio_corrective_max": float(band.damping_ratio_corrective_max),
                "multi_channel_count": int(ch_count),
                "multi_channel_required": int(policy.require_multi_channel),
                "wide_area_candidate": int(wide_area_candidate),
                "review_required": int(cand["review_required"]),
                "operator_alert_reasons": ",".join(str(x) for x in cand["reasons"]),
                "support_reason": str(support_ev.get("reason", "")) if isinstance(support_ev, dict) else "",
                "support_transition_reason": str(support_ev.get("transition_reason", "")) if isinstance(support_ev, dict) else "",
            }
        )
    out.sort(key=lambda e: (float(_as_float(e.get("start_t", np.nan))), int(e.get("interval_id", -1))))
    return out


__all__ = [
    "BaselineStore",
    "evaluate_operator_alerts",
]
