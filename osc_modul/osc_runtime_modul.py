"""Runtime/orchestration functions for modular streaming detector."""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Callable, Sequence

import numpy as np

from .osc_alert_burst_policy_modul import evaluate_burst_alerts
from .osc_alert_policy_modul import BaselineStore, evaluate_operator_alerts
from .osc_config_modul import (
    AlertBandConfig,
    AlertPolicyConfig,
    BaselineConfig,
    BurstConfig,
    BurstPolicyConfig,
    DEFAULT_DEVICE,
    DEFAULT_RISK_CUT,
    DEFAULT_UPDATE_SEC,
    PRESET_CHOICES,
    PRESET_SAFE,
    STREAM_INPUT_MODE_CHOICES,
    STREAM_INPUT_MODE_LIVE_CSV_TAIL,
    STREAM_INPUT_MODE_REPLAY_CSV,
    STREAM_INPUT_MODE_TEST_SIGNAL,
    WIN_SEC,
    DetectorConfig,
    LongConfig,
    ModalPreprocessConfig,
    MPPostAnalysisConfig,
    PronyPostAnalysisConfig,
    PeriodicityConfig,
    StreamConfig,
    ThresholdConfig,
    _apply_detector_overrides,
    default_alert_band_profiles,
    _resolve_mp_runtime_controls,
    _resolve_prony_runtime_controls,
    make_preset_config,
)
from .osc_core_burst_modul import (
    build_burst_decision_context,
    build_burst_tick_features,
    emit_burst_events,
    emit_burst_stream_end_open_event,
    step_burst_fsm,
)
from .osc_core_fsm_modul import (
    _build_risk_event_metrics,
    _emit_or_stitch_interval,
    build_decision_context,
    compute_tick_features,
    emit_events,
    step_fsm,
)
from .osc_core_mp_modul import _IntervalMPPostRuntime
from .osc_core_prony_modul import _IntervalPronyPostRuntime
from .osc_core_signal_modul import BASE_SEC, RMS_WIN_SEC, _is_risk_active_phase, compute_band_rms_energies
from .osc_io_modul import (
    _guess_default_stream_csv,
    build_update_batches_from_test_signal,
    build_update_batches_from_voltage_csv,
    iter_live_update_batches_from_csv_tail,
)
from .osc_state_modul import (
    BURST_PHASE_OFF,
    BURST_PHASE_ACTIVE,
    BurstChannelState,
    ChannelStreamState,
    MP_RUNTIME_MODE_CHOICES,
    QualityCacheSnapshot,
    PHASE_OFF,
    PHASE_OFF_CONFIRMED,
)


_EVENT_LOG_PREFIXES = (
    "[ALERT]",
    "[INTERVAL]",
    "[INTERVAL_OPEN]",
    "[BURST]",
    "[BURST_INTERVAL]",
    "[BURST_OPEN]",
    "[BURST_ALERT]",
    "[BURST_STITCH]",
    "[MP_POST]",
    "[PRONY_POST]",
    "[INTERVAL_ENERGY]",
    "[OPERATOR_ALERT]",
    "[STREAM_DONE]",
    "[BASELINE]",
)


def _is_event_log_line(line: str) -> bool:
    """Return True when log line should be shown in event-only console mode."""

    s = str(line).strip()
    return any(s.startswith(p) for p in _EVENT_LOG_PREFIXES)


def _build_status_logger(
    *,
    status_cb: Callable[[str], None] | None,
    console_event_only: bool,
    log_to_file: bool,
    log_file_path: str,
) -> Callable[[str], None]:
    """Build unified status logger: file sink + console/event filter + callback."""

    path = str(log_file_path).strip()
    warned = {"io": False}
    if bool(log_to_file) and path:
        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            # Start each run with a fresh log file (avoid unbounded append-only logs).
            with open(abs_path, "w", encoding="utf-8"):
                pass
        except Exception:
            warned["io"] = True
            print(f"[WARN] log file init failed: {abs_path}")
    else:
        abs_path = ""

    def _status(msg: str) -> None:
        line = str(msg)
        if bool(log_to_file) and abs_path:
            try:
                with open(abs_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                if not warned["io"]:
                    warned["io"] = True
                    print(f"[WARN] log file write failed: {abs_path}")
        if status_cb is not None:
            status_cb(line)
            return
        if (not bool(console_event_only)) or _is_event_log_line(line):
            print(line)

    return _status


def _validate_detector_runtime_values(
    *,
    stream_cfg: StreamConfig,
    threshold_cfg: ThresholdConfig,
    long_cfg: LongConfig,
    periodicity_cfg: PeriodicityConfig,
    modal_preprocess_cfg: ModalPreprocessConfig,
    mp_cfg: MPPostAnalysisConfig,
    prony_cfg: PronyPostAnalysisConfig,
    burst_cfg: BurstConfig,
    burst_policy_cfg: BurstPolicyConfig,
    alert_policy_cfg: AlertPolicyConfig,
    baseline_cfg: BaselineConfig,
    alert_band_cfgs: Sequence[AlertBandConfig],
    cut_on: float,
    cut_off: float,
) -> None:
    """Validate runtime config via section-level validate methods."""

    cfg_obj = DetectorConfig(
        stream=stream_cfg,
        threshold=threshold_cfg,
        long=long_cfg,
        periodicity=periodicity_cfg,
        modal_preprocess=modal_preprocess_cfg,
        mp_post=mp_cfg,
        prony_post=prony_cfg,
        burst=burst_cfg,
        burst_policy=burst_policy_cfg,
        alert_policy=alert_policy_cfg,
        baseline=baseline_cfg,
        alert_bands=tuple(alert_band_cfgs),
    )
    cfg_obj.validate(cut_on=float(cut_on), cut_off=float(cut_off))


def _validate_detector_runtime_args(
    *,
    stream_cfg: StreamConfig,
    threshold_cfg: ThresholdConfig,
    long_cfg: LongConfig,
    periodicity_cfg: PeriodicityConfig,
    modal_preprocess_cfg: ModalPreprocessConfig,
    mp_cfg: MPPostAnalysisConfig,
    prony_cfg: PronyPostAnalysisConfig,
    burst_cfg: BurstConfig,
    burst_policy_cfg: BurstPolicyConfig,
    alert_policy_cfg: AlertPolicyConfig,
    baseline_cfg: BaselineConfig,
    alert_band_cfgs: Sequence[AlertBandConfig],
    cut_on: float,
    cut_off: float,
) -> None:
    """Validate grouped runtime config and derived cuts."""

    _validate_detector_runtime_values(
        stream_cfg=stream_cfg,
        threshold_cfg=threshold_cfg,
        long_cfg=long_cfg,
        periodicity_cfg=periodicity_cfg,
        modal_preprocess_cfg=modal_preprocess_cfg,
        mp_cfg=mp_cfg,
        prony_cfg=prony_cfg,
        burst_cfg=burst_cfg,
        burst_policy_cfg=burst_policy_cfg,
        alert_policy_cfg=alert_policy_cfg,
        baseline_cfg=baseline_cfg,
        alert_band_cfgs=alert_band_cfgs,
        cut_on=float(cut_on),
        cut_off=float(cut_off),
    )


def _build_event_metrics_kwargs(
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
    rms_decay_on_ok: int,
    rms_decay_off_hint: int,
    rms_decay_event: float,
    rms_decay_event_r2: float,
    rms_decay_event_n: int,
    rms_decay_event_win_sec: float,
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
) -> dict[str, object]:
    """Centralized risk-event metric argument assembly."""

    return {
        "score": float(score),
        "confidence": float(confidence),
        "confidence_raw": float(confidence_raw),
        "confidence_cal": float(confidence_cal),
        "cal_on_conf_thr": float(cal_on_conf_thr),
        "on_support": float(on_support),
        "on_support_ema": float(on_support_ema),
        "on_soft_votes_sum": int(on_soft_votes_sum),
        "on_soft_votes_n": int(on_soft_votes_n),
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
        "evidence": float(evidence),
        "long_ratio_on": float(long_ratio_on),
        "long_ratio_off": float(long_ratio_off),
        "long_ratio_off_recent": float(long_ratio_off_recent),
        "long_off_n_recent": int(long_off_n_recent),
        "long_off_votes_sum": int(long_off_votes_sum),
        "long_off_votes_n": int(long_off_votes_n),
        "acf_peak": float(acf_peak),
        "acf_period_sec": float(acf_period_sec),
        "acf_lag_steps": int(acf_lag_steps),
        "acf_n": int(acf_n),
        "long_zmax": float(long_zmax),
        "long_n": int(long_n),
        "baseline_n": int(baseline_n),
    }


def _build_tick_metrics_kwargs(
    *,
    tick,
    st: ChannelStreamState,
    transition_reason: str,
) -> dict[str, object]:
    """Build event-metrics kwargs from one tick without local field fan-out."""

    return _build_event_metrics_kwargs(
        score=float(tick.score),
        confidence=float(tick.confidence),
        confidence_raw=float(tick.confidence_raw),
        confidence_cal=float(tick.confidence_cal),
        cal_on_conf_thr=float(tick.cal_on_conf_thr),
        on_support=float(tick.on_support),
        on_support_ema=float(tick.on_support_ema),
        on_soft_votes_sum=int(tick.on_soft_vote_sum),
        on_soft_votes_n=int(len(st.on_soft_votes)),
        c_acf=float(tick.c_acf),
        c_spec=float(tick.c_spec),
        c_env=float(tick.c_env),
        c_fft=float(tick.c_fft),
        c_freq_agree=float(tick.c_freq_agree),
        f_welch=float(tick.f_welch),
        f_zc=float(tick.f_zc),
        f_fft=float(tick.f_fft),
        A_tail=float(tick.A_tail),
        D_tail=float(tick.D_tail),
        rms_decay=float(tick.rms_decay),
        rms_decay_r2=float(tick.rms_decay_r2),
        rms_decay_n=int(tick.rms_decay_n),
        rms_decay_on_ok=int(tick.rms_decay_on_ok),
        rms_decay_off_hint=int(tick.rms_decay_off_hint),
        rms_decay_event=float(tick.rms_decay_event),
        rms_decay_event_r2=float(tick.rms_decay_event_r2),
        rms_decay_event_n=int(tick.rms_decay_event_n),
        rms_decay_event_win_sec=float(tick.rms_decay_event_win_sec),
        reason=str(tick.reason),
        transition_reason=str(transition_reason),
        evidence=float(st.evidence),
        long_ratio_on=float(tick.long_ratio_on),
        long_ratio_off=float(tick.long_ratio_off),
        long_ratio_off_recent=float(tick.long_ratio_off_recent),
        long_off_n_recent=int(tick.long_off_n_recent),
        long_off_votes_sum=int(st.long_off_votes.sum),
        long_off_votes_n=int(len(st.long_off_votes)),
        acf_peak=float(tick.acf_peak),
        acf_period_sec=float(tick.acf_period_sec),
        acf_lag_steps=int(tick.acf_lag_steps),
        acf_n=int(tick.acf_n),
        long_zmax=float(tick.long_zmax),
        long_n=int(tick.long_n),
        baseline_n=int(len(st.long_baseline_hist)),
    )


def _build_cached_metrics_kwargs(
    *,
    st: ChannelStreamState,
    quality: QualityCacheSnapshot | None,
    transition_reason: str,
    reason: str,
) -> dict[str, object]:
    """Build event metrics from cached quality when no current tick exists."""

    q = quality if quality is not None else QualityCacheSnapshot()
    return _build_event_metrics_kwargs(
        score=float("nan"),
        confidence=float(q.confidence),
        confidence_raw=float(q.confidence_raw),
        confidence_cal=float(q.confidence_cal),
        cal_on_conf_thr=float("nan"),
        on_support=float("nan"),
        on_support_ema=float(st.on_support_ema),
        on_soft_votes_sum=int(st.on_soft_votes.sum),
        on_soft_votes_n=int(len(st.on_soft_votes)),
        c_acf=float(q.c_acf),
        c_spec=float(q.c_spec),
        c_env=float(q.c_env),
        c_fft=float(q.c_fft),
        c_freq_agree=float(q.c_freq_agree),
        f_welch=float(q.f_welch),
        f_zc=float(q.f_zc),
        f_fft=float(q.f_fft),
        A_tail=float("nan"),
        D_tail=float("nan"),
        rms_decay=float(q.rms_decay_local),
        rms_decay_r2=float(q.rms_decay_local_r2),
        rms_decay_n=int(q.rms_decay_local_n),
        rms_decay_on_ok=int(q.rms_decay_local_on_ok),
        rms_decay_off_hint=int(q.rms_decay_local_off_hint),
        rms_decay_event=float(q.rms_decay_event),
        rms_decay_event_r2=float(q.rms_decay_event_r2),
        rms_decay_event_n=int(q.rms_decay_event_n),
        rms_decay_event_win_sec=float(q.rms_decay_event_win_sec),
        reason=str(reason),
        transition_reason=str(transition_reason),
        evidence=float(st.evidence),
        long_ratio_on=float("nan"),
        long_ratio_off=float("nan"),
        long_ratio_off_recent=float("nan"),
        long_off_n_recent=0,
        long_off_votes_sum=int(st.long_off_votes.sum),
        long_off_votes_n=int(len(st.long_off_votes)),
        acf_peak=float(q.acf_peak),
        acf_period_sec=float(q.acf_period_sec),
        acf_lag_steps=int(q.acf_lag_steps),
        acf_n=int(q.acf_n),
        long_zmax=float("nan"),
        long_n=int(len(st.long_score_hist)),
        baseline_n=int(len(st.long_baseline_hist)),
    )


def _run_streaming_alert_demo_one_channel_cfg_impl(
    vcsv: str,
    *,
    device: str = DEFAULT_DEVICE,
    target_channel: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
    stream_cfg: StreamConfig | None = None,
    threshold_cfg: ThresholdConfig | None = None,
    long_cfg: LongConfig | None = None,
    periodicity_cfg: PeriodicityConfig | None = None,
    modal_preprocess_cfg: ModalPreprocessConfig | None = None,
    mp_cfg: MPPostAnalysisConfig | None = None,
    prony_cfg: PronyPostAnalysisConfig | None = None,
    burst_cfg: BurstConfig | None = None,
    burst_policy_cfg: BurstPolicyConfig | None = None,
    alert_policy_cfg: AlertPolicyConfig | None = None,
    baseline_cfg: BaselineConfig | None = None,
    alert_band_cfgs: Sequence[AlertBandConfig] | None = None,
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
    - event-aware RMS-decay contributes to OFF decay-path hints (with dedicated relaxed threshold)
    - forced OFF from repeated `excluded_damped` is guarded by long activity state
    """
    s = copy.deepcopy(stream_cfg) if stream_cfg is not None else StreamConfig()
    th = copy.deepcopy(threshold_cfg) if threshold_cfg is not None else ThresholdConfig()
    lg = copy.deepcopy(long_cfg) if long_cfg is not None else LongConfig()
    pq = copy.deepcopy(periodicity_cfg) if periodicity_cfg is not None else PeriodicityConfig()
    pp = copy.deepcopy(modal_preprocess_cfg) if modal_preprocess_cfg is not None else ModalPreprocessConfig()
    mp = copy.deepcopy(mp_cfg) if mp_cfg is not None else MPPostAnalysisConfig()
    pr = copy.deepcopy(prony_cfg) if prony_cfg is not None else PronyPostAnalysisConfig()
    bt = copy.deepcopy(burst_cfg) if burst_cfg is not None else BurstConfig()
    bp = copy.deepcopy(burst_policy_cfg) if burst_policy_cfg is not None else BurstPolicyConfig()
    ap = copy.deepcopy(alert_policy_cfg) if alert_policy_cfg is not None else AlertPolicyConfig()
    bl = copy.deepcopy(baseline_cfg) if baseline_cfg is not None else BaselineConfig()
    band_profiles = tuple(copy.deepcopy(list(alert_band_cfgs))) if alert_band_cfgs is not None else default_alert_band_profiles()
    update_sec = float(s.update_sec)
    window_sec = float(s.window_sec)
    realtime_sleep = bool(s.realtime_sleep)
    print_tick = bool(s.print_tick)
    console_event_only = bool(s.console_event_only)
    log_to_file = bool(s.log_to_file)
    log_file_path = str(s.log_file_path).strip()
    status = _build_status_logger(
        status_cb=status_cb,
        console_event_only=bool(console_event_only),
        log_to_file=bool(log_to_file),
        log_file_path=str(log_file_path),
    )
    min_interval_sec_for_alert = float(s.min_interval_sec_for_alert)
    stitch_gap_sec = float(s.stitch_gap_sec)

    base_cut = float(DEFAULT_RISK_CUT if th.risk_cut is None else th.risk_cut)
    cut_off = float(base_cut if th.risk_cut_off is None else th.risk_cut_off)
    cut_on = float((cut_off * 3.0) if th.risk_cut_on is None else th.risk_cut_on)
    if float(th.re_on_grace_sec) <= 0.0:
        th.re_on_confirm_min_sec = float(th.on_confirm_min_sec)

    _validate_detector_runtime_args(
        stream_cfg=s,
        threshold_cfg=th,
        long_cfg=lg,
        periodicity_cfg=pq,
        modal_preprocess_cfg=pp,
        mp_cfg=mp,
        prony_cfg=pr,
        burst_cfg=bt,
        burst_policy_cfg=bp,
        alert_policy_cfg=ap,
        baseline_cfg=bl,
        alert_band_cfgs=band_profiles,
        cut_on=float(cut_on),
        cut_off=float(cut_off),
    )
    baseline_store = BaselineStore.load(
        baseline_cfg=bl,
        status_cb=status,
    )
    baseline_band_defs = tuple(
        (str(b.band_name), float(b.freq_low_hz), float(b.freq_high_hz))
        for b in band_profiles
        if str(b.band_name).strip()
    )

    selected = [target_channel] if (target_channel and str(target_channel).strip()) else None
    source_mode = str(s.input_mode).strip().lower()
    source_controls_timing = False
    expected_update_total = 0
    if source_mode == STREAM_INPUT_MODE_REPLAY_CSV:
        batches = build_update_batches_from_voltage_csv(
            vcsv,
            device=device,
            update_sec=float(update_sec),
            target_channels=selected,
            max_channels=1,
        )
        if not batches:
            status("[STREAM] no batches were created")
            return []
        batch_iter = iter(batches)
        expected_update_total = int(len(batches))
        status(f"[SOURCE] mode={source_mode} | updates={int(expected_update_total)}")
    elif source_mode == STREAM_INPUT_MODE_TEST_SIGNAL:
        batches = build_update_batches_from_test_signal(
            device=device,
            update_sec=float(update_sec),
            duration_sec=float(s.test_duration_sec),
            sampling_hz=float(s.test_sampling_hz),
            freq_hz=float(s.test_signal_freq_hz),
            amp=float(s.test_signal_amp),
            noise_std=float(s.test_noise_std),
            seed=int(s.test_seed),
            channel_name=(str(target_channel).strip() if (target_channel and str(target_channel).strip()) else "V1"),
        )
        if not batches:
            status("[STREAM] no test batches were created")
            return []
        batch_iter = iter(batches)
        expected_update_total = int(len(batches))
        status(
            f"[SOURCE] mode={source_mode} | updates={int(expected_update_total)} | "
            f"freq={float(s.test_signal_freq_hz):.3f}Hz | amp={float(s.test_signal_amp):.6f}"
        )
    elif source_mode == STREAM_INPUT_MODE_LIVE_CSV_TAIL:
        if (not str(vcsv).strip()):
            raise ValueError("vcsv is required when input_mode=live_csv_tail")
        batch_iter = iter_live_update_batches_from_csv_tail(
            vcsv,
            device=device,
            update_sec=float(update_sec),
            target_channels=selected,
            max_channels=1,
            max_updates=int(s.live_max_updates),
            status_cb=status,
        )
        source_controls_timing = True
        expected_update_total = int(s.live_max_updates) if int(s.live_max_updates) > 0 else 0
        status(
            f"[SOURCE] mode={source_mode} | max_updates={int(s.live_max_updates)} | "
            f"csv={os.path.abspath(str(vcsv)) if str(vcsv).strip() else ''}"
        )
    else:
        raise ValueError(f"unsupported input_mode: {source_mode}")

    states: dict[tuple[str, str], ChannelStreamState] = {}
    burst_states: dict[tuple[str, str], BurstChannelState] = {}
    events: list[dict] = []
    raw_risk_interval_count = 0
    raw_burst_interval_count = 0
    suppressed_interval_count = 0
    last_interval_by_key: dict[tuple[str, str], dict] = {}
    last_burst_interval_by_key: dict[tuple[str, str], dict[str, object]] = {}
    next_interval_id = 1
    next_burst_interval_id = 1
    mp_runtime_mode_eff, mp_async_enabled_eff, mp_queue_maxsize_eff, mp_finalize_wait_sec_eff = _resolve_mp_runtime_controls(
        mp_runtime_mode=str(mp.mp_runtime_mode),
        mp_async_enabled=bool(mp.mp_async_enabled),
        mp_queue_maxsize=int(mp.mp_queue_maxsize),
        mp_finalize_wait_sec=float(mp.mp_finalize_wait_sec),
    )
    mp_runtime = _IntervalMPPostRuntime(
        mp_enabled=bool(mp.mp_enabled),
        mp_min_interval_sec=float(mp.mp_min_interval_sec),
        mp_onset_skip_sec=float(mp.mp_onset_skip_sec),
        mp_onset_window_sec=float(mp.mp_onset_window_sec),
        mp_onset_min_window_sec=float(mp.mp_onset_min_window_sec),
        mp_fallback_use_rms_event_window=bool(mp.mp_fallback_use_rms_event_window),
        mp_fallback_default_window_sec=float(mp.mp_fallback_default_window_sec),
        mp_model_order=int(mp.mp_model_order),
        mp_order_selection_enabled=bool(mp.mp_order_selection_enabled),
        mp_order_candidates=mp.mp_order_candidates,
        mp_max_modes=int(mp.mp_max_modes),
        mp_freq_low_hz=float(mp.mp_freq_low_hz),
        mp_freq_high_hz=float(mp.mp_freq_high_hz),
        mp_min_samples=int(mp.mp_min_samples),
        mp_dt_cv_max=float(mp.mp_dt_cv_max),
        mp_signal_std_min=float(mp.mp_signal_std_min),
        mp_singular_ratio_min=float(mp.mp_singular_ratio_min),
        mp_downsample_enabled=bool(mp.mp_downsample_enabled),
        mp_target_fs_hz=float(mp.mp_target_fs_hz),
        mp_downsample_lpf_cutoff_hz=float(mp.mp_downsample_lpf_cutoff_hz),
        mp_downsample_lpf_order=int(mp.mp_downsample_lpf_order),
        modal_preprocess_enabled=bool(pp.enabled),
        modal_preprocess_method=str(pp.method),
        modal_preprocess_hampel_half_window=int(pp.hampel_half_window),
        modal_preprocess_hampel_nsigma=float(pp.hampel_nsigma),
        modal_preprocess_max_repair_points=int(pp.max_repair_points),
        modal_preprocess_max_repair_fraction=float(pp.max_repair_fraction),
        modal_preprocess_repair_mode=str(pp.repair_mode),
        modal_preprocess_keep_raw_summary=bool(pp.keep_raw_summary),
        mp_async_enabled=bool(mp_async_enabled_eff),
        mp_queue_maxsize=int(mp_queue_maxsize_eff),
        mp_finalize_wait_sec=float(mp_finalize_wait_sec_eff),
    )
    prony_runtime_mode_eff, prony_async_enabled_eff, prony_queue_maxsize_eff, prony_finalize_wait_sec_eff = _resolve_prony_runtime_controls(
        prony_runtime_mode=str(pr.prony_runtime_mode),
        prony_async_enabled=bool(pr.prony_async_enabled),
        prony_queue_maxsize=int(pr.prony_queue_maxsize),
        prony_finalize_wait_sec=float(pr.prony_finalize_wait_sec),
    )
    prony_runtime = _IntervalPronyPostRuntime(
        prony_enabled=bool(pr.prony_enabled),
        prony_min_interval_sec=float(pr.prony_min_interval_sec),
        prony_onset_skip_sec=float(pr.prony_onset_skip_sec),
        prony_onset_window_sec=float(pr.prony_onset_window_sec),
        prony_onset_min_window_sec=float(pr.prony_onset_min_window_sec),
        prony_fallback_use_rms_event_window=bool(pr.prony_fallback_use_rms_event_window),
        prony_fallback_default_window_sec=float(pr.prony_fallback_default_window_sec),
        prony_model_order=int(pr.prony_model_order),
        prony_order_candidates=pr.prony_order_candidates,
        prony_max_modes=int(pr.prony_max_modes),
        prony_freq_low_hz=float(pr.prony_freq_low_hz),
        prony_freq_high_hz=float(pr.prony_freq_high_hz),
        prony_min_samples=int(pr.prony_min_samples),
        prony_dt_cv_max=float(pr.prony_dt_cv_max),
        prony_signal_std_min=float(pr.prony_signal_std_min),
        prony_root_mag_max=float(pr.prony_root_mag_max),
        modal_preprocess_enabled=bool(pp.enabled),
        modal_preprocess_method=str(pp.method),
        modal_preprocess_hampel_half_window=int(pp.hampel_half_window),
        modal_preprocess_hampel_nsigma=float(pp.hampel_nsigma),
        modal_preprocess_max_repair_points=int(pp.max_repair_points),
        modal_preprocess_max_repair_fraction=float(pp.max_repair_fraction),
        modal_preprocess_repair_mode=str(pp.repair_mode),
        modal_preprocess_keep_raw_summary=bool(pp.keep_raw_summary),
        prony_async_enabled=bool(prony_async_enabled_eff),
        prony_queue_maxsize=int(prony_queue_maxsize_eff),
        prony_finalize_wait_sec=float(prony_finalize_wait_sec_eff),
    )

    interval_capture_by_key: dict[tuple[str, str], dict[str, object]] = {}
    interval_samples_by_interval_id: dict[int, list[tuple[float, float]]] = {}

    def _sorted_sample_pairs(samples: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
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

    def _sync_interval_energy_capture(
        *,
        key: tuple[str, str],
        active_start_t: float,
        is_interval_active: bool,
        tick_samples: Sequence[tuple[float, float]],
    ) -> None:
        if bool(is_interval_active) and np.isfinite(active_start_t):
            cap = interval_capture_by_key.get(key)
            if cap is None:
                cap = {"start_t": float(active_start_t), "samples": []}
                interval_capture_by_key[key] = cap
            else:
                prev_start = float(cap.get("start_t", np.nan))
                if (not np.isfinite(prev_start)) or (float(active_start_t) < prev_start):
                    cap["start_t"] = float(active_start_t)
            start_t_eff = float(cap.get("start_t", active_start_t))
            samples = cap.get("samples")
            if not isinstance(samples, list):
                samples = []
                cap["samples"] = samples
            for t, v in tick_samples:
                t_f = float(t)
                v_f = float(v)
                if np.isfinite(t_f) and np.isfinite(v_f) and (t_f >= start_t_eff):
                    samples.append((t_f, v_f))
            return
        interval_capture_by_key.pop(key, None)

    def _submit_interval_post(interval_ev: dict, rms_event_win_sec: float) -> None:
        interval_copy = dict(interval_ev)
        interval_id = int(interval_copy.get("interval_id", -1))
        key = (str(interval_copy.get("device", "")), str(interval_copy.get("channel", "")))
        seg_capture = interval_capture_by_key.pop(key, None)
        seg_samples = seg_capture.get("samples", []) if isinstance(seg_capture, dict) else []
        if not isinstance(seg_samples, list):
            seg_samples = []
        merged_samples = list(interval_samples_by_interval_id.get(int(interval_id), []))
        if seg_samples:
            merged_samples.extend((float(t), float(v)) for t, v in seg_samples)
        interval_samples_by_interval_id[int(interval_id)] = _sorted_sample_pairs(merged_samples)
        mp_runtime.submit_interval(interval_ev, float(rms_event_win_sec))
        prony_runtime.submit_interval(interval_ev, float(rms_event_win_sec))
    max_keep_sec = max(
        float(window_sec) * 2.0,
        float(window_sec) + float(BASE_SEC) + float(RMS_WIN_SEC),
        float(th.rms_decay_window_sec) + float(th.rms_decay_rms_win_sec) + float(th.rms_decay_step_sec),
        float(th.rms_decay_event_max_window_sec) + float(th.rms_decay_event_rms_win_sec) + float(th.rms_decay_event_step_sec),
    )

    total_updates = 0
    for upd_idx, batch in enumerate(batch_iter, start=1):
        total_updates = int(upd_idx)
        if not batch:
            if realtime_sleep and (not bool(source_controls_timing)):
                time.sleep(float(update_sec))
            continue

        touched_keys = set()
        batch_samples_by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}
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
            key_samples = batch_samples_by_key.get(key)
            if key_samples is None:
                key_samples = []
                batch_samples_by_key[key] = key_samples
            key_samples.append((float(t_f), float(v_f)))
            touched_keys.add(key)

        for key in touched_keys:
            st = states[key]
            bst = burst_states.get(key)
            if bst is None:
                bst = BurstChannelState()
                burst_states[key] = bst
            tick_samples = batch_samples_by_key.get(key, [])
            tick = compute_tick_features(
                st,
                channel_name=str(key[1]),
                max_keep_sec=float(max_keep_sec),
                window_sec=float(window_sec),
                upd_idx=int(upd_idx),
                cut_on=float(cut_on),
                cut_off=float(cut_off),
                update_sec=float(update_sec),
                threshold_cfg=th,
                long_cfg=lg,
                periodicity_cfg=pq,
                baseline_store=baseline_store,
                baseline_band_defs=baseline_band_defs,
            )
            if tick is None:
                _sync_interval_energy_capture(
                    key=key,
                    active_start_t=(
                        float(st.active_start_t)
                        if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
                        else float("nan")
                    ),
                    is_interval_active=bool(st.active_start_t is not None),
                    tick_samples=tick_samples,
                )
                mp_runtime.sync_interval_capture(
                    key=key,
                    active_start_t=(
                        float(st.active_start_t)
                        if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
                        else float("nan")
                    ),
                    is_interval_active=bool(st.active_start_t is not None),
                    tick_samples=tick_samples,
                )
                prony_runtime.sync_interval_capture(
                    key=key,
                    active_start_t=(
                        float(st.active_start_t)
                        if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
                        else float("nan")
                    ),
                    is_interval_active=bool(st.active_start_t is not None),
                    tick_samples=tick_samples,
                )
                continue
            phase_now = str(st.phase)
            transition_reason = str(tick.reason)
            decision = build_decision_context(
                tick=tick,
                st=st,
                threshold_cfg=th,
                long_cfg=lg,
                periodicity_cfg=pq,
            )
            gate_flags = decision
            on_entry_vote_sum = int(decision.on_entry_vote_sum)
            warmup_cold_start_allowed = bool(not np.isfinite(st.last_off_t))
            warmup_handoff_active = bool(
                bool(lg.warmup_long_enabled)
                and bool(warmup_cold_start_allowed)
                and bool(tick.long_ready)
                and (int(st.long_ready_streak) <= int(lg.warmup_handoff_grace_ticks))
            )

            phase_now, transition_reason = step_fsm(
                st,
                phase_now=str(phase_now),
                t1=float(tick.t1),
                upd_idx=int(upd_idx),
                reason=str(tick.reason),
                score_reason_ok=bool(tick.score_reason_ok),
                score=float(tick.score),
                cut_off_cmp=float(tick.cut_off_cmp),
                cal_on_active=bool(tick.cal_on_active),
                on_support_ema=float(tick.on_support_ema),
                cal_on_support_hold_min=float(pq.cal_on_support_hold_min),
                on_confirm_min_sec=float(th.on_confirm_min_sec),
                re_on_confirm_min_sec=float(th.re_on_confirm_min_sec),
                re_on_require_short_trigger=bool(th.re_on_require_short_trigger),
                re_on_require_accel=bool(th.re_on_require_accel),
                re_on_grace_sec=float(th.re_on_grace_sec),
                on_soft_confirmed=bool(tick.on_soft_confirmed),
                cal_on_support_confirm_min=float(pq.cal_on_support_confirm_min),
                off_hold_down_sec=float(th.off_hold_down_sec),
                short_trigger=bool(tick.short_trigger),
                short_high=bool(tick.short_high),
                collapse_ok=bool(tick.collapse_ok),
                off_vote_core=bool(tick.off_vote_core),
                long_off_confirmed=bool(tick.long_off_confirmed),
                force_off_now=bool(tick.force_off_now),
                on_consecutive_required=int(th.on_consecutive_required),
                off_confirm_min_sec=float(th.off_confirm_min_sec),
                gate_flags=gate_flags,
            )

            _sync_interval_energy_capture(
                key=key,
                active_start_t=(
                    float(st.active_start_t)
                    if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
                    else float("nan")
                ),
                is_interval_active=bool(st.active_start_t is not None),
                tick_samples=tick_samples,
            )
            mp_runtime.sync_interval_capture(
                key=key,
                active_start_t=(
                    float(st.active_start_t)
                    if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
                    else float("nan")
                ),
                is_interval_active=bool(st.active_start_t is not None),
                tick_samples=tick_samples,
            )
            prony_runtime.sync_interval_capture(
                key=key,
                active_start_t=(
                    float(st.active_start_t)
                    if (st.active_start_t is not None) and np.isfinite(st.active_start_t)
                    else float("nan")
                ),
                is_interval_active=bool(st.active_start_t is not None),
                tick_samples=tick_samples,
            )

            risk_now = bool(_is_risk_active_phase(phase_now))

            if print_tick:
                off_vote_str = f"{int(st.long_off_votes.sum)}/{len(st.long_off_votes)}"
                warmup_vote_str = f"{int(tick.warmup_vote_sum)}/{len(st.warmup_on_votes)}"
                long_gate_mode = ("W" if bool(tick.warmup_mode) else ("H" if warmup_handoff_active else "L"))
                off_age_sec = (
                    float(tick.t1 - float(st.last_off_t))
                    if np.isfinite(st.last_off_t)
                    else float("nan")
                )
                post_off_rearm_active = bool(
                    np.isfinite(off_age_sec)
                    and (float(off_age_sec) >= 0.0)
                    and (float(off_age_sec) < float(lg.post_off_rearm_sec))
                )
                status(
                    f"[TICK] upd={upd_idx:03d} | dev={key[0]} | ch={key[1]} | "
                    f"t_end={float(tick.t1):.3f} | score={float(tick.score):.3e} | e={float(tick.e_t):+.3f} | S={float(st.evidence):+.3f} | "
                    f"phase={phase_now} | on_votes={st.on_candidate_streak}/{len(st.on_short_votes)} | "
                    f"damped_streak={st.damped_streak} | coll_streak={st.periodicity_collapse_streak} | "
                    f"Lon={float(tick.long_ratio_on):.2f} | Loff={float(tick.long_ratio_off):.2f} | LoffR={float(tick.long_ratio_off_recent):.2f} | "
                    f"Conf={float(tick.confidence):.2f}[raw={float(tick.confidence_raw):.2f},cal={float(tick.confidence_cal):.2f}]"
                    f"(acf={float(tick.c_acf):.2f},spec={float(tick.c_spec):.2f},env={float(tick.c_env):.2f},fft={float(tick.c_fft):.2f},xv={float(tick.c_freq_agree):.2f}) | "
                    f"f={float(tick.f_welch):.2f}/{float(tick.f_zc):.2f}/{float(tick.f_fft):.2f}Hz | ACF={float(tick.acf_peak):.2f}@{float(tick.acf_period_sec):.1f}s | "
                    f"RMSd={float(tick.rms_decay):+.3f}[n={int(tick.rms_decay_n)},r2={float(tick.rms_decay_r2):.2f}] | "
                    f"CALthr={float(tick.cal_on_conf_thr):.2f} | Soft={float(tick.on_support):.2f}/{float(tick.on_support_ema):.2f} | "
                    f"SV={int(tick.on_soft_vote_sum)}/{len(st.on_soft_votes)} | Wv={warmup_vote_str}({int(bool(tick.warmup_on_confirmed))}) | "
                    f"CUT[on={float(tick.cut_on_cmp):.2e},off={float(tick.cut_off_cmp):.2e}] | "
                    f"Lg={long_gate_mode}:{int(gate_flags.on_long_gate_ok)} | rearm={int(post_off_rearm_active)} | off_age_sec={float(off_age_sec):.3f} | "
                    f"Ev={int(on_entry_vote_sum)}/4 | Lv={off_vote_str} | "
                    f"G[ac={int(gate_flags.accel_ok)},cf={int(gate_flags.on_conf_ok)},su={int(gate_flags.on_support_ok)},"
                    f"lg={int(gate_flags.on_long_gate_ok)},lr={int(gate_flags.long_ready)},ra={int(gate_flags.re_on_active)},"
                    f"rs={int(gate_flags.re_on_short_ok)},rx={int(gate_flags.re_on_accel_ok)},"
                    f"ov={int(bool(tick.off_vote_core))},oc={int(bool(tick.long_off_confirmed))},fo={int(bool(tick.force_off_now))}] | "
                    f"dS={float(tick.delta_s_log):+.2f} | dE={float(tick.delta_e):+.2f} | Lzmax={float(tick.long_zmax):.2f} | "
                    f"Ln={int(tick.long_n)} | Bn={len(st.long_baseline_hist)} | reason={tick.reason} | risk={'ON' if risk_now else 'OFF'}"
                )

            metrics_kwargs = _build_tick_metrics_kwargs(
                tick=tick,
                st=st,
                transition_reason=str(transition_reason),
            )
            raw_risk_interval_count, suppressed_interval_count, next_interval_id = emit_events(
                events=events,
                on_event=on_event,
                st=st,
                key=key,
                upd_idx=int(upd_idx),
                t1=float(tick.t1),
                risk_now=bool(risk_now),
                transition_reason=str(transition_reason),
                min_interval_sec_for_alert=float(s.min_interval_sec_for_alert),
                raw_risk_interval_count=int(raw_risk_interval_count),
                suppressed_interval_count=int(suppressed_interval_count),
                last_interval_by_key=last_interval_by_key,
                next_interval_id=int(next_interval_id),
                stitch_gap_sec=float(s.stitch_gap_sec),
                metrics_kwargs=metrics_kwargs,
                interval_post_submit=_submit_interval_post,
                status_cb=status,
            )

            if bool(bt.burst_enabled):
                burst_feat = build_burst_tick_features(
                    tick=tick,
                    st=bst,
                    burst_cfg=bt,
                )
                burst_decision = build_burst_decision_context(
                    st=bst,
                    feat=burst_feat,
                    burst_cfg=bt,
                )
                burst_phase_now, burst_transition_reason = step_burst_fsm(
                    st=bst,
                    feat=burst_feat,
                    decision=burst_decision,
                    upd_idx=int(upd_idx),
                    burst_cfg=bt,
                )
                raw_burst_interval_count, next_burst_interval_id = emit_burst_events(
                    events=events,
                    on_event=on_event,
                    st=bst,
                    key=key,
                    upd_idx=int(upd_idx),
                    t1=float(tick.t1),
                    phase_now=str(burst_phase_now),
                    transition_reason=str(burst_transition_reason),
                    feat=burst_feat,
                    raw_burst_interval_count=int(raw_burst_interval_count),
                    last_burst_interval_by_key=last_burst_interval_by_key,
                    next_burst_interval_id=int(next_burst_interval_id),
                    merge_gap_sec=float(bt.burst_merge_gap_sec),
                    status_cb=status,
                )
                bst.phase = str(burst_phase_now) if str(burst_phase_now) != BURST_PHASE_OFF else BURST_PHASE_OFF

            if str(phase_now) == PHASE_OFF_CONFIRMED:
                st.phase = PHASE_OFF
            elif (not risk_now) and str(phase_now) == PHASE_OFF:
                st.phase = PHASE_OFF
            else:
                st.phase = str(phase_now)
            st.last_risk_on = bool(risk_now)

        if realtime_sleep and (not bool(source_controls_timing)):
            time.sleep(float(update_sec))

    for key, st in states.items():
        if (not st.last_risk_on) or (not st.ring):
            continue
        end_t = float(st.ring[-1][0])
        start_t = float(st.active_start_t) if st.active_start_t is not None else float("nan")
        start_idx = int(st.active_start_update_idx) if st.active_start_update_idx is not None else -1
        duration_sec = float(end_t - start_t) if np.isfinite(start_t) else float("nan")
        raw_risk_interval_count += 1
        q = st.last_quality if isinstance(st.last_quality, QualityCacheSnapshot) else None
        rms_decay_event_win_sec = float(q.rms_decay_event_win_sec) if q is not None else float("nan")
        if np.isfinite(duration_sec) and (duration_sec < float(min_interval_sec_for_alert)):
            suppressed_interval_count += 1
            status(
                f"[SUPPRESS] short_open_interval | dev={key[0]} | ch={key[1]} | "
                f"start={start_t:.3f}(upd={start_idx:03d}) | end={end_t:.3f}(stream_end) | "
                f"duration={duration_sec:.3f}s < min={float(min_interval_sec_for_alert):.3f}s"
            )
            continue
        if not st.on_event_emitted and np.isfinite(start_t):
            delayed_metrics = _build_cached_metrics_kwargs(
                st=st,
                quality=q,
                transition_reason="delayed_emit_before_open_close",
                reason="stream_end_delayed_emit",
            )
            on_ev = {
                "event": "risk_on",
                "update_idx": int(start_idx if start_idx >= 0 else total_updates),
                "device": str(key[0]),
                "channel": str(key[1]),
                "t_end": float(start_t),
                "start_t": start_t,
                "start_update_idx": start_idx,
                **_build_risk_event_metrics(**delayed_metrics),
            }
            events.append(on_ev)
            if on_event is not None:
                on_event(on_ev)
        open_metrics = _build_cached_metrics_kwargs(
            st=st,
            quality=q,
            transition_reason="stream_end_open",
            reason="stream_end",
        )
        ev = {
            "event": "risk_interval_open",
            "device": str(key[0]),
            "channel": str(key[1]),
            "start_t": start_t,
            "start_update_idx": start_idx,
            "end_t": float(end_t),
            "end_update_idx": int(total_updates),
            "duration_sec": duration_sec,
            **_build_risk_event_metrics(**open_metrics),
            "end_reason": "stream_end_open",
        }
        events.append(ev)
        status(
            f"[INTERVAL_OPEN] dev={key[0]} | ch={key[1]} | start={start_t:.3f}(upd={start_idx:03d}) | "
            f"end={end_t:.3f}(stream_end) | duration={duration_sec:.3f}s"
        )
        if np.isfinite(start_t):
            interval_ev, next_interval_id = _emit_or_stitch_interval(
                key=key,
                start_t=float(start_t),
                start_update_idx=int(start_idx),
                end_t=float(end_t),
                end_update_idx=int(total_updates),
                end_reason="stream_end_open",
                status="open",
                events=events,
                last_interval_by_key=last_interval_by_key,
                next_interval_id=int(next_interval_id),
                stitch_gap_sec=float(stitch_gap_sec),
                status_cb=status,
            )
            _submit_interval_post(interval_ev, float(rms_decay_event_win_sec))
        if on_event is not None:
            on_event(ev)

    if bool(bt.burst_enabled):
        for key, bst in burst_states.items():
            if str(bst.phase) != BURST_PHASE_ACTIVE:
                continue
            end_t = float("nan")
            st_main = states.get(key)
            if isinstance(st_main, ChannelStreamState) and st_main.ring:
                end_t = float(st_main.ring[-1][0])
            elif bst.active_start_t is not None and np.isfinite(bst.active_start_t):
                end_t = float(bst.active_start_t)
            if not np.isfinite(end_t):
                continue
            raw_burst_interval_count += 1
            next_burst_interval_id = emit_burst_stream_end_open_event(
                events=events,
                on_event=on_event,
                st=bst,
                key=key,
                end_t=float(end_t),
                end_update_idx=int(total_updates),
                last_burst_interval_by_key=last_burst_interval_by_key,
                next_burst_interval_id=int(next_burst_interval_id),
                merge_gap_sec=float(bt.burst_merge_gap_sec),
                status_cb=status,
            )
            bst.phase = BURST_PHASE_OFF

    mp_records = mp_runtime.finalize()
    if mp_records:
        events.extend(mp_records)
        status(
            f"[MP_POST] records={len(mp_records)} | mode={mp_runtime_mode_eff} | "
            f"non_blocking={int(bool(mp_async_enabled_eff))}"
        )
    prony_records = prony_runtime.finalize()
    if prony_records:
        events.extend(prony_records)
        status(
            f"[PRONY_POST] records={len(prony_records)} | mode={prony_runtime_mode_eff} | "
            f"non_blocking={int(bool(prony_async_enabled_eff))}"
        )

    interval_final_by_id: dict[int, dict[str, object]] = {}
    for ev in events:
        if str(ev.get("event", "")) == "interval_final":
            interval_final_by_id[int(ev.get("interval_id", -1))] = ev
    if interval_final_by_id and interval_samples_by_interval_id:
        mp_map = {
            int(ev.get("interval_id", -1)): dict(ev)
            for ev in mp_records
            if isinstance(ev, dict)
        }
        prony_map = {
            int(ev.get("interval_id", -1)): dict(ev)
            for ev in prony_records
            if isinstance(ev, dict)
        }

        def _pick_modal_freq_for_interval(interval_id: int) -> float:
            mp_rec = mp_map.get(int(interval_id))
            pr_rec = prony_map.get(int(interval_id))
            mp_ok = isinstance(mp_rec, dict) and (str(mp_rec.get("mp_status", "")) == "ok")
            pr_ok = isinstance(pr_rec, dict) and (str(pr_rec.get("prony_status", "")) == "ok")

            if pr_ok and mp_ok:
                pr_r2 = float(pr_rec.get("prony_fit_r2", np.nan))
                mp_r2 = float(mp_rec.get("mp_fit_r2", np.nan))
                if np.isfinite(pr_r2) and np.isfinite(mp_r2):
                    chosen = pr_rec if pr_r2 >= mp_r2 else mp_rec
                else:
                    chosen = pr_rec
            elif pr_ok:
                chosen = pr_rec
            elif mp_ok:
                chosen = mp_rec
            else:
                return float("nan")

            if chosen is pr_rec:
                return float(pr_rec.get("prony_dominant_freq_hz", np.nan))
            return float(mp_rec.get("mp_dominant_freq_hz", np.nan))

        def _select_band_name(freq_hz: float) -> str:
            if np.isfinite(freq_hz):
                for b in band_profiles:
                    if (float(freq_hz) >= float(b.freq_low_hz)) and (float(freq_hz) <= float(b.freq_high_hz)):
                        return str(b.band_name)
            for b in band_profiles:
                if str(b.band_name).strip().lower() == "inter_area":
                    return str(b.band_name)
            return str(band_profiles[0].band_name)

        band_defs = [
            (str(b.band_name), float(b.freq_low_hz), float(b.freq_high_hz))
            for b in band_profiles
        ]
        min_band_energy_samples = int(max(16, round(float(window_sec) / max(float(update_sec), 1e-6))))
        interval_energy_count = 0
        for interval_id, interval_ev in interval_final_by_id.items():
            samples = interval_samples_by_interval_id.get(int(interval_id), [])
            if not samples:
                continue
            arr = np.asarray(samples, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            t_win = arr[:, 0]
            v_win = arr[:, 1]
            band_energy_by_name = compute_band_rms_energies(
                t_win,
                v_win,
                bands=band_defs,
                min_samples=int(min_band_energy_samples),
                dt_cv_max=0.25,
                detrend_linear=True,
            )
            if not isinstance(band_energy_by_name, dict):
                continue
            dominant_freq_hz = _pick_modal_freq_for_interval(int(interval_id))
            chosen_band_name = _select_band_name(float(dominant_freq_hz))
            energy_value = float(band_energy_by_name.get(str(chosen_band_name), np.nan))
            interval_ev["band_energy_by_name"] = {
                str(k): float(v) if np.isfinite(v) else float("nan")
                for k, v in band_energy_by_name.items()
            }
            interval_ev["band_name"] = str(chosen_band_name)
            interval_ev["band_energy"] = float(energy_value)
            interval_ev["energy_value"] = float(energy_value)
            if np.isfinite(dominant_freq_hz):
                interval_ev["dominant_freq_hz"] = float(dominant_freq_hz)
            if np.isfinite(energy_value):
                interval_energy_count += 1
        status(
            f"[INTERVAL_ENERGY] computed={interval_energy_count}/{len(interval_final_by_id)} | "
            f"bands={len(band_defs)} | min_samples={int(min_band_energy_samples)}"
        )

    operator_alert_records = evaluate_operator_alerts(
        events=events,
        alert_policy_cfg=ap,
        alert_band_cfgs=band_profiles,
        baseline_store=baseline_store,
    )
    if operator_alert_records:
        events.extend(operator_alert_records)
        status(
            f"[OPERATOR_ALERT] records={len(operator_alert_records)} | enabled={int(bool(ap.operator_alert_enabled))}"
        )

    burst_alert_records = evaluate_burst_alerts(
        events=events,
        burst_policy_cfg=bp,
        status_cb=status,
    )
    if burst_alert_records:
        events.extend(burst_alert_records)
        status(
            f"[BURST_ALERT] records={len(burst_alert_records)} | enabled={int(bool(bp.burst_policy_enabled))}"
        )

    raw_risk_interval_exists = int(raw_risk_interval_count > 0)
    raw_burst_interval_exists = int(raw_burst_interval_count > 0)
    suppressed_event_exists = int(suppressed_interval_count > 0)
    updates_done = int(total_updates if total_updates > 0 else expected_update_total)
    status(
        f"[STREAM_DONE] updates={updates_done} | events={len(events)} | "
        f"raw_intervals={raw_risk_interval_count} | raw_burst_intervals={raw_burst_interval_count} | "
        f"suppressed_intervals={suppressed_interval_count} | "
        f"raw_risk_interval_exists={raw_risk_interval_exists} | suppressed_event_exists={suppressed_event_exists}"
        f" | raw_burst_interval_exists={raw_burst_interval_exists}"
    )
    return events


def _run_streaming_alert_demo_one_channel_impl(
    vcsv: str,
    *,
    cfg: DetectorConfig | None = None,
    device: str = DEFAULT_DEVICE,
    target_channel: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """Internal config-first impl boundary for streaming detector execution."""

    cfg_eff = copy.deepcopy(cfg) if cfg is not None else DetectorConfig()

    return _run_streaming_alert_demo_one_channel_cfg_impl(
        vcsv,
        device=str(device),
        target_channel=target_channel,
        on_event=on_event,
        status_cb=status_cb,
        stream_cfg=cfg_eff.stream,
        threshold_cfg=cfg_eff.threshold,
        long_cfg=cfg_eff.long,
        periodicity_cfg=cfg_eff.periodicity,
        modal_preprocess_cfg=cfg_eff.modal_preprocess,
        mp_cfg=cfg_eff.mp_post,
        prony_cfg=cfg_eff.prony_post,
        burst_cfg=cfg_eff.burst,
        burst_policy_cfg=cfg_eff.burst_policy,
        alert_policy_cfg=cfg_eff.alert_policy,
        baseline_cfg=cfg_eff.baseline,
        alert_band_cfgs=cfg_eff.alert_bands,
    )


def run_streaming_alert_demo_one_channel(
    vcsv: str,
    *,
    cfg: DetectorConfig | None = None,
    device: str = DEFAULT_DEVICE,
    target_channel: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
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
        status_cb=status_cb,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for the streaming detector runner."""

    p = argparse.ArgumentParser(description="Streaming oscillation detector runner (minimal CLI)")
    p.add_argument("--csv", default="", help="Input CSV path. If omitted, auto-discovery is used.")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="Device id label")
    p.add_argument("--channel", default="", help="Target channel name (optional)")
    p.add_argument("--preset", choices=PRESET_CHOICES, default=PRESET_SAFE, help="Detector preset profile")
    p.add_argument("--input-mode", choices=STREAM_INPUT_MODE_CHOICES, default=STREAM_INPUT_MODE_REPLAY_CSV, help="Input source mode: replay_csv, live_csv_tail, or test_signal")
    p.add_argument("--update-sec", type=float, default=DEFAULT_UPDATE_SEC, help="Stream update period in seconds")
    p.add_argument("--window-sec", type=float, default=WIN_SEC, help="Sliding evaluation window in seconds")
    p.add_argument("--live-max-updates", type=int, default=0, help="For live_csv_tail mode: stop after N updates (0 means run forever)")
    p.add_argument("--test-duration-sec", type=float, default=None, help="For test_signal mode: synthetic signal duration")
    p.add_argument("--test-sampling-hz", type=float, default=None, help="For test_signal mode: sampling rate")
    p.add_argument("--test-signal-freq-hz", type=float, default=None, help="For test_signal mode: sine frequency")
    p.add_argument("--test-signal-amp", type=float, default=None, help="For test_signal mode: sine amplitude")
    p.add_argument("--test-noise-std", type=float, default=None, help="For test_signal mode: gaussian noise std")
    p.add_argument("--test-seed", type=int, default=None, help="For test_signal mode: random seed")
    p.add_argument("--realtime-sleep", action="store_true", help="Sleep each update_sec during replay")
    p.add_argument("--no-print-tick", action="store_true", help="Disable per-tick logs")
    p.add_argument("--console-events-only", action="store_true", help="Show only event-level logs in console (still saved in file)")
    p.add_argument("--full-console-log", action="store_true", help="Show all logs in console (including tick logs)")
    p.add_argument("--log-file", default="", help="Runtime log file path")
    p.add_argument("--no-log-file", action="store_true", help="Disable runtime log file output")
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
    p.add_argument("--mp-post", action="store_true", help="Enable post-interval Matrix Pencil analysis (non-blocking)")
    p.add_argument("--mp-post-sync", action="store_true", help="Run MP post-analysis synchronously (debug only)")
    p.add_argument("--mp-policy", choices=MP_RUNTIME_MODE_CHOICES, default="", help="MP post-analysis policy mode: live(non-blocking) or replay(completion-oriented)")
    p.add_argument("--mp-model-order", type=int, default=None, help="Override Matrix Pencil model order")
    p.add_argument("--mp-order-selection", action="store_true", help="Enable adaptive MP rank selection (shared-SVD candidate sweep)")
    p.add_argument("--mp-order-candidates", default="", help="Comma-separated MP rank candidates (e.g. 4,6,8,10)")
    p.add_argument("--prony-post", action="store_true", help="Enable post-interval Prony analysis (non-blocking)")
    p.add_argument("--prony-post-sync", action="store_true", help="Run Prony post-analysis synchronously (debug only)")
    p.add_argument("--prony-policy", choices=MP_RUNTIME_MODE_CHOICES, default="", help="Prony post-analysis policy mode: live(non-blocking) or replay(completion-oriented)")
    p.add_argument("--prony-model-order", type=int, default=None, help="Override Prony model order")
    p.add_argument("--prony-order-candidates", default="", help="Comma-separated Prony order sweep candidates (e.g. 8,6,4,2)")
    p.add_argument("--no-burst", action="store_true", help="Disable sidecar high-frequency burst detector")
    p.add_argument("--no-burst-policy", action="store_true", help="Disable burst advisory/investigate policy layer")
    p.add_argument("--operator-alert", action="store_true", help="Enable operator policy alert layer")
    p.add_argument("--baseline-file", default="", help="Optional baseline JSON path for operator policy")
    return p


def main() -> None:
    """CLI entrypoint: build config, run detector, and print summary."""

    args = _build_arg_parser().parse_args()

    input_mode = str(args.input_mode).strip().lower()
    csv_path = str(args.csv).strip() if str(args.csv).strip() else _guess_default_stream_csv()
    if input_mode in {STREAM_INPUT_MODE_REPLAY_CSV, STREAM_INPUT_MODE_LIVE_CSV_TAIL}:
        if not csv_path:
            raise FileNotFoundError(
                "No stream CSV found. Provide --csv or set config/env STREAM_INPUT_CSV."
            )
        if (input_mode == STREAM_INPUT_MODE_REPLAY_CSV) and (not os.path.isfile(csv_path)):
            raise FileNotFoundError(
                f"Replay CSV not found: {csv_path}"
            )
        print(f"[MAIN] csv={os.path.abspath(csv_path)}")
    else:
        csv_path = ""
        print("[MAIN] csv=<test_signal_mode>")
    print(f"[MAIN] device={args.device} | update_sec={args.update_sec} | preset={args.preset} | input_mode={input_mode}")

    cfg = make_preset_config(str(args.preset))
    overrides: dict[str, object] = {
        "input_mode": str(input_mode),
        "update_sec": float(args.update_sec),
        "window_sec": float(args.window_sec),
        "live_max_updates": int(args.live_max_updates),
        "realtime_sleep": bool(args.realtime_sleep),
        "print_tick": (not bool(args.no_print_tick)),
    }
    if args.test_duration_sec is not None:
        overrides["test_duration_sec"] = float(args.test_duration_sec)
    if args.test_sampling_hz is not None:
        overrides["test_sampling_hz"] = float(args.test_sampling_hz)
    if args.test_signal_freq_hz is not None:
        overrides["test_signal_freq_hz"] = float(args.test_signal_freq_hz)
    if args.test_signal_amp is not None:
        overrides["test_signal_amp"] = float(args.test_signal_amp)
    if args.test_noise_std is not None:
        overrides["test_noise_std"] = float(args.test_noise_std)
    if args.test_seed is not None:
        overrides["test_seed"] = int(args.test_seed)
    if bool(args.console_events_only):
        overrides["console_event_only"] = True
    if bool(args.full_console_log):
        overrides["console_event_only"] = False
    if str(args.log_file).strip():
        overrides["log_file_path"] = str(args.log_file).strip()
    if bool(args.no_log_file):
        overrides["log_to_file"] = False
    if args.min_interval_sec_for_alert is not None:
        overrides["min_interval_sec_for_alert"] = float(args.min_interval_sec_for_alert)
    if args.risk_cut is not None:
        overrides["risk_cut"] = float(args.risk_cut)
    if bool(args.no_rms_decay_gate):
        overrides["rms_decay_gate_enabled"] = False
    if args.rms_decay_window_sec is not None:
        overrides["rms_decay_window_sec"] = float(args.rms_decay_window_sec)
    if args.rms_decay_on_max is not None:
        overrides["rms_decay_on_max"] = float(args.rms_decay_on_max)
    if args.rms_decay_off_min is not None:
        overrides["rms_decay_off_min"] = float(args.rms_decay_off_min)
    if bool(args.confidence_use_calibration):
        overrides["confidence_use_calibration"] = True
    if args.confidence_w_acf is not None:
        overrides["confidence_w_acf"] = float(args.confidence_w_acf)
    if args.cal_on_support_score_acf_bonus is not None:
        overrides["cal_on_support_score_acf_bonus"] = float(args.cal_on_support_score_acf_bonus)
    if bool(args.no_cal_on_soft_mode):
        overrides["cal_on_soft_mode"] = False
    if bool(args.mp_post):
        overrides["mp_enabled"] = True
    if str(args.mp_policy).strip():
        overrides["mp_runtime_mode"] = str(args.mp_policy).strip().lower()
    if bool(args.mp_post_sync):
        overrides["mp_async_enabled"] = False
    if args.mp_model_order is not None:
        overrides["mp_model_order"] = int(args.mp_model_order)
    if bool(args.mp_order_selection):
        overrides["mp_order_selection_enabled"] = True
    if str(args.mp_order_candidates).strip():
        toks = [x.strip() for x in str(args.mp_order_candidates).split(",") if str(x).strip()]
        if not toks:
            raise ValueError("mp_order_candidates parse failed: empty list")
        overrides["mp_order_candidates"] = tuple(int(x) for x in toks)
    if bool(args.prony_post):
        overrides["prony_enabled"] = True
    if str(args.prony_policy).strip():
        overrides["prony_runtime_mode"] = str(args.prony_policy).strip().lower()
    if bool(args.prony_post_sync):
        overrides["prony_async_enabled"] = False
    if args.prony_model_order is not None:
        overrides["prony_model_order"] = int(args.prony_model_order)
    if str(args.prony_order_candidates).strip():
        toks = [x.strip() for x in str(args.prony_order_candidates).split(",") if str(x).strip()]
        if not toks:
            raise ValueError("prony_order_candidates parse failed: empty list")
        overrides["prony_order_candidates"] = tuple(int(x) for x in toks)
    if bool(args.no_burst):
        overrides["burst_enabled"] = False
    if bool(args.no_burst_policy):
        overrides["burst_policy_enabled"] = False
    if bool(args.operator_alert):
        overrides["operator_alert_enabled"] = True
    if str(args.baseline_file).strip():
        overrides["baseline_file_path"] = str(args.baseline_file).strip()

    if overrides:
        cfg = _apply_detector_overrides(cfg, overrides)

    events = run_streaming_alert_demo_one_channel(
        csv_path,
        cfg=cfg,
        device=str(args.device),
        target_channel=(str(args.channel).strip() if str(args.channel).strip() else None),
    )
    print(f"[MAIN] done | events={len(events)}")


if __name__ == "__main__":
    main()

__all__ = [
    "_validate_detector_runtime_args",
    "_run_streaming_alert_demo_one_channel_cfg_impl",
    "_run_streaming_alert_demo_one_channel_impl",
    "run_streaming_alert_demo_one_channel",
    "_guess_default_stream_csv",
    "_build_arg_parser",
    "main",
]
