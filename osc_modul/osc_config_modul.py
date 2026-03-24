"""Config dataclasses and preset helpers for modular streaming detector."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Sequence

import numpy as np

from .osc_state_modul import (
    MP_RUNTIME_MODE_CHOICES,
    MP_RUNTIME_MODE_LIVE,
    MP_RUNTIME_MODE_REPLAY,
)
from .osc_core_postprep_modul import (
    POSTPREP_METHOD_CHOICES,
    POSTPREP_METHOD_HAMPEL,
    POSTPREP_REPAIR_MODE_CHOICES,
    POSTPREP_REPAIR_MODE_MEDIAN,
)

# Signal scoring constants kept aligned with the monolith defaults.
WIN_SEC = 8.0
DEFAULT_RISK_CUT = 1e-6
DEFAULT_DEVICE = "WMU-SAMPLE"
DEFAULT_UPDATE_SEC = 2.0
DEFAULT_REALTIME_SLEEP = False
DEFAULT_PRINT_TICK = True
STREAM_INPUT_MODE_REPLAY_CSV = "replay_csv"
STREAM_INPUT_MODE_LIVE_CSV_TAIL = "live_csv_tail"
STREAM_INPUT_MODE_TEST_SIGNAL = "test_signal"
STREAM_INPUT_MODE_CHOICES = (
    STREAM_INPUT_MODE_REPLAY_CSV,
    STREAM_INPUT_MODE_LIVE_CSV_TAIL,
    STREAM_INPUT_MODE_TEST_SIGNAL,
)
# Optional default CSV path for local project runs.
# Leave empty to use --csv / STREAM_INPUT_CSV / local auto-discovery.
DEFAULT_STREAM_INPUT_CSV = r""
BASELINE_MISSING_POLICY_WARN_AND_FALLBACK = "warn_and_fallback"
BASELINE_MISSING_POLICY_DISABLE = "disable"
BASELINE_MISSING_POLICY_ERROR = "error"
BASELINE_MISSING_POLICY_CHOICES = (
    BASELINE_MISSING_POLICY_WARN_AND_FALLBACK,
    BASELINE_MISSING_POLICY_DISABLE,
    BASELINE_MISSING_POLICY_ERROR,
)


@dataclass
class StreamConfig:
    """Runtime stream execution controls."""

    update_sec: float = DEFAULT_UPDATE_SEC
    window_sec: float = WIN_SEC
    input_mode: str = STREAM_INPUT_MODE_REPLAY_CSV
    live_max_updates: int = 0
    realtime_sleep: bool = DEFAULT_REALTIME_SLEEP
    print_tick: bool = DEFAULT_PRINT_TICK
    console_event_only: bool = True
    log_to_file: bool = True
    log_file_path: str = "logs/stream_runtime.log"
    min_interval_sec_for_alert: float = 8.0
    stitch_gap_sec: float = 6.0
    test_duration_sec: float = 120.0
    test_sampling_hz: float = 256.0
    test_signal_freq_hz: float = 8.0
    test_signal_amp: float = 0.005
    test_noise_std: float = 0.0005
    test_seed: int = 42

    def validate(self) -> None:
        """Validate stream/runtime controls."""

        mode = str(self.input_mode).strip().lower()
        if mode not in STREAM_INPUT_MODE_CHOICES:
            raise ValueError(f"input_mode must be one of {STREAM_INPUT_MODE_CHOICES!r}")
        if (not np.isfinite(self.update_sec)) or (float(self.update_sec) <= 0.0):
            raise ValueError("update_sec must be positive")
        if (not np.isfinite(self.window_sec)) or (float(self.window_sec) <= 0.0):
            raise ValueError("window_sec must be positive")
        if int(self.live_max_updates) < 0:
            raise ValueError("live_max_updates must be >= 0")
        if bool(self.log_to_file) and (not str(self.log_file_path).strip()):
            raise ValueError("log_file_path must be non-empty when log_to_file is enabled")
        if (not np.isfinite(self.min_interval_sec_for_alert)) or (float(self.min_interval_sec_for_alert) < 0.0):
            raise ValueError("min_interval_sec_for_alert must be >= 0")
        if (not np.isfinite(self.stitch_gap_sec)) or (float(self.stitch_gap_sec) < 0.0):
            raise ValueError("stitch_gap_sec must be >= 0")
        if (not np.isfinite(self.test_duration_sec)) or (float(self.test_duration_sec) <= 0.0):
            raise ValueError("test_duration_sec must be positive")
        if (not np.isfinite(self.test_sampling_hz)) or (float(self.test_sampling_hz) <= 0.0):
            raise ValueError("test_sampling_hz must be positive")
        if (not np.isfinite(self.test_signal_freq_hz)) or (float(self.test_signal_freq_hz) <= 0.0):
            raise ValueError("test_signal_freq_hz must be positive")
        if (not np.isfinite(self.test_signal_amp)) or (float(self.test_signal_amp) < 0.0):
            raise ValueError("test_signal_amp must be >= 0")
        if (not np.isfinite(self.test_noise_std)) or (float(self.test_noise_std) < 0.0):
            raise ValueError("test_noise_std must be >= 0")


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
    rms_decay_event_off_min: float = 0.025
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
    re_on_confirm_min_sec: float = 1.0
    re_on_require_short_trigger: bool = False
    re_on_require_accel: bool = False
    re_on_grace_sec: float = 0.0
    damped_force_off_streak: int = 2
    off_hold_down_sec: float = 4.0
    off_confirm_min_sec: float = 2.0
    off_periodicity_collapse_conf_raw_max: float = 0.45
    off_periodicity_collapse_streak_required: int = 1
    excluded_damped_evidence_penalty: float = 0.15
    excluded_damped_hard_penalty_streak: int = 4
    force_off_require_long_not_on: bool = True
    force_off_long_on_ratio: float | None = None

    def validate(self, *, cut_on: float, cut_off: float) -> None:
        """Validate short-stage/evidence thresholds and derived cutoffs."""

        if (not np.isfinite(cut_off)) or (float(cut_off) <= 0.0):
            raise ValueError("risk_cut_off must be positive")
        if (not np.isfinite(cut_on)) or (float(cut_on) <= 0.0):
            raise ValueError("risk_cut_on must be positive")
        if float(cut_on) < float(cut_off):
            raise ValueError("risk_cut_on must be >= risk_cut_off")
        if int(self.short_dynamic_min_baseline) <= 0:
            raise ValueError("short_dynamic_min_baseline must be >= 1")
        if (not np.isfinite(self.short_dynamic_on_z)) or (float(self.short_dynamic_on_z) < 0.0):
            raise ValueError("short_dynamic_on_z must be >= 0")
        if (not np.isfinite(self.short_dynamic_off_z)) or (float(self.short_dynamic_off_z) < 0.0):
            raise ValueError("short_dynamic_off_z must be >= 0")
        if float(self.short_dynamic_off_z) > float(self.short_dynamic_on_z):
            raise ValueError("short_dynamic_off_z must be <= short_dynamic_on_z")
        if (not np.isfinite(self.short_dynamic_on_max_mult)) or (float(self.short_dynamic_on_max_mult) < 1.0):
            raise ValueError("short_dynamic_on_max_mult must be >= 1")
        if (not np.isfinite(self.short_dynamic_off_max_mult)) or (float(self.short_dynamic_off_max_mult) < 1.0):
            raise ValueError("short_dynamic_off_max_mult must be >= 1")
        if float(self.short_dynamic_off_max_mult) > float(self.short_dynamic_on_max_mult):
            raise ValueError("short_dynamic_off_max_mult must be <= short_dynamic_on_max_mult")
        if (not np.isfinite(self.short_damped_cut_relax)) or (float(self.short_damped_cut_relax) <= 0.0) or (float(self.short_damped_cut_relax) > 1.0):
            raise ValueError("short_damped_cut_relax must be in (0, 1]")
        if (not np.isfinite(self.rms_decay_window_sec)) or (float(self.rms_decay_window_sec) <= 0.0):
            raise ValueError("rms_decay_window_sec must be positive")
        if (not np.isfinite(self.rms_decay_rms_win_sec)) or (float(self.rms_decay_rms_win_sec) <= 0.0):
            raise ValueError("rms_decay_rms_win_sec must be positive")
        if (not np.isfinite(self.rms_decay_step_sec)) or (float(self.rms_decay_step_sec) <= 0.0):
            raise ValueError("rms_decay_step_sec must be positive")
        if int(self.rms_decay_min_windows) <= 0:
            raise ValueError("rms_decay_min_windows must be >= 1")
        if not np.isfinite(self.rms_decay_on_max):
            raise ValueError("rms_decay_on_max must be finite")
        if not np.isfinite(self.rms_decay_off_min):
            raise ValueError("rms_decay_off_min must be finite")
        if float(self.rms_decay_off_min) > float(self.rms_decay_on_max):
            raise ValueError("rms_decay_off_min must be <= rms_decay_on_max")
        if (not np.isfinite(self.rms_decay_cache_refresh_sec)) or (float(self.rms_decay_cache_refresh_sec) <= 0.0):
            raise ValueError("rms_decay_cache_refresh_sec must be positive")
        if (not np.isfinite(self.rms_decay_event_max_window_sec)) or (float(self.rms_decay_event_max_window_sec) <= 0.0):
            raise ValueError("rms_decay_event_max_window_sec must be positive")
        if (not np.isfinite(self.rms_decay_event_min_window_sec)) or (float(self.rms_decay_event_min_window_sec) <= 0.0):
            raise ValueError("rms_decay_event_min_window_sec must be positive")
        if float(self.rms_decay_event_max_window_sec) < float(self.rms_decay_event_min_window_sec):
            raise ValueError("rms_decay_event_max_window_sec must be >= rms_decay_event_min_window_sec")
        if (not np.isfinite(self.rms_decay_event_rms_win_sec)) or (float(self.rms_decay_event_rms_win_sec) <= 0.0):
            raise ValueError("rms_decay_event_rms_win_sec must be positive")
        if (not np.isfinite(self.rms_decay_event_step_sec)) or (float(self.rms_decay_event_step_sec) <= 0.0):
            raise ValueError("rms_decay_event_step_sec must be positive")
        if int(self.rms_decay_event_min_windows) <= 0:
            raise ValueError("rms_decay_event_min_windows must be >= 1")
        if not np.isfinite(self.rms_decay_event_off_min):
            raise ValueError("rms_decay_event_off_min must be finite")
        if float(self.rms_decay_event_off_min) < 0.0:
            raise ValueError("rms_decay_event_off_min must be >= 0")
        if (not np.isfinite(self.evidence_alpha)) or not (0.0 < float(self.evidence_alpha) < 1.0):
            raise ValueError("evidence_alpha must be in (0, 1)")
        if (not np.isfinite(self.evidence_clip)) or (float(self.evidence_clip) <= 0):
            raise ValueError("evidence_clip must be positive")
        if (not np.isfinite(self.theta_on)) or (not np.isfinite(self.theta_off)):
            raise ValueError("theta_on/theta_off must be finite")
        if float(self.theta_off) >= float(self.theta_on):
            raise ValueError("theta_off must be smaller than theta_on")
        if int(self.on_consecutive_required) <= 0:
            raise ValueError("on_consecutive_required must be >= 1")
        if int(self.on_short_votes_window) <= 0:
            raise ValueError("on_short_votes_window must be >= 1")
        if int(self.on_consecutive_required) > int(self.on_short_votes_window):
            raise ValueError("on_consecutive_required must be <= on_short_votes_window")
        if (not np.isfinite(self.on_confirm_min_sec)) or (float(self.on_confirm_min_sec) < 0.0):
            raise ValueError("on_confirm_min_sec must be >= 0")
        if (not np.isfinite(self.re_on_confirm_min_sec)) or (float(self.re_on_confirm_min_sec) < 0.0):
            raise ValueError("re_on_confirm_min_sec must be >= 0")
        if (not np.isfinite(self.re_on_grace_sec)) or (float(self.re_on_grace_sec) < 0.0):
            raise ValueError("re_on_grace_sec must be >= 0")
        if not np.isfinite(self.on_accel_score_log_min):
            raise ValueError("on_accel_score_log_min must be finite")
        if not np.isfinite(self.on_accel_evidence_min):
            raise ValueError("on_accel_evidence_min must be finite")
        if int(self.damped_force_off_streak) <= 0:
            raise ValueError("damped_force_off_streak must be >= 1")
        if (not np.isfinite(self.off_hold_down_sec)) or (float(self.off_hold_down_sec) < 0.0):
            raise ValueError("off_hold_down_sec must be >= 0")
        if (not np.isfinite(self.off_confirm_min_sec)) or (float(self.off_confirm_min_sec) < 0.0):
            raise ValueError("off_confirm_min_sec must be >= 0")
        if (not np.isfinite(self.off_periodicity_collapse_conf_raw_max)) or (float(self.off_periodicity_collapse_conf_raw_max) < 0.0) or (float(self.off_periodicity_collapse_conf_raw_max) > 1.0):
            raise ValueError("off_periodicity_collapse_conf_raw_max must be in [0, 1]")
        if int(self.off_periodicity_collapse_streak_required) <= 0:
            raise ValueError("off_periodicity_collapse_streak_required must be >= 1")
        if (not np.isfinite(self.excluded_damped_evidence_penalty)) or (float(self.excluded_damped_evidence_penalty) < 0.0):
            raise ValueError("excluded_damped_evidence_penalty must be >= 0")
        if int(self.excluded_damped_hard_penalty_streak) <= 0:
            raise ValueError("excluded_damped_hard_penalty_streak must be >= 1")
        if self.force_off_long_on_ratio is not None:
            if (not np.isfinite(self.force_off_long_on_ratio)) or (float(self.force_off_long_on_ratio) < 0.0) or (float(self.force_off_long_on_ratio) > 1.0):
                raise ValueError("force_off_long_on_ratio must be in [0, 1]")


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
    warmup_on_votes_required: int = 2
    warmup_on_votes_window: int = 3
    warmup_on_confirm_min_sec: float = 6.0
    warmup_handoff_grace_ticks: int = 2
    warmup_cancel_on_excluded_damped: bool = True
    post_off_rearm_sec: float = 16.0
    long_z_off: float = 1.0
    long_off_ratio: float = 0.20
    long_off_recent_window_sec: float = 24.0
    long_off_votes_required: int = 2
    long_off_votes_window: int = 4
    long_off_recent_min_points: int = 4
    baseline_include_quiet_on: bool = False
    baseline_quiet_on_max_sec: float = 45.0
    baseline_post_off_cooldown_sec: float = 45.0

    def validate(self) -> None:
        """Validate long-window/warmup settings."""

        if (not np.isfinite(self.long_window_sec)) or (float(self.long_window_sec) <= 0):
            raise ValueError("long_window_sec must be positive")
        if int(self.long_min_points) <= 0:
            raise ValueError("long_min_points must be >= 1")
        if int(self.long_min_baseline) <= 0:
            raise ValueError("long_min_baseline must be >= 1")
        if int(self.long_baseline_max) <= 0:
            raise ValueError("long_baseline_max must be >= 1")
        if (not np.isfinite(self.long_off_recent_window_sec)) or (float(self.long_off_recent_window_sec) <= 0.0):
            raise ValueError("long_off_recent_window_sec must be positive")
        if int(self.long_off_votes_required) <= 0:
            raise ValueError("long_off_votes_required must be >= 1")
        if int(self.long_off_votes_window) <= 0:
            raise ValueError("long_off_votes_window must be >= 1")
        if int(self.long_off_votes_required) > int(self.long_off_votes_window):
            raise ValueError("long_off_votes_required must be <= long_off_votes_window")
        if int(self.long_off_recent_min_points) <= 0:
            raise ValueError("long_off_recent_min_points must be >= 1")
        if (not np.isfinite(self.long_z_on)) or (not np.isfinite(self.long_z_off)):
            raise ValueError("long_z_on/long_z_off must be finite")
        if float(self.long_z_off) > float(self.long_z_on):
            raise ValueError("long_z_off should be <= long_z_on")
        if (not np.isfinite(self.long_on_ratio)) or (float(self.long_on_ratio) <= 0.0) or (float(self.long_on_ratio) > 1.0):
            raise ValueError("long_on_ratio must be in (0, 1]")
        if (not np.isfinite(self.long_off_ratio)) or (float(self.long_off_ratio) < 0.0) or (float(self.long_off_ratio) > 1.0):
            raise ValueError("long_off_ratio must be in [0, 1]")
        if int(self.warmup_on_min_points) <= 0:
            raise ValueError("warmup_on_min_points must be >= 1")
        if int(self.warmup_min_baseline) <= 0:
            raise ValueError("warmup_min_baseline must be >= 1")
        if not np.isfinite(self.warmup_on_z):
            raise ValueError("warmup_on_z must be finite")
        if (not np.isfinite(self.warmup_on_ratio)) or (float(self.warmup_on_ratio) <= 0.0) or (float(self.warmup_on_ratio) > 1.0):
            raise ValueError("warmup_on_ratio must be in (0, 1]")
        if int(self.warmup_on_votes_required) <= 0:
            raise ValueError("warmup_on_votes_required must be >= 1")
        if int(self.warmup_on_votes_window) <= 0:
            raise ValueError("warmup_on_votes_window must be >= 1")
        if int(self.warmup_on_votes_required) > int(self.warmup_on_votes_window):
            raise ValueError("warmup_on_votes_required must be <= warmup_on_votes_window")
        if (not np.isfinite(self.warmup_on_confirm_min_sec)) or (float(self.warmup_on_confirm_min_sec) < 0.0):
            raise ValueError("warmup_on_confirm_min_sec must be >= 0")
        if int(self.warmup_handoff_grace_ticks) < 0:
            raise ValueError("warmup_handoff_grace_ticks must be >= 0")
        if (not np.isfinite(self.post_off_rearm_sec)) or (float(self.post_off_rearm_sec) < 0.0):
            raise ValueError("post_off_rearm_sec must be >= 0")
        if (not np.isfinite(self.baseline_post_off_cooldown_sec)) or (float(self.baseline_post_off_cooldown_sec) < 0.0):
            raise ValueError("baseline_post_off_cooldown_sec must be >= 0")
        if bool(self.baseline_include_quiet_on):
            if (not np.isfinite(self.baseline_quiet_on_max_sec)) or (float(self.baseline_quiet_on_max_sec) <= 0.0):
                raise ValueError("baseline_quiet_on_max_sec must be positive when baseline_include_quiet_on is enabled")


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
    cal_on_support_hold_min: float = 0.45
    cal_on_support_confirm_min: float = 0.56
    cal_on_support_ema_alpha: float = 0.65
    cal_on_confirm_votes_required: int = 2
    cal_on_confirm_votes_window: int = 3
    reentry_support_suppress_enabled: bool = True
    reentry_support_tail_window_sec: float = 120.0
    reentry_support_residual_long_min: float = 0.15
    reentry_support_growth_score_log_min: float = 0.03
    reentry_support_growth_evidence_min: float = 0.02
    reentry_support_growth_long_min: float = 0.01
    reentry_support_long_floor: float = 0.25
    reentry_support_score_floor: float = 0.60
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

    def validate(self) -> None:
        """Validate periodicity/confidence calibration controls."""

        if int(self.acf_min_points) <= 0:
            raise ValueError("acf_min_points must be >= 1")
        if (not np.isfinite(self.acf_min_period_sec)) or (float(self.acf_min_period_sec) <= 0.0):
            raise ValueError("acf_min_period_sec must be positive")
        if (not np.isfinite(self.acf_max_period_sec)) or (float(self.acf_max_period_sec) <= 0.0):
            raise ValueError("acf_max_period_sec must be positive")
        if float(self.acf_max_period_sec) <= float(self.acf_min_period_sec):
            raise ValueError("acf_max_period_sec must be larger than acf_min_period_sec")
        if (not np.isfinite(self.confidence_on_min)) or (float(self.confidence_on_min) < 0.0) or (float(self.confidence_on_min) > 1.0):
            raise ValueError("confidence_on_min must be in [0, 1]")
        if (not np.isfinite(self.confidence_off_max)) or (float(self.confidence_off_max) < 0.0) or (float(self.confidence_off_max) > 1.0):
            raise ValueError("confidence_off_max must be in [0, 1]")
        if float(self.confidence_off_max) > float(self.confidence_on_min):
            raise ValueError("confidence_off_max should be <= confidence_on_min")
        if (not np.isfinite(self.confidence_w_acf)) or (float(self.confidence_w_acf) < 0.0):
            raise ValueError("confidence_w_acf must be >= 0")
        if (not np.isfinite(self.confidence_w_spec)) or (float(self.confidence_w_spec) < 0.0):
            raise ValueError("confidence_w_spec must be >= 0")
        if (not np.isfinite(self.confidence_w_env)) or (float(self.confidence_w_env) < 0.0):
            raise ValueError("confidence_w_env must be >= 0")
        if (not np.isfinite(self.confidence_w_fft)) or (float(self.confidence_w_fft) < 0.0):
            raise ValueError("confidence_w_fft must be >= 0")
        if (float(self.confidence_w_acf) + float(self.confidence_w_spec) + float(self.confidence_w_env) + float(self.confidence_w_fft)) <= 0.0:
            raise ValueError("sum of confidence weights must be > 0")
        if int(self.confidence_cal_min_points) <= 0:
            raise ValueError("confidence_cal_min_points must be >= 1")
        if int(self.confidence_cal_hist_max) <= 0:
            raise ValueError("confidence_cal_hist_max must be >= 1")
        if (not np.isfinite(self.confidence_raw_on_min_when_cal)) or (float(self.confidence_raw_on_min_when_cal) < 0.0) or (float(self.confidence_raw_on_min_when_cal) > 1.0):
            raise ValueError("confidence_raw_on_min_when_cal must be in [0, 1]")
        if (not np.isfinite(self.confidence_raw_off_max_when_cal)) or (float(self.confidence_raw_off_max_when_cal) < 0.0) or (float(self.confidence_raw_off_max_when_cal) > 1.0):
            raise ValueError("confidence_raw_off_max_when_cal must be in [0, 1]")
        if (not np.isfinite(self.cal_on_noise_scale_low)) or (not np.isfinite(self.cal_on_noise_scale_high)):
            raise ValueError("cal_on_noise_scale_low/high must be finite")
        if float(self.cal_on_noise_scale_high) <= float(self.cal_on_noise_scale_low):
            raise ValueError("cal_on_noise_scale_high must be larger than cal_on_noise_scale_low")
        if (not np.isfinite(self.cal_on_conf_adapt_gain)) or (float(self.cal_on_conf_adapt_gain) < 0.0):
            raise ValueError("cal_on_conf_adapt_gain must be >= 0")
        if (not np.isfinite(self.cal_on_conf_floor)) or (float(self.cal_on_conf_floor) < 0.0) or (float(self.cal_on_conf_floor) > 1.0):
            raise ValueError("cal_on_conf_floor must be in [0, 1]")
        if (not np.isfinite(self.cal_on_conf_ceil)) or (float(self.cal_on_conf_ceil) < 0.0) or (float(self.cal_on_conf_ceil) > 1.0):
            raise ValueError("cal_on_conf_ceil must be in [0, 1]")
        if float(self.cal_on_conf_floor) > float(self.cal_on_conf_ceil):
            raise ValueError("cal_on_conf_floor must be <= cal_on_conf_ceil")
        if (not np.isfinite(self.cal_on_support_score_w)) or (float(self.cal_on_support_score_w) < 0.0):
            raise ValueError("cal_on_support_score_w must be >= 0")
        if (not np.isfinite(self.cal_on_support_score_acf_bonus)) or (float(self.cal_on_support_score_acf_bonus) < 0.0) or (float(self.cal_on_support_score_acf_bonus) > 1.0):
            raise ValueError("cal_on_support_score_acf_bonus must be in [0, 1]")
        if (not np.isfinite(self.cal_on_support_long_w)) or (float(self.cal_on_support_long_w) < 0.0):
            raise ValueError("cal_on_support_long_w must be >= 0")
        if (not np.isfinite(self.cal_on_support_conf_w)) or (float(self.cal_on_support_conf_w) < 0.0):
            raise ValueError("cal_on_support_conf_w must be >= 0")
        if (not np.isfinite(self.cal_on_support_accel_w)) or (float(self.cal_on_support_accel_w) < 0.0):
            raise ValueError("cal_on_support_accel_w must be >= 0")
        if (float(self.cal_on_support_score_w) + float(self.cal_on_support_long_w) + float(self.cal_on_support_conf_w) + float(self.cal_on_support_accel_w)) <= 0.0:
            raise ValueError("sum of CAL_ON support weights must be > 0")
        if (not np.isfinite(self.cal_on_support_enter_min)) or (float(self.cal_on_support_enter_min) < 0.0) or (float(self.cal_on_support_enter_min) > 1.0):
            raise ValueError("cal_on_support_enter_min must be in [0, 1]")
        if (not np.isfinite(self.cal_on_support_confirm_min)) or (float(self.cal_on_support_confirm_min) < 0.0) or (float(self.cal_on_support_confirm_min) > 1.0):
            raise ValueError("cal_on_support_confirm_min must be in [0, 1]")
        if (not np.isfinite(self.cal_on_support_hold_min)) or (float(self.cal_on_support_hold_min) < 0.0) or (float(self.cal_on_support_hold_min) > 1.0):
            raise ValueError("cal_on_support_hold_min must be in [0, 1]")
        if float(self.cal_on_support_confirm_min) < float(self.cal_on_support_enter_min):
            raise ValueError("cal_on_support_confirm_min should be >= cal_on_support_enter_min")
        if (not np.isfinite(self.cal_on_support_ema_alpha)) or (float(self.cal_on_support_ema_alpha) < 0.0) or (float(self.cal_on_support_ema_alpha) >= 1.0):
            raise ValueError("cal_on_support_ema_alpha must be in [0, 1)")
        if int(self.cal_on_confirm_votes_required) <= 0:
            raise ValueError("cal_on_confirm_votes_required must be >= 1")
        if int(self.cal_on_confirm_votes_window) <= 0:
            raise ValueError("cal_on_confirm_votes_window must be >= 1")
        if int(self.cal_on_confirm_votes_required) > int(self.cal_on_confirm_votes_window):
            raise ValueError("cal_on_confirm_votes_required must be <= cal_on_confirm_votes_window")
        if (not np.isfinite(self.reentry_support_tail_window_sec)) or (float(self.reentry_support_tail_window_sec) < 0.0):
            raise ValueError("reentry_support_tail_window_sec must be >= 0")
        if (not np.isfinite(self.reentry_support_residual_long_min)) or (float(self.reentry_support_residual_long_min) < 0.0) or (float(self.reentry_support_residual_long_min) > 1.0):
            raise ValueError("reentry_support_residual_long_min must be in [0, 1]")
        if (not np.isfinite(self.reentry_support_growth_score_log_min)) or (float(self.reentry_support_growth_score_log_min) < 0.0):
            raise ValueError("reentry_support_growth_score_log_min must be >= 0")
        if (not np.isfinite(self.reentry_support_growth_evidence_min)) or (float(self.reentry_support_growth_evidence_min) < 0.0):
            raise ValueError("reentry_support_growth_evidence_min must be >= 0")
        if (not np.isfinite(self.reentry_support_growth_long_min)) or (float(self.reentry_support_growth_long_min) < 0.0):
            raise ValueError("reentry_support_growth_long_min must be >= 0")
        if (not np.isfinite(self.reentry_support_long_floor)) or (float(self.reentry_support_long_floor) < 0.0) or (float(self.reentry_support_long_floor) > 1.0):
            raise ValueError("reentry_support_long_floor must be in [0, 1]")
        if (not np.isfinite(self.reentry_support_score_floor)) or (float(self.reentry_support_score_floor) < 0.0) or (float(self.reentry_support_score_floor) > 1.0):
            raise ValueError("reentry_support_score_floor must be in [0, 1]")
        if float(self.reentry_support_score_floor) < float(self.reentry_support_long_floor):
            raise ValueError("reentry_support_score_floor should be >= reentry_support_long_floor")
        if (not np.isfinite(self.freq_band_low_hz)) or (float(self.freq_band_low_hz) <= 0.0):
            raise ValueError("freq_band_low_hz must be positive")
        if (not np.isfinite(self.freq_band_high_hz)) or (float(self.freq_band_high_hz) <= 0.0):
            raise ValueError("freq_band_high_hz must be positive")
        if float(self.freq_band_high_hz) <= float(self.freq_band_low_hz):
            raise ValueError("freq_band_high_hz must be larger than freq_band_low_hz")
        if (not np.isfinite(self.periodicity_quality_cache_refresh_sec)) or (float(self.periodicity_quality_cache_refresh_sec) <= 0.0):
            raise ValueError("periodicity_quality_cache_refresh_sec must be positive")


@dataclass
class ModalPreprocessConfig:
    """Shared post-analysis preprocess for MP/Prony selected windows."""

    enabled: bool = True
    method: str = POSTPREP_METHOD_HAMPEL
    hampel_half_window: int = 4
    hampel_nsigma: float = 8.0
    max_repair_points: int = 24
    max_repair_fraction: float = 0.01
    repair_mode: str = POSTPREP_REPAIR_MODE_MEDIAN
    keep_raw_summary: bool = True

    def validate(self) -> None:
        """Validate conservative modal post preprocess settings."""

        method = str(self.method).strip().lower()
        if method not in POSTPREP_METHOD_CHOICES:
            raise ValueError(f"method must be one of {POSTPREP_METHOD_CHOICES!r}")
        if int(self.hampel_half_window) < 1:
            raise ValueError("hampel_half_window must be >= 1")
        if (not np.isfinite(self.hampel_nsigma)) or (float(self.hampel_nsigma) <= 0.0):
            raise ValueError("hampel_nsigma must be positive")
        if int(self.max_repair_points) < 0:
            raise ValueError("max_repair_points must be >= 0")
        if (not np.isfinite(self.max_repair_fraction)) or (float(self.max_repair_fraction) < 0.0) or (float(self.max_repair_fraction) > 1.0):
            raise ValueError("max_repair_fraction must be in [0, 1]")
        repair_mode = str(self.repair_mode).strip().lower()
        if repair_mode not in POSTPREP_REPAIR_MODE_CHOICES:
            raise ValueError(f"repair_mode must be one of {POSTPREP_REPAIR_MODE_CHOICES!r}")


@dataclass
class MPPostAnalysisConfig:
    """Post-interval Matrix Pencil analysis settings."""

    mp_enabled: bool = False
    mp_runtime_mode: str = MP_RUNTIME_MODE_LIVE
    mp_min_interval_sec: float = 8.0
    mp_onset_skip_sec: float = 1.5
    mp_onset_window_sec: float = 12.0
    mp_onset_min_window_sec: float = 6.0
    mp_fallback_use_rms_event_window: bool = True
    mp_fallback_default_window_sec: float = 12.0
    mp_model_order: int = 8
    mp_order_selection_enabled: bool = False
    mp_order_candidates: tuple[int, ...] | None = (4, 6, 8, 10)
    mp_max_modes: int = 3
    mp_freq_low_hz: float = 0.15
    mp_freq_high_hz: float = 20.0
    mp_min_samples: int = 64
    mp_dt_cv_max: float = 0.15
    mp_signal_std_min: float = 1e-5
    mp_singular_ratio_min: float = 1e-6
    mp_downsample_enabled: bool = False
    mp_target_fs_hz: float = 80.0
    mp_downsample_lpf_cutoff_hz: float = 26.0
    mp_downsample_lpf_order: int = 4
    mp_async_enabled: bool = True
    mp_queue_maxsize: int = 64
    mp_finalize_wait_sec: float = 2.0

    def validate(self) -> None:
        """Validate MP runtime mode selection."""

        mode = str(self.mp_runtime_mode).strip().lower()
        if mode not in MP_RUNTIME_MODE_CHOICES:
            raise ValueError(f"mp_runtime_mode must be one of {MP_RUNTIME_MODE_CHOICES!r}")
        if int(self.mp_model_order) < 1:
            raise ValueError("mp_model_order must be >= 1")
        if self.mp_order_candidates is not None:
            if isinstance(self.mp_order_candidates, (str, bytes)):
                raise ValueError("mp_order_candidates must be a sequence of integers")
            vals = [int(x) for x in self.mp_order_candidates]
            if not vals:
                raise ValueError("mp_order_candidates must not be empty when provided")
            if any(int(v) < 1 for v in vals):
                raise ValueError("all mp_order_candidates must be >= 1")
        if (not np.isfinite(self.mp_target_fs_hz)) or (float(self.mp_target_fs_hz) <= 0.0):
            raise ValueError("mp_target_fs_hz must be positive")
        if (not np.isfinite(self.mp_downsample_lpf_cutoff_hz)) or (float(self.mp_downsample_lpf_cutoff_hz) <= 0.0):
            raise ValueError("mp_downsample_lpf_cutoff_hz must be positive")
        if int(self.mp_downsample_lpf_order) < 1:
            raise ValueError("mp_downsample_lpf_order must be >= 1")
        if bool(self.mp_downsample_enabled):
            nyq_target = 0.5 * float(self.mp_target_fs_hz)
            if float(self.mp_downsample_lpf_cutoff_hz) >= nyq_target:
                raise ValueError("mp_downsample_lpf_cutoff_hz must be < 0.5 * mp_target_fs_hz when downsampling is enabled")


@dataclass
class PronyPostAnalysisConfig:
    """Post-interval Prony analysis settings."""

    prony_enabled: bool = False
    prony_runtime_mode: str = MP_RUNTIME_MODE_LIVE
    prony_min_interval_sec: float = 8.0
    prony_onset_skip_sec: float = 1.5
    prony_onset_window_sec: float = 12.0
    prony_onset_min_window_sec: float = 6.0
    prony_fallback_use_rms_event_window: bool = True
    prony_fallback_default_window_sec: float = 12.0
    prony_model_order: int = 8
    prony_order_candidates: tuple[int, ...] | None = (24, 16, 12, 8, 6, 4, 2)
    prony_max_modes: int = 3
    prony_freq_low_hz: float = 0.15
    prony_freq_high_hz: float = 20.0
    prony_min_samples: int = 64
    prony_dt_cv_max: float = 0.15
    prony_signal_std_min: float = 1e-5
    prony_root_mag_max: float = 1.2
    prony_async_enabled: bool = True
    prony_queue_maxsize: int = 64
    prony_finalize_wait_sec: float = 2.0

    def validate(self) -> None:
        """Validate Prony runtime mode selection."""

        mode = str(self.prony_runtime_mode).strip().lower()
        if mode not in MP_RUNTIME_MODE_CHOICES:
            raise ValueError(f"prony_runtime_mode must be one of {MP_RUNTIME_MODE_CHOICES!r}")
        if int(self.prony_model_order) < 2:
            raise ValueError("prony_model_order must be >= 2")
        if self.prony_order_candidates is not None:
            if isinstance(self.prony_order_candidates, (str, bytes)):
                raise ValueError("prony_order_candidates must be a sequence of integers")
            vals = [int(x) for x in self.prony_order_candidates]
            if not vals:
                raise ValueError("prony_order_candidates must not be empty when provided")
            if any(int(v) < 2 for v in vals):
                raise ValueError("all prony_order_candidates must be >= 2")


@dataclass
class BurstConfig:
    """High-frequency short-burst detector controls (sidecar to main FSM)."""

    burst_enabled: bool = True
    burst_freq_low_hz: float = 5.0
    burst_freq_high_hz: float = 20.0
    burst_freq_match_tol_hz: float = 2.0
    burst_freq_stability_tol_hz: float = 3.0
    burst_recent_window_ticks: int = 6
    burst_recent_required_ticks: int = 4
    burst_confirm_min_sec: float = 8.0
    burst_hold_gap_sec: float = 4.0
    burst_merge_gap_sec: float = 6.0
    burst_entry_support_min: float = 0.55
    burst_hold_support_min: float = 0.42
    burst_entry_conf_min: float = 0.45
    burst_hold_conf_min: float = 0.35
    burst_conf_w_spec: float = 0.45
    burst_conf_w_fft: float = 0.35
    burst_conf_w_env: float = 0.20
    burst_require_accel_on_entry: bool = True
    burst_require_short_trigger: bool = True
    burst_require_reason_ok: bool = True

    def validate(self) -> None:
        """Validate burst detector thresholds and temporal guards."""

        if (not np.isfinite(self.burst_freq_low_hz)) or (float(self.burst_freq_low_hz) <= 0.0):
            raise ValueError("burst_freq_low_hz must be positive")
        if (not np.isfinite(self.burst_freq_high_hz)) or (float(self.burst_freq_high_hz) <= 0.0):
            raise ValueError("burst_freq_high_hz must be positive")
        if float(self.burst_freq_high_hz) <= float(self.burst_freq_low_hz):
            raise ValueError("burst_freq_high_hz must be larger than burst_freq_low_hz")
        if (not np.isfinite(self.burst_freq_match_tol_hz)) or (float(self.burst_freq_match_tol_hz) < 0.0):
            raise ValueError("burst_freq_match_tol_hz must be >= 0")
        if (not np.isfinite(self.burst_freq_stability_tol_hz)) or (float(self.burst_freq_stability_tol_hz) < 0.0):
            raise ValueError("burst_freq_stability_tol_hz must be >= 0")
        if int(self.burst_recent_window_ticks) <= 0:
            raise ValueError("burst_recent_window_ticks must be >= 1")
        if int(self.burst_recent_required_ticks) <= 0:
            raise ValueError("burst_recent_required_ticks must be >= 1")
        if int(self.burst_recent_required_ticks) > int(self.burst_recent_window_ticks):
            raise ValueError("burst_recent_required_ticks must be <= burst_recent_window_ticks")
        if (not np.isfinite(self.burst_confirm_min_sec)) or (float(self.burst_confirm_min_sec) < 0.0):
            raise ValueError("burst_confirm_min_sec must be >= 0")
        if (not np.isfinite(self.burst_hold_gap_sec)) or (float(self.burst_hold_gap_sec) < 0.0):
            raise ValueError("burst_hold_gap_sec must be >= 0")
        if (not np.isfinite(self.burst_merge_gap_sec)) or (float(self.burst_merge_gap_sec) < 0.0):
            raise ValueError("burst_merge_gap_sec must be >= 0")
        if (not np.isfinite(self.burst_entry_support_min)) or (float(self.burst_entry_support_min) < 0.0) or (float(self.burst_entry_support_min) > 1.0):
            raise ValueError("burst_entry_support_min must be in [0, 1]")
        if (not np.isfinite(self.burst_hold_support_min)) or (float(self.burst_hold_support_min) < 0.0) or (float(self.burst_hold_support_min) > 1.0):
            raise ValueError("burst_hold_support_min must be in [0, 1]")
        if float(self.burst_hold_support_min) > float(self.burst_entry_support_min):
            raise ValueError("burst_hold_support_min must be <= burst_entry_support_min")
        if (not np.isfinite(self.burst_entry_conf_min)) or (float(self.burst_entry_conf_min) < 0.0) or (float(self.burst_entry_conf_min) > 1.0):
            raise ValueError("burst_entry_conf_min must be in [0, 1]")
        if (not np.isfinite(self.burst_hold_conf_min)) or (float(self.burst_hold_conf_min) < 0.0) or (float(self.burst_hold_conf_min) > 1.0):
            raise ValueError("burst_hold_conf_min must be in [0, 1]")
        if float(self.burst_hold_conf_min) > float(self.burst_entry_conf_min):
            raise ValueError("burst_hold_conf_min must be <= burst_entry_conf_min")
        if (not np.isfinite(self.burst_conf_w_spec)) or (float(self.burst_conf_w_spec) < 0.0):
            raise ValueError("burst_conf_w_spec must be >= 0")
        if (not np.isfinite(self.burst_conf_w_fft)) or (float(self.burst_conf_w_fft) < 0.0):
            raise ValueError("burst_conf_w_fft must be >= 0")
        if (not np.isfinite(self.burst_conf_w_env)) or (float(self.burst_conf_w_env) < 0.0):
            raise ValueError("burst_conf_w_env must be >= 0")
        if (float(self.burst_conf_w_spec) + float(self.burst_conf_w_fft) + float(self.burst_conf_w_env)) <= 0.0:
            raise ValueError("sum of burst_conf_w_* must be > 0")


@dataclass
class BurstPolicyConfig:
    """Operator-facing burst policy layer on top of burst_interval_final events."""

    burst_policy_enabled: bool = True
    burst_emit_advisory: bool = True
    burst_emit_investigate: bool = True
    burst_repeat_window_sec: float = 120.0
    burst_investigate_repeat_count: int = 2
    burst_policy_freq_match_tol_hz: float = 1.0
    burst_multi_channel_window_sec: float = 15.0
    burst_multi_channel_min_count: int = 2
    burst_suppress_covered_by_sustained: bool = True
    burst_suppress_window_sec: float = 15.0
    burst_suppress_freq_match_tol_hz: float = 1.0

    def validate(self) -> None:
        """Validate burst policy controls."""

        if (not np.isfinite(self.burst_repeat_window_sec)) or (float(self.burst_repeat_window_sec) < 0.0):
            raise ValueError("burst_repeat_window_sec must be >= 0")
        if int(self.burst_investigate_repeat_count) <= 0:
            raise ValueError("burst_investigate_repeat_count must be >= 1")
        if (not np.isfinite(self.burst_policy_freq_match_tol_hz)) or (float(self.burst_policy_freq_match_tol_hz) < 0.0):
            raise ValueError("burst_policy_freq_match_tol_hz must be >= 0")
        if (not np.isfinite(self.burst_multi_channel_window_sec)) or (float(self.burst_multi_channel_window_sec) < 0.0):
            raise ValueError("burst_multi_channel_window_sec must be >= 0")
        if int(self.burst_multi_channel_min_count) <= 0:
            raise ValueError("burst_multi_channel_min_count must be >= 1")
        if (not np.isfinite(self.burst_suppress_window_sec)) or (float(self.burst_suppress_window_sec) < 0.0):
            raise ValueError("burst_suppress_window_sec must be >= 0")
        if (not np.isfinite(self.burst_suppress_freq_match_tol_hz)) or (float(self.burst_suppress_freq_match_tol_hz) < 0.0):
            raise ValueError("burst_suppress_freq_match_tol_hz must be >= 0")


@dataclass
class AlertBandConfig:
    """Band-level operator alert thresholds and persistence settings."""

    band_name: str = "inter_area"
    freq_low_hz: float = 0.15
    freq_high_hz: float = 1.0
    persist_sec_level1: float = 60.0
    persist_sec_level2: float = 120.0
    energy_level1_z: float = 3.0
    energy_level2_z: float = 4.0
    use_modal_post: bool = True
    enable_damping_gate: bool = True
    damping_ratio_investigate_max: float = 0.05
    damping_ratio_corrective_max: float = 0.03

    def validate(self) -> None:
        """Validate operator-alert thresholds for one frequency band."""

        name = str(self.band_name).strip()
        if not name:
            raise ValueError("band_name must be non-empty")
        if (not np.isfinite(self.freq_low_hz)) or (float(self.freq_low_hz) <= 0.0):
            raise ValueError("freq_low_hz must be positive")
        if (not np.isfinite(self.freq_high_hz)) or (float(self.freq_high_hz) <= 0.0):
            raise ValueError("freq_high_hz must be positive")
        if float(self.freq_high_hz) <= float(self.freq_low_hz):
            raise ValueError("freq_high_hz must be larger than freq_low_hz")
        if (not np.isfinite(self.persist_sec_level1)) or (float(self.persist_sec_level1) < 0.0):
            raise ValueError("persist_sec_level1 must be >= 0")
        if (not np.isfinite(self.persist_sec_level2)) or (float(self.persist_sec_level2) < 0.0):
            raise ValueError("persist_sec_level2 must be >= 0")
        if float(self.persist_sec_level2) < float(self.persist_sec_level1):
            raise ValueError("persist_sec_level2 must be >= persist_sec_level1")
        if (not np.isfinite(self.energy_level1_z)) or (float(self.energy_level1_z) < 0.0):
            raise ValueError("energy_level1_z must be >= 0")
        if (not np.isfinite(self.energy_level2_z)) or (float(self.energy_level2_z) < 0.0):
            raise ValueError("energy_level2_z must be >= 0")
        if float(self.energy_level2_z) < float(self.energy_level1_z):
            raise ValueError("energy_level2_z must be >= energy_level1_z")
        if (not np.isfinite(self.damping_ratio_investigate_max)) or (float(self.damping_ratio_investigate_max) < 0.0) or (float(self.damping_ratio_investigate_max) > 1.0):
            raise ValueError("damping_ratio_investigate_max must be in [0, 1]")
        if (not np.isfinite(self.damping_ratio_corrective_max)) or (float(self.damping_ratio_corrective_max) < 0.0) or (float(self.damping_ratio_corrective_max) > 1.0):
            raise ValueError("damping_ratio_corrective_max must be in [0, 1]")
        if float(self.damping_ratio_corrective_max) > float(self.damping_ratio_investigate_max):
            raise ValueError("damping_ratio_corrective_max must be <= damping_ratio_investigate_max")


def default_alert_band_profiles() -> tuple[AlertBandConfig, ...]:
    """Default NERC-style broad-band split used by operator-alert policy."""

    return (
        AlertBandConfig(
            band_name="very_low",
            freq_low_hz=0.01,
            freq_high_hz=0.15,
            persist_sec_level1=60.0,
            persist_sec_level2=120.0,
        ),
        AlertBandConfig(
            band_name="inter_area",
            freq_low_hz=0.15,
            freq_high_hz=1.0,
            persist_sec_level1=60.0,
            persist_sec_level2=120.0,
        ),
        AlertBandConfig(
            band_name="local_control",
            freq_low_hz=1.0,
            freq_high_hz=5.0,
            persist_sec_level1=45.0,
            persist_sec_level2=90.0,
        ),
        AlertBandConfig(
            band_name="high_freq",
            freq_low_hz=5.0,
            freq_high_hz=20.0,
            persist_sec_level1=30.0,
            persist_sec_level2=60.0,
        ),
    )


@dataclass
class AlertPolicyConfig:
    """Operator-facing alert policy layer (separate from detector FSM)."""

    operator_alert_enabled: bool = True
    require_multi_channel: bool = False
    multi_channel_min_count: int = 2
    multi_channel_window_sec: float = 60.0
    dominant_freq_match_tol_hz: float = 0.05
    emit_advisory: bool = True
    emit_investigate: bool = True
    emit_corrective: bool = True
    review_required_on_low_damping: bool = True
    modal_min_fit_r2: float = 0.75
    modal_min_mode_count: int = 1
    modal_min_signal_std: float = 0.0

    def validate(self) -> None:
        """Validate operator-policy controls and reliability gates."""

        if int(self.multi_channel_min_count) <= 0:
            raise ValueError("multi_channel_min_count must be >= 1")
        if (not np.isfinite(self.multi_channel_window_sec)) or (float(self.multi_channel_window_sec) < 0.0):
            raise ValueError("multi_channel_window_sec must be >= 0")
        if (not np.isfinite(self.dominant_freq_match_tol_hz)) or (float(self.dominant_freq_match_tol_hz) < 0.0):
            raise ValueError("dominant_freq_match_tol_hz must be >= 0")
        if (not np.isfinite(self.modal_min_fit_r2)) or (float(self.modal_min_fit_r2) < 0.0) or (float(self.modal_min_fit_r2) > 1.0):
            raise ValueError("modal_min_fit_r2 must be in [0, 1]")
        if int(self.modal_min_mode_count) <= 0:
            raise ValueError("modal_min_mode_count must be >= 1")
        if (not np.isfinite(self.modal_min_signal_std)) or (float(self.modal_min_signal_std) < 0.0):
            raise ValueError("modal_min_signal_std must be >= 0")


@dataclass
class BaselineConfig:
    """External baseline source controls for operator-alert energy thresholds."""

    baseline_enabled: bool = True
    baseline_file_path: str = ""
    baseline_missing_policy: str = BASELINE_MISSING_POLICY_WARN_AND_FALLBACK
    baseline_fallback_mean_energy: float = 0.0
    baseline_fallback_std_energy: float = 1.0
    baseline_fallback_std_epsilon: float = 1e-9

    def validate(self) -> None:
        """Validate baseline loading/fallback options."""

        policy = str(self.baseline_missing_policy).strip().lower()
        if policy not in BASELINE_MISSING_POLICY_CHOICES:
            raise ValueError(f"baseline_missing_policy must be one of {BASELINE_MISSING_POLICY_CHOICES!r}")
        if not np.isfinite(self.baseline_fallback_mean_energy):
            raise ValueError("baseline_fallback_mean_energy must be finite")
        if (not np.isfinite(self.baseline_fallback_std_energy)) or (float(self.baseline_fallback_std_energy) < 0.0):
            raise ValueError("baseline_fallback_std_energy must be >= 0")
        if (not np.isfinite(self.baseline_fallback_std_epsilon)) or (float(self.baseline_fallback_std_epsilon) <= 0.0):
            raise ValueError("baseline_fallback_std_epsilon must be positive")


@dataclass
class DetectorConfig:
    """Top-level grouped configuration for the streaming detector."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    long: LongConfig = field(default_factory=LongConfig)
    periodicity: PeriodicityConfig = field(default_factory=PeriodicityConfig)
    modal_preprocess: ModalPreprocessConfig = field(default_factory=ModalPreprocessConfig)
    mp_post: MPPostAnalysisConfig = field(default_factory=MPPostAnalysisConfig)
    prony_post: PronyPostAnalysisConfig = field(default_factory=PronyPostAnalysisConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)
    burst_policy: BurstPolicyConfig = field(default_factory=BurstPolicyConfig)
    alert_policy: AlertPolicyConfig = field(default_factory=AlertPolicyConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    alert_bands: tuple[AlertBandConfig, ...] = field(default_factory=default_alert_band_profiles)

    def validate(self, *, cut_on: float, cut_off: float) -> None:
        """Validate all section configs with derived runtime cuts."""

        self.stream.validate()
        self.threshold.validate(cut_on=float(cut_on), cut_off=float(cut_off))
        self.long.validate()
        self.periodicity.validate()
        self.modal_preprocess.validate()
        self.mp_post.validate()
        self.prony_post.validate()
        self.burst.validate()
        self.burst_policy.validate()
        self.alert_policy.validate()
        self.baseline.validate()
        if not isinstance(self.alert_bands, Sequence) or isinstance(self.alert_bands, (str, bytes)):
            raise ValueError("alert_bands must be a sequence of AlertBandConfig")
        if len(self.alert_bands) <= 0:
            raise ValueError("alert_bands must include at least one band profile")
        for band in self.alert_bands:
            if not isinstance(band, AlertBandConfig):
                raise ValueError("alert_bands items must be AlertBandConfig")
            band.validate()


def _detector_config_override_target_map() -> dict[str, tuple[str, str]]:
    """Map flat override names to `(section, field)` in `DetectorConfig`."""

    cfg_obj = DetectorConfig()
    sections = {
        "stream": cfg_obj.stream,
        "threshold": cfg_obj.threshold,
        "long": cfg_obj.long,
        "periodicity": cfg_obj.periodicity,
        "modal_preprocess": cfg_obj.modal_preprocess,
        "mp_post": cfg_obj.mp_post,
        "prony_post": cfg_obj.prony_post,
        "burst": cfg_obj.burst,
        "burst_policy": cfg_obj.burst_policy,
        "alert_policy": cfg_obj.alert_policy,
        "baseline": cfg_obj.baseline,
    }
    out: dict[str, tuple[str, str]] = {}
    for sec_name, sec_obj in sections.items():
        for f in fields(sec_obj):
            if f.name in out:
                raise RuntimeError(f"duplicate config field name: {f.name}")
            out[f.name] = (sec_name, f.name)
    return out


_CFG_OVERRIDE_TARGETS = _detector_config_override_target_map()


def _apply_detector_overrides(cfg_obj: DetectorConfig, overrides: dict[str, object]) -> DetectorConfig:
    """Apply flat keyword overrides into a deep-copied detector config."""

    out = copy.deepcopy(cfg_obj)
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


def _resolve_mp_runtime_controls(
    *,
    mp_runtime_mode: str,
    mp_async_enabled: bool,
    mp_queue_maxsize: int,
    mp_finalize_wait_sec: float,
) -> tuple[str, bool, int, float]:
    """Resolve MP runtime controls by execution mode (live vs replay)."""

    mode = str(mp_runtime_mode).strip().lower()
    if mode not in MP_RUNTIME_MODE_CHOICES:
        raise ValueError(f"mp_runtime_mode must be one of {MP_RUNTIME_MODE_CHOICES!r}")

    async_eff = bool(mp_async_enabled)
    queue_eff = int(max(1, mp_queue_maxsize))
    wait_eff = float(max(0.0, mp_finalize_wait_sec))

    if mode == MP_RUNTIME_MODE_REPLAY:
        async_eff = False
        queue_eff = int(max(256, queue_eff))
        wait_eff = float(max(20.0, wait_eff))

    return str(mode), bool(async_eff), int(queue_eff), float(wait_eff)


def _resolve_prony_runtime_controls(
    *,
    prony_runtime_mode: str,
    prony_async_enabled: bool,
    prony_queue_maxsize: int,
    prony_finalize_wait_sec: float,
) -> tuple[str, bool, int, float]:
    """Resolve Prony runtime controls by execution mode (live vs replay)."""

    return _resolve_mp_runtime_controls(
        mp_runtime_mode=str(prony_runtime_mode),
        mp_async_enabled=bool(prony_async_enabled),
        mp_queue_maxsize=int(prony_queue_maxsize),
        mp_finalize_wait_sec=float(prony_finalize_wait_sec),
    )


PRESET_SAFE = "safe"
PRESET_BALANCED = "balanced"
PRESET_SENSITIVE = "sensitive"
PRESET_CHOICES = (PRESET_SAFE, PRESET_BALANCED, PRESET_SENSITIVE)


def preset_safe() -> DetectorConfig:
    """Return conservative production-safe defaults."""

    return DetectorConfig()


def preset_balanced() -> DetectorConfig:
    """Return medium-sensitivity defaults with calibrated confidence."""

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
    """Build detector config from preset name."""

    n = str(name).strip().lower()
    if n == PRESET_SAFE:
        return preset_safe()
    if n == PRESET_BALANCED:
        return preset_balanced()
    if n == PRESET_SENSITIVE:
        return preset_sensitive()
    raise ValueError(f"Unknown preset: {name}")

__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_UPDATE_SEC",
    "DEFAULT_REALTIME_SLEEP",
    "DEFAULT_PRINT_TICK",
    "STREAM_INPUT_MODE_REPLAY_CSV",
    "STREAM_INPUT_MODE_LIVE_CSV_TAIL",
    "STREAM_INPUT_MODE_TEST_SIGNAL",
    "STREAM_INPUT_MODE_CHOICES",
    "DEFAULT_RISK_CUT",
    "BASELINE_MISSING_POLICY_WARN_AND_FALLBACK",
    "BASELINE_MISSING_POLICY_DISABLE",
    "BASELINE_MISSING_POLICY_ERROR",
    "BASELINE_MISSING_POLICY_CHOICES",
    "PRESET_SAFE",
    "PRESET_BALANCED",
    "PRESET_SENSITIVE",
    "PRESET_CHOICES",
    "StreamConfig",
    "ThresholdConfig",
    "LongConfig",
    "PeriodicityConfig",
    "ModalPreprocessConfig",
    "MPPostAnalysisConfig",
    "PronyPostAnalysisConfig",
    "BurstConfig",
    "BurstPolicyConfig",
    "AlertBandConfig",
    "AlertPolicyConfig",
    "BaselineConfig",
    "default_alert_band_profiles",
    "DetectorConfig",
    "_detector_config_override_target_map",
    "_apply_detector_overrides",
    "_resolve_mp_runtime_controls",
    "_resolve_prony_runtime_controls",
    "preset_safe",
    "preset_balanced",
    "preset_sensitive",
    "make_preset_config",
]
