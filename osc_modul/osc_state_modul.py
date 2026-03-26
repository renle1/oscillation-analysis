"""State/phase types for modular streaming detector.

Canonical API contract for new code:
- Channel state access: ``st.signal`` / ``st.votes`` / ``st.cache``
- Tick feature access: ``tick.signal`` / ``tick.quality`` / ``tick.vote``
- Tick-quality reads: semantic alias names (``gate_*`` / ``feature_*`` / ``state_*``)

Legacy flat forwarding and legacy storage names remain for compatibility only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

# Finite-state machine phases.
PHASE_OFF = "OFF"
PHASE_ON_CANDIDATE = "ON_CANDIDATE"
PHASE_ON_CONFIRMED = "ON_CONFIRMED"
PHASE_OFF_CANDIDATE = "OFF_CANDIDATE"
PHASE_OFF_CONFIRMED = "OFF_CONFIRMED"
BURST_PHASE_OFF = "BURST_OFF"
BURST_PHASE_CANDIDATE = "BURST_CANDIDATE"
BURST_PHASE_ACTIVE = "BURST_ACTIVE"

MP_RUNTIME_MODE_LIVE = "live"
MP_RUNTIME_MODE_REPLAY = "replay"
MP_RUNTIME_MODE_CHOICES = (MP_RUNTIME_MODE_LIVE, MP_RUNTIME_MODE_REPLAY)

SampleTuple = tuple[str, str, float, float]
UpdateBatch = list[SampleTuple]
UpdateBatchList = list[UpdateBatch]


class VoteWindow(deque[int]):
    """Deque that keeps an incremental sum for O(1) vote-window totals."""

    def __init__(self, iterable=()) -> None:
        vals = [int(x) for x in iterable]
        super().__init__(vals)
        self.sum = int(sum(vals))

    def append(self, x: int) -> None:  # type: ignore[override]
        v = int(x)
        self.sum += v
        super().append(v)

    def appendleft(self, x: int) -> None:  # type: ignore[override]
        v = int(x)
        self.sum += v
        super().appendleft(v)

    def pop(self) -> int:  # type: ignore[override]
        v = int(super().pop())
        self.sum -= v
        return v

    def popleft(self) -> int:  # type: ignore[override]
        v = int(super().popleft())
        self.sum -= v
        return v

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self.sum = 0

    def extend(self, iterable) -> None:  # type: ignore[override]
        for x in iterable:
            self.append(int(x))

    def extendleft(self, iterable) -> None:  # type: ignore[override]
        for x in iterable:
            self.appendleft(int(x))


@dataclass
class QualityCacheSnapshot:
    """Typed cache for periodicity quality + RMS support metrics."""

    confidence: float = float("nan")
    confidence_raw: float = float("nan")
    confidence_cal: float = float("nan")
    confidence_used: float = float("nan")
    c_acf: float = float("nan")
    c_spec: float = float("nan")
    c_env: float = float("nan")
    c_fft: float = float("nan")
    c_freq_agree: float = float("nan")
    f_welch: float = float("nan")
    f_zc: float = float("nan")
    f_fft: float = float("nan")
    acf_peak: float = float("nan")
    acf_period_sec: float = float("nan")
    acf_lag_steps: int = -1
    acf_n: int = 0
    rms_decay_local: float = float("nan")
    rms_decay_local_r2: float = float("nan")
    rms_decay_local_n: int = 0
    rms_decay_local_on_ok: int = 0
    rms_decay_local_off_hint: int = 0
    rms_decay_event: float = float("nan")
    rms_decay_event_r2: float = float("nan")
    rms_decay_event_n: int = 0
    rms_decay_event_win_sec: float = float("nan")


@dataclass
class SignalState:
    """FSM/core detector state that drives transitions."""

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
    warmup_on_start_t: float | None = None
    warmup_on_start_update_idx: int | None = None
    periodicity_collapse_streak: int = 0
    long_ready_streak: int = 0
    prev_score_log: float = float("nan")
    prev_e_t: float = float("nan")
    prev_long_ratio_on: float = float("nan")
    on_support_ema: float = 0.0
    last_off_t: float = float("nan")


@dataclass
class VoteState:
    """Binary vote windows grouped for debug/test visibility."""

    on_short_votes: VoteWindow = field(default_factory=VoteWindow)
    on_soft_votes: VoteWindow = field(default_factory=VoteWindow)
    long_off_votes: VoteWindow = field(default_factory=VoteWindow)
    warmup_on_votes: VoteWindow = field(default_factory=VoteWindow)


@dataclass
class CacheState:
    """Historical/cache state used by feature extraction."""

    long_score_hist: deque[tuple[float, float]] = field(default_factory=deque)
    long_off_recent_hist: deque[tuple[float, float]] = field(default_factory=deque)
    long_baseline_hist: deque[float] = field(default_factory=deque)
    last_quality: QualityCacheSnapshot | None = None
    last_quality_t_end: float = float("nan")
    last_rms_decay: float = float("nan")
    last_rms_decay_r2: float = float("nan")
    last_rms_decay_n: int = 0
    last_rms_decay_t_end: float = float("nan")
    last_rms_decay_event: float = float("nan")
    last_rms_decay_event_r2: float = float("nan")
    last_rms_decay_event_n: int = 0
    last_rms_decay_event_win_sec: float = float("nan")
    last_rms_decay_event_t_end: float = float("nan")
    rms_decay_event_peak: float = float("nan")
    conf_raw_hist: deque[float] = field(default_factory=deque)


@dataclass
class ChannelStreamState:
    """Per-channel stream state grouped into signal/vote/cache blocks.

    Prefer explicit section access (`st.signal`, `st.votes`, `st.cache`).
    Flat attribute forwarding is retained for legacy compatibility only.
    New runtime code must not use forwarded flat fields.
    """

    signal: SignalState = field(default_factory=SignalState)
    votes: VoteState = field(default_factory=VoteState)
    cache: CacheState = field(default_factory=CacheState)

    _SIGNAL_FIELDS: ClassVar[set[str]] = {
        "ring",
        "ring_time_sorted",
        "last_risk_on",
        "phase",
        "evidence",
        "active_start_t",
        "active_start_update_idx",
        "off_candidate_start_t",
        "off_candidate_start_update_idx",
        "on_event_emitted",
        "damped_streak",
        "on_candidate_streak",
        "warmup_on_start_t",
        "warmup_on_start_update_idx",
        "periodicity_collapse_streak",
        "long_ready_streak",
        "prev_score_log",
        "prev_e_t",
        "prev_long_ratio_on",
        "on_support_ema",
        "last_off_t",
    }
    _VOTE_FIELDS: ClassVar[set[str]] = {
        "on_short_votes",
        "on_soft_votes",
        "long_off_votes",
        "warmup_on_votes",
    }
    _CACHE_FIELDS: ClassVar[set[str]] = {
        "long_score_hist",
        "long_off_recent_hist",
        "long_baseline_hist",
        "last_quality",
        "last_quality_t_end",
        "last_rms_decay",
        "last_rms_decay_r2",
        "last_rms_decay_n",
        "last_rms_decay_t_end",
        "last_rms_decay_event",
        "last_rms_decay_event_r2",
        "last_rms_decay_event_n",
        "last_rms_decay_event_win_sec",
        "last_rms_decay_event_t_end",
        "rms_decay_event_peak",
        "conf_raw_hist",
    }
    _VOTE_SUM_FIELDS: ClassVar[dict[str, str]] = {
        "on_short_vote_sum": "on_short_votes",
        "on_soft_vote_sum": "on_soft_votes",
        "long_off_vote_sum": "long_off_votes",
        "warmup_on_vote_sum": "warmup_on_votes",
    }

    def __getattr__(self, name: str):
        # Legacy compatibility path only: prefer explicit section access in new code.
        if name in self._SIGNAL_FIELDS:
            return getattr(self.signal, name)
        if name in self._VOTE_FIELDS:
            return getattr(self.votes, name)
        if name in self._CACHE_FIELDS:
            return getattr(self.cache, name)
        vote_field = self._VOTE_SUM_FIELDS.get(name)
        if vote_field is not None:
            return int(getattr(self.votes, vote_field).sum)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        # Legacy compatibility path only: prefer explicit section access in new code.
        if name in {"signal", "votes", "cache"}:
            super().__setattr__(name, value)
            return
        if name in self._SIGNAL_FIELDS and "signal" in self.__dict__:
            setattr(self.signal, name, value)
            return
        if name in self._VOTE_FIELDS and "votes" in self.__dict__:
            setattr(self.votes, name, value)
            return
        if name in self._CACHE_FIELDS and "cache" in self.__dict__:
            setattr(self.cache, name, value)
            return
        super().__setattr__(name, value)


@dataclass
class BurstSignalState:
    """Burst detector state (independent from sustained-risk FSM state)."""

    phase: str = BURST_PHASE_OFF
    candidate_start_t: float | None = None
    candidate_start_update_idx: int | None = None
    candidate_last_seen_t: float = float("nan")
    active_start_t: float | None = None
    active_start_update_idx: int | None = None
    active_last_seen_t: float = float("nan")
    on_event_emitted: bool = False
    freq_ema_hz: float = float("nan")
    freq_peak_hz: float = float("nan")
    support_peak: float = float("nan")
    confidence_peak: float = float("nan")
    score_peak: float = float("nan")
    short_trigger_hits: int = 0
    accel_hits: int = 0
    candidate_true_ticks: int = 0
    last_off_t: float = float("nan")


@dataclass
class BurstVoteState:
    """Binary vote windows used by burst entry density checks."""

    candidate_votes: VoteWindow = field(default_factory=VoteWindow)


@dataclass
class BurstChannelState:
    """Per-channel burst-sidecar state grouped independently from main FSM."""

    signal: BurstSignalState = field(default_factory=BurstSignalState)
    votes: BurstVoteState = field(default_factory=BurstVoteState)

    _SIGNAL_FIELDS: ClassVar[set[str]] = set(BurstSignalState.__dataclass_fields__.keys())
    _VOTE_FIELDS: ClassVar[set[str]] = set(BurstVoteState.__dataclass_fields__.keys())
    _VOTE_SUM_FIELDS: ClassVar[dict[str, str]] = {
        "candidate_vote_sum": "candidate_votes",
    }

    def __getattr__(self, name: str):
        if name in self._SIGNAL_FIELDS:
            return getattr(self.signal, name)
        if name in self._VOTE_FIELDS:
            return getattr(self.votes, name)
        vote_field = self._VOTE_SUM_FIELDS.get(name)
        if vote_field is not None:
            return int(getattr(self.votes, vote_field).sum)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name in {"signal", "votes"}:
            super().__setattr__(name, value)
            return
        if name in self._SIGNAL_FIELDS and "signal" in self.__dict__:
            setattr(self.signal, name, value)
            return
        if name in self._VOTE_FIELDS and "votes" in self.__dict__:
            setattr(self.votes, name, value)
            return
        super().__setattr__(name, value)


@dataclass
class DecisionContext:
    """FSM gate decisions grouped for one tick."""

    accel_ok: bool
    on_conf_ok: bool
    on_support_ok: bool
    long_ready: bool
    on_long_gate_ok: bool
    accel_evidence_ok: bool
    re_on_active: bool
    re_on_short_ok: bool
    re_on_accel_ok: bool
    on_entry_ready: bool
    on_entry_vote_sum: int


@dataclass
class TickSignalState:
    """Per-tick signal/base statistics before confidence calibration."""

    t1: float
    score: float
    A_tail: float
    D_tail: float
    reason: str
    score_reason_ok: bool
    cut_on_cmp: float
    cut_off_cmp: float
    long_ratio_on: float
    long_ratio_off: float
    long_ratio_off_recent: float
    long_off_n_recent: int
    long_zmax: float
    long_n: int
    baseline_n: int
    short_high: bool
    short_trigger: bool
    collapse_ok: bool
    off_vote_core: bool
    long_off_confirmed: bool
    force_off_now: bool


@dataclass
class TickQualityState:
    """Per-tick quality/confidence/evidence bundle.

    Prefer alias properties (`gate_*`, `feature_*`, `state_*`) in new code.
    Flat storage fields are legacy-compatibility only and should not be read directly.
    """

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
    rms_decay_on_ok: bool
    rms_decay_off_hint: bool
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

    @property
    def gate_confidence_used_now(self) -> float:
        return float(self.conf_now)

    @gate_confidence_used_now.setter
    def gate_confidence_used_now(self, value: float) -> None:
        self.conf_now = float(value)

    @property
    def gate_confidence_raw_now(self) -> float:
        return float(self.raw_now)

    @gate_confidence_raw_now.setter
    def gate_confidence_raw_now(self, value: float) -> None:
        self.raw_now = float(value)

    @property
    def gate_long_baseline_ready(self) -> bool:
        return bool(self.long_ready)

    @gate_long_baseline_ready.setter
    def gate_long_baseline_ready(self, value: bool) -> None:
        self.long_ready = bool(value)

    @property
    def state_cold_start_warmup_active(self) -> bool:
        return bool(self.warmup_mode)

    @state_cold_start_warmup_active.setter
    def state_cold_start_warmup_active(self, value: bool) -> None:
        self.warmup_mode = bool(value)

    @property
    def gate_onset_acceleration_ok(self) -> bool:
        return bool(self.accel_ok)

    @gate_onset_acceleration_ok.setter
    def gate_onset_acceleration_ok(self, value: bool) -> None:
        self.accel_ok = bool(value)

    @property
    def gate_calibration_enabled(self) -> bool:
        return bool(self.use_cal_gate)

    @gate_calibration_enabled.setter
    def gate_calibration_enabled(self, value: bool) -> None:
        self.use_cal_gate = bool(value)

    @property
    def gate_calibration_active(self) -> bool:
        return bool(self.cal_on_active)

    @gate_calibration_active.setter
    def gate_calibration_active(self, value: bool) -> None:
        self.cal_on_active = bool(value)

    @property
    def gate_calibration_confidence_threshold(self) -> float:
        return float(self.cal_on_conf_thr)

    @gate_calibration_confidence_threshold.setter
    def gate_calibration_confidence_threshold(self, value: float) -> None:
        self.cal_on_conf_thr = float(value)

    @property
    def feature_support_score(self) -> float:
        return float(self.on_support)

    @feature_support_score.setter
    def feature_support_score(self, value: float) -> None:
        self.on_support = float(value)

    @property
    def feature_support_ema(self) -> float:
        return float(self.on_support_ema)

    @feature_support_ema.setter
    def feature_support_ema(self, value: float) -> None:
        self.on_support_ema = float(value)

    @property
    def feature_score_log_delta(self) -> float:
        return float(self.delta_s_log)

    @feature_score_log_delta.setter
    def feature_score_log_delta(self, value: float) -> None:
        self.delta_s_log = float(value)

    @property
    def feature_evidence_delta(self) -> float:
        return float(self.delta_e)

    @feature_evidence_delta.setter
    def feature_evidence_delta(self, value: float) -> None:
        self.delta_e = float(value)


@dataclass
class TickVoteState:
    """Per-tick vote aggregates."""

    warmup_vote_sum: int
    warmup_on_confirmed: bool
    on_soft_vote_sum: int
    on_soft_confirmed: bool

    @property
    def gate_warmup_entry_confirmed(self) -> bool:
        return bool(self.warmup_on_confirmed)

    @gate_warmup_entry_confirmed.setter
    def gate_warmup_entry_confirmed(self, value: bool) -> None:
        self.warmup_on_confirmed = bool(value)

    @property
    def gate_warmup_entry_vote_sum(self) -> int:
        return int(self.warmup_vote_sum)

    @gate_warmup_entry_vote_sum.setter
    def gate_warmup_entry_vote_sum(self, value: int) -> None:
        self.warmup_vote_sum = int(value)

    @property
    def gate_soft_entry_confirmed(self) -> bool:
        return bool(self.on_soft_confirmed)

    @gate_soft_entry_confirmed.setter
    def gate_soft_entry_confirmed(self, value: bool) -> None:
        self.on_soft_confirmed = bool(value)

    @property
    def gate_soft_entry_vote_sum(self) -> int:
        return int(self.on_soft_vote_sum)

    @gate_soft_entry_vote_sum.setter
    def gate_soft_entry_vote_sum(self, value: int) -> None:
        self.on_soft_vote_sum = int(value)


@dataclass
class TickFeatures:
    """Per-tick features grouped by signal/quality/vote layers.

    Prefer explicit layer access (`tick.signal`, `tick.quality`, `tick.vote`).
    Flat forwarding remains for legacy compatibility only.
    New runtime code must not use forwarded flat fields.
    """

    signal: TickSignalState
    quality: TickQualityState
    vote: TickVoteState

    _SIGNAL_FIELDS: ClassVar[set[str]] = set(TickSignalState.__dataclass_fields__.keys())
    _QUALITY_FIELDS: ClassVar[set[str]] = set(TickQualityState.__dataclass_fields__.keys())
    _VOTE_FIELDS: ClassVar[set[str]] = set(TickVoteState.__dataclass_fields__.keys())

    def __getattr__(self, name: str):
        # Legacy compatibility path only: prefer explicit layer access in new code.
        if name in self._SIGNAL_FIELDS:
            return getattr(self.signal, name)
        if name in self._QUALITY_FIELDS:
            return getattr(self.quality, name)
        if name in self._VOTE_FIELDS:
            return getattr(self.vote, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        # Legacy compatibility path only: prefer explicit layer access in new code.
        if name in {"signal", "quality", "vote"}:
            super().__setattr__(name, value)
            return
        if name in self._SIGNAL_FIELDS and "signal" in self.__dict__:
            setattr(self.signal, name, value)
            return
        if name in self._QUALITY_FIELDS and "quality" in self.__dict__:
            setattr(self.quality, name, value)
            return
        if name in self._VOTE_FIELDS and "vote" in self.__dict__:
            setattr(self.vote, name, value)
            return
        super().__setattr__(name, value)


__all__ = [
    "SampleTuple",
    "UpdateBatch",
    "UpdateBatchList",
    "PHASE_OFF",
    "PHASE_ON_CANDIDATE",
    "PHASE_ON_CONFIRMED",
    "PHASE_OFF_CANDIDATE",
    "PHASE_OFF_CONFIRMED",
    "BURST_PHASE_OFF",
    "BURST_PHASE_CANDIDATE",
    "BURST_PHASE_ACTIVE",
    "MP_RUNTIME_MODE_LIVE",
    "MP_RUNTIME_MODE_REPLAY",
    "MP_RUNTIME_MODE_CHOICES",
    "VoteWindow",
    "QualityCacheSnapshot",
    "SignalState",
    "VoteState",
    "CacheState",
    "ChannelStreamState",
    "BurstSignalState",
    "BurstVoteState",
    "BurstChannelState",
    "DecisionContext",
    "TickSignalState",
    "TickQualityState",
    "TickVoteState",
    "TickFeatures",
]
