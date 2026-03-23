"""I/O adapter functions for modular streaming detector."""

from __future__ import annotations

import glob
import os
import time
from typing import Sequence

import numpy as np
import pandas as pd

from .osc_config_modul import (
    DEFAULT_DEVICE,
    DEFAULT_STREAM_INPUT_CSV,
    DEFAULT_UPDATE_SEC,
)
from .osc_core_signal_modul import infer_time_col, to_float_np
from .osc_state_modul import SampleTuple, UpdateBatch, UpdateBatchList

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))

class StreamListReceiver:
    """Simple receiver for `on_sample(device, channel, t, v)` stream interface."""

    def __init__(self) -> None:
        self.samples: list[SampleTuple] = []

    def on_sample(self, device: str, channel: str, t: float, v: float) -> None:
        self.samples.append((str(device), str(channel), float(t), float(v)))



def _pick_channel_columns(
    vdf: pd.DataFrame,
    *,
    tcol: str,
    target_channels: Sequence[str] | None,
    max_channels: int | None,
) -> list[str]:
    """Select numeric channel columns with metadata-aware prioritization."""

    raw_cols = [str(c) for c in vdf.columns if str(c) != str(tcol)]
    if not raw_cols:
        return []

    numeric_meta: dict[str, tuple[int, float, bool, bool]] = {}
    for c in raw_cols:
        arr = to_float_np(vdf[c])
        fm = np.isfinite(arr)
        cnt = int(np.count_nonzero(fm))
        if cnt <= 0:
            continue
        std = float(np.nanstd(arr[fm])) if cnt > 0 else 0.0
        name = c.strip().lower()
        is_voltage_like = ("volt" in name) or ("v_pu" in name) or ("voltage" in name) or (name == "v")
        is_quality_like = ("quality" in name) or (name in {"q", "flag", "status"})
        numeric_meta[c] = (cnt, std, is_voltage_like, is_quality_like)

    if not numeric_meta:
        return []

    if target_channels is not None:
        out = []
        for c in target_channels:
            cs = str(c).strip()
            if cs and cs in numeric_meta:
                out.append(cs)
    else:
        out = list(numeric_meta.keys())
        non_quality = [c for c in out if (not numeric_meta[c][3])]
        # Prefer non-metadata numeric channels by default when auto-selecting.
        if non_quality:
            out = non_quality
        out.sort(
            key=lambda c: (
                1 if numeric_meta[c][2] else 0,            # voltage-like first
                1 if (not numeric_meta[c][3]) else 0,      # non-quality first
                1 if numeric_meta[c][1] > 0 else 0,        # non-flat first
                numeric_meta[c][1],                        # larger std first
                numeric_meta[c][0],                        # more valid points first
            ),
            reverse=True,
        )

    if max_channels is not None:
        mch = int(max_channels)
        if mch <= 0:
            return []
        out = out[:mch]
    return out


def build_update_batches_from_voltage_csv(
    vcsv: str,
    *,
    device: str = DEFAULT_DEVICE,
    update_sec: float = DEFAULT_UPDATE_SEC,
    target_channels: Sequence[str] | None = None,
    max_channels: int | None = 1,
) -> UpdateBatchList:
    """
    Convert CSV into update batches.

    Returns:
    - list of updates; each update is a list of `(device, channel, t, v)`.
    """
    if (not np.isfinite(update_sec)) or (float(update_sec) <= 0.0):
        raise ValueError("update_sec must be positive")

    vdf = pd.read_csv(vcsv, encoding="utf-8-sig")
    if vdf.shape[0] == 0 or vdf.shape[1] < 2:
        return []

    tcol = infer_time_col(vdf)
    channel_cols = _pick_channel_columns(
        vdf,
        tcol=tcol,
        target_channels=target_channels,
        max_channels=max_channels,
    )
    if not channel_cols:
        return []

    t_arr = to_float_np(vdf[tcol])
    m_t = np.isfinite(t_arr)
    if int(np.count_nonzero(m_t)) == 0:
        return []

    rows = vdf.loc[m_t, channel_cols].copy()
    rows.insert(0, "__t__", t_arr[m_t])
    rows = rows.sort_values("__t__", kind="mergesort")

    t_start = float(rows["__t__"].iloc[0])
    if rows.empty:
        return []

    t_last = float(rows["__t__"].iloc[-1])
    eff_duration = max(1e-12, max(0.0, t_last - t_start) + 1e-12)
    n_updates = int(max(1, np.ceil(eff_duration / float(update_sec))))
    batches: UpdateBatchList = [[] for _ in range(n_updates)]
    t_rows = rows["__t__"].to_numpy(dtype=float, copy=False)
    idx_rows = ((t_rows - float(t_start)) / float(update_sec)).astype(np.int64, copy=False)
    idx_valid = (idx_rows >= 0) & (idx_rows < int(n_updates))
    if not np.any(idx_valid):
        return batches

    vals = rows[channel_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
    dev_s = str(device)
    valid_rows = np.flatnonzero(idx_valid)
    ch_names = [str(c) for c in channel_cols]
    for r in valid_rows:
        idx = int(idx_rows[r])
        t = float(t_rows[r])
        rowv = vals[r]
        for j, ch_s in enumerate(ch_names):
            fv = float(rowv[j])
            if not np.isfinite(fv):
                continue
            batches[idx].append((dev_s, ch_s, t, fv))
    return batches


def build_update_batches_from_test_signal(
    *,
    device: str = DEFAULT_DEVICE,
    update_sec: float = DEFAULT_UPDATE_SEC,
    duration_sec: float = 120.0,
    sampling_hz: float = 256.0,
    freq_hz: float = 8.0,
    amp: float = 0.005,
    noise_std: float = 0.0005,
    seed: int = 42,
    channel_name: str = "V1",
) -> UpdateBatchList:
    """Build synthetic update batches for test-mode runtime verification."""

    if (not np.isfinite(update_sec)) or (float(update_sec) <= 0.0):
        raise ValueError("update_sec must be positive")
    if (not np.isfinite(duration_sec)) or (float(duration_sec) <= 0.0):
        raise ValueError("duration_sec must be positive")
    if (not np.isfinite(sampling_hz)) or (float(sampling_hz) <= 0.0):
        raise ValueError("sampling_hz must be positive")
    if (not np.isfinite(freq_hz)) or (float(freq_hz) <= 0.0):
        raise ValueError("freq_hz must be positive")
    if (not np.isfinite(amp)) or (float(amp) < 0.0):
        raise ValueError("amp must be >= 0")
    if (not np.isfinite(noise_std)) or (float(noise_std) < 0.0):
        raise ValueError("noise_std must be >= 0")

    dt = float(1.0 / float(sampling_hz))
    n = int(max(1, np.floor(float(duration_sec) * float(sampling_hz))))
    t = np.arange(n, dtype=float) * float(dt)
    rng = np.random.default_rng(int(seed))
    base = float(amp) * np.sin(2.0 * np.pi * float(freq_hz) * t)
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=n) if float(noise_std) > 0.0 else 0.0
    v = base + noise

    n_updates = int(max(1, np.ceil((float(t[-1]) + 1e-12) / float(update_sec))))
    batches: UpdateBatchList = [[] for _ in range(n_updates)]
    idx_rows = (t / float(update_sec)).astype(np.int64, copy=False)
    valid = (idx_rows >= 0) & (idx_rows < int(n_updates))
    if not np.any(valid):
        return batches
    dev_s = str(device)
    ch_s = str(channel_name)
    for r in np.flatnonzero(valid):
        idx = int(idx_rows[r])
        fv = float(v[r])
        if not np.isfinite(fv):
            continue
        batches[idx].append((dev_s, ch_s, float(t[r]), float(fv)))
    return batches


def iter_live_update_batches_from_csv_tail(
    vcsv: str,
    *,
    device: str = DEFAULT_DEVICE,
    update_sec: float = DEFAULT_UPDATE_SEC,
    target_channels: Sequence[str] | None = None,
    max_channels: int | None = 1,
    max_updates: int = 0,
    status_cb=None,
):
    """
    Yield one update batch every `update_sec` by tail-following a growing CSV.

    Behavior:
    - Rebuilds CSV->batches snapshot each tick.
    - Emits next unseen update index if available, else emits empty batch.
    - `max_updates<=0` means unbounded streaming.
    """

    if (not np.isfinite(update_sec)) or (float(update_sec) <= 0.0):
        raise ValueError("update_sec must be positive")
    if int(max_updates) < 0:
        raise ValueError("max_updates must be >= 0")
    status = print if status_cb is None else status_cb
    next_unseen_idx = 0
    emitted = 0
    next_tick_deadline = time.monotonic()
    warned_missing = False

    while (int(max_updates) <= 0) or (int(emitted) < int(max_updates)):
        path = str(vcsv).strip()
        if (not path) or (not os.path.isfile(path)):
            if not warned_missing:
                status(f"[LIVE_SOURCE] waiting for csv path: {path}")
                warned_missing = True
            batch: UpdateBatch = []
        else:
            warned_missing = False
            try:
                snapshot_batches = build_update_batches_from_voltage_csv(
                    path,
                    device=str(device),
                    update_sec=float(update_sec),
                    target_channels=target_channels,
                    max_channels=max_channels,
                )
            except Exception as exc:
                status(f"[LIVE_SOURCE] snapshot read failed, emit empty batch | err={exc}")
                snapshot_batches = []
            if int(next_unseen_idx) < int(len(snapshot_batches)):
                batch = list(snapshot_batches[int(next_unseen_idx)])
                next_unseen_idx += 1
            else:
                batch = []
        emitted += 1
        yield batch

        next_tick_deadline += float(update_sec)
        sleep_sec = float(next_tick_deadline - time.monotonic())
        if float(sleep_sec) > 0.0:
            time.sleep(float(sleep_sec))
        else:
            # Backlog/slow processing: reset deadline to avoid drift explosion.
            next_tick_deadline = time.monotonic()


def ingest_batches_to_list(
    batches: UpdateBatchList,
    *,
    receiver: StreamListReceiver | None = None,
) -> list[SampleTuple]:
    """Flatten update batches into sample tuples via stream receiver API."""

    recv = receiver if receiver is not None else StreamListReceiver()
    for batch in batches:
        for device, channel, t, v in batch:
            recv.on_sample(device, channel, t, v)
    return recv.samples



def _guess_default_stream_csv() -> str | None:
    """Pick a default stream CSV path so `python module.py` runs immediately."""
    candidates: list[str] = []

    cfg_csv = str(DEFAULT_STREAM_INPUT_CSV).strip()
    if cfg_csv:
        candidates.append(cfg_csv)

    env_csv = os.environ.get("STREAM_INPUT_CSV", "").strip()
    if env_csv:
        candidates.append(env_csv)

    local_dirs = [
        os.path.join(_PROJECT_ROOT, "tester", "derived_wmu_3ch_csv"),
        os.path.join(_PROJECT_ROOT, "tester"),
        os.path.join(_PROJECT_ROOT, "csv"),
    ]
    for d in local_dirs:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)):
            candidates.append(p)

    seen = set()
    for p in candidates:
        ap = os.path.abspath(str(p))
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            return ap
    return None

__all__ = [
    "StreamListReceiver",
    "infer_time_col",
    "_pick_channel_columns",
    "build_update_batches_from_voltage_csv",
    "build_update_batches_from_test_signal",
    "iter_live_update_batches_from_csv_tail",
    "ingest_batches_to_list",
    "_guess_default_stream_csv",
]
