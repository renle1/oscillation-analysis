# -*- coding: utf-8 -*-
"""
lineup_parallel_Fin_usecols_ver.py

라인업 후처리를 모듈 형태로 수행한다.
- 입력 CSV를 1개 이상 받는다. (공통 스키마: case, channel)
- case별 원본 전압 CSV를 찾아 채널 위험 요약을 계산한다.
- 언더슛 허용 규칙을 적용한다.
  undershoot duration <= (undershoot_tol_cycles / nominal_hz) 이면 OK 처리
- 원본 전압 CSV는 헤더를 먼저 읽고, 필요한 컬럼만 선택 로드(usecols)한다.

기본 import 사용 예시:
    import lineup_parallel_Fin_usecols_ver as lineup
    lineup.run_lineup(
        in_csv_paths=[r"...\\kpg_second_filter.csv", r"...\\another.csv"],
        voltage_dir=[r"...\\csv\\KPGC_xxx", r"...\\csv\\KPGC_yyy"],
        out_csv=r"...\\out\\GlobalRisk_Scenario_Vmax_Vmin_tag_duration.csv",
    )
"""

from __future__ import annotations

import csv
import glob
import heapq
import os
import re
import sqlite3
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


try:
    import config_paths as _cfg_paths
except Exception:
    _cfg_paths = None


# =========================
# 사용자 제어 변수(파일 상단)
# =========================
# 이 값을 바꾸면 스크립트 실행/모듈 실행(별도 cfg 미전달) 기본 동작이 바뀐다.
NOMINAL_HZ = 60.0
UNDERSHOOT_TOL_CYCLES = 2.0
MAX_IN_FLIGHT_FACTOR = 2
MAX_TASKS_PER_CHILD = 200
TEMP_FLUSH_ROWS = 2000
INPUT_CHUNK_ROWS = 50000
SORT_CHUNK_ROWS = 50000
DB_COMMIT_EVERY_CHUNKS = 20


_WORKER_VOLTAGE_INDEX_CACHE = {"dirs_key": None, "index": None}


def _new_pool(max_workers: int) -> ProcessPoolExecutor:
    kwargs = {"max_workers": int(max_workers)}
    if int(MAX_TASKS_PER_CHILD) > 0:
        kwargs["max_tasks_per_child"] = int(MAX_TASKS_PER_CHILD)
    try:
        return ProcessPoolExecutor(**kwargs)
    except TypeError:
        kwargs.pop("max_tasks_per_child", None)
        return ProcessPoolExecutor(**kwargs)


def _safe_float(v, default: float = np.nan) -> float:
    try:
        fv = float(v)
    except Exception:
        return default
    if not np.isfinite(fv):
        return default
    return fv


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class LineupConfig:
    skip_sec: float = 2.0
    win_sec: float = 8.0

    overshoot_max: float = 1.20
    undershoot_warn: float = 0.90
    undershoot_sev: float = 0.80

    nominal_hz: float = NOMINAL_HZ
    undershoot_tol_cycles: float = UNDERSHOOT_TOL_CYCLES

    # 호환성과 속도를 위해 고정 dt 옵션 유지
    dt: float = 1.0 / 120.0

    exclude_bus_nos: tuple[int, ...] = ()
    case_basename_only: bool = True
    max_workers: int = 4 # 병렬 워커


def _as_float_array(x: pd.Series) -> np.ndarray:
    """시리즈를 float numpy 배열로 변환한다. 변환 불가 값은 NaN으로 처리한다."""
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def _infer_time_col_from_columns(cols: Sequence[str]) -> str:
    """컬럼 목록에서 시간 컬럼명을 추정한다. 대표 키가 없으면 첫 번째 컬럼을 사용한다."""
    lower_map = {str(c).strip().lower(): c for c in cols}
    for key in ("time", "t", "sec", "seconds"):
        if key in lower_map:
            return lower_map[key]
    return cols[0]


def _infer_time_col(df: pd.DataFrame) -> str:
    """데이터프레임에서 시간 컬럼명을 추정한다."""
    return _infer_time_col_from_columns(list(df.columns))


def _bus_no_from_text(text: str):
    """채널 문자열에서 BUS 번호를 추출한다. 찾지 못하면 None을 반환한다."""
    s = str(text)
    m = re.search(r"\bBUS\s*([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bBUS([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bVOLT(?:AGE)?\s*([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"^\s*([0-9]{1,5})\s*\|", s)
    if m:
        return int(m.group(1))
    return None


def _short_case(case_val: str, case_basename_only: bool) -> str:
    """케이스 값을 출력용 짧은 이름으로 정규화한다."""
    c = str(case_val).strip()
    if not c:
        return c

    name = os.path.basename(c) if case_basename_only else c
    name = re.sub(r"_Voltage\.csv$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.csv$", "", name, flags=re.IGNORECASE)
    return name


def _normalize_voltage_dirs(voltage_dir: Sequence[str] | str) -> list[str]:
    """전압 입력 폴더(단일/복수)를 절대경로 목록으로 정규화한다."""
    if isinstance(voltage_dir, str):
        items = [voltage_dir]
    else:
        items = [str(x) for x in voltage_dir]

    out = []
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        if any(ch in s for ch in "*?[]"):
            for p in sorted(glob.glob(s)):
                if os.path.isdir(p):
                    out.append(p)
            continue
        if os.path.isdir(s):
            out.append(s)

    norm = []
    seen = set()
    for d in out:
        ad = os.path.abspath(d)
        if ad in seen:
            continue
        seen.add(ad)
        norm.append(ad)
    return norm


def _strip_voltage_suffix(stem: str) -> str:
    """파일 stem에서 후행 '_Voltage'를 제거한 키를 반환한다."""
    return re.sub(r"_voltage$", "", str(stem).strip(), flags=re.IGNORECASE)


def _build_voltage_csv_index(voltage_dirs: Sequence[str]) -> dict:
    """전압 CSV 탐색 인덱스를 1회 생성한다."""
    by_name = {}
    by_stem = {}
    by_core = {}

    for d in voltage_dirs:
        for p in sorted(glob.glob(os.path.join(str(d), "*.csv"))):
            ap = os.path.abspath(p)
            name = os.path.basename(ap)
            stem = os.path.splitext(name)[0]
            core = _strip_voltage_suffix(stem)

            name_k = name.lower()
            stem_k = stem.lower()
            core_k = core.lower()

            by_name.setdefault(name_k, []).append(ap)
            by_stem.setdefault(stem_k, []).append(ap)
            by_core.setdefault(core_k, []).append(ap)

    for mp in (by_name, by_stem, by_core):
        for k in list(mp.keys()):
            uniq = sorted(set(mp[k]))
            mp[k] = uniq

    return {"by_name": by_name, "by_stem": by_stem, "by_core": by_core}


def _get_worker_voltage_index(voltage_dirs: Sequence[str]) -> dict:
    """워커 프로세스에서 전압 CSV 인덱스를 지연 생성/재사용한다."""
    dirs_key = tuple(os.path.abspath(str(d)) for d in voltage_dirs)
    cache_key = _WORKER_VOLTAGE_INDEX_CACHE.get("dirs_key")
    if cache_key == dirs_key and _WORKER_VOLTAGE_INDEX_CACHE.get("index") is not None:
        return _WORKER_VOLTAGE_INDEX_CACHE["index"]

    idx = _build_voltage_csv_index(dirs_key)
    _WORKER_VOLTAGE_INDEX_CACHE["dirs_key"] = dirs_key
    _WORKER_VOLTAGE_INDEX_CACHE["index"] = idx
    return idx


def _resolve_voltage_csv(case_val: str, voltage_dirs: Sequence[str], voltage_index: dict | None = None):
    """케이스명으로 원본 전압 CSV 경로를 결정적으로 탐색한다."""
    c = str(case_val).strip()
    if not c:
        return None, "voltage_csv_not_found"

    if os.path.isabs(c) and os.path.isfile(c):
        return c, None

    idx = voltage_index or _build_voltage_csv_index(voltage_dirs)
    by_name = idx.get("by_name", {})
    by_stem = idx.get("by_stem", {})
    by_core = idx.get("by_core", {})

    rel_name = os.path.basename(c)
    if rel_name and rel_name.lower().endswith(".csv"):
        name_hits = by_name.get(rel_name.lower(), [])
        if len(name_hits) == 1:
            return name_hits[0], None
        if len(name_hits) > 1:
            return None, "voltage_csv_ambiguous_match"

    stem_raw = os.path.splitext(rel_name)[0] if rel_name else os.path.splitext(c)[0]
    stem = str(stem_raw).strip()
    if not stem:
        return None, "voltage_csv_not_found"

    stem_k = stem.lower()
    core_k = _strip_voltage_suffix(stem).lower()
    stem_voltage_k = (core_k + "_voltage") if core_k else ""

    cand = []
    cand.extend(by_stem.get(stem_k, []))
    cand.extend(by_stem.get(stem_voltage_k, []))
    cand.extend(by_core.get(core_k, []))

    hits = sorted(set(cand))
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, "voltage_csv_ambiguous_match"
    return None, "voltage_csv_not_found"


def _pick_voltage_column_from_columns(columns: Sequence[str], tcol: str, bus_no: int):
    """컬럼 목록에서 지정 BUS 번호에 가장 잘 맞는 전압 컬럼명을 선택한다."""
    pat_volt = re.compile(r"\bVOLT(?:AGE)?\s*%d\b" % int(bus_no), flags=re.IGNORECASE)
    pat_bus = re.compile(r"\bBUS\s*%d\b" % int(bus_no), flags=re.IGNORECASE)
    pat_bus_join = re.compile(r"\bBUS%d\b" % int(bus_no), flags=re.IGNORECASE)

    cand = []
    for c in columns:
        if c == tcol:
            continue
        s = str(c)
        if pat_volt.search(s) or pat_bus.search(s) or pat_bus_join.search(s):
            cand.append(s)
    if not cand:
        return None

    def rank(colname: str):
        r0 = 0
        if pat_volt.search(colname):
            r0 -= 10
        if pat_bus.search(colname) or pat_bus_join.search(colname):
            r0 -= 5
        return (r0, len(colname), colname)

    cand.sort(key=rank)
    return cand[0]


def _pick_voltage_column_for_bus(vdf: pd.DataFrame, tcol: str, bus_no: int):
    """데이터프레임에서 지정 BUS 번호에 가장 잘 맞는 전압 컬럼명을 선택한다."""
    return _pick_voltage_column_from_columns(list(vdf.columns), tcol, bus_no)


def _read_voltage_header(vcsv: str) -> list[str]:
    """원본 전압 CSV의 헤더만 읽어 컬럼 목록을 반환한다."""
    try:
        hdf = pd.read_csv(vcsv, encoding="utf-8-sig", nrows=0)
    except Exception:
        hdf = pd.read_csv(vcsv, encoding="utf-8", nrows=0)
    return [str(c) for c in hdf.columns]


def _read_voltage_data(vcsv: str, usecols: list[str] | None) -> pd.DataFrame:
    """원본 전압 CSV를 읽는다. usecols를 지정하면 필요한 컬럼만 읽는다."""
    try:
        return pd.read_csv(vcsv, encoding="utf-8-sig", usecols=usecols)
    except Exception:
        return pd.read_csv(vcsv, encoding="utf-8", usecols=usecols)


def _tag_dev_rank(vmax: float, vmin: float, cfg: LineupConfig, undershoot_active: bool):
    """Vmax/Vmin 기준으로 태그, 편차(dev), 정렬 우선순위(rank)를 계산한다."""
    tags = []

    overshoot_dev = 0.0
    undershoot_dev = 0.0

    if np.isfinite(vmax) and (vmax > cfg.overshoot_max):
        tags.append("OVERSHOOT")
        overshoot_dev = float(vmax - cfg.overshoot_max)

    if undershoot_active and np.isfinite(vmin):
        if vmin < cfg.undershoot_sev:
            tags.append("UNDERSHOOT_SEVERE")
            undershoot_dev = float(cfg.undershoot_warn - vmin)
        elif vmin < cfg.undershoot_warn:
            tags.append("UNDERSHOOT_WARN")
            undershoot_dev = float(cfg.undershoot_warn - vmin)

    if not tags:
        return "OK", 0.0, 9

    tag = "|".join(tags)
    dev = max(overshoot_dev, undershoot_dev)

    if ("OVERSHOOT" in tag) and ("UNDERSHOOT_SEVERE" in tag):
        rank = 0
    elif ("OVERSHOOT" in tag) and ("UNDERSHOOT_WARN" in tag):
        rank = 1
    else:
        rank = 2

    return tag, float(dev), int(rank)


def _is_internal_error_tag(tag: str) -> bool:
    """내부 처리 오류 성격의 태그인지 판별한다."""
    if tag is None:
        return True
    s = str(tag)
    bad_keys = (
        "parse_fail",
        "not_found",
        "ambiguous",
        "too_small",
        "too_few",
        "window_too_small",
        "voltage_col_not_found",
        "time_invalid",
        "read_fail",
        "worker_error",
    )
    return any(k in s for k in bad_keys)


def _normalize_input_paths(in_csv_paths: Sequence[str] | str) -> list[str]:
    """입력 경로(단일/복수/와일드카드)를 절대경로 목록으로 정규화한다."""
    if isinstance(in_csv_paths, str):
        items = [in_csv_paths]
    else:
        items = [str(x) for x in in_csv_paths]

    out = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if any(ch in item for ch in "*?[]"):
            out.extend(sorted(glob.glob(item)))
        else:
            out.append(item)

    # 중복 제거 + 절대경로 정규화
    norm = []
    seen = set()
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            norm.append(ap)
    return norm

def _prepare_case_channel_db(paths: Sequence[str], db_path: str) -> tuple[sqlite3.Connection, int]:
    """입력 CSV들을 청크로 읽어 case/channel 고유쌍을 디스크(SQLite)에 적재한다."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS case_channel ("
        "case_val TEXT NOT NULL, "
        "channel TEXT NOT NULL, "
        "PRIMARY KEY(case_val, channel))"
    )
    conn.commit()

    pending_chunk_commits = 0
    for path in paths:
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
            enc = "utf-8-sig"
        except Exception:
            header = pd.read_csv(path, nrows=0, encoding="utf-8")
            enc = "utf-8"

        if "case" not in header.columns or "channel" not in header.columns:
            conn.close()
            raise ValueError("input CSV missing required columns(case, channel): %s" % path)

        for chunk in pd.read_csv(path, usecols=["case", "channel"], chunksize=int(INPUT_CHUNK_ROWS), encoding=enc):
            if chunk is None or len(chunk) == 0:
                continue
            part = chunk.copy()
            part["case"] = part["case"].astype(str).str.strip()
            part["channel"] = part["channel"].astype(str).str.strip()
            part = part[(part["case"] != "") & (part["channel"] != "")]
            if len(part) == 0:
                continue

            rows = list(map(tuple, part[["case", "channel"]].itertuples(index=False, name=None)))
            conn.executemany(
                "INSERT OR IGNORE INTO case_channel(case_val, channel) VALUES (?, ?)",
                rows,
            )
            pending_chunk_commits += 1
            if pending_chunk_commits >= int(DB_COMMIT_EVERY_CHUNKS):
                conn.commit()
                pending_chunk_commits = 0

    if pending_chunk_commits > 0:
        conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT case_val) FROM case_channel")
    total_cases = int(cur.fetchone()[0])
    cur.close()
    return conn, total_cases


def _iter_case_groups_from_db(conn: sqlite3.Connection) -> Iterable[tuple[str, list[str]]]:
    """SQLite에 저장된 case/channel을 case 단위 그룹으로 순차 반환한다."""
    cur = conn.cursor()
    cur.execute("SELECT case_val, channel FROM case_channel ORDER BY case_val, rowid")
    prev_case = None
    chs: list[str] = []
    try:
        for case_val, channel in cur:
            c = str(case_val)
            ch = str(channel)
            if prev_case is None:
                prev_case = c
                chs = [ch]
                continue
            if c == prev_case:
                chs.append(ch)
            else:
                yield prev_case, chs
                prev_case = c
                chs = [ch]
        if prev_case is not None:
            yield prev_case, chs
    finally:
        cur.close()


def _lineup_row_sort_key(row: dict) -> tuple:
    tag = str(row.get("tag", ""))
    is_err = 1 if _is_internal_error_tag(tag) else 0
    rank = _safe_int(row.get("_rank", 99), default=99)
    dev = _safe_float(row.get("_dev", 0.0), default=0.0)
    t_any = _safe_float(row.get("t_any_violation", 0.0), default=0.0)
    channel = str(row.get("channel", ""))
    case = str(row.get("case", ""))
    vmax = _safe_float(row.get("Vmax", None), default=float("-inf"))
    vmin = _safe_float(row.get("Vmin", None), default=float("inf"))
    return (is_err, rank, -dev, -t_any, channel, case, -vmax, vmin)


def _external_sort_lineup_rows(in_csv: str, out_csv: str, *, chunk_rows: int = SORT_CHUNK_ROWS) -> None:
    """라인업 임시 결과 CSV를 외부 정렬(청크 정렬 + k-way merge)로 정렬한다."""
    in_csv = os.path.abspath(in_csv)
    out_csv = os.path.abspath(out_csv)
    parent = os.path.dirname(out_csv) or "."

    temp_parts: list[str] = []
    base_cols = ["channel", "case", "Vmax", "Vmin", "tag", "_dev", "_rank", "t_any_violation"]

    try:
        for chunk in pd.read_csv(in_csv, chunksize=int(chunk_rows), encoding="utf-8-sig"):
            if chunk is None or len(chunk) == 0:
                continue
            for c in base_cols:
                if c not in chunk.columns:
                    chunk[c] = np.nan

            chunk["_is_err"] = chunk["tag"].apply(_is_internal_error_tag)
            chunk = chunk.sort_values(
                by=["_is_err", "_rank", "_dev", "t_any_violation", "channel", "case", "Vmax", "Vmin"],
                ascending=[True, True, False, False, True, True, False, True],
                kind="mergesort",
            )
            chunk = chunk[base_cols]

            part = tempfile.NamedTemporaryFile(
                prefix="__lineup_part_",
                suffix=".csv",
                dir=parent,
                delete=False,
            ).name
            chunk.to_csv(part, index=False, encoding="utf-8")
            temp_parts.append(part)

        with open(out_csv, "w", newline="", encoding="utf-8-sig") as fw:
            writer = csv.DictWriter(fw, fieldnames=base_cols)
            writer.writeheader()

            if not temp_parts:
                return

            file_handles = []
            readers = []
            heap = []
            try:
                for idx, path in enumerate(temp_parts):
                    fh = open(path, "r", newline="", encoding="utf-8")
                    rd = csv.DictReader(fh)
                    file_handles.append(fh)
                    readers.append(rd)
                    row = next(rd, None)
                    if row is None:
                        continue
                    heapq.heappush(heap, (_lineup_row_sort_key(row), idx, row))

                while heap:
                    _, idx, row = heapq.heappop(heap)
                    writer.writerow(row)
                    nxt = next(readers[idx], None)
                    if nxt is None:
                        continue
                    heapq.heappush(heap, (_lineup_row_sort_key(nxt), idx, nxt))
            finally:
                for fh in file_handles:
                    try:
                        fh.close()
                    except Exception:
                        pass
    finally:
        for p in temp_parts:
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

def _process_one_case(
    case_raw: str,
    channels: list[str],
    voltage_dirs: Sequence[str],
    cfg: LineupConfig,
    exclude_set: set[int],
):
    """한 개 case의 모든 채널을 평가해 라인업 출력 행 목록을 생성한다."""
    out_rows = []
    case_short = _short_case(case_raw, cfg.case_basename_only)
    voltage_index = _get_worker_voltage_index(voltage_dirs)
    undershoot_tol_sec = (
        float(cfg.undershoot_tol_cycles) / float(cfg.nominal_hz)
        if float(cfg.nominal_hz) > 0
        else 0.0
    )

    ch_items = []
    for ch in channels:
        ch_str = str(ch).strip()
        bus_no = _bus_no_from_text(ch_str)

        if bus_no is None:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "bus_no_parse_fail",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
            continue

        if exclude_set and (int(bus_no) in exclude_set):
            continue

        ch_items.append((ch_str, int(bus_no)))

    if not ch_items:
        return out_rows

    vcsv, resolve_err = _resolve_voltage_csv(case_raw, voltage_dirs, voltage_index=voltage_index)
    if (not vcsv) or (not os.path.isfile(vcsv)):
        err_tag = resolve_err or "voltage_csv_not_found"
        for ch_str, _ in ch_items:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": err_tag,
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
        return out_rows

    try:
        header_cols = _read_voltage_header(vcsv)
    except Exception:
        for ch_str, _ in ch_items:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "voltage_csv_read_fail",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
        return out_rows

    if len(header_cols) < 2:
        for ch_str, _ in ch_items:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "voltage_csv_too_small",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
        return out_rows

    tcol = _infer_time_col_from_columns(header_cols)

    # 채널에서 필요한 BUS 컬럼만 선별해 최소 컬럼으로 로드한다.
    bus_to_vcol = {}
    for _, bus_no in ch_items:
        if bus_no not in bus_to_vcol:
            bus_to_vcol[bus_no] = _pick_voltage_column_from_columns(header_cols, tcol, bus_no)

    if not any(bus_to_vcol.values()):
        for ch_str, _ in ch_items:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "voltage_col_not_found_for_bus",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
        return out_rows

    usecols = [tcol] + [v for v in bus_to_vcol.values() if v]
    usecols = list(dict.fromkeys(usecols))

    try:
        vdf = _read_voltage_data(vcsv, usecols=usecols)
    except Exception:
        # usecols 로드 실패 시 전체 로드로 한 번 더 시도
        try:
            vdf = _read_voltage_data(vcsv, usecols=None)
        except Exception:
            for ch_str, _ in ch_items:
                out_rows.append(
                    {
                        "channel": ch_str,
                        "case": case_short,
                        "Vmax": np.nan,
                        "Vmin": np.nan,
                        "tag": "voltage_csv_read_fail",
                        "_dev": 0.0,
                        "_rank": 99,
                        "t_any_violation": 0.0,
                    }
                )
            return out_rows

    if vdf.shape[0] < 10 or vdf.shape[1] < 2:
        for ch_str, _ in ch_items:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "voltage_csv_too_small",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
        return out_rows

    # fallback 전체 로드가 된 경우를 대비해 시간 컬럼 재확인
    if tcol not in vdf.columns:
        tcol = _infer_time_col(vdf)

    t = _as_float_array(vdf[tcol])

    if not np.isfinite(t).any():
        for ch_str, _ in ch_items:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "time_invalid",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
        return out_rows

    t_valid = t[np.isfinite(t)]
    t0 = float(np.min(t_valid)) + float(cfg.skip_sec)
    t1 = t0 + float(cfg.win_sec)

    for ch_str, bus_no in ch_items:
        vcol = bus_to_vcol.get(bus_no)
        if (not vcol) or (vcol not in vdf.columns):
            vcol = _pick_voltage_column_for_bus(vdf, tcol, bus_no)
        if not vcol:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "voltage_col_not_found_for_bus",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
            continue

        v = _as_float_array(vdf[vcol])
        ok = np.isfinite(t) & np.isfinite(v)
        tt, vv = t[ok], v[ok]
        if len(tt) < 5:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "too_few_samples",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
            continue

        idx = np.argsort(tt)
        tt, vv = tt[idx], vv[idx]

        mw = (tt >= t0) & (tt <= t1)
        if np.count_nonzero(mw) < 5:
            out_rows.append(
                {
                    "channel": ch_str,
                    "case": case_short,
                    "Vmax": np.nan,
                    "Vmin": np.nan,
                    "tag": "window_too_small",
                    "_dev": 0.0,
                    "_rank": 99,
                    "t_any_violation": 0.0,
                }
            )
            continue

        vw = vv[mw]
        vmax = float(np.nanmax(vw))
        vmin = float(np.nanmin(vw))

        m_over = vw > float(cfg.overshoot_max)
        m_under = vw < float(cfg.undershoot_warn)

        t_over = float(np.count_nonzero(m_over) * float(cfg.dt))
        t_under = float(np.count_nonzero(m_under) * float(cfg.dt))

        # 새 규칙:
        # 언더슛 지속시간이 2사이클(기본 60Hz 기준) 이하면 OK로 본다.
        undershoot_active = t_under > float(undershoot_tol_sec)

        tag, dev, rank = _tag_dev_rank(vmax, vmin, cfg, undershoot_active)

        if undershoot_active:
            t_any = float(np.count_nonzero(m_over | m_under) * float(cfg.dt))
        else:
            t_any = t_over

        out_rows.append(
            {
                "channel": ch_str,
                "case": case_short,
                "Vmax": vmax,
                "Vmin": vmin,
                "tag": tag,
                "_dev": float(dev),
                "_rank": int(rank),
                "t_any_violation": float(t_any),
            }
        )

    return out_rows


def _write_lineup_outputs_from_sorted(
    sorted_csv: str,
    out_csv: str,
    within_limit_out_csv: str | None,
) -> int:
    """정렬된 라인업 결과를 최종 출력 파일들로 분기 저장하고 main 행 개수를 반환한다."""
    keep_cols = ["channel", "case", "Vmax", "Vmin", "tag", "t_any_violation"]
    out_abs = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    main_count = 0
    with open(sorted_csv, "r", newline="", encoding="utf-8-sig") as fr:
        reader = csv.DictReader(fr)

        if within_limit_out_csv:
            within_abs = os.path.abspath(within_limit_out_csv)
            os.makedirs(os.path.dirname(within_abs) or ".", exist_ok=True)
            with open(out_abs, "w", newline="", encoding="utf-8-sig") as fw_main, open(
                within_abs, "w", newline="", encoding="utf-8-sig"
            ) as fw_ok:
                w_main = csv.DictWriter(fw_main, fieldnames=keep_cols)
                w_ok = csv.DictWriter(fw_ok, fieldnames=keep_cols)
                w_main.writeheader()
                w_ok.writeheader()

                for row in reader:
                    out_row = {k: row.get(k, "") for k in keep_cols}
                    tag = str(out_row.get("tag", ""))
                    t_any = _safe_float(out_row.get("t_any_violation", 0.0), default=0.0)
                    within = (tag == "OK") and (t_any <= 0.0)
                    if within:
                        w_ok.writerow(out_row)
                    else:
                        w_main.writerow(out_row)
                        main_count += 1
        else:
            with open(out_abs, "w", newline="", encoding="utf-8-sig") as fw_main:
                w_main = csv.DictWriter(fw_main, fieldnames=keep_cols)
                w_main.writeheader()
                for row in reader:
                    out_row = {k: row.get(k, "") for k in keep_cols}
                    w_main.writerow(out_row)
                    main_count += 1

    return int(main_count)


def run_lineup(
    in_csv_paths: Sequence[str] | str,
    voltage_dir: Sequence[str] | str,
    out_csv: str,
    *,
    cfg: LineupConfig | None = None,
    within_limit_out_csv: str | None = None,
    return_df: bool = True,
) -> pd.DataFrame:
    """라인업 후처리 전체를 실행하고 결과 CSV를 저장한 뒤 결과 DataFrame을 반환한다."""
    cfg = cfg or LineupConfig()
    voltage_dirs = _normalize_voltage_dirs(voltage_dir)
    if not voltage_dirs:
        raise FileNotFoundError("no voltage input directories found")

    paths = _normalize_input_paths(in_csv_paths)
    if not paths:
        raise FileNotFoundError("no input CSV files provided")

    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError("input CSV not found: %s" % p)

    exclude_set = set(int(x) for x in cfg.exclude_bus_nos)

    temp_cols = ["channel", "case", "Vmax", "Vmin", "tag", "_dev", "_rank", "t_any_violation"]
    out_abs = os.path.abspath(out_csv)
    out_dir = os.path.dirname(out_abs) or "."
    os.makedirs(out_dir, exist_ok=True)
    temp_rows_csv = os.path.join(out_dir, "__lineup_rows_tmp_%d.csv" % os.getpid())
    sorted_rows_csv = os.path.join(out_dir, "__lineup_rows_sorted_%d.csv" % os.getpid())
    input_db = os.path.join(out_dir, "__lineup_case_channel_%d.sqlite" % os.getpid())
    pd.DataFrame(columns=temp_cols).to_csv(temp_rows_csv, index=False, encoding="utf-8-sig")

    row_buf = []
    rows_written = 0
    total_cases = 0
    db_conn = None

    def _flush_rows() -> None:
        nonlocal row_buf, rows_written
        if not row_buf:
            return
        bdf = pd.DataFrame(row_buf)
        for c in temp_cols:
            if c not in bdf.columns:
                bdf[c] = np.nan
        bdf = bdf[temp_cols]
        bdf.to_csv(temp_rows_csv, mode="a", index=False, header=False, encoding="utf-8")
        rows_written += len(bdf)
        row_buf = []

    workers = max(1, int(cfg.max_workers))
    max_in_flight = max(workers, workers * int(MAX_IN_FLIGHT_FACTOR))
    try:
        db_conn, total_cases = _prepare_case_channel_db(paths, input_db)
        if total_cases <= 0:
            raise ValueError("no valid case/channel rows found in inputs")

        with _new_pool(max_workers=workers) as ex:
            item_iter = iter(_iter_case_groups_from_db(db_conn))
            fut_map = {}

            def _submit_next() -> bool:
                try:
                    case_raw, channels = next(item_iter)
                except StopIteration:
                    return False
                fut = ex.submit(
                    _process_one_case,
                    case_raw,
                    channels,
                    voltage_dirs,
                    cfg,
                    exclude_set,
                )
                fut_map[fut] = (case_raw, channels)
                return True

            while len(fut_map) < max_in_flight and _submit_next():
                pass

            while fut_map:
                done_set, _ = wait(set(fut_map.keys()), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    case_raw, channels = fut_map.pop(fut)
                    try:
                        row_buf.extend(fut.result())
                    except Exception as e:
                        # 개별 워커 실패가 전체 중단으로 번지지 않게 케이스 단위 오류로 기록
                        case_short = _short_case(case_raw, cfg.case_basename_only)
                        for ch in channels:
                            row_buf.append(
                                {
                                    "channel": str(ch),
                                    "case": case_short,
                                    "Vmax": np.nan,
                                    "Vmin": np.nan,
                                    "tag": "worker_error:%s" % type(e).__name__,
                                    "_dev": 0.0,
                                    "_rank": 99,
                                    "t_any_violation": 0.0,
                                }
                            )
                    if len(row_buf) >= int(TEMP_FLUSH_ROWS):
                        _flush_rows()

                while len(fut_map) < max_in_flight and _submit_next():
                    pass

        _flush_rows()
        _external_sort_lineup_rows(temp_rows_csv, sorted_rows_csv, chunk_rows=SORT_CHUNK_ROWS)
        _write_lineup_outputs_from_sorted(sorted_rows_csv, out_csv, within_limit_out_csv)
    finally:
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass
        if os.path.isfile(temp_rows_csv):
            try:
                os.remove(temp_rows_csv)
            except Exception:
                pass
        if os.path.isfile(sorted_rows_csv):
            try:
                os.remove(sorted_rows_csv)
            except Exception:
                pass
        for p in (input_db, input_db + "-wal", input_db + "-shm"):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    if return_df:
        try:
            return pd.read_csv(out_csv, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(out_csv, encoding="utf-8")

    return pd.DataFrame(columns=["channel", "case", "Vmax", "Vmin", "tag", "t_any_violation"])


def _default_inputs() -> tuple[list[str], list[str], str]:
    """config_paths 기반 기본 입력/출력 경로를 구성한다."""
    if _cfg_paths is None:
        return [], [], "GlobalRisk_Scenario_Vmax_Vmin_tag_duration.csv"
    if hasattr(_cfg_paths, "get_lineup_input_csvs"):
        in_csvs = list(_cfg_paths.get_lineup_input_csvs())
    else:
        in_csvs = [str(_cfg_paths.SECOND_CSV)]
    if hasattr(_cfg_paths, "get_csv_input_dirs"):
        voltage_dirs = list(_cfg_paths.get_csv_input_dirs())
    else:
        voltage_dirs = [str(_cfg_paths.CSV_DIR)]
    out_csv = os.path.join(str(_cfg_paths.OUT_DIR), "GlobalRisk_Scenario_Vmax_Vmin_tag_duration.csv")
    return in_csvs, voltage_dirs, out_csv


def main() -> None:
    """스크립트 단독 실행 진입점."""
    in_csvs, voltage_dirs, out_csv = _default_inputs()
    if not in_csvs or not voltage_dirs:
        raise RuntimeError("Default paths are not configured. Use run_lineup(...) as module call.")

    out_df = run_lineup(
        in_csv_paths=in_csvs,
        voltage_dir=voltage_dirs,
        out_csv=out_csv,
        cfg=LineupConfig(),
    )

    print("DONE")
    print(" - inputs:", in_csvs)
    print(" - output:", out_csv)
    print(" - rows:", len(out_df))


if __name__ == "__main__":
    main()
