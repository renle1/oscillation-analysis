# -*- coding: utf-8 -*-
"""
진동 후보 2차 필터 모듈(no-first).

역할:
- 1차 필터 없이 입력 전압 CSV에서 바로 2차 평가를 수행한다.
- run_second_filter_no_first(...)가 전체 처리의 진입점이다.

구현 요약:
- SQLite 임시 DB와 bounded in-flight 워커 패턴으로 메모리 사용량을 제어한다.
- 케이스/채널 단위 예외를 격리해 일부 실패가 전체 실행을 중단시키지 않도록 한다.
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
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import config_paths_no_first as cfg
except Exception:
    import config_paths as cfg


OUT_CSV = os.path.join(cfg.OUT_DIR, "kpg_second_filter_all_cases.csv")
OUT_LOG_TXT = os.path.join(cfg.OUT_DIR, "kpg_second_filter_all_cases_log.txt")
OUT_CASE_SUMMARY = os.path.join(cfg.OUT_DIR, "kpg_second_filter_all_cases_summary.csv")

WRITE_SECOND_LOG = bool(getattr(cfg, "WRITE_SECOND_LOG", False))
WRITE_SECOND_SUMMARY = bool(getattr(cfg, "WRITE_SECOND_SUMMARY", False))

MAX_WORKERS = 6
MAX_IN_FLIGHT_FACTOR = 2
MAX_TASKS_PER_CHILD = 200
RESULT_FLUSH_ROWS = 2000
SORT_CHUNK_ROWS = 50000
FIXED_DT_SEC = float(getattr(cfg, "FIXED_DT_SEC", 1.0 / 120.0))

SKIP_SEC = 2.0
WIN_SEC = 8.0
BASE_SEC = 6.0
RMS_WIN_SEC = 0.25
TAIL_FRAC = 0.30
MIN_SIGN_CHANGES = 3
DAMPED_RATIO_CUT = 0.70
EPS_AMP = 1e-6
SCORE_CUT = 1e-6
LOG_LOW_MAX_PER_CASE = 12
USE_TOPK_PER_CASE = False
TOPK_PER_CASE = 30
EXCLUDE_BUS_NOS = []


def _new_pool(max_workers: int) -> ProcessPoolExecutor:
    """프로세스 풀 생성. 지원 환경이면 max_tasks_per_child를 함께 사용한다."""
    kwargs = {"max_workers": int(max_workers)}
    if int(MAX_TASKS_PER_CHILD) > 0:
        kwargs["max_tasks_per_child"] = int(MAX_TASKS_PER_CHILD)
    try:
        return ProcessPoolExecutor(**kwargs)
    except TypeError:
        kwargs.pop("max_tasks_per_child", None)
        return ProcessPoolExecutor(**kwargs)


def _safe_float(v, default: float = float("-inf")) -> float:
    """안전한 float 변환 함수. 변환 실패/비유한값이면 default를 반환한다."""
    try:
        fv = float(v)
    except Exception:
        return default
    if not np.isfinite(fv):
        return default
    return fv


def _external_sort_csv_numeric_desc(in_csv: str, key_col: str, *, chunk_rows: int = SORT_CHUNK_ROWS) -> None:
    """대용량 CSV를 청크 정렬/병합으로 key_col 내림차순 정렬한다."""
    in_csv = os.path.abspath(in_csv)
    if not os.path.isfile(in_csv):
        return

    parent = os.path.dirname(in_csv) or "."
    temp_paths: list[str] = []
    out_sorted = tempfile.NamedTemporaryFile(
        prefix="__second_sorted_",
        suffix=".csv",
        dir=parent,
        delete=False,
    ).name

    header_cols = []
    try:
        for chunk in pd.read_csv(in_csv, chunksize=int(chunk_rows), encoding="utf-8-sig"):
            if chunk is None or len(chunk) == 0:
                continue
            if not header_cols:
                header_cols = [str(c) for c in chunk.columns]
            if key_col not in chunk.columns:
                chunk[key_col] = np.nan
            chunk[key_col] = pd.to_numeric(chunk[key_col], errors="coerce")
            chunk = chunk.sort_values([key_col], ascending=[False], kind="mergesort")

            part = tempfile.NamedTemporaryFile(
                prefix="__second_part_",
                suffix=".csv",
                dir=parent,
                delete=False,
            ).name
            chunk.to_csv(part, index=False, encoding="utf-8")
            temp_paths.append(part)

        if not temp_paths:
            if not header_cols:
                try:
                    hdf = pd.read_csv(in_csv, nrows=0, encoding="utf-8-sig")
                    header_cols = [str(c) for c in hdf.columns]
                except Exception:
                    header_cols = []
            pd.DataFrame(columns=header_cols).to_csv(out_sorted, index=False, encoding="utf-8-sig")
            os.replace(out_sorted, in_csv)
            return

        file_handles = []
        readers = []
        heap = []
        try:
            for idx, path in enumerate(temp_paths):
                fh = open(path, "r", newline="", encoding="utf-8")
                rd = csv.DictReader(fh)
                file_handles.append(fh)
                readers.append(rd)
                row = next(rd, None)
                if row is None:
                    continue
                kval = _safe_float(row.get(key_col, None), default=float("-inf"))
                heapq.heappush(heap, (-kval, idx, row))

            with open(out_sorted, "w", newline="", encoding="utf-8-sig") as fw:
                writer = csv.DictWriter(fw, fieldnames=header_cols)
                writer.writeheader()
                while heap:
                    _, idx, row = heapq.heappop(heap)
                    writer.writerow(row)
                    nxt = next(readers[idx], None)
                    if nxt is None:
                        continue
                    kval = _safe_float(nxt.get(key_col, None), default=float("-inf"))
                    heapq.heappush(heap, (-kval, idx, nxt))
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

        os.replace(out_sorted, in_csv)
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        if os.path.isfile(out_sorted):
            try:
                os.remove(out_sorted)
            except Exception:
                pass


def infer_time_col(df: pd.DataFrame) -> str:
    """시간 컬럼명을 추정한다."""
    for c in df.columns:
        if "time" in str(c).lower():
            return c
    return df.columns[0]


def to_float_np(s: pd.Series) -> np.ndarray:
    """시리즈를 숫자형 numpy 배열로 변환한다."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def duration_over_mask(t: np.ndarray, mask: np.ndarray) -> float:
    """마스크가 True인 구간의 시간 길이 합을 계산한다."""
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    return float(np.sum(dt[mask[:-1]]))


def tail_sign_changes(x_tail: np.ndarray, eps: float) -> int:
    """tail 구간에서 유효 신호의 부호 변화 횟수를 계산한다."""
    xt = np.asarray(x_tail, dtype=float)
    xt = xt[np.isfinite(xt)]
    if xt.size < 10:
        return 0
    xt = xt[np.abs(xt) >= eps]
    if xt.size < 10:
        return 0
    sgn = np.sign(xt)
    return int(np.sum(sgn[1:] * sgn[:-1] < 0))


def parse_bus_no_from_colname(col: str):
    """채널 컬럼명에서 BUS 번호를 추출한다."""
    s = str(col)
    m = re.search(r"\bBUS\s*([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bBUS([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bVOLT\s*([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bVOLTAGE\s*([0-9]+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def score_one_channel_equiv(tw: np.ndarray, vw: np.ndarray, dt: float, t0: float, t1: float):
    """
    2차 필터 규칙으로 채널 점수 `(score, A_tail, D_tail, reason)`를 계산한다.

    계산식 요약:
    - 기준 전압: `base = median(vw[tw <= tw[0] + BASE_SEC])`
    - 중심화 신호: `x = vw - base`
    - 추세 제거:
      `k = odd(max(3, round(BASE_SEC / dt)))`
      `trend = rolling_median(x, window=k)`
      `residual = x - trend`
    - tail 구간: `tw >= t1 - TAIL_FRAC * (t1 - t0)`
    - 진동성 게이트: `sc = sign_changes(residual_tail, eps=EPS_AMP)`, `sc < MIN_SIGN_CHANGES`면 제외
    - RMS:
      `win_n = max(1, round(RMS_WIN_SEC / dt))`
      `rms = sqrt(rolling_mean(residual^2, window=win_n))`
    - 크기 지표:
      `A_head = Q80(rms_head)`, `A_tail = Q80(rms_tail)`
      `ratio = A_tail / (A_head + EPS_AMP)`
      `ratio <= DAMPED_RATIO_CUT`이면 감쇠로 제외
    - 지속시간 지표:
      `thr = Q70(rms_tail)`
      `mask_hi = isfinite(rms) & (rms >= thr) & mtail`
      `D_tail = sum(dt_i for i where mask_hi[i] is True)`
    - 최종 점수: `score = (A_tail^2) * (D_tail^3)`
    """
    tb1 = tw[0] + float(BASE_SEC)
    mb = tw <= tb1
    if np.count_nonzero(mb) < 5:
        mb = np.arange(tw.size) < max(5, int(0.25 * tw.size))
    base = float(np.nanmedian(vw[mb]))
    x = vw - base

    k = int(max(3, round(float(BASE_SEC) / float(dt))))
    if (k % 2) == 0:
        k += 1
    trend = pd.Series(x).rolling(window=k, center=True, min_periods=1).median().to_numpy(dtype=float)
    residual = x - trend

    tail_start_t = float(t1) - float(TAIL_FRAC) * float(t1 - t0)
    mtail = tw >= tail_start_t

    sc = tail_sign_changes(residual[mtail], eps=float(EPS_AMP))
    if sc < int(MIN_SIGN_CHANGES):
        return 0.0, np.nan, 0.0, "no_osc_sign_change"

    win_n = int(max(1, round(float(RMS_WIN_SEC) / float(dt))))
    rms = pd.Series(residual * residual).rolling(window=win_n, min_periods=win_n).mean()
    rms = np.sqrt(rms.to_numpy(dtype=float))

    head_end_t = float(t0) + (1.0 - float(TAIL_FRAC)) * float(t1 - t0)
    mhead = tw <= head_end_t
    rms_head = rms[mhead]
    rms_tail = rms[mtail]

    A_head = float(np.nanquantile(rms_head, 0.80))
    A_tail = float(np.nanquantile(rms_tail, 0.80))

    ratio = float(A_tail / (A_head + float(EPS_AMP)))
    if ratio <= float(DAMPED_RATIO_CUT):
        return 0.0, (A_tail if np.isfinite(A_tail) else np.nan), 0.0, "excluded_damped"

    thr = float(np.nanquantile(rms_tail, 0.70))
    mask_hi = np.isfinite(rms) & (rms >= thr) & mtail
    D_tail = duration_over_mask(tw, mask_hi)

    if (not np.isfinite(A_tail)) or (A_tail <= 0.0) or (D_tail <= 0.0):
        return 0.0, (A_tail if np.isfinite(A_tail) else np.nan), float(D_tail), "weak_tail"

    score = (A_tail ** 2.0) * (D_tail ** 3.0)
    if not np.isfinite(score):
        score = 0.0

    return float(score), float(A_tail), float(D_tail), "ok"


def _empty_case_summary(case_val: str, status_reason: str) -> dict:
    """계산 불가/실패 case용 기본 summary row를 만든다."""
    return {
        "case": str(case_val),
        "best_score": 0.0,
        "best_channel": "",
        "total_scanned": 0,
        "ok": 0,
        "kept_ok": 0,
        "cut_low_ok": 0,
        "damped": 0,
        "noosc": 0,
        "has_other_risky": 0,
        "status_reason": str(status_reason),
    }


def process_one_case(case_val: str, vcsv: str, exclude_set: set):
    """단일 case의 채널 점수를 계산해 kept rows와 요약 정보를 반환한다."""
    kept = []
    log_lines = []

    try:
        vdf = pd.read_csv(vcsv, encoding="utf-8-sig")
        if vdf.shape[0] < 2 or vdf.shape[1] < 2:
            reason = "csv_too_small"
            log_lines.append(f"[CASE_SKIP] {case_val} | {reason} | file={vcsv}")
            return kept, log_lines, _empty_case_summary(case_val, reason)

        tcol = infer_time_col(vdf)
        t_all = to_float_np(vdf[tcol])
        t_valid = t_all[np.isfinite(t_all)]
        if t_valid.size == 0:
            reason = "time_invalid"
            log_lines.append(f"[CASE_SKIP] {case_val} | {reason} | file={vcsv}")
            return kept, log_lines, _empty_case_summary(case_val, reason)

        t0 = float(t_valid[0]) + float(SKIP_SEC)
        t1 = t0 + float(WIN_SEC)
        mwin = (t_all >= t0) & (t_all <= t1)

        tw = t_all[mwin]
        if np.count_nonzero(np.isfinite(tw)) < 5:
            reason = "window_too_small"
            log_lines.append(f"[CASE_SKIP] {case_val} | {reason} | file={vcsv}")
            return kept, log_lines, _empty_case_summary(case_val, reason)

        dt = float(FIXED_DT_SEC)
        if (not np.isfinite(dt)) or (dt <= 0):
            reason = "time_invalid"
            log_lines.append(f"[CASE_SKIP] {case_val} | {reason} | file={vcsv}")
            return kept, log_lines, _empty_case_summary(case_val, reason)

        low_ok = []
        n_total = 0
        n_ok = 0
        n_ok_kept = 0
        n_damped = 0
        n_noosc = 0
        n_channel_fail = 0
        best_score = -1.0
        best_channel = ""

        for c in vdf.columns:
            if c == tcol:
                continue
            bus_no = parse_bus_no_from_colname(c)
            if bus_no is not None and exclude_set and (int(bus_no) in exclude_set):
                continue
            n_total += 1

            v_all = to_float_np(vdf[c])
            vw = v_all[mwin]
            try:
                score, A_tail, D_tail, reason = score_one_channel_equiv(tw, vw, dt, t0, t1)
            except Exception as e:
                n_channel_fail += 1
                if n_channel_fail <= int(LOG_LOW_MAX_PER_CASE):
                    log_lines.append(
                        f"  CHANNEL_FAIL | case={case_val} | channel={c} | {type(e).__name__}: {e}"
                    )
                continue

            if reason == "ok":
                n_ok += 1
                if score > best_score:
                    best_score = float(score)
                    best_channel = str(c)
                if float(score) >= float(SCORE_CUT):
                    kept.append(
                        {
                            "case": str(case_val),
                            "channel": str(c),
                            "score": float(score),
                            "A_tail": float(A_tail) if np.isfinite(A_tail) else np.nan,
                            "D_tail": float(D_tail),
                            "status_reason": "ok",
                        }
                    )
                    n_ok_kept += 1
                else:
                    low_ok.append((float(score), str(c)))
            else:
                if reason == "excluded_damped":
                    n_damped += 1
                elif reason == "no_osc_sign_change":
                    n_noosc += 1

        low_ok.sort(key=lambda x: x[0], reverse=True)
        cut_cnt = len(low_ok)
        log_lines.append(
            f"[CASE] {case_val} | total_scanned={n_total} | ok={n_ok} | kept(ok>=cut)={n_ok_kept} | "
            f"cut_low_ok={cut_cnt} | damped={n_damped} | noosc={n_noosc} | channel_fail={n_channel_fail} | best={best_score:.3e}"
        )

        if cut_cnt > 0:
            topm = low_ok[: int(LOG_LOW_MAX_PER_CASE)]
            for sc, ch in topm:
                log_lines.append(f"  CUT_LOW_OK | case={case_val} | channel={ch} | score={sc:.3e} < {SCORE_CUT:.3e}")
            if cut_cnt > int(LOG_LOW_MAX_PER_CASE):
                log_lines.append(f"  ... ({cut_cnt - int(LOG_LOW_MAX_PER_CASE)} more low-ok channels omitted)")

        summary = {
            "case": str(case_val),
            "best_score": float(best_score) if best_score >= 0 else 0.0,
            "best_channel": str(best_channel),
            "total_scanned": int(n_total),
            "ok": int(n_ok),
            "kept_ok": int(n_ok_kept),
            "cut_low_ok": int(cut_cnt),
            "damped": int(n_damped),
            "noosc": int(n_noosc),
            "has_other_risky": int(n_ok_kept >= 2),
            "status_reason": "ok" if n_channel_fail == 0 else "partial_channel_fail",
        }

        if USE_TOPK_PER_CASE and len(kept) > int(TOPK_PER_CASE):
            kept.sort(key=lambda r: r["score"], reverse=True)
            kept = kept[: int(TOPK_PER_CASE)]

        return kept, log_lines, summary
    except Exception as e:
        reason = "worker_issue:%s" % type(e).__name__
        log_lines.append(f"[CASE_FAIL] {case_val} | {reason} | file={vcsv} | msg={e}")
        return [], log_lines, _empty_case_summary(case_val, reason)


def _default_voltage_dirs() -> list[str]:
    """기본 전압 입력 디렉터리 목록을 반환한다."""
    if hasattr(cfg, "get_csv_input_dirs"):
        return list(cfg.get_csv_input_dirs())
    return [os.path.abspath(cfg.CSV_DIR)]


def _resolve_voltage_dirs(voltage_dirs: Sequence[str] | str | None) -> list[str]:
    """전압 입력 경로를 절대경로 리스트로 정규화한다."""
    if voltage_dirs is None:
        items = _default_voltage_dirs()
    elif isinstance(voltage_dirs, str):
        items = [voltage_dirs]
    else:
        items = [str(x) for x in voltage_dirs]

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


def _discover_voltage_files(voltage_dirs: Sequence[str]) -> list[str]:
    """입력 디렉터리들에서 전압 CSV를 수집해 절대경로 목록으로 반환한다."""
    files = []
    for d in voltage_dirs:
        hits = sorted(glob.glob(os.path.join(d, "*_Voltage.csv")))
        if not hits:
            hits = sorted(glob.glob(os.path.join(d, "*.csv")))
        files.extend(hits)

    out = []
    seen = set()
    for p in files:
        ap = os.path.abspath(p)
        if ap in seen or not os.path.isfile(ap):
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _prepare_case_jobs_db(voltage_dirs: Sequence[str], db_path: str) -> tuple[sqlite3.Connection, int]:
    """no-first 모드용 case_jobs(SQLite) 작업 큐를 생성한다."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS case_jobs ("
        "case_val TEXT PRIMARY KEY, "
        "vcsv TEXT NOT NULL)"
    )
    conn.commit()

    files = _discover_voltage_files(voltage_dirs)
    rows = [(p, p) for p in files]
    if rows:
        conn.executemany("INSERT OR IGNORE INTO case_jobs(case_val, vcsv) VALUES (?, ?)", rows)
        conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM case_jobs")
    total_cases = int(cur.fetchone()[0])
    cur.close()
    return conn, total_cases


def _run_pipeline(
    *,
    voltage_dirs: Sequence[str],
    out_csv: str,
    max_workers_override: int | None = None,
) -> str:
    """no-first 2차 필터 메인 파이프라인을 실행하고 결과 경로를 반환한다."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if WRITE_SECOND_LOG:
        os.makedirs(os.path.dirname(OUT_LOG_TXT) or ".", exist_ok=True)
    if WRITE_SECOND_SUMMARY:
        os.makedirs(os.path.dirname(OUT_CASE_SUMMARY) or ".", exist_ok=True)

    if max_workers_override is not None:
        max_workers = int(max_workers_override)
    elif MAX_WORKERS:
        max_workers = int(MAX_WORKERS)
    else:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    max_in_flight = max(max_workers, max_workers * int(MAX_IN_FLIGHT_FACTOR))
    print("workers:", max_workers, "| in_flight:", max_in_flight)
    if int(MAX_TASKS_PER_CHILD) > 0:
        print("max_tasks_per_child:", int(MAX_TASKS_PER_CHILD))
    else:
        print("max_tasks_per_child: disabled")

    result_cols = ["case", "channel", "score", "A_tail", "D_tail", "status_reason"]
    summary_cols = [
        "case",
        "best_score",
        "best_channel",
        "total_scanned",
        "ok",
        "kept_ok",
        "cut_low_ok",
        "damped",
        "noosc",
        "has_other_risky",
        "status_reason",
    ]

    pd.DataFrame(columns=result_cols).to_csv(out_csv, index=False, encoding="utf-8-sig")
    if WRITE_SECOND_SUMMARY:
        pd.DataFrame(columns=summary_cols).to_csv(OUT_CASE_SUMMARY, index=False, encoding="utf-8-sig")
    if WRITE_SECOND_LOG:
        with open(OUT_LOG_TXT, "w", encoding="utf-8") as f_init:
            f_init.write("")

    result_buf = []
    summary_buf = []
    rows_saved = 0
    summary_saved = 0
    done = 0

    exclude_set = set(EXCLUDE_BUS_NOS) if isinstance(EXCLUDE_BUS_NOS, (list, tuple, set)) else set()

    def _flush_results() -> None:
        nonlocal result_buf, rows_saved
        if not result_buf:
            return
        bdf = pd.DataFrame(result_buf)
        for c in result_cols:
            if c not in bdf.columns:
                bdf[c] = np.nan
        bdf = bdf[result_cols]
        bdf.to_csv(out_csv, mode="a", index=False, header=False, encoding="utf-8")
        rows_saved += len(bdf)
        result_buf = []

    def _flush_summary() -> None:
        nonlocal summary_buf, summary_saved
        if (not WRITE_SECOND_SUMMARY) or (not summary_buf):
            return
        sdf = pd.DataFrame(summary_buf)
        for c in summary_cols:
            if c not in sdf.columns:
                sdf[c] = np.nan
        sdf = sdf[summary_cols]
        sdf.to_csv(OUT_CASE_SUMMARY, mode="a", index=False, header=False, encoding="utf-8")
        summary_saved += len(sdf)
        summary_buf = []

    job_db = os.path.join(cfg.OUT_DIR, "__second_nofirst_jobs_%d.sqlite" % os.getpid())
    job_conn = None
    log_fh = open(OUT_LOG_TXT, "a", encoding="utf-8") if WRITE_SECOND_LOG else None
    total_cases = 0

    try:
        job_conn, total_cases = _prepare_case_jobs_db(voltage_dirs, job_db)
        print("cases_selected:", total_cases)
        if total_cases == 0:
            raise FileNotFoundError("No voltage CSV files found in input directories.")

        with _new_pool(max_workers=max_workers) as ex:
            case_cur = job_conn.cursor()
            case_cur.execute("SELECT case_val, vcsv FROM case_jobs")
            case_iter = iter(case_cur)
            fut_map = {}

            def _submit_next() -> bool:
                try:
                    case_val, vcsv = next(case_iter)
                except StopIteration:
                    return False
                case_s = str(case_val)
                vcsv_s = str(vcsv)
                fut = ex.submit(process_one_case, case_s, vcsv_s, exclude_set)
                fut_map[fut] = (case_s, vcsv_s)
                return True

            while len(fut_map) < max_in_flight and _submit_next():
                pass

            while fut_map:
                done_set, _ = wait(set(fut_map.keys()), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    case_val, vcsv = fut_map.pop(fut)
                    try:
                        kept, log_lines, summary = fut.result()
                    except Exception as e:
                        reason = "worker_issue:%s" % type(e).__name__
                        kept = []
                        log_lines = [f"[CASE_FAIL] {case_val} | {reason} | file={vcsv} | msg={e}"]
                        summary = _empty_case_summary(case_val, reason)

                    result_buf.extend(kept)
                    if len(result_buf) >= int(RESULT_FLUSH_ROWS):
                        _flush_results()

                    if log_fh is not None:
                        for line in log_lines:
                            log_fh.write(line + "\n")

                    if WRITE_SECOND_SUMMARY:
                        summary_buf.append(summary)
                        if len(summary_buf) >= int(RESULT_FLUSH_ROWS):
                            _flush_summary()

                    done += 1
                    if (done % 50 == 0) or (done == total_cases):
                        print("progress:", done, "/", total_cases, "| rows_saved:", rows_saved)

                while len(fut_map) < max_in_flight and _submit_next():
                    pass
    finally:
        _flush_results()
        _flush_summary()
        if log_fh is not None:
            log_fh.close()
        if job_conn is not None:
            try:
                job_conn.close()
            except Exception:
                pass
        for p in (job_db, job_db + "-wal", job_db + "-shm"):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    if rows_saved > 0:
        _external_sort_csv_numeric_desc(out_csv, "score", chunk_rows=SORT_CHUNK_ROWS)
    else:
        pd.DataFrame(columns=result_cols).to_csv(out_csv, index=False, encoding="utf-8-sig")

    if WRITE_SECOND_SUMMARY:
        if summary_saved > 0:
            _external_sort_csv_numeric_desc(OUT_CASE_SUMMARY, "best_score", chunk_rows=SORT_CHUNK_ROWS)
        else:
            pd.DataFrame(columns=summary_cols).to_csv(OUT_CASE_SUMMARY, index=False, encoding="utf-8-sig")

    print("DONE")
    print("cases_processed:", total_cases)
    print("rows_saved(score>=cut):", rows_saved)
    print("saved_csv:", out_csv)
    print("saved_log:", OUT_LOG_TXT if WRITE_SECOND_LOG else "disabled")
    print("saved_summary:", OUT_CASE_SUMMARY if WRITE_SECOND_SUMMARY else "disabled")
    return os.path.abspath(out_csv)


def run_second_filter_no_first(
    *,
    reload_source: bool = False,
    max_workers: int | None = None,
    voltage_dirs: Sequence[str] | str | None = None,
    out_csv: str | None = None,
) -> str:
    """외부 호출용 엔트리포인트. no-first 2차 필터를 실행한다."""
    _ = reload_source
    dirs = _resolve_voltage_dirs(voltage_dirs)
    if not dirs:
        raise FileNotFoundError("No voltage input directories were found.")

    cfg.ensure_dirs(create_input_dir=False)
    out_path = os.path.abspath(out_csv) if out_csv else os.path.abspath(OUT_CSV)
    result = _run_pipeline(voltage_dirs=dirs, out_csv=out_path, max_workers_override=max_workers)
    if not os.path.isfile(result):
        raise FileNotFoundError("2nd(no-first) output not found after run: %s" % result)
    return result


def main() -> None:
    """스크립트 직접 실행 진입점."""
    out_csv = run_second_filter_no_first()
    print("DONE")
    print("saved_csv:", out_csv)


if __name__ == "__main__":
    main()
