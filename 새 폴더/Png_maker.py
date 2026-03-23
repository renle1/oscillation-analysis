# -*- coding: utf-8 -*-
"""
Png_maker.py (Python 3.12)

吏??湲곕뒫:
1) ?붿빟 CSV 湲곕컲 異쒕젰 (湲곗〈 湲곕뒫 ?좎?)
2) 耳?댁뒪紐??꾪꽣(CASE_FILTER_REGEX)濡??먰븯??耳?댁뒪留??좏깮
3) ?좏깮 耳?댁뒪??紐⑤뱺 梨꾨꼸 PNG 異쒕젰 ?듭뀡(PLOT_ALL_CHANNELS=True)
"""

import glob
import os
import re
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import pandas as pd


# =========================
# ?ㅼ젙 (?ш린留??섏젙)
# =========================
SUMMARY_CSV = r""  # ?뚯씪??議댁옱?섎㈃ summary 紐⑤뱶, ?놁쑝硫?raw-scan 紐⑤뱶
VOLTAGE_ROOT = r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\csv\KPGC_20260127_020013"
OUT_IMG_DIR = r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\code\tester"

# ?? r"(?i)L(?:INEONLY_)?(?:80_82|82_80)"
CASE_FILTER_REGEX = r""

# False: summary??channel(?먮뒗 best_channel)留?洹몃┝
# True : ?좏깮??case??紐⑤뱺 梨꾨꼸??梨꾨꼸蹂?PNG濡?洹몃┝
PLOT_ALL_CHANNELS = True

FIG_DPI = 160
TIME_COL_CANDIDATES = ("time", "Time", "TIME", "t", "T")
EXT = "png"

MAX_WORKERS = 4
MAX_IN_FLIGHT_FACTOR = 2
# =========================


def safe_filename(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def parse_channel_text(ch_text: str):
    s = str(ch_text).strip()
    chan_no = None
    chan_label = s

    if "|" in s:
        left, right = s.split("|", 1)
        if left.strip():
            chan_no = left.strip()
        chan_label = right.strip()

    bus = None
    m = re.search(r"\bBUS\s*([0-9]+)\b", chan_label, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bBUS([0-9]+)\b", chan_label, flags=re.IGNORECASE)
    if m:
        try:
            bus = int(m.group(1))
        except Exception:
            bus = None

    return chan_no, chan_label, bus


def find_voltage_column(df: pd.DataFrame, chan_label: str, bus: int | None) -> str | None:
    if chan_label in df.columns:
        return chan_label

    norm_target = re.sub(r"\s+", " ", str(chan_label)).strip().lower()
    for c in df.columns:
        norm_c = re.sub(r"\s+", " ", str(c)).strip().lower()
        if norm_target == norm_c:
            return c

    if bus is not None:
        pat = re.compile(rf"\bBUS\s*{bus}\b|\bBUS{bus}\b", flags=re.IGNORECASE)
        candidates = [c for c in df.columns if pat.search(str(c))]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            volt_first = [c for c in candidates if "VOLT" in str(c).upper()]
            return volt_first[0] if volt_first else candidates[0]

    return None


def resolve_case_path(case_value: str) -> str | None:
    p = str(case_value).strip()

    if os.path.isabs(p) and os.path.exists(p):
        return p

    p2 = os.path.join(VOLTAGE_ROOT, p)
    if os.path.exists(p2):
        return p2

    if not p.lower().endswith(".csv"):
        p3 = os.path.join(VOLTAGE_ROOT, p + "_Voltage.csv")
        if os.path.exists(p3):
            return p3

    return None


def discover_voltage_files(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        return []
    files = sorted(
        {
            os.path.abspath(p)
            for p in glob.glob(os.path.join(root_dir, "**", "*_Voltage.csv"), recursive=True)
            if os.path.isfile(p)
        }
    )
    if not files:
        files = sorted(
            {
                os.path.abspath(p)
                for p in glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
                if os.path.isfile(p)
            }
        )
    return files


def case_match(case_text: str) -> bool:
    pat = str(CASE_FILTER_REGEX).strip()
    if not pat:
        return True
    try:
        return re.search(pat, str(case_text)) is not None
    except re.error:
        # ?섎せ???뺢퇋?앹씠硫??꾪꽣瑜?臾댁떆?섏? ?딄퀬 ?ㅽ뙣 泥섎━
        return False


def build_tasks() -> tuple[list[tuple[str, str | None]], str]:
    tasks: list[tuple[str, str | None]] = []
    use_summary = bool(str(SUMMARY_CSV).strip()) and os.path.isfile(SUMMARY_CSV)

    if use_summary:
        summ = pd.read_csv(SUMMARY_CSV)
        if "case" not in summ.columns:
            raise ValueError("SUMMARY_CSV?먮뒗 'case' 而щ읆???덉뼱????")

        ch_col = None
        if "best_channel" in summ.columns:
            ch_col = "best_channel"
        elif "channel" in summ.columns:
            ch_col = "channel"

        if (not PLOT_ALL_CHANNELS) and (ch_col is None):
            raise ValueError("PLOT_ALL_CHANNELS=False?대㈃ SUMMARY_CSV??best_channel/channel???꾩슂??")

        if PLOT_ALL_CHANNELS:
            cases = []
            seen = set()
            for c in summ["case"].astype(str).tolist():
                c2 = c.strip()
                if not c2:
                    continue
                if not case_match(c2):
                    continue
                if c2 in seen:
                    continue
                seen.add(c2)
                cases.append(c2)
            tasks = [(c, None) for c in cases]
            mode = "summary-all-channels"
        else:
            for _, row in summ.iterrows():
                c = str(row["case"]).strip()
                if not c or not case_match(c):
                    continue
                tasks.append((c, str(row[ch_col])))
            mode = "summary-single-channel"

        return tasks, mode

    # raw-scan fallback
    files = discover_voltage_files(VOLTAGE_ROOT)
    if not files:
        raise FileNotFoundError(f"VOLTAGE_ROOT?먯꽌 CSV瑜?李얠? 紐삵븿: {VOLTAGE_ROOT}")

    files = [p for p in files if case_match(os.path.basename(p))]
    tasks = [(p, None if PLOT_ALL_CHANNELS else "") for p in files]
    mode = "raw-all-channels" if PLOT_ALL_CHANNELS else "raw-single-channel(梨꾨꼸?뺣낫?놁쓬)"
    return tasks, mode


def worker_plot_one(case_value, channel_text, idx: int, idx_width: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_path = resolve_case_path(case_value)
    if case_path is None:
        return (False, f"[SKIP] ?뚯씪 ?놁쓬: {case_value}")

    try:
        df = pd.read_csv(case_path)
    except Exception as e:
        return (False, f"[ERROR] read fail: {os.path.basename(case_path)} | {type(e).__name__}: {e}")

    if df.shape[1] < 2:
        return (False, f"[SKIP] 梨꾨꼸 ?놁쓬: {os.path.basename(case_path)}")

    tcol = find_time_column(df)
    try:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    except Exception:
        return (False, f"[SKIP] time parse fail: {os.path.basename(case_path)}")

    case_base = os.path.splitext(os.path.basename(case_path))[0]
    all_channel_mode = (channel_text is None) or (str(channel_text).strip() == "")

    if all_channel_mode:
        ch_cols = [c for c in df.columns if c != tcol]
        if not ch_cols:
            return (False, f"[SKIP] 梨꾨꼸 ?놁쓬: {os.path.basename(case_path)}")

        case_out_dir = os.path.join(OUT_IMG_DIR, safe_filename(case_base))
        os.makedirs(case_out_dir, exist_ok=True)

        saved = 0
        for c_idx, ch in enumerate(ch_cols, start=1):
            try:
                v = pd.to_numeric(df[ch], errors="coerce").to_numpy()
                plt.figure()
                plt.plot(t, v)
                plt.xlabel(tcol)
                plt.ylabel("Value")
                plt.title(f"{case_base}\n{ch}")
                out_name = safe_filename(f"{c_idx:03d}__{case_base}__{ch}.{EXT}")
                out_path = os.path.join(case_out_dir, out_name)
                plt.tight_layout()
                plt.savefig(out_path, dpi=FIG_DPI)
                plt.close()
                saved += 1
            except Exception:
                plt.close("all")
                continue

        return (True, f"[OK] {case_base} | channels={len(ch_cols)} | saved={saved}")

    # single-channel 紐⑤뱶(summary 湲곕컲)
    chan_no, chan_label, bus = parse_channel_text(str(channel_text))
    vcol = find_voltage_column(df, chan_label, bus)
    if vcol is None:
        volt_cols = [c for c in df.columns if "VOLT" in str(c).upper()]
        if len(volt_cols) == 1:
            vcol = volt_cols[0]
        else:
            return (False, f"[SKIP] 梨꾨꼸 紐?李얠쓬: {os.path.basename(case_path)} / target='{chan_label}'")

    try:
        v = pd.to_numeric(df[vcol], errors="coerce").to_numpy()
        plt.figure()
        plt.plot(t, v)
        plt.xlabel(tcol)
        plt.ylabel("Voltage (pu)")
        plt.title(f"{case_base}\n{chan_no + ' | ' if chan_no else ''}{vcol}")
        bus_tag = f"BUS{bus}" if bus is not None else "BUS?"
        ch_tag = f"CH{chan_no}" if chan_no else "CHNA"
        order_tag = f"{idx:0{idx_width}d}"
        out_name = safe_filename(f"{order_tag}__{case_base}__{bus_tag}__{ch_tag}.{EXT}")
        out_path = os.path.join(OUT_IMG_DIR, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=FIG_DPI)
        plt.close()
        return (True, f"[OK] {out_name}")
    except Exception as e:
        plt.close("all")
        return (False, f"[ERROR] plot fail: {os.path.basename(case_path)} | {type(e).__name__}: {e}")


def main_png() -> None:
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    tasks, mode = build_tasks()
    if not tasks:
        raise ValueError("議곌굔??留욌뒗 case媛 ?놁쓬. SUMMARY_CSV/CASE_FILTER_REGEX瑜??뺤씤?섏꽭??")

    print(f"[MODE] {mode}")
    print(f"[TASKS] {len(tasks)}")

    total = len(tasks)
    idx_width = len(str(total)) if total > 0 else 1
    max_in_flight = max(MAX_WORKERS, MAX_WORKERS * int(MAX_IN_FLIGHT_FACTOR))

    done = 0
    ok = 0

    try:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            task_iter = iter(enumerate(tasks, start=1))
            fut_map = {}

            def _submit_next() -> bool:
                try:
                    i, (case_value, ch_text) = next(task_iter)
                except StopIteration:
                    return False
                fut = ex.submit(worker_plot_one, case_value, ch_text, i, idx_width)
                fut_map[fut] = i
                return True

            while len(fut_map) < max_in_flight and _submit_next():
                pass

            while fut_map:
                done_set, _ = wait(set(fut_map.keys()), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    _ = fut_map.pop(fut, None)
                    done += 1
                    success, msg = fut.result()
                    if success:
                        ok += 1
                    print(f"[PROGRESS] {done}/{total} | ok={ok} | {msg}")
                while len(fut_map) < max_in_flight and _submit_next():
                    pass
    except (PermissionError, OSError) as e:
        print(f"[WARN] parallel ?ㅽ뙣 -> ?⑥씪 ?꾨줈?몄뒪濡?吏꾪뻾: {type(e).__name__}: {e}")
        for i, (case_value, ch_text) in enumerate(tasks, start=1):
            done += 1
            success, msg = worker_plot_one(case_value, ch_text, i, idx_width)
            if success:
                ok += 1
            print(f"[PROGRESS] {done}/{total} | ok={ok} | {msg}")

    print(f"\nDONE: {ok}/{total} tasks | out={OUT_IMG_DIR}")


if __name__ == "__main__":
    main_png()

