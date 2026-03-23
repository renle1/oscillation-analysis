# -*- coding: utf-8 -*-
"""
make_png.py (Python 3.12)

Summary CSV -> per-case voltage plot -> PNG export (parallel)
- SUMMARY_CSV?먮뒗 理쒖냼??'case' 而щ읆???덉뼱????- 梨꾨꼸 而щ읆? 'best_channel'???덉쑝硫?洹멸구 ?곌퀬, ?놁쑝硫?'channel'???ъ슜
?쇰떒 ?닿굅濡?媛??
"""

import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================
# ?ㅼ젙 (?ш린留??섏젙)
# =========================
SUMMARY_CSV = r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\RESULTS\time_domain.csv"
VOLTAGE_ROOT = r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\csv\KPGC_20260127_020013"
OUT_IMG_DIR = r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\kpg_png_grah\new_algorithm_ver2"

FIG_DPI = 160
TIME_COL_CANDIDATES = ("time", "Time", "TIME", "t", "T")
EXT = "png"

MAX_WORKERS = 6   # ?붿뒪??CPU 蹂닿퀬 2~6 沅뚯옣
# =========================


def safe_filename(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_channel_text(ch_text: str):
    """
    ?쒓뎅???ㅻ챸:
    - best_channel 臾몄옄?댁씠 ?덉쟾 ?щ㎎("074 | VOLT ...")?대뱺,
      洹몃깷 而щ읆紐?"VOLT 59 [BUS59 ...]")?대뱺 ????泥섎━
    - 諛섑솚: (chan_no_str_or_None, chan_label_str, bus_int_or_None)
    """
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


def resolve_case_path(case_value: str) -> str | None:
    """
    ?쒓뎅???ㅻ챸:
    - summary??case媛
      1) ?덈?寃쎈줈
      2) VOLTAGE_ROOT 諛묒쓽 ?곷?寃쎈줈/?뚯씪紐?      3) ?뺤옣???놁씠 base留??덈뒗 寃쎌슦 (_Voltage.csv ?먮룞 遺李?
      瑜?紐⑤몢 吏??    """
    p = str(case_value).strip()

    if os.path.isabs(p) and os.path.exists(p):
        return p

    p2 = os.path.join(VOLTAGE_ROOT, p)
    if os.path.exists(p2):
        return p2

    # ?뺤옣???녿뒗 base??寃쎌슦: _Voltage.csv 遺숈뿬蹂닿린
    if not p.lower().endswith(".csv"):
        p3 = os.path.join(VOLTAGE_ROOT, p + "_Voltage.csv")
        if os.path.exists(p3):
            return p3

    return None


def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def find_voltage_column(df: pd.DataFrame, chan_label: str, bus: int | None) -> str | None:
    """
    ?쒓뎅???ㅻ챸:
    - 1) chan_label??df 而щ읆??洹몃?濡??덉쑝硫?洹멸구 ?ъ슜 (理쒖슦??
    - 2) 怨듬갚 ?뺢퇋?????숈씪?섎㈃ ?ъ슜
    - 3) BUS 踰덊샇媛 ?덉쑝硫?BUS 留ㅼ묶?쇰줈 ?꾨낫 李얘린
    """
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
        elif len(candidates) > 1:
            volt_first = [c for c in candidates if "VOLT" in str(c).upper()]
            return volt_first[0] if volt_first else candidates[0]

    return None


def worker_plot_one(case_value, channel_text, idx: int, idx_width: int):
    """
    ?뚯빱 ?꾨줈?몄뒪?먯꽌 ?ㅽ뻾???⑥닔
    諛섑솚: (ok:bool, msg:str)
    """
    # ?뚯빱 ?덉뿉??matplotlib import (異⑸룎/珥덇린???댁뒋 以꾩씠湲?
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_path = resolve_case_path(case_value)
    if case_path is None:
        return (False, f"[SKIP] ?뚯씪 ?놁쓬: {case_value}")

    chan_no, chan_label, bus = parse_channel_text(channel_text)

    try:
        df = pd.read_csv(case_path)

        tcol = find_time_column(df)
        vcol = find_voltage_column(df, chan_label, bus)

        if vcol is None:
            # 留덉?留?fallback: voltage 鍮꾩듂??而щ읆 ?섎굹?쇰룄 ?≪븘蹂닿린(?덈Т 怨듦꺽?곸씠硫?吏?뚮룄 ??
            volt_cols = [c for c in df.columns if "VOLT" in str(c).upper()]
            if len(volt_cols) == 1:
                vcol = volt_cols[0]
            else:
                return (False, f"[SKIP] 梨꾨꼸 紐?李얠쓬: {os.path.basename(case_path)} / target='{chan_label}'")

        t = df[tcol].values
        v = df[vcol].values

        base = os.path.splitext(os.path.basename(case_path))[0]
        title = f"{base}\n{chan_no + ' | ' if chan_no else ''}{vcol}"

        plt.figure()
        plt.plot(t, v)
        plt.xlabel(tcol)
        plt.ylabel("Voltage (pu)")
        plt.title(title)

        bus_tag = f"BUS{bus}" if bus is not None else "BUS?"
        ch_tag = f"CH{chan_no}" if chan_no else "CHNA"
        order_tag = f"{idx:0{idx_width}d}"
        out_name = safe_filename(f"{order_tag}__{base}__{bus_tag}__{ch_tag}.{EXT}")
        out_path = os.path.join(OUT_IMG_DIR, out_name)

        plt.tight_layout()
        plt.savefig(out_path, dpi=FIG_DPI)
        plt.close()

        return (True, f"[OK] {out_name}")

    except Exception as e:
        return (False, f"[ERROR] {os.path.basename(case_path)} : {type(e).__name__}: {e}")


def main_png():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    summ = pd.read_csv(SUMMARY_CSV)

    if "case" not in summ.columns:
        raise ValueError("SUMMARY_CSV?먮뒗 'case' 而щ읆???덉뼱????")

    # best_channel ?곗꽑, ?놁쑝硫?channel ?ъ슜
    if "best_channel" in summ.columns:
        ch_col = "best_channel"
    elif "channel" in summ.columns:
        ch_col = "channel"
    else:
        raise ValueError("SUMMARY_CSV?먮뒗 'best_channel' ?먮뒗 'channel' 而щ읆???덉뼱????")

    tasks = [(row["case"], row[ch_col]) for _, row in summ.iterrows()]
    total = len(tasks)
    idx_width = len(str(total)) if total > 0 else 1

    ok = 0
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(worker_plot_one, case_value, ch_text, i + 1, idx_width)
            for i, (case_value, ch_text) in enumerate(tasks)
        ]

        for fu in as_completed(futures):
            done += 1
            success, msg = fu.result()
            if success:
                ok += 1

            if done % 20 == 0 or (not success):
                print(f"[PROGRESS] {done}/{total} | ok={ok} | {msg}")

    print(f"\nDONE: {ok}/{total} plots saved to: {OUT_IMG_DIR}")


if __name__ == "__main__":
    main_png()

