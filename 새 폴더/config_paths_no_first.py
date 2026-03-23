# -*- coding: utf-8 -*-
r"""
No-first(`1李??앸왂`) ?뚯씠?꾨씪???꾩슜 寃쎈줈/異쒕젰 ?ㅼ젙 紐⑤뱢.

?듭떖 ??ぉ:
- `CSV_INPUT_DIRS`: no-first 2李??됯? ????꾩븬 CSV ?낅젰 ?대뜑(蹂듭닔 媛??
- `NOFIRST_SECOND_CSV`: no-first 2李?寃곌낵 CSV
- `NOFIRST_LINEUP_OUT_CSV`, `NOFIRST_LINEUP_WITHIN_LIMIT_CSV`: lineup 寃곌낵 CSV
"""

import os
from typing import Optional, Sequence


def _pick_latest_run_tag(csv_base: str) -> Optional[str]:
    """`csv_base` ?섏쐞 ?대뜑 以??섏젙 ?쒓컙??媛??理쒖떊???대뜑紐낆쓣 諛섑솚?쒕떎."""
    if not os.path.isdir(csv_base):
        return None
    dirs = []
    for name in os.listdir(csv_base):
        path = os.path.join(csv_base, name)
        if os.path.isdir(path):
            dirs.append(path)
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.basename(dirs[0])


"""?꾨줈?앺듃 湲곗? 猷⑦듃 寃쎈줈."""
ROOT = os.path.dirname(os.path.abspath(__file__))
OSC_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
CSV_BASE = os.path.join(OSC_ROOT, "csv")
RUNNER_BASE = os.path.join(OSC_ROOT, "runner")

"""?ㅽ뻾 ???RUN_TAG ?좏깮 ?뺤콉."""
AUTO_PICK_LATEST_RUN_TAG = False
DEFAULT_RUN_TAG = "KPGC_20260127_020013"

if AUTO_PICK_LATEST_RUN_TAG:
    RUN_TAG = _pick_latest_run_tag(CSV_BASE) or DEFAULT_RUN_TAG
else:
    RUN_TAG = DEFAULT_RUN_TAG

"""RUN_TAG 湲곕컲 ?뚯깮 寃쎈줈."""
CSV_DIR = os.path.join(CSV_BASE, RUN_TAG)
RUN_DIR = os.path.join(RUNNER_BASE, RUN_TAG)
OUT_DIR = os.path.join(RUN_DIR, "out")

"""No-first ?뚯씠?꾨씪??異쒕젰 寃쎈줈."""
NOFIRST_SECOND_CSV = os.path.join(OUT_DIR, "kpg_second_filter_all_cases.csv")
NOFIRST_LINEUP_OUT_CSV = os.path.join(OUT_DIR, "GlobalRisk_Scenario_V_tag_duration_nofirst.csv")
NOFIRST_LINEUP_WITHIN_LIMIT_CSV = os.path.join(
    OUT_DIR,
    "GlobalRisk_Scenario_oscillating_within_limits_nofirst.csv",
)

"""怨듯넻 ?ㅽ뻾 ?듭뀡."""
MODULE_MAX_WORKERS = 5
FIXED_DT_SEC = 1.0 / 120.0
WRITE_SECOND_LOG = False
WRITE_SECOND_SUMMARY = False

"""?낅젰 寃쎈줈 ?ㅼ젙."""
CSV_INPUT_DIRS = ["C:\\Users\\pspo\\Desktop\\pspo\\MinHwan\\Trips\\csv\\KPGC_20260122_141002"] #湲곕낯 寃쎈줈: CSV_DIR , ?꾩슂 ???대뜑 寃쎈줈 吏??

def ensure_dirs(create_input_dir: bool = False) -> None:
    """?꾩슂??異쒕젰 ?붾젆?곕━瑜??앹꽦?섍퀬, ?듭뀡???곕씪 ?낅젰 ?붾젆?곕━???앹꽦?쒕떎."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if create_input_dir:
        os.makedirs(CSV_DIR, exist_ok=True)


def _as_path_list(v: Sequence[str] | str | None) -> list[str]:
    """臾몄옄???쒗?ㅻ? ?뺢퇋?뷀빐 鍮?媛믪씠 ?쒓굅??寃쎈줈 臾몄옄??由ъ뒪?몃? 諛섑솚?쒕떎."""
    if v is None:
        return []
    if isinstance(v, str):
        items = [v]
    else:
        items = [str(x) for x in v]
    return [x.strip() for x in items if str(x).strip()]


def get_csv_input_dirs() -> list[str]:
    """`CSV_INPUT_DIRS`瑜??뺢퇋?뷀빐 ?덈?寃쎈줈/以묐났 ?쒓굅??由ъ뒪?몃줈 諛섑솚?쒕떎."""
    items = _as_path_list(globals().get("CSV_INPUT_DIRS", [CSV_DIR])) or [CSV_DIR]
    out = []
    seen = set()
    for p in items:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out

