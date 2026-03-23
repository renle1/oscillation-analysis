# -*- coding: utf-8 -*-
# python 2.7
"""
PSSE .out -> grouped CSV converter.

Applied upgrades:
- Standard CLI (case/input/exports/out-dir)
- Runtime pre-check (PSSE paths + dyntools import)
- Run reports (summary + per-file results + unclassified channels)
- Regex rule-based channel classification
- Settings split to make_csv_config.py

주요 흐름:
1) 런타임 점검(PSSE 경로/dyntools import)
2) 입력 대상 결정(단일 .out 또는 폴더)
3) 채널 분류 후 그룹 CSV 생성
4) 실행 요약/파일별 결과/미분류 채널 로그 저장
"""

import os
import sys
import csv
import glob
import time
import re
import traceback
import argparse
from datetime import datetime

import make_csv_config as appcfg

try:
    # Python 2/3 모두에서 unicode 타입 이름을 동일하게 다루기 위한 호환 처리
    unicode
except NameError:
    unicode = str


DYTOOLS = None
TS_RE = re.compile(r"(\d{8})_(\d{6})$")
EXPORT_KEYS = ("VOLT", "FREQ", "P", "Q")

# 정규식 기반 채널 분류 규칙
# Q를 P보다 먼저 판별해 P/Q 오분류를 줄인다.
RULES_VOLT = [
    re.compile(r"\bVOLT(AGE)?\b", re.IGNORECASE),
    re.compile(r"\bVPU\b", re.IGNORECASE),
    re.compile(r"\bBUS\s*V(OLT)?\b", re.IGNORECASE),
    re.compile(r"\bBUS\s*\d+\s*V\b", re.IGNORECASE),
]
RULES_FREQ = [
    re.compile(r"\bFREQ(UENCY)?\b", re.IGNORECASE),
    re.compile(r"\bHZ\b", re.IGNORECASE),
]
RULES_Q = [
    re.compile(r"\bQ(ELEC)?\b", re.IGNORECASE),
    re.compile(r"\bMVAR\b", re.IGNORECASE),
    re.compile(r"\bVAR\b", re.IGNORECASE),
]
RULES_P = [
    re.compile(r"\bP(ELEC|OWR)?\b", re.IGNORECASE),
    re.compile(r"\bMW\b", re.IGNORECASE),
]


def _to_csv_cell(value):
    # CSV writer에 안전하게 들어갈 값으로 정규화
    if value is None:
        return ""
    if isinstance(value, unicode):
        return value.encode("utf-8")
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return repr(value)


def _write_rows_utf8_bom(path, header, rows):
    # 엑셀 호환을 위해 UTF-8 BOM으로 기록
    f = open(path, "wb")
    try:
        f.write(u"\ufeff".encode("utf-8"))
        writer = csv.writer(f)
        writer.writerow([_to_csv_cell(h) for h in header])
        for row in rows:
            writer.writerow([_to_csv_cell(x) for x in row])
    finally:
        f.close()


def _normalize_label(label):
    if label is None:
        return ""
    return unicode(label).strip()


def _match_any(regex_list, text):
    for rx in regex_list:
        if rx.search(text):
            return True
    return False


def classify_channel(label):
    # 우선순위: VOLT -> FREQ -> Q -> P -> OTHER
    s = _normalize_label(label)
    if _match_any(RULES_VOLT, s):
        return "VOLT"
    if _match_any(RULES_FREQ, s):
        return "FREQ"
    if _match_any(RULES_Q, s):
        return "Q"
    if _match_any(RULES_P, s):
        return "P"
    return "OTHER"


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(path))]


def _dir_ts(path):
    name = os.path.basename(path.rstrip("\\/"))
    m = TS_RE.search(name)
    if not m:
        return datetime.min
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _normalize_exports(export_flags):
    base = {"VOLT": False, "FREQ": False, "P": False, "Q": False}
    if not export_flags:
        defaults = dict(getattr(appcfg, "EXPORT_DEFAULTS", {}))
        for key in EXPORT_KEYS:
            base[key] = bool(defaults.get(key, False))
        return base

    for key in EXPORT_KEYS:
        base[key] = bool(export_flags.get(key, False))
    return base


def parse_export_spec(spec, defaults):
    # --exports 문자열을 내부 플래그(dict)로 변환
    # 예: "voltage,freq", "all", "none"
    out = {"VOLT": False, "FREQ": False, "P": False, "Q": False}
    for key in EXPORT_KEYS:
        out[key] = bool(defaults.get(key, False))

    if spec is None:
        return out

    tokens = [t.strip().upper() for t in unicode(spec).split(",") if t.strip()]
    if not tokens:
        return out

    if "ALL" in tokens:
        for key in EXPORT_KEYS:
            out[key] = True
        return out

    if "NONE" in tokens:
        for key in EXPORT_KEYS:
            out[key] = False
        tokens = [t for t in tokens if t != "NONE"]

    # 명시 토큰이 있으면 "해당 목록만 활성화"로 처리
    selected = set()
    for token in tokens:
        if token in ("V", "VOLT", "VOLTAGE"):
            selected.add("VOLT")
        elif token in ("F", "FREQ", "FREQUENCY"):
            selected.add("FREQ")
        elif token == "P":
            selected.add("P")
        elif token == "Q":
            selected.add("Q")
        else:
            raise ValueError("Unknown export token: %s" % token)

    if selected:
        for key in EXPORT_KEYS:
            out[key] = (key in selected)
    return out


def ensure_runtime(psse_path=None, psspy_path=None):
    # 실행 전 필수 환경 검증:
    # - PSSE/PSSPY 경로 존재
    # - PATH/sys.path 설정
    # - dyntools import 가능 여부
    global DYTOOLS

    psse = psse_path or getattr(appcfg, "PSSE_PATH", "")
    psspy = psspy_path or getattr(appcfg, "PSSPY_PATH", "")

    errors = []
    if not psse or not os.path.isdir(psse):
        errors.append("PSSE_PATH not found: %s" % psse)
    if not psspy or not os.path.isdir(psspy):
        errors.append("PSSPY_PATH not found: %s" % psspy)
    if errors:
        raise RuntimeError(" | ".join(errors))

    if psse not in sys.path:
        sys.path.append(psse)
    if psspy not in sys.path:
        sys.path.append(psspy)

    path_env = os.environ.get("PATH", "")
    if psse not in path_env:
        os.environ["PATH"] = path_env + ";" + psse if path_env else psse

    try:
        import dyntools as _dyntools
    except Exception as e:
        raise RuntimeError("Failed to import dyntools: %s" % repr(e))

    DYTOOLS = _dyntools


def write_group_csv(csv_path, time_arr, chan_data, chan_id_map, channel_numbers):
    # time + 선택 채널들로 단일 그룹 CSV 생성
    header = ["time"]
    for ch in channel_numbers:
        label = chan_id_map.get(ch, "CH_%s" % str(ch))
        header.append("%s | %s" % (str(ch).zfill(3), label))

    n = len(time_arr)
    rows = []
    for i in range(n):
        row = [time_arr[i]]
        for ch in channel_numbers:
            series = chan_data.get(ch, [])
            row.append(series[i] if i < len(series) else "")
        rows.append(row)

    _write_rows_utf8_bom(csv_path, header, rows)


def convert_one_out_to_csvs(out_path, out_dir, export_flags, unknown_counter):
    # .out 1개를 읽어 그룹별 CSV를 생성하고 통계값을 반환
    if DYTOOLS is None:
        raise RuntimeError("dyntools is not initialized. Call ensure_runtime() first.")

    base = os.path.splitext(os.path.basename(out_path))[0]
    chnf = DYTOOLS.CHNF(out_path)
    _short_title, chan_id_map, chan_data = chnf.get_data()

    if "time" not in chan_data:
        raise RuntimeError("time channel not found in %s" % out_path)
    time_arr = chan_data["time"]

    groups = {"VOLT": [], "FREQ": [], "P": [], "Q": []}
    for key in chan_data.keys():
        if key == "time":
            continue
        label = chan_id_map.get(key, "CH_%s" % str(key))
        g = classify_channel(label)
        if g in groups:
            groups[g].append(key)
        else:
            label_key = _normalize_label(label)
            unknown_counter[label_key] = unknown_counter.get(label_key, 0) + 1

    for g in groups:
        groups[g].sort()

    written = []
    if export_flags["VOLT"]:
        if groups["VOLT"]:
            path_v = os.path.join(out_dir, base + "_Voltage.csv")
            write_group_csv(path_v, time_arr, chan_data, chan_id_map, groups["VOLT"])
            written.append(path_v)
        else:
            print("[SKIP] No VOLT channels in %s" % out_path)

    if export_flags["FREQ"]:
        if groups["FREQ"]:
            path_f = os.path.join(out_dir, base + "_Frequency.csv")
            write_group_csv(path_f, time_arr, chan_data, chan_id_map, groups["FREQ"])
            written.append(path_f)
        else:
            print("[SKIP] No FREQ channels in %s" % out_path)

    if export_flags["P"]:
        if groups["P"]:
            path_p = os.path.join(out_dir, base + "_P.csv")
            write_group_csv(path_p, time_arr, chan_data, chan_id_map, groups["P"])
            written.append(path_p)
        else:
            print("[SKIP] No P channels in %s" % out_path)

    if export_flags["Q"]:
        if groups["Q"]:
            path_q = os.path.join(out_dir, base + "_Q.csv")
            write_group_csv(path_q, time_arr, chan_data, chan_id_map, groups["Q"])
            written.append(path_q)
        else:
            print("[SKIP] No Q channels in %s" % out_path)

    return {
        "written_count": len(written),
        "group_volt": len(groups["VOLT"]),
        "group_freq": len(groups["FREQ"]),
        "group_p": len(groups["P"]),
        "group_q": len(groups["Q"]),
    }


def _pick_in_dir(case_cfg, in_dir_override=None):
    # 입력 폴더 결정 우선순위:
    # 1) 함수 인자 override
    # 2) preset IN_DIR_OVERRIDE
    # 3) BASE_DIR/RUN_GLOB 기준 최신 run 폴더
    if in_dir_override and os.path.isdir(in_dir_override):
        return in_dir_override

    preset_override = case_cfg.get("IN_DIR_OVERRIDE")
    if preset_override and os.path.isdir(preset_override):
        return preset_override

    base_dir = case_cfg["BASE_DIR"]
    run_glob = case_cfg["RUN_GLOB"]
    run_dirs = [d for d in glob.glob(os.path.join(base_dir, run_glob)) if os.path.isdir(d)]
    if not run_dirs:
        raise RuntimeError("No run folders found under: %s (pattern=%s)" % (base_dir, run_glob))

    run_dirs = sorted(run_dirs, key=_dir_ts)
    return run_dirs[-1]


def _build_output_dir(csv_root_dir, case_tag, out_dir_override=None):
    # 출력 폴더 생성(override가 없으면 CASE_TAG_타임스탬프)
    if out_dir_override:
        out_dir = os.path.abspath(out_dir_override)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(csv_root_dir, case_tag + "_" + stamp)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return out_dir


def _collect_out_files(in_dir):
    # 입력 폴더에서 .out 파일만 수집 후 자연 정렬
    out_files = []
    for fn in os.listdir(in_dir):
        if fn.lower().endswith(".out"):
            out_files.append(os.path.join(in_dir, fn))
    return sorted(out_files, key=natural_key)


def _write_file_results(out_dir, file_results):
    # 파일 단위 성공/실패와 채널 그룹 통계를 CSV로 저장
    path = os.path.join(out_dir, "run_file_results.csv")
    header = [
        "index",
        "out_file",
        "status",
        "error",
        "written_count",
        "group_volt",
        "group_freq",
        "group_p",
        "group_q",
    ]
    rows = []
    for i, r in enumerate(file_results, 1):
        rows.append([
            i,
            r.get("out_file", ""),
            r.get("status", ""),
            r.get("error", ""),
            r.get("written_count", 0),
            r.get("group_volt", 0),
            r.get("group_freq", 0),
            r.get("group_p", 0),
            r.get("group_q", 0),
        ])
    _write_rows_utf8_bom(path, header, rows)
    return path


def _write_run_summary(out_dir, summary):
    # 실행 1회 요약(입력/출력/성공률/옵션)을 1행 CSV로 저장
    path = os.path.join(out_dir, "run_summary.csv")
    header = [
        "start_time",
        "end_time",
        "duration_sec",
        "active_case",
        "case_tag",
        "input_mode",
        "input_path",
        "output_dir",
        "total_out_files",
        "success_count",
        "fail_count",
        "export_volt",
        "export_freq",
        "export_p",
        "export_q",
    ]
    row = [[
        summary.get("start_time", ""),
        summary.get("end_time", ""),
        summary.get("duration_sec", 0.0),
        summary.get("active_case", ""),
        summary.get("case_tag", ""),
        summary.get("input_mode", ""),
        summary.get("input_path", ""),
        summary.get("output_dir", ""),
        summary.get("total_out_files", 0),
        summary.get("success_count", 0),
        summary.get("fail_count", 0),
        int(bool(summary.get("export_volt", False))),
        int(bool(summary.get("export_freq", False))),
        int(bool(summary.get("export_p", False))),
        int(bool(summary.get("export_q", False))),
    ]]
    _write_rows_utf8_bom(path, header, row)
    return path


def _write_unclassified_channels(out_dir, unknown_counter):
    # 분류 규칙에 걸리지 않은 채널 라벨 빈도를 저장
    path = os.path.join(out_dir, "unclassified_channels.csv")
    header = ["label", "count"]
    pairs = sorted(unknown_counter.items(), key=lambda x: (-x[1], x[0].lower()))
    rows = [[k, v] for (k, v) in pairs]
    _write_rows_utf8_bom(path, header, rows)
    return path


def _resolve_input_target(case_cfg, in_path_override):
    # 입력 소스를 단일 파일/폴더 모드로 판별
    if in_path_override:
        p = os.path.abspath(in_path_override)
        if os.path.isfile(p) and p.lower().endswith(".out"):
            return "single_out", p
        if os.path.isdir(p):
            return "directory", p
        raise RuntimeError("Invalid input path: %s" % in_path_override)

    picked = _pick_in_dir(case_cfg, in_dir_override=None)
    return "directory", picked


def save_CSV(
    active_case=None,
    in_path_override=None,
    case_tag=None,
    out_dir_override=None,
    export_flags=None,
    psse_path=None,
    psspy_path=None
):
    # 외부에서 호출하는 메인 실행 함수
    # 실제 변환 + 보고서 출력까지 책임진다.
    start_epoch = time.time()
    start_text = time.strftime("%Y-%m-%d %H:%M:%S")

    case_key = active_case or appcfg.ACTIVE_CASE
    presets = dict(getattr(appcfg, "CASE_PRESETS", {}))
    if case_key not in presets:
        raise ValueError("Unknown ACTIVE_CASE: %s (use one of %s)" % (case_key, ", ".join(sorted(presets.keys()))))
    case_cfg = presets[case_key]

    ensure_runtime(psse_path=psse_path, psspy_path=psspy_path)

    ex_flags = _normalize_exports(export_flags)
    tag = case_tag or case_cfg["CASE_TAG"]
    csv_root = getattr(appcfg, "CSV_ROOT_DIR", ".")

    input_mode, input_target = _resolve_input_target(case_cfg, in_path_override)
    out_dir = _build_output_dir(csv_root, tag, out_dir_override=out_dir_override)

    print("ACTIVE_CASE:", case_key)
    print("INPUT_MODE :", input_mode)
    print("INPUT_PATH :", input_target)
    print("OUT_DIR    :", out_dir)
    print("EXPORTS    : VOLT=%d FREQ=%d P=%d Q=%d" % (
        int(ex_flags["VOLT"]), int(ex_flags["FREQ"]), int(ex_flags["P"]), int(ex_flags["Q"])
    ))

    if input_mode == "single_out":
        out_files = [input_target]
    else:
        out_files = _collect_out_files(input_target)

    if not out_files:
        print("No .out files found in: %s" % input_target)
        end_epoch = time.time()
        summary = {
            "start_time": start_text,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(end_epoch - start_epoch, 3),
            "active_case": case_key,
            "case_tag": tag,
            "input_mode": input_mode,
            "input_path": input_target,
            "output_dir": out_dir,
            "total_out_files": 0,
            "success_count": 0,
            "fail_count": 0,
            "export_volt": ex_flags["VOLT"],
            "export_freq": ex_flags["FREQ"],
            "export_p": ex_flags["P"],
            "export_q": ex_flags["Q"],
        }
        _write_run_summary(out_dir, summary)
        return out_dir

    print("Found %d .out files." % len(out_files))
    print("FIRST 10 OUT:", [os.path.basename(p) for p in out_files[:10]])

    file_results = []
    unknown_counter = {}
    fail = 0

    for i, out_path in enumerate(out_files, 1):
        print("[%d/%d] %s" % (i, len(out_files), os.path.basename(out_path)))
        row = {"out_file": out_path}
        try:
            stats = convert_one_out_to_csvs(out_path, out_dir, ex_flags, unknown_counter)
            row.update(stats)
            row["status"] = "ok"
            row["error"] = ""
            print("[OK] %s -> CSV saved in %s" % (out_path, out_dir))
        except Exception as e:
            fail += 1
            row["status"] = "fail"
            row["error"] = repr(e)
            row["written_count"] = 0
            row["group_volt"] = 0
            row["group_freq"] = 0
            row["group_p"] = 0
            row["group_q"] = 0
            print("[FAIL] %s" % out_path)
            print("Reason:", repr(e))
            traceback.print_exc()
        file_results.append(row)

    success = len(out_files) - fail
    end_epoch = time.time()

    summary = {
        "start_time": start_text,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(end_epoch - start_epoch, 3),
        "active_case": case_key,
        "case_tag": tag,
        "input_mode": input_mode,
        "input_path": input_target,
        "output_dir": out_dir,
        "total_out_files": len(out_files),
        "success_count": success,
        "fail_count": fail,
        "export_volt": ex_flags["VOLT"],
        "export_freq": ex_flags["FREQ"],
        "export_p": ex_flags["P"],
        "export_q": ex_flags["Q"],
    }

    summary_path = _write_run_summary(out_dir, summary)
    if bool(getattr(appcfg, "WRITE_FILE_RESULTS", True)):
        results_path = _write_file_results(out_dir, file_results)
    else:
        results_path = "disabled"

    if bool(getattr(appcfg, "WRITE_UNCLASSIFIED_LOG", True)):
        unclassified_path = _write_unclassified_channels(out_dir, unknown_counter)
    else:
        unclassified_path = "disabled"

    print("Done. total=%d fail=%d success=%d" % (len(out_files), fail, success))
    print("saved_summary    :", summary_path)
    print("saved_file_result:", results_path)
    print("saved_unclassified:", unclassified_path)
    return out_dir


def _build_parser():
    # 명시 옵션 + 기존 positional 호출을 함께 지원
    parser = argparse.ArgumentParser(
        description="Convert PSSE .out files to grouped CSV files (Python 2.7, PSSE33)."
    )
    parser.add_argument("--case", dest="case_key", default=None, help="Case preset key (ieee9/ieee118/kpgc)")
    parser.add_argument("--input-dir", dest="input_dir", default=None, help="Directory containing .out files")
    parser.add_argument("--input-out", dest="input_out", default=None, help="Single .out file path")
    parser.add_argument(
        "--exports",
        dest="exports",
        default=None,
        help="Comma list: voltage,freq,p,q | all | none",
    )
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Exact output directory path")
    parser.add_argument("--case-tag", dest="case_tag", default=None, help="Output folder prefix tag")
    parser.add_argument("--psse-path", dest="psse_path", default=None, help="Override PSSE PSSBIN path")
    parser.add_argument("--psspy-path", dest="psspy_path", default=None, help="Override PSSE PSSPY path")
    parser.add_argument("--check-only", dest="check_only", action="store_true", help="Only run runtime pre-check")

    # 기존 호출 호환: python make_CSV.py <input> [case_tag]
    parser.add_argument("legacy_input", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("legacy_case_tag", nargs="?", default=None, help=argparse.SUPPRESS)
    return parser


def _resolve_cli_input(args):
    # CLI 인자를 내부 입력 경로/케이스 태그로 정리
    in_path = None
    case_tag = args.case_tag

    if args.input_dir and args.input_out:
        raise ValueError("Use either --input-dir or --input-out, not both.")

    if args.input_out:
        in_path = args.input_out
    elif args.input_dir:
        in_path = args.input_dir
    elif args.legacy_input:
        p = args.legacy_input
        if os.path.isdir(p) or (os.path.isfile(p) and p.lower().endswith(".out")):
            in_path = p
        else:
            print("Warning: invalid legacy input path, using preset latest run: %s" % p)

    if (not case_tag) and args.legacy_case_tag:
        case_tag = args.legacy_case_tag

    return in_path, case_tag


if __name__ == "__main__":
    # CLI 진입점: 파싱 -> 체크 모드/실행 모드 분기
    parser = _build_parser()
    args = parser.parse_args()

    try:
        export_defaults = dict(getattr(appcfg, "EXPORT_DEFAULTS", {}))
        export_flags = parse_export_spec(args.exports, export_defaults)
        in_path, case_tag = _resolve_cli_input(args)

        if args.check_only:
            ensure_runtime(psse_path=args.psse_path, psspy_path=args.psspy_path)
            print("Runtime pre-check passed.")
            sys.exit(0)

        save_CSV(
            active_case=args.case_key,
            in_path_override=in_path,
            case_tag=case_tag,
            out_dir_override=args.out_dir,
            export_flags=export_flags,
            psse_path=args.psse_path,
            psspy_path=args.psspy_path,
        )
    except Exception as cli_err:
        print("[ERROR]", repr(cli_err))
        traceback.print_exc()
        sys.exit(1)
