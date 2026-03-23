# -*- coding: utf-8 -*-
# PSSE33 + PSSPY27 (Python 2.7)
#
# 湲곕뒫:
# 1) N-1 ?ㅼ틪: GENONLY + LINEONLY
# 2) Seed CSV 湲곕컲 N-2: seed(湲곗〈 ?쒕굹由ъ삤) + 異붽? N-1(諛쒖쟾湲??좊줈) 瑜?TRIP_TIME??"?숈떆?? ?몃┰
#
# ?ъ슜踰?
# - RUN_MODE = "N1"  ?먮뒗  "SEED_N2"
# - SEED_CSV / SEED_CASE_COL ?ㅼ젙
#
# 二쇱쓽:
# - 蹂?肄붾뱶??"seed ?뚯씪紐?洹쒖튃"???꾨옒 以??섎굹?????뚯떛??
#     GENONLY_G<bus>_<id>   ?? GENONLY_G128_1
#     LINEONLY_L<fb>_<tb>_<ckt>   ?? LINEONLY_L91_147_1
#   seed 媛믪씠 "..._Voltage.csv" 媛숈? ?묐??ш? 遺숈뼱 ?덉뼱???먮룞 ?쒓굅??

import os
import sys
import time
import csv
from datetime import datetime

# ============================================================
# PSSE PATH
# ============================================================
PSSE_PATH  = r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN"
PSSPY_PATH = r"C:\Program Files (x86)\PTI\PSSE33\PSSPY27"

sys.path.append(PSSE_PATH)
sys.path.append(PSSPY_PATH)
os.environ["PATH"] += ";" + PSSE_PATH

import psspy

_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()

psspy.psseinit(100000)

# ============================================================
# CASE PRESETS
# ============================================================
CASE_PRESETS = {
    "ieee9": {
        "CASE_TYPE": "RAW",
        "RAW": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\raws\ieee9 1.raw",
        "DYR": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\dyrs\ieee9 1.dyr",
        "OUT_ROOT": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\outs\ieee9_scan",
    },
    "ieee118": {
        "CASE_TYPE": "SAV",
        "SAV": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\savs\IEEE118 1.sav",
        "DYR": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\dyrs\IEEE118 1.dyr",
        "OUT_ROOT": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\outs\ieee118_scan",
    },
    "kpgc": {
        "CASE_TYPE": "RAW",
        "RAW": r"C:\Users\pspo\Desktop\pspo\DB\kpg\raw\KPG193-PSS(R)E 2 2.raw",
        "DYR": r"C:\Users\pspo\Desktop\pspo\DB\kpg\dyr\KPG_DYR 2 2.dyr",
        "OUT_ROOT": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\outs\kpgc_scan",
    }
}

# ============================================================
# USER SETTINGS (?ш린留??섏젙)
# ============================================================
ACTIVE_CASE = "kpgc"

# ?ㅽ뻾 紐⑤뱶
# "N1"      : GENONLY + LINEONLY
# "SEED_N2" : seed + 異붽? N-1 (?숈떆 ?몃┰)
# "N1_N2"   : N1 ?꾨즺 ??SEED_N2源뚯? ?곗냽 ?ㅽ뻾
# ?ъ슜 ??
#   RUN_MODE = "N1"       # N-1留??ㅽ뻾
#   RUN_MODE = "SEED_N2"  # seed 湲곕컲 N-2留??ㅽ뻾
#   RUN_MODE = "N1_N2"    # N-1 ?ㅽ뻾 ??N-2 ?곗냽 ?ㅽ뻾 !!!!!!!!!!!!!! 吏꾩쭨 紐⑤뱺 耳?댁뒪媛 ?뚯븘媛??議곗떖 !!!!!!!!!!!!!
# 李멸퀬:
#   "BOTH", "N1+SEED_N2" ??"N1_N2"? ?숈씪?섍쾶 ?숈옉
RUN_MODE = "BOTH"

TRIP_TIME = 1
T_END     = 10.0
DT        = 0.00833333  # 李멸퀬??PSSE run()?먮뒗 吏곸젒 ???)

# --- Target generators (imported from Y:\PSPO\minhwan_NAS\code\kpg_N2_trip.py) ---
Target_GENS = [
    (3,  "1"), (3,  "2"), (3,  "3"),
    (4,  "1"), (4,  "2"),
    (7,  "1"),
    (8,  "1"), (8,  "2"), (8,  "3"), (8,  "4"),
    (10, "1"),
    (19, "1"),
    (20, "1"),
    (21, "1"), (21, "2"), (21, "3"), (21, "4"),
    (26, "1"),
    (27, "1"), (27, "2"), (27, "3"), (27, "4"),
    (36, "1"),
    (37, "1"), (37, "2"),
    (39, "1"), (39, "2"),
    (40, "1"), (40, "2"), (40, "3"), (40, "4"), (40, "5"), (40, "6"),
    (48, "1"), (48, "2"),
    (49, "1"), (49, "2"), (49, "3"), (49, "4"), (49, "5"), (49, "6"),
    (53, "1"), (53, "2"), (53, "3"), (53, "4"), (53, "5"), (53, "6"), (53, "7"), (53, "8"), (53, "9"),
    (58, "1"),
    (59, "1"), (59, "2"), (59, "3"), (59, "4"), (59, "5"), (59, "6"),
    (64, "1"),
    (65, "1"),
    (68, "1"), (68, "2"),
    (70, "1"),
    (71, "1"), (71, "2"), (71, "3"),
    (79, "1"),
    (100,"1"), (100,"2"), (100,"3"), (100,"4"), (100,"5"), (100,"6"), (100,"7"),
    (106,"1"),
    (108,"1"),
    (128,"1"),
    (133,"1"),
    (134,"1"), (134,"2"), (134,"3"),
    (137,"1"),
    (156,"1"),
    (166,"1"), (166,"2"), (166,"3"), (166,"4"), (166,"5"), (166,"6"),
    (167,"1"), (167,"2"), (167,"3"), (167,"4"),
    (181,"1"), (181,"2"),
    (190,"1"), (190,"2"), (190,"3"), (190,"4"),
    (193,"1"), (193,"2"), (193,"3"), (193,"4"),
    (82201,"1"),
    (82202,"1"),
    (82203,"1"),
    (82204,"1"),
    (82205,"1"),
    (82206,"1"),
    (82207,"1"),
    (124201,"1"),
    (124202,"1"),
    (124203,"1"),
    (124204,"1"),
    (124205,"1"),
    (124206,"1"),
    (175201,"1"),
    (175202,"1"),
    (175203,"1"),
    (175204,"1"),
    (175205,"1"),
    (175206,"1"),
    (175207,"1"),
]

# --- ????좊줈 (None?대㈃ 紐⑤뱺 active line) ---
Target_Lines = None

# --- 梨꾨꼸 ?ㅼ젙 (吏湲덉? ?꾩븬留? ---
CHSB_VOLT = [-1, -1, -1, 1, 13, 0]

# --- Seed CSV 湲곕컲 N-2 ?듭뀡 ---
SEED_CSV = r"C:\Users\pspo\Desktop\pspo\MinHwan\OSC\runner\KPGC_20260127_020013\out\kpg_first_filter_top10pct.csv"
SEED_CASE_COL = "case"     # seed csv???쒕굹由ъ삤 ?대쫫 而щ읆
ADD_MODE = "ALL"           # "ALL" / "GEN" / "LINE" (異붽? N-1 ?꾨낫 醫낅쪟)
MAX_ADD_PER_SEED = 1000    # seed 1媛쒕떦 異붽? 議고빀 ???쒗븳
# --- N1 -> CSV -> SEED_N2 chaining options ---
AUTO_CONVERT_OUT_TO_CSV = True
BOTH_USE_N1_SEED_CSV = True
N1_SEED_CSV_NAME = "seed_from_n1.csv"
N1_SEED_CSV_PATH = None    # e.g. r"C:\temp\seed_from_n1.csv"

# --- run path record for validation ---
RUN_PATH_RECORD_FILE = None  # e.g. r"C:\temp\trip_run_paths.txt"

# --- 諛쒖쟾湲??꾨낫 ?뚯뒪 ---
# "AUTO"   : 耳?댁뒪???꾩껜 machine ?꾨낫瑜?議고쉶 ?? in-service 諛쒖쟾湲곕쭔 ?ъ슜
# "TARGET" : Target_GENS留??ъ슜
GEN_SOURCE_MODE = "AUTO"   # "AUTO" or "TARGET"
AUTO_GEN_FALLBACK_TO_TARGET = True
GEN_LIST_TXT_NAME = "in_service_generators.txt"

# ============================================================
# CFG
# ============================================================
CFG = CASE_PRESETS[ACTIVE_CASE]

CASE_TYPE = CFG["CASE_TYPE"]
RAW_PATH  = CFG.get("RAW")
SAV_PATH  = CFG.get("SAV")
DYR_PATH  = CFG["DYR"]
OUT_ROOT  = CFG["OUT_ROOT"]

# ============================================================
# helpers
# ============================================================
def ok(ierr, msg):
    # PSSE API??蹂댄넻 int ierr 諛섑솚. ierr!=0 ?대㈃ ?ㅽ뙣.
    if ierr:
        raise Exception("%s failed (ierr=%d)" % (msg, int(ierr)))

def _assert_file(path, name):
    if path is None:
        return
    if not os.path.isfile(path):
        raise Exception("%s not found: %s" % (name, path))

def validate_paths():
    if CASE_TYPE == "RAW":
        _assert_file(RAW_PATH, "RAW")
    elif CASE_TYPE == "SAV":
        _assert_file(SAV_PATH, "SAV")
    else:
        raise Exception("Invalid CASE_TYPE: %s (use 'RAW' or 'SAV')" % str(CASE_TYPE))

    _assert_file(DYR_PATH, "DYR")

    if not os.path.isdir(OUT_ROOT):
        os.makedirs(OUT_ROOT)

def make_out_dir(tag):
    run_stamp = "%s_%s_%s" % (
        ACTIVE_CASE.upper(),
        tag,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir = os.path.join(OUT_ROOT, run_stamp)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return out_dir

def setup_psse_outputs(out_dir):
    # 6踰?=progress)?쇰줈 ?뚯씪濡?鍮쇰㈃ ?곕??먯씠 議곗슜?댁???寃??뺤긽??
    # ?곕??먮줈 蹂닿퀬 ?띠쑝硫????⑥닔瑜??몄텧?섏? ?딄굅?? 6 ???1(terminal) ?깆쓣 ?⑥빞 ??
    psspy.progress_output(6, os.path.join(out_dir, "PROGRESS.txt"), [0, 0])
    psspy.report_output(6, os.path.join(out_dir, "REPORT.txt"), [2, 0])
    psspy.prompt_output(6, os.path.join(out_dir, "LOG.txt"), [2, 0])

def _unwrap(a):
    if isinstance(a, (list, tuple)) and len(a) == 1:
        return a[0]
    return a

# ============================================================
# CASE LOAD / PF / DYR / DYNAMIC CONVERT
# ============================================================
def load_case_pf():
    # clean channels
    psspy.delete_all_plot_channels()

    # load network
    if CASE_TYPE == "SAV":
        ok(psspy.case(SAV_PATH), "psspy.case")
    else:
        ok(psspy.read(0, RAW_PATH), "psspy.read")

    # power flow (fdns)
    ok(psspy.fdns([0,0,0,1,1,0,99,0]), "psspy.fdns")

def dynamic_convert():
    # 諛섎뱶??DYR 癒쇱? 濡쒕뱶
    ok(psspy.dyre_new([1,1,1,1], DYR_PATH, "", "", ""), "psspy.dyre_new")

    # convert sequence (?쒖?)
    ok(psspy.cong(0), "psspy.cong")

    # conl? ?쇰? 踰꾩쟾?먯꽌 tuple 諛섑솚(?? (ierr, something))?????덉쑝???덉쟾?섍쾶 泥섎━
    r = psspy.conl(0,1,1,[0,0],[100.0,0.0,0.0,100.0]);  ok(r[0] if isinstance(r, tuple) else r, "psspy.conl(1)")
    r = psspy.conl(0,1,2,[0,0],[100.0,0.0,0.0,100.0]);  ok(r[0] if isinstance(r, tuple) else r, "psspy.conl(2)")
    r = psspy.conl(0,1,3,[0,0],[100.0,0.0,0.0,100.0]);  ok(r[0] if isinstance(r, tuple) else r, "psspy.conl(3)")

    ok(psspy.ordr(0), "psspy.ordr")
    ok(psspy.fact(),   "psspy.fact")
    ok(psspy.tysl(0),  "psspy.tysl")

# ============================================================
# NETWORK LISTS
# ============================================================
def get_all_lines():
    ierr, fbus = psspy.abrnint(-1, 1, 1, 1, 1, ['FROMNUMBER'])
    ierr, tbus = psspy.abrnint(-1, 1, 1, 1, 1, ['TONUMBER'])
    ierr, stat = psspy.abrnint(-1, 1, 1, 1, 1, ['STATUS'])
    ierr, cid  = psspy.abrnchar(-1, 1, 1, 1, 1, ['ID'])

    fbus = _unwrap(fbus)
    tbus = _unwrap(tbus)
    stat = _unwrap(stat)
    cid  = _unwrap(cid)

    lines = []
    for fb, tb, st, ck in zip(fbus, tbus, stat, cid):
        if int(st) == 1:
            lines.append((int(fb), int(tb), str(ck).strip()))
    return lines

def filter_active_gens(target_gens):
    active = []
    for bus, gid in target_gens:
        ierr, stat = psspy.macint(int(bus), str(gid), 'STATUS')
        if ierr != 0:
            print("API error: bus=%s id=%s ierr=%s" % (bus, gid, ierr))
            continue
        if int(stat) == 1:
            active.append((int(bus), str(gid)))
    return active

def discover_all_generators():
    """
    ?꾩옱 耳?댁뒪?먯꽌 machine ?꾨낫 (bus, id)瑜?紐⑤몢 ?섏쭛?쒕떎.
    """
    ierr_b, buses = psspy.amachint(-1, 4, ['NUMBER'])
    if ierr_b != 0:
        raise Exception("psspy.amachint(NUMBER) failed ierr=%d" % int(ierr_b))

    ierr_i, ids = psspy.amachchar(-1, 4, ['ID'])
    if ierr_i != 0:
        raise Exception("psspy.amachchar(ID) failed ierr=%d" % int(ierr_i))

    buses = _unwrap(buses)
    ids = _unwrap(ids)

    gens = []
    for b, gid in zip(buses, ids):
        gens.append((int(b), str(gid).strip()))
    return gens

def build_in_service_gen_dict(gen_candidates):
    """
    紐⑤뱺 ?꾨낫??macint('STATUS')瑜??섑뻾??in-service 諛쒖쟾湲곕쭔 dict濡?諛섑솚?쒕떎.
    """
    gen_dict = {}
    ierr_counts = {}

    for bus, gid in gen_candidates:
        ierr, stat = psspy.macint(int(bus), str(gid), 'STATUS')
        ierr = int(ierr)
        ierr_counts[ierr] = ierr_counts.get(ierr, 0) + 1

        if ierr == 0 and int(stat) == 1:
            key = (int(bus), str(gid).strip())
            gen_dict[key] = {"status": 1, "ierr": 0}

    return gen_dict, ierr_counts

def sorted_gen_keys(gen_dict):
    return sorted(gen_dict.keys(), key=lambda x: (int(x[0]), str(x[1])))

def dump_in_service_gen_txt(path, gen_dict, total_candidates, ierr_counts):
    rows = sorted_gen_keys(gen_dict)

    f = open(path, 'wb')
    try:
        f.write("# in-service generators built by macint('STATUS')\n")
        f.write("# total_candidates=%d\n" % int(total_candidates))
        f.write("# in_service_count=%d\n" % int(len(rows)))
        if ierr_counts:
            ks = sorted(ierr_counts.keys())
            f.write("# ierr_counts=" + ",".join(["%d:%d" % (int(k), int(ierr_counts[k])) for k in ks]) + "\n")
        f.write("bus,id,status\n")
        for bus, gid in rows:
            f.write("%d,%s,1\n" % (int(bus), str(gid)))
    finally:
        f.close()


def convert_out_dir_to_csv(out_dir, mode_tag):
    """
    Convert .out files under out_dir by passing an explicit input path.
    This avoids any latest-folder guess.
    """
    from make_CSV import save_CSV

    csv_out_dir = os.path.join(out_dir, "csv")
    case_tag = "%s_%s" % (ACTIVE_CASE.upper(), str(mode_tag).upper())
    result = save_CSV(
        active_case=ACTIVE_CASE,
        in_path_override=out_dir,
        case_tag=case_tag,
        out_dir_override=csv_out_dir,
    )
    if result:
        csv_out_dir = result
    return csv_out_dir


def _collect_n1_seed_names(n1_out_dir):
    names = []
    for fn in os.listdir(n1_out_dir):
        if not fn.lower().endswith(".out"):
            continue
        base = os.path.splitext(fn)[0]
        if base.startswith("GENONLY_G") or base.startswith("LINEONLY_L"):
            names.append(base)
    return sorted(names)


def build_seed_csv_from_n1_out(n1_out_dir, seed_csv_path, case_col):
    seed_names = _collect_n1_seed_names(n1_out_dir)
    if not seed_names:
        raise Exception("No N1 seed candidates found in: %s" % str(n1_out_dir))

    parent = os.path.dirname(seed_csv_path)
    if parent and (not os.path.isdir(parent)):
        os.makedirs(parent)

    f = open(seed_csv_path, "wb")
    try:
        # UTF-8 BOM for Excel compatibility
        f.write("\xef\xbb\xbf")
        w = csv.writer(f)
        w.writerow([str(case_col)])
        for case_name in seed_names:
            w.writerow([case_name])
    finally:
        f.close()

    return len(seed_names)


def write_run_path_record(path_info):
    if RUN_PATH_RECORD_FILE:
        record_path = RUN_PATH_RECORD_FILE
    else:
        record_path = os.path.join(
            OUT_ROOT,
            "run_paths_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    parent = os.path.dirname(record_path)
    if parent and (not os.path.isdir(parent)):
        os.makedirs(parent)

    f = open(record_path, "ab")
    try:
        f.write("=== run %s ===\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for k in sorted(path_info.keys()):
            v = path_info.get(k, "")
            f.write("%s=%s\n" % (str(k), str(v)))
        f.write("\n")
    finally:
        f.close()

    return record_path
# ============================================================
# CONTINGENCY ACTIONS
# ============================================================
def trip_generator(bus, gid):
    ierr = psspy.dist_machine_trip(int(bus), str(gid))
    if ierr == 0:
        return 0
    # ?고쉶: STATUS=0
    return psspy.machine_chng_2(int(bus), str(gid), [0,_i,_i,_i,_i,_i], [_f]*12)

def trip_line(fb, tb, ckt):
    return psspy.dist_branch_trip(int(fb), int(tb), str(ckt))

# ============================================================
# RUN CORE
# ============================================================
def start_channels_and_out(out_path):
    psspy.delete_all_plot_channels()
    psspy.chsb(0, 1, CHSB_VOLT)

    ierr = psspy.strt(0, out_path)
    if ierr != 0:
        return ierr

    ierr = psspy.run(0, float(TRIP_TIME), 1, 1, 0)
    return ierr

def finish_run():
    return psspy.run(0, float(T_END), 1, 1, 0)

# ============================================================
# N-1: GENONLY + LINEONLY
# ============================================================
def run_n1(out_dir, gens_active, lines):
    total_cases = len(gens_active) + len(lines)
    case_idx = 0

    # 1) GENONLY
    for bus, gid in gens_active:
        case_idx += 1
        t0 = time.time()

        load_case_pf()
        dynamic_convert()

        run_tag = "GENONLY_G%s_%s" % (bus, gid)
        out_path = os.path.join(out_dir, "%s.out" % run_tag)

        ierr = start_channels_and_out(out_path)
        ok(ierr, "start/run(pre)")

        trip_generator(bus, gid)

        ierr = finish_run()
        ok(ierr, "run(post)")

        print("[%d/%d] %s => OK (%.2fs, %s)" %
              (case_idx, total_cases, run_tag, time.time()-t0, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # 2) LINEONLY
    for fb, tb, ckt in lines:
        case_idx += 1
        t0 = time.time()

        load_case_pf()
        dynamic_convert()

        run_tag = "LINEONLY_L%s_%s_%s" % (fb, tb, ckt)
        out_path = os.path.join(out_dir, "%s.out" % run_tag)

        ierr = start_channels_and_out(out_path)
        ok(ierr, "start/run(pre)")

        trip_line(fb, tb, ckt)

        ierr = finish_run()
        ok(ierr, "run(post)")

        print("[%d/%d] %s => OK (%.2fs, %s)" %
              (case_idx, total_cases, run_tag, time.time()-t0, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ============================================================
# SEED ?뚯떛 (CSV??case 臾몄옄??-> seed contingency)
# ============================================================
def _strip_seed_name(s):
    base = os.path.basename(str(s))
    for suf in ["_Voltage.csv", "_VOLTAGE.csv", ".csv", ".out"]:
        if base.endswith(suf):
            base = base[:-len(suf)]
    return base

def parse_seed_case(case_str):
    """
    return:
      ("GEN", bus, gid) or ("LINE", fb, tb, ckt) or None
    """
    name = _strip_seed_name(case_str)

    if name.startswith("GENONLY_G"):
        try:
            tail = name[len("GENONLY_G"):]   # "128_1"
            parts = tail.split("_")
            bus = int(parts[0])
            gid = str(parts[1])
            return ("GEN", bus, gid)
        except:
            return None

    if name.startswith("LINEONLY_L"):
        try:
            tail = name[len("LINEONLY_L"):]  # "91_147_1"
            parts = tail.split("_")
            fb = int(parts[0])
            tb = int(parts[1])
            ckt = str(parts[2])
            return ("LINE", fb, tb, ckt)
        except:
            return None

    return None

def same_contingency(a, b):
    if a[0] != b[0]:
        return False
    if a[0] == "GEN":
        return (int(a[1]) == int(b[1])) and (str(a[2]) == str(b[2]))
    else:
        return (int(a[1]) == int(b[1])) and (int(a[2]) == int(b[2])) and (str(a[3]) == str(b[3]))

def build_add_list(lines, gens_active):
    add_list = []
    if ADD_MODE in ["ALL", "GEN"]:
        for bus, gid in gens_active:
            add_list.append(("GEN", int(bus), str(gid)))
    if ADD_MODE in ["ALL", "LINE"]:
        for fb, tb, ckt in lines:
            add_list.append(("LINE", int(fb), int(tb), str(ckt)))
    return add_list

def trip_contingency(info):
    if info[0] == "GEN":
        trip_generator(info[1], info[2])
    else:
        trip_line(info[1], info[2], info[3])

def make_run_tag_seed_n2(seed_info, add_info):
    if seed_info[0] == "GEN":
        seed_tag = "G%s_%s" % (seed_info[1], seed_info[2])
    else:
        seed_tag = "L%s_%s_%s" % (seed_info[1], seed_info[2], seed_info[3])

    if add_info[0] == "GEN":
        add_tag = "G%s_%s" % (add_info[1], add_info[2])
    else:
        add_tag = "L%s_%s_%s" % (add_info[1], add_info[2], add_info[3])

    return "N2_SEED_%s__ADD_%s" % (seed_tag, add_tag)

# ============================================================
# SEED_N2 ?ㅽ뻾
# ============================================================
def run_seed_n2(out_dir, gens_active, lines, seed_csv_path=None):
    import pandas as pd
    seed_csv_to_use = seed_csv_path or SEED_CSV
    if not os.path.isfile(seed_csv_to_use):
        raise Exception("Seed CSV not found: %s" % str(seed_csv_to_use))

    print("SEED_CSV_USED            :", seed_csv_to_use)
    df = pd.read_csv(seed_csv_to_use, encoding="utf-8-sig")
    if SEED_CASE_COL not in df.columns:
        raise Exception("SEED_CASE_COL not found: %s" % SEED_CASE_COL)

    # seed list ?뚯떛
    seed_list = []
    bad_seed = 0
    for v in df[SEED_CASE_COL].values:
        info = parse_seed_case(v)
        if info is not None:
            seed_list.append(info)
        else:
            bad_seed += 1

    if not seed_list:
        raise Exception("No valid seed parsed from seed csv")

    add_list = build_add_list(lines, gens_active)
    if not add_list:
        raise Exception("ADD list is empty (check ADD_MODE / gens / lines)")

    # ============================================================
    #  ?꾩껜 寃쎌슦????怨꾩궛/異쒕젰 (?ㅽ뻾 ?꾩뿉)
    # ============================================================
    total_seed = len(seed_list)
    total_add  = len(add_list)

    # seed ?섎굹??(?숈씪 contingency ?쒖쇅) + MAX_ADD_PER_SEED 而??곸슜
    total_pairs = 0
    for seed_info in seed_list:
        # ?숈씪 contingency ?쒖쇅??add ?꾨낫 ??
        valid_add = 0
        for add_info in add_list:
            if same_contingency(seed_info, add_info):
                continue
            valid_add += 1

        # seed???쒗븳 ?곸슜
        if MAX_ADD_PER_SEED is not None:
            valid_add = min(valid_add, int(MAX_ADD_PER_SEED))

        total_pairs += valid_add

    print("")
    print("========== SEED_N2 CASE COUNT ==========")
    print("seed rows in CSV         :", len(df))
    print("parsed seed count        :", total_seed)
    print("unparsed seed count      :", bad_seed)
    print("add candidate count      :", total_add, "(ADD_MODE=%s)" % ADD_MODE)
    print("MAX_ADD_PER_SEED         :", MAX_ADD_PER_SEED)
    print("TOTAL N-2 cases to run   :", total_pairs)
    print("OUT_DIR                  :", out_dir)
    print("========================================")
    print("")

    # ============================================================
    # ?꾨옒遺???ㅼ젣 ?ㅽ뻾
    # ============================================================
    total_done = 0

    for seed_info in seed_list:
        add_cnt = 0

        for add_info in add_list:
            if same_contingency(seed_info, add_info):
                continue

            add_cnt += 1
            if MAX_ADD_PER_SEED is not None and int(add_cnt) > int(MAX_ADD_PER_SEED):
                break

            t0 = time.time()

            load_case_pf()
            dynamic_convert()

            run_tag = make_run_tag_seed_n2(seed_info, add_info)
            out_path = os.path.join(out_dir, "%s.out" % run_tag)

            ierr = start_channels_and_out(out_path)
            if ierr != 0:
                print("[FAIL pre] %s ierr=%d" % (run_tag, int(ierr)))
                continue

            # TRIP_TIME ?쒖젏 ?숈떆 ?몃┰
            trip_contingency(seed_info)
            trip_contingency(add_info)

            ierr = finish_run()
            if ierr != 0:
                print("[FAIL post] %s ierr=%d" % (run_tag, int(ierr)))
                continue

            total_done += 1
            print("[N2 %d/%d] %s => OK (%.2fs, %s)" %
                  (total_done, total_pairs, run_tag, time.time()-t0,
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ============================================================
# MAIN
# ============================================================
def main():
    validate_paths()

    # ??踰?濡쒕뱶?댁꽌 由ъ뒪?몃뱾 ?뺣낫 (active gens / active lines)
    load_case_pf()

    macint_ierr_counts = {}
    total_gen_candidates = 0

    if str(GEN_SOURCE_MODE).upper() == "AUTO":
        try:
            gen_candidates = discover_all_generators()
            total_gen_candidates = len(gen_candidates)
            gen_dict, macint_ierr_counts = build_in_service_gen_dict(gen_candidates)
            gens_active = sorted_gen_keys(gen_dict)
            print("GEN_SOURCE_MODE=AUTO | candidates=%d in_service=%d" % (total_gen_candidates, len(gens_active)))
        except Exception as e:
            if AUTO_GEN_FALLBACK_TO_TARGET:
                print("[WARN] AUTO generator discovery failed, fallback to TARGET_GENS: %s" % str(e))
                gens_active = filter_active_gens(Target_GENS)
                total_gen_candidates = len(Target_GENS)
                gen_dict = {}
                for b, g in gens_active:
                    gen_dict[(int(b), str(g))] = {"status": 1, "ierr": 0}
            else:
                raise
    else:
        gens_active = filter_active_gens(Target_GENS)
        total_gen_candidates = len(Target_GENS)
        gen_dict = {}
        for b, g in gens_active:
            gen_dict[(int(b), str(g))] = {"status": 1, "ierr": 0}

    if Target_Lines is None:
        lines = get_all_lines()
    else:
        lines = Target_Lines

    if not gens_active:
        raise Exception("No in-service generators found (GEN_SOURCE_MODE=%s)" % str(GEN_SOURCE_MODE))
    if not lines:
        raise Exception("Target_Lines is empty")

    run_mode_key = str(RUN_MODE).strip().upper()
    if run_mode_key == "N1":
        run_plan = ["N1"]
    elif run_mode_key == "SEED_N2":
        run_plan = ["SEED_N2"]
    elif run_mode_key in ("N1_N2", "BOTH", "N1+SEED_N2"):
        run_plan = ["N1", "SEED_N2"]
    else:
        raise Exception("Invalid RUN_MODE: %s (use N1 / SEED_N2 / N1_N2/BOTH)" % str(RUN_MODE))

    finished_out_dirs = []
    run_path_info = {
        "RUN_MODE": str(RUN_MODE),
        "ACTIVE_CASE": str(ACTIVE_CASE),
        "OUT_ROOT": str(OUT_ROOT),
    }
    seed_csv_for_n2 = SEED_CSV

    for mode in run_plan:
        # output dir + psse output redirect
        if mode == "N1":
            out_dir = make_out_dir("N1_Line_%d_Gen_%d" % (len(lines), len(gens_active)))
        else:
            out_dir = make_out_dir("SEEDN2_Line_%d_Gen_%d" % (len(lines), len(gens_active)))

        setup_psse_outputs(out_dir)

        # save in-service generator list
        gen_txt_path = os.path.join(out_dir, GEN_LIST_TXT_NAME)
        dump_in_service_gen_txt(gen_txt_path, gen_dict, total_gen_candidates, macint_ierr_counts)
        print("saved generator list:", gen_txt_path)

        if mode == "N1":
            run_n1(out_dir, gens_active, lines)
            run_path_info["N1_OUT_DIR"] = out_dir

            n1_csv_dir = ""
            if AUTO_CONVERT_OUT_TO_CSV:
                try:
                    n1_csv_dir = convert_out_dir_to_csv(out_dir, "N1")
                    print("saved N1 csv dir       :", n1_csv_dir)
                except Exception as e:
                    print("N1 CSV conversion failed: %s" % str(e))
            if n1_csv_dir:
                run_path_info["N1_CSV_DIR"] = n1_csv_dir

            if ("SEED_N2" in run_plan) and BOTH_USE_N1_SEED_CSV:
                seed_csv_target = N1_SEED_CSV_PATH
                if not seed_csv_target:
                    seed_base_dir = n1_csv_dir if n1_csv_dir else out_dir
                    seed_csv_target = os.path.join(seed_base_dir, N1_SEED_CSV_NAME)

                seed_count = build_seed_csv_from_n1_out(out_dir, seed_csv_target, SEED_CASE_COL)
                seed_csv_for_n2 = seed_csv_target
                print("saved seed csv from N1 :", seed_csv_for_n2, "(rows=%d)" % int(seed_count))
                run_path_info["SEED_FROM_N1_CSV"] = seed_csv_for_n2
                run_path_info["SEED_FROM_N1_ROWS"] = int(seed_count)
        elif mode == "SEED_N2":
            run_path_info["SEED_CSV_USED"] = seed_csv_for_n2
            run_seed_n2(out_dir, gens_active, lines, seed_csv_path=seed_csv_for_n2)
            run_path_info["SEED_N2_OUT_DIR"] = out_dir

            if AUTO_CONVERT_OUT_TO_CSV:
                try:
                    n2_csv_dir = convert_out_dir_to_csv(out_dir, "SEED_N2")
                    print("saved N2 csv dir       :", n2_csv_dir)
                    run_path_info["SEED_N2_CSV_DIR"] = n2_csv_dir
                except Exception as e:
                    print("N2 CSV conversion failed: %s" % str(e))

        finished_out_dirs.append(out_dir)

    run_path_info["OUT_DIRS"] = ";".join(finished_out_dirs)
    record_path = write_run_path_record(run_path_info)
    print("saved run path record   :", record_path)

    print("All done. OUT_DIRS = %s" % ", ".join(finished_out_dirs))

if __name__ == "__main__":
    main()

