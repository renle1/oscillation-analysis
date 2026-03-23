# -*- coding: utf-8 -*-
"""
Configuration for make_CSV.py (Python 2.7 compatible).

Keep operational presets here so make_CSV.py can stay logic-focused.
"""

PSSE_PATH = r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN"
PSSPY_PATH = r"C:\Program Files (x86)\PTI\PSSE33\PSSPY27"

ACTIVE_CASE = "kpgc"
CSV_ROOT_DIR = r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\csv"

CASE_PRESETS = {
    "ieee9": {
        "CASE_TAG": "IEEE_9",
        "BASE_DIR": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\outs\ieee9_scan",
        "IN_DIR_OVERRIDE": None,
        "RUN_GLOB": r"IEEE9_Line_*_Gen_*",
    },
    "ieee118": {
        "CASE_TAG": "IEEE_118",
        "BASE_DIR": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\outs\ieee118_scan",
        "IN_DIR_OVERRIDE": r"C:\Users\pspo\Desktop\pspo\MinHwan\Trips\outs\ieee118_scan\IEEE118_Line_177_Gen_1_20260111_214232",
        "RUN_GLOB": r"IEEE118_Line_*_Gen_*",
    },
    "kpgc": {
        "CASE_TAG": "KPGC",
        "BASE_DIR": r"C:\Users\pspo\Desktop\pspo\MinHwan\OSC\outs\kpgc_scan",
        "IN_DIR_OVERRIDE": None,
        "RUN_GLOB": r"KPGC_*Line_*_Gen_*",
    },
}

EXPORT_DEFAULTS = {
    "VOLT": True,
    "FREQ": False,
    "P": False,
    "Q": False,
}

# Output side artifacts
WRITE_FILE_RESULTS = True
WRITE_UNCLASSIFIED_LOG = True

