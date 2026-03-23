# -*- coding: utf-8 -*-
"""
no-first 모듈 파이프라인 실행 러너.

실행 순서:
1) 2차 필터(no-first)
2) lineup 집계

각 단계는 별도 subprocess에서 순차 실행해 장애 영향을 분리한다.
모든 단계 워커 수는 MODULE_MAX_WORKERS 단일 값으로 통일한다.
"""

import json
import os
import subprocess
import sys

import config_paths_no_first as cfg


ROOT = os.path.dirname(os.path.abspath(__file__))


_SUBPROC_BOOTSTRAP = r"""
import importlib
import json
import sys

payload = json.loads(sys.argv[1])
module = importlib.import_module(payload["module"])
kwargs = payload.get("kwargs") or {}
cfg_kwargs = payload.get("cfg_kwargs")

if cfg_kwargs is not None:
    kwargs["cfg"] = module.LineupConfig(**cfg_kwargs)

func = getattr(module, payload["func"])
result = func(**kwargs)
if isinstance(result, str):
    print("RESULT_PATH:", result)
"""


def _run_module_function_subprocess(
    module_name: str,
    func_name: str,
    *,
    kwargs: dict | None = None,
    cfg_kwargs: dict | None = None,
) -> None:
    """지정 모듈 함수를 별도 subprocess에서 실행한다."""
    payload = {
        "module": str(module_name),
        "func": str(func_name),
        "kwargs": kwargs or {},
        "cfg_kwargs": cfg_kwargs,
    }
    cmd = [
        sys.executable,
        "-u",
        "-c",
        _SUBPROC_BOOTSTRAP,
        json.dumps(payload, ensure_ascii=True),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    print("\n>>>", f"{module_name}.{func_name} (subprocess)")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _count_csv_rows(path: str) -> int:
    """CSV 데이터 행 수(헤더 제외)를 반환한다."""
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def _input_dirs() -> list[str]:
    """입력 전압 CSV 디렉터리 목록을 반환한다."""
    if hasattr(cfg, "get_csv_input_dirs"):
        return list(cfg.get_csv_input_dirs())
    return [os.path.abspath(cfg.CSV_DIR)]


def _safe_int(v, default: int) -> int:
    """정수 변환 실패 시 default를 반환한다."""
    try:
        return int(v)
    except Exception:
        return int(default)


def _resolve_module_workers() -> int:
    """MODULE_MAX_WORKERS를 최소 1로 보정해 반환한다."""
    return max(1, _safe_int(getattr(cfg, "MODULE_MAX_WORKERS", 3), 3))


def main() -> None:
    """no-first(2차 -> lineup) 파이프라인을 순차 실행한다."""
    os.environ.setdefault("PYTHONHASHSEED", "0")

    input_dirs = _input_dirs()
    module_workers = _resolve_module_workers()
    if not input_dirs:
        raise FileNotFoundError("No input directories configured.")
    for d in input_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError("CSV input dir not found: %s" % d)

    cfg.ensure_dirs(create_input_dir=False)

    print("[RUN_TAG]    ", cfg.RUN_TAG)
    print("[IN_DIRS]    ", input_dirs)
    print("[OUT_DIR]    ", os.path.abspath(cfg.OUT_DIR))
    print("[SECOND_OUT] ", os.path.abspath(cfg.NOFIRST_SECOND_CSV))
    print("[LINEUP_OUT] ", os.path.abspath(cfg.NOFIRST_LINEUP_OUT_CSV))
    print("[LINEUP_OK0] ", os.path.abspath(cfg.NOFIRST_LINEUP_WITHIN_LIMIT_CSV))
    print("[WORKERS]    ", module_workers)

    print("\n[RUN] 2nd gate runner (no-first module subprocess)")
    _run_module_function_subprocess(
        "OSC_second_filter_runner_module_no_first_ver",
        "run_second_filter_no_first",
        kwargs={
            "max_workers": module_workers,
            "voltage_dirs": input_dirs,
            "out_csv": cfg.NOFIRST_SECOND_CSV,
        },
    )
    second_out = os.path.abspath(cfg.NOFIRST_SECOND_CSV)
    if not os.path.isfile(second_out):
        raise FileNotFoundError("2nd(no-first) output not found: %s" % second_out)
    print("[DONE] 2nd(no-first)")
    print("[SECOND_OUT]", second_out)
    print()

    print("[RUN] lineup post-processing (module subprocess)")
    _run_module_function_subprocess(
        "lineup_parallel_Fin_usecols_ver",
        "run_lineup",
        kwargs={
            "in_csv_paths": [second_out],
            "voltage_dir": input_dirs,
            "out_csv": cfg.NOFIRST_LINEUP_OUT_CSV,
            "within_limit_out_csv": cfg.NOFIRST_LINEUP_WITHIN_LIMIT_CSV,
            "return_df": False,
        },
        cfg_kwargs={"max_workers": module_workers},
    )
    if not os.path.isfile(cfg.NOFIRST_LINEUP_OUT_CSV):
        raise FileNotFoundError("lineup output not found: %s" % os.path.abspath(cfg.NOFIRST_LINEUP_OUT_CSV))
    print("[DONE] lineup")
    print("[LINEUP_ROWS]", _count_csv_rows(cfg.NOFIRST_LINEUP_OUT_CSV))
    print("[LINEUP_OUT ]", os.path.abspath(cfg.NOFIRST_LINEUP_OUT_CSV))
    print("[LINEUP_OK0 ]", os.path.abspath(cfg.NOFIRST_LINEUP_WITHIN_LIMIT_CSV))
    print()

    print("ALL DONE")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
