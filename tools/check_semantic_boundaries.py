"""Guard semantic API boundaries for detector/runtime code.

Checks:
1) Forbid new flat forwarding usage:
   - disallow tick.<flat> (except tick.signal/tick.quality/tick.vote)
   - disallow st.<flat> (except st.signal/st.votes/st.cache)
2) Forbid new calls to legacy export wrapper _build_risk_event_metrics(...).
3) Keep TickQualityState construction centralized in
   _build_tick_quality_state_from_snapshot().
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "osc_modul"
ALLOWED_TICK_SECTIONS = {"signal", "quality", "vote"}
ALLOWED_STATE_SECTIONS = {"signal", "votes", "cache"}


class BoundaryVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[tuple[int, str]] = []
        self._fn_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn_stack.append(node.name)
        self.generic_visit(node)
        self._fn_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._fn_stack.append(node.name)
        self.generic_visit(node)
        self._fn_stack.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            if node.value.id == "tick" and node.attr not in ALLOWED_TICK_SECTIONS:
                self.errors.append(
                    (
                        node.lineno,
                        (
                            f"legacy flat access `tick.{node.attr}` is not allowed; "
                            "use tick.signal/tick.quality/tick.vote"
                        ),
                    )
                )
            if node.value.id == "st" and node.attr not in ALLOWED_STATE_SECTIONS:
                self.errors.append(
                    (
                        node.lineno,
                        (
                            f"legacy flat access `st.{node.attr}` is not allowed; "
                            "use st.signal/st.votes/st.cache"
                        ),
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_legacy_wrapper_call(node.func):
            self.errors.append(
                (
                    node.lineno,
                    "legacy wrapper `_build_risk_event_metrics(...)` call is not allowed",
                )
            )
        if self._is_tick_quality_state_ctor(node.func) and not self._inside_quality_factory():
            self.errors.append(
                (
                    node.lineno,
                    (
                        "TickQualityState construction must stay in "
                        "_build_tick_quality_state_from_snapshot()"
                    ),
                )
            )
        self.generic_visit(node)

    @staticmethod
    def _is_legacy_wrapper_call(func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id == "_build_risk_event_metrics"
        if isinstance(func, ast.Attribute):
            return func.attr == "_build_risk_event_metrics"
        return False

    @staticmethod
    def _is_tick_quality_state_ctor(func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id == "TickQualityState"
        if isinstance(func, ast.Attribute):
            return func.attr == "TickQualityState"
        return False

    def _inside_quality_factory(self) -> bool:
        return (
            self.path.name == "osc_core_fsm_modul.py"
            and "_build_tick_quality_state_from_snapshot" in self._fn_stack
        )


def _iter_target_files() -> list[Path]:
    return sorted(
        p
        for p in TARGET_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def main() -> int:
    failures: list[tuple[Path, int, str]] = []
    for path in _iter_target_files():
        src = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError as exc:
            failures.append((path, int(exc.lineno or 1), f"SyntaxError: {exc.msg}"))
            continue
        visitor = BoundaryVisitor(path=path)
        visitor.visit(tree)
        for lineno, message in visitor.errors:
            failures.append((path, lineno, message))

    if failures:
        print("semantic boundary check: FAILED")
        for path, lineno, message in failures:
            rel = path.relative_to(ROOT)
            print(f"- {rel}:{lineno}: {message}")
        return 1

    print("semantic boundary check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
