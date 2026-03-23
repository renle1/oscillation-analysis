"""Modular boundaries for OSC streaming detector.

Keep package imports side-effect free so `python -m osc_modul.osc_runtime_modul`
does not preload the runtime module via package import.
"""


def main() -> None:
    """Lazy package-level entrypoint."""

    from .osc_runtime_modul import main as _main

    _main()


__all__ = ["main"]
