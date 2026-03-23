"""Modular entrypoint for the streaming detector.

Original monolithic implementation remains in `OSC_streaming.py`.
This module intentionally re-exports only stable public interfaces.
"""

try:
    from .osc_config_modul import (
        DetectorConfig,
        PRESET_BALANCED,
        PRESET_CHOICES,
        PRESET_SAFE,
        PRESET_SENSITIVE,
        make_preset_config,
    )
    from .osc_runtime_modul import main, run_streaming_alert_demo_one_channel
except ImportError:
    # Fallback for direct script execution modes.
    from osc_modul.osc_config_modul import (
        DetectorConfig,
        PRESET_BALANCED,
        PRESET_CHOICES,
        PRESET_SAFE,
        PRESET_SENSITIVE,
        make_preset_config,
    )
    from osc_modul.osc_runtime_modul import main, run_streaming_alert_demo_one_channel

__all__ = [
    "DetectorConfig",
    "PRESET_SAFE",
    "PRESET_BALANCED",
    "PRESET_SENSITIVE",
    "PRESET_CHOICES",
    "make_preset_config",
    "run_streaming_alert_demo_one_channel",
    "main",
]


if __name__ == "__main__":
    main()
