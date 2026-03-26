import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from osc_modul.OSC_streaming_modul import run_streaming_alert_demo_one_channel, make_preset_config
except ImportError:
    from OSC_streaming_modul import run_streaming_alert_demo_one_channel, make_preset_config


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture streaming events to JSON for regression checks.")
    p.add_argument("csv", help="Input CSV path")
    p.add_argument("channel", help="Target channel (e.g. V1)")
    p.add_argument("preset", help="Preset (safe|balanced|sensitive)")
    p.add_argument("out_json", help="Output JSON file path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    events: list[dict] = []
    cfg = make_preset_config(args.preset)

    def on_event(ev: dict) -> None:
        events.append(_to_jsonable(ev))

    run_streaming_alert_demo_one_channel(
        args.csv,
        cfg=cfg,
        target_channel=args.channel,
        on_event=on_event,
        input_mode="replay_csv",
        update_sec=2.0,
        window_sec=8.0,
        print_tick=False,
        realtime_sleep=False,
        console_event_only=True,
        log_to_file=True,
        log_file_path="logs/stream_runtime.log",
        min_interval_sec_for_alert=8.0,
        stitch_gap_sec=6.0,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"WROTE: {out_path} (events={len(events)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
