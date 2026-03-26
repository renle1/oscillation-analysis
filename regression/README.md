# Regression Stability Harness

This folder is for regression stability checks (not accuracy improvement).

## Goal

1. Same input -> same event count
2. Same input -> same event order and interval count
3. Same input -> same interval start/end timestamps
4. Runtime should not noticeably degrade

## Fixed Conditions

Keep these identical for baseline and candidate runs:

- input file path
- channel
- preset
- input mode
- update sec
- window sec
- print tick
- realtime sleep

Default fixed stream values used by `capture_events.py`:

- `input_mode="replay_csv"`
- `update_sec=2.0`
- `window_sec=8.0`
- `print_tick=False`
- `realtime_sleep=False`
- `console_event_only=True`
- `log_to_file=True`
- `log_file_path="logs/stream_runtime.log"`
- `min_interval_sec_for_alert=8.0`
- `stitch_gap_sec=6.0`

## Folder Layout

- `baseline/`: outputs from pre-change code
- `candidate/`: outputs from post-change code
- `scripts/`: external harness scripts
- `reports/`: comparison summary files

## Python 3.12

Use this interpreter in this environment:

```powershell
$PY = "C:\Users\pspo\AppData\Local\Programs\Python\Python312\python.exe"
```

## CLI Baseline/Candidate Capture

Use module execution (from repo root):

```powershell
& $PY -m osc_modul.OSC_streaming_modul --csv "<CASE_CSV>" --channel V1 --preset safe --input-mode replay_csv --update-sec 2.0 --window-sec 8.0 --no-print-tick > regression\baseline\case_cli.txt
& $PY -m osc_modul.OSC_streaming_modul --csv "<CASE_CSV>" --channel V1 --preset safe --input-mode replay_csv --update-sec 2.0 --window-sec 8.0 --no-print-tick > regression\candidate\case_cli.txt
```

Compare:

```powershell
Compare-Object (Get-Content regression\baseline\case_cli.txt) (Get-Content regression\candidate\case_cli.txt)
```

## JSON Event Capture

```powershell
& $PY regression\scripts\capture_events.py "<CASE_CSV>" V1 safe regression\baseline\case_events.json
& $PY regression\scripts\capture_events.py "<CASE_CSV>" V1 safe regression\candidate\case_events.json
```

Compare:

```powershell
& $PY regression\scripts\compare_events.py regression\baseline\case_events.json regression\candidate\case_events.json
```

## Runtime Check (PowerShell)

```powershell
Measure-Command { & $PY -m osc_modul.OSC_streaming_modul --csv "<CASE_CSV>" --channel V1 --preset safe --input-mode replay_csv --update-sec 2.0 --window-sec 8.0 --no-print-tick > regression\candidate\case_cli.txt }
```

## Recommended 3 Cases

- Spain-Portugal surrogate
- OG&E 13.33 Hz surrogate
- North Scotland 8 Hz surrogate

## Decision

- PASS: CLI match + JSON exact match + no meaningful runtime degradation
- WARN: event count same, but timing/reason drift
- FAIL: count/order/interval/timing mismatch or large runtime regression
