# OSC Streaming Modular Structure

## Prerequisites

- Python: 3.12 recommended
- Required packages:

```powershell
py -3.12 -m pip install -U numpy pandas
```

## Module Map

- `osc_modul/osc_state_modul.py`
  - Main FSM states and burst sidecar states
- `osc_modul/osc_config_modul.py`
  - Detector config sections and presets
  - Includes `BurstConfig` and `BurstPolicyConfig`
  - Includes input source mode config (`replay_csv`, `live_csv_tail`, `test_signal`)
- `osc_modul/osc_core_signal_modul.py`
  - Signal feature/math helpers
- `osc_modul/osc_core_fsm_modul.py`
  - Sustained-risk main FSM
- `osc_modul/osc_core_burst_modul.py`
  - Short high-frequency burst sidecar detector
  - Emits `burst_on`, `burst_off`, `burst_interval_open`, `burst_interval_final`
- `osc_modul/osc_alert_policy_modul.py`
  - Risk interval operator policy (`operator_alert`)
- `osc_modul/osc_alert_burst_policy_modul.py`
  - Burst interval policy (`burst_operator_alert`)
- `osc_modul/osc_core_mp_modul.py`
  - Matrix Pencil post-analysis
- `osc_modul/osc_core_prony_modul.py`
  - Prony post-analysis
- `osc_modul/osc_core_postprep_modul.py`
  - MP/Prony preprocessing
- `osc_modul/osc_io_modul.py`
  - CSV replay batches, live CSV tail batches, test signal batches
- `osc_modul/osc_runtime_modul.py`
  - Runtime orchestration (main FSM + burst sidecar)

## Semantic Contract (Official API)

- Official grouped access paths:
  - `tick.signal`, `tick.quality`, `tick.vote`
  - `st.signal`, `st.votes`, `st.cache`
- Timestamp semantics (staged migration):
  - `st.signal.active_start_t` remains the current behavior-driving anchor.
  - `st.signal.candidate_start_t`, `st.signal.confirmed_start_t`, `st.signal.capture_start_t`
    are shadow timelines for parallel observation/debug only.
- Vote source-of-truth vs debug mirror:
  - Source-of-truth: `st.votes.on_short_votes.sum`
  - Debug mirror: `st.signal.on_candidate_streak`
- Context layering:
  - `DecisionContext`: ON-entry gate summary for current tick.
  - `TransitionContext`: off-age/rearm/handoff/re-on transition context.
- Official detector meaning names in new code:
  - `gate_*`, `feature_*`, `state_*`
- Compatibility-only layers (do not add new usage):
  - Flat forwarding on `TickFeatures` / `ChannelStreamState`
  - Legacy export wrapper `_build_risk_event_metrics(...)`
- Producer/storage boundary:
  - Semantic quality snapshots are lowered into legacy storage only through
    `_build_tick_quality_state_from_snapshot()`.

## Input Modes

- `replay_csv`
  - Test/replay mode from static CSV
- `live_csv_tail`
  - Production-like mode
  - Reads a growing CSV every `update_sec` and emits one update batch per tick
  - Use `--live-max-updates` for bounded runs
- `test_signal`
  - Built-in synthetic signal mode for quick runtime checks
  - No CSV required

## Run Examples

Help:

```powershell
py -3.12 -m osc_modul.osc_runtime_modul --help
```

Semantic boundary guard:

```powershell
& "C:\Users\danB\AppData\Local\Programs\Python\Python312\python.exe" tools\check_semantic_boundaries.py
```

Replay test:

```powershell
py -3.12 -m osc_modul.osc_runtime_modul --input-mode replay_csv --csv "<path>" --no-print-tick
```

Live CSV tail (production-like):

```powershell
py -3.12 -m osc_modul.osc_runtime_modul --input-mode live_csv_tail --csv "<path>" --update-sec 2 --live-max-updates 0
```

Built-in test signal:

```powershell
py -3.12 -m osc_modul.osc_runtime_modul --input-mode test_signal --test-duration-sec 120 --update-sec 2 --no-print-tick
```

Burst control:

```powershell
py -3.12 -m osc_modul.osc_runtime_modul --no-burst
py -3.12 -m osc_modul.osc_runtime_modul --no-burst-policy
```
