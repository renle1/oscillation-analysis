# Streaming Detector Conversation Log

Last updated: 2026-03-17

## Scope (Pruned)
- This log was pruned to keep only recent items relevant to current modularization/refactor work.
- Older sections that were far from current work (mostly 2026-03-04 to 2026-03-09) were removed.

## Append Policy
Append new session updates to this same file.

---

## Session Note (2026-03-10, next-step review order agreed)

### User review order
1. Check whether `min_interval_sec_for_alert=8` is the direct cause of `event=0`.
2. Identify per-file blocking gates among `cf`, `lr`, `wr`, `rg`, `sn`.
3. Prefer tuning `long_ready` / baseline bottlenecks and `short_no_long_ready` realism before global confidence relaxation.

### Assistant assessment
- The order is appropriate and risk-aware.
- `min_interval_sec_for_alert` is expected to affect event emission timing more than core ON decision; verify with A/B (`8` vs `0`) first.
- Main tuning leverage should remain in structural bottlenecks (`long_ready`/baseline and controlled `short_no_long_ready` path), with global confidence relaxation treated as a later option.

## Session Update (2026-03-10, md-followup ordered validation + application)

### Scope
- Followed the agreed review order in `Session Note (2026-03-10)` and executed each step end-to-end.
- Target action: test first, then apply only no-regression changes.

### 1) `min_interval_sec_for_alert` direct-cause check (`8` vs `0`)
- A/B on `C:\Users\pspo\Desktop\pspo\DB\sample_data` CSV set using `safe` preset.
- Observed direct effect:
  - `sample_5ch_256Hz_60s.csv`: `events 0 -> 3` when `8 -> 0`
  - `wmu_voltage_mock_hard_5min_3ch_256Hz.csv`: `events 0 -> 3` when `8 -> 0`
- Interpretation:
  - `min_interval_sec_for_alert=8` is a direct emission suppressor for short ON intervals on some files.
  - It is not the only reason for `events=0` across all files (`wmu_bottleneck_5min_256Hz_5ch.csv` and others remained `0`).

### 2) Per-file blocking gates among `cf/lr/wr/rg/sn` (baseline `min_interval=8`)
- Tick-log gate parsing (`G[...]`) summary:
  - `sample_5ch_256Hz_60s.csv`: dominant blocks at `lr/sn` (both 100%), `wr` mostly 0
  - `wmu_bottleneck_5min_256Hz_5ch.csv`: `cf/lr/wr/sn` all 100% blocking
  - `wmu_voltage_mock_5min_256Hz.csv`: dominant `cf` blocking
  - `wmu_voltage_mock_hard_5min_3ch_256Hz.csv`: mostly `cf` blocking
  - `wmu_voltage_random_1min_256Hz.csv`: dominant `lr/wr/sn` blocking
- `rg` was not a dominant blocker on this set.

### 3) Structural tuning only (`long_ready` / baseline / `short_no_long_ready`)
- Swept structural parameters around current defaults (no confidence-relaxation):
  - `warmup_min_baseline`, `warmup_on_ratio`, `short_dynamic_min_baseline`
  - additional warmup structural controls (`warmup_on_votes_*`, `warmup_on_confirm_min_sec`)
- Result:
  - Earlier hard-file ON start (`53.996s -> 51.996s` or `49.996s`) was achievable only with new false positives (notably `sample_5ch_256Hz_60s.csv`).
  - No strictly better (earlier + no-regression) structural point was found on current sample set.

### Applied change
- Added CLI override to reproduce step-1 verification without editing code defaults:
  - `OSC_streaming.py`: `--min-interval-sec-for-alert`
- Default behavior unchanged when option is omitted.

## Session Update (2026-03-10, summary disambiguation columns added)

### Applied
- Added stream-level disambiguation fields in `OSC_streaming.py` final summary print:
  - `raw_risk_interval_exists`
  - `suppressed_event_exists`
  - plus counts: `raw_intervals`, `suppressed_intervals`
- Updated `cscv/_tmp_recheck_cases.py` summary row to include the same columns.

### Why
- Prevent confusion when `events=0` but a raw risk interval existed and was suppressed by `min_interval_sec_for_alert`.

### Quick check
- `sample_5ch_256Hz_60s.csv` (`safe`, default min=8):
  - `[STREAM_DONE] ... events=0 ... raw_risk_interval_exists=1 ... suppressed_event_exists=1`
- This confirms zero emitted events can still carry suppressed raw intervals.

## Session Note (2026-03-10, user-provided optimization roadmap)

# OSC_streaming.py 理쒖쟻??諛?援ъ“ ?⑥닚??寃??
## ?붿빟

?꾩옱 蹂묐ぉ? ?쒖“嫄댁씠 留롮븘?쒋??먯껜蹂대떎, **臾닿굅??怨꾩궛??留?tick留덈떎 留롮씠 ?ㅼ떆 ?섑뻾?섍퀬**, **鍮꾩듂??寃곗젙???щ윭 ?④퀎?먯꽌 以묐났 怨꾩궛?섎뒗 援ъ“**????媛源앸떎. ?뱁엳 硫붿씤 援ы쁽 ?⑥닔媛 ?ㅼ젙 寃利? feature 怨꾩궛, 寃뚯씠???먯젙, ?곹깭 ?꾩씠, ?대깽??異쒕젰源뚯? ?쒓납??紐곕젮 ?덉뼱 蹂묐ぉ ?꾩튂 ?뚯븙怨?由ы뙥?좊쭅???대졄??  

---

## ?듭떖 臾몄젣

### 1. 硫붿씤 ?⑥닔????븷??怨쇰룄?섍쾶 吏묒쨷??
`_run_streaming_alert_demo_one_channel_impl`媛

* 湲??뚮씪誘명꽣 ?쒓렇?덉쿂
* ?洹쒕え validation
* ?ㅽ듃由щ컢 猷⑦봽
* score/feature 怨꾩궛
* warmup/long/CAL_ON ?먯젙
* FSM ?꾩씠
* ?대깽??emission

??紐⑤몢 ?대떦?섍퀬 ?덈떎. ??援ъ“??媛?낆꽦???⑥뼱?⑤┫ 肉??꾨땲?? ?뱀젙 蹂묐ぉ留?遺꾨━??理쒖쟻?뷀븯湲??대졄寃?留뚮뱺??  

### 2. 留?tick留덈떎 臾닿굅??怨꾩궛??諛섎났??
`score_one_channel_equiv()` ?덉뿉??`pd.Series(...).rolling(...).median()`???ъ슜?섍퀬 ?덈떎. ?ㅽ듃由щ컢 猷⑦봽?먯꽌 pandas 媛앹껜瑜?留ㅻ쾲 ?앹꽦?섍퀬 rolling median???뚮━??寃껋? 鍮꾩슜???щ떎. ?ㅼ떆媛?泥섎━ ?깅뒫 痢〓㈃?먯꽌 媛??癒쇱? ?섏떖?????덈뒗 蹂묐ぉ 以??섎굹?? 

### 3. 鍮꾩떬 二쇨린???덉쭏 怨꾩궛???꾨낫 援ш컙???꾨땲?대룄 ?먯＜ 媛쒖엯??
二쇨린???덉쭏 怨꾩궛? detrend, FFT 湲곕컲 band limit, Welch peak, zero-crossing frequency, ACF peak, envelope stability ?깆쓣 ?ы븿?쒕떎. ??怨꾩궛 ?ㅽ깮? 臾닿쾪怨? 罹먯떆 ?ъ궗??議곌굔??蹂댁닔?곸씠???꾨낫 援ш컙?먯꽌???ш퀎?곗씠 ?먯＜ ?쇱뼱?????덈떎.     

### 4. 鍮꾩듂???먯젙???щ윭 ?④퀎?먯꽌 ?ㅼ떆 怨꾩궛??
肄붾뱶??????꾨옒 ?쒖꽌濡??섏궗寃곗젙??留뚮뱺??

* `long_ready`
* `warmup_core`
* CAL_ON soft support
* `on_conf_ok`
* `on_long_gate_ok`
* `short_no_long_ready`
* `rms_decay_local_on_ok`
* 理쒖쥌 `on_entry_ready`

利?鍮꾩듂???낅젰??諛뷀깢?쇰줈 ?щ윭 遺덈━?몄쓣 ?꾩쟻??寃곗젙??諛섎났 援ъ꽦?쒕떎. ?쇰━?곸쑝濡쒕뒗 留욎쓣 ???덉?留? 援ъ“?곸쑝濡쒕뒗 以묒꺽怨?以묐났??而ㅼ쭊??    

### 5. ?대깽??異쒕젰 濡쒖쭅??湲멸퀬 以묐났??
`risk_on`, `risk_off`, `risk_interval_open` ?앹꽦 ???좎궗??硫뷀듃由?payload瑜?諛섎났?곸쑝濡?議곕┰?쒕떎. suppress 泥섎━? raw interval 湲곕줉源뚯? 遺숈쑝硫댁꽌 event emission 愿??肄붾뱶媛 湲몄뼱吏怨??좎?蹂댁닔?깆씠 ?⑥뼱吏꾨떎.  

### 6. ?ъ슜?섏? ?딄굅???먮쫫???먮━???뚮씪誘명꽣媛 ?덉쓬

`warmup_on_conf_raw_min`? ?ㅼ젙媛믪쑝濡?議댁옱?섍퀬 validation/plumbing?먮룄 ?ㅼ뼱媛吏留? warmup core ?먯젙?앹뿉???ㅼ젣濡??ъ슜?섏? ?딅뒗?? ?꾩옱 湲곗??쇰줈??dead parameter??媛源앸떎. ?대윴 媛믪? 肄붾뱶 ?몄? 遺?섎쭔 ?믪씤??  

---

## ?깅뒫 愿?먯뿉??癒쇱? ?섏떖??遺遺?
### 1?쒖쐞: pandas rolling median ?쒓굅

媛??吏곸젒?곸씤 ?깅뒫 媛쒖꽑 媛??吏?먯씠??

?꾩옱:

* `pd.Series(...).rolling(...).median()`

?꾨낫:

* rolling mean?쇰줈 ?⑥닚??* numpy 湲곕컲 洹쇱궗 detrend
* short stage ?꾩슜 ?鍮꾩슜 detrend濡?援먯껜

?뺥솗?꾨낫???ㅼ떆媛꾩꽦????以묒슂?섎떎硫? ?ш린???삳뒗 泥닿컧 ?대뱷??媛????媛?μ꽦???믩떎. 

### 2?쒖쐞: deque ??numpy 蹂??諛섎났 媛먯냼

留?tick留덈떎 `np.fromiter`濡?window瑜??ъ깮?깊븯???⑦꽩? ?꾩쟻 鍮꾩슜???щ떎. ring buffer瑜?numpy 湲곕컲?쇰줈 ?좎??섎㈃ ??鍮꾩슜??以꾩씪 ???덈떎. 

### 3?쒖쐞: 二쇨린???덉쭏 怨꾩궛??吏???됯?

吏湲덉? 臾닿굅??periodicity quality瑜??덈Т ?대Ⅸ ?④퀎?먯꽌 ?먯＜ 怨꾩궛?쒕떎.
?ㅼ쓬怨?媛숈? cheap pre-gate瑜?癒쇱? ?먮뒗 寃껋씠 ?ル떎.

* `score_reason_ok == False`?대㈃ skip
* `score < low_floor`?대㈃ skip
* 理쒓렐 N tick ?곗냽 ?쏀븳 ?곹깭硫?skip

利? ?쒗썑蹂댁꽦???대뒓 ?뺣룄 ?뺣낫?먯쓣 ?뚮쭔 鍮꾩떬 怨꾩궛?앹쑝濡??ㅻ줈 誘몃（??寃?留욌떎. 

---

## 援ъ“ ?⑥닚??諛⑺뼢

### 1. Feature / Gate / FSM / Emit 4??遺꾨━

?꾩옱??紐⑤뱺 ?④퀎媛 ??猷⑦봽 ?덉뿉 ?ㅼ꽎???덈떎.
?꾨옒泥섎읆 遺꾨━?섎뒗 寃껋씠 醫뗫떎.

#### Feature stage

怨꾩궛留??섑뻾:

* short score
* long stats
* confidence
* RMS decay

#### Gate stage

遺덈━??寃뚯씠?몃쭔 ?뺣━:

* `short_trigger`
* `long_ready`
* `warmup_confirmed`
* `conf_ok`
* `rms_on_ok`
* `off_vote_core`
* `force_off_now`

#### FSM stage

寃뚯씠??寃곌낵留?諛쏆븘 ?곹깭 ?꾩씠:

* OFF
* ON_CANDIDATE
* ON_CONFIRMED
* ON
* OFF_CANDIDATE

#### Emit stage

event payload ?앹꽦 諛?suppress/raw interval 泥섎━

?대젃寃??섎㈃ ?먯젙 ?쇰━? 異쒕젰 ?쇰━瑜?遺꾨━?????덇퀬, ?붾쾭源낅룄 ?ъ썙吏꾨떎.

### 2. GateFlags 援ъ“泥??꾩엯

?⑹뼱吏?遺덈━?몄쓣 ??媛앹껜濡?臾띠쑝硫?媛?낆꽦???ш쾶 醫뗭븘吏꾨떎.

?덉떆:

```python
@dataclass
class GateFlags:
    short_trigger: bool
    long_ready: bool
    warmup_confirmed: bool
    conf_ok: bool
    rms_on_ok: bool
    off_vote_core: bool
    force_off_now: bool
```

?댄썑 FSM? `GateFlags`留?蹂닿퀬 ?吏곸씠寃??섎㈃ ?쒕떎.

### 3. validation ?꾩쟾 遺꾨━

?꾩옱 validation? ?ㅽ뻾 猷⑦봽? 媛숈? ?⑥닔 ?덉뿉 ?덉뼱 ?⑥닔媛 吏?섏튂寃?而ㅼ죱??
`validate_detector_runtime_args()` 媛숈? 蹂꾨룄 ?⑥닔濡?鍮쇰뒗 寃껋씠 醫뗫떎. 

### 4. emission helper 遺꾨━

?꾨옒 helper濡??섎늻??寃껋씠 醫뗫떎.

* `emit_risk_on(...)`
* `emit_risk_off(...)`
* `emit_open_interval(...)`

?대젃寃??섎㈃ raw/suppressed interval 媛쒕뀗??異붽??대룄 援ъ“媛 ??吏?遺꾪빐吏꾨떎. 

---

## 異붿쿇 由ы뙥?좊쭅 ?곗꽑?쒖쐞

### 1?④퀎

`score_one_channel_equiv()`?먯꽌 pandas rolling median ?쒓굅

### 2?④퀎

periodicity quality瑜?cheap pre-gate ?ㅻ줈 ?대룞

### 3?④퀎

硫붿씤 impl ?⑥닔 遺꾨━:

* `extract_features`
* `evaluate_gates`
* `step_fsm`
* `emit_events`

### 4?④퀎

dead parameter ?쒓굅 ?먮뒗 ?ㅼ젣 ?ъ슜 ?곌껐

* `warmup_on_conf_raw_min`

### 5?④퀎

?대깽??emission helper ?뺣━ 諛?payload 以묐났 ?쒓굅

---

## 寃곕줎

?꾩옱 臾몄젣???쒖“嫄댁씠 留롫떎?앸뒗 寃??먯껜蹂대떎, **臾닿굅??feature 怨꾩궛怨?寃뚯씠???먯젙, FSM, ?대깽??異쒕젰?????ㅽ듃由щ컢 猷⑦봽 ?덉뿉??以묒꺽?섍퀬 ?ш퀎?곕릺??援ъ“**???덈떎.
媛??癒쇱? ?먮뙂 怨녹?:

1. pandas rolling median ?쒓굅
2. 鍮꾩떬 periodicity 怨꾩궛??吏???됯?
3. 嫄곕???impl ?⑥닔???④퀎蹂?遺꾪빐

????媛吏?? ???쒖꽌濡?媛硫?**肄붾뱶媛 ??踰덉옟?댁?怨? 泥닿컧 ?깅뒫 媛쒖꽑??媛???ш쾶 ?살쓣 媛?μ꽦**???믩떎.

## Session Update (2026-03-10, optimization roadmap step 1-2)

### Applied Step 1
- Replaced pandas centered rolling median in `score_one_channel_equiv()` with NumPy centered rolling mean helper.
- New helper: `_rolling_mean_centered_np(...)`.

### Applied Step 2
- Added cheap pre-gate before heavy periodicity quality compute.
- In weak non-candidate ticks, heavy periodicity stack is skipped and cached quality is reused.
- Candidate/risk/calibration contexts still compute quality as before.

### Validation
- `python.exe -m py_compile OSC_streaming.py` -> PASS
- Sample-set regression (`C:\Users\pspo\Desktop\pspo\DB\sample_data`) preserved:
  - hard single-channel: `events=3`, first `risk_on.start_t=53.996s`
  - other listed no-event files remain `events=0`

## Session Update (2026-03-10, performance validation for logic changes)

### Validation policy
- For logic changes, run A/B timing against a baseline-equivalent module before accepting.

### This round benchmark
- Dataset: `C:\Users\pspo\Desktop\pspo\DB\sample_data` (6 CSV)
- Repeats: 3 full passes
- Baseline emulation:
  - short detrend uses pandas centered rolling median
  - no cheap-skip branch for periodicity quality
- Current:
  - NumPy centered rolling mean detrend
  - cheap pre-gate skip for weak non-candidate ticks

### Result
- `BASELINE_SEC = 19.6157`
- `CURRENT_SEC = 16.2432`
- `DELTA_SEC = -3.3724`
- `SPEEDUP_PCT = +17.19%`
- `OUTPUT_EQUIV = 1` (same events/first_on on benchmark set)

## Session Update (2026-03-10, accuracy-first decision)

### User policy
- Accuracy is prioritized over speed for upcoming logic changes.

### Robustness findings
- Extended A/B matrix: 12 cases x 2 presets (`safe`, `balanced`).
- `rolling mean` detrend introduced behavior drift on several `balanced` cases.
- `cheap periodicity skip` alone preserved output equivalence on the matrix.

### Action taken
- Reverted detrend back to pandas centered rolling median.
- Kept cheap-skip branch only, since it passed equivalence checks.

### Rule for next steps
- For each logic change: accept only if equivalence/regression checks pass on the expanded matrix.
- If mismatch appears, rollback or gate it behind explicit opt-in.

## Session Update (2026-03-10, step 3 start: validation extraction)

### Applied
- Extracted runtime argument checks from `_run_streaming_alert_demo_one_channel_impl` into helper:
  - `_validate_detector_runtime_args(...)`
- Impl now calls validation via:
  - `_validate_detector_runtime_args(**locals())`

### Intent
- Structural split for readability/maintainability without changing detector decision logic.

### Validation
- `python.exe -m py_compile OSC_streaming.py` -> PASS
- Quick regression (safe preset) unchanged on `sample_data` 6 files.
- `cscv/_tmp_recheck_cases.py` outputs unchanged summary counts/flags.

## Session Update (2026-03-10, explicit append by user request)

### Requested action
- User requested: append current progress into md file.

### Appended summary
- Extracted runtime validation into `_validate_detector_runtime_args(...)`.
- Impl now calls `_validate_detector_runtime_args(**locals())` before stream loop setup.
- Regression checks remained unchanged on current quick suite.

## Session Update (2026-03-10, step3 structural split continuation with equivalence checks)

### Applied
- Added `GateFlags` dataclass and switched FSM/tick-log gate consumption to `gate_flags` references.
- Moved nested helpers out of `_run_streaming_alert_demo_one_channel_impl(...)` to module scope:
  - `_finite_or_nan(...)`
  - `_build_risk_event_metrics(...)`
  - `_emit_or_stitch_interval(...)`
- Added `WindowSlice` dataclass and `_extract_window_slice(...)`, then replaced inline ring/window preparation block with helper call.

### Validation protocol
- Fixed baseline snapshot first:
  - `cscv/_tmp_step3_baseline.json` (18 rows = `safe/balanced` x 9 CSV)
- After each refactor chunk, re-ran:
  - `python -m py_compile OSC_streaming.py`
  - same 18-row snapshot (`_tmp_step3_after*.json`)
  - strict key-field diff check on:
    - `events`, `risk_on`, `risk_off`, `open`, `closed`, `final`
    - `raw_intervals`, `suppressed_intervals`, `raw_exists`, `suppressed_exists`
    - `first_on_start_t`

### Result
- All rounds: `DIFF=0` on the tracked key fields.
- Decision: keep all above structural changes (behavior-equivalent on current regression set).

### Next-step candidate
- Continue step 3 by extracting explicit phase helpers (feature/gate/fsm/emit boundaries), while preserving the same equivalence-check policy.

## Session Update (2026-03-10, split-first round completed)

### Scope
- Per user instruction, focused only on function splitting first (no dead-parameter cleanup in this round).

### Applied split
- `_extract_features(...)` introduced and wired into tick loop entry.
- `_evaluate_gates(...)` introduced and wired for gate computation (`GateFlags`, `on_entry_vote_sum`).
- `_step_fsm(...)` introduced and wired for OFF/ON candidate/confirmed phase transitions.
- `_emit_events(...)` introduced and wired for in-loop `risk_on`/`risk_off` + interval stitch/suppress handling.

### Validation
- `python -m py_compile OSC_streaming.py` -> PASS
- Regression snapshot:
  - baseline: `cscv/_tmp_step3_baseline.json`
  - after split: `cscv/_tmp_step3_after_split.json`
- Key-field diff: `DIFF=0`
  - checked fields: events/types/count flags + first `risk_on.start_t`

### Note
- Next round can proceed to:
  - dead parameter cleanup (`warmup_on_conf_raw_min`)
  - additional payload duplication reduction for stream-end/open paths

## Session Note (2026-03-10, user-directed next-step priorities)

### 1) Per-tick 蹂몄껜 異붽? 遺꾪빐媛 ?곗꽑
- ?꾩옱 helper媛 ?앷꼈吏留?猷⑦봽 ?대? 以묎컙 怨꾩궛痢듭씠 ?ъ쟾???먭퍖?ㅻ뒗 ?먯쓣 ?ъ슜?먯? ?⑹쓽.
- ?ㅼ쓬 遺꾪빐 紐⑺몴(理쒖냼 3媛?:
  - `_update_baseline_and_long_stats(...)`
  - `_compute_quality_and_support(...)`
  - `_compute_off_path(...)`
- ?붿?:
  - `_extract_features / _evaluate_gates / _step_fsm / _emit_events`留뚯쑝濡쒕뒗 以묎컙 怨꾩궛 ?⑹뼱由ш? ?꾩쭅 ??

### 2) dead parameter ?뺣━ ?먯튃
- ?????ぉ: `warmup_on_conf_raw_min`
- ?좏깮吏????以??섎굹濡?紐낇솗??
  - ?ㅼ젣 誘몄궗?⑹씠硫??쒓굅
  - ?④린?ㅻ㈃ `deprecated` 紐낆떆 + 湲곕낯 寃쎈줈 鍮꾪솢???좎?
- ?쒖꽕?뺤? ?덈뒗???듭떖 ?먯젙??誘몄궗?⒱??곹깭瑜??쇳븯??寃껋씠 紐⑺몴.

### 3) legacy alias ?뺣━ 怨꾪쉷 ?섎┰
- `rms_decay_local*`怨?legacy alias `rms_decay*` ?숈떆 ?좎? ?곹깭??怨쇰룄湲곗쟻?쇰줈 ?덉슜.
- ?몃? ?뚮퉬???뺣━ ?댄썑 ?⑥씪 ?ㅻ줈 異뺤냼 ?덉젙.
- 利됱떆 ??젣媛 ?꾨땲???쒖젣嫄??쒖젏/議곌굔?앹쓣 癒쇱? ?뺤젙?섎뒗 ?④퀎濡?湲곕줉.

### 4) ?깅뒫 理쒖쟻?붾뒗 援ъ“ ?뺣━ ?댄썑
- ?⑥? ?깅뒫 ?꾨낫:
  - `score_one_channel_equiv()`??pandas rolling median
  - `_extract_window_slice()`??tick??`np.fromiter` ?ъ깮??- ?먮떒:
  - 泥??꾨낫??湲곗〈 A/B?먯꽌 drift ?대젰 議댁옱
  - ?섏㎏ ?꾨낫??援ы쁽 ?쒕룄 ?믪쓬
- 寃곕줎:
  - ?뺥솗???곗꽑 ?먯튃 ?좎?
  - 援ъ“ 遺꾪빐 ?꾨즺 ??蹂꾨룄 A/B ?몃옓?먯꽌 ?섑뻾

## Session Update (2026-03-10, prioritized cleanup execution)

### Applied (priority order)
1. `warmup_on_conf_raw_min` rollback ?좎?:
- `warmup_core` ?먯젙?앹뿉??raw-confidence hard gate瑜??ㅼ떆 ?곌껐?섏? ?딆쓬.
- 肄붾뱶 二쇱꽍?쇰줈 rollback ?좎? ?섎룄瑜?紐낆떆.

2. Per-tick 蹂몄껜 以묎컙 怨꾩궛 異붽? 遺꾪빐:
- ?좉퇋 helper 異붽?:
  - `_update_baseline_and_long_stats(...)`
  - `_compute_quality_and_support(...)`
  - `_compute_off_path(...)`
- 湲곗〈 猷⑦봽 ?대???baseline/long ?듦퀎, quality/support, OFF path 怨꾩궛 釉붾줉??helper ?몄텧濡?移섑솚.

3. dead parameter 泥섎━:
- `LongConfig.warmup_on_conf_raw_min`??deprecated 二쇱꽍 異붽?.
- ?고??꾩뿉???대떦 媛믪쓣 湲곕낯媛?0.70)怨??ㅻⅤ寃?二쇰㈃ deprecation 寃쎄퀬瑜?異쒕젰?섎룄濡?異붽?:
  - `_warn_deprecated_warmup_params(...)`

4. legacy alias ?뺣━ 怨꾪쉷 諛섏쁺:
- canonical key ?곗꽑/legacy fallback??怨듯넻 helper濡??듯빀:
  - `_apply_legacy_rms_aliases(...)`
  - `_set_quality_rms_metrics(...)`
  - `_extract_quality_rms_metrics(...)`
- 肄붾뱶 ?댁뿉 ?④퀎??異뺤냼 怨꾪쉷(?꾪뻾 蹂묓뻾 -> ?뚮퉬???닿? -> fallback ?쒓굅)??二쇱꽍?쇰줈 怨좎젙.

### Note
- ?대쾲 ?쇱슫?쒕뒗 援ъ“ 遺꾪빐/?뺣━ 以묒떖?대ŉ, ?숈옉 蹂寃쎌? ?섎룄?섏? ?딆쓬.
- ?깅뒫 理쒖쟻???몃옓? 湲곗〈 ?⑹쓽?濡?援ъ“ ?뺣━ ?댄썑 蹂꾨룄 A/B?먯꽌 ?섑뻾.

## Session Update (2026-03-10, impl API boundary cleanup)

### Scope
- Follow-up to user review: prioritize API boundary cleanup over logic tuning.

### Applied
1. Impl entry moved to config-first:
- `_run_streaming_alert_demo_one_channel_impl(...)` now accepts `cfg` directly.
- Flat-argument runner is retained as internal-only helper:
  - `_run_streaming_alert_demo_one_channel_flat_impl(...)`

2. Stale default mismatch risk reduced:
- Internal flat helper defaults aligned for known mismatch fields:
  - `warmup_min_baseline=10`
  - `warmup_on_ratio=0.57`
- Public path now always enters cfg-based impl boundary.

3. `warmup_on_conf_raw_min` core-path separation:
- Removed from flat impl signature and runtime validator argument list.
- Deprecated handling remains at cfg boundary warning only:
  - `_warn_deprecated_warmup_params(...)`

4. Public wrapper simplification:
- `run_streaming_alert_demo_one_channel(...)` no longer forwards dozens of flat kwargs directly to core logic.
- Wrapper now resolves overrides into `DetectorConfig` and calls cfg-based impl.

### Validation
- `python -m py_compile OSC_streaming.py` -> PASS
- Smoke run (`WMU_SIM_01_V1.csv`) -> PASS (`events=3`, no runtime error)

## Session Update (2026-03-11, legacy alias payload slimming)

### Scope
- Apply alias cleanup earlier by default, while keeping explicit compatibility mode.

### Applied
1. Default emission policy changed:
- Event payload now emits canonical RMS keys (`rms_decay_local*`) by default.
- Legacy alias keys (`rms_decay*`) are no longer emitted unless explicitly enabled.

2. Compatibility toggle added:
- New stream config flag:
  - `StreamConfig.emit_legacy_rms_aliases` (default `False`)
- New CLI option:
  - `--emit-legacy-rms-aliases`

3. Internal quality payload slimmed:
- `_set_quality_rms_metrics(...)` now writes canonical keys only.
- Legacy read fallback remains in `_extract_quality_rms_metrics(...)` for compatibility.

### Note
- This reduces payload bloat immediately without removing the migration escape hatch.
- Next removal step can drop legacy fallback paths after external consumer migration is confirmed.

## Session Note (2026-03-12, hard singlech dip interpretation)

### User observation
- In `wmu_singlech_hard_to_detect_5min_256Hz.csv`, amplitude visibly dips around ~160s.

### Current detector interpretation
- The dip is treated as a temporary weakening within one continuous ON interval, not as a separate OFF->ON split.
- Current summary remains:
  - `risk_on_n=1`
  - `risk_off_n=0`
  - `open_interval_end=299.996...`

### Why this happens (current logic)
- OFF transition path is conservative and requires compound conditions + dwell:
  - OFF core vote condition (`off_vote_core`) + vote persistence
  - ON confirmed hold-down before OFF-candidate transition
  - OFF-candidate dwell before OFF-confirmed

### Presentation wording
- "Waveform shows a local amplitude dip, but detector currently keeps one continuous risk interval (no OFF split)."

## Session Update (2026-03-12, A/B comparison and B adoption)

### Scope
- Compared two CAL_ON-side tuning ideas in sequence on the current code:
  - Option A: raise `confidence_w_acf` from `0.34` to `0.40`
  - Option B: add ACF bonus to CAL_ON score-support path (no hard gate)

### Implemented
1. Added CAL_ON score-support ACF bonus knob:
- `cal_on_support_score_acf_bonus` introduced in periodicity config/runtime path.
- ACF bonus is applied only inside `score_support` composition, then clipped to `[0, 1]`.
- Runtime validation added (`0 <= bonus <= 1`).

2. Added CLI overrides for quick experiments:
- `--confidence-w-acf`
- `--cal-on-support-score-acf-bonus`

3. Ran 9-case comparison matrix (safe preset + confidence calibration enabled):
- 3 WMU SIM files + 6 sample_data files.
- Variants:
  - baseline_cal (`w_acf=0.34`, `acf_bonus=0.00`)
  - option_a (`w_acf=0.40`, `acf_bonus=0.00`)
  - option_b (`w_acf=0.34`, `acf_bonus=0.08`)
- Result: key detection outputs (`events`, `risk_on_n`, `risk_off_n`, `first_on_start_t`, `open_interval_end_t`) were identical across all 9 cases.

4. Adopted Option B default:
- Set `cal_on_support_score_acf_bonus` default to `0.08`.
- Kept Option A as optional override (did not globally raise `confidence_w_acf` default).

### Validation
- `py -3 -m py_compile OSC_streaming.py` -> PASS
- Smoke (`WMU_SIM_01_V2.csv`, calibration on) -> PASS with expected ON/OFF events.

## Session Update (2026-03-12, presentation CSV regeneration on current code)

### Scope
- Regenerated presentation summary CSVs from current codebase state.

### Action
- Rebuilt:
  - `cscv/presentation_summary_core.csv`
  - `cscv/presentation_summary_gate.csv`
- Source set:
  - `sample_5ch_256Hz_60s.csv`
  - `wmu_bottleneck_5min_256Hz_5ch.csv`
  - `wmu_singlech_hard_to_detect_5min_256Hz.csv`
  - `wmu_voltage_mock_5min_256Hz.csv`
  - `wmu_voltage_mock_hard_5min_3ch_256Hz.csv`
  - `wmu_voltage_random_1min_256Hz.csv`

### Result
- Regenerated presentation CSV values matched the previous presentation snapshot fields.

## Session Note (2026-03-12, event emission vs detection interpretation)

### Clarification
- Tick-level detection indicators and emitted `events` are intentionally different layers.
- Current event emission enforces minimum interval duration (`min_interval_sec_for_alert`, default `8.0s`).
- Therefore short ON intervals can be detected internally but suppressed from final event list (`[SUPPRESS] short_interval` / `short_open_interval`).

### Product direction discussed
- For real application UX, avoid exposing multiple user-facing event types.
- Keep single external alert semantics, but consider internal pending/stitch/promote logic to reduce missed promotion while preserving precision guardrails.

## Session Note (2026-03-12, helper regroup refactor with parity check)

### Refactor intent
- Agreed to reduce over-fragmented helpers and regroup by concept: compute / decision / emit.
- Keep behavior unchanged while simplifying tick pipeline readability.

### Structural changes in `OSC_streaming.py`
- Added 2 per-tick data types:
  - `TickFeatures`
  - `DecisionContext`
- Consolidated flow into 4 core functions:
  - `compute_tick_features(...)`
  - `build_decision_context(...)`
  - `step_fsm(...)`
  - `emit_events(...)`
- Rewired main loop to follow the 4-step pipeline order.

### Behavior parity check
- Compared refactored file vs baseline (`OSC_streaming copy.py`) on:
  - `tester/derived_wmu_3ch_csv/WMU_SIM_01_V1.csv`
  - `tester/derived_wmu_3ch_csv/WMU_SIM_01_V2.csv`
  - `tester/derived_wmu_3ch_csv/WMU_SIM_01_V3.csv`
- Verified parity for event-bearing lines:
  - `ALERT`
  - `INTERVAL` / `INTERVAL_OPEN`
  - `SUPPRESS`
  - final `MAIN done | events=...`
- Result: event outputs matched across all 3 inputs.
- Note: `STREAM_DONE` summary print format differs (extra counters), but detection/event decisions are unchanged.

## Session Note (2026-03-12, decision slimming and support simplification)

### Scope
- Continued refactor on top of the 4-step pipeline to reduce decision complexity for runtime maintainability.
- Target direction: keep transition behavior explainable by core axes (`confidence`, `support`, `long readiness`, `acceleration`).

### Structural updates in `OSC_streaming.py`
- Slimmed `DecisionContext` to ON-entry core gates only:
  - `accel_ok`, `on_conf_ok`, `on_support_ok`, `long_ready`, `on_long_gate_ok`, `on_entry_ready`, `on_entry_vote_sum`
- Updated `build_decision_context(...)` to evaluate ON-entry with those 4 axes.
- Kept OFF transition signals (`off_vote_core`, `long_off_confirmed`, `force_off_now`) outside `DecisionContext` and passed directly to `step_fsm(...)`.

### Support model simplification
- Replaced weighted support blend with a fixed equal-weight mean over:
  - score support
  - long support
  - confidence support
  - acceleration support
- Intent: reduce tuning surface and make support interpretation stable across datasets.

### Validation
- `py -3 -m py_compile OSC_streaming.py` -> PASS
- Smoke runs:
  - `WMU_SIM_01_V1.csv` -> PASS (`events=3`, `raw_intervals=1`, `suppressed_intervals=0`)
  - `WMU_SIM_01_V2.csv` -> PASS (`events=3`, `raw_intervals=1`, `suppressed_intervals=0`)
  - `WMU_SIM_01_V3.csv` -> PASS (`events=3`, `raw_intervals=2`, `suppressed_intervals=1`)

### Observed effect
- ON entry became more conservative in at least one case (`V1` first ON shifted later).
- Interpretation: expected tradeoff from stricter/slimmer ON gating.

## Session Update (2026-03-12, decision-axis cleanup follow-up)

### Scope
- Apply the requested post-refactor cleanup so ON-entry axes remain interpretable.

### Applied
1. Separated confidence/support semantics in ON-entry decision:
- In `build_decision_context(...)`, `on_conf_ok` no longer includes `on_support_ok`.
- `on_conf_ok` now remains confidence-only (`conf_now`, optional raw guard).
- `on_support_ok` remains support-only (`on_support_ema` vs `cal_on_support_enter_min`).

2. Documented CAL_ON support threshold roles inline:
- `cal_on_support_enter_min`: OFF -> ON_CANDIDATE entry gate.
- `cal_on_support_hold_min`: keep ON_CANDIDATE alive (revert guard).
- `cal_on_support_confirm_min`: ON_CANDIDATE -> ON_CONFIRMED soft confirm gate.

### Note
- High-priority runtime risk items (legacy `build_decision_context` args, old `gate_flags.*` OFF fields in FSM/log) were already aligned in current `OSC_streaming.py`.

## Session Update (2026-03-17, 진행 현황 정리: 완료/미완료 분리)

### 지금까지 완료한 내용
1. 모듈 경계/구조 정리(이전 배치 완료)
- `osc_modul/OSC_streaming_modul.py`의 `import *` 제거, 명시적 public re-export 구조로 정리.
- `osc_runtime_modul.py`에서 `_guess_default_stream_csv()` 중복 정의 제거( `osc_io_modul.py` 단일 소스 사용).
- runtime/fsm의 이벤트/상태 로그 출력 경로를 `status_cb` 기반으로 분리(코어에서 직접 `print` 최소화).

2. 설정 전달 구조 개선(이전 배치 완료)
- 내부 실행 진입을 config-first로 재정리:
  - `_run_streaming_alert_demo_one_channel_cfg_impl(...)`
  - `_run_streaming_alert_demo_one_channel_impl(...)`
  - `run_streaming_alert_demo_one_channel(...)`
- 기본 risk cut 상수 `DEFAULT_RISK_CUT`를 config 모듈로 올려 누락 참조 정리.

3. 안정화 수정(이전 배치 완료)
- `SCORE_CUT/BASE_SEC/RMS_WIN_SEC/PHASE_*` 관련 누락 참조 경로 정리.
- `_is_risk_active_phase`에서 상태 상수 import 누락 보완.
- MP post submit 경로에서 샘플 복사 일부 축소(중복 list 복사 완화).

4. 이번 배치에서 시작한 추가 리팩토링(진행중)
- `compute_tick_features(...)` 시그니처를 cfg 묶음 기반으로 축소 시작:
  - `threshold_cfg`, `long_cfg`, `periodicity_cfg` 전달
  - runtime 호출부 1차 반영 완료
- `build_decision_context(...)`를 `tick + st + cfg` 기반으로 개편 시작.
- `osc_core_fsm_modul.py` 내부에서 cfg 참조(`th/lg/pq`)로 치환 작업 진행.

### 현재 상태 점검 결과
- `py -3 -m py_compile ...` : PASS (문법 컴파일 기준)
- 스모크 런타임:
  - `run_streaming_alert_demo_one_channel(..., print_tick=False, status_cb=lambda _m: None)`
  - 결과: **FAIL**
  - 에러: `TypeError: build_decision_context() got an unexpected keyword argument 'conf_now'`

### 아직 해야 할 내용 (즉시 우선순위)
1. runtime 호출부 시그니처 불일치 해소 (최우선)
- `osc_runtime_modul.py`에서 `build_decision_context(...)` 호출을 새 시그니처(`tick, st, threshold_cfg, long_cfg, periodicity_cfg`)로 맞출 것.

2. 중복 unpack 축소(사용자 요청 핵심 1단계)
- runtime에서 `tick` 필드를 전부 로컬 변수로 복제하는 구간 축소.
- 1회 사용 값은 `tick.xxx` 직접 접근으로 전환.

3. cfg 묶음 전달 마무리(사용자 요청 핵심 2단계)
- `compute_tick_features` 내부에서 남은 개별 인자 참조 정리 완료.
- 필요 시 `_compute_quality_and_support`도 cfg 묶음 기준으로 정리.

4. 구조 정리 후 검증
- 컴파일 + 스모크 재실행으로 `events` 생성까지 확인.
- 최소 1개 입력 CSV (`WMU_SIM_01_TEST.csv`) 기준 런타임 정상 동작을 PASS로 되돌릴 것.

### 참고
- 현재 코드는 “컴파일 통과 + 런타임 실패(TypeError)” 상태이므로, 다음 작업은 시그니처 정합성 복구를 최우선으로 진행해야 함.

## Session Update (2026-03-17, runtime 시그니처 정합 복구 + 3.12 스모크 PASS)

### 조치 내용
1. `build_decision_context(...)` 호출 정합성 복구
- 파일: `osc_modul/osc_runtime_modul.py`
- 기존 구형 keyword 인자 전달(`conf_now=...`, `raw_now=...` 등) 제거.
- 신형 호출로 변경:
  - `build_decision_context(tick=tick, st=st, threshold_cfg=th, long_cfg=lg, periodicity_cfg=pq)`

2. tick unpack 축소(1차)
- `build_decision_context` 호출 전용으로만 쓰이던 1회성 로컬 복제 변수 제거:
  - `conf_now`, `raw_now`, `use_cal_gate`, `accel_ok`, `long_ready`
- `warmup_handoff_active` 계산은 `tick.long_ready` 직접 참조로 전환.

### 검증
1. 컴파일
- `C:\Users\pspo\AppData\Local\Programs\Python\Python312\python.exe -m py_compile ...` : PASS

2. CLI 스모크
- `python3.12 -m osc_modul.osc_runtime_modul --csv tester/derived_wmu_3ch_csv/WMU_SIM_01_TEST.csv --no-print-tick`
- 결과: PASS (`events=3`, `raw_intervals=1`, `suppressed_intervals=0`)
- 이벤트: `RISK_ON`, `RISK_OFF`, `INTERVAL`

3. 함수 직접 호출 스모크(이전 실패 케이스 재검증)
- `run_streaming_alert_demo_one_channel(..., print_tick=False, status_cb=lambda _m: None)`
- 결과: PASS (`events=3`, `['risk_on', 'risk_off', 'interval_final']`)

### 현재 상태
- 이전 blocker였던 `TypeError: build_decision_context() got an unexpected keyword argument 'conf_now'`는 해소됨.
- 코드 상태는 “컴파일 PASS + 런타임 스모크 PASS”로 복구됨.

## Session Update (2026-03-17, 내부 구조 2차 정리: config-first 강화 + tick fan-out 축소)

### 적용 내용
1. runtime 검증 경계 정리
- `osc_runtime_modul.py`
- 기존 긴 flat 시그니처 검증 함수를 내부 값 검증 전용으로 분리:
  - `_validate_detector_runtime_values(...)` (flattened value validation)
- 외부 호출 경계는 config-first로 축소:
  - `_validate_detector_runtime_args(stream_cfg, threshold_cfg, long_cfg, periodicity_cfg, mp_cfg, cut_on, cut_off)`
- `_run_streaming_alert_demo_one_channel_cfg_impl(...)`에서는 더 이상 `locals()` 기반 대량 전달을 사용하지 않음.

2. 런타임 본체의 config local fan-out 대폭 축소
- `th/lg/pq/mp`를 가능한 한 직접 참조하도록 변경.
- 함수 시작부의 수십 개 config-field 로컬 재할당 블록 제거.
- 유지한 파생값은 `cut_on/cut_off` 같은 실제 런타임 파생값 중심으로 축소.

3. tick unpacking 축소(루프 본체)
- per-tick 루프에서 `tick.*`를 대량 로컬 변수로 재복제하던 구간을 정리.
- `step_fsm(...)`, tick 로그, 이벤트 메트릭 조립에서 `tick` 직접 참조로 전환.
- `_build_tick_metrics_kwargs(...)` 헬퍼 추가로 metrics fan-out을 한곳으로 모음.

4. tuple 기반 품질 스냅샷 반환 제거(1차)
- `osc_core_fsm_modul.py`
- `_compute_quality_and_support(...)` 반환형을 순서 의존 `tuple`에서 dataclass로 전환:
  - `_QualitySupportSnapshot`
- `compute_tick_features(...)`는 index/순서 unpack 대신 named attribute 접근으로 변경.

5. 자잘한 정리
- `build_decision_context(...)`에서 실사용되지 않던 `threshold_cfg` 인자 제거.
- CLI `main()` 설정 적용 경로를 수동 필드 대입에서 `_apply_detector_overrides(...)` 단일 경로로 통일.

### 검증
1. 컴파일
- `python3.12 -m py_compile osc_modul/osc_runtime_modul.py osc_modul/osc_core_fsm_modul.py osc_modul/OSC_streaming_modul.py` : PASS

2. 스모크
- 함수 호출:
  - `run_streaming_alert_demo_one_channel(..., print_tick=False, status_cb=lambda _m: None)`
  - 결과: `events=3`, `['risk_on', 'risk_off', 'interval_final']`
- CLI 호출:
  - `python3.12 -m osc_modul.osc_runtime_modul --csv tester/derived_wmu_3ch_csv/WMU_SIM_01_TEST.csv --no-print-tick`
  - 결과: PASS (`events=3`, interval summary 정상)

## Session Update (2026-03-17, Prony post-analysis 백엔드 추가: MP 경계 유지)

### 적용 내용
1. Prony 전용 코어 모듈 추가
- `osc_modul/osc_core_prony_modul.py` 신규 생성
- 설계 원칙:
  - 검출기/FSM 경로는 변경하지 않음
  - interval post-analysis 경계에만 Prony를 추가
  - 창 선택은 MP와 동일 함수(`_select_mp_analysis_window`) 재사용
- 구현:
  - `_IntervalPronyPostRuntime` (비동기/동기 submit/finalize 구조)
  - `_run_prony_on_window(...)` (AR 계수 -> 근 추정 -> 연속시간 감쇠/주파수 변환 -> LS 진폭 적합)
  - 결과 레코드 스키마는 MP 대응형으로 `prony_*` 필드 구성
    - `prony_status/reason/n_samples/dt_sec/dt_cv/signal_std/model_order_used/fit_r2/mode_count/modes`
  - 대표모드 보조 기준:
    - interval의 `f_welch` 우선, 없으면 `acf_period_sec` 역수 기준으로 nearest mode 요약 저장

2. 설정 계층 확장
- `osc_modul/osc_config_modul.py`
- `PronyPostAnalysisConfig` 추가 및 `DetectorConfig.prony_post` 섹션 추가
- `DetectorConfig.validate()`에 `prony_post.validate()` 연결
- flat override 맵에 `prony_post` 포함
- 런타임 모드 해석 helper 추가:
  - `_resolve_prony_runtime_controls(...)`

3. 런타임 결합
- `osc_modul/osc_runtime_modul.py`
- MP 런타임 옆에 Prony 런타임 생성
- interval submit 시 MP/Prony 둘 다 호출하는 `_submit_interval_post(...)` 래퍼 추가
- tick 단계 raw sample capture도 MP/Prony 모두 동기화
- finalize 시 `interval_analysis_mp` + `interval_analysis_prony` 모두 events에 append
- CLI 옵션 추가:
  - `--prony-post`
  - `--prony-post-sync`
  - `--prony-policy`
  - `--prony-model-order`

### 검증
1. 컴파일
- `C:\Users\pspo\AppData\Local\Programs\Python\Python312\python.exe -m py_compile osc_modul\osc_core_prony_modul.py osc_modul\osc_config_modul.py osc_modul\osc_runtime_modul.py` : PASS

2. 스모크 (동시 MP+Prony)
- 호출:
  - `run_streaming_alert_demo_one_channel('tester\\derived_wmu_3ch_csv\\WMU_SIM_01_V1.csv', cfg=...)`
  - 설정: `min_interval_sec_for_alert=0`, `mp_enabled=True`, `prony_enabled=True`, 둘 다 sync
- 결과:
  - `[MP_POST] records=1`
  - `[PRONY_POST] records=1`
  - events 내 `interval_analysis_mp=1`, `interval_analysis_prony=1` 확인

## Session Update (2026-03-17, Prony 안정화 1차: 진단 + order sweep + conjugate pair)

### 적용 내용
1. Prony 진단 필드 추가
- `osc_modul/osc_core_prony_modul.py`
- `no_modes_in_band` 등 skip/failed 시 원인 해석을 위해 아래 필드 기록:
  - `prony_roots_total`
  - `prony_roots_after_mag`
  - `prony_roots_after_freq`
  - `prony_roots_after_pair`
  - `prony_order_used` / `prony_model_order_used`
  - `prony_best_attempt_idx`
  - `prony_attempt_count`, `prony_attempts`

2. 단일 order -> order sweep 전환
- `_run_prony_on_window(...)`가 단일 `prony_model_order` 1회 대신,
  `prony_order_candidates` 순회로 다중 시도
- 시도별로 AR fit -> roots -> 필터 -> amplitude fit 수행
- 성공 후보가 여러 개면 `fit_r2` 우선, 그다음 dominant amplitude 기준으로 best 선택

3. conjugate-pair 필터 도입
- 진동 모드는 복소공액쌍 기준으로만 채택
- 공액쌍이 없으면 `prony_reason=no_conjugate_pairs_in_band`로 skip

4. 설정/CLI 확장
- `osc_modul/osc_config_modul.py`
  - `prony_order_candidates` 추가 (기본: `(24,16,12,8,6,4,2)`)
- `osc_modul/osc_runtime_modul.py`
  - Prony 런타임에 `prony_order_candidates` 전달
  - CLI `--prony-order-candidates` 추가 (예: `8,6,4,2`)

### 검증
1. 컴파일
- `python3.12 -m py_compile osc_modul/osc_core_prony_modul.py osc_modul/osc_config_modul.py osc_modul/osc_runtime_modul.py` : PASS

2. MP/Prony 동시 스모크 비교 (`WMU_SIM_01_V1/V2/V3.csv`, `min_interval_sec_for_alert=0`)
- V1 interval1: MP `ok`, Prony `ok` (best order 24, roots 24/24/2/2)
- V2 interval1: MP `ok`, Prony `ok` (best order 24, roots 24/24/4/4)
- V3 interval1: MP `ok`, Prony `ok` (best order 24, roots 24/24/2/2)
- V3 interval2(5.996s): MP/Prony 모두 `interval_too_short`

