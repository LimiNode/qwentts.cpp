# Bounded early-EOS checkpoint mechanics

Research-only mechanics, not a production policy or a streaming codec fork.
Build `QWEN_RESEARCH_FRAME_SEEDS=ON` and `eos-research-probe`; ordinary builds
do not expose this internal API. No public C ABI or default changes.

## Supported state boundary

The original buffered stage supports one slot, zero committed PCM, complete16-code boundaries0..8,
Talker attention window at most512rows. Snapshots retain host slot/prompt,
reference ownership, RNG key/subsequence, histories, complete codes, pending
logits/IDs, Talker padded KV/write head, Predictor KV and hidden bridge.
Restore repairs owned pointers and rejects another engine/job, streaming
codec lanes, partial boundaries, forced prefixes, prefix-KV and scheduled
reseeding. Parameters and external reference buffers must outlive snapshots.

Codec state is deliberately **not** snapshotted. Retain complete uncommitted
codes and redecode the whole sequence through the existing buffered codec.
Thus the experiment tests one-prefill AR restoration, not live rollback or
common-prefix PCM continuity. Up to9 host snapshots may retain approximately
530MB of device payload for the bounded window; this is not a cheap production
ring buffer or a peak-VRAM measurement.

## Single-lane live checkpoint seam

The same internal snapshot capture can now retain one admitted streaming lane
at complete AR boundaries0..8, between `tts_engine_step` calls. In addition to
AR state it owns an independent codec mirror (convolution contexts, carry,
transformer KV ring and position), the lane mapping, pending complete code
rows, live flag and shared ramp target. It does not borrow a reference-cache
LRU slot. Checkpoint destruction frees the mirror. Public ABI and ordinary
builds remain unchanged; this interface is compiled only with
`QWEN_RESEARCH_FRAME_SEEDS`.

`TtsResearchSnapshotInfo` distinguishes `frame`, `decoded_frames`,
`pending_frames` and `codec_target`. Under the production ramp, AR frame2
means one decoded frame and one pending frame, not two codec callbacks.
`bytes` includes the independent codec allocation; copying and retaining AR
state remains diagnostic overhead, not a latency optimization.

The existing `tts_engine_research_restore` stays buffered-only. The separate
`tts_engine_research_stream_restore(e, snapshot, continuation_seed,
committed_samples, restore_ms)` rejects any irreversible commit beyond
`decoded_frames * 1920` before restoring state. It accepts the same active
job or its internally retired natural-EOS result; cancellation, errors,
assisted/forced EOS, token limits, another job and any newer admission are
rejected. The current cancellation callback is polled before restoration;
one original deadline remains the caller's responsibility. No callback state
or external deadline is copied back from the checkpoint.

The caller owns bounded held PCM, copies callback samples before their scratch
storage is reused, and preserves its monotonically increasing commit cursor.
After a successful restore it keeps all committed bytes unchanged and
reinstates only the checkpoint's held PCM from the current cursor to the saved
decoded boundary. Pending rows then pass through the restored ordinary codec
lane. Params and external reference data remain valid and unchanged, except
for caller-current callback/cancellation targets. The caller must not publish
a terminal event until recovery is resolved; an already published terminal
cannot be revoked. Engine calls remain single-threaded and non-reentrant.

This seam alone establishes no live replay result, detector, bounded-buffer
policy, speech coverage, physical playback, TTFA or release qualification.

## Probe input and output

Optional per-group `recovery` requires `max_batch=1` and one explicit seed.
Unknown keys and incompatible dump/prefix/schedule modes fail preflight.

```json
{"recovery":{"policy":"replay","snapshot_frame":1}}
```

Replay restores a saved boundary without changing seed or subsequence. Both
full complete-code history and f32PCM must match byte-for-byte. Divergence
preserves outputs, returns2, and blocks interpretation of recovery policies.

Other policies are `eos-resample`, `rollback-1`, `rollback-2`, each with explicit
nonnegative `continuation_seed` and optional `trigger_max_frames`1..8. They
trigger at most once after a successful natural EOS within that frame bound.
All work shares the original monotonic deadline. Current-decision resampling
restores at the EOS boundary and uses another key; it does not mask EOS or
guarantee recovery. A universal short-EOS trigger is only a discovery probe,
not a detector qualified on legitimate short replies.

Each group retains `*-original.f32`, `*-original.i32`, final `*.f32` and
`*-codes.i32` (frame-major16codes). `research_recovery` records both outcomes,
seed/boundary, snapshot/restore times, copied bytes and exact replay checks.
`execution_mode=buffered-uncommitted`, `committed_samples=0`, empty `chunks`
and `streaming_qualified=false` prevent a false TTFA/commit claim. Snapshot
times include synchronous device-to-host copies; they are not GPU kernel time.

Use the separate original/final code files for recovery analysis. Legacy
`parse_frames` assumes monotonically increasing frames per serial and is not
compatible with FULL_AR_TRACE logs that include a restored attempt. The new
runner leaves full-frame logging OFF and uses bounded decision observations.

## Bounded runner

`python tests/run_eos_recovery.py --help` describes explicit repo/probe/runtime
and new-output paths. Plan-only is the default; only `--execute` loads models.
It pins exact RU-stress, original Q8codec and Kraftwerk prepared inputs, hashes
runtime/models before/after, and retains every flushed outcome. Loaded static
probe identity is distinct from the source checkout identity.

The full discovery schedule is14streaming controls,21unchanged replay pairs
(boundaries0/1/2, seeds2/14/72/75/78/42/64), then21conditional intervention
pairs. Retained baseline raw hashes and all replay bytes are prerequisite
gates; a mismatch leaves subsequent stages `not_run`. `--smoke` instead uses
only five known short controls,10streaming groups and15replays, cap8frames;
it is not the full matrix or full-text recovery. Both variants preflight a
worst-case PCM/log allowance plus512MiB free reserve. Max1800s overall,
600s/process and60s/logical group; watchdogs terminate only the owned probe.

No live suffix buffer, actual speculative dual fork,
classifier, physical playback, semantic coverage or held-out qualification is
implemented by this stage. The Bridge frozen plan owns the subsequent work.

## Full retry and private codec-state follow-up

`--retry-comparison` preserves the14controls/21replays, then compares rollback1
and `full-retry` twice for each seed (28groups), reversing policy order in
repeat1. Retry retains the original artifacts and performs a fresh admission
and prefill on the same engine, without AR snapshots. One original cancellation
deadline covers both attempts; prefill itself is not interruptible. Metadata
distinguishes prefill_count2 from restore_frame=-1 and records retry admission
and prefill spans. Use group_total_ms for both-attempt cost, not final job time.
The runner independently checks original bytes, repeats, and retry seed64
against same-run fresh buffered64. Default historical schedules are unchanged.

`codec-stream-replay-probe` reuses existing private streaming-codec snapshots;
it does not add another decoder or public API. `run_codec_replay.py` freezes
six retained discovery code histories, explicitly transposes frame-major AR
files, primes the full reference, then compares snapshot restoration and
reset/reconstruction. It checks all existing snapshot tensor bytes/position,
complete PCM, and a changed continuation against independently decoded fresh
candidate under the same schedule. FixedT1 boundaries0/1/2 and production-ramp
0/1/3 are separate; ineligible common-prefix boundaries are skipped. These are
teacher-forced one-lane tests, not live AR rollback, commit-buffer, multi-lane,
timing or listening qualification. Both runners are plan-only without --execute.

Quality review: the new retry branch extends the existing attempt owner and
keeps its deadline/cleanup, rather than copying the synthesis loop. The separate
codec probe reuses production state walkers and decoder; host inspection is
deliberately diagnostic. No engine runtime source, public ABI or defaults change.

## Measured discovery matrix: 2026-11-22

CMP 50HX UUID `GPU-40361931-6cb5-ac58-a059-5ba3e70986fb`, RU-stress Q8_0,
original Q8_0 codec and unchanged Kraftwerk conditioning. Payloads remain in
the Bridge checkout's ignored
`.tmp/base-acceptance/eos-recovery-20261122/matrix-001/`. See the
[Bridge results report](https://github.com/LimiNode/qwen3-tts-bridge-cpp/blob/research/native-prefill-latency/docs/reports/native-eos-recovery-mechanics-20261122.md)
for the full identity and evidence ledger.

The static probe was built from native source
`382e20a3fd52d341820d35f6ea5a2d5c23d42447`; the runner checkout was clean at
`b16d8af6c365e9deb0fd503df9a695c1092ad4f4` (later runner/tests/CI changes only).
Probe SHA-256:
`ef53929578325efba9c74028997adbb8ee486c3470ef108ba462505deaf94435`.
Probe, runtime and model identities were unchanged after execution.

- Streaming controls: 14/14 matched retained frame counts, natural EOS and
  raw PCM SHA-256, including both repetitions.
- Unchanged replay: 21/21 matched original complete 16-code histories and
  f32 PCM byte-for-byte, restoring boundaries 0/1/2 without another prefill.
- Recovery groups: 21/21 mechanically valid; 15 triggered interventions and
  six healthy controls did not intervene. Initial collection left listening
  unassessed; the later user observations below do not alter mechanical data.

The following are final frame counts, not semantic success counts. Each
triggered method used continuation seed 64; all final outcomes were natural EOS.

| Original seed | Original frames | EOS resample | Rollback1 | Rollback2 |
| --- | --- | --- | --- | --- |
| 2 | 3 | 3 | 87 | 87 |
| 14 | 3 | 3 | 104 | 2 |
| 72 | 6 | 6 | 71 | 84 |
| 75 | 2 | 94 | 85 | 84 |
| 78 | 3 | 97 | 86 | 85 |

Healthy seeds 42/64 retained 101/84 frames respectively, with original/final
codes and PCM identical in all three policy groups. Every triggered output
preserved the complete-code prefix before its actual restore boundary.
Buffered original PCM did **not** equal retained streaming PCM in any of the
42 replay/recovery groups; observed original Talker decisions matched, and
buffered original full-code histories were stable per seed. Thus unchanged
buffered replay passes, but buffered/streaming codec equivalence does not.

Across replay/recovery groups, summed snapshot capture cost was 166.3–515.1 ms
per original attempt; actual restore cost was 36.4–42.2 ms per restoration.
Retained host copies of device payload totaled 178,151,424–534,454,272 bytes per
group. These diagnostic synchronous copies are not an optimized ring buffer,
GPU-kernel-only timing, process-memory measurement or peak-VRAM measurement.

This establishes bounded buffered mechanics only. No live commit/rollback,
codec fork, TTFA improvement, general text coverage, held-out quality or release gate
is qualified. Public ABI, sampler defaults and Bridge engine pin remain
unchanged; no experimental policy is promoted.

## Scoped user listening follow-up

On2026-11-22 the user assessed all17 blinded WAVs in the Bridge shortlist;
copy/source byte counts and SHA-256 were reverified. Rollback1 has5/5 OK
observations on known seeds2/14/72/75/78, rollback2 has4/5 (seed14 failed),
and EOS resample has2/5 (seeds2/14/72 failed). Healthy streaming64 control
is OK; buffered-original14 control fails. Total12 OK and5 failed includes
both controls; policy denominators include only their own five candidates.

OK supports subjective requested-text recognizability in response to the
requested-text/extraneous-sounds question, not detailed stress, voice identity,
prosody, all artifact classes or release quality. The Bridge per-file report
`docs/reports/native-eos-recovery-listening-20261122.json` and three separate
indexed receipts retain positive rollback1 `partial_pass` and the failures.
Known-case positive listening makes rollback1 the next research candidate,
not a general EOS fix. Codec continuity, snapshot overhead and held-out/live
streaming gates are unchanged. No new GPU execution, pin/default or ABI change.

## New-text corpus and checkpoint transfer study

`tests/run_live_recovery_corpus.py` uses the existing live probe and collector
for 72 baseline groups, 12 unchanged replays and 36 conditional rollback groups.
Text/seed pairs own distinct controls; short replies are not excluded from the
naive <=8-frame intervention policy. Neither intervention nor a longer WAV is
semantic recovery evidence. Every missing cell remains explicit.

Research KV capture/restore now batches identical per-head bytes with pitched
GGML get_2d/set_2d transfers. Precision, padded rows, codec ownership and payload
size are unchanged. Model-free sentinel tests cover packing/restore equivalence;
the frozen Bridge plan additionally requires A/B/B/A CUDA code/PCM equality
before corpus execution. Capture timings are summed over each group's snapshots,
not single-checkpoint latency. Both runners accept an optional shared absolute
`--deadline-monotonic`, so identity preflight cannot restart the GPU budget.
Production ABI/defaults and the Bridge engine pin remain unchanged.
