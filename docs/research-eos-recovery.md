# Buffered early-EOS checkpoint mechanics

Research-only stage1, not a production policy or a streaming codec fork.
Build `QWEN_RESEARCH_FRAME_SEEDS=ON` and `eos-research-probe`; ordinary builds
do not expose this internal API. No public C ABI or default changes.

## Supported state boundary

One buffered slot, zero committed PCM, complete16-code boundaries0..8,
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

No standalone retry, live suffix buffer, actual speculative dual fork,
classifier, physical playback, semantic coverage or held-out qualification is
implemented by this stage. The Bridge frozen plan owns the subsequent work.

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
  six healthy controls did not intervene. All new audio remains unlistened;
  longer output is not established text recovery.

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
codec fork, TTFA improvement, text coverage, held-out quality or release gate
is qualified. Public ABI, sampler defaults and Bridge engine pin remain
unchanged; no experimental policy is promoted.
