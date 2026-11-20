# Experimental frame-indexed Philox seeds, schedule v1

Research only; no early-EOS, speech-quality or performance improvement is
claimed. This is not a production sampler/default or an ABI extension.
Build with `-DQWEN_RESEARCH_FRAME_SEEDS=ON`. An ordinary OFF build never reads
the environment variable and retains its original resolved request seed.

In a research build, `QWEN_RESEARCH_FRAME_SEED_POLICY` is read once per slot
admission. Unset, empty and `baseline` preserve the request seed. Invalid
values fail admission with `QT_STATUS_INVALID_PARAMS` before model work.
Set this environment before process startup; do not mutate the process
environment concurrently with admission on the engine's compute thread.

Let B be the request's resolved 64-bit seed, f the zero-based codec frame,
and `M(B,i)` the SplitMix64 finalizer of
`B + 0x9e3779b97f4a7c15 * (i+1)`, modulo 2^64. The finalizer is exactly the
two multiply/xor-shift stages in `src/research_frame_seed.h`.

| Policy | Frame key |
| --- | --- |
| `baseline` | B |
| `per-frame` | M(B,f), including frame zero |
| `early-2` | M(B,min(f,1)); hold the second key from frame two onward |
| `early-4` | M(B,min(f,3)); hold the fourth key from frame four onward |
| `switch-after:N` | B when f<N, otherwise M(B,N), held thereafter |

N is a canonical decimal integer 0..128 (no signs, spaces or leading zeroes).
`switch-after:1` preserves all first-frame RNG draws and switches on frame 1;
`switch-after:2` switches only after two complete frames. No counter is reset.
The schedule uses the codec frame, never the chunk or callback index.
Talker uses subsequence 16*f; the 15 Code Predictor draws use 16*f+1..16*f+15
with the **same frame key**. Existing forced-frame replay remains diagnostic.

Policies intentionally change explicit-seed semantics only in this separately
compiled and explicitly enabled experiment. Record B, policy, schedule version
and runtime hash, not only B. The key is retained as all 64 bits, including its
high bit. The [FrameSeed] trace records policy/base once per slot and at most
128 per-frame keys/subsequences with stable slot serials. It performs no extra
Philox draws and does not alter sampling logits, filters or EOS suppression.

This does not implement a risk detector, retry, snapshot/fork, or PCM switching.
Two-slot testing must separately verify actual concurrent admission, complete
prefix identity, codec state and playback timing. A longer WAV is not proof
of spoken-text coverage or successful recovery.

## Prefix and batch probe

`QWEN_RESEARCH_PREFIX_FILE` optionally names a UTF-8/ASCII file with one
complete frame per line (16 whitespace-separated IDs in this model). Its
limit is 64 KiB / 128 frames; malformed, empty, truncated, out-of-range or
dump-conflicting input fails admission. The file is captured at admission,
not reread at each frame. Unlike dump_dir it enables no tensor dumps or
diagnostic predictor graph. The complete sampled frame is overridden before
Talker history and codec consumption; compare a baseline prefix with its own
replay to verify this seam. It is teacher forcing, not a state snapshot.

The conditional `eos-research-probe manifest.json NEW-output-dir` target uses
the internal engine directly, admitting all one/two lanes before any step.
It shares loaded weights and runs genuine batched steps, not queued QTB jobs.
Each group has one steady-clock deadline across both admissions and all steps,
bounded float PCM capture, callback times and exactly one retired job per lane.
The parent runner must also use an independent process watchdog: cooperative
cancellation cannot interrupt a stuck device call. No physical playback occurs.

Manifest fields: `talker`, `codec`, `reference_text_file` (unmodified UTF-8),
`speaker_f32` (raw little-endian f32), `reference_i32` (raw little-endian
codebook-major i32), `reference_frames`, `max_batch` (1/2), `max_tokens`
(4..2048), `deadline_ms` (1..120000), `stream_max_chunk_frames` (1/2/4/8),
and `groups` (1..256 objects with `text_file`, `language`, and `seeds`).
Use only explicit nonnegative seeds in this probe. The caller records input
hashes, compiler/backend/runtime identities and all environment settings.

Outputs are per-lane raw f32 PCM and results.jsonl (group/lane/seed, status,
finish reason, frame count, callback chunks and time from common group start).
No output directory is overwritten. `admitted_ms` includes both serial
prefills; callback time is first PCM, while the native ttfa metric is first
codes. Per-lane phase spans charge shared batched work and must not be summed.
`group_total_ms` is the common elapsed span. First group is first use; retain
it separately from warm statistics. Logs identify stable job serials and
complete early frames when QWEN_TTS_FULL_AR_TRACE is set.

Two pre-admitted identical-prefix slots duplicate prefill and codec work.
They establish neither an efficient generation fork nor zero added TTFA.
Changing a lane's future seed does not prove that its speech is complete.
