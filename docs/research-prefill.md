# Bounded native prefill experiments

Research only. No sampler, model, public C ABI, engine pin, or ordinary build
default is promoted. The Bridge report `native-prefill-study-20261121.md`
owns the measured CMP results; this file owns reproduction and switch semantics.

## Build and observation

Both `QWEN_RESEARCH_PREFILL` and `QWEN_RESEARCH_NVTX` default OFF. Ordinary
builds do not read the prefill experiment environment. The internal observation
accessor is not a public DLL export. Internal tools inherit the research layout
through qwen-core's compile definition.

Use a clean build after internal-header changes. On the measurement machine,
localized MSVC/Ninja include scanning recorded zero pipeline header dependencies;
an incremental rebuild mixed a new probe layout with a stale static object.
The failed controls were retained as invalid, and explicitly rebuilding both
pipeline/qwen objects repaired them. Do not treat a changed version string as
proof that all affected objects were rebuilt.

For the GPU probe, configure the standalone engine with `GGML_CUDA=ON`,
`QWEN_RESEARCH_FRAME_SEEDS=ON`, and `QWEN_RESEARCH_PREFILL=ON`; use an explicit
GPU architecture and initialized host compiler environment. Build
`eos-research-probe` and `test-prefill-settings-contract`. For the separate
Nsight experiment add `QWEN_RESEARCH_NVTX=ON`; this requires CUDA and its
existing NVTX headers, not a new runtime or SDK dependency.

| Environment | Values / meaning |
| --- | --- |
| `QWEN_PREFILL_PROFILE` | `0` (default) or `1`; host-wall phases and saved final logits |
| `QWEN_PREFILL_LAST_ONLY` | `0` (default) or `1`; final norm/head only for the last row; full dumps retain the original graph |
| `QWEN_PREFILL_KV` | `f32` (default) or `f16`; Talker KV storage, including later decode; Predictor KV unchanged |
| `QWEN_PREFILL_WINDOW` | `256` (default) or `64`; attention-window alignment, not a smaller total sequence capacity |
| `QWEN_PREFILL_RETAIN` | `0` (default) or `1`; one compatible prefill graph/scheduler/allocation, keyed by shape and relevant execution settings |
| `QWEN_PREFILL_PREFIX_ROWS` | absent (default) or integer `1..4095`, also strictly smaller than the actual prompt |
| `QWEN_PREFILL_PREFIX_SOURCE` | `split` (default) or `full`; an explicit value requires prefix rows |
| `QWEN_PREFILL_NVTX` | `1` enables labels only in an NVTX-enabled build; use in separate profiler runs |

Invalid settings fail atomically. Set the environment before loading the
pipeline; do not mutate it concurrently. Retained graphs are bypassed for
full diagnostics; only one shape is retained. This is not a general cache.

`compute_wait_ms` includes the existing synchronous graph call and waiting;
it is not pure GPU execution. Phase observation does not add synchronization
or full-tensor readbacks. The probe copies the existing last-logit readback
and records its overhead in matched Observer OFF/ON controls. Full dumps,
frame tracing and Nsight are separate from latency measurements.
Nsight Systems 2026.4.1 explicitly refuses CUDA tracing on CMP hardware
(`ERR_NVCMPGPU`). The retained attempt provides NVTX host spans, not CUDA
kernel timings. Profiler temporary files are routed into its task directory.

## Prefix controls

Prefix reuse requires a GPU, one slot, no tensor dump, and byte-identical
leading input embeddings. First discover the actual common rows on different
texts/languages/conditioning with observation only. A speaker name alone does
not establish reusable context. The research cache rejects the first mismatch
and remains invalid until pipeline reload; it does not silently choose a new
prefix, fall back, or implement a production voice/language cache.

`split` first computes only the chosen prefix, snapshots its KV on device,
then computes the suffix. `full` instead executes and returns the original
full first prefill and snapshots its leading KV rows. Subsequent requests in
both variants restore the snapshot and execute the suffix with absolute
positions. This distinguishes prefix-construction shape changes from suffix
execution shape changes. Device restore/preparation/snapshot timings are
separate; `prefill_ms` includes the wrapper, whereas observation core spans
describe the suffix (or the original full first request in `full`).

Causal attention does not imply floating-point equivalence. Compare final
logits, forced or sampled codes, raw PCM and EOS separately. A faster result
or equal argmax does not qualify this path. There is no release cache,
cross-request cancellation/reset contract or public configuration in this PR.

## Reproducible harnesses

- `tests/run_prefill_research.py --help`: frozen CMP UUID, exact models and
  conditioning hashes, new output directory, bounded subprocess/total runtime,
  first-use plus twelve warm same-input pairs, shape/language/EOS controls,
  final baseline, raw PCM/WAV/logits and all failures retained. `--causal`
  enables separate bounded complete-code traces; these are not latency runs.
- `--prefix-rows N --variants baseline prefix prefix-full baseline-end`:
  explicitly chosen post-discovery comparison. `--russian-only` tests the
  observed same-language prefix without pretending it also matches English.
- `--speaker-only`: same Base checkpoint and speaker embedding without ICL
  reference text/codes, a conditioning-cost control, not equivalent voice quality.
- `--disable-cuda-graphs`: native control for a comparator compiled with
  GGML CUDA graphs OFF. Keep native default-graph results separately.
- `tests/check_prefill_rejection.py --help`: two bounded accepted Russian
  requests, then expected English-prefix rejection. The probe exits at rejection,
  so subsequent poisoned-handle behavior is explicitly not hardware-tested.
- `tests/profile_prefill.py --help`: three four-frame controls with NVTX,
  independent owned-process-tree watchdog and SQLite export. Max-token endings
  are intentional diagnostics, not successful speech synthesis.
- `tests/run_speech_comparison.py --help` and
  [comparator tools](../tests/speech_comparison/README.md): pinned source,
  matched checkpoint/tensor payloads, every attempt retained and no hidden
  warmup. Different RNG algorithms mean integer seeds are not matched draws.

`state=completed` means the process produced all planned result rows. Always
inspect error, max-token, PCM, EOS, coverage and numerical comparisons separately;
successful process exit is not a qualification result. Twelve identical warm
repeats measure jitter, not a distribution across twelve independent texts.
New artifacts remain unlistened until exact-file human observations are recorded.
