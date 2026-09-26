# Experimental EOS convergence guard

The Talker sampler keeps its normal stochastic policy by default.  An
application may opt in to a bounded termination policy through the ABI-6 tail
of `qt_tts_params` (the `qwen-tts` CLI exposes this as `--eos-guard`).

The policy is deliberately fail-safe rather than a claim that the model's EOS
distribution is correct for every utterance:

1. estimate an expected frame count from the prompt text (`4` frames per text
   token, floor `24` frames, and a `1.5` voice-clone multiplier by default);
2. leave logits unchanged before `0.6 * expected` frames;
3. linearly add at most `+25` to the EOS logit by `1.2 * expected` frames;
4. terminate with `QT_FINISH_EOS_FORCED` at `1.5 * expected` frames.

The terminal categories are kept distinct in the C ABI:
`QT_FINISH_EOS`, `QT_FINISH_MAX_TOKENS`, and `QT_FINISH_EOS_FORCED`.  A forced
termination is not reported as natural EOS.

These defaults are an initial, configurable experiment.  They are based on
the independently published `darkautism/qwen3-tts` implementation, which uses
the same `0.6 / 1.2 / 1.5`, `+25`, and clone `1.5` starting values.  They are
not yet a release acceptance policy for this GGML runtime.  The existing
hardware evidence shows a wide stochastic duration distribution (including
26-frame natural EOS and 2048-frame exhaustion for one short prompt), so a
CMP 50HX sweep must tune the estimator and thresholds against intelligibility
and truncation before enabling the guard in a worker profile.

The guard is opt-in and does not change sampler behaviour when disabled.
`suppress_initial_eos` now matches upstream `min_new_tokens=2` by masking EOS
on Talker steps 0 and 1; EOS is eligible from step 2 onward.

## Local CMP 50HX smoke

The first bounded CUDA smoke was run after commit `142066a` with the Release
`sm_75` build, Q8/Q8 1.7B Base GGUFs, and the exported ICL voice latents. The
model and codec SHA-256 values were `4b9a33a236908dd9435a42f7a396e38038329d053b704342a6413c08544c4fda`
and `1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94`.
The target was the short Russian Kraftwerk sentence used by the historical
EOS evidence; `max_new_tokens` was 2048 unless stated otherwise.

| Seed / policy | Result | Frames | Audio |
|---|---|---:|---:|
| 1000, guard enabled | natural EOS | 101 | 8.08 s |
| 1002, guard enabled | natural EOS | 25 | 2.00 s |
| 1006, guard enabled | natural EOS | 89 | 7.12 s |
| 1008, guard enabled | natural EOS | 98 | 7.84 s |
| 1008, guard disabled, `max_new=256` | max tokens | 256 | 20.48 s |
| 1000, guard enabled, boost `0` | forced EOS | 171 | 13.68 s |

This is a bounded regression smoke, not a quality or release-acceptance claim.
It demonstrates that the opt-in policy prevents the observed 256-frame
runaway on this exact Q8/CUDA setup and that forced termination is surfaced as
`eos_forced`, while normal EOS remains `natural_eos`. The generated WAVs and
stderr logs are intentionally kept outside the repository.
