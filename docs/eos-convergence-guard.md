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
