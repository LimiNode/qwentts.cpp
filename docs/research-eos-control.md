# Research-only Talker EOS control

This is experimental policy, not a qualified early-EOS fix. Compile with
`QWEN_RESEARCH_EOS_CONTROL=ON` (default OFF). The option affects the internal
static/shared engine implementation, not `qwen.h`, ABI 6 or default parameters.
Without the definition, the original sampler call is compiled unchanged.

Set `QWEN_RESEARCH_EOS_CONTROL` before admission:

| Value | Policy |
| --- | --- |
| unset, empty, `baseline` | Original sampler and original RNG schedule |
| `penalty:2`, `penalty:4`, `penalty:6` | Subtract the stated raw-logit value from EOS at every Talker decision |
| `alternative-once:8` | Once per request, if baseline selects EOS at frame 2–7, exclude EOS for this selection and rerun sampling |

Other values reject admission. The eight-frame diagnostic window is frozen
before GPU experiments, not a calibrated definition of premature EOS. A
legitimate short answer can be altered. Constant penalties can delay natural
completion, induce garbage or reach the generation cap; save these failures.

Order: reserved-token mask -> minimum-two-frame mask -> existing guard ->
research raw-logit EOS subtraction -> repetition penalty -> temperature ->
top-k -> top-p -> exponential weights -> categorical draw. The normal guard
is disabled in nonbaseline research runs. EOS is not in ordinary audio history;
the raw penalty is not scaled or normalized as a probability.

The alternative restores the same pre-chain logits, masks EOS and reruns the
chain using the original Philox uniform/subsequence. It does not take another
draw, increment the subsequence twice, run prefill/forward again, retry the
request, rewind KV or modify codec state. The next Talker decision has ordinary
EOS eligibility, whether or not that ends the utterance. If no finite alternative
exists, preserve EOS without spending the intervention. Predictor is unchanged.

Nonbaseline admission requires N=1 and stochastic Talker/Predictor. It rejects
the existing EOS guard, forced Talker/Predictor histories, frame reseeding and
prefix reuse. The research runner must also freeze all otherwise unchanged
parameters and prohibit recovery/greedy/alternate conditioning arms. The
one-shot flag belongs to the slot and is reset by every new admission.

`[ResearchEOS]` logs policy/seed on admission and initial/effective selections,
frame, intervention/used flags, penalty, uniform and subsequence for the first
eight decisions and later EOS decisions. This is bounded per decision, not a
tensor dump. The observer's suppressed logits represent the *effective*
pre-chain input and its weights match that input. An alternative's original
sample is retained in the explicit log; the observer alone is not its complete
counterfactual history. For penalty arms, `initial` means the penalized sampler
selection, not an unpenalized paired baseline trajectory.

Model-free test: `test-research-eos-control` / CTest
`research_eos_control_contract`. It checks strict parsing, baseline token/weight/
uniform identity, raw penalty placement, retained masks, one-shot bounds/state,
restored logits and repetition/top-k/top-p with the same draw. These tests do
not establish speech completeness, voice identity, GPU cost or EOS recovery.

This branch is stacked on retained ICL/prefill research `1416928`, not isolated
Engine #51. Freeze those modes off; do not merge or pin this whole research
stack into a production build.
