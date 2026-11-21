# Matched speech.cpp comparator tools

These research-only entry points are not linked into qwen-core or its normal
CLI/DLL. The comparator checkout is pinned to speech.cpp
`fd747c11e50543f59f2bbe233a9c7228dc479a71`, GGML
`79b50d239d0b88852814c873db0d85d43ed66d9b`. No upstream source is patched.

Use the same retained CustomVoice checkpoint revision
`0c0e3051f131929182e2c023b9537f8b1c68adfe` with both original converters.
`prepare_customvoice_models.py --help` documents its path arguments; it
verifies the checkpoint manifest before conversion. Use `python -X utf8`
on Windows. The conversion environment uses the original projects' Python
requirements, not a new worker or SDK runtime dependency.

`compare_talker_weights.py` maps tensor names and compares complete payloads,
shapes and storage types. Run on F32 and again on actual Q8 outputs.
`audit_quantized_model.py` records structural inventory and hashes.
`prepare_native_f16_codec.py` prepares/checks a native-layout decoder with the
comparator's F16/F32 storage policy. Inspect its receipt rather than assuming
the same name means the same axes or Snake parameter representation.
The small `speech_quantize_probe.cpp` calls unchanged upstream quantization
and the Qwen layout when a full CUDA build would be unnecessary for conversion.

After building the pinned speech.cpp with CUDA, initialize Visual Studio x64
and run `build_probe.ps1` with explicit checkout/build/toolkit/output paths.
The wrapper builds only the direct entry point against the existing libraries.
It retains linker warnings, including the observed CUDA LNK4098 warning;
it does not suppress a CRT warning or change upstream defaults.

Put the chosen CUDA toolkit's runtime directory on PATH and set
`CUDA_VISIBLE_DEVICES` to the exact CMP UUID. Invoke:

```text
speech_direct_probe MODEL gpu NEW_OUTPUT ryan ru 64 13 EXACT_UTF8_TEXT_FILE
```

`ru`/`en` are the comparator's language names; native uses `russian`/`english`.
The first of thirteen runs is first-use, the other twelve are persistent warm
repeats. A process watchdog is still required. Every run retains WAV/raw float
PCM and JSONL, including failure/limit outcomes. A natural-EOS result does not
establish coverage. Seed 64 is not a matched random sequence between MT19937-64
and Philox. Keep source/build/model/runtime/probe hashes with each run.
