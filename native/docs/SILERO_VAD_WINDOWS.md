# Silero VAD on Windows: quick start

This guide builds the `af_vad` interface, the Silero VAD adapter, and the
`vad_probe` validation tool on Windows 10 or later.

The adapter is written against the ONNX Runtime C API. Follow these steps on
the target Windows machine.

## What gets built

- `audioflow.dll` — the single shared library containing capture, VAD, and
  chunker. The Silero adapter implements `include/audioflow/af_vad.h`.
- `vad_probe.exe` — reads a 16 kHz mono WAV and prints one probability per
  512-new-sample model step.

The adapter validates the model's input/output names and shapes at load time.
It accepts the Silero VAD streaming interface (v5 and v6):

```text
inputs:  input [1,576] float32   (64 rolling context + 512 new samples)
         state [2,1,128] float32
         sr    int64 scalar or [1]
outputs: output [1,1] float32
         stateN [2,1,128] float32
```

Silero VAD v5/v6 keeps a 64-sample rolling context in addition to the LSTM state.
The ONNX input axis is dynamic, so feeding `[1,512]` runs without an error but
silently returns degraded probabilities. The adapter maintains the context
and feeds `[1,576]` to the model. Reset clears both the state and the context.

## Prerequisites

- Windows 10 or later, x64.
- Visual Studio 2022 with the "Desktop development with C++" workload.
- CMake 3.20 or later on `PATH`.
- PowerShell 5.1 or later.
- Optional: ffmpeg or Audacity to produce a test WAV.

Pin both dependencies. This guide uses ONNX Runtime 1.20.1 and Silero VAD
v6.2.2 (streaming model); the adapter is also verified against ONNX Runtime
1.31.0. v5 uses the same ONNX I/O. Substitute your pinned versions, but keep
the ONNX Runtime headers, import library, and DLL from the same package.

## 1. Install ONNX Runtime

Download the Windows x64 package into `C:\deps`:

```powershell
$ortVersion = "1.20.1"
$zip = "C:\deps\onnxruntime.zip"

New-Item -ItemType Directory -Force C:\deps | Out-Null
Invoke-WebRequest `
  "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-$ortVersion.zip" `
  -OutFile $zip

Expand-Archive $zip -DestinationPath C:\deps
Rename-Item "C:\deps\onnxruntime-win-x64-$ortVersion" "C:\deps\onnxruntime"
```

Verify these paths exist:

```text
C:\deps\onnxruntime\include\onnxruntime_c_api.h
C:\deps\onnxruntime\lib\onnxruntime.lib
C:\deps\onnxruntime\lib\onnxruntime.dll
```

The DLL may also be under `C:\deps\onnxruntime\bin`. The CMake build searches
both.

## 2. Get the Silero VAD model

```powershell
git clone --depth 1 --branch v6.2.2 `
  https://github.com/snakers4/silero-vad.git C:\deps\silero-vad

New-Item -ItemType Directory -Force C:\deps\silero | Out-Null
Copy-Item `
  C:\deps\silero-vad\src\silero_vad\data\silero_vad.onnx `
  C:\deps\silero\silero_vad.onnx

Get-FileHash C:\deps\silero\silero_vad.onnx -Algorithm SHA256
```

Record the hash in your build notes or installer manifest. If the model path
differs in your checkout, use the v6 streaming ONNX file that contains the
`input`, `state`, and `sr` inputs. Do not use `silero_vad_16k_sequence.onnx`;
that is the offline sequence model, not the streaming VAD. A v4 model will be
rejected with a clear error.

## 3. Build

From the repository root, in a Developer PowerShell:

```powershell
cmake -S native -B native\build -A x64 `
  -DORT_ROOT=C:\deps\onnxruntime

cmake --build native\build --config Release
```

Silero VAD is mandatory. CMake fails at configure time if ONNX Runtime is not
found.

Expected artifacts:

```text
native\build\Release\audioflow.dll
native\build\Release\audioflow.lib
native\build\Release\vad_probe.exe
native\build\Release\capture_dump.exe
native\build\Release\onnxruntime.dll
```

`audioflow.dll` contains capture, VAD, and chunker. For Python, keep
`audioflow.dll`, `onnxruntime.dll`, and `silero_vad.onnx` in one directory and
add it with `os.add_dll_directory()` before `ctypes.CDLL`.

If `onnxruntime.dll` was not copied next to the executable, copy it manually:

```powershell
Copy-Item C:\deps\onnxruntime\lib\onnxruntime.dll `
  native\build\Release\
```

If the DLL is under `bin`, copy it from there.

## 4. Prepare a test WAV

The probe requires PCM, 16 kHz, mono, 16-bit:

```powershell
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le sample16k.wav
```

With Audacity: set project rate to 16000 Hz, export as "WAV (Microsoft)" with
encoding "Signed 16-bit PCM", mono.

## 5. Run the probe

```powershell
native\build\Release\vad_probe.exe `
  C:\deps\silero\silero_vad.onnx `
  sample16k.wav
```

Output:

```text
window  probability  state
     0  0.01234
     1  0.88421      speech
     2  0.91342      speech
     3  0.20118
windows=4 speech_windows=2 samples=70000
```

The last partial window is zero-padded by the probe. For a quick sanity check:

- Silence and room tone should produce low probabilities.
- Speech should clear 0.5.
- Music with vocals may also clear 0.5. The VAD detects speech, not intent.

## 6. Integrate with capture

Create one VAD instance on the chunker thread. The capture module delivers
events on its own delivery thread; enqueue those frames to the chunker thread
and run VAD there. Do not share a VAD instance across threads.

```c
#include "audioflow/af_vad.h"

af_vad_config vad_config = {0};
af_vad *vad = NULL;

vad_config.model_path = "silero_vad.onnx";
vad_config.sample_rate = 16000;
vad_config.intra_op_threads = 1;

if (af_vad_create(&vad_config, &vad) != AF_VAD_OK) {
    /* af_vad_last_error() has the reason */
}
```

The capture frame defaults to 32 ms, which is one Silero step. Set it
explicitly when overriding:

```c
af_capture_config capture_config = {0};
capture_config.frame_ms = 32;   /* 512 new samples at 16 kHz */
capture_config.emit_silence = 1;
```

The adapter prepends the 64-sample rolling context internally, so the model
receives 576 samples per step. Do not feed 512 samples directly to the raw
ONNX model in another binding; that path silently degrades.

Feed each audio event to the VAD:

```c
float probability = 0.0f;
int updated = 0;

if (af_vad_process(vad,
                   event->samples,
                   event->sample_count,
                   &probability,
                   &updated) != AF_VAD_OK) {
    /* log af_vad_last_error(); reset the VAD and continue */
}

if (updated) {
    /* probability is fresh for this 32 ms window */
}
```

On a capture discontinuity or a new session:

```c
af_vad_reset(vad);
```

A suggested starting policy:

- Onset threshold: 0.5-0.6.
- Offset threshold: 0.35-0.4.
- Require two consecutive speech windows to open a chunk.
- Require about 400 ms of below-threshold windows to close it.
- Keep the 250 ms pre-roll; it covers detection delay.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ONNX Runtime not found` | `ORT_ROOT` is wrong; check `include/onnxruntime_c_api.h` and `lib/onnxruntime.lib`. |
| `onnxruntime.dll` not found at runtime | Copy it next to `vad_probe.exe`, or add its directory to `PATH`. |
| `ONNX Runtime ... does not support API version ...` | Header and DLL are from different ONNX Runtime versions. Use one package for both. |
| `expected Silero VAD streaming model (3 inputs, 2 outputs)` | The model is v4, the offline sequence model, or another export. Use a v5/v6 streaming `silero_vad.onnx`. |
| `expected input ... named 'state'` | The model export has different names. Use the pinned streaming model. |
| `CreateSessionFromArray` failed | Corrupt model, incompatible opset, or mismatched ONNX Runtime version. Re-download both. |
| `WAV must be PCM, 16 kHz, mono, 16-bit` | Re-export the test file with the required format. |
| Speech probabilities stay low | Check the WAV is really 16 kHz mono; check levels; confirm the model is a v5/v6 streaming export. If you call the raw ONNX model outside this adapter, feed 576 samples (64 context + 512 new), not 512. |
| Probabilities are uniformly near zero (~5e-4) | The model is being fed 512 samples without the 64-sample context. Use `af_vad_process`, which maintains the context. |
| Unicode model path fails | Use an ASCII path, or pass the model through `af_vad_config.model_data` instead of `model_path`. |

## Licensing

- Silero VAD, including `silero_vad.onnx`, is MIT licensed.
- ONNX Runtime is MIT licensed.
- The ONNX Runtime package ships `LICENSE` and `ThirdPartyNotices.txt`.
  Redistribute both with `onnxruntime.dll`.
- The full texts are in
  [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) at the repository
  root.

## Notes

- The adapter supports the Silero VAD v5/v6 streaming interface at 16 kHz
  only.
- The model input is 576 samples: 64 rolling context + 512 new. The adapter
  maintains both that context and the LSTM state.
- The model is loaded into memory with `CreateSessionFromArray`; it is not
  streamed from disk during inference.
- The VAD instance carries recurrent state and context. Reset it whenever the
  capture timeline breaks.
- The probe feeds 512 new samples per call and zero-pads the final partial
  step. The adapter supplies the 64-sample context.
