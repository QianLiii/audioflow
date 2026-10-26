# AudioFlow native library

C11 implementation of the audio pre-processing path: WASAPI loopback capture,
Silero VAD, and chunker. It builds as one shared library, `audioflow`
(`audioflow.dll` on Windows), callable from Python via `ctypes`.

The capture layer captures the default Windows render endpoint, converts it to
16 kHz mono s16le, and delivers fixed-size PCM frames to a callback on a
dedicated event thread. VAD and chunking run on the frames.

## Boundary

- Input: Windows system output (endpoint loopback).
- Output: 16 kHz mono s16le frames, 10/20/32/40 ms, default 32 ms.
- The default 32 ms (512 new samples) is one Silero VAD step. The adapter adds
  the model's 64-sample rolling context internally.
- No disk access. All buffering is in memory.
- One capture instance. The caller owns the process lifetime.
- Event callback runs on the delivery thread. It must not block and must not
  call `af_capture_stop`.

## Public API

`include/audioflow/wasapi_capture.h`:

- `af_capture_start(config, &capture)`
- `af_capture_stop(capture)`
- `af_capture_last_error()`

Events:

- `AF_EVENT_AUDIO`: PCM frame, monotonic `frame_index`, optional QPC position.
- `AF_EVENT_FORMAT`: input rate/channels and whether WASAPI performed the
  conversion.
- `AF_EVENT_DISCONTINUITY`: default-device change, device invalidation, or a
  capture error.
- `AF_EVENT_DROPPED`: the delivery ring discarded events because the callback
  fell behind.

## Format path

1. Try to initialize the loopback client at 16 kHz mono s16le with
   `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` and
   `AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY`. If the engine accepts it, WASAPI
   performs the conversion.
2. Otherwise, initialize at the device mix format and convert in-process:
   float32 or s16le, any channel count, downmixed to mono, then resampled to
   16 kHz by a windowed-sinc low-pass filter and linear interpolation.

The fallback resampler is intentionally simple. It is not a high-end DSP
component, but it removes the aliasing a bare decimator would introduce.

## Threading and buffering

- Capture thread: COM MTA, WASAPI event-driven loop, optional MMCSS "Audio"
  registration.
- Delivery thread: pops events from a bounded ring and invokes the callback.
- Ring overflow drops the oldest event and is reported as `AF_EVENT_DROPPED`.
- On a device change or `AUDCLNT_E_DEVICE_INVALIDATED`, the capture client is
  torn down, rebuilt, and a discontinuity event is emitted. The frame index
  continues; the QPC timeline jumps.

## Silence

WASAPI loopback produces no packets when no application is rendering. With
`emit_silence` enabled, the capture loop generates zero frames after 30 ms of
idleness so the downstream VAD and chunker see a continuous timeline.

## VAD

`include/audioflow/af_vad.h` defines the speech probability interface.
**Silero VAD** (v6.2.2 streaming model; v5-compatible I/O) through ONNX
Runtime is mandatory. No other VAD backend is included.

- ONNX Runtime is required for every build. Set `-DORT_ROOT=<onnxruntime root>`.
- `vad_probe.exe` validates a model against a 16 kHz mono WAV.
- The adapter validates the streaming model's input/output names and shapes
  at load.
- Silero VAD v5/v6 expects 576 samples per model call: 64 rolling context +
  512 new.
  The adapter maintains the context. Configure capture at 32 ms so each event
  is one 512-sample step; feed arbitrary sample counts if needed.
- VAD instances are stateful and not thread-safe. Run one on the chunker
  thread and call `af_vad_reset()` on capture discontinuities.

Windows installation and validation steps are in
[`docs/SILERO_VAD_WINDOWS.md`](docs/SILERO_VAD_WINDOWS.md).

## Chunker

`include/audioflow/af_chunker.h` turns 16 kHz frames plus a per-frame speech
probability into chunk events.

Defaults:

- 512-sample frames (32 ms).
- 250 ms pre-roll.
- 2 consecutive speech frames to open.
- 400 ms below the offset threshold to close.
- 500 ms minimum speech.
- 4 s maximum chunk length.
- 300 ms overlap on forced cuts.
- Forced cut at the lowest-energy 100 ms window in the last 500 ms.
- 100 ms trailing pad on silence closes.

Events:

- `AF_CHUNK_EVENT_OPENED` — a chunk is open; partial slots can start.
- `AF_CHUNK_EVENT_CLOSED` — finalized PCM; reason is `SILENCE`, `FORCED`, or
  `FLUSH`. The event owns `pcm`; release it with `af_chunk_event_release`.
- `AF_CHUNK_EVENT_DISCARDED` — range dropped for `MIN_SPEECH` or
  `DISCONTINUITY`; no PCM.
- `AF_CHUNK_EVENT_SNAPSHOT` — a copy of the open chunk prefix for a partial
  request; owns `pcm`.

Integration on the chunker thread:

```c
float probability = 0.0f;
int vad_updated = 0;
af_chunk_event events[4];
size_t event_count = 0;

af_vad_process(vad, frame, frame_samples, &probability, &vad_updated);
af_chunker_push(chunker, frame, frame_samples, probability, qpc,
                events, 4, &event_count);
```

The default `frame_ms = 32` makes one capture event one VAD step and one
chunker frame. On a capture discontinuity, call `af_vad_reset()` and
`af_chunker_reset()`.

The chunker is single-threaded. Do not call it from the capture callback;
enqueue frames to the chunker thread. `af_chunker_snapshot()` copies the open
chunk prefix for the scheduler's partial slots.

## ASR service contract

The reference backend is a local `llama-server` serving a Qwen3-ASR GGUF over
`POST /v1/chat/completions` (one `input_audio` part carrying a base64 WAV). It
holds no server state: the client re-sends the accumulated audio on every
decode and receives cumulative text. `python/audioflow/asr/llama_cpp.py`
implements that contract.

The table below documents the alternative vLLM demo service. The client's
failure policy (no idempotency key, no abort endpoint, orphan sessions expire on
a timer) was written against that service.

| Endpoint | Request | Response |
|---|---|---|
| `POST /api/start` | no body | `{"session_id": "..."}` |
| `POST /api/chunk?session_id=...` | raw float32 PCM, 16 kHz mono, `application/octet-stream` | cumulative `{"language", "text"}` |
| `POST /api/finish?session_id=...` | no body | final `{"language", "text"}` |

Integration rules:

- One ASR session maps to one chunker chunk.
- Start the session lazily on the first audio submission, not on every OPENED
  event.
- Send only new audio: `[submitted_samples, current_end)`, converted from s16le
  to float32. Do not resend the growing prefix.
- Advance `submitted_samples` only after a successful `/api/chunk` response.
- On close, send the remaining delta, call `/api/finish`, and commit its text.
- On drop, call `/api/finish` best-effort. The demo has no abort endpoint and
  expires abandoned sessions after 10 minutes.
- The demo has no idempotency key. Do not retry an ambiguous `/api/chunk`
  timeout; finish the session and emit the last cumulative text as degraded.
- No disk access and no local audio URL server.

The client-side ASR module and scheduler are not implemented yet.

## Build

ONNX Runtime is mandatory. The Windows setup is covered in
[`docs/SILERO_VAD_WINDOWS.md`](docs/SILERO_VAD_WINDOWS.md).

Linux, with ONNX Runtime installed:

```sh
cmake -S native -B native/build \
  -DORT_ROOT=/path/to/onnxruntime \
  -DAUDIOFLOW_BUILD_TESTS=ON
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

Windows, single shared library and examples (default):

```bat
cmake -S native -B native\build -A x64 -DORT_ROOT=C:\path\to\onnxruntime
cmake --build native\build --config Release
```

Outputs:

```text
native\build\Release\audioflow.dll
native\build\Release\audioflow.lib
native\build\Release\vad_probe.exe
native\build\Release\capture_dump.exe
native\build\Release\capture_wav.exe    (record loopback to a 16 kHz mono WAV)
native\build\Release\onnxruntime.dll   (copied for the examples)
```

`capture_wav.exe [out.wav] [seconds]` writes exactly the PCM that the Python
ASR layer receives, so a live session can be replayed offline:

```bat
native\build\Release\capture_wav.exe live.wav 15
```

`audioflow.dll` is the one library for capture, VAD, and chunker. Its public
exports are:

```text
af_capture_start, af_capture_stop, af_capture_last_error
af_vad_create, af_vad_destroy, af_vad_reset, af_vad_window_samples,
af_vad_process, af_vad_last_error
af_chunker_create, af_chunker_destroy, af_chunker_push, af_chunker_snapshot,
af_chunker_flush, af_chunker_reset, af_chunk_event_release
```

For a static library, add `-DBUILD_SHARED_LIBS=OFF`. On Linux the default
shared output is `libaudioflow.so`; capture is omitted because WASAPI is
Windows-only.

## Python / FFI

Load the DLL with `ctypes`:

```python
import ctypes
import os

os.add_dll_directory(r"C:\path\to\audioflow")
lib = ctypes.CDLL(r"C:\path\to\audioflow\audioflow.dll")
```

Keep `audioflow.dll`, `onnxruntime.dll`, and `silero_vad.onnx` together.
`audioflow.dll` depends on `onnxruntime.dll`; the model path is passed to
`af_vad_create`.

The example prints one line per second with frame count and RMS, plus format,
discontinuity, and drop events. Press Ctrl+C to stop.

## Limitations

- Endpoint loopback only. Per-application capture needs process loopback
  (`ActivateAudioInterfaceAsync`), a separate source implementation.
- No echo cancellation; endpoint loopback does not need it.
- The fallback conversion supports float32 and s16le mix formats. Other mix
  formats (24-bit packed, 32-bit integer) return `AF_ERR_FORMAT`.
- The capture callback must be fast. A slow callback fills the ring; the
  oldest events are dropped.

## Licensing

Silero VAD (including the model) and ONNX Runtime are MIT licensed. See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the full texts.
AudioFlow itself is MIT licensed; see [`../LICENSE`](../LICENSE).
