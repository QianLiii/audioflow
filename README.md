# AudioFlow

Real-time transcription of what your PC is playing.

WASAPI loopback captures the render endpoint, Silero VAD finds speech, the
chunker cuts it into chunks, a single-lane scheduler feeds each chunk to a local
ASR model, and the transcript is committed incrementally. Nothing is written to
disk: audio, partial text, and the transcript live in memory only.

```
WASAPI ──512 samples──► VAD ──probability──► chunker ──snapshot/final──► scheduler
                                                                           │  HTTP
                                                                           ▼
                                                                  llama.cpp + Qwen3-ASR
                                                                           │
                     view ◄── assembler ◄── partial / final / gap events ◄─┘
```

Status: the pipeline runs end to end on Windows and has been measured on live
audio. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and
[ASR_INTEGRATION.md](ASR_INTEGRATION.md) for the ASR backend decisions.

## What is in the box

| Layer | Where | What it does |
|---|---|---|
| Native library | `native/` | WASAPI capture, Silero VAD, chunker. One DLL, C ABI. |
| Bindings | `python/audioflow/native.py` | ctypes, with struct-size assertions. |
| ASR client | `python/audioflow/asr/` | llama.cpp HTTP backend, plus a deterministic fake. |
| Scheduler | `python/audioflow/scheduler.py` | One request in flight, partial slots, deadlines, degraded finals and gaps. |
| Assembler | `python/audioflow/assembler.py` | Ordered transcript, partial replacement, overlap trimming. |
| Session | `python/audioflow/session.py` | Ties it together; source-agnostic. |
| Tools | `python/tools/` | `watch`, `live_asr`, `transcribe_wav`, `latency_probe`. |

## Requirements

- Windows 10 or later, x64.
- Visual Studio 2022/2026 with the "Desktop development with C++" workload, and
  CMake 3.20+ on `PATH`.
- **ONNX Runtime** Windows x64 package (headers, `onnxruntime.lib`,
  `onnxruntime.dll`). Verified with 1.31.0; 1.20.1 is the version the docs were
  written against.
- **Python 3.9+**. Verified with 3.14. No packages are required: the tools run
  from the repo root and add the package directory to `sys.path` themselves.
- **llama.cpp** Windows build (Vulkan or ROCm backend) and a **Qwen3-ASR GGUF**
  model. Any Vulkan-capable GPU works; CPU-only is untested.

The Silero VAD model ships with the repo at `native/model/silero_vad.onnx`, so
there is nothing extra to download for the capture/VAD side.

## Quick start

### 1. Build the native library

```bat
cmake -S native -B native\build -A x64 -DORT_ROOT=C:\deps\onnxruntime
cmake --build native\build --config Release
ctest --test-dir native\build -C Release
```

Artifacts land in `native\build\Release\`:

```
audioflow.dll        the library
audioflow.lib        import library
vad_probe.exe        score a 16 kHz mono WAV with VAD
capture_dump.exe     print frame rate and RMS from the live endpoint
capture_wav.exe      record the endpoint to a 16 kHz mono WAV
onnxruntime.dll      copied next to the examples
```

If `native\build` is not writable on your machine, build into `build\` instead
and pass `-B build`; everything below works the same with that path.

### 2. Start the ASR server

Get the two pieces:

- llama.cpp Windows binaries: <https://github.com/ggml-org/llama.cpp/releases>
  → `llama-b####-bin-win-vulkan-x64.zip` (use the ROCm or CPU build if that
  suits your hardware better).
- The model: <https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF> →
  `Qwen3-ASR-1.7B-Q8_0.gguf` **and** `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`. The
  `mmproj` file is the audio projector; the server will not accept audio
  without it.

Lay them out like this (this is the only layout the commands assume):

```
build\llama\bin\llama-server.exe
build\models\Qwen3-ASR-1.7B-Q8_0.gguf
build\models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf
```

Run the server and leave it open:

```powershell
.\build\llama\bin\llama-server.exe `
  -m .\build\models\Qwen3-ASR-1.7B-Q8_0.gguf `
  --mmproj .\build\models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf `
  -ngl 99 --host 127.0.0.1 --port 8080
```

`-ngl 99` offloads every layer to the GPU. Drop it to run on the CPU; expect a
large latency increase and a much smaller comfortable chunk rate.

#### Choosing the GPU

A machine with more than one Vulkan or SYCL device (a discrete card plus an
integrated one, for example) needs an explicit choice, otherwise llama.cpp uses
whichever it finds first. List the devices:

```powershell
.\build\llama\bin\llama-server.exe --list-devices
```

The output names them by backend prefix, for example:

```
Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, ...)
  Vulkan1: Intel(R) UHD Graphics 770 (16384 MiB, ...)
```

Pass the name to the server:

```powershell
.\build\llama\bin\llama-server.exe `
  -m .\build\models\Qwen3-ASR-1.7B-Q8_0.gguf `
  --mmproj .\build\models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf `
  -ngl 99 --device Vulkan1 --host 127.0.0.1 --port 8080
```

Notes:

- Names carry the backend prefix (`Vulkan0`, `SYCL0`, `CUDA0`) and change with
  the build and the driver, so read them from `--list-devices` rather than
  hard-coding.
- Integrated GPUs work: the Vulkan build runs on any Windows GPU with a Vulkan
  driver, and the SYCL build explicitly targets Intel iGPUs. They share system
  memory bandwidth, so expect several times the decode time of a discrete card.
  If that is not fast enough, switch to `Qwen3-ASR-0.6B`, raise `--partial-ms`,
  or shorten `--max-chunk-ms` until `latency_probe` reports no lost finals.
- `--no-mmproj-offload` keeps the audio projector on the CPU, which is worth a
  try if the multimodal path misbehaves on an unusual device.
- Splitting one model across devices (`--device dev1,dev2` plus
  `--tensor-split`) is not useful at 1.7B and is not recommended here.

Check it is up:

```powershell
curl.exe http://127.0.0.1:8080/health
```

### 3. Transcribe

Start something playing, then from the repo root:

```powershell
python python\tools\watch.py --partial-ms 250 --min-partial-age 300
```

You get a repainting transcript: committed text, the tentative tail dimmed and
marked `[partial]`, degraded finals in red, gaps in yellow, and a header with
latency and gap counters. `Ctrl+C` stops the session, prints the final
transcript and the metrics, and leaves a JSONL event log in `build\logs\`
(IDs and timings only, never text).

Useful alternatives:

```powershell
python python\tools\live_asr.py                 # one line per event, no repaint
python python\tools\transcribe_wav.py file.wav  # one file, one request
python python\tools\watch.py --wav file.wav     # replay a file through the live path
python python\tools\latency_probe.py --seconds 60   # latency distribution
```

### 4. Verify the stack without any audio

```powershell
native\build\Release\capture_wav.exe out.wav 10     # records the endpoint
native\build\Release\vad_probe.exe native\model\silero_vad.onnx out.wav
python python\tools\transcribe_wav.py out.wav
```

If all three succeed, capture, VAD, and ASR are each working.

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--partial-ms` | 500 | Partial cadence `P`. Lower is snappier and costs more decodes. 250 is safe on a Vulkan GPU. |
| `--min-partial-age` | 800 | Delay between a chunk opening and its first partial. 300 cuts first-word latency by ~700 ms. |
| `--max-chunk-ms` | 4000 | Maximum chunk length `U`. Longer chunks mean fewer finals but a later final and more audio re-fed per partial. |

Recommended latency profile:

```powershell
python python\tools\watch.py --partial-ms 250 --min-partial-age 300
```

On an RX 9070 XT with Qwen3-ASR-1.7B Q8_0, that measured:

| Metric | Value |
|---|---|
| ASR request, 1–4 s of audio | 100–150 ms (roughly linear, ~0.02 s per audio second) |
| Partial age (audio end → text shown) | 130 ms p50 |
| First partial (chunk start → text) | 662 ms p50, i.e. ~410 ms after speech onset |
| Slot slip | 14 ms p50 |
| Final latency (chunk close → text) | ~430 ms |

## Tests

```powershell
ctest --test-dir native\build -C Release          # ring, resampler, chunker

cd python
python -m tests.test_scheduler                     # 9 scheduler tests
python -m tests.test_assembler                     # 11 assembler/metrics tests
```

The scheduler tests run against the fake backend with a virtual clock, so they
are deterministic and need neither a GPU nor a server.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error C2198` in `af_vad_silero.c` | ONNX Runtime headers changed `CastTypeInfoToTensorInfo`. Build with `-DAF_ORT_CAST_TENSOR_INFO_STATUS=1`, or `0` for headers older than 1.22. |
| `cannot open include file: onnxruntime_c_api.h` | `ORT_ROOT` is wrong. It must contain `include\onnxruntime_c_api.h` and `lib\onnxruntime.lib`. |
| `python` opens the Microsoft Store | The WindowsApps placeholder shadows your install. Use the full interpreter path, e.g. `& 'E:\Tools\python\python.exe'`. |
| `llama-server not reachable` | The server process is not running, or it is on another port; pass `--url http://127.0.0.1:PORT`. |
| The wrong GPU is doing the work | List devices with `--list-devices` and pin one with `--device`. |
| `failed to load model` / no audio accepted | The `--mmproj` argument is missing or does not match the GGUF. |
| Transcript stays empty while audio plays | Check the endpoint with `capture_dump.exe`: RMS near zero means nothing is rendering, or the default device is wrong. |
| `Access is denied` creating `native\build` | A sandbox or ACL is blocking the directory; build into `build\` instead. |
| First partial feels slow | Lower `--min-partial-age` (try 300) before touching `--partial-ms`. |

## Repository layout

```
native/                 C11 library: capture, VAD, chunker, tests, examples
  include/audioflow/    public C ABI
  src/                  implementation
  model/silero_vad.onnx Silero VAD v6.2.2 streaming model
python/
  audioflow/            bindings, ASR client, scheduler, assembler, metrics, session
  tests/                scheduler and assembler tests
  tools/                watch, live_asr, transcribe_wav, latency_probe
build/                  build output, llama.cpp, GGUF models, logs (git-ignored)
testdata/               local audio fixtures (git-ignored)
```

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — session contract, scheduler policy, event
  semantics, defaults.
- [ASR_INTEGRATION.md](ASR_INTEGRATION.md) — Qwen3-ASR streaming investigation,
  measured decode cost, and why the reference backend is llama.cpp.
- [native/README.md](native/README.md) — the C API, build options, and the Silero
  VAD adapter.

## License

AudioFlow is MIT licensed; see [LICENSE](LICENSE). Silero VAD and ONNX Runtime
are MIT licensed as well; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
