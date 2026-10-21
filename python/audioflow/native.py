"""ctypes bindings for audioflow.dll.

Only what the live loop needs: capture, VAD, and chunker. The C ABI is in
``native/include/audioflow/``; every struct is mirrored field for field and its
size is asserted at import time, so a layout drift fails loudly instead of
corrupting memory.

Threading: the capture callback runs on the DLL's delivery thread. It must not
block, so it hands events to a queue and returns.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    byref,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int16,
    c_int32,
    c_size_t,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)

# -- constants -------------------------------------------------------------

AF_OK = 0

AF_EVENT_AUDIO = 0
AF_EVENT_DISCONTINUITY = 1
AF_EVENT_DROPPED = 2
AF_EVENT_FORMAT = 3

AF_DISC_DEVICE_CHANGED = 0
AF_DISC_DEVICE_INVALIDATED = 1
AF_DISC_CAPTURE_ERROR = 2
AF_DISC_OVERFLOW = 3
AF_DISC_STOPPED = 4

AF_CAPTURE_SAMPLE_RATE = 16000
AF_CAPTURE_MAX_FRAME_SAMPLES = 640
AF_CAPTURE_DEVICE_ID_MAX = 256

AF_CHUNK_EVENT_OPENED = 0
AF_CHUNK_EVENT_CLOSED = 1
AF_CHUNK_EVENT_DISCARDED = 2
AF_CHUNK_EVENT_SNAPSHOT = 3

AF_CHUNK_REASON_SILENCE = 0
AF_CHUNK_REASON_FORCED = 1
AF_CHUNK_REASON_FLUSH = 2
AF_CHUNK_REASON_DISCONTINUITY = 3
AF_CHUNK_REASON_MIN_SPEECH = 4
AF_CHUNK_REASON_PARTIAL = 5
AF_CHUNK_REASON_NONE = 6


# -- structs ---------------------------------------------------------------


class AfEvent(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("frame_index", c_uint64),
        ("qpc_position", c_uint64),
        ("sample_count", c_uint32),
        ("samples", c_int16 * AF_CAPTURE_MAX_FRAME_SAMPLES),
        ("input_sample_rate", c_uint32),
        ("input_channels", c_uint16),
        ("autoconverted", c_int32),
        ("reason", c_int32),
        ("dropped_frames", c_uint32),
        ("device_id", c_char * AF_CAPTURE_DEVICE_ID_MAX),
    ]


AfEventCallback = ctypes.CFUNCTYPE(None, POINTER(AfEvent), c_void_p)


class AfCaptureConfig(ctypes.Structure):
    _fields_ = [
        ("device_id", c_char_p),
        ("frame_ms", c_uint32),
        ("ring_capacity", c_uint32),
        ("emit_silence", c_int),
        ("on_event", AfEventCallback),
        ("user_data", c_void_p),
    ]


class AfVadConfig(ctypes.Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("model_data", c_void_p),
        ("model_size", c_size_t),
        ("sample_rate", c_uint32),
        ("intra_op_threads", c_uint32),
    ]


class AfChunkerConfig(ctypes.Structure):
    _fields_ = [
        ("sample_rate", c_uint32),
        ("frame_samples", c_uint32),
        ("pre_roll_ms", c_uint32),
        ("onset_frames", c_uint32),
        ("offset_ms", c_uint32),
        ("min_speech_ms", c_uint32),
        ("max_chunk_ms", c_uint32),
        ("overlap_ms", c_uint32),
        ("trailing_pad_ms", c_uint32),
        ("force_cut_window_ms", c_uint32),
        ("force_cut_search_ms", c_uint32),
        ("onset_threshold", c_float),
        ("offset_threshold", c_float),
        ("min_speech_ratio", c_float),
    ]


class AfChunkEvent(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("reason", c_int),
        ("chunk_id", c_uint64),
        ("start_sample", c_uint64),
        ("end_sample", c_uint64),
        ("start_qpc", c_uint64),
        ("end_qpc", c_uint64),
        ("sample_rate", c_uint32),
        ("speech_ms", c_uint32),
        ("speech_ratio", c_float),
        ("pcm", POINTER(c_int16)),
        ("sample_count", c_size_t),
    ]


_EXPECTED_SIZES = {AfEvent: 1584, AfChunkEvent: 80}


def _check_layout() -> None:
    for cls, expected in _EXPECTED_SIZES.items():
        actual = ctypes.sizeof(cls)
        if actual != expected:
            raise RuntimeError(
                f"{cls.__name__} size {actual} != expected {expected}; "
                "the C ABI changed, update the bindings"
            )


# -- library ---------------------------------------------------------------


class NativeError(RuntimeError):
    pass


class AudioFlow:
    """Owns the loaded DLL and the three native handles."""

    def __init__(self, dll_dir: str | None = None) -> None:
        if dll_dir is None:
            here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dll_dir = os.path.join(here, "build", "Release")
        self.dll_dir = dll_dir
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
        path = os.path.join(dll_dir, "audioflow.dll")
        self.lib = ctypes.CDLL(path)
        _check_layout()
        self._bind()
        self._capture = None
        self._vad = None
        self._chunker = None
        self._callback = None

    def _bind(self) -> None:
        lib = self.lib

        lib.af_capture_start.argtypes = [POINTER(AfCaptureConfig), POINTER(c_void_p)]
        lib.af_capture_start.restype = c_int
        lib.af_capture_stop.argtypes = [c_void_p]
        lib.af_capture_stop.restype = None
        lib.af_capture_last_error.argtypes = []
        lib.af_capture_last_error.restype = c_char_p

        lib.af_vad_create.argtypes = [POINTER(AfVadConfig), POINTER(c_void_p)]
        lib.af_vad_create.restype = c_int
        lib.af_vad_destroy.argtypes = [c_void_p]
        lib.af_vad_process.argtypes = [
            c_void_p, POINTER(c_int16), c_size_t, POINTER(c_float), POINTER(c_int)
        ]
        lib.af_vad_process.restype = c_int
        lib.af_vad_reset.argtypes = [c_void_p]
        lib.af_vad_last_error.argtypes = []
        lib.af_vad_last_error.restype = c_char_p

        event_array = POINTER(AfChunkEvent)
        lib.af_chunker_create.argtypes = [POINTER(AfChunkerConfig), POINTER(c_void_p)]
        lib.af_chunker_create.restype = c_int
        lib.af_chunker_destroy.argtypes = [c_void_p]
        lib.af_chunker_push.argtypes = [
            c_void_p, POINTER(c_int16), c_size_t, c_float, c_uint64,
            event_array, c_size_t, POINTER(c_size_t),
        ]
        lib.af_chunker_push.restype = c_int
        lib.af_chunker_snapshot.argtypes = [c_void_p, event_array]
        lib.af_chunker_snapshot.restype = c_int
        lib.af_chunker_flush.argtypes = [c_void_p, event_array, c_size_t, POINTER(c_size_t)]
        lib.af_chunker_flush.restype = c_int
        lib.af_chunker_reset.argtypes = [c_void_p, event_array, c_size_t, POINTER(c_size_t)]
        lib.af_chunker_reset.restype = c_int
        lib.af_chunk_event_release.argtypes = [event_array]
        lib.af_chunk_event_release.restype = None

    # -- capture -----------------------------------------------------------

    def capture_start(self, on_event, frame_ms: int = 32, emit_silence: bool = True) -> None:
        config = AfCaptureConfig()
        config.device_id = None
        config.frame_ms = frame_ms
        config.ring_capacity = 256
        config.emit_silence = 1 if emit_silence else 0
        self._callback = AfEventCallback(on_event)
        config.on_event = self._callback
        config.user_data = None

        handle = c_void_p()
        result = self.lib.af_capture_start(byref(config), byref(handle))
        if result != AF_OK:
            raise NativeError(f"af_capture_start failed: {result} "
                              f"({self.lib.af_capture_last_error().decode()})")
        self._capture = handle

    def capture_stop(self) -> None:
        if self._capture is not None:
            self.lib.af_capture_stop(self._capture)
            self._capture = None
            self._callback = None

    # -- vad ---------------------------------------------------------------

    def vad_open(self, model_path: str, threads: int = 1) -> None:
        config = AfVadConfig()
        config.model_path = model_path.encode("utf-8")
        config.model_data = None
        config.model_size = 0
        config.sample_rate = AF_CAPTURE_SAMPLE_RATE
        config.intra_op_threads = threads

        handle = c_void_p()
        result = self.lib.af_vad_create(byref(config), byref(handle))
        if result != AF_OK:
            raise NativeError(f"af_vad_create failed: {result} "
                              f"({self.lib.af_vad_last_error().decode()})")
        self._vad = handle

    def vad_process(self, samples, count: int) -> float:
        buf = (c_int16 * count).from_address(ctypes.addressof(samples))
        probability = c_float(0.0)
        updated = c_int(0)
        result = self.lib.af_vad_process(self._vad, buf, count, byref(probability), byref(updated))
        if result != AF_OK:
            raise NativeError(f"af_vad_process failed: {result} "
                              f"({self.lib.af_vad_last_error().decode()})")
        return probability.value

    def vad_reset(self) -> None:
        self.lib.af_vad_reset(self._vad)

    # -- chunker -----------------------------------------------------------

    def chunker_open(self, **overrides) -> None:
        config = AfChunkerConfig()
        config.sample_rate = AF_CAPTURE_SAMPLE_RATE
        config.frame_samples = 512
        config.pre_roll_ms = overrides.get("pre_roll_ms", 250)
        config.onset_frames = overrides.get("onset_frames", 2)
        config.offset_ms = overrides.get("offset_ms", 400)
        config.min_speech_ms = overrides.get("min_speech_ms", 500)
        config.max_chunk_ms = overrides.get("max_chunk_ms", 4000)
        config.overlap_ms = overrides.get("overlap_ms", 300)
        config.trailing_pad_ms = overrides.get("trailing_pad_ms", 100)
        config.force_cut_window_ms = overrides.get("force_cut_window_ms", 100)
        config.force_cut_search_ms = overrides.get("force_cut_search_ms", 500)
        config.onset_threshold = overrides.get("onset_threshold", 0.5)
        config.offset_threshold = overrides.get("offset_threshold", 0.35)
        config.min_speech_ratio = overrides.get("min_speech_ratio", 0.0)

        handle = c_void_p()
        result = self.lib.af_chunker_create(byref(config), byref(handle))
        if result != AF_OK:
            raise NativeError(f"af_chunker_create failed: {result}")
        self._chunker = handle

    def chunker_push(self, samples, count: int, probability: float, qpc: int):
        events = (AfChunkEvent * 4)()
        produced = c_size_t(0)
        buf = (c_int16 * count).from_address(ctypes.addressof(samples))
        result = self.lib.af_chunker_push(
            self._chunker, buf, count, probability, qpc, events, 4, byref(produced)
        )
        if result != AF_OK:
            raise NativeError(f"af_chunker_push failed: {result}")
        return events, produced.value

    def chunker_snapshot(self):
        event = AfChunkEvent()
        result = self.lib.af_chunker_snapshot(self._chunker, byref(event))
        if result != 0:
            return None
        return event

    def chunker_flush(self):
        events = (AfChunkEvent * 4)()
        produced = c_size_t(0)
        self.lib.af_chunker_flush(self._chunker, events, 4, byref(produced))
        return events, produced.value

    def chunker_reset(self):
        events = (AfChunkEvent * 4)()
        produced = c_size_t(0)
        self.lib.af_chunker_reset(self._chunker, events, 4, byref(produced))
        return events, produced.value

    @staticmethod
    def event_pcm(event) -> bytes:
        """Copy a chunk event's PCM into a bytes object.

        The event owns the buffer (malloc'd in C); the caller must call
        ``release_event`` afterwards.
        """
        if not event.pcm or event.sample_count == 0:
            return b""
        return ctypes.string_at(event.pcm, event.sample_count * 2)

    def release_event(self, event) -> None:
        self.lib.af_chunk_event_release(byref(event))

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        self.capture_stop()
        if self._chunker is not None:
            self.lib.af_chunker_destroy(self._chunker)
            self._chunker = None
        if self._vad is not None:
            self.lib.af_vad_destroy(self._vad)
            self._vad = None
