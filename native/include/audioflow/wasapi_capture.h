#ifndef AUDIOFLOW_WASAPI_CAPTURE_H
#define AUDIOFLOW_WASAPI_CAPTURE_H

/*
 * AudioFlow WASAPI loopback capture.
 *
 * The module captures the default render endpoint (system output) and emits
 * fixed-size 16 kHz mono s16le frames. It owns its capture and delivery
 * threads. The callback runs on the delivery thread and must not block.
 *
 * Windows 10 or later. The module is Windows-only; the ring and resampler
 * used internally are portable and have their own tests.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AF_API
#  if defined(_WIN32) && defined(AF_BUILD_SHARED)
#    if defined(AF_BUILDING_LIBRARY)
#      define AF_API __declspec(dllexport)
#    else
#      define AF_API __declspec(dllimport)
#    endif
#  else
#    define AF_API
#  endif
#endif

#define AF_CAPTURE_SAMPLE_RATE 16000u
#define AF_CAPTURE_FRAME_MS 32u
#define AF_CAPTURE_FRAME_SAMPLES \
    (AF_CAPTURE_SAMPLE_RATE / 1000u * AF_CAPTURE_FRAME_MS)
#define AF_CAPTURE_VAD_FRAME_MS 32u /* 512 samples: Silero VAD window */
#define AF_CAPTURE_MAX_FRAME_SAMPLES 640u /* 40 ms at 16 kHz */
#define AF_CAPTURE_DEVICE_ID_MAX 256u

typedef enum af_result {
    AF_OK = 0,
    AF_ERR_INVALID_ARG = 1,
    AF_ERR_NO_MEMORY = 2,
    AF_ERR_COM = 3,
    AF_ERR_DEVICE = 4,
    AF_ERR_FORMAT = 5,
    AF_ERR_THREAD = 6,
    AF_ERR_ALREADY_RUNNING = 7,
    AF_ERR_NOT_RUNNING = 8,
    AF_ERR_UNSUPPORTED = 9,
    AF_ERR_TIMEOUT = 10
} af_result;

typedef enum af_event_type {
    AF_EVENT_AUDIO = 0,
    AF_EVENT_DISCONTINUITY = 1,
    AF_EVENT_DROPPED = 2,
    AF_EVENT_FORMAT = 3
} af_event_type;

typedef enum af_discontinuity_reason {
    AF_DISC_DEVICE_CHANGED = 0,
    AF_DISC_DEVICE_INVALIDATED = 1,
    AF_DISC_CAPTURE_ERROR = 2,
    AF_DISC_OVERFLOW = 3,
    AF_DISC_STOPPED = 4
} af_discontinuity_reason;

typedef struct af_event {
    af_event_type type;

    /* Audio events. */
    uint64_t frame_index;       /* monotonic frame counter, 32 ms frames */
    uint64_t qpc_position;      /* 100 ns units; 0 when unknown */
    uint32_t sample_count;      /* number of valid samples in samples[] */
    int16_t samples[AF_CAPTURE_MAX_FRAME_SAMPLES];

    /* Format events. */
    uint32_t input_sample_rate; /* rate delivered by WASAPI */
    uint16_t input_channels;
    int32_t autoconverted;      /* nonzero when WASAPI did the conversion */

    /* Discontinuity and drop events. */
    int32_t reason;             /* af_discontinuity_reason */
    uint32_t dropped_frames;    /* AF_EVENT_DROPPED: events, usually frames */

    char device_id[AF_CAPTURE_DEVICE_ID_MAX]; /* UTF-8, NUL-terminated */
} af_event;

typedef void (*af_event_callback)(const af_event *event, void *user_data);

typedef struct af_capture_config {
    /*
     * Endpoint ID to capture. NULL selects the default console render
     * endpoint. The string is UTF-8 and is copied during start.
     */
    const char *device_id;

    /*
     * 10, 20, 32, or 40 ms. 0 selects 32 ms, which is one Silero VAD step
     * (512 new samples at 16 kHz).
     */
    uint32_t frame_ms;

    /* Delivery ring capacity in events. 0 selects 128. */
    uint32_t ring_capacity;

    /*
     * Nonzero emits silence frames while the loopback endpoint is idle.
     * Set to 0 to disable. VAD-based segmentation needs these frames.
     */
    int emit_silence;

    af_event_callback on_event;
    void *user_data;
} af_capture_config;

typedef struct af_capture af_capture;

/*
 * Starts capture and waits until the first audio client is initialized.
 * Returns AF_OK and stores the handle on success. Events may be delivered on
 * the event thread before this function returns; user_data must be valid.
 */
AF_API af_result af_capture_start(const af_capture_config *config,
                                  af_capture **out_capture);

/*
 * Stops capture, joins both threads, and frees the handle. May be called with
 * NULL. Do not call from the event callback.
 */
AF_API void af_capture_stop(af_capture *capture);

/*
 * Returns a thread-local copy of the last diagnostic message. The pointer is
 * valid until the next call from the same thread.
 */
AF_API const char *af_capture_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIOFLOW_WASAPI_CAPTURE_H */
