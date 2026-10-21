#ifndef AUDIOFLOW_AF_CHUNKER_H
#define AUDIOFLOW_AF_CHUNKER_H

/*
 * Frame-to-chunk segmentation for AudioFlow.
 *
 * The chunker consumes fixed-size 16 kHz mono s16le frames plus a speech
 * probability per frame. It applies VAD hysteresis, pre-roll, minimum speech,
 * maximum chunk length, and overlap on forced cuts. It emits chunk events.
 *
 * One instance is single-threaded. Run push, snapshot, flush, and reset on the
 * same chunker thread. The VAD and chunker are separate; the caller runs
 * af_vad_process and passes the probability here.
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

typedef enum af_chunker_result {
    AF_CHUNKER_OK = 0,
    AF_CHUNKER_ERR_INVALID_ARG = 1,
    AF_CHUNKER_ERR_NO_MEMORY = 2,
    AF_CHUNKER_ERR_STATE = 3,
    AF_CHUNKER_ERR_CAPACITY = 4
} af_chunker_result;

typedef enum af_chunk_event_type {
    AF_CHUNK_EVENT_OPENED = 0,
    AF_CHUNK_EVENT_CLOSED = 1,
    AF_CHUNK_EVENT_DISCARDED = 2,
    AF_CHUNK_EVENT_SNAPSHOT = 3
} af_chunk_event_type;

typedef enum af_chunk_reason {
    AF_CHUNK_REASON_SILENCE = 0,
    AF_CHUNK_REASON_FORCED = 1,
    AF_CHUNK_REASON_FLUSH = 2,
    AF_CHUNK_REASON_DISCONTINUITY = 3,
    AF_CHUNK_REASON_MIN_SPEECH = 4,
    AF_CHUNK_REASON_PARTIAL = 5,
    AF_CHUNK_REASON_NONE = 6
} af_chunk_reason;

typedef struct af_chunk_event {
    af_chunk_event_type type;
    af_chunk_reason reason;
    uint64_t chunk_id;

    /* Session sample timeline at 16 kHz. */
    uint64_t start_sample;
    uint64_t end_sample;

    /* QPC positions in 100 ns units; 0 when unknown. */
    uint64_t start_qpc;
    uint64_t end_qpc;

    uint32_t sample_rate;
    uint32_t speech_ms;
    float speech_ratio;

    /*
     * CLOSED and SNAPSHOT events own this buffer. Release the event with
     * af_chunk_event_release() when done. OPENED and DISCARDED leave it NULL.
     */
    int16_t *pcm;
    size_t sample_count;
} af_chunk_event;

typedef struct af_chunker_config {
    uint32_t sample_rate;         /* 0 or 16000 */
    uint32_t frame_samples;       /* 0 -> 512 (32 ms at 16 kHz) */
    uint32_t pre_roll_ms;         /* 0 -> 250 */
    uint32_t onset_frames;        /* 0 -> 2 consecutive speech frames */
    uint32_t offset_ms;           /* 0 -> 400 ms of silence closes */
    uint32_t min_speech_ms;       /* 0 -> 500 */
    uint32_t max_chunk_ms;        /* 0 -> 4000 */
    uint32_t overlap_ms;          /* 0 -> 300, forced cuts only */
    uint32_t trailing_pad_ms;     /* 0 -> 100 */
    uint32_t force_cut_window_ms; /* 0 -> 100 */
    uint32_t force_cut_search_ms; /* 0 -> 500 */
    float onset_threshold;        /* <= 0 -> 0.5 */
    float offset_threshold;       /* <= 0 -> 0.35 */
    float min_speech_ratio;       /* 0 disables the ratio gate */
} af_chunker_config;

typedef struct af_chunker af_chunker;

AF_API af_chunker_result af_chunker_create(const af_chunker_config *config,
                                           af_chunker **out_chunker);
AF_API void af_chunker_destroy(af_chunker *chunker);

/*
 * Pushes exactly config->frame_samples samples.
 *
 * events must have room for at least 2 events; 4 is safe. Closed-chunk and
 * snapshot PCM ownership transfers to the caller. Call af_chunk_event_release
 * for every CLOSED or SNAPSHOT event.
 *
 * Returns AF_CHUNKER_ERR_CAPACITY before changing state when event_capacity
 * is less than 2.
 */
AF_API af_chunker_result af_chunker_push(af_chunker *chunker,
                                         const int16_t *samples,
                                         size_t count,
                                         float speech_probability,
                                         uint64_t qpc_position,
                                         af_chunk_event *events,
                                         size_t event_capacity,
                                         size_t *event_count);

/*
 * Copies the current open chunk prefix into event as AF_CHUNK_EVENT_SNAPSHOT.
 * Returns AF_CHUNKER_ERR_STATE when no chunk is open.
 */
AF_API af_chunker_result af_chunker_snapshot(af_chunker *chunker,
                                             af_chunk_event *event);

/*
 * Closes the open chunk at the current position, with reason FLUSH.
 * Returns zero events when no chunk is open.
 */
AF_API af_chunker_result af_chunker_flush(af_chunker *chunker,
                                          af_chunk_event *events,
                                          size_t event_capacity,
                                          size_t *event_count);

/*
 * Discards the open chunk with reason DISCONTINUITY and clears pre-roll.
 * The session sample timeline continues. The caller should reset the VAD.
 */
AF_API af_chunker_result af_chunker_reset(af_chunker *chunker,
                                          af_chunk_event *events,
                                          size_t event_capacity,
                                          size_t *event_count);

/* Frees event->pcm if present. Safe to call on a zeroed event. */
AF_API void af_chunk_event_release(af_chunk_event *event);

#ifdef __cplusplus
}
#endif

#endif /* AUDIOFLOW_AF_CHUNKER_H */
