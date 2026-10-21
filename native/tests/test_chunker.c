#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audioflow/af_chunker.h"

#define FRAME_SAMPLES 512u
#define FRAME_QPC_DELTA 320000ull /* 32 ms in 100 ns units */

static int failures;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                    \
        }                                                                  \
    } while (0)

typedef struct stats {
    size_t opened;
    size_t closed;
    size_t discarded;
    size_t snapshots;
    size_t silence_count;
    size_t forced_count;
    size_t flush_count;
    size_t min_speech_count;
    size_t discontinuity_count;
    size_t last_closed_samples;
    uint64_t last_closed_start;
    uint32_t last_closed_speech_ms;
} stats;

static void fill_frame(int16_t *frame, int amplitude)
{
    size_t i;
    for (i = 0; i < FRAME_SAMPLES; i++)
        frame[i] = (i & 1u) ? (int16_t)amplitude : (int16_t)-amplitude;
}

static void account_event(const af_chunk_event *event, stats *st)
{
    switch (event->type) {
    case AF_CHUNK_EVENT_OPENED:
        st->opened++;
        break;
    case AF_CHUNK_EVENT_CLOSED:
        st->closed++;
        st->last_closed_samples = event->sample_count;
        st->last_closed_start = event->start_sample;
        st->last_closed_speech_ms = event->speech_ms;
        if (event->reason == AF_CHUNK_REASON_SILENCE)
            st->silence_count++;
        else if (event->reason == AF_CHUNK_REASON_FORCED)
            st->forced_count++;
        else if (event->reason == AF_CHUNK_REASON_FLUSH)
            st->flush_count++;
        break;
    case AF_CHUNK_EVENT_DISCARDED:
        st->discarded++;
        if (event->reason == AF_CHUNK_REASON_MIN_SPEECH)
            st->min_speech_count++;
        else if (event->reason == AF_CHUNK_REASON_DISCONTINUITY)
            st->discontinuity_count++;
        break;
    case AF_CHUNK_EVENT_SNAPSHOT:
        st->snapshots++;
        break;
    }
}

static void feed(af_chunker *chunker, const int16_t *frame, float probability,
                 size_t frames, uint64_t *qpc, stats *st)
{
    size_t i;

    for (i = 0; i < frames; i++) {
        af_chunk_event events[4];
        size_t count = 0;
        size_t k;
        af_chunker_result result;

        memset(events, 0, sizeof(events));
        result = af_chunker_push(chunker, frame, FRAME_SAMPLES, probability,
                                 *qpc, events, 4, &count);
        CHECK(result == AF_CHUNKER_OK);

        for (k = 0; k < count; k++) {
            account_event(&events[k], st);
            if (events[k].type == AF_CHUNK_EVENT_CLOSED ||
                events[k].type == AF_CHUNK_EVENT_SNAPSHOT) {
                af_chunk_event_release(&events[k]);
            }
        }
        *qpc += FRAME_QPC_DELTA;
    }
}

static void push_one(af_chunker *chunker, const int16_t *frame,
                     float probability, uint64_t *qpc, stats *st)
{
    af_chunk_event events[4];
    size_t count = 0;
    size_t k;

    memset(events, 0, sizeof(events));
    CHECK(af_chunker_push(chunker, frame, FRAME_SAMPLES, probability, *qpc,
                          events, 4, &count) == AF_CHUNKER_OK);
    for (k = 0; k < count; k++) {
        account_event(&events[k], st);
        if (events[k].type == AF_CHUNK_EVENT_CLOSED ||
            events[k].type == AF_CHUNK_EVENT_SNAPSHOT) {
            af_chunk_event_release(&events[k]);
        }
    }
    *qpc += FRAME_QPC_DELTA;
}

static af_chunker *make_chunker(void)
{
    af_chunker_config config;
    af_chunker *chunker = NULL;
    af_chunker_result result;

    memset(&config, 0, sizeof(config));
    config.sample_rate = 16000;
    config.frame_samples = FRAME_SAMPLES;
    config.pre_roll_ms = 250;
    config.onset_frames = 2;
    config.offset_ms = 400;
    config.min_speech_ms = 500;
    config.max_chunk_ms = 4000;
    config.overlap_ms = 300;
    config.trailing_pad_ms = 100;
    config.force_cut_window_ms = 100;
    config.force_cut_search_ms = 500;
    config.onset_threshold = 0.5f;
    config.offset_threshold = 0.35f;
    config.min_speech_ratio = 0.0f;

    result = af_chunker_create(&config, &chunker);
    CHECK(result == AF_CHUNKER_OK);
    return chunker;
}

static void test_silence_close(void)
{
    af_chunker *chunker = make_chunker();
    int16_t silence[FRAME_SAMPLES];
    int16_t speech[FRAME_SAMPLES];
    stats st;
    uint64_t qpc = 0;

    memset(&st, 0, sizeof(st));
    memset(silence, 0, sizeof(silence));
    fill_frame(speech, 10000);

    feed(chunker, silence, 0.05f, 32, &qpc, &st);
    feed(chunker, speech, 0.90f, 32, &qpc, &st);
    feed(chunker, silence, 0.05f, 32, &qpc, &st);

    CHECK(st.opened == 1);
    CHECK(st.closed == 1);
    CHECK(st.discarded == 0);
    CHECK(st.silence_count == 1);
    CHECK(st.last_closed_samples > 20000);
    CHECK(st.last_closed_samples < 28000);
    CHECK(st.last_closed_speech_ms >= 1000);
    CHECK(st.last_closed_speech_ms <= 1100);

    printf("silence close: start=%llu samples=%zu speech_ms=%u\n",
           (unsigned long long)st.last_closed_start, st.last_closed_samples,
           st.last_closed_speech_ms);

    af_chunker_destroy(chunker);
}

static void test_min_speech(void)
{
    af_chunker *chunker = make_chunker();
    int16_t silence[FRAME_SAMPLES];
    int16_t speech[FRAME_SAMPLES];
    stats st;
    uint64_t qpc = 0;

    memset(&st, 0, sizeof(st));
    memset(silence, 0, sizeof(silence));
    fill_frame(speech, 10000);

    feed(chunker, silence, 0.05f, 10, &qpc, &st);
    feed(chunker, speech, 0.90f, 2, &qpc, &st);
    feed(chunker, silence, 0.05f, 20, &qpc, &st);

    CHECK(st.opened == 1);
    CHECK(st.closed == 0);
    CHECK(st.discarded == 1);
    CHECK(st.min_speech_count == 1);

    printf("min speech: opened=%zu discarded=%zu\n", st.opened,
           st.discarded);

    af_chunker_destroy(chunker);
}

static void test_forced_cut(void)
{
    af_chunker *chunker = make_chunker();
    int16_t silence[FRAME_SAMPLES];
    int16_t speech[FRAME_SAMPLES];
    stats st;
    uint64_t qpc = 0;
    af_chunk_event events[4];
    size_t count = 0;
    size_t forced_samples;
    size_t flush_samples;

    memset(&st, 0, sizeof(st));
    memset(silence, 0, sizeof(silence));
    fill_frame(speech, 10000);

    feed(chunker, silence, 0.05f, 10, &qpc, &st);
    feed(chunker, speech, 0.90f, 156, &qpc, &st);

    CHECK(st.opened == 2);
    CHECK(st.closed == 1);
    CHECK(st.forced_count == 1);
    CHECK(st.last_closed_samples >= 56000);
    CHECK(st.last_closed_samples <= 64000);
    CHECK(st.last_closed_speech_ms >= 3500);
    forced_samples = st.last_closed_samples;

    memset(events, 0, sizeof(events));
    CHECK(af_chunker_flush(chunker, events, 4, &count) == AF_CHUNKER_OK);
    CHECK(count == 1);
    flush_samples = events[0].sample_count;
    account_event(&events[0], &st);
    if (events[0].type == AF_CHUNK_EVENT_CLOSED ||
        events[0].type == AF_CHUNK_EVENT_SNAPSHOT) {
        af_chunk_event_release(&events[0]);
    }

    CHECK(st.closed == 2);
    CHECK(st.flush_count == 1);

    printf("forced cut: closed_samples=%zu flush_samples=%zu\n",
           forced_samples, flush_samples);

    af_chunker_destroy(chunker);
}

static void test_forced_cut_low_energy(void)
{
    af_chunker *chunker = make_chunker();
    int16_t silence[FRAME_SAMPLES];
    int16_t speech[FRAME_SAMPLES];
    int16_t quiet[FRAME_SAMPLES];
    stats st;
    uint64_t qpc = 0;
    size_t i;

    memset(&st, 0, sizeof(st));
    memset(silence, 0, sizeof(silence));
    fill_frame(speech, 10000);
    fill_frame(quiet, 100);

    feed(chunker, silence, 0.05f, 10, &qpc, &st);

    for (i = 0; i < 140; i++) {
        const int16_t *frame = (i >= 110 && i <= 113) ? quiet : speech;
        push_one(chunker, frame, 0.90f, &qpc, &st);
    }

    CHECK(st.forced_count == 1);
    CHECK(st.last_closed_samples > 58000);
    CHECK(st.last_closed_samples < 63000);

    printf("low-energy forced cut: samples=%zu\n",
           st.last_closed_samples);

    af_chunker_destroy(chunker);
}

static void test_snapshot_and_reset(void)
{
    af_chunker *chunker = make_chunker();
    int16_t silence[FRAME_SAMPLES];
    int16_t speech[FRAME_SAMPLES];
    stats st;
    uint64_t qpc = 0;
    af_chunk_event event;
    af_chunk_event reset_events[2];
    size_t count = 0;
    size_t snapshot_samples;

    memset(&st, 0, sizeof(st));
    memset(silence, 0, sizeof(silence));
    fill_frame(speech, 10000);

    feed(chunker, silence, 0.05f, 10, &qpc, &st);
    feed(chunker, speech, 0.90f, 5, &qpc, &st);

    memset(&event, 0, sizeof(event));
    CHECK(af_chunker_snapshot(chunker, &event) == AF_CHUNKER_OK);
    CHECK(event.type == AF_CHUNK_EVENT_SNAPSHOT);
    CHECK(event.reason == AF_CHUNK_REASON_PARTIAL);
    CHECK(event.chunk_id != 0);
    CHECK(event.sample_count > 0);
    CHECK(event.pcm != NULL);
    snapshot_samples = event.sample_count;
    af_chunk_event_release(&event);

    memset(reset_events, 0, sizeof(reset_events));
    CHECK(af_chunker_reset(chunker, reset_events, 2, &count) ==
          AF_CHUNKER_OK);
    CHECK(count == 1);
    CHECK(reset_events[0].type == AF_CHUNK_EVENT_DISCARDED);
    CHECK(reset_events[0].reason == AF_CHUNK_REASON_DISCONTINUITY);
    account_event(&reset_events[0], &st);
    CHECK(st.discontinuity_count == 1);
    af_chunk_event_release(&reset_events[0]);

    printf("snapshot/reset: snapshot_samples=%zu\n", snapshot_samples);

    af_chunker_destroy(chunker);
}

static void test_event_capacity(void)
{
    af_chunker *chunker = make_chunker();
    int16_t frame[FRAME_SAMPLES];
    af_chunk_event event;
    size_t count = 0;

    fill_frame(frame, 10000);
    CHECK(af_chunker_push(chunker, frame, FRAME_SAMPLES, 0.9f, 0, &event,
                          1, &count) == AF_CHUNKER_ERR_CAPACITY);

    af_chunker_destroy(chunker);
}

int main(void)
{
    test_silence_close();
    test_min_speech();
    test_forced_cut();
    test_forced_cut_low_energy();
    test_snapshot_and_reset();
    test_event_capacity();

    if (failures) {
        fprintf(stderr, "%d chunker test(s) failed\n", failures);
        return 1;
    }
    printf("chunker tests passed\n");
    return 0;
}
