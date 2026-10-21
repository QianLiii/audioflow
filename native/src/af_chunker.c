#include "audioflow/af_chunker.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AF_ENERGY_HISTORY 32u

typedef struct af_energy_entry {
    size_t offset;
    uint64_t energy;
} af_energy_entry;

struct af_chunker {
    af_chunker_config cfg;

    uint32_t sample_rate;
    uint32_t frame_samples;

    size_t pre_roll_samples;
    size_t offset_samples;
    size_t min_speech_samples;
    size_t max_chunk_samples;
    size_t overlap_samples;
    size_t trailing_pad_samples;
    size_t force_window_samples;
    size_t force_search_samples;
    size_t force_window_frames;

    uint32_t onset_frames;
    float onset_threshold;
    float offset_threshold;
    float min_speech_ratio;

    uint64_t total_samples;
    uint64_t next_chunk_id;

    int open;
    int16_t *chunk;
    size_t chunk_len;
    size_t chunk_cap;
    uint64_t chunk_start_sample;
    uint64_t chunk_id;
    uint64_t chunk_start_qpc;
    uint64_t chunk_end_qpc;

    size_t chunk_speech_samples;
    size_t chunk_silence_run;
    size_t last_speech_end;
    uint32_t onset_run;

    int16_t *pre_roll;
    size_t pre_roll_len;
    size_t pre_roll_cap;

    af_energy_entry energy[AF_ENERGY_HISTORY];
    size_t energy_count;
};

static uint32_t af_cfg_u32(uint32_t value, uint32_t fallback)
{
    return value ? value : fallback;
}

static size_t af_ms_to_samples(uint32_t ms, uint32_t sample_rate)
{
    size_t samples = (size_t)ms * (size_t)sample_rate / 1000u;
    return samples ? samples : 1u;
}

static uint64_t af_qpc_delta(size_t samples, uint32_t sample_rate)
{
    return (uint64_t)samples * 10000000ull / (uint64_t)sample_rate;
}

static uint64_t af_frame_energy(const int16_t *samples, size_t count)
{
    uint64_t energy = 0;
    size_t i;

    for (i = 0; i < count; i++) {
        int64_t value = samples[i];
        energy += (uint64_t)(value * value);
    }
    return energy;
}

static void af_reset_state(af_chunker *chunker)
{
    chunker->open = 0;
    chunker->chunk = NULL;
    chunker->chunk_len = 0;
    chunker->chunk_cap = 0;
    chunker->chunk_start_sample = 0;
    chunker->chunk_id = 0;
    chunker->chunk_start_qpc = 0;
    chunker->chunk_end_qpc = 0;
    chunker->chunk_speech_samples = 0;
    chunker->chunk_silence_run = 0;
    chunker->last_speech_end = 0;
    chunker->onset_run = 0;
    chunker->pre_roll_len = 0;
    chunker->energy_count = 0;
}

static void af_add_energy(af_chunker *chunker, size_t offset,
                          uint64_t energy)
{
    if (chunker->energy_count == AF_ENERGY_HISTORY) {
        memmove(&chunker->energy[0], &chunker->energy[1],
                (AF_ENERGY_HISTORY - 1u) * sizeof(chunker->energy[0]));
        chunker->energy_count--;
    }
    chunker->energy[chunker->energy_count].offset = offset;
    chunker->energy[chunker->energy_count].energy = energy;
    chunker->energy_count++;
}

static void af_append_pre_roll(af_chunker *chunker, const int16_t *samples,
                               size_t count)
{
    if (count >= chunker->pre_roll_cap) {
        memcpy(chunker->pre_roll,
               samples + count - chunker->pre_roll_cap,
               chunker->pre_roll_cap * sizeof(int16_t));
        chunker->pre_roll_len = chunker->pre_roll_cap;
        return;
    }

    if (chunker->pre_roll_len + count > chunker->pre_roll_cap) {
        size_t drop = chunker->pre_roll_len + count - chunker->pre_roll_cap;
        memmove(chunker->pre_roll, chunker->pre_roll + drop,
                (chunker->pre_roll_len - drop) * sizeof(int16_t));
        chunker->pre_roll_len -= drop;
    }

    memcpy(chunker->pre_roll + chunker->pre_roll_len, samples,
           count * sizeof(int16_t));
    chunker->pre_roll_len += count;
}

static af_chunker_result af_append_chunk(af_chunker *chunker,
                                         const int16_t *samples,
                                         size_t count)
{
    if (chunker->chunk_len + count > chunker->chunk_cap) {
        size_t capacity = chunker->chunk_len + count;
        int16_t *buffer;

        capacity += capacity / 2u + chunker->frame_samples;
        buffer = (int16_t *)realloc(chunker->chunk,
                                    capacity * sizeof(int16_t));
        if (!buffer)
            return AF_CHUNKER_ERR_NO_MEMORY;
        chunker->chunk = buffer;
        chunker->chunk_cap = capacity;
    }

    memcpy(chunker->chunk + chunker->chunk_len, samples,
           count * sizeof(int16_t));
    chunker->chunk_len += count;
    return AF_CHUNKER_OK;
}

static void af_fill_opened(af_chunker *chunker, af_chunk_event *event)
{
    size_t length = chunker->chunk_len;
    size_t speech = chunker->chunk_speech_samples;

    memset(event, 0, sizeof(*event));
    event->type = AF_CHUNK_EVENT_OPENED;
    event->reason = AF_CHUNK_REASON_NONE;
    event->chunk_id = chunker->chunk_id;
    event->start_sample = chunker->chunk_start_sample;
    event->end_sample = chunker->chunk_start_sample + length;
    event->start_qpc = chunker->chunk_start_qpc;
    event->end_qpc = chunker->chunk_end_qpc;
    event->sample_rate = chunker->sample_rate;
    event->speech_ms = (uint32_t)((uint64_t)speech * 1000u /
                                  chunker->sample_rate);
    event->speech_ratio =
        length ? (float)speech / (float)length : 0.0f;
    event->pcm = NULL;
    event->sample_count = 0;
}

static af_chunker_result af_open_chunk(af_chunker *chunker, uint64_t qpc,
                                       uint64_t current_energy)
{
    size_t copy = chunker->pre_roll_samples < chunker->pre_roll_len
                      ? chunker->pre_roll_samples
                      : chunker->pre_roll_len;
    int16_t *buffer;
    size_t energy_offset;

    if (copy == 0)
        return AF_CHUNKER_ERR_STATE;

    buffer = (int16_t *)malloc(copy * sizeof(int16_t));
    if (!buffer)
        return AF_CHUNKER_ERR_NO_MEMORY;

    memcpy(buffer, chunker->pre_roll + chunker->pre_roll_len - copy,
           copy * sizeof(int16_t));

    chunker->chunk = buffer;
    chunker->chunk_len = copy;
    chunker->chunk_cap = copy;
    chunker->chunk_start_sample = chunker->total_samples - copy;
    chunker->chunk_id = chunker->next_chunk_id++;
    chunker->chunk_speech_samples =
        (size_t)chunker->onset_run * chunker->frame_samples;
    if (chunker->chunk_speech_samples > copy)
        chunker->chunk_speech_samples = copy;
    chunker->chunk_silence_run = 0;
    chunker->last_speech_end = copy;
    chunker->chunk_start_qpc =
        qpc > af_qpc_delta(copy, chunker->sample_rate)
            ? qpc - af_qpc_delta(copy, chunker->sample_rate)
            : 0;
    chunker->chunk_end_qpc =
        qpc ? qpc + af_qpc_delta(chunker->frame_samples,
                                 chunker->sample_rate)
            : 0;
    chunker->open = 1;
    chunker->onset_run = 0;
    chunker->energy_count = 0;

    energy_offset = copy >= chunker->frame_samples
                        ? copy - chunker->frame_samples
                        : 0;
    af_add_energy(chunker, energy_offset, current_energy);
    return AF_CHUNKER_OK;
}

static size_t af_find_force_cut(af_chunker *chunker)
{
    size_t earliest = chunker->chunk_len > chunker->force_search_samples
                          ? chunker->chunk_len - chunker->force_search_samples
                          : 0;
    size_t window = chunker->force_window_frames;
    size_t best = SIZE_MAX;
    uint64_t best_energy = UINT64_MAX;
    size_t i;
    size_t j;

    if (window == 0)
        window = 1;
    if (chunker->energy_count < window)
        return chunker->chunk_len;

    for (i = 0; i + window <= chunker->energy_count; i++) {
        size_t start = chunker->energy[i].offset;
        size_t end = chunker->energy[i + window - 1].offset +
                     chunker->frame_samples;
        uint64_t sum = 0;

        if (start < earliest || end > chunker->chunk_len)
            continue;

        for (j = 0; j < window; j++)
            sum += chunker->energy[i + j].energy;

        if (sum < best_energy) {
            best_energy = sum;
            best = i;
        }
    }

    if (best == SIZE_MAX)
        return chunker->chunk_len;

    {
        size_t start = chunker->energy[best].offset;
        size_t end = chunker->energy[best + window - 1].offset +
                     chunker->frame_samples;
        size_t cut = start + (end - start) / 2u;

        if (cut < chunker->overlap_samples)
            cut = chunker->overlap_samples;
        if (cut > chunker->chunk_len)
            cut = chunker->chunk_len;
        return cut;
    }
}

static void af_close_chunk(af_chunker *chunker, af_chunk_reason reason,
                           af_chunk_event *events, size_t event_capacity,
                           size_t *event_count)
{
    size_t total = chunker->chunk_len;
    size_t cut = chunker->last_speech_end + chunker->trailing_pad_samples;
    size_t speech = chunker->chunk_speech_samples;
    float ratio;
    af_chunk_event *event;

    if (event_capacity == 0)
        return;

    if (cut > total)
        cut = total;
    if (cut == 0)
        cut = total;
    if (speech > cut)
        speech = cut;
    ratio = cut ? (float)speech / (float)cut : 0.0f;

    event = &events[*event_count];
    memset(event, 0, sizeof(*event));
    event->chunk_id = chunker->chunk_id;
    event->start_sample = chunker->chunk_start_sample;
    event->start_qpc = chunker->chunk_start_qpc;
    event->sample_rate = chunker->sample_rate;
    event->speech_ms = (uint32_t)((uint64_t)speech * 1000u /
                                  chunker->sample_rate);
    event->speech_ratio = ratio;

    if (speech < chunker->min_speech_samples ||
        (chunker->min_speech_ratio > 0.0f &&
         ratio < chunker->min_speech_ratio)) {
        event->type = AF_CHUNK_EVENT_DISCARDED;
        event->reason = reason == AF_CHUNK_REASON_FLUSH
                            ? AF_CHUNK_REASON_FLUSH
                            : AF_CHUNK_REASON_MIN_SPEECH;
        event->end_sample = chunker->chunk_start_sample + total;
        event->end_qpc = chunker->chunk_start_qpc
                             ? chunker->chunk_start_qpc +
                                   af_qpc_delta(total, chunker->sample_rate)
                             : 0;
        event->pcm = NULL;
        event->sample_count = 0;
        free(chunker->chunk);
    } else {
        event->type = AF_CHUNK_EVENT_CLOSED;
        event->reason = reason;
        event->end_sample = chunker->chunk_start_sample + cut;
        event->end_qpc = chunker->chunk_start_qpc
                             ? chunker->chunk_start_qpc +
                                   af_qpc_delta(cut, chunker->sample_rate)
                             : 0;
        event->pcm = chunker->chunk;
        event->sample_count = cut;
    }

    (*event_count)++;
    af_reset_state(chunker);
}

static af_chunker_result af_force_close(af_chunker *chunker, uint64_t qpc,
                                        af_chunk_event *events,
                                        size_t event_capacity,
                                        size_t *event_count)
{
    size_t cut = af_find_force_cut(chunker);
    size_t tail_start = cut > chunker->overlap_samples
                            ? cut - chunker->overlap_samples
                            : 0;
    size_t tail_len = chunker->chunk_len - tail_start;
    int16_t *tail;
    size_t speech = chunker->chunk_speech_samples;
    float ratio;
    af_chunk_event *closed;
    af_chunk_event *opened;
    uint64_t old_start = chunker->chunk_start_sample;
    uint64_t old_start_qpc = chunker->chunk_start_qpc;
    uint64_t old_id = chunker->chunk_id;

    if (event_capacity < 2)
        return AF_CHUNKER_ERR_CAPACITY;

    if (tail_len == 0) {
        af_close_chunk(chunker, AF_CHUNK_REASON_FORCED, events,
                       event_capacity, event_count);
        return AF_CHUNKER_OK;
    }

    tail = (int16_t *)malloc(tail_len * sizeof(int16_t));
    if (!tail)
        return AF_CHUNKER_ERR_NO_MEMORY;
    memcpy(tail, chunker->chunk + tail_start,
           tail_len * sizeof(int16_t));

    if (speech > cut)
        speech = cut;
    ratio = cut ? (float)speech / (float)cut : 0.0f;

    closed = &events[*event_count];
    memset(closed, 0, sizeof(*closed));
    closed->type = AF_CHUNK_EVENT_CLOSED;
    closed->reason = AF_CHUNK_REASON_FORCED;
    closed->chunk_id = old_id;
    closed->start_sample = old_start;
    closed->end_sample = old_start + cut;
    closed->start_qpc = old_start_qpc;
    closed->end_qpc = old_start_qpc
                          ? old_start_qpc +
                                af_qpc_delta(cut, chunker->sample_rate)
                          : 0;
    closed->sample_rate = chunker->sample_rate;
    closed->speech_ms = (uint32_t)((uint64_t)speech * 1000u /
                                   chunker->sample_rate);
    closed->speech_ratio = ratio;
    closed->pcm = chunker->chunk;
    closed->sample_count = cut;
    (*event_count)++;

    chunker->chunk = tail;
    chunker->chunk_len = tail_len;
    chunker->chunk_cap = tail_len;
    chunker->chunk_start_sample = old_start + tail_start;
    chunker->chunk_id = chunker->next_chunk_id++;
    chunker->chunk_speech_samples = tail_len;
    chunker->chunk_silence_run = 0;
    chunker->last_speech_end = tail_len;
    chunker->chunk_start_qpc =
        old_start_qpc
            ? old_start_qpc +
                  af_qpc_delta(tail_start, chunker->sample_rate)
            : 0;
    chunker->chunk_end_qpc =
        qpc ? qpc + af_qpc_delta(chunker->frame_samples,
                                 chunker->sample_rate)
            : 0;
    chunker->energy_count = 0;
    chunker->open = 1;

    opened = &events[*event_count];
    af_fill_opened(chunker, opened);
    (*event_count)++;
    return AF_CHUNKER_OK;
}

af_chunker_result af_chunker_create(const af_chunker_config *config,
                                    af_chunker **out_chunker)
{
    af_chunker *chunker;
    af_chunker_config defaults;
    size_t pre_roll_cap;
    uint32_t sample_rate;
    uint32_t frame_samples;
    uint32_t onset_frames;

    if (!out_chunker)
        return AF_CHUNKER_ERR_INVALID_ARG;
    *out_chunker = NULL;

    memset(&defaults, 0, sizeof(defaults));
    if (config)
        defaults = *config;

    sample_rate = defaults.sample_rate ? defaults.sample_rate : 16000u;
    if (sample_rate != 16000u)
        return AF_CHUNKER_ERR_INVALID_ARG;

    frame_samples = defaults.frame_samples ? defaults.frame_samples : 512u;
    if (frame_samples == 0 || frame_samples > 4096u)
        return AF_CHUNKER_ERR_INVALID_ARG;

    chunker = (af_chunker *)calloc(1, sizeof(*chunker));
    if (!chunker)
        return AF_CHUNKER_ERR_NO_MEMORY;

    chunker->cfg = defaults;
    chunker->sample_rate = sample_rate;
    chunker->frame_samples = frame_samples;

    chunker->pre_roll_samples =
        af_ms_to_samples(af_cfg_u32(defaults.pre_roll_ms, 250u),
                         sample_rate);
    chunker->offset_samples =
        af_ms_to_samples(af_cfg_u32(defaults.offset_ms, 400u),
                         sample_rate);
    chunker->min_speech_samples =
        af_ms_to_samples(af_cfg_u32(defaults.min_speech_ms, 500u),
                         sample_rate);
    chunker->max_chunk_samples =
        af_ms_to_samples(af_cfg_u32(defaults.max_chunk_ms, 4000u),
                         sample_rate);
    chunker->overlap_samples =
        af_ms_to_samples(af_cfg_u32(defaults.overlap_ms, 300u),
                         sample_rate);
    chunker->trailing_pad_samples =
        af_ms_to_samples(af_cfg_u32(defaults.trailing_pad_ms, 100u),
                         sample_rate);
    chunker->force_window_samples =
        af_ms_to_samples(af_cfg_u32(defaults.force_cut_window_ms, 100u),
                         sample_rate);
    chunker->force_search_samples =
        af_ms_to_samples(af_cfg_u32(defaults.force_cut_search_ms, 500u),
                         sample_rate);

    onset_frames = af_cfg_u32(defaults.onset_frames, 2u);
    if (onset_frames == 0)
        onset_frames = 1;
    chunker->onset_frames = onset_frames;

    chunker->onset_threshold =
        defaults.onset_threshold > 0.0f ? defaults.onset_threshold : 0.5f;
    chunker->offset_threshold =
        defaults.offset_threshold > 0.0f ? defaults.offset_threshold : 0.35f;
    if (chunker->onset_threshold > 1.0f)
        chunker->onset_threshold = 1.0f;
    if (chunker->offset_threshold > chunker->onset_threshold)
        chunker->offset_threshold = chunker->onset_threshold;

    chunker->min_speech_ratio = defaults.min_speech_ratio;
    if (chunker->min_speech_ratio < 0.0f)
        chunker->min_speech_ratio = 0.0f;
    if (chunker->min_speech_ratio > 1.0f) {
        free(chunker);
        return AF_CHUNKER_ERR_INVALID_ARG;
    }

    if (chunker->max_chunk_samples <= chunker->min_speech_samples ||
        chunker->overlap_samples >= chunker->max_chunk_samples) {
        free(chunker);
        return AF_CHUNKER_ERR_INVALID_ARG;
    }

    if (chunker->force_search_samples < chunker->force_window_samples)
        chunker->force_search_samples = chunker->force_window_samples;

    chunker->force_window_frames =
        chunker->force_window_samples / frame_samples;
    if (chunker->force_window_frames == 0)
        chunker->force_window_frames = 1;

    pre_roll_cap = chunker->pre_roll_samples + frame_samples;
    chunker->pre_roll =
        (int16_t *)malloc(pre_roll_cap * sizeof(int16_t));
    if (!chunker->pre_roll) {
        free(chunker);
        return AF_CHUNKER_ERR_NO_MEMORY;
    }
    chunker->pre_roll_cap = pre_roll_cap;
    chunker->pre_roll_len = 0;

    chunker->next_chunk_id = 1;
    *out_chunker = chunker;
    return AF_CHUNKER_OK;
}

void af_chunker_destroy(af_chunker *chunker)
{
    if (!chunker)
        return;
    free(chunker->chunk);
    free(chunker->pre_roll);
    free(chunker);
}

af_chunker_result af_chunker_push(af_chunker *chunker,
                                  const int16_t *samples,
                                  size_t count,
                                  float speech_probability,
                                  uint64_t qpc_position,
                                  af_chunk_event *events,
                                  size_t event_capacity,
                                  size_t *event_count)
{
    size_t produced = 0;
    uint64_t energy;
    size_t frame_offset;
    af_chunker_result result;

    if (!chunker || !samples || !events || !event_count)
        return AF_CHUNKER_ERR_INVALID_ARG;
    *event_count = 0;
    if (count != chunker->frame_samples)
        return AF_CHUNKER_ERR_INVALID_ARG;
    if (event_capacity < 2)
        return AF_CHUNKER_ERR_CAPACITY;
    if (!chunker->pre_roll)
        return AF_CHUNKER_ERR_STATE;

    chunker->total_samples += count;
    energy = af_frame_energy(samples, count);

    if (!chunker->open) {
        af_append_pre_roll(chunker, samples, count);

        if (speech_probability >= chunker->onset_threshold) {
            chunker->onset_run++;
            if (chunker->onset_run >= chunker->onset_frames) {
                result = af_open_chunk(chunker, qpc_position, energy);
                if (result != AF_CHUNKER_OK)
                    return result;
                af_fill_opened(chunker, &events[produced]);
                produced++;
            }
        } else {
            chunker->onset_run = 0;
        }

        *event_count = produced;
        return AF_CHUNKER_OK;
    }

    frame_offset = chunker->chunk_len;
    result = af_append_chunk(chunker, samples, count);
    if (result != AF_CHUNKER_OK)
        return result;
    af_add_energy(chunker, frame_offset, energy);

    if (speech_probability >= chunker->onset_threshold) {
        chunker->chunk_speech_samples += count;
        chunker->last_speech_end = chunker->chunk_len;
        chunker->chunk_silence_run = 0;
    } else if (speech_probability >= chunker->offset_threshold) {
        chunker->last_speech_end = chunker->chunk_len;
        chunker->chunk_silence_run = 0;
    } else {
        chunker->chunk_silence_run += count;
    }

    chunker->chunk_end_qpc =
        qpc_position
            ? qpc_position +
                  af_qpc_delta(count, chunker->sample_rate)
            : 0;

    if (chunker->chunk_silence_run >= chunker->offset_samples) {
        af_close_chunk(chunker, AF_CHUNK_REASON_SILENCE, events,
                       event_capacity, &produced);
    } else if (chunker->chunk_len >= chunker->max_chunk_samples) {
        result = af_force_close(chunker, qpc_position, events,
                                event_capacity, &produced);
        if (result != AF_CHUNKER_OK)
            return result;
    }

    *event_count = produced;
    return AF_CHUNKER_OK;
}

af_chunker_result af_chunker_snapshot(af_chunker *chunker,
                                      af_chunk_event *event)
{
    int16_t *copy;
    size_t speech;

    if (!chunker || !event)
        return AF_CHUNKER_ERR_INVALID_ARG;
    if (!chunker->open)
        return AF_CHUNKER_ERR_STATE;

    copy = (int16_t *)malloc(chunker->chunk_len * sizeof(int16_t));
    if (!copy)
        return AF_CHUNKER_ERR_NO_MEMORY;
    memcpy(copy, chunker->chunk,
           chunker->chunk_len * sizeof(int16_t));

    speech = chunker->chunk_speech_samples;
    if (speech > chunker->chunk_len)
        speech = chunker->chunk_len;

    memset(event, 0, sizeof(*event));
    event->type = AF_CHUNK_EVENT_SNAPSHOT;
    event->reason = AF_CHUNK_REASON_PARTIAL;
    event->chunk_id = chunker->chunk_id;
    event->start_sample = chunker->chunk_start_sample;
    event->end_sample = chunker->chunk_start_sample + chunker->chunk_len;
    event->start_qpc = chunker->chunk_start_qpc;
    event->end_qpc = chunker->chunk_end_qpc;
    event->sample_rate = chunker->sample_rate;
    event->speech_ms = (uint32_t)((uint64_t)speech * 1000u /
                                  chunker->sample_rate);
    event->speech_ratio =
        chunker->chunk_len
            ? (float)speech / (float)chunker->chunk_len
            : 0.0f;
    event->pcm = copy;
    event->sample_count = chunker->chunk_len;
    return AF_CHUNKER_OK;
}

af_chunker_result af_chunker_flush(af_chunker *chunker,
                                   af_chunk_event *events,
                                   size_t event_capacity,
                                   size_t *event_count)
{
    if (!chunker || !events || !event_count)
        return AF_CHUNKER_ERR_INVALID_ARG;
    *event_count = 0;
    if (!chunker->open)
        return AF_CHUNKER_OK;
    if (event_capacity < 1)
        return AF_CHUNKER_ERR_CAPACITY;
    af_close_chunk(chunker, AF_CHUNK_REASON_FLUSH, events,
                   event_capacity, event_count);
    return AF_CHUNKER_OK;
}

af_chunker_result af_chunker_reset(af_chunker *chunker,
                                   af_chunk_event *events,
                                   size_t event_capacity,
                                   size_t *event_count)
{
    af_chunk_event *event;

    if (!chunker || !events || !event_count)
        return AF_CHUNKER_ERR_INVALID_ARG;
    *event_count = 0;

    if (!chunker->open) {
        af_reset_state(chunker);
        return AF_CHUNKER_OK;
    }
    if (event_capacity < 1)
        return AF_CHUNKER_ERR_CAPACITY;

    event = &events[0];
    memset(event, 0, sizeof(*event));
    event->type = AF_CHUNK_EVENT_DISCARDED;
    event->reason = AF_CHUNK_REASON_DISCONTINUITY;
    event->chunk_id = chunker->chunk_id;
    event->start_sample = chunker->chunk_start_sample;
    event->end_sample = chunker->chunk_start_sample + chunker->chunk_len;
    event->start_qpc = chunker->chunk_start_qpc;
    event->end_qpc = chunker->chunk_start_qpc
                         ? chunker->chunk_start_qpc +
                               af_qpc_delta(chunker->chunk_len,
                                            chunker->sample_rate)
                         : 0;
    event->sample_rate = chunker->sample_rate;
    event->speech_ms =
        (uint32_t)((uint64_t)chunker->chunk_speech_samples * 1000u /
                   chunker->sample_rate);
    event->speech_ratio =
        chunker->chunk_len
            ? (float)chunker->chunk_speech_samples /
                  (float)chunker->chunk_len
            : 0.0f;

    *event_count = 1;
    free(chunker->chunk);
    af_reset_state(chunker);
    return AF_CHUNKER_OK;
}

void af_chunk_event_release(af_chunk_event *event)
{
    if (!event)
        return;
    free(event->pcm);
    event->pcm = NULL;
    event->sample_count = 0;
}
