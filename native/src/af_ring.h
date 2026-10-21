#ifndef AF_RING_H
#define AF_RING_H

#include <stddef.h>
#include <stdint.h>

#include "audioflow/wasapi_capture.h"
#include "af_platform.h"

typedef struct af_ring {
    af_event *items;
    size_t capacity;
    size_t head; /* next write */
    size_t tail; /* next read */
    size_t count;
    uint64_t dropped;
    int stopped;
    af_mutex lock;
    af_cond not_empty;
} af_ring;

int af_ring_init(af_ring *ring, size_t capacity);
void af_ring_destroy(af_ring *ring);

/*
 * Copies the event into the ring. When full, the oldest event is discarded
 * and counted in dropped. Returns 0 on success, -1 when the ring is stopped.
 */
int af_ring_push(af_ring *ring, const af_event *event);

/*
 * Returns 0 when an event was copied to out, 1 on timeout, -1 when stopped.
 */
int af_ring_pop(af_ring *ring, af_event *out, uint32_t timeout_ms);

void af_ring_stop(af_ring *ring);
uint64_t af_ring_dropped(af_ring *ring);

#endif /* AF_RING_H */
