#include "af_ring.h"

#include <stdlib.h>
#include <string.h>

int af_ring_init(af_ring *ring, size_t capacity)
{
    if (!ring || capacity < 2)
        return -1;

    memset(ring, 0, sizeof(*ring));
    ring->items = (af_event *)calloc(capacity, sizeof(*ring->items));
    if (!ring->items)
        return -1;
    ring->capacity = capacity;

    if (af_mutex_init(&ring->lock) != 0) {
        free(ring->items);
        ring->items = NULL;
        return -1;
    }
    if (af_cond_init(&ring->not_empty) != 0) {
        af_mutex_destroy(&ring->lock);
        free(ring->items);
        ring->items = NULL;
        return -1;
    }
    return 0;
}

void af_ring_destroy(af_ring *ring)
{
    if (!ring)
        return;
    af_cond_destroy(&ring->not_empty);
    af_mutex_destroy(&ring->lock);
    free(ring->items);
    memset(ring, 0, sizeof(*ring));
}

int af_ring_push(af_ring *ring, const af_event *event)
{
    int rc = 0;

    if (!ring || !event)
        return -1;

    af_mutex_lock(&ring->lock);
    if (ring->stopped) {
        rc = -1;
        goto done;
    }

    if (ring->count == ring->capacity) {
        /* Keep the newest audio: discard the oldest event. */
        ring->tail = (ring->tail + 1) % ring->capacity;
        ring->count--;
        ring->dropped++;
    }

    ring->items[ring->head] = *event;
    ring->head = (ring->head + 1) % ring->capacity;
    ring->count++;
    af_cond_signal(&ring->not_empty);

done:
    af_mutex_unlock(&ring->lock);
    return rc;
}

int af_ring_pop(af_ring *ring, af_event *out, uint32_t timeout_ms)
{
    int rc = 0;

    if (!ring || !out)
        return -1;

    af_mutex_lock(&ring->lock);
    while (ring->count == 0 && !ring->stopped) {
        int wait_rc = af_cond_wait_ms(&ring->not_empty, &ring->lock, timeout_ms);
        if (wait_rc == -1) {
            if (ring->count == 0 && !ring->stopped)
                rc = 1; /* timeout */
            break;
        }
        if (wait_rc == -2) {
            rc = -1;
            break;
        }
    }

    if (rc == 0) {
        if (ring->count == 0) {
            rc = -1; /* stopped and empty */
        } else {
            *out = ring->items[ring->tail];
            ring->tail = (ring->tail + 1) % ring->capacity;
            ring->count--;
        }
    }

    af_mutex_unlock(&ring->lock);
    return rc;
}

void af_ring_stop(af_ring *ring)
{
    if (!ring)
        return;
    af_mutex_lock(&ring->lock);
    ring->stopped = 1;
    af_cond_broadcast(&ring->not_empty);
    af_mutex_unlock(&ring->lock);
}

uint64_t af_ring_dropped(af_ring *ring)
{
    uint64_t dropped;

    if (!ring)
        return 0;
    af_mutex_lock(&ring->lock);
    dropped = ring->dropped;
    af_mutex_unlock(&ring->lock);
    return dropped;
}
