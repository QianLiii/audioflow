#include <stdio.h>
#include <string.h>

#include "af_ring.h"

static int failures;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                    \
        }                                                                  \
    } while (0)

int main(void)
{
    af_ring ring;
    af_event event;
    uint64_t i;
    int rc;

    CHECK(af_ring_init(&ring, 4) == 0);

    for (i = 0; i < 6; i++) {
        memset(&event, 0, sizeof(event));
        event.type = AF_EVENT_AUDIO;
        event.sample_count = (uint32_t)i;
        CHECK(af_ring_push(&ring, &event) == 0);
    }

    CHECK(af_ring_dropped(&ring) == 2);

    for (i = 2; i < 6; i++) {
        memset(&event, 0, sizeof(event));
        rc = af_ring_pop(&ring, &event, 10);
        CHECK(rc == 0);
        CHECK(event.sample_count == (uint32_t)i);
    }

    memset(&event, 0, sizeof(event));
    rc = af_ring_pop(&ring, &event, 10);
    CHECK(rc == 1); /* empty timeout */

    af_ring_stop(&ring);
    rc = af_ring_pop(&ring, &event, 10);
    CHECK(rc == -1);

    af_ring_destroy(&ring);

    if (failures) {
        fprintf(stderr, "%d ring test(s) failed\n", failures);
        return 1;
    }
    printf("ring tests passed\n");
    return 0;
}
