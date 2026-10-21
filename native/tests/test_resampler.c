#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "af_resampler.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                    \
        }                                                                  \
    } while (0)

typedef struct result {
    int16_t *samples;
    size_t count;
    size_t capacity;
} result;

static void result_init(result *r, size_t capacity)
{
    r->samples = (int16_t *)malloc(capacity * sizeof(int16_t));
    r->count = 0;
    r->capacity = capacity;
}

static void result_free(result *r)
{
    free(r->samples);
    r->samples = NULL;
    r->count = 0;
    r->capacity = 0;
}

static int run_tone(uint32_t in_rate, uint32_t out_rate, double frequency,
                    size_t in_samples, result *out)
{
    af_resampler *resampler = af_resampler_create(in_rate, out_rate);
    float *block;
    size_t block_size = in_rate / 100; /* 10 ms */
    size_t pos = 0;
    size_t phase_base = 0;

    if (!resampler)
        return -1;

    block = (float *)malloc(block_size * sizeof(float));
    if (!block) {
        af_resampler_destroy(resampler);
        return -1;
    }

    while (pos < in_samples) {
        size_t n = in_samples - pos;
        size_t out_capacity;
        size_t produced;
        size_t i;

        if (n > block_size)
            n = block_size;

        for (i = 0; i < n; i++) {
            size_t absolute = phase_base + i;
            block[i] = (float)(0.5 * sin(2.0 * M_PI * frequency *
                                         (double)absolute / (double)in_rate));
        }

        out_capacity = (size_t)ceil((double)n * (double)out_rate /
                                    (double)in_rate) + 8u;
        if (out->count + out_capacity > out->capacity)
            out_capacity = out->capacity - out->count;

        produced = af_resampler_process(resampler, block, n,
                                        out->samples + out->count,
                                        out_capacity);
        out->count += produced;
        pos += n;
        phase_base += n;
    }

    free(block);
    af_resampler_destroy(resampler);
    return 0;
}

static void analyze(const result *r, size_t skip, double *rms,
                    double *zero_crossings)
{
    double sum = 0.0;
    size_t crossings = 0;
    size_t i;

    for (i = skip; i < r->count; i++) {
        double v = (double)r->samples[i] / 32768.0;
        sum += v * v;
    }
    *rms = sqrt(sum / (double)(r->count - skip));

    for (i = skip + 1; i < r->count; i++) {
        if ((r->samples[i - 1] < 0 && r->samples[i] >= 0) ||
            (r->samples[i - 1] >= 0 && r->samples[i] < 0))
            crossings++;
    }
    *zero_crossings = (double)crossings;
}

static void test_tone(uint32_t in_rate, double frequency)
{
    const size_t in_samples = in_rate; /* one second */
    result out;
    double rms;
    double crossings;
    double expected_rms = 0.5 / sqrt(2.0);
    double expected_crossings;
    size_t skip = 200;

    result_init(&out, 24000);
    CHECK(run_tone(in_rate, 16000, frequency, in_samples, &out) == 0);
    CHECK(out.count > 15000 && out.count < 16500);

    if (out.count <= skip) {
        result_free(&out);
        return;
    }

    analyze(&out, skip, &rms, &crossings);
    expected_crossings = 2.0 * frequency *
                         (double)(out.count - skip) / 16000.0;

    printf("  %u -> 16000: count=%zu rms=%.4f (expected %.4f) "
           "crossings=%.0f (expected %.0f)\n",
           in_rate, out.count, rms, expected_rms, crossings,
           expected_crossings);

    CHECK(fabs(rms - expected_rms) < 0.03);
    CHECK(fabs(crossings - expected_crossings) <
          expected_crossings * 0.06 + 4.0);

    result_free(&out);
}

static void test_dc(void)
{
    af_resampler *r = af_resampler_create(48000, 16000);
    float *input;
    result out;
    size_t i;
    double sum = 0.0;
    size_t inspected = 0;

    CHECK(r != NULL);
    if (!r)
        return;

    input = (float *)malloc(48000 * sizeof(float));
    CHECK(input != NULL);
    if (!input) {
        af_resampler_destroy(r);
        return;
    }
    for (i = 0; i < 48000; i++)
        input[i] = 0.5f;

    result_init(&out, 20000);
    out.count = af_resampler_process(r, input, 48000, out.samples,
                                     out.capacity);
    CHECK(out.count > 15000);

    for (i = out.count / 2; i < out.count; i++) {
        sum += (double)out.samples[i];
        inspected++;
    }
    printf("  DC: count=%zu mean=%.1f (expected ~16384)\n", out.count,
           sum / (double)inspected);
    CHECK(fabs(sum / (double)inspected - 16383.5) < 100.0);

    result_free(&out);
    free(input);
    af_resampler_destroy(r);
}

static void test_passthrough(void)
{
    af_resampler *r = af_resampler_create(16000, 16000);
    float input[4] = {0.0f, 0.25f, -0.25f, 0.5f};
    int16_t output[4] = {0};

    CHECK(r != NULL);
    if (!r)
        return;

    CHECK(af_resampler_process(r, input, 4, output, 4) == 4);
    CHECK(output[0] == 0);
    CHECK(abs((int)output[1] - 8192) <= 1);
    CHECK(abs((int)output[2] + 8192) <= 1);
    CHECK(abs((int)output[3] - 16384) <= 1);

    af_resampler_destroy(r);
}

int main(void)
{
    printf("resampler tests:\n");
    test_tone(48000, 440.0);
    test_tone(44100, 440.0);
    test_dc();
    test_passthrough();

    if (failures) {
        fprintf(stderr, "%d resampler test(s) failed\n", failures);
        return 1;
    }
    printf("resampler tests passed\n");
    return 0;
}
