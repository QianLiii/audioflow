#include "af_resampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AF_RESAMPLER_TAPS 127u
#define AF_RESAMPLER_FILT_RING 256u

struct af_resampler {
    uint32_t in_rate;
    uint32_t out_rate;
    int passthrough;
    double delay;
    double step;

    float coeff[AF_RESAMPLER_TAPS];

    float hist[AF_RESAMPLER_TAPS];
    uint32_t hist_pos;

    uint64_t filt_count;
    float filt_ring[AF_RESAMPLER_FILT_RING];
    double next_pos;
};

static int16_t float_to_s16(float value)
{
    if (value > 1.0f)
        value = 1.0f;
    else if (value < -1.0f)
        value = -1.0f;
    return (int16_t)lrintf(value * 32767.0f);
}

static void design_lowpass(af_resampler *r)
{
    const uint32_t taps = AF_RESAMPLER_TAPS;
    const double center = (double)(taps - 1) / 2.0;
    const double min_rate = (double)(r->in_rate < r->out_rate ? r->in_rate
                                                               : r->out_rate);
    const double fc = 0.45 * min_rate / (double)r->in_rate;
    double sum = 0.0;
    uint32_t n;
    uint32_t k;

    for (n = 0; n < taps; n++) {
        double x = (double)n - center;
        double t = 2.0 * fc * x;
        double sinc;
        double window;
        double h;

        if (fabs(t) < 1e-12)
            sinc = 1.0;
        else
            sinc = sin(M_PI * t) / (M_PI * t);

        window = 0.42 -
                 0.5 * cos(2.0 * M_PI * (double)n / (double)(taps - 1)) +
                 0.08 * cos(4.0 * M_PI * (double)n / (double)(taps - 1));

        h = 2.0 * fc * sinc * window;
        r->coeff[n] = (float)h;
        sum += h;
    }

    if (fabs(sum) > 1e-12) {
        for (k = 0; k < taps; k++)
            r->coeff[k] = (float)((double)r->coeff[k] / sum);
    }
}

af_resampler *af_resampler_create(uint32_t in_rate, uint32_t out_rate)
{
    af_resampler *r;

    if (in_rate == 0 || out_rate == 0)
        return NULL;

    r = (af_resampler *)calloc(1, sizeof(*r));
    if (!r)
        return NULL;

    r->in_rate = in_rate;
    r->out_rate = out_rate;
    r->passthrough = (in_rate == out_rate);
    r->step = (double)in_rate / (double)out_rate;
    r->delay = (double)(AF_RESAMPLER_TAPS - 1) / 2.0;

    if (!r->passthrough)
        design_lowpass(r);

    af_resampler_reset(r);
    return r;
}

void af_resampler_destroy(af_resampler *resampler)
{
    free(resampler);
}

void af_resampler_reset(af_resampler *r)
{
    if (!r)
        return;
    memset(r->hist, 0, sizeof(r->hist));
    memset(r->filt_ring, 0, sizeof(r->filt_ring));
    r->hist_pos = 0;
    r->filt_count = 0;
    r->next_pos = r->delay;
}

size_t af_resampler_process(af_resampler *r,
                            const float *in,
                            size_t in_count,
                            int16_t *out,
                            size_t out_capacity)
{
    size_t produced = 0;
    size_t i;

    if (!r || !in || !out || out_capacity == 0)
        return 0;

    if (r->passthrough) {
        size_t n = in_count < out_capacity ? in_count : out_capacity;
        for (i = 0; i < n; i++)
            out[i] = float_to_s16(in[i]);
        return n;
    }

    for (i = 0; i < in_count; i++) {
        float y = 0.0f;
        uint32_t k;
        uint32_t idx;

        r->hist[r->hist_pos] = in[i];
        for (k = 0; k < AF_RESAMPLER_TAPS; k++) {
            idx = (r->hist_pos + AF_RESAMPLER_TAPS - k) % AF_RESAMPLER_TAPS;
            y += r->coeff[k] * r->hist[idx];
        }
        r->hist_pos = (r->hist_pos + 1u) % AF_RESAMPLER_TAPS;

        r->filt_ring[r->filt_count % AF_RESAMPLER_FILT_RING] = y;
        r->filt_count++;

        while (r->filt_count > (uint64_t)floor(r->next_pos) + 1u) {
            uint64_t i0 = (uint64_t)floor(r->next_pos);
            double frac = r->next_pos - (double)i0;
            float s0;
            float s1;
            float value;

            if (r->filt_count - i0 > AF_RESAMPLER_FILT_RING)
                break; /* cannot happen with the configured rates */

            s0 = r->filt_ring[i0 % AF_RESAMPLER_FILT_RING];
            s1 = r->filt_ring[(i0 + 1u) % AF_RESAMPLER_FILT_RING];
            value = s0 + (float)frac * (s1 - s0);

            if (produced >= out_capacity)
                return produced;

            out[produced++] = float_to_s16(value);
            r->next_pos += r->step;
        }
    }

    return produced;
}
