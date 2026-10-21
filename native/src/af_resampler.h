#ifndef AF_RESAMPLER_H
#define AF_RESAMPLER_H

#include <stddef.h>
#include <stdint.h>

/*
 * Streaming mono float32 -> mono s16le resampler.
 *
 * This is the fallback path used when WASAPI cannot deliver the canonical
 * 16 kHz mono format through AUTOCONVERTPCM. It applies a windowed-sinc
 * low-pass filter at the input rate and linearly interpolates to the output
 * rate. It is not a high-end resampler, but it removes the aliasing that a
 * bare decimator would introduce.
 *
 * af_resampler_process requires out_capacity to be at least
 * ceil(in_count * out_rate / in_rate) + 4 samples.
 */

typedef struct af_resampler af_resampler;

af_resampler *af_resampler_create(uint32_t in_rate, uint32_t out_rate);
void af_resampler_destroy(af_resampler *resampler);
void af_resampler_reset(af_resampler *resampler);

size_t af_resampler_process(af_resampler *resampler,
                            const float *in,
                            size_t in_count,
                            int16_t *out,
                            size_t out_capacity);

#endif /* AF_RESAMPLER_H */
