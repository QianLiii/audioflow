#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audioflow/af_vad.h"

static uint16_t read_u16le(const unsigned char *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_u32le(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int read_wav(const char *path, int16_t **out_samples,
                    size_t *out_count)
{
    FILE *file;
    long length;
    unsigned char *buffer;
    size_t read_count;
    size_t position = 12;
    const unsigned char *fmt = NULL;
    size_t fmt_size = 0;
    const unsigned char *data = NULL;
    size_t data_size = 0;
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint16_t bits;
    size_t sample_count;
    int16_t *samples;
    size_t i;

    *out_samples = NULL;
    *out_count = 0;

    file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open WAV: %s\n", path);
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    length = ftell(file);
    if (length < 12) {
        fclose(file);
        fprintf(stderr, "WAV is too short\n");
        return -1;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }

    buffer = (unsigned char *)malloc((size_t)length);
    if (!buffer) {
        fclose(file);
        return -1;
    }
    read_count = fread(buffer, 1, (size_t)length, file);
    fclose(file);
    if (read_count != (size_t)length) {
        free(buffer);
        return -1;
    }

    if (memcmp(buffer, "RIFF", 4) != 0 ||
        memcmp(buffer + 8, "WAVE", 4) != 0) {
        free(buffer);
        fprintf(stderr, "not a RIFF/WAVE file\n");
        return -1;
    }

    while (position + 8 <= (size_t)length) {
        const unsigned char *id = buffer + position;
        uint32_t chunk_size = read_u32le(buffer + position + 4);
        size_t payload = position + 8;

        if (chunk_size > (size_t)length - payload)
            break;

        if (memcmp(id, "fmt ", 4) == 0) {
            fmt = buffer + payload;
            fmt_size = chunk_size;
        } else if (memcmp(id, "data", 4) == 0) {
            data = buffer + payload;
            data_size = chunk_size == 0xFFFFFFFFu
                            ? (size_t)length - payload
                            : chunk_size;
        }

        position = payload + chunk_size;
        if ((chunk_size & 1u) && position < (size_t)length)
            position++;
    }

    if (!fmt || fmt_size < 16 || !data) {
        free(buffer);
        fprintf(stderr, "WAV is missing fmt or data\n");
        return -1;
    }

    audio_format = read_u16le(fmt);
    channels = read_u16le(fmt + 2);
    sample_rate = read_u32le(fmt + 4);
    bits = read_u16le(fmt + 14);

    if (audio_format != 1 || channels != 1 || sample_rate != 16000 ||
        bits != 16) {
        free(buffer);
        fprintf(stderr,
                "WAV must be PCM, 16 kHz, mono, 16-bit "
                "(got format=%u rate=%u channels=%u bits=%u)\n",
                audio_format, sample_rate, channels, bits);
        return -1;
    }

    if (data_size & 1u) {
        free(buffer);
        fprintf(stderr, "WAV data size is not sample-aligned\n");
        return -1;
    }

    sample_count = data_size / 2u;
    samples = (int16_t *)malloc(sample_count * sizeof(int16_t));
    if (!samples) {
        free(buffer);
        return -1;
    }
    for (i = 0; i < sample_count; i++)
        samples[i] = (int16_t)read_u16le(data + 2u * i);

    free(buffer);
    *out_samples = samples;
    *out_count = sample_count;
    return 0;
}

int main(int argc, char **argv)
{
    af_vad_config config;
    af_vad *vad = NULL;
    int16_t *samples = NULL;
    size_t sample_count = 0;
    uint32_t window;
    int16_t *frame = NULL;
    size_t position = 0;
    size_t window_index = 0;
    size_t speech_windows = 0;
    int failed = 0;

    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <silero_vad.onnx> <16k-mono-s16.wav>\n",
                argv[0]);
        return 2;
    }

    memset(&config, 0, sizeof(config));
    config.model_path = argv[1];
    config.sample_rate = 16000;
    config.intra_op_threads = 1;

    if (af_vad_create(&config, &vad) != AF_VAD_OK) {
        fprintf(stderr, "VAD init failed: %s\n", af_vad_last_error());
        return 1;
    }

    if (read_wav(argv[2], &samples, &sample_count) != 0) {
        af_vad_destroy(vad);
        return 1;
    }

    window = af_vad_window_samples(vad);
    if (window == 0) {
        fprintf(stderr, "VAD returned a zero window\n");
        free(samples);
        af_vad_destroy(vad);
        return 1;
    }

    frame = (int16_t *)calloc(window, sizeof(int16_t));
    if (!frame) {
        free(samples);
        af_vad_destroy(vad);
        return 1;
    }

    printf("window  probability  state\n");
    while (position < sample_count) {
        size_t remaining = sample_count - position;
        size_t count = remaining < window ? remaining : window;
        float probability = 0.0f;
        int updated = 0;
        af_vad_result result;

        memset(frame, 0, (size_t)window * sizeof(int16_t));
        memcpy(frame, samples + position, count * sizeof(int16_t));

        result = af_vad_process(vad, frame, window, &probability, &updated);
        if (result != AF_VAD_OK) {
            fprintf(stderr, "VAD process failed: %s\n",
                    af_vad_last_error());
            failed = 1;
            break;
        }

        if (updated) {
            int speech = probability >= 0.5f;
            printf("%6zu  %.5f      %s\n", window_index, probability,
                   speech ? "speech" : "");
            if (speech)
                speech_windows++;
            window_index++;
        }
        position += count;
    }

    printf("windows=%zu speech_windows=%zu samples=%zu\n", window_index,
           speech_windows, sample_count);

    free(frame);
    free(samples);
    af_vad_destroy(vad);
    return failed ? 1 : 0;
}
