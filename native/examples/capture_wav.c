/*
 * capture_wav: record the default render endpoint to a 16 kHz mono WAV.
 *
 * Development tool. It exercises the full capture path (WASAPI loopback,
 * resampling, frame assembly, delivery ring) and writes the same PCM the
 * Python ASR layer will receive, so a recording can be replayed offline.
 *
 * Usage:
 *   capture_wav.exe [output.wav] [seconds]
 *
 * Defaults: capture.wav, 15 seconds. Press Ctrl+C to stop early; the WAV
 * header is finalized either way.
 */

#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audioflow/wasapi_capture.h"

static FILE *g_out;
static unsigned long g_samples;
static unsigned long g_frames;
static unsigned long g_dropped;
static unsigned long g_discontinuities;
static volatile LONG g_running = 1;

static void write_u32(FILE *f, unsigned long v)
{
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xffu);
    b[1] = (unsigned char)((v >> 8) & 0xffu);
    b[2] = (unsigned char)((v >> 16) & 0xffu);
    b[3] = (unsigned char)((v >> 24) & 0xffu);
    fwrite(b, 1, 4, f);
}

static void write_u16(FILE *f, unsigned int v)
{
    unsigned char b[2];
    b[0] = (unsigned char)(v & 0xffu);
    b[1] = (unsigned char)((v >> 8) & 0xffu);
    fwrite(b, 1, 2, f);
}

static void write_header(FILE *f, unsigned long data_bytes)
{
    fseek(f, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, f);
    write_u32(f, 36u + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    write_u32(f, 16u);
    write_u16(f, 1u);             /* PCM */
    write_u16(f, 1u);             /* mono */
    write_u32(f, AF_CAPTURE_SAMPLE_RATE);
    write_u32(f, AF_CAPTURE_SAMPLE_RATE * 2u);
    write_u16(f, 2u);             /* block align */
    write_u16(f, 16u);            /* bits per sample */
    fwrite("data", 1, 4, f);
    write_u32(f, data_bytes);
}

static void on_event(const af_event *event, void *user_data)
{
    (void)user_data;

    switch (event->type) {
    case AF_EVENT_AUDIO:
        if (event->sample_count == 0)
            break;
        fwrite(event->samples, sizeof(int16_t), event->sample_count, g_out);
        g_samples += event->sample_count;
        g_frames++;
        break;
    case AF_EVENT_DROPPED:
        g_dropped += event->dropped_frames;
        break;
    case AF_EVENT_DISCONTINUITY:
        g_discontinuities++;
        fprintf(stderr, "discontinuity reason=%d\n", event->reason);
        break;
    case AF_EVENT_FORMAT:
        fprintf(stderr, "format rate=%u channels=%u autoconverted=%d\n",
                event->input_sample_rate, event->input_channels,
                event->autoconverted);
        break;
    default:
        break;
    }
}

static BOOL WINAPI on_console_ctrl(DWORD type)
{
    (void)type;
    InterlockedExchange(&g_running, 0);
    return TRUE;
}

int main(int argc, char **argv)
{
    af_capture_config config;
    af_capture *capture = NULL;
    af_result result;
    const char *path = argc > 1 ? argv[1] : "capture.wav";
    int seconds = argc > 2 ? atoi(argv[2]) : 15;
    int i;

    if (seconds <= 0)
        seconds = 15;

    g_out = fopen(path, "wb");
    if (!g_out) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    write_header(g_out, 0);

    memset(&config, 0, sizeof(config));
    config.frame_ms = 32;         /* one Silero VAD step */
    config.emit_silence = 1;
    config.on_event = on_event;

    SetConsoleCtrlHandler(on_console_ctrl, TRUE);

    result = af_capture_start(&config, &capture);
    if (result != AF_OK) {
        fprintf(stderr, "capture start failed: %d (%s)\n", (int)result,
                af_capture_last_error());
        fclose(g_out);
        return 1;
    }

    fprintf(stderr, "recording %d s to %s\n", seconds, path);
    for (i = 0; i < seconds * 10 && g_running; i++)
        Sleep(100);

    af_capture_stop(capture);

    write_header(g_out, (unsigned long)g_samples * 2u);
    fclose(g_out);

    fprintf(stderr, "wrote %.2f s (%lu frames, %lu samples), dropped=%lu, "
                    "discontinuities=%lu\n",
            (double)g_samples / (double)AF_CAPTURE_SAMPLE_RATE, g_frames,
            g_samples, g_dropped, g_discontinuities);
    return 0;
}
