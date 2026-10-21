#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audioflow/wasapi_capture.h"

static volatile LONG g_running = 1;
static uint32_t g_frame_ms = 32;
static uint64_t g_frames_in_report;
static double g_sum_squares;
static uint64_t g_total_frames;

static BOOL WINAPI on_console_ctrl(DWORD type)
{
    (void)type;
    InterlockedExchange(&g_running, 0);
    return TRUE;
}

static void on_event(const af_event *event, void *user_data)
{
    (void)user_data;

    switch (event->type) {
    case AF_EVENT_AUDIO: {
        uint32_t report_frames = 1000u / g_frame_ms;
        uint32_t i;

        for (i = 0; i < event->sample_count; i++) {
            double v = (double)event->samples[i] / 32768.0;
            g_sum_squares += v * v;
        }
        g_frames_in_report++;
        g_total_frames++;

        if (g_frames_in_report >= report_frames) {
            double rms = sqrt(g_sum_squares /
                              (double)(g_frames_in_report *
                                       event->sample_count));
            printf("audio frames=%llu rms=%.4f qpc=%llu\n",
                   (unsigned long long)g_total_frames, rms,
                   (unsigned long long)event->qpc_position);
            fflush(stdout);
            g_frames_in_report = 0;
            g_sum_squares = 0.0;
        }
        break;
    }
    case AF_EVENT_FORMAT:
        printf("format rate=%u channels=%u autoconverted=%d device=%s\n",
               event->input_sample_rate, event->input_channels,
               event->autoconverted, event->device_id);
        fflush(stdout);
        break;
    case AF_EVENT_DISCONTINUITY:
        printf("discontinuity reason=%d device=%s\n", event->reason,
               event->device_id);
        fflush(stdout);
        break;
    case AF_EVENT_DROPPED:
        printf("dropped frames=%u\n", event->dropped_frames);
        fflush(stdout);
        break;
    default:
        break;
    }
}

int main(int argc, char **argv)
{
    af_capture_config config;
    af_capture *capture = NULL;
    af_result result;

    if (argc > 1)
        g_frame_ms = (uint32_t)atoi(argv[1]);

    memset(&config, 0, sizeof(config));
    config.frame_ms = g_frame_ms;
    config.ring_capacity = 128;
    config.emit_silence = 1;
    config.on_event = on_event;

    SetConsoleCtrlHandler(on_console_ctrl, TRUE);

    result = af_capture_start(&config, &capture);
    if (result != AF_OK) {
        fprintf(stderr, "capture start failed: %d (%s)\n", (int)result,
                af_capture_last_error());
        return 1;
    }

    printf("capturing; press Ctrl+C to stop\n");
    fflush(stdout);

    while (InterlockedCompareExchange(&g_running, 1, 1))
        Sleep(100);

    af_capture_stop(capture);
    printf("stopped after %llu frames\n",
           (unsigned long long)g_total_frames);
    return 0;
}
