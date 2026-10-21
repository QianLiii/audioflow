#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <objbase.h>
#include <propsys.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <avrt.h>
#include <mmreg.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audioflow/wasapi_capture.h"
#include "af_platform.h"
#include "af_ring.h"
#include "af_resampler.h"

#ifndef AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
#define AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM 0x80000000u
#endif
#ifndef AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
#define AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY 0x08000000u
#endif
#ifndef AUDCLNT_S_BUFFER_EMPTY
#define AUDCLNT_S_BUFFER_EMPTY 0x08890001L
#endif

/* Local GUIDs keep the module independent of a specific uuid.lib layout. */
static const CLSID AF_CLSID_MMDeviceEnumerator = {
    0xBCDE0395, 0xE52F, 0x467C,
    {0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}};
static const IID AF_IID_IMMDeviceEnumerator = {
    0xA95664D2, 0x9614, 0x4F35,
    {0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}};
static const IID AF_IID_IAudioClient = {
    0x1CB9AD4C, 0xDBFA, 0x4C32,
    {0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}};
static const IID AF_IID_IAudioCaptureClient = {
    0xC8ADBD64, 0xE71E, 0x48A0,
    {0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17}};
static const IID AF_IID_IMMNotificationClient = {
    0x7991EEC9, 0x7E89, 0x4D85,
    {0x83, 0x90, 0x6C, 0x70, 0x3C, 0xEC, 0x60, 0xC0}};
static const IID AF_IID_IUnknown = {
    0x00000000, 0x0000, 0x0000,
    {0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

static const GUID AF_SUBTYPE_PCM = {
    0x00000001, 0x0000, 0x0010,
    {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
static const GUID AF_SUBTYPE_IEEE_FLOAT = {
    0x00000003, 0x0000, 0x0010,
    {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};

struct af_capture;

typedef struct af_notifier {
    IMMNotificationClientVtbl *lpVtbl;
    LONG ref;
    struct af_capture *capture;
} af_notifier;

struct af_capture {
    af_capture_config config;
    char device_id[AF_CAPTURE_DEVICE_ID_MAX];
    char current_device_id[AF_CAPTURE_DEVICE_ID_MAX];
    uint32_t frame_samples;
    uint64_t qpc_period;

    af_thread capture_thread;
    af_thread event_thread;
    af_ring ring;

    af_mutex state_lock;
    af_cond state_cond;
    af_result start_result;
    int start_ready;

    volatile LONG stopping;

    HANDLE stop_event;
    HANDLE rebuild_event;
    HANDLE capture_event;

    IMMDeviceEnumerator *enumerator;
    IMMDevice *device;
    IAudioClient *audio_client;
    IAudioCaptureClient *capture_client;
    af_notifier *notifier;
    WAVEFORMATEX *mix_format;
    int client_started;

    uint32_t input_rate;
    uint16_t input_channels;
    int input_is_float;
    int input_bits;
    int autoconverted;
    af_resampler *resampler;

    float *mono_buf;
    size_t mono_cap;
    int16_t *resample_buf;
    size_t resample_cap;

    int16_t *accum;
    size_t accum_count;
    size_t accum_cap;
    uint64_t accum_qpc;

    uint64_t frame_index;
    uint64_t next_silence_qpc;
    uint64_t last_audio_qpc;
};

enum {
    AF_DRAIN_OK = 0,
    AF_DRAIN_REBUILD = 1,
    AF_DRAIN_ERROR = -1
};

static char g_last_error[512];
static AF_THREAD_LOCAL char g_tls_error[512];
static SRWLOCK g_error_lock = SRWLOCK_INIT;

static void set_last_error(const char *fmt, ...)
{
    char buf[512];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    AcquireSRWLockExclusive(&g_error_lock);
    memcpy(g_last_error, buf, sizeof(g_last_error));
    g_last_error[sizeof(g_last_error) - 1] = 0;
    ReleaseSRWLockExclusive(&g_error_lock);
}

static void set_last_error_hr(const char *what, HRESULT hr)
{
    set_last_error("%s failed (HRESULT 0x%08lX)", what,
                   (unsigned long)hr);
}

const char *af_capture_last_error(void)
{
    AcquireSRWLockShared(&g_error_lock);
    memcpy(g_tls_error, g_last_error, sizeof(g_tls_error));
    g_tls_error[sizeof(g_tls_error) - 1] = 0;
    ReleaseSRWLockShared(&g_error_lock);
    return g_tls_error;
}

static int is_stopping(af_capture *cap)
{
    return InterlockedCompareExchange(&cap->stopping, 0, 0) != 0;
}

static void copy_string(char *dst, size_t cap, const char *src)
{
    size_t n;

    if (!dst || cap == 0)
        return;
    if (!src) {
        dst[0] = 0;
        return;
    }
    n = strlen(src);
    if (n >= cap)
        n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = 0;
}

static int wide_to_utf8(const wchar_t *src, char *dst, size_t cap)
{
    int n;

    if (!dst || cap == 0)
        return -1;
    dst[0] = 0;
    if (!src)
        return -1;
    n = WideCharToMultiByte(CP_UTF8, 0, src, -1, dst, (int)cap, NULL, NULL);
    if (n <= 0) {
        dst[0] = 0;
        return -1;
    }
    return 0;
}

static wchar_t *utf8_to_wide(const char *src)
{
    wchar_t *dst;
    int n;

    if (!src)
        return NULL;
    n = MultiByteToWideChar(CP_UTF8, 0, src, -1, NULL, 0);
    if (n <= 0)
        return NULL;
    dst = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!dst)
        return NULL;
    if (MultiByteToWideChar(CP_UTF8, 0, src, -1, dst, n) <= 0) {
        free(dst);
        return NULL;
    }
    return dst;
}

static void get_device_id(IMMDevice *device, char *dst, size_t cap)
{
    LPWSTR id = NULL;

    if (!dst || cap == 0)
        return;
    dst[0] = 0;
    if (!device)
        return;
    if (SUCCEEDED(device->lpVtbl->GetId(device, &id)) && id) {
        wide_to_utf8(id, dst, cap);
        CoTaskMemFree(id);
    }
}

/* ------------------------------------------------------------------------- */
/* IMMNotificationClient                                                     */
/* ------------------------------------------------------------------------- */

static HRESULT STDMETHODCALLTYPE notifier_QueryInterface(
    IMMNotificationClient *This, REFIID riid, void **ppv)
{
    if (!ppv)
        return E_POINTER;
    if (IsEqualIID(riid, &AF_IID_IUnknown) ||
        IsEqualIID(riid, &AF_IID_IMMNotificationClient)) {
        *ppv = This;
        This->lpVtbl->AddRef(This);
        return S_OK;
    }
    *ppv = NULL;
    return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE notifier_AddRef(IMMNotificationClient *This)
{
    af_notifier *n = (af_notifier *)This;
    return (ULONG)InterlockedIncrement(&n->ref);
}

static ULONG STDMETHODCALLTYPE notifier_Release(IMMNotificationClient *This)
{
    af_notifier *n = (af_notifier *)This;
    LONG ref = InterlockedDecrement(&n->ref);

    if (ref == 0)
        free(n);
    return (ULONG)ref;
}

static HRESULT STDMETHODCALLTYPE notifier_OnDeviceStateChanged(
    IMMNotificationClient *This, LPCWSTR device_id, DWORD new_state)
{
    af_notifier *n = (af_notifier *)This;
    (void)device_id;
    if (new_state != DEVICE_STATE_ACTIVE && n->capture)
        SetEvent(n->capture->rebuild_event);
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE notifier_OnDeviceAdded(
    IMMNotificationClient *This, LPCWSTR device_id)
{
    (void)This;
    (void)device_id;
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE notifier_OnDeviceRemoved(
    IMMNotificationClient *This, LPCWSTR device_id)
{
    af_notifier *n = (af_notifier *)This;
    (void)device_id;
    if (n->capture)
        SetEvent(n->capture->rebuild_event);
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE notifier_OnDefaultDeviceChanged(
    IMMNotificationClient *This, EDataFlow flow, ERole role,
    LPCWSTR default_device_id)
{
    af_notifier *n = (af_notifier *)This;
    (void)default_device_id;
    if (flow == eRender && (role == eConsole || role == eMultimedia)) {
        if (n->capture)
            SetEvent(n->capture->rebuild_event);
    }
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE notifier_OnPropertyValueChanged(
    IMMNotificationClient *This, LPCWSTR device_id, const PROPERTYKEY key)
{
    (void)This;
    (void)device_id;
    (void)key;
    return S_OK;
}

static IMMNotificationClientVtbl g_notifier_vtbl = {
    notifier_QueryInterface,
    notifier_AddRef,
    notifier_Release,
    notifier_OnDeviceStateChanged,
    notifier_OnDeviceAdded,
    notifier_OnDeviceRemoved,
    notifier_OnDefaultDeviceChanged,
    notifier_OnPropertyValueChanged
};

/* ------------------------------------------------------------------------- */
/* Buffers and events                                                        */
/* ------------------------------------------------------------------------- */

static int ensure_mono_buffer(af_capture *cap, size_t samples)
{
    float *buf;

    if (cap->mono_cap >= samples)
        return 0;
    buf = (float *)realloc(cap->mono_buf, samples * sizeof(float));
    if (!buf) {
        set_last_error("out of memory growing mono buffer");
        return -1;
    }
    cap->mono_buf = buf;
    cap->mono_cap = samples;
    return 0;
}

static int ensure_resample_buffer(af_capture *cap, size_t samples)
{
    int16_t *buf;

    if (cap->resample_cap >= samples)
        return 0;
    buf = (int16_t *)realloc(cap->resample_buf, samples * sizeof(int16_t));
    if (!buf) {
        set_last_error("out of memory growing resample buffer");
        return -1;
    }
    cap->resample_buf = buf;
    cap->resample_cap = samples;
    return 0;
}

static int ensure_accum_buffer(af_capture *cap, size_t extra)
{
    size_t needed = cap->accum_count + extra;
    size_t capacity;
    int16_t *buf;

    if (cap->accum_cap >= needed)
        return 0;
    capacity = needed + cap->frame_samples * 2u;
    buf = (int16_t *)realloc(cap->accum, capacity * sizeof(int16_t));
    if (!buf) {
        set_last_error("out of memory growing PCM accumulator");
        return -1;
    }
    cap->accum = buf;
    cap->accum_cap = capacity;
    return 0;
}

static void push_audio(af_capture *cap, const int16_t *samples,
                       size_t count, uint64_t qpc)
{
    af_event event;

    if (count > AF_CAPTURE_MAX_FRAME_SAMPLES)
        count = AF_CAPTURE_MAX_FRAME_SAMPLES;

    memset(&event, 0, sizeof(event));
    event.type = AF_EVENT_AUDIO;
    event.frame_index = cap->frame_index++;
    event.qpc_position = qpc;
    event.sample_count = (uint32_t)count;
    event.input_sample_rate = cap->input_rate;
    event.input_channels = cap->input_channels;
    event.autoconverted = cap->autoconverted;
    if (count)
        memcpy(event.samples, samples, count * sizeof(int16_t));
    copy_string(event.device_id, sizeof(event.device_id),
                cap->current_device_id);
    af_ring_push(&cap->ring, &event);
}

static void push_discontinuity(af_capture *cap, int32_t reason)
{
    af_event event;

    memset(&event, 0, sizeof(event));
    event.type = AF_EVENT_DISCONTINUITY;
    event.reason = reason;
    copy_string(event.device_id, sizeof(event.device_id),
                cap->current_device_id);
    af_ring_push(&cap->ring, &event);
}

static void push_format(af_capture *cap)
{
    af_event event;

    memset(&event, 0, sizeof(event));
    event.type = AF_EVENT_FORMAT;
    event.input_sample_rate = cap->input_rate;
    event.input_channels = cap->input_channels;
    event.autoconverted = cap->autoconverted;
    copy_string(event.device_id, sizeof(event.device_id),
                cap->current_device_id);
    af_ring_push(&cap->ring, &event);
}

static void append_pcm(af_capture *cap, const int16_t *samples,
                       size_t count, uint64_t qpc)
{
    if (count == 0)
        return;
    if (ensure_accum_buffer(cap, count) != 0)
        return;

    if (cap->accum_count == 0)
        cap->accum_qpc = qpc ? qpc : af_now_qpc();

    if (samples) {
        memcpy(cap->accum + cap->accum_count, samples,
               count * sizeof(int16_t));
    } else {
        memset(cap->accum + cap->accum_count, 0, count * sizeof(int16_t));
    }
    cap->accum_count += count;

    while (cap->accum_count >= cap->frame_samples) {
        push_audio(cap, cap->accum, cap->frame_samples, cap->accum_qpc);
        cap->accum_count -= cap->frame_samples;
        if (cap->accum_count) {
            memmove(cap->accum, cap->accum + cap->frame_samples,
                    cap->accum_count * sizeof(int16_t));
        }
        cap->accum_qpc += cap->qpc_period;
    }
}

/* ------------------------------------------------------------------------- */
/* Format handling                                                           */
/* ------------------------------------------------------------------------- */

static void parse_format(const WAVEFORMATEX *format, uint32_t *rate,
                         uint16_t *channels, int *is_float, int *bits)
{
    *rate = format->nSamplesPerSec;
    *channels = format->nChannels;
    *is_float = 0;
    *bits = format->wBitsPerSample;

    if (format->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) {
        *is_float = 1;
    } else if (format->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
               format->cbSize >=
                   (WORD)(sizeof(WAVEFORMATEXTENSIBLE) -
                          sizeof(WAVEFORMATEX))) {
        const WAVEFORMATEXTENSIBLE *ext =
            (const WAVEFORMATEXTENSIBLE *)format;
        if (IsEqualGUID(&ext->SubFormat, &AF_SUBTYPE_IEEE_FLOAT))
            *is_float = 1;
        else if (IsEqualGUID(&ext->SubFormat, &AF_SUBTYPE_PCM))
            *is_float = 0;
    }
}

static int convert_packet(af_capture *cap, const BYTE *data, UINT32 frames,
                          DWORD flags, UINT64 qpc)
{
    int silent = (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;

    if (!data)
        silent = 1;

    if (flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR)
        qpc = 0;

    if (cap->autoconverted) {
        const int16_t *src = silent ? NULL : (const int16_t *)data;
        append_pcm(cap, src, frames, qpc);
        return 0;
    }

    if (ensure_mono_buffer(cap, frames) != 0)
        return -1;

    if (silent) {
        memset(cap->mono_buf, 0, (size_t)frames * sizeof(float));
    } else if (cap->input_is_float) {
        const float *src = (const float *)data;
        uint16_t channels = cap->input_channels;
        UINT32 i;
        uint16_t c;

        for (i = 0; i < frames; i++) {
            float sum = 0.0f;
            for (c = 0; c < channels; c++)
                sum += src[(size_t)i * channels + c];
            cap->mono_buf[i] = sum / (float)channels;
        }
    } else if (cap->input_bits == 16) {
        const int16_t *src = (const int16_t *)data;
        uint16_t channels = cap->input_channels;
        UINT32 i;
        uint16_t c;

        for (i = 0; i < frames; i++) {
            float sum = 0.0f;
            for (c = 0; c < channels; c++)
                sum += (float)src[(size_t)i * channels + c];
            cap->mono_buf[i] = sum / ((float)channels * 32768.0f);
        }
    } else {
        set_last_error("unsupported PCM format: %u bits, %u channels",
                       (unsigned)cap->input_bits,
                       (unsigned)cap->input_channels);
        return -1;
    }

    {
        size_t out_capacity =
            (size_t)((double)frames * 16000.0 /
                     (double)cap->input_rate) + 8u;
        size_t out_count;

        if (ensure_resample_buffer(cap, out_capacity) != 0)
            return -1;

        out_count = af_resampler_process(cap->resampler, cap->mono_buf,
                                         frames, cap->resample_buf,
                                         cap->resample_cap);
        append_pcm(cap, cap->resample_buf, out_count, qpc);
    }

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Capture client lifecycle                                                  */
/* ------------------------------------------------------------------------- */

static void teardown_client(af_capture *cap)
{
    if (cap->audio_client && cap->client_started)
        cap->audio_client->lpVtbl->Stop(cap->audio_client);

    if (cap->capture_client) {
        cap->capture_client->lpVtbl->Release(cap->capture_client);
        cap->capture_client = NULL;
    }
    if (cap->audio_client) {
        cap->audio_client->lpVtbl->Release(cap->audio_client);
        cap->audio_client = NULL;
    }
    if (cap->device) {
        cap->device->lpVtbl->Release(cap->device);
        cap->device = NULL;
    }
    if (cap->capture_event) {
        CloseHandle(cap->capture_event);
        cap->capture_event = NULL;
    }
    if (cap->mix_format) {
        CoTaskMemFree(cap->mix_format);
        cap->mix_format = NULL;
    }
    if (cap->resampler) {
        af_resampler_destroy(cap->resampler);
        cap->resampler = NULL;
    }

    free(cap->mono_buf);
    free(cap->resample_buf);
    free(cap->accum);
    cap->mono_buf = NULL;
    cap->mono_cap = 0;
    cap->resample_buf = NULL;
    cap->resample_cap = 0;
    cap->accum = NULL;
    cap->accum_cap = 0;
    cap->accum_count = 0;
    cap->client_started = 0;
    cap->input_rate = 0;
    cap->input_channels = 0;
    cap->input_is_float = 0;
    cap->input_bits = 0;
    cap->autoconverted = 0;
}

static af_result ensure_capture(af_capture *cap)
{
    HRESULT hr;
    IMMDevice *device = NULL;
    IAudioClient *client = NULL;
    IAudioCaptureClient *capture_client = NULL;
    HANDLE event = NULL;
    WAVEFORMATEX *mix_format = NULL;
    af_resampler *resampler = NULL;
    int autoconverted = 0;
    int client_started = 0;
    uint32_t input_rate = 0;
    uint16_t input_channels = 0;
    int input_is_float = 0;
    int input_bits = 0;
    DWORD flags;

    if (!cap->enumerator) {
        hr = CoCreateInstance(&AF_CLSID_MMDeviceEnumerator, NULL,
                              CLSCTX_ALL, &AF_IID_IMMDeviceEnumerator,
                              (void **)&cap->enumerator);
        if (FAILED(hr) || !cap->enumerator) {
            set_last_error_hr("CoCreateInstance(MMDeviceEnumerator)", hr);
            return AF_ERR_COM;
        }
    }

    if (!cap->notifier) {
        af_notifier *notifier =
            (af_notifier *)calloc(1, sizeof(*notifier));
        if (!notifier) {
            set_last_error("out of memory creating device notifier");
            return AF_ERR_NO_MEMORY;
        }
        notifier->lpVtbl = &g_notifier_vtbl;
        notifier->ref = 1;
        notifier->capture = cap;
        hr = cap->enumerator->lpVtbl->RegisterEndpointNotificationCallback(
            cap->enumerator, (IMMNotificationClient *)notifier);
        if (FAILED(hr)) {
            set_last_error_hr("RegisterEndpointNotificationCallback", hr);
            notifier->lpVtbl->Release((IMMNotificationClient *)notifier);
            return AF_ERR_COM;
        }
        cap->notifier = notifier;
    }

    if (cap->device_id[0]) {
        wchar_t *wide_id = utf8_to_wide(cap->device_id);
        if (!wide_id) {
            set_last_error("could not convert device id to UTF-16");
            return AF_ERR_INVALID_ARG;
        }
        hr = cap->enumerator->lpVtbl->GetDevice(
            cap->enumerator, wide_id, &device);
        free(wide_id);
    } else {
        hr = cap->enumerator->lpVtbl->GetDefaultAudioEndpoint(
            cap->enumerator, eRender, eConsole, &device);
    }
    if (FAILED(hr) || !device) {
        set_last_error_hr("get render endpoint", hr);
        if (device)
            device->lpVtbl->Release(device);
        return AF_ERR_DEVICE;
    }

    hr = device->lpVtbl->Activate(device, &AF_IID_IAudioClient,
                                  CLSCTX_ALL, NULL, (void **)&client);
    if (FAILED(hr) || !client) {
        set_last_error_hr("IAudioClient activation", hr);
        device->lpVtbl->Release(device);
        return AF_ERR_COM;
    }

    flags = AUDCLNT_STREAMFLAGS_LOOPBACK |
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK;

    {
        WAVEFORMATEX canonical;
        memset(&canonical, 0, sizeof(canonical));
        canonical.wFormatTag = WAVE_FORMAT_PCM;
        canonical.nChannels = 1;
        canonical.nSamplesPerSec = AF_CAPTURE_SAMPLE_RATE;
        canonical.wBitsPerSample = 16;
        canonical.nBlockAlign = 2;
        canonical.nAvgBytesPerSec =
            AF_CAPTURE_SAMPLE_RATE * canonical.nBlockAlign;
        canonical.cbSize = 0;

        hr = client->lpVtbl->Initialize(
            client, AUDCLNT_SHAREMODE_SHARED,
            flags | (DWORD)AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                (DWORD)AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
            0, 0, &canonical, NULL);

        if (SUCCEEDED(hr)) {
            autoconverted = 1;
            input_rate = AF_CAPTURE_SAMPLE_RATE;
            input_channels = 1;
            input_is_float = 0;
            input_bits = 16;
        }
    }

    if (!autoconverted) {
        /*
         * A failed Initialize may leave the client unusable, so activate a
         * fresh IAudioClient before retrying with the engine mix format.
         */
        client->lpVtbl->Release(client);
        client = NULL;
        hr = device->lpVtbl->Activate(device, &AF_IID_IAudioClient,
                                      CLSCTX_ALL, NULL, (void **)&client);
        if (FAILED(hr) || !client) {
            set_last_error_hr("IAudioClient re-activation", hr);
            device->lpVtbl->Release(device);
            return AF_ERR_COM;
        }

        hr = client->lpVtbl->GetMixFormat(client, &mix_format);
        if (FAILED(hr) || !mix_format) {
            set_last_error_hr("GetMixFormat", hr);
            client->lpVtbl->Release(client);
            device->lpVtbl->Release(device);
            return AF_ERR_FORMAT;
        }

        hr = client->lpVtbl->Initialize(client, AUDCLNT_SHAREMODE_SHARED,
                                        flags, 0, 0, mix_format, NULL);
        if (FAILED(hr)) {
            set_last_error_hr("IAudioClient::Initialize(mix format)", hr);
            CoTaskMemFree(mix_format);
            client->lpVtbl->Release(client);
            device->lpVtbl->Release(device);
            return AF_ERR_FORMAT;
        }

        parse_format(mix_format, &input_rate, &input_channels,
                     &input_is_float, &input_bits);

        if (input_rate == 0 || input_channels == 0) {
            set_last_error("invalid mix format: rate=%u channels=%u",
                           (unsigned)input_rate,
                           (unsigned)input_channels);
            client->lpVtbl->Release(client);
            device->lpVtbl->Release(device);
            CoTaskMemFree(mix_format);
            return AF_ERR_FORMAT;
        }

        if (!(input_is_float && input_bits == 32) &&
            !(!input_is_float && input_bits == 16)) {
            set_last_error(
                "unsupported mix format: %u Hz, %u channels, %d bits, %s",
                (unsigned)input_rate, (unsigned)input_channels,
                input_bits, input_is_float ? "float" : "PCM");
            client->lpVtbl->Release(client);
            device->lpVtbl->Release(device);
            CoTaskMemFree(mix_format);
            return AF_ERR_FORMAT;
        }

        resampler = af_resampler_create(input_rate,
                                        AF_CAPTURE_SAMPLE_RATE);
        if (!resampler) {
            set_last_error("could not create resampler");
            client->lpVtbl->Release(client);
            device->lpVtbl->Release(device);
            CoTaskMemFree(mix_format);
            return AF_ERR_NO_MEMORY;
        }
    }

    event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!event) {
        set_last_error("CreateEvent failed");
        if (resampler)
            af_resampler_destroy(resampler);
        client->lpVtbl->Release(client);
        device->lpVtbl->Release(device);
        if (mix_format)
            CoTaskMemFree(mix_format);
        return AF_ERR_COM;
    }

    hr = client->lpVtbl->SetEventHandle(client, event);
    if (FAILED(hr)) {
        set_last_error_hr("SetEventHandle", hr);
        CloseHandle(event);
        if (resampler)
            af_resampler_destroy(resampler);
        client->lpVtbl->Release(client);
        device->lpVtbl->Release(device);
        if (mix_format)
            CoTaskMemFree(mix_format);
        return AF_ERR_COM;
    }

    hr = client->lpVtbl->GetService(client, &AF_IID_IAudioCaptureClient,
                                    (void **)&capture_client);
    if (FAILED(hr) || !capture_client) {
        set_last_error_hr("GetService(IAudioCaptureClient)", hr);
        CloseHandle(event);
        if (resampler)
            af_resampler_destroy(resampler);
        client->lpVtbl->Release(client);
        device->lpVtbl->Release(device);
        if (mix_format)
            CoTaskMemFree(mix_format);
        return AF_ERR_COM;
    }

    hr = client->lpVtbl->Start(client);
    if (FAILED(hr)) {
        set_last_error_hr("IAudioClient::Start", hr);
        capture_client->lpVtbl->Release(capture_client);
        CloseHandle(event);
        if (resampler)
            af_resampler_destroy(resampler);
        client->lpVtbl->Release(client);
        device->lpVtbl->Release(device);
        if (mix_format)
            CoTaskMemFree(mix_format);
        return AF_ERR_COM;
    }
    client_started = 1;

    cap->device = device;
    cap->audio_client = client;
    cap->capture_client = capture_client;
    cap->capture_event = event;
    cap->mix_format = mix_format;
    cap->resampler = resampler;
    cap->client_started = client_started;
    cap->autoconverted = autoconverted;
    cap->input_rate = input_rate;
    cap->input_channels = input_channels;
    cap->input_is_float = input_is_float;
    cap->input_bits = input_bits;
    cap->accum_count = 0;
    cap->accum_qpc = 0;
    cap->last_audio_qpc = af_now_qpc();
    cap->next_silence_qpc = cap->last_audio_qpc;

    get_device_id(device, cap->current_device_id,
                  sizeof(cap->current_device_id));
    push_format(cap);
    return AF_OK;
}

static int drain_packets(af_capture *cap)
{
    int got_audio = 0;

    for (;;) {
        UINT32 packet_frames = 0;
        BYTE *data = NULL;
        UINT32 frames = 0;
        DWORD flags = 0;
        UINT64 qpc = 0;
        HRESULT hr;

        hr = cap->capture_client->lpVtbl->GetNextPacketSize(
            cap->capture_client, &packet_frames);
        if (hr == AUDCLNT_E_DEVICE_INVALIDATED)
            return AF_DRAIN_REBUILD;
        if (FAILED(hr)) {
            set_last_error_hr("GetNextPacketSize", hr);
            return AF_DRAIN_ERROR;
        }
        if (packet_frames == 0)
            break;

        hr = cap->capture_client->lpVtbl->GetBuffer(
            cap->capture_client, &data, &frames, &flags, NULL, &qpc);
        if (hr == AUDCLNT_E_DEVICE_INVALIDATED)
            return AF_DRAIN_REBUILD;
        if (hr == AUDCLNT_S_BUFFER_EMPTY)
            break;
        if (FAILED(hr)) {
            set_last_error_hr("IAudioCaptureClient::GetBuffer", hr);
            return AF_DRAIN_ERROR;
        }

        if (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY)
            push_discontinuity(cap, AF_DISC_CAPTURE_ERROR);

        if (convert_packet(cap, data, frames, flags, qpc) != 0) {
            cap->capture_client->lpVtbl->ReleaseBuffer(
                cap->capture_client, frames);
            return AF_DRAIN_ERROR;
        }

        cap->capture_client->lpVtbl->ReleaseBuffer(
            cap->capture_client, frames);
        got_audio = 1;
    }

    if (got_audio) {
        cap->last_audio_qpc = af_now_qpc();
        cap->next_silence_qpc = cap->last_audio_qpc;
    }
    return AF_DRAIN_OK;
}

static void maybe_emit_silence(af_capture *cap)
{
    uint64_t now;
    int emitted = 0;
    int16_t zeros[AF_CAPTURE_MAX_FRAME_SAMPLES];

    if (!cap->config.emit_silence || !cap->client_started)
        return;

    now = af_now_qpc();
    if (now < cap->last_audio_qpc + 300000ull)
        return; /* 30 ms of audio or less: not idle yet */

    if (cap->next_silence_qpc + cap->qpc_period < now)
        cap->next_silence_qpc = now - cap->qpc_period;

    memset(zeros, 0, sizeof(zeros));
    while (cap->next_silence_qpc + cap->qpc_period <= now &&
           emitted < 5) {
        push_audio(cap, zeros, cap->frame_samples,
                   cap->next_silence_qpc);
        cap->next_silence_qpc += cap->qpc_period;
        emitted++;
    }
}

/* ------------------------------------------------------------------------- */
/* Threads                                                                   */
/* ------------------------------------------------------------------------- */

static void signal_start(af_capture *cap, af_result result)
{
    af_mutex_lock(&cap->state_lock);
    cap->start_result = result;
    cap->start_ready = 1;
    af_cond_broadcast(&cap->state_cond);
    af_mutex_unlock(&cap->state_lock);
}

static void event_thread_main(void *arg)
{
    af_capture *cap = (af_capture *)arg;
    uint64_t last_dropped = 0;

    for (;;) {
        af_event event;
        uint64_t dropped = af_ring_dropped(&cap->ring);
        int rc;

        if (dropped != last_dropped) {
            af_event drop_event;
            memset(&drop_event, 0, sizeof(drop_event));
            drop_event.type = AF_EVENT_DROPPED;
            drop_event.dropped_frames =
                (uint32_t)(dropped - last_dropped);
            cap->config.on_event(&drop_event, cap->config.user_data);
            last_dropped = dropped;
        }

        rc = af_ring_pop(&cap->ring, &event, 100);
        if (rc == -1)
            break;
        if (rc == 0)
            cap->config.on_event(&event, cap->config.user_data);
    }
}

static void capture_cleanup(af_capture *cap, int com_initialized,
                            HANDLE mmcss)
{
    teardown_client(cap);

    if (cap->enumerator && cap->notifier) {
        cap->enumerator->lpVtbl->UnregisterEndpointNotificationCallback(
            cap->enumerator, (IMMNotificationClient *)cap->notifier);
    }
    if (cap->notifier) {
        cap->notifier->lpVtbl->Release(
            (IMMNotificationClient *)cap->notifier);
        cap->notifier = NULL;
    }
    if (cap->enumerator) {
        cap->enumerator->lpVtbl->Release(cap->enumerator);
        cap->enumerator = NULL;
    }

    if (mmcss)
        AvRevertMmThreadCharacteristics(mmcss);
    if (com_initialized)
        CoUninitialize();
}

static void capture_thread_main(void *arg)
{
    af_capture *cap = (af_capture *)arg;
    HRESULT hr;
    int com_initialized = 0;
    DWORD mmcss_task = 0;
    HANDLE mmcss = NULL;
    af_result first;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr))
        com_initialized = 1;
    else if (hr != RPC_E_CHANGED_MODE) {
        set_last_error_hr("CoInitializeEx", hr);
        signal_start(cap, AF_ERR_COM);
        return;
    } else {
        /* An existing apartment is usable for these calls. */
        com_initialized = 0;
    }

    mmcss = AvSetMmThreadCharacteristicsW(L"Audio", &mmcss_task);

    first = ensure_capture(cap);
    signal_start(cap, first);
    if (first != AF_OK) {
        capture_cleanup(cap, com_initialized, mmcss);
        return;
    }

    while (!is_stopping(cap)) {
        HANDLE handles[3];
        DWORD wait_result;
        int drain_result;

        if (!cap->audio_client) {
            if (ensure_capture(cap) != AF_OK) {
                af_sleep_ms(250);
                continue;
            }
        }

        handles[0] = cap->stop_event;
        handles[1] = cap->rebuild_event;
        handles[2] = cap->capture_event;

        wait_result = WaitForMultipleObjects(3, handles, FALSE, 10);
        if (wait_result == WAIT_OBJECT_0)
            break;

        if (wait_result == WAIT_OBJECT_0 + 1) {
            ResetEvent(cap->rebuild_event);
            push_discontinuity(cap, AF_DISC_DEVICE_CHANGED);
            teardown_client(cap);
            continue;
        }

        if (wait_result == WAIT_OBJECT_0 + 2) {
            drain_result = drain_packets(cap);
            if (drain_result == AF_DRAIN_REBUILD) {
                push_discontinuity(cap, AF_DISC_DEVICE_INVALIDATED);
                teardown_client(cap);
                continue;
            }
            if (drain_result == AF_DRAIN_ERROR) {
                push_discontinuity(cap, AF_DISC_CAPTURE_ERROR);
                teardown_client(cap);
                af_sleep_ms(250);
                continue;
            }
        } else if (wait_result == WAIT_FAILED) {
            push_discontinuity(cap, AF_DISC_CAPTURE_ERROR);
            teardown_client(cap);
            af_sleep_ms(250);
            continue;
        }

        maybe_emit_silence(cap);
    }

    capture_cleanup(cap, com_initialized, mmcss);
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                */
/* ------------------------------------------------------------------------- */

static void destroy_capture(af_capture *cap)
{
    if (!cap)
        return;
    if (cap->stop_event) {
        CloseHandle(cap->stop_event);
        cap->stop_event = NULL;
    }
    if (cap->rebuild_event) {
        CloseHandle(cap->rebuild_event);
        cap->rebuild_event = NULL;
    }
    af_ring_destroy(&cap->ring);
    af_cond_destroy(&cap->state_cond);
    af_mutex_destroy(&cap->state_lock);
    free(cap);
}

af_result af_capture_start(const af_capture_config *config,
                           af_capture **out_capture)
{
    af_capture *cap;
    uint32_t frame_ms;
    uint32_t ring_capacity;
    uint64_t deadline;

    if (!config || !out_capture || !config->on_event)
        return AF_ERR_INVALID_ARG;
    *out_capture = NULL;

    frame_ms = config->frame_ms ? config->frame_ms : AF_CAPTURE_FRAME_MS;
    if (frame_ms != 10 && frame_ms != 20 && frame_ms != 32 &&
        frame_ms != 40)
        return AF_ERR_INVALID_ARG;

    cap = (af_capture *)calloc(1, sizeof(*cap));
    if (!cap)
        return AF_ERR_NO_MEMORY;

    cap->config = *config;
    cap->frame_samples = AF_CAPTURE_SAMPLE_RATE / 1000u * frame_ms;
    cap->qpc_period = (uint64_t)cap->frame_samples * 10000000ull /
                      AF_CAPTURE_SAMPLE_RATE;
    copy_string(cap->device_id, sizeof(cap->device_id),
                config->device_id);

    ring_capacity = config->ring_capacity ? config->ring_capacity : 128;
    if (ring_capacity < 4)
        ring_capacity = 4;

    if (af_ring_init(&cap->ring, ring_capacity) != 0) {
        free(cap);
        return AF_ERR_NO_MEMORY;
    }
    if (af_mutex_init(&cap->state_lock) != 0) {
        af_ring_destroy(&cap->ring);
        free(cap);
        return AF_ERR_NO_MEMORY;
    }
    if (af_cond_init(&cap->state_cond) != 0) {
        af_mutex_destroy(&cap->state_lock);
        af_ring_destroy(&cap->ring);
        free(cap);
        return AF_ERR_NO_MEMORY;
    }

    cap->stop_event = CreateEventW(NULL, TRUE, FALSE, NULL);
    cap->rebuild_event = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!cap->stop_event || !cap->rebuild_event) {
        set_last_error("CreateEvent failed");
        destroy_capture(cap);
        return AF_ERR_COM;
    }

    if (af_thread_create(&cap->event_thread, event_thread_main, cap) != 0) {
        set_last_error("could not create event thread");
        destroy_capture(cap);
        return AF_ERR_THREAD;
    }

    if (af_thread_create(&cap->capture_thread, capture_thread_main,
                         cap) != 0) {
        set_last_error("could not create capture thread");
        InterlockedExchange(&cap->stopping, 1);
        af_ring_stop(&cap->ring);
        af_thread_join(&cap->event_thread);
        destroy_capture(cap);
        return AF_ERR_THREAD;
    }

    deadline = af_now_qpc() + 50000000ull; /* 5 seconds */
    af_mutex_lock(&cap->state_lock);
    while (!cap->start_ready && af_now_qpc() < deadline)
        af_cond_wait_ms(&cap->state_cond, &cap->state_lock, 100);

    if (!cap->start_ready) {
        af_mutex_unlock(&cap->state_lock);
        set_last_error("timed out waiting for the capture client");
        af_capture_stop(cap);
        return AF_ERR_TIMEOUT;
    }

    if (cap->start_result != AF_OK) {
        af_result result = cap->start_result;
        af_mutex_unlock(&cap->state_lock);
        af_capture_stop(cap);
        return result;
    }
    af_mutex_unlock(&cap->state_lock);

    *out_capture = cap;
    return AF_OK;
}

void af_capture_stop(af_capture *cap)
{
    if (!cap)
        return;

    InterlockedExchange(&cap->stopping, 1);
    if (cap->stop_event)
        SetEvent(cap->stop_event);
    af_ring_stop(&cap->ring);

    af_thread_join(&cap->capture_thread);
    af_thread_join(&cap->event_thread);

    destroy_capture(cap);
}
