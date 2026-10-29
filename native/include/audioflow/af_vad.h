#ifndef AUDIOFLOW_AF_VAD_H
#define AUDIOFLOW_AF_VAD_H

/*
 * Speech probability interface for AudioFlow.
 *
 * Silero VAD (v5 or v6 streaming interface) through ONNX Runtime is the VAD
 * implementation. Silero VAD and ONNX Runtime are MIT licensed; see
 * THIRD_PARTY_NOTICES.md at the repository root.
 *
 * Input is 16 kHz mono s16le. The caller may feed arbitrary sample counts;
 * the adapter buffers them into 512-sample steps. Silero v5/v6 also requires a
 * 64-sample rolling context, which the adapter maintains internally. The
 * model itself receives 576 samples per step: 64 context + 512 new.
 * A VAD instance is stateful and is not thread-safe.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AF_API
#  if defined(_WIN32) && defined(AF_BUILD_SHARED)
#    if defined(AF_BUILDING_LIBRARY)
#      define AF_API __declspec(dllexport)
#    else
#      define AF_API __declspec(dllimport)
#    endif
#  else
#    define AF_API
#  endif
#endif

typedef enum af_vad_result {
    AF_VAD_OK = 0,
    AF_VAD_ERR_INVALID_ARG = 1,
    AF_VAD_ERR_NO_MEMORY = 2,
    AF_VAD_ERR_MODEL_IO = 3,
    AF_VAD_ERR_MODEL_FORMAT = 4,
    AF_VAD_ERR_ORT = 5,
    AF_VAD_ERR_STATE = 6
} af_vad_result;

typedef struct af_vad af_vad;

typedef struct af_vad_config {
    /*
     * Model source. If model_data is non-NULL, it is copied and model_path is
     * ignored. Otherwise model_path is read from disk. model_path is UTF-8.
     */
    const char *model_path;
    const void *model_data;
    size_t model_size;

    /*
     * 0 or 16000. The Silero v5/v6 16 kHz streaming model uses 512 new
     * samples per step plus a 64-sample rolling context maintained by the
     * adapter.
     */
    uint32_t sample_rate;

    /* ONNX Runtime intra-op threads. 0 selects 1. */
    uint32_t intra_op_threads;
} af_vad_config;

/*
 * Loads a Silero VAD streaming model (v5 or v6) and creates a VAD instance.
 * Returns AF_VAD_OK on success. On failure, returns an error and stores NULL
 * in *out_vad.
 */
AF_API af_vad_result af_vad_create(const af_vad_config *config,
                                   af_vad **out_vad);

AF_API void af_vad_destroy(af_vad *vad);

/* Clears the recurrent state and any partial model window. */
AF_API void af_vad_reset(af_vad *vad);

/*
 * Number of new samples per model step, e.g. 512. This is the caller-facing
 * step size, not the model's 576-sample input (64 context + 512 new).
 * Returns 0 for NULL.
 */
AF_API uint32_t af_vad_window_samples(const af_vad *vad);

/*
 * Feeds 16 kHz mono s16le samples.
 *
 * probability receives the most recent speech probability in [0, 1]. It is 0
 * until the first full model window has been processed.
 *
 * updated is optional. It is set to 1 when this call ran at least one model
 * window, and 0 when the samples only filled the internal buffer.
 *
 * Returns AF_VAD_OK on success. On error, the instance should be reset before
 * reuse.
 */
AF_API af_vad_result af_vad_process(af_vad *vad,
                                    const int16_t *samples,
                                    size_t count,
                                    float *probability,
                                    int *updated);

/*
 * Returns a thread-local diagnostic for the last failure on this thread.
 * The pointer is valid until the next VAD call from the same thread.
 */
AF_API const char *af_vad_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIOFLOW_AF_VAD_H */
