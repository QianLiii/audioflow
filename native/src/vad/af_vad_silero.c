/*
 * Silero VAD and ONNX Runtime are MIT licensed.
 * See THIRD_PARTY_NOTICES.md at the repository root.
 */

#include "audioflow/af_vad.h"

#include <onnxruntime_c_api.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wchar.h>
#endif

#define AF_VAD_MAX_MODEL_BYTES (256u * 1024u * 1024u)
#define AF_VAD_MAX_DIMS 8u

#if defined(_MSC_VER)
#define AF_VAD_TLS __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define AF_VAD_TLS _Thread_local
#else
#define AF_VAD_TLS
#endif

static AF_VAD_TLS char g_vad_error[512];

struct af_vad {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *memory_info;

    OrtValue *input_tensor;
    OrtValue *state_tensor;
    OrtValue *sr_tensor;

    void *model_bytes;
    size_t model_size;

    float *input_buf;       /* model input: context + new samples */
    float *pending_buf;     /* new samples waiting for a full step */
    float *context_buf;     /* previous step's last 64 samples */
    float *state_buf;
    int64_t *state_shape;
    size_t state_elems;
    size_t state_rank;
    size_t window_samples;      /* new samples per step, e.g. 512 */
    size_t context_samples;     /* rolling context, e.g. 64 */
    size_t model_input_samples; /* context + window, e.g. 576 */

    int sr_rank;
    int64_t sr_value;

    size_t fill;
    float last_probability;
    int has_probability;
};

static void set_error(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(g_vad_error, sizeof(g_vad_error), fmt, ap);
    va_end(ap);
    g_vad_error[sizeof(g_vad_error) - 1] = 0;
}

const char *af_vad_last_error(void)
{
    return g_vad_error;
}

static void set_error_status(af_vad *vad, OrtStatus *status,
                             const char *what)
{
    const char *message = vad->api->GetErrorMessage(status);
    set_error("%s: %s", what, message ? message : "unknown ONNX Runtime error");
    vad->api->ReleaseStatus(status);
}

#ifdef _WIN32
static wchar_t *utf8_to_wide(const char *text)
{
    wchar_t *wide;
    int count;

    if (!text)
        return NULL;
    count = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
    if (count <= 0)
        return NULL;
    wide = (wchar_t *)malloc((size_t)count * sizeof(wchar_t));
    if (!wide)
        return NULL;
    if (MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, count) <= 0) {
        free(wide);
        return NULL;
    }
    return wide;
}

static FILE *open_binary_file(const char *path)
{
    wchar_t *wide = utf8_to_wide(path);
    FILE *file;

    if (!wide)
        return NULL;
    file = _wfopen(wide, L"rb");
    free(wide);
    return file;
}
#else
static FILE *open_binary_file(const char *path)
{
    return fopen(path, "rb");
}
#endif

static af_vad_result read_model_file(const char *path, void **out_data,
                                     size_t *out_size)
{
    FILE *file;
    long length;
    void *data;
    size_t read_count;

    *out_data = NULL;
    *out_size = 0;

    file = open_binary_file(path);
    if (!file) {
        set_error("cannot open model file: %s", path);
        return AF_VAD_ERR_MODEL_IO;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        set_error("cannot seek model file: %s", path);
        return AF_VAD_ERR_MODEL_IO;
    }
    length = ftell(file);
    if (length <= 0 || (unsigned long)length > AF_VAD_MAX_MODEL_BYTES) {
        fclose(file);
        set_error("model file has invalid size: %s", path);
        return AF_VAD_ERR_MODEL_IO;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        set_error("cannot rewind model file: %s", path);
        return AF_VAD_ERR_MODEL_IO;
    }

    data = malloc((size_t)length);
    if (!data) {
        fclose(file);
        set_error("out of memory reading model file");
        return AF_VAD_ERR_NO_MEMORY;
    }

    read_count = fread(data, 1, (size_t)length, file);
    fclose(file);
    if (read_count != (size_t)length) {
        free(data);
        set_error("short read from model file: %s", path);
        return AF_VAD_ERR_MODEL_IO;
    }

    *out_data = data;
    *out_size = (size_t)length;
    return AF_VAD_OK;
}

static int check_io_name(af_vad *vad, size_t index, int is_input,
                         const char *expected)
{
    OrtAllocator *allocator = NULL;
    OrtStatus *status;
    char *name = NULL;
    int matches;

    status = vad->api->GetAllocatorWithDefaultOptions(&allocator);
    if (status) {
        set_error_status(vad, status, "GetAllocatorWithDefaultOptions");
        return -1;
    }

    if (is_input) {
        status = vad->api->SessionGetInputName(vad->session, index,
                                               allocator, &name);
    } else {
        status = vad->api->SessionGetOutputName(vad->session, index,
                                                allocator, &name);
    }
    if (status) {
        set_error_status(vad, status,
                         is_input ? "SessionGetInputName"
                                  : "SessionGetOutputName");
        return -1;
    }

    matches = name && strcmp(name, expected) == 0;
    if (!matches) {
        set_error("expected %s %zu named '%s', got '%s'",
                  is_input ? "input" : "output", index, expected,
                  name ? name : "(null)");
    }

    if (name)
        allocator->Free(allocator, name);
    return matches ? 0 : -1;
}

static int get_tensor_meta(af_vad *vad, OrtTypeInfo *type_info,
                           ONNXTensorElementDataType *element_type,
                           size_t *rank, int64_t *dims, size_t dims_capacity)
{
    const OrtTensorTypeAndShapeInfo *info;
    OrtStatus *status;
    size_t count = 0;

    /*
     * ONNX Runtime headers expose two shapes for this call. Current headers
     * (verified with 1.31) return OrtStatus and write the tensor info through
     * an out-parameter; older headers (1.20 and earlier) return the pointer
     * directly. The default assumes the newer form for ORT_API_VERSION >= 22;
     * define AF_ORT_CAST_TENSOR_INFO_STATUS to 0 or 1 to override it when
     * building against an untested version.
     */
#ifndef AF_ORT_CAST_TENSOR_INFO_STATUS
#  if ORT_API_VERSION >= 22
#    define AF_ORT_CAST_TENSOR_INFO_STATUS 1
#  else
#    define AF_ORT_CAST_TENSOR_INFO_STATUS 0
#  endif
#endif

#if AF_ORT_CAST_TENSOR_INFO_STATUS
    status = vad->api->CastTypeInfoToTensorInfo(type_info, &info);
    if (status) {
        set_error_status(vad, status, "CastTypeInfoToTensorInfo");
        return -1;
    }
#else
    info = vad->api->CastTypeInfoToTensorInfo(type_info);
#endif
    if (!info) {
        set_error("model I/O is not a tensor");
        return -1;
    }

    status = vad->api->GetTensorElementType(info, element_type);
    if (status) {
        set_error_status(vad, status, "GetTensorElementType");
        return -1;
    }

    status = vad->api->GetDimensionsCount(info, &count);
    if (status) {
        set_error_status(vad, status, "GetDimensionsCount");
        return -1;
    }
    if (count > dims_capacity) {
        set_error("tensor rank %zu exceeds supported %zu", count,
                  dims_capacity);
        return -1;
    }
    if (count > 0) {
        status = vad->api->GetDimensions(info, dims, count);
        if (status) {
            set_error_status(vad, status, "GetDimensions");
            return -1;
        }
    }

    *rank = count;
    return 0;
}

static af_vad_result inspect_model(af_vad *vad)
{
    OrtStatus *status;
    OrtTypeInfo *type_info = NULL;
    ONNXTensorElementDataType element_type;
    size_t rank;
    int64_t dims[AF_VAD_MAX_DIMS];
    size_t input_count = 0;
    size_t output_count = 0;
    size_t i;

    status = vad->api->SessionGetInputCount(vad->session, &input_count);
    if (status) {
        set_error_status(vad, status, "SessionGetInputCount");
        return AF_VAD_ERR_ORT;
    }
    status = vad->api->SessionGetOutputCount(vad->session, &output_count);
    if (status) {
        set_error_status(vad, status, "SessionGetOutputCount");
        return AF_VAD_ERR_ORT;
    }

    if (input_count != 3 || output_count != 2) {
        set_error("expected Silero VAD streaming model "
                  "(3 inputs, 2 outputs); got %zu, %zu",
                  input_count, output_count);
        return AF_VAD_ERR_MODEL_FORMAT;
    }

    if (check_io_name(vad, 0, 1, "input") != 0 ||
        check_io_name(vad, 1, 1, "state") != 0 ||
        check_io_name(vad, 2, 1, "sr") != 0 ||
        check_io_name(vad, 0, 0, "output") != 0 ||
        check_io_name(vad, 1, 0, "stateN") != 0) {
        return AF_VAD_ERR_MODEL_FORMAT;
    }

    /* input: float32 [1, 512] (second dimension may be dynamic). */
    status = vad->api->SessionGetInputTypeInfo(vad->session, 0, &type_info);
    if (status) {
        set_error_status(vad, status, "SessionGetInputTypeInfo(input)");
        return AF_VAD_ERR_ORT;
    }
    if (get_tensor_meta(vad, type_info, &element_type, &rank, dims,
                        AF_VAD_MAX_DIMS) != 0) {
        vad->api->ReleaseTypeInfo(type_info);
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->api->ReleaseTypeInfo(type_info);
    type_info = NULL;

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank != 2 ||
        (dims[0] != 1 && dims[0] != -1) ||
        (dims[1] != 576 && dims[1] != -1)) {
        set_error(
            "input must be float32 [1, 576] "
            "(64 context + 512 new samples)");
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->context_samples = 64;
    vad->model_input_samples = 576;
    vad->window_samples = 512;

    /* state: float32 [2, 1, 128]. */
    status = vad->api->SessionGetInputTypeInfo(vad->session, 1, &type_info);
    if (status) {
        set_error_status(vad, status, "SessionGetInputTypeInfo(state)");
        return AF_VAD_ERR_ORT;
    }
    if (get_tensor_meta(vad, type_info, &element_type, &rank, dims,
                        AF_VAD_MAX_DIMS) != 0) {
        vad->api->ReleaseTypeInfo(type_info);
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->api->ReleaseTypeInfo(type_info);
    type_info = NULL;

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank != 3 ||
        dims[0] != 2 || (dims[1] != 1 && dims[1] != -1) ||
        dims[2] != 128) {
        set_error("state must be float32 [2, 1, 128]");
        return AF_VAD_ERR_MODEL_FORMAT;
    }

    vad->state_rank = 3;
    vad->state_shape = (int64_t *)malloc(3 * sizeof(int64_t));
    if (!vad->state_shape) {
        set_error("out of memory allocating state shape");
        return AF_VAD_ERR_NO_MEMORY;
    }
    vad->state_shape[0] = 2;
    vad->state_shape[1] = 1;
    vad->state_shape[2] = 128;
    vad->state_elems = 2u * 1u * 128u;

    /* sr: int64 scalar or [1]. */
    status = vad->api->SessionGetInputTypeInfo(vad->session, 2, &type_info);
    if (status) {
        set_error_status(vad, status, "SessionGetInputTypeInfo(sr)");
        return AF_VAD_ERR_ORT;
    }
    if (get_tensor_meta(vad, type_info, &element_type, &rank, dims,
                        AF_VAD_MAX_DIMS) != 0) {
        vad->api->ReleaseTypeInfo(type_info);
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->api->ReleaseTypeInfo(type_info);
    type_info = NULL;

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || rank > 1 ||
        (rank == 1 && dims[0] != 1)) {
        set_error("sr must be an int64 scalar");
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->sr_rank = (int)rank;

    /* output: float32 [1, 1]. */
    status = vad->api->SessionGetOutputTypeInfo(vad->session, 0, &type_info);
    if (status) {
        set_error_status(vad, status, "SessionGetOutputTypeInfo(output)");
        return AF_VAD_ERR_ORT;
    }
    if (get_tensor_meta(vad, type_info, &element_type, &rank, dims,
                        AF_VAD_MAX_DIMS) != 0) {
        vad->api->ReleaseTypeInfo(type_info);
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->api->ReleaseTypeInfo(type_info);
    type_info = NULL;

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank < 1 ||
        rank > 2 ||
        (rank == 1 && dims[0] != 1 && dims[0] != -1) ||
        (rank == 2 && ((dims[0] != 1 && dims[0] != -1) ||
                       (dims[1] != 1 && dims[1] != -1)))) {
        set_error("output must be float32 [1, 1]");
        return AF_VAD_ERR_MODEL_FORMAT;
    }

    /* stateN: float32, same shape as state. */
    status = vad->api->SessionGetOutputTypeInfo(vad->session, 1, &type_info);
    if (status) {
        set_error_status(vad, status, "SessionGetOutputTypeInfo(stateN)");
        return AF_VAD_ERR_ORT;
    }
    if (get_tensor_meta(vad, type_info, &element_type, &rank, dims,
                        AF_VAD_MAX_DIMS) != 0) {
        vad->api->ReleaseTypeInfo(type_info);
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    vad->api->ReleaseTypeInfo(type_info);
    type_info = NULL;

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        rank != vad->state_rank) {
        set_error("stateN must match the state tensor");
        return AF_VAD_ERR_MODEL_FORMAT;
    }
    for (i = 0; i < rank; i++) {
        if (dims[i] != -1 && dims[i] != vad->state_shape[i]) {
            set_error("stateN must match the state tensor");
            return AF_VAD_ERR_MODEL_FORMAT;
        }
    }

    return AF_VAD_OK;
}

static af_vad_result create_tensors(af_vad *vad)
{
    OrtStatus *status;
    int64_t input_shape[2];
    int64_t sr_shape[1] = {1};
    const int64_t *sr_shape_ptr = vad->sr_rank == 1 ? sr_shape : NULL;

    vad->input_buf =
        (float *)calloc(vad->model_input_samples, sizeof(float));
    vad->pending_buf =
        (float *)calloc(vad->window_samples, sizeof(float));
    vad->context_buf =
        (float *)calloc(vad->context_samples, sizeof(float));
    vad->state_buf = (float *)calloc(vad->state_elems, sizeof(float));
    if (!vad->input_buf || !vad->pending_buf || !vad->context_buf ||
        !vad->state_buf) {
        set_error("out of memory allocating VAD buffers");
        return AF_VAD_ERR_NO_MEMORY;
    }

    input_shape[0] = 1;
    input_shape[1] = (int64_t)vad->model_input_samples;

    status = vad->api->CreateTensorWithDataAsOrtValue(
        vad->memory_info, vad->input_buf,
        vad->model_input_samples * sizeof(float), input_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vad->input_tensor);
    if (status) {
        set_error_status(vad, status, "CreateTensor(input)");
        return AF_VAD_ERR_ORT;
    }

    status = vad->api->CreateTensorWithDataAsOrtValue(
        vad->memory_info, vad->state_buf,
        vad->state_elems * sizeof(float), vad->state_shape,
        vad->state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &vad->state_tensor);
    if (status) {
        set_error_status(vad, status, "CreateTensor(state)");
        return AF_VAD_ERR_ORT;
    }

    status = vad->api->CreateTensorWithDataAsOrtValue(
        vad->memory_info, &vad->sr_value, sizeof(int64_t), sr_shape_ptr,
        (size_t)vad->sr_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &vad->sr_tensor);
    if (status) {
        set_error_status(vad, status, "CreateTensor(sr)");
        return AF_VAD_ERR_ORT;
    }

    return AF_VAD_OK;
}

static af_vad_result run_window(af_vad *vad)
{
    static const char *input_names[3] = {"input", "state", "sr"};
    static const char *output_names[2] = {"output", "stateN"};
    const OrtValue *inputs[3];
    OrtValue *outputs[2] = {NULL, NULL};
    OrtStatus *status;
    float *probability = NULL;
    float *next_state = NULL;

    inputs[0] = vad->input_tensor;
    inputs[1] = vad->state_tensor;
    inputs[2] = vad->sr_tensor;

    status = vad->api->Run(vad->session, NULL, input_names, inputs, 3,
                           output_names, 2, outputs);
    if (status) {
        if (outputs[0])
            vad->api->ReleaseValue(outputs[0]);
        if (outputs[1])
            vad->api->ReleaseValue(outputs[1]);
        set_error_status(vad, status, "OrtRun");
        return AF_VAD_ERR_ORT;
    }

    status = vad->api->GetTensorMutableData(outputs[0],
                                            (void **)&probability);
    if (status) {
        set_error_status(vad, status, "GetTensorMutableData(output)");
        if (outputs[0])
            vad->api->ReleaseValue(outputs[0]);
        if (outputs[1])
            vad->api->ReleaseValue(outputs[1]);
        return AF_VAD_ERR_ORT;
    }

    status = vad->api->GetTensorMutableData(outputs[1],
                                            (void **)&next_state);
    if (status) {
        set_error_status(vad, status, "GetTensorMutableData(stateN)");
        if (outputs[0])
            vad->api->ReleaseValue(outputs[0]);
        if (outputs[1])
            vad->api->ReleaseValue(outputs[1]);
        return AF_VAD_ERR_ORT;
    }

    vad->last_probability = probability[0];
    if (vad->last_probability < 0.0f)
        vad->last_probability = 0.0f;
    else if (vad->last_probability > 1.0f)
        vad->last_probability = 1.0f;
    vad->has_probability = 1;

    memcpy(vad->state_buf, next_state,
           vad->state_elems * sizeof(float));

    if (outputs[0])
        vad->api->ReleaseValue(outputs[0]);
    if (outputs[1])
        vad->api->ReleaseValue(outputs[1]);
    return AF_VAD_OK;
}

af_vad_result af_vad_create(const af_vad_config *config, af_vad **out_vad)
{
    af_vad *vad;
    af_vad_result result = AF_VAD_ERR_ORT;
    const OrtApiBase *api_base;
    OrtSessionOptions *options = NULL;
    OrtStatus *status;
    uint32_t sample_rate;
    int threads;

    if (!out_vad)
        return AF_VAD_ERR_INVALID_ARG;
    *out_vad = NULL;

    if (!config) {
        set_error("VAD config is required");
        return AF_VAD_ERR_INVALID_ARG;
    }
    if (!config->model_data && !config->model_path) {
        set_error("model_data or model_path is required");
        return AF_VAD_ERR_INVALID_ARG;
    }

    sample_rate = config->sample_rate ? config->sample_rate : 16000u;
    if (sample_rate != 16000u) {
        set_error("Silero VAD adapter supports 16000 Hz only");
        return AF_VAD_ERR_INVALID_ARG;
    }

    vad = (af_vad *)calloc(1, sizeof(*vad));
    if (!vad) {
        set_error("out of memory allocating VAD");
        return AF_VAD_ERR_NO_MEMORY;
    }

    api_base = OrtGetApiBase();
    if (!api_base) {
        set_error("OrtGetApiBase returned NULL");
        free(vad);
        return AF_VAD_ERR_ORT;
    }
    vad->api = api_base->GetApi(ORT_API_VERSION);
    if (!vad->api) {
        set_error("ONNX Runtime %s does not support API version %d",
                  api_base->GetVersionString(), ORT_API_VERSION);
        free(vad);
        return AF_VAD_ERR_ORT;
    }

    if (config->model_data) {
        if (config->model_size == 0 ||
            config->model_size > AF_VAD_MAX_MODEL_BYTES) {
            set_error("invalid embedded model size");
            af_vad_destroy(vad);
            return AF_VAD_ERR_INVALID_ARG;
        }
        vad->model_bytes = malloc(config->model_size);
        if (!vad->model_bytes) {
            set_error("out of memory copying model");
            af_vad_destroy(vad);
            return AF_VAD_ERR_NO_MEMORY;
        }
        memcpy(vad->model_bytes, config->model_data, config->model_size);
        vad->model_size = config->model_size;
    } else {
        result = read_model_file(config->model_path, &vad->model_bytes,
                                 &vad->model_size);
        if (result != AF_VAD_OK) {
            af_vad_destroy(vad);
            return result;
        }
    }

    status = vad->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "audioflow-vad",
                                 &vad->env);
    if (status) {
        set_error_status(vad, status, "CreateEnv");
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }

    status = vad->api->CreateSessionOptions(&options);
    if (status) {
        set_error_status(vad, status, "CreateSessionOptions");
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }

    threads = config->intra_op_threads
                  ? (int)config->intra_op_threads
                  : 1;
    status = vad->api->SetIntraOpNumThreads(options, threads);
    if (status) {
        set_error_status(vad, status, "SetIntraOpNumThreads");
        vad->api->ReleaseSessionOptions(options);
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }
    status = vad->api->SetInterOpNumThreads(options, 1);
    if (status) {
        set_error_status(vad, status, "SetInterOpNumThreads");
        vad->api->ReleaseSessionOptions(options);
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }
    status = vad->api->SetSessionGraphOptimizationLevel(options,
                                                        ORT_ENABLE_ALL);
    if (status) {
        set_error_status(vad, status, "SetSessionGraphOptimizationLevel");
        vad->api->ReleaseSessionOptions(options);
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }

    status = vad->api->CreateSessionFromArray(
        vad->env, vad->model_bytes, vad->model_size, options,
        &vad->session);
    vad->api->ReleaseSessionOptions(options);
    options = NULL;
    if (status) {
        set_error_status(vad, status, "CreateSessionFromArray");
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }

    result = inspect_model(vad);
    if (result != AF_VAD_OK) {
        af_vad_destroy(vad);
        return result;
    }

    status = vad->api->CreateCpuMemoryInfo(OrtArenaAllocator,
                                           OrtMemTypeDefault,
                                           &vad->memory_info);
    if (status) {
        set_error_status(vad, status, "CreateCpuMemoryInfo");
        af_vad_destroy(vad);
        return AF_VAD_ERR_ORT;
    }

    vad->sr_value = (int64_t)sample_rate;

    result = create_tensors(vad);
    if (result != AF_VAD_OK) {
        af_vad_destroy(vad);
        return result;
    }

    *out_vad = vad;
    return AF_VAD_OK;
}

void af_vad_destroy(af_vad *vad)
{
    if (!vad)
        return;

    if (vad->api) {
        if (vad->input_tensor)
            vad->api->ReleaseValue(vad->input_tensor);
        if (vad->state_tensor)
            vad->api->ReleaseValue(vad->state_tensor);
        if (vad->sr_tensor)
            vad->api->ReleaseValue(vad->sr_tensor);
        if (vad->session)
            vad->api->ReleaseSession(vad->session);
        if (vad->memory_info)
            vad->api->ReleaseMemoryInfo(vad->memory_info);
        if (vad->env)
            vad->api->ReleaseEnv(vad->env);
    }

    free(vad->model_bytes);
    free(vad->input_buf);
    free(vad->pending_buf);
    free(vad->context_buf);
    free(vad->state_buf);
    free(vad->state_shape);
    free(vad);
}

void af_vad_reset(af_vad *vad)
{
    if (!vad)
        return;
    if (vad->state_buf && vad->state_elems)
        memset(vad->state_buf, 0, vad->state_elems * sizeof(float));
    if (vad->context_buf && vad->context_samples)
        memset(vad->context_buf, 0,
               vad->context_samples * sizeof(float));
    if (vad->pending_buf && vad->window_samples)
        memset(vad->pending_buf, 0,
               vad->window_samples * sizeof(float));
    vad->fill = 0;
    vad->last_probability = 0.0f;
    vad->has_probability = 0;
}

uint32_t af_vad_window_samples(const af_vad *vad)
{
    if (!vad)
        return 0;
    return (uint32_t)vad->window_samples;
}

af_vad_result af_vad_process(af_vad *vad, const int16_t *samples,
                             size_t count, float *probability, int *updated)
{
    size_t i;
    int ran = 0;

    if (!vad || !probability || (count > 0 && !samples))
        return AF_VAD_ERR_INVALID_ARG;
    if (!vad->input_buf || !vad->pending_buf || !vad->context_buf ||
        !vad->window_samples || !vad->context_samples)
        return AF_VAD_ERR_STATE;

    for (i = 0; i < count; i++) {
        af_vad_result result;

        vad->pending_buf[vad->fill++] = (float)samples[i] / 32768.0f;
        if (vad->fill == vad->window_samples) {
            memcpy(vad->input_buf, vad->context_buf,
                   vad->context_samples * sizeof(float));
            memcpy(vad->input_buf + vad->context_samples,
                   vad->pending_buf,
                   vad->window_samples * sizeof(float));

            result = run_window(vad);
            if (result != AF_VAD_OK) {
                vad->fill = 0;
                return result;
            }

            memcpy(vad->context_buf,
                   vad->input_buf +
                       (vad->model_input_samples -
                        vad->context_samples),
                   vad->context_samples * sizeof(float));
            vad->fill = 0;
            ran = 1;
        }
    }

    *probability = vad->has_probability ? vad->last_probability : 0.0f;
    if (updated)
        *updated = ran;
    return AF_VAD_OK;
}
