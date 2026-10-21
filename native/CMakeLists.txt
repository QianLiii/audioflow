cmake_minimum_required(VERSION 3.20)
project(audioflow_native LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(AUDIOFLOW_BUILD_TESTS "Build portable unit tests" ON)
option(AUDIOFLOW_BUILD_EXAMPLES "Build the capture and VAD examples" ON)

# Silero VAD is mandatory.
set(ORT_ROOT "" CACHE PATH "ONNX Runtime installation root")

find_path(ORT_INCLUDE_DIR onnxruntime_c_api.h
    HINTS ${ORT_ROOT}/include
)
find_library(ORT_LIBRARY NAMES onnxruntime
    HINTS ${ORT_ROOT}/lib
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR
        "Silero VAD requires ONNX Runtime. "
        "Set ORT_ROOT to the extracted ONNX Runtime package.")
endif()

add_library(audioflow
    src/af_ring.c
    src/af_resampler.c
    src/af_chunker.c
    src/vad/af_vad_silero.c
)

target_include_directories(audioflow
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${ORT_INCLUDE_DIR}
)
target_compile_features(audioflow PUBLIC c_std_11)
target_link_libraries(audioflow PRIVATE ${ORT_LIBRARY})

if(WIN32)
    target_sources(audioflow PRIVATE src/wasapi_capture.c)
    target_compile_definitions(audioflow PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(audioflow PRIVATE ole32 avrt)
    set_target_properties(audioflow PROPERTIES PREFIX "")
else()
    find_package(Threads REQUIRED)
    target_compile_definitions(audioflow PUBLIC _POSIX_C_SOURCE=200809L)
    target_link_libraries(audioflow PUBLIC Threads::Threads m)
    if(UNIX)
        get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
        set_target_properties(audioflow PROPERTIES
            BUILD_RPATH "${ORT_LIBRARY_DIR}"
        )
    endif()
endif()

if(BUILD_SHARED_LIBS)
    target_compile_definitions(audioflow PRIVATE
        AF_BUILD_SHARED
        AF_BUILDING_LIBRARY
    )
    target_compile_definitions(audioflow INTERFACE
        AF_BUILD_SHARED
    )
endif()

if(MINGW AND BUILD_SHARED_LIBS)
    # Keep the ABI to explicitly exported AF_API symbols only.
    target_link_options(audioflow PRIVATE -Wl,--exclude-all-symbols)
endif()

if(MSVC)
    target_compile_options(audioflow PRIVATE /W4)
else()
    target_compile_options(audioflow PRIVATE -Wall -Wextra)
endif()

if(WIN32)
    find_file(ORT_RUNTIME_DLL NAMES onnxruntime.dll
        HINTS ${ORT_ROOT}/lib ${ORT_ROOT}/bin
    )
endif()

function(audioflow_copy_runtime target)
    if(WIN32 AND ORT_RUNTIME_DLL)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ORT_RUNTIME_DLL} $<TARGET_FILE_DIR:${target}>
        )
    endif()
endfunction()

if(AUDIOFLOW_BUILD_TESTS)
    enable_testing()

    # Internal portable modules, linked by tests without exporting them from
    # the public DLL or requiring ONNX Runtime.
    add_library(audioflow_internal STATIC
        src/af_ring.c
        src/af_resampler.c
        src/af_chunker.c
    )
    target_include_directories(audioflow_internal PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(audioflow_internal PUBLIC c_std_11)
    if(WIN32)
        target_compile_definitions(audioflow_internal PRIVATE _WIN32_WINNT=0x0A00)
    else()
        target_compile_definitions(audioflow_internal PUBLIC _POSIX_C_SOURCE=200809L)
        target_link_libraries(audioflow_internal PUBLIC Threads::Threads m)
    endif()
    if(MSVC)
        target_compile_options(audioflow_internal PRIVATE /W4)
    else()
        target_compile_options(audioflow_internal PRIVATE -Wall -Wextra)
    endif()

    add_executable(test_ring tests/test_ring.c)
    target_link_libraries(test_ring PRIVATE audioflow_internal)
    add_test(NAME ring COMMAND test_ring)

    add_executable(test_resampler tests/test_resampler.c)
    target_link_libraries(test_resampler PRIVATE audioflow_internal)
    add_test(NAME resampler COMMAND test_resampler)

    add_executable(test_chunker tests/test_chunker.c)
    target_link_libraries(test_chunker PRIVATE audioflow_internal)
    add_test(NAME chunker COMMAND test_chunker)
endif()

if(AUDIOFLOW_BUILD_EXAMPLES)
    add_executable(vad_probe examples/vad_probe.c)
    target_link_libraries(vad_probe PRIVATE audioflow)
    audioflow_copy_runtime(vad_probe)

    if(WIN32)
        add_executable(capture_dump examples/capture_dump.c)
        target_link_libraries(capture_dump PRIVATE audioflow)
        audioflow_copy_runtime(capture_dump)

        add_executable(capture_wav examples/capture_wav.c)
        target_link_libraries(capture_wav PRIVATE audioflow)
        audioflow_copy_runtime(capture_wav)
    endif()
endif()
