# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# The pattern source is a CUDA kernel; the module project itself is CXX-only.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(holoscan_cuda_sink_example LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(cuda_sink_example
    cuda_sink_example.cpp
    pattern_source.cu
)
target_compile_features(cuda_sink_example PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(cuda_sink_example
    PRIVATE
        CUDA::cudart
        holoscan::core
        holoscan::sensor_io
        holoscan::visualization::holoviz
        holoscan::visualization::op_holoviz
        Threads::Threads
)
holoviz_copy_runtime_dlls(cuda_sink_example)
holoscan_configure_example_install(cuda_sink_example
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/cuda_sink_example/cpp"
)

# The rendering run opens a HoloViz window (or takes a display) and needs one, so it has no CTest
# entry. Argument validation happens before any CUDA, graph, or window work, so malformed inputs
# are checked on any host: each case must be rejected with a diagnostic naming the option.
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    # option:value pairs; ':' keeps each pair one list element (';' would flatten the list).
    set(_cuda_sink_bad_args
        seconds:abc
        seconds:5junk
        seconds:-1
        seconds:99999999999999999999
        device:abc
        device:-1
        device:99999999999
        hz:abc
        hz:-1
        hz:99999999999
        fps:abc
        fps:0
        fps:1001
    )
    set(_cuda_sink_case 0)
    foreach(_cuda_sink_bad IN LISTS _cuda_sink_bad_args)
        string(REPLACE ":" ";" _cuda_sink_pair "${_cuda_sink_bad}")
        list(GET _cuda_sink_pair 0 _cuda_sink_option)
        list(GET _cuda_sink_pair 1 _cuda_sink_value)
        math(EXPR _cuda_sink_case "${_cuda_sink_case} + 1")
        set(_cuda_sink_test EXAMPLE_CUDA_SINK_ARGS.rejects_${_cuda_sink_option}_${_cuda_sink_case})
        add_test(NAME ${_cuda_sink_test}
            COMMAND cuda_sink_example --${_cuda_sink_option} ${_cuda_sink_value})
        set_tests_properties(${_cuda_sink_test}
            PROPERTIES
                PASS_REGULAR_EXPRESSION "invalid --${_cuda_sink_option} value: '${_cuda_sink_value}'"
                LABELS "examples;visualization"
                TIMEOUT 30)
    endforeach()
    unset(_cuda_sink_bad_args)
    unset(_cuda_sink_bad)
    unset(_cuda_sink_pair)
    unset(_cuda_sink_case)
    unset(_cuda_sink_option)
    unset(_cuda_sink_value)
    unset(_cuda_sink_test)
endif()

install(TARGETS cuda_sink_example
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/cuda_sink_example/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)
