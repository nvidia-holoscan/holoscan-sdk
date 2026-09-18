# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_gray_scale_example LANGUAGES CXX CUDA)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

# The grayscale source is a CUDA kernel; the module project itself is CXX-only.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

add_executable(gray_scale_example
    gray_scale_example.cpp
    gray_scale_source.cu
)
target_compile_features(gray_scale_example PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(gray_scale_example
    PRIVATE
        CUDA::cudart
        holoscan::core
        holoscan::visualization::op_holoviz
        Threads::Threads
)
holoviz_copy_runtime_dlls(gray_scale_example)
holoscan_configure_example_install(gray_scale_example
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/gray_scale_example/cpp"
)

# The rendering run opens a HoloViz window and needs a display, so it has no CTest entry. Argument
# validation happens before any CUDA, graph, or window work, so malformed inputs are checked on any
# host: each case must be rejected with a diagnostic naming the option.
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    # option:value pairs; ':' keeps each pair one list element (';' would flatten the list).
    set(_gray_scale_bad_args
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
    set(_gray_scale_case 0)
    foreach(_gray_scale_bad IN LISTS _gray_scale_bad_args)
        string(REPLACE ":" ";" _gray_scale_pair "${_gray_scale_bad}")
        list(GET _gray_scale_pair 0 _gray_scale_option)
        list(GET _gray_scale_pair 1 _gray_scale_value)
        math(EXPR _gray_scale_case "${_gray_scale_case} + 1")
        set(_gray_scale_test EXAMPLE_GRAY_SCALE_ARGS.rejects_${_gray_scale_option}_${_gray_scale_case})
        add_test(NAME ${_gray_scale_test}
            COMMAND gray_scale_example --${_gray_scale_option} ${_gray_scale_value})
        set_tests_properties(${_gray_scale_test}
            PROPERTIES
                PASS_REGULAR_EXPRESSION "invalid --${_gray_scale_option} value: '${_gray_scale_value}'"
                LABELS "holoviz;visualization;example")
    endforeach()
    unset(_gray_scale_bad_args)
    unset(_gray_scale_bad)
    unset(_gray_scale_pair)
    unset(_gray_scale_case)
    unset(_gray_scale_option)
    unset(_gray_scale_value)
    unset(_gray_scale_test)
endif()

install(TARGETS gray_scale_example
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/gray_scale_example/cpp"
    COMPONENT holoscan-examples
)
