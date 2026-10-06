# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(holoscan_text_layer_example LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(text_layer_example
    text_layer_example.cpp
    text_source.cu
)
target_compile_features(text_layer_example PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(text_layer_example
    PRIVATE
        CUDA::cudart
        holoscan::core
        holoscan::sensor_io
        holoscan::visualization::op_holoviz
        Threads::Threads
)
holoviz_copy_runtime_dlls(text_layer_example)
holoscan_configure_example_install(text_layer_example
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/text_layer_example/cpp"
)

# The rendering run opens a HoloViz window and needs a display, so it has no CTest entry. Argument
# validation happens before any CUDA, graph, or window work, so malformed inputs are checked on any
# host: each case must be rejected with a diagnostic naming the option.
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    # option:value pairs; ':' keeps each pair one list element (';' would flatten the list).
    set(_text_layer_bad_args
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
    set(_text_layer_case 0)
    foreach(_text_layer_bad IN LISTS _text_layer_bad_args)
        string(REPLACE ":" ";" _text_layer_pair "${_text_layer_bad}")
        list(GET _text_layer_pair 0 _text_layer_option)
        list(GET _text_layer_pair 1 _text_layer_value)
        math(EXPR _text_layer_case "${_text_layer_case} + 1")
        set(_text_layer_test EXAMPLE_TEXT_LAYER_ARGS.rejects_${_text_layer_option}_${_text_layer_case})
        add_test(NAME ${_text_layer_test}
            COMMAND text_layer_example --${_text_layer_option} ${_text_layer_value})
        set_tests_properties(${_text_layer_test}
            PROPERTIES
                PASS_REGULAR_EXPRESSION "invalid --${_text_layer_option} value: '${_text_layer_value}'"
                LABELS "examples;visualization"
                TIMEOUT 30)
    endforeach()
    unset(_text_layer_bad_args)
    unset(_text_layer_bad)
    unset(_text_layer_pair)
    unset(_text_layer_case)
    unset(_text_layer_option)
    unset(_text_layer_value)
    unset(_text_layer_test)
endif()

install(TARGETS text_layer_example
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/text_layer_example/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)
