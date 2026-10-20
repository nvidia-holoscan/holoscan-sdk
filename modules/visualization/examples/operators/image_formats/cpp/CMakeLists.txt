# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_image_formats LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(image_formats
    image_formats.cpp
)
target_compile_features(image_formats PRIVATE cxx_std_20)
target_link_libraries(image_formats
    PRIVATE
        CUDA::cudart
        holoscan::core
        holoscan::visualization::op_holoviz
        Threads::Threads
)
holoviz_copy_runtime_dlls(image_formats)
holoscan_configure_example_install(image_formats
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/image_formats/cpp"
)

# The rendering run opens a HoloViz window and needs a display, so it has no CTest entry. Argument
# validation happens before any CUDA, graph, or window work, so malformed inputs are checked on any
# host: each case must be rejected with a diagnostic naming the option.
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    # option:value pairs; ':' keeps each pair one list element (';' would flatten the list).
    set(_image_formats_bad_args
        seconds:abc
        seconds:5junk
        seconds:-1
        seconds:99999999999999999999
        device:abc
        device:-1
        device:99999999999
        format:bogus
        memory:bogus
    )
    set(_image_formats_case 0)
    foreach(_image_formats_bad IN LISTS _image_formats_bad_args)
        string(REPLACE ":" ";" _image_formats_pair "${_image_formats_bad}")
        list(GET _image_formats_pair 0 _image_formats_option)
        list(GET _image_formats_pair 1 _image_formats_value)
        math(EXPR _image_formats_case "${_image_formats_case} + 1")
        set(_image_formats_test
            EXAMPLE_IMAGE_FORMATS_ARGS.rejects_${_image_formats_option}_${_image_formats_case})
        add_test(NAME ${_image_formats_test}
            COMMAND image_formats --${_image_formats_option} ${_image_formats_value})
        set_tests_properties(${_image_formats_test}
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "invalid --${_image_formats_option} value: '${_image_formats_value}'"
                LABELS "examples;visualization"
                TIMEOUT 30)
    endforeach()
    unset(_image_formats_bad_args)
    unset(_image_formats_bad)
    unset(_image_formats_pair)
    unset(_image_formats_case)
    unset(_image_formats_option)
    unset(_image_formats_value)
    unset(_image_formats_test)
endif()

install(TARGETS image_formats
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/image_formats/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)
