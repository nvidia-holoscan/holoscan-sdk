# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Python port imports the built bindings: Core's `holoscan` package and the visualization
# provider `holoscan_visualization` beside it under HOLOSCAN_PYTHON_MODULE_ROOT. The SDK build sets
# that root before adding this module and builds the provider extension afterwards (it depends on
# the module's operator target), so the root is the right gate here. A standalone module build or
# HOLOSCAN_BUILD_PYTHON=OFF leaves it undefined and this directory is a no-op.
if(NOT HOLOSCAN_BUILD_PYTHON OR NOT DEFINED HOLOSCAN_PYTHON_MODULE_ROOT)
    return()
endif()

# The rendering run opens a HoloViz window and needs a display, so the shared helper's test is the
# headless --help path instead: it exits before any graph or window work, so it is an import and
# argument-parse smoke test. Depending on the provider target keeps a focused example build
# complete.
holoscan_add_python_example(
    NAME gray_scale_example
    SCRIPT gray_scale_example.py
    TEST_NAME EXAMPLE_GRAY_SCALE_PYTHON_ARGS.help
    ARGS --help
    LABELS examples visualization python
    INSTALL_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/gray_scale_example/python"
    INSTALL_COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
    DEPENDS holoscan_visualization_python
)

# Argument validation happens before any graph or window work, so malformed inputs are checked on
# any host with the same diagnostics as the C++ binary (minus --device, which the host-tensor port
# does not take).
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    set_tests_properties(EXAMPLE_GRAY_SCALE_PYTHON_ARGS.help
        PROPERTIES PASS_REGULAR_EXPRESSION "--fullscreen")

    set(_gray_scale_py "${CMAKE_CURRENT_SOURCE_DIR}/gray_scale_example.py")
    set(_gray_scale_py_environment
        "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
        "PYTHONUNBUFFERED=1"
    )
    # option:value pairs; ':' keeps each pair one list element (';' would flatten the list).
    set(_gray_scale_py_bad_args
        seconds:abc
        seconds:5junk
        seconds:-1
        seconds:99999999999999999999
        hz:abc
        hz:-1
        hz:99999999999
        fps:abc
        fps:0
        fps:1001
    )
    set(_gray_scale_py_case 0)
    foreach(_gray_scale_py_bad IN LISTS _gray_scale_py_bad_args)
        string(REPLACE ":" ";" _gray_scale_py_pair "${_gray_scale_py_bad}")
        list(GET _gray_scale_py_pair 0 _gray_scale_py_option)
        list(GET _gray_scale_py_pair 1 _gray_scale_py_value)
        math(EXPR _gray_scale_py_case "${_gray_scale_py_case} + 1")
        set(_gray_scale_py_test
            EXAMPLE_GRAY_SCALE_PYTHON_ARGS.rejects_${_gray_scale_py_option}_${_gray_scale_py_case})
        add_test(NAME ${_gray_scale_py_test}
            COMMAND "${Python3_EXECUTABLE}" "${_gray_scale_py}"
                --${_gray_scale_py_option} ${_gray_scale_py_value})
        set_tests_properties(${_gray_scale_py_test}
            PROPERTIES
                ENVIRONMENT "${_gray_scale_py_environment}"
                PASS_REGULAR_EXPRESSION
                    "invalid --${_gray_scale_py_option} value: '${_gray_scale_py_value}'"
                LABELS "examples;visualization;python"
                TIMEOUT 30)
    endforeach()

    unset(_gray_scale_py)
    unset(_gray_scale_py_environment)
    unset(_gray_scale_py_bad_args)
    unset(_gray_scale_py_bad)
    unset(_gray_scale_py_pair)
    unset(_gray_scale_py_case)
    unset(_gray_scale_py_option)
    unset(_gray_scale_py_value)
    unset(_gray_scale_py_test)
endif()
