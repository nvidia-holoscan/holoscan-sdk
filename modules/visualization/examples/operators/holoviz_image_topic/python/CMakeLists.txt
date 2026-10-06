# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Python-only example: it needs the holoscan_visualization provider package rather than a native
# target of its own. That extension is configured later than this directory, so gate on the option
# rather than on the target existing yet.
if(NOT HOLOSCAN_BUILD_PYTHON)
    return()
endif()

set(_holoviz_image_topic_script "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_image_topic.py")

install(FILES "${_holoviz_image_topic_script}"
    DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/holoviz_image_topic/python"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    # Lifecycle coverage only: this starts the operator, runs briefly and stops. Nothing publishes
    # to the topic, so it asserts nothing about rendered output or window-event handling.
    #
    # The example opens a window, so it reports 77 when no display is reachable and CTest records
    # the run as skipped rather than failed.
    add_test(NAME EXAMPLE_HOLOVIZ_IMAGE_TOPIC.starts_and_stops_cleanly
        COMMAND "${Python3_EXECUTABLE}" "${_holoviz_image_topic_script}" --seconds 1)
    set_tests_properties(EXAMPLE_HOLOVIZ_IMAGE_TOPIC.starts_and_stops_cleanly
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
            SKIP_RETURN_CODE 77
            LABELS "examples;visualization;python"
            TIMEOUT 120)

    add_test(NAME EXAMPLE_HOLOVIZ_IMAGE_TOPIC.rejects_seconds
        COMMAND "${Python3_EXECUTABLE}" "${_holoviz_image_topic_script}" --seconds 0)
    set_tests_properties(EXAMPLE_HOLOVIZ_IMAGE_TOPIC.rejects_seconds
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
            PASS_REGULAR_EXPRESSION "invalid --seconds value: '0'"
            LABELS "examples;visualization;python"
            TIMEOUT 60)
endif()

unset(_holoviz_image_topic_script)
