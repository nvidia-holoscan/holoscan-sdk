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
    # Direct tutorial smoke: wait on the session, request stop, and inspect final diagnostics.
    # This does not confirm global START before the interval or exercise rendered output.
    #
    # The example opens a window. A retained missing-route diagnostic yields 77, which CTest
    # records as skipped.
    add_test(NAME EXAMPLE_HOLOVIZ_IMAGE_TOPIC.starts_and_stops_cleanly
        COMMAND "${Python3_EXECUTABLE}" "${_holoviz_image_topic_script}" --seconds 1)
    set_tests_properties(EXAMPLE_HOLOVIZ_IMAGE_TOPIC.starts_and_stops_cleanly
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
            SKIP_RETURN_CODE 77
            LABELS "examples;visualization;python;display"
            # The strict driver imports the same camera service.
            RESOURCE_LOCK "holoviz-image-topic-service"
            TIMEOUT 120)
endif()

unset(_holoviz_image_topic_script)
