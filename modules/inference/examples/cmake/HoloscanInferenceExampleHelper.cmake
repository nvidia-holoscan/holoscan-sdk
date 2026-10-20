# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holoscan_inference_add_example_test config_file)
    get_filename_component(config_name "${config_file}" NAME_WE)
    set(test_name "HoloscanInferenceExample.${config_name}")

    add_test(NAME "${test_name}"
        COMMAND $<TARGET_FILE:holoinfer_example>
            --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_file}"
            --ticks 2
    )
    set_tests_properties("${test_name}"
        PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            LABELS "examples;inference"
            RESOURCE_LOCK "holoinfer-example-gpu"
            TIMEOUT 120
    )
endfunction()
