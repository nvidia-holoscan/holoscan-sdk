# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_TEST_BUILD_DIR OR HOLOSCAN_TEST_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_BUILD_DIR is required")
endif()

foreach(rejected_target IN ITEMS
        third_party_tensor_schema_ordinary_traits_rejected
        third_party_selected_tensor_schema_ordinary_traits_rejected)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${HOLOSCAN_TEST_BUILD_DIR}"
            --target "${rejected_target}"
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_stdout
        ERROR_VARIABLE build_stderr
    )
    if(build_result EQUAL 0)
        message(FATAL_ERROR
            "Ordinary traits unexpectedly accepted Tensor-bearing target ${rejected_target}"
        )
    endif()
    set(build_output "${build_stdout}\n${build_stderr}")
    if(NOT build_output MATCHES "HOLOSCAN_SCHEMA_TENSOR_REQUIRES_DESCRIPTOR_TRAITS")
        message(FATAL_ERROR
            "Tensor-bearing target ${rejected_target} failed without the expected diagnostic:"
            "\n${build_output}"
        )
    endif()
endforeach()
