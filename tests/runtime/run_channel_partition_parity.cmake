# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED RUNTIME_TEST OR NOT DEFINED PARTITION_SELECTOR OR NOT DEFINED SEMANTIC_FILTERS)
    message(FATAL_ERROR "runtime test, partition selector, and semantic filters are required")
endif()

string(REPLACE "|" ";" semantic_filters "${SEMANTIC_FILTERS}")
foreach(semantic_filter IN LISTS semantic_filters)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
            "${PARTITION_SELECTOR}"
            "${RUNTIME_TEST}"
            "--gtest_filter=ChannelPartitionPlanParityTest.AcceptedCoverFacts:${semantic_filter}"
        RESULT_VARIABLE semantic_result
        OUTPUT_VARIABLE semantic_output
        ERROR_VARIABLE semantic_error
    )
    if(NOT semantic_result EQUAL 0)
        message(STATUS "${semantic_output}")
        message(STATUS "${semantic_error}")
        message(FATAL_ERROR "partition parity semantic failed: ${semantic_filter}")
    endif()
endforeach()
