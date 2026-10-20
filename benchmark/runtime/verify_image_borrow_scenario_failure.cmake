# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RUNTIME_GRAPH_BENCHMARK OR
   HOLOSCAN_RUNTIME_GRAPH_BENCHMARK STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RUNTIME_GRAPH_BENCHMARK is required")
endif()

foreach(scenario IN ITEMS pod tensor)
    execute_process(
        COMMAND "${HOLOSCAN_RUNTIME_GRAPH_BENCHMARK}"
                --borrow-image-input --scenario "${scenario}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "Image borrowing was unexpectedly accepted for ${scenario}")
    endif()
    string(FIND "${stdout}\n${stderr}" "--borrow-image-input requires --scenario image" offset)
    if(offset EQUAL -1)
        message(FATAL_ERROR "Missing Image borrowing diagnostic for ${scenario}:\n${stdout}\n${stderr}")
    endif()
endforeach()
