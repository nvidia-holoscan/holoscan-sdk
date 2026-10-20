# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required IN ITEMS V1_CODEC V2_CODEC WORK_DIR)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(v1_buffer "${WORK_DIR}/v1.bin")
set(v2_buffer "${WORK_DIR}/v2.bin")

function(run_codec label)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${label} failed with status ${result}")
    endif()
endfunction()

run_codec("V1 write" "${V1_CODEC}" write "${v1_buffer}")
run_codec("V2 read of V1" "${V2_CODEC}" read "${v1_buffer}" v1)
run_codec("V2 write" "${V2_CODEC}" write "${v2_buffer}")
run_codec("V1 read of V2" "${V1_CODEC}" read "${v2_buffer}" v2)
