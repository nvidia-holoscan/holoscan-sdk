# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required IN ITEMS FLATC SOURCE_DIR)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

set(baseline "${SOURCE_DIR}/evolution_v1.fbs")

function(expect_conform candidate)
    execute_process(
        COMMAND "${FLATC}" --require-explicit-ids --conform "${baseline}"
                "${SOURCE_DIR}/${candidate}.fbs"
        RESULT_VARIABLE result
        ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${candidate} should conform but failed: ${error}")
    endif()
endfunction()

function(expect_nonconform candidate)
    execute_process(
        COMMAND "${FLATC}" --require-explicit-ids --conform "${baseline}"
                "${SOURCE_DIR}/${candidate}.fbs"
        RESULT_VARIABLE result
        ERROR_VARIABLE error)
    if(result EQUAL 0)
        message(FATAL_ERROR "${candidate} unexpectedly conforms")
    endif()
    if(NOT error MATCHES "schemas don't conform")
        message(FATAL_ERROR "${candidate} failed for the wrong reason: ${error}")
    endif()
endfunction()

expect_conform(evolution_v2)
expect_conform(evolution_renamed)
expect_nonconform(incompatible_removed_reused_id)
expect_nonconform(incompatible_changed_default)
expect_nonconform(incompatible_field_type)
expect_nonconform(incompatible_struct)
expect_nonconform(incompatible_union_discriminant)

# FlatBuffers 25.12.19 does not compare the required attribute in ConformTo.
# schema_evolution_characterization applies the dedicated Holoscan policy check.
