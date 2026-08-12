# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE OR
   HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE is required")
endif()

execute_process(
    COMMAND "${HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE}"
    RESULT_VARIABLE host_result
    OUTPUT_VARIABLE host_output
    ERROR_VARIABLE host_error
    TIMEOUT 20
)
string(CONCAT host_log "${host_output}" "${host_error}")
string(CONCAT expected_run_one
    "run=1 pre_overlay_activations=0 selected=[1-9][0-9]* reused=[1-9][0-9]* "
    "tensor_eos=closed control_eos=closed reset=clean")
string(CONCAT expected_run_two
    "run=2 pre_overlay_activations=0 selected=[1-9][0-9]* reused=[1-9][0-9]* "
    "tensor_eos=closed control_eos=closed reset=clean")
if(NOT host_result EQUAL 0 OR
   NOT host_log MATCHES "${expected_run_one}" OR
   NOT host_log MATCHES "${expected_run_two}" OR
   NOT host_log MATCHES
       "route=holomq://retained-tensor-compositor\\.overlay-[^ \r\n]+ frame_memory=host" OR
   NOT host_log MATCHES "retained Tensor compositor complete")
    message(FATAL_ERROR
        "Retained-Tensor README host command or expected output failed: ${host_log}")
endif()
message(STATUS "${host_log}")

function(make_route_suffix output_variable nonce)
    string(TIMESTAMP route_seed_time "%s%f" UTC)
    string(SHA256 route_seed_digest
        "${CMAKE_CURRENT_LIST_FILE}|${CMAKE_CURRENT_BINARY_DIR}|${route_seed_time}|${nonce}")
    string(SUBSTRING "${route_seed_digest}" 0 7 route_seed_hex)
    math(EXPR route_seed "0x${route_seed_hex}")
    string(RANDOM
        LENGTH 16
        ALPHABET 0123456789abcdef
        RANDOM_SEED "${route_seed}"
        generated_suffix
    )
    set("${output_variable}" "${generated_suffix}" PARENT_SCOPE)
endfunction()

make_route_suffix(route_suffix "authored-route")
make_route_suffix(collision_probe_suffix "collision-probe")
if(route_suffix STREQUAL collision_probe_suffix)
    message(FATAL_ERROR "Path- and time-seeded route suffixes collided")
endif()
set(authored_route "holomq://my-calibration-service-${route_suffix}")
execute_process(
    COMMAND
        "${HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE}"
        --route
        "${authored_route}"
    RESULT_VARIABLE route_result
    OUTPUT_VARIABLE route_output
    ERROR_VARIABLE route_error
    TIMEOUT 20
)
string(CONCAT route_log "${route_output}" "${route_error}")
if(NOT route_result EQUAL 0 OR
   NOT route_log MATCHES "route=${authored_route} frame_memory=host" OR
   NOT route_log MATCHES "retained Tensor compositor complete")
    message(FATAL_ERROR
        "Retained-Tensor README authored-route command or expected output failed: ${route_log}")
endif()
message(STATUS "${route_log}")
message(STATUS "retained Tensor compositor README host and authored-route modes passed")
