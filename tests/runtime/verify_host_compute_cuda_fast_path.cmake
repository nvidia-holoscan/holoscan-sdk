# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# REQ: CHAN-103
if(NOT DEFINED RUN_SESSION_SOURCE OR NOT DEFINED GRAPH_EXEC_HEADER)
    message(FATAL_ERROR "run-session source and GraphExec header paths are required")
endif()

file(READ "${RUN_SESSION_SOURCE}" run_session_source)
file(READ "${GRAPH_EXEC_HEADER}" graph_exec_header)
string(REGEX REPLACE "[ \t\r\n]+" " " normalized_run_session_source "${run_session_source}")
string(REGEX REPLACE "[ \t\r\n]+" " " normalized_graph_exec_header "${graph_exec_header}")

set(host_fast_path
    "if (!requested.valid()) { active_ = true; restored_ = true; return; } if (cudaGetDevice(&previous_) != cudaSuccess)"
)
string(FIND "${normalized_run_session_source}" "${host_fast_path}" host_fast_path_offset)
if(host_fast_path_offset EQUAL -1)
    message(FATAL_ERROR
        "host-only compute selection must return before the first CUDA device query")
endif()

string(FIND "${normalized_run_session_source}"
    ".cuda_device = instance.compute_device," cold_binding_offset)
if(cold_binding_offset EQUAL -1)
    message(FATAL_ERROR
        "compute dispatch must consume the cold-bound RuntimeOperatorInstance device")
endif()

string(FIND "${normalized_graph_exec_header}" "DeviceId compute_device{};" invalid_default_offset)
if(invalid_default_offset EQUAL -1)
    message(FATAL_ERROR
        "RuntimeOperatorInstance host-only compute device must remain invalid by default")
endif()
