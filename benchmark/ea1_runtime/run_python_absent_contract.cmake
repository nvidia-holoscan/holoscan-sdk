# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_EA1_RUNTIME_SOURCE_DIR OR
   NOT DEFINED HOLOSCAN_EA1_RUNTIME_BINARY_DIR OR
   NOT DEFINED HOLOSCAN_GENERATOR)
    message(FATAL_ERROR "Python-absent benchmark contract inputs are incomplete")
endif()

file(REMOVE_RECURSE "${HOLOSCAN_EA1_RUNTIME_BINARY_DIR}")
set(_configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_EA1_RUNTIME_SOURCE_DIR}/python_absent_contract"
    -B "${HOLOSCAN_EA1_RUNTIME_BINARY_DIR}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DHOLOSCAN_EA1_RUNTIME_SOURCE_DIR=${HOLOSCAN_EA1_RUNTIME_SOURCE_DIR}")
if(HOLOSCAN_GENERATOR_PLATFORM)
    list(APPEND _configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(HOLOSCAN_GENERATOR_TOOLSET)
    list(APPEND _configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(HOLOSCAN_MAKE_PROGRAM)
    list(APPEND _configure_command "-DCMAKE_MAKE_PROGRAM=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(HOLOSCAN_CXX_COMPILER)
    list(APPEND _configure_command "-DCMAKE_CXX_COMPILER=${HOLOSCAN_CXX_COMPILER}")
endif()

execute_process(
    COMMAND ${_configure_command}
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_stdout
    ERROR_VARIABLE _configure_stderr)
if(NOT _configure_result EQUAL 0)
    message(FATAL_ERROR
        "Python-absent benchmark contract configure failed (${_configure_result})\n"
        "stdout:\n${_configure_stdout}\n"
        "stderr:\n${_configure_stderr}")
endif()
