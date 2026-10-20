# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_SOURCE_DIR
        HOLOSCAN_SUBPROJECT_SOURCE_DIR
        HOLOSCAN_SUBPROJECT_WORK_DIR
        HOLOSCAN_VERIFY_INSTALL_SOURCES_SCRIPT)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO STDOUT)

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${HOLOSCAN_SUBPROJECT_SOURCE_DIR}"
        -B "${HOLOSCAN_SUBPROJECT_WORK_DIR}"
        "-DHOLOSCAN_SOURCE_DIR=${HOLOSCAN_SOURCE_DIR}"
    RESULT_VARIABLE holoscan_subproject_configure_result
)
if(NOT holoscan_subproject_configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure Holoscan as a subproject")
endif()

set(HOLOSCAN_INSTALL_SCRIPT "${HOLOSCAN_SUBPROJECT_WORK_DIR}/holoscan-sdk/cmake_install.cmake")
include("${HOLOSCAN_VERIFY_INSTALL_SOURCES_SCRIPT}")
