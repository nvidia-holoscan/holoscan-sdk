# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_CONSUMER_SOURCE_DIR
        HOLOSCAN_CONSUMER_WORK_DIR
        HOLOSCAN_GENERATOR
        HOLOSCAN_CUDA_TOOLKIT_ROOT
        HOLOSCAN_CTEST_COMMAND)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

# Echo subprocess commands by default so verbose CTest output shows each invocation before it runs.
set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO STDOUT)

set(install_prefix "${HOLOSCAN_CONSUMER_WORK_DIR}/install")
set(consumer_build_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/build")
file(REMOVE_RECURSE "${install_prefix}" "${consumer_build_dir}")

# Build the complete holoscan-core install target set before component installation. This keeps the
# test usable after focused builds, where the schema DSO may not otherwise exist yet. sensor_io is
# named for the same reason and not because this consumer uses a sensor type: the install exports one
# holoscan-targets.cmake spanning every exported target, and find_package rejects it if any referenced
# file is absent.
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --target core holoscan_schema sensor_io
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${sdk_build_command} RESULT_VARIABLE sdk_build_result)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build SDK targets required by the installed consumer")
endif()

foreach(component IN ITEMS
        holoscan
        holoscan-dependencies
        iceoryx2-lib
        dev
        holoscan-holomq
        holoscan-core
        holoscan-sensor-io)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    # Run the installation and capture log output to reduce log noise
    message(STATUS "Run: ${install_command}")
    execute_process(
        COMMAND ${install_command}
        COMMAND_ECHO NONE
        OUTPUT_VARIABLE install_output
        ERROR_VARIABLE install_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE install_result
    )
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to install SDK component ${component}\n"
            "stdout:\n${install_output}\n"
            "stderr:\n${install_error}"
        )
    else()
        message(STATUS "Installed SDK component ${component}")
    endif()
endforeach()

foreach(required_install_path IN ITEMS
        "${install_prefix}/include/holoscan/core/expected.hpp"
        "${install_prefix}/include/holoscan/core/expected_macros.hpp"
        "${install_prefix}/include/holoscan/logger/logger.hpp"
        "${install_prefix}/lib/cmake/holoscan/holoscan-config.cmake")
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed consumer prerequisite is missing: ${required_install_path}")
    endif()
endforeach()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
    -B "${consumer_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${install_prefix}"
    "-DCUDAToolkit_ROOT:PATH=${HOLOSCAN_CUDA_TOOLKIT_ROOT}"
)
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
# Visual Studio and explicit toolchain files own compiler selection. Preserve the parent compiler
# only for single-config/no-toolchain package tests, where doing so improves local reproducibility
# without overriding generator or toolchain policy.
set(toolchain_selects_compiler FALSE)
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    set(toolchain_selects_compiler TRUE)
endif()
if(DEFINED HOLOSCAN_CXX_COMPILER
        AND NOT HOLOSCAN_CXX_COMPILER STREQUAL ""
        AND NOT HOLOSCAN_GENERATOR MATCHES "^Visual Studio"
        AND NOT toolchain_selects_compiler)
    list(APPEND configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}"
    )
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()

execute_process(COMMAND ${configure_command} RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure the isolated installed-package consumer")
endif()

set(consumer_build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}")
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND consumer_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${consumer_build_command} RESULT_VARIABLE consumer_build_result)
if(NOT consumer_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build the isolated installed-package consumer")
endif()

set(consumer_test_command
    "${HOLOSCAN_CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND consumer_test_command -C "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${consumer_test_command} RESULT_VARIABLE consumer_test_result)
if(NOT consumer_test_result EQUAL 0)
    message(FATAL_ERROR "Installed-package consumer execution failed")
endif()
