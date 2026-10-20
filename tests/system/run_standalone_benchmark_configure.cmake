# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BENCHMARK_SOURCE_DIR
        HOLOSCAN_BENCHMARK_WORK_DIR
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_CUDA_TOOLKIT_ROOT
        HOLOSCAN_GENERATOR
        HOLOSCAN_GOOGLE_BENCHMARK_SOURCE_DIR
        HOLOSCAN_INSTALL_LIB_DIR
        HOLOSCAN_STANDALONE_BENCHMARK_VERIFY)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO STDOUT)

set(install_prefix "${HOLOSCAN_BENCHMARK_WORK_DIR}/install")
set(benchmark_build_dir "${HOLOSCAN_BENCHMARK_WORK_DIR}/build")
file(REMOVE_RECURSE "${install_prefix}" "${benchmark_build_dir}")

set(sdk_build_targets core holoscan_schema sensor_io config config_io params params_io)
if(DEFINED HOLOSCAN_BUILD_SHARED_LIBS AND NOT HOLOSCAN_BUILD_SHARED_LIBS)
    list(APPEND sdk_build_targets spdlog)
endif()
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}"
    --target ${sdk_build_targets}
    --parallel 4
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${sdk_build_command}
    RESULT_VARIABLE sdk_build_result
    TIMEOUT 600
)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build SDK targets required by the standalone benchmarks")
endif()

foreach(component IN ITEMS
        holoscan
        holoscan-dependencies
        iceoryx2-lib
        dev
        holoscan-holomq
        holoscan-core
        holoscan-sensor-io
        holoscan-config
        holoscan-config-io)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${install_command}
        RESULT_VARIABLE install_result
        TIMEOUT 300
    )
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endforeach()

set(holoscan_package_dir
    "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
)
if(NOT EXISTS "${holoscan_package_dir}/holoscan-config.cmake")
    message(FATAL_ERROR
        "Standalone benchmark package is missing: ${holoscan_package_dir}/holoscan-config.cmake"
    )
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_BENCHMARK_SOURCE_DIR}"
    -B "${benchmark_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-Dholoscan_DIR:PATH=${holoscan_package_dir}"
    "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
    "-DCUDAToolkit_ROOT:PATH=${HOLOSCAN_CUDA_TOOLKIT_ROOT}"
    "-DFETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON"
    "-DFETCHCONTENT_SOURCE_DIR_BENCHMARK:PATH=${HOLOSCAN_GOOGLE_BENCHMARK_SOURCE_DIR}"
    "-DBUILD_TESTING:BOOL=OFF"
    "-DHOLOSCAN_BUILD_PYTHON:BOOL=OFF"
    "-DCMAKE_PROJECT_INCLUDE:FILEPATH=${HOLOSCAN_STANDALONE_BENCHMARK_VERIFY}"
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
set(toolchain_selects_compiler FALSE)
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    set(toolchain_selects_compiler TRUE)
endif()
if(DEFINED HOLOSCAN_CXX_COMPILER AND
   NOT HOLOSCAN_CXX_COMPILER STREQUAL "" AND
   NOT HOLOSCAN_GENERATOR MATCHES "^Visual Studio" AND
   NOT toolchain_selects_compiler)
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

execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    TIMEOUT 300
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure standalone benchmarks against the installed SDK")
endif()
