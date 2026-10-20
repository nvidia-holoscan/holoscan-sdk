# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${HOLOSCAN_SDK_SOURCE_DIR}/tests/cmake/consumer_coverage.cmake")

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_CONSUMER_SOURCE_DIR
        HOLOSCAN_CONSUMER_WORK_DIR
        HOLOSCAN_GENERATOR
        HOLOSCAN_GENERATOR_IS_MULTI_CONFIG)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()
unset(required_variable)

set(package_work_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/package")
set(install_prefix "${package_work_dir}/install")
set(consumer_build_dir "${package_work_dir}/build")
file(REMOVE_RECURSE "${package_work_dir}")
file(MAKE_DIRECTORY "${consumer_build_dir}")
holoscan_exclude_consumer_coverage("${HOLOSCAN_BUILD_DIR}" "${consumer_build_dir}")

# The shared holoscan-targets.cmake references every Core-exported DSO, so the Core closure
# (config, params, sensor_io) is built alongside the inference targets.
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --parallel 4
    --target core config config_io params params_io sensor_io holoinfer_utils holoinfer
    op_inference op_tensor_preprocessor)
if(HOLOSCAN_BUILD_EXAMPLES)
    list(APPEND sdk_build_command holoinfer_example)
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${sdk_build_command}
    RESULT_VARIABLE sdk_build_result
)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build inference targets for the installed-package consumer")
endif()

# iceoryx2's package config and headers use its upstream `dev` component; its libraries and
# exported targets stay in upstream-defined components while HoloMQ uses its SDK-owned component.
# holoscan-core exports targets that link holoscan::config/config_io and requires the sensor_io
# DSO, so every Core-bearing prefix carries those components as well.
foreach(component IN ITEMS holoscan holoscan-dependencies iceoryx2-lib dev holoscan-holomq
        holoscan-core holoscan-sensor-io holoscan-config holoscan-config-io)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}")
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endforeach()
unset(component)

foreach(inference_payload IN ITEMS
        "${install_prefix}/include/holoinfer/holoinfer.hpp"
        "${install_prefix}/include/holoscan/operators/inference/inference.hpp"
        "${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}/holoscan-inference-targets.cmake")
    if(EXISTS "${inference_payload}")
        message(FATAL_ERROR "The holoscan-core installation unexpectedly contains ${inference_payload}")
    endif()
endforeach()
unset(inference_payload)

set(install_command
    "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
    --prefix "${install_prefix}"
    --component holoscan-inference)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Failed to install SDK component holoscan-inference")
endif()

foreach(required_inference_payload IN ITEMS
        "${install_prefix}/include/holoinfer/holoinfer.hpp"
        "${install_prefix}/include/holoinfer/holoinfer_export.h"
        "${install_prefix}/include/holoscan/operators/inference/inference.hpp"
        "${install_prefix}/include/holoscan/utils/holoinfer_utils.hpp"
        "${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}/holoscan-inference-config.cmake"
        "${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}/holoscan-inference-config-version.cmake"
        "${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}/holoscan-inference-targets.cmake")
    if(NOT EXISTS "${required_inference_payload}")
        message(FATAL_ERROR "Installed inference payload is missing: ${required_inference_payload}")
    endif()
endforeach()
unset(required_inference_payload)

if(HOLOSCAN_BUILD_EXAMPLES)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component holoscan-inference-examples)
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component holoscan-inference-examples")
    endif()

endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
    -B "${consumer_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
    "-DBUILD_TESTING:BOOL=ON")
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_CXX_COMPILER AND NOT HOLOSCAN_CXX_COMPILER STREQUAL "")
    list(APPEND configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
endif()
execute_process(COMMAND ${configure_command} RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure the installed inference package consumer")
endif()

set(consumer_build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}" --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND consumer_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${consumer_build_command} RESULT_VARIABLE consumer_build_result)
if(NOT consumer_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build the installed inference package consumer")
endif()

set(consumer_test_command "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND consumer_test_command --build-config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${consumer_test_command} RESULT_VARIABLE consumer_test_result)
if(NOT consumer_test_result EQUAL 0)
    message(FATAL_ERROR "Installed inference package consumer test failed")
endif()

set(module_consumer_build_dir "${package_work_dir}/module-build")
file(MAKE_DIRECTORY "${module_consumer_build_dir}")
holoscan_exclude_consumer_coverage("${HOLOSCAN_BUILD_DIR}" "${module_consumer_build_dir}")
set(module_configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
    -B "${module_consumer_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
    "-DHOLOSCAN_PACKAGE_FIND_MODE:STRING=module"
    "-DHOLOSCAN_MODULE_PACKAGE_DIR:PATH=${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}"
    "-DBUILD_TESTING:BOOL=ON")
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND module_configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND module_configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND module_configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_CXX_COMPILER AND NOT HOLOSCAN_CXX_COMPILER STREQUAL "")
    list(APPEND module_configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND module_configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND module_configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
endif()
if(NOT HOLOSCAN_GENERATOR_IS_MULTI_CONFIG AND
   DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND module_configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${module_configure_command} RESULT_VARIABLE module_configure_result)
if(NOT module_configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure the standalone inference package consumer")
endif()

set(module_consumer_build_command
    "${CMAKE_COMMAND}" --build "${module_consumer_build_dir}" --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND module_consumer_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${module_consumer_build_command} RESULT_VARIABLE module_consumer_build_result)
if(NOT module_consumer_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build the standalone inference package consumer")
endif()

set(module_consumer_test_command
    "${CMAKE_CTEST_COMMAND}" --test-dir "${module_consumer_build_dir}" --output-on-failure)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND module_consumer_test_command --build-config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${module_consumer_test_command} RESULT_VARIABLE module_consumer_test_result)
if(NOT module_consumer_test_result EQUAL 0)
    message(FATAL_ERROR "Standalone inference package consumer test failed")
endif()
