# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/consumer_coverage.cmake")

foreach(_required IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_CONSUMER_SOURCE_DIR
        HOLOSCAN_CONSUMER_WORK_DIR
        HOLOSCAN_COMPONENT_MODES
        HOLOSCAN_GENERATOR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

string(REPLACE "," ";" _available_modes "${HOLOSCAN_COMPONENT_MODES}")
foreach(_required_mode IN ITEMS config full)
  if(NOT _required_mode IN_LIST _available_modes)
    message(FATAL_ERROR "HOLOSCAN_COMPONENT_MODES must include ${_required_mode}")
  endif()
endforeach()

# Every target exported by holoscan-core is built for every mode below: installing a component
# whose target was never built leaves a dangling import file, and the shared holoscan-targets.cmake
# declares core, params, params_io, sensor_io, and sensor_io_schemas together. This test varies
# which *requestable* components a consumer names, not which Core DSOs an install carries.
set(_build_targets core holoscan_schema sensor_io config config_io params params_io)
foreach(_mode IN LISTS _available_modes)
  if(_mode STREQUAL "tuner")
    list(APPEND _build_targets experimental_params_web_tuner)
  elseif(NOT _mode MATCHES "^(config|config_io|params|params_io|full)$")
    message(FATAL_ERROR "Unknown available component mode: ${_mode}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _build_targets)

set(_sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --target ${_build_targets})
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
  list(APPEND _sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${_sdk_build_command} RESULT_VARIABLE _build_result)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR "Failed to build targets for the installed component matrix")
endif()

# holoscan-core exports holoscan::params and holoscan::params_io, which link the separately
# installed config components, so every Core-bearing prefix carries both config components.
set(_core_install_components
    holoscan
    holoscan-dependencies
    iceoryx2-lib
    dev
    holoscan-holomq
    holoscan-core
    holoscan-sensor-io
    holoscan-config
    holoscan-config-io)

foreach(_mode IN LISTS _available_modes)
  set(_install_prefix "${HOLOSCAN_CONSUMER_WORK_DIR}/${_mode}/install")
  set(_consumer_build_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/${_mode}/build")
  file(REMOVE_RECURSE "${_install_prefix}" "${_consumer_build_dir}")
  file(MAKE_DIRECTORY "${_consumer_build_dir}")
  holoscan_exclude_consumer_coverage("${HOLOSCAN_BUILD_DIR}" "${_consumer_build_dir}")

  if(_mode STREQUAL "config")
    set(_install_components holoscan holoscan-config)
  elseif(_mode STREQUAL "config_io")
    set(_install_components holoscan holoscan-config holoscan-config-io)
  elseif(_mode STREQUAL "params" OR _mode STREQUAL "params_io")
    set(_install_components ${_core_install_components})
  elseif(_mode STREQUAL "tuner")
    set(_install_components ${_core_install_components}
        holoscan-experimental-params-web-tuner)
  elseif(_mode STREQUAL "full")
    set(_install_components ${_core_install_components})
    if("tuner" IN_LIST _available_modes)
      list(APPEND _install_components holoscan-experimental-params-web-tuner)
    endif()
  endif()
  list(REMOVE_DUPLICATES _install_components)

  foreach(_component IN LISTS _install_components)
    set(_install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${_install_prefix}" --component "${_component}")
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
      list(APPEND _install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${_install_command} RESULT_VARIABLE _install_result)
    if(NOT _install_result EQUAL 0)
      message(FATAL_ERROR "Failed to install ${_component} for ${_mode} mode")
    endif()
  endforeach()

  set(_configure_command
      "${CMAKE_COMMAND}"
      -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
      -B "${_consumer_build_dir}"
      -G "${HOLOSCAN_GENERATOR}"
      "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${_install_prefix}"
      "-DHOLOSCAN_TEST_COMPONENT_MODE:STRING=${_mode}"
      "-DHOLOSCAN_TEST_AVAILABLE_COMPONENTS:STRING=${HOLOSCAN_COMPONENT_MODES}")
  if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND _configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
  endif()
  if(DEFINED HOLOSCAN_CXX_COMPILER AND NOT HOLOSCAN_CXX_COMPILER STREQUAL "")
    list(APPEND _configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
  endif()
  if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND _configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
  endif()
  if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND _configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
  endif()
  if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND _configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
  endif()
  if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND _configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
  endif()
  if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND _configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
  endif()

  foreach(_configure_pass RANGE 1 2)
    execute_process(COMMAND ${_configure_command} RESULT_VARIABLE _configure_result)
    if(NOT _configure_result EQUAL 0)
      message(FATAL_ERROR
          "Failed to configure ${_mode} consumer on pass ${_configure_pass}")
    endif()
  endforeach()

  set(_consumer_build_command "${CMAKE_COMMAND}" --build "${_consumer_build_dir}")
  if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND _consumer_build_command --config "${CMAKE_BUILD_TYPE}")
  endif()
  execute_process(COMMAND ${_consumer_build_command} RESULT_VARIABLE _consumer_build_result)
  if(NOT _consumer_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build isolated ${_mode} consumer")
  endif()

  get_filename_component(_cmake_bin_dir "${CMAKE_COMMAND}" DIRECTORY)
  set(_consumer_test_command
      "${_cmake_bin_dir}/ctest${CMAKE_EXECUTABLE_SUFFIX}"
      --test-dir "${_consumer_build_dir}" --output-on-failure)
  if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND _consumer_test_command -C "${CMAKE_BUILD_TYPE}")
  endif()
  execute_process(COMMAND ${_consumer_test_command} RESULT_VARIABLE _consumer_test_result)
  if(NOT _consumer_test_result EQUAL 0)
    message(FATAL_ERROR "Failed to run isolated ${_mode} consumer")
  endif()

  if(_mode STREQUAL "full")
    set(_negative_install_prefix "${_install_prefix}")
  endif()
endforeach()

set(_recognized_modes tuner)
foreach(_mode IN LISTS _recognized_modes)
  if(_mode IN_LIST _available_modes)
    continue()
  endif()
  if(_mode STREQUAL "tuner")
    set(_component_name experimental_params_web_tuner)
  else()
    set(_component_name "${_mode}")
  endif()
  set(_negative_mode "missing_${_component_name}")
  set(_negative_build_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/negative/${_mode}")
  file(REMOVE_RECURSE "${_negative_build_dir}")
  execute_process(
      COMMAND "${CMAKE_COMMAND}"
          -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
          -B "${_negative_build_dir}"
          -G "${HOLOSCAN_GENERATOR}"
          "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${_negative_install_prefix}"
          "-DHOLOSCAN_TEST_COMPONENT_MODE:STRING=${_negative_mode}"
      RESULT_VARIABLE _negative_result)
  if(_negative_result EQUAL 0)
    message(FATAL_ERROR "Recognized absent component unexpectedly configured: ${_component_name}")
  endif()
endforeach()

set(_unknown_build_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/negative/unknown")
file(REMOVE_RECURSE "${_unknown_build_dir}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
        -B "${_unknown_build_dir}"
        -G "${HOLOSCAN_GENERATOR}"
        "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${_negative_install_prefix}"
        "-DHOLOSCAN_TEST_COMPONENT_MODE:STRING=missing_unknown"
    RESULT_VARIABLE _unknown_result)
if(_unknown_result EQUAL 0)
  message(FATAL_ERROR "Unknown required Holoscan component unexpectedly configured")
endif()
