# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/consumer_coverage.cmake")

foreach(_required IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_BUILD_TARGETS
        HOLOSCAN_CONSUMER_SOURCE_DIR
        HOLOSCAN_CONSUMER_WORK_DIR
        HOLOSCAN_GENERATOR
        HOLOSCAN_CXX_COMPILER)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

set(_sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --target ${HOLOSCAN_BUILD_TARGETS} --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
  list(APPEND _sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${_sdk_build_command} RESULT_VARIABLE _build_result)
if(NOT _build_result STREQUAL "0")
  message(FATAL_ERROR "Failed to build targets for the installed Sensor I/O consumer")
endif()

set(_install_prefix "${HOLOSCAN_CONSUMER_WORK_DIR}/install")
set(_consumer_build_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/build")
# A previous successful run must not hide an incomplete build or installation.
file(REMOVE_RECURSE "${_install_prefix}" "${_consumer_build_dir}")
file(MAKE_DIRECTORY "${_consumer_build_dir}")
holoscan_exclude_consumer_coverage("${HOLOSCAN_BUILD_DIR}" "${_consumer_build_dir}")

set(_install_components
    holoscan
    holoscan-dependencies
    iceoryx2-lib
    dev
    holoscan-holomq
    holoscan-core
    holoscan-sensor-io
    holoscan-config
    holoscan-config-io)
foreach(_component IN LISTS _install_components)
  set(_install_command
      "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
      --prefix "${_install_prefix}" --component "${_component}")
  if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND _install_command --config "${CMAKE_BUILD_TYPE}")
  endif()
  execute_process(COMMAND ${_install_command} RESULT_VARIABLE _install_result)
  if(NOT _install_result STREQUAL "0")
    message(FATAL_ERROR "Failed to install ${_component} for the Sensor I/O consumer")
  endif()
endforeach()

set(_configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
    -B "${_consumer_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH:PATH=${_install_prefix}")
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
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()
foreach(_package IN ITEMS iceoryx2-cxx iceoryx2-c iceoryx2-bb-cxx)
  if(DEFINED ${_package}_DIR AND NOT "${${_package}_DIR}" STREQUAL "")
    list(APPEND _configure_command "-D${_package}_DIR:PATH=${${_package}_DIR}")
  endif()
endforeach()
execute_process(COMMAND ${_configure_command} RESULT_VARIABLE _configure_result)
if(NOT _configure_result STREQUAL "0")
  message(FATAL_ERROR "Failed to configure the installed Sensor I/O consumer")
endif()

set(_consumer_build_command "${CMAKE_COMMAND}" --build "${_consumer_build_dir}" --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
  list(APPEND _consumer_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${_consumer_build_command} RESULT_VARIABLE _consumer_build_result)
if(NOT _consumer_build_result STREQUAL "0")
  message(FATAL_ERROR "Failed to build the installed Sensor I/O consumer")
endif()

get_filename_component(_cmake_bin_dir "${CMAKE_COMMAND}" DIRECTORY)
set(_consumer_test_command
    "${_cmake_bin_dir}/ctest${CMAKE_EXECUTABLE_SUFFIX}"
    --test-dir "${_consumer_build_dir}" --output-on-failure)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
  list(APPEND _consumer_test_command -C "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${_consumer_test_command} RESULT_VARIABLE _consumer_test_result)
if(NOT _consumer_test_result STREQUAL "0")
  message(FATAL_ERROR "Failed to run the installed Sensor I/O consumer")
endif()
