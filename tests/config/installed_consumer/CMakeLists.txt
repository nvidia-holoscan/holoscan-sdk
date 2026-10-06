# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_config_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
  message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()
if(NOT DEFINED HOLOSCAN_TEST_COMPONENT_MODE OR HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "")
  message(FATAL_ERROR "HOLOSCAN_TEST_COMPONENT_MODE must name the package-consumer mode")
endif()

# Lightweight configuration consumers must not acquire the full Core dependency graph. Neither
# parser provider is a downstream package dependency in any mode.
if(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "config" OR
   HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "config_io")
  set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit TRUE)
  set(CMAKE_DISABLE_FIND_PACKAGE_Threads TRUE)
  set(CMAKE_DISABLE_FIND_PACKAGE_CCCL TRUE)
endif()
set(CMAKE_DISABLE_FIND_PACKAGE_yaml-cpp TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_reflectcpp TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_yyjson TRUE)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)
list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")

set(_all_component_targets
    holoscan::core
    holoscan::config
    holoscan::config_io
    holoscan::params
    holoscan::params_io
    holoscan::experimental_params_web_tuner
    holoscan::sensor_io
    holoscan::sensor_io_schemas)
set(_all_metadata
    holoscan-config.json
    holoscan-config-io.json
    holoscan-experimental-params-web-tuner.json)
# Every Core-bearing prefix imports the whole shared holoscan-targets.cmake, and the config
# components it links.
set(_core_closure_targets
    holoscan::core
    holoscan::config
    holoscan::config_io
    holoscan::params
    holoscan::params_io
    holoscan::sensor_io
    holoscan::sensor_io_schemas)
set(_core_closure_metadata holoscan-config.json holoscan-config-io.json)

if(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "missing_unknown")
  find_package(holoscan REQUIRED CONFIG COMPONENTS definitely_unknown
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  message(FATAL_ERROR "Unknown component unexpectedly resolved")
elseif(HOLOSCAN_TEST_COMPONENT_MODE MATCHES "^missing_(.+)$")
  set(_missing_component "${CMAKE_MATCH_1}")
  find_package(holoscan REQUIRED CONFIG COMPONENTS "${_missing_component}"
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  message(FATAL_ERROR "Recognized absent component unexpectedly resolved: ${_missing_component}")
elseif(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "config")
  find_package(holoscan REQUIRED CONFIG COMPONENTS config
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  set(_expected_targets holoscan::config)
  set(_expected_metadata holoscan-config.json)
  set(_link_target holoscan::config)
elseif(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "config_io")
  find_package(holoscan REQUIRED CONFIG COMPONENTS config_io
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  set(_expected_targets holoscan::config holoscan::config_io)
  set(_expected_metadata holoscan-config.json holoscan-config-io.json)
  set(_link_target holoscan::config_io)
  add_compile_definitions(HOLOSCAN_TEST_USE_CONFIG_IO=1)
elseif(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "params")
  # params is a pass-through name for a target in the Core export set, so requesting it yields the
  # whole Core closure rather than a narrower one.
  find_package(holoscan REQUIRED CONFIG COMPONENTS params
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  set(_expected_targets ${_core_closure_targets})
  set(_expected_metadata ${_core_closure_metadata})
  set(_link_target holoscan::params)
  add_compile_definitions(HOLOSCAN_TEST_USE_PARAMS=1)
elseif(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "params_io")
  find_package(holoscan REQUIRED CONFIG COMPONENTS params_io
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  set(_expected_targets ${_core_closure_targets})
  set(_expected_metadata ${_core_closure_metadata})
  set(_link_target holoscan::params_io)
  add_compile_definitions(HOLOSCAN_TEST_USE_PARAMS_IO=1)
elseif(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "tuner")
  find_package(holoscan REQUIRED CONFIG COMPONENTS experimental_params_web_tuner
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  set(_expected_targets ${_core_closure_targets} holoscan::experimental_params_web_tuner)
  set(_expected_metadata ${_core_closure_metadata}
                         holoscan-experimental-params-web-tuner.json)
  set(_link_target holoscan::experimental_params_web_tuner)
  add_compile_definitions(HOLOSCAN_TEST_USE_TUNER=1)
elseif(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "full")
  find_package(holoscan REQUIRED CONFIG
      PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
  set(_expected_targets ${_core_closure_targets})
  set(_expected_metadata ${_core_closure_metadata})
  string(REPLACE "," ";" _available_modes "${HOLOSCAN_TEST_AVAILABLE_COMPONENTS}")
  if("tuner" IN_LIST _available_modes)
    list(APPEND _expected_targets holoscan::experimental_params_web_tuner)
    list(APPEND _expected_metadata holoscan-experimental-params-web-tuner.json)
  endif()
  # Exercise every imported component's link interface in the aggregate mode. The consumer body
  # uses the configuration API, while the complete target list proves that the Core closure and
  # the optional tuner coexist without hidden ordering requirements.
  set(_link_target ${_expected_targets})

  # The Core export set declares the sensor targets, so a Core-bearing prefix must not also carry
  # a stale standalone export file for them.
  get_filename_component(_holoscan_package_dir "${holoscan_CONFIG}" DIRECTORY)
  foreach(_stale_export IN ITEMS
      holoscan-sensor-io-targets.cmake
      holoscan-params-targets.cmake
      holoscan-params-io-targets.cmake)
    if(EXISTS "${_holoscan_package_dir}/${_stale_export}")
      message(FATAL_ERROR "Stale standalone export file installed: ${_stale_export}")
    endif()
  endforeach()
  file(READ "${_holoscan_package_dir}/holoscan-targets.cmake" _holoscan_shared_exports)
  foreach(_core_target IN ITEMS holoscan::params holoscan::params_io holoscan::sensor_io)
    if(NOT _holoscan_shared_exports MATCHES "${_core_target}")
      message(FATAL_ERROR
          "The shared holoscan-targets.cmake does not declare ${_core_target}")
    endif()
  endforeach()
else()
  message(FATAL_ERROR "Unknown HOLOSCAN_TEST_COMPONENT_MODE=${HOLOSCAN_TEST_COMPONENT_MODE}")
endif()

if(HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "config" OR
   HOLOSCAN_TEST_COMPONENT_MODE STREQUAL "config_io")
  if(TARGET holoscan::fbs_tools OR COMMAND holoscan_add_flatbuffer_schema)
    message(FATAL_ERROR
        "Lightweight component falsely advertised unavailable schema tooling")
  endif()
elseif(NOT TARGET flatbuffers::flatc OR
       NOT TARGET holoscan::fbs_tools OR
       NOT COMMAND holoscan_add_flatbuffer_schema)
  message(FATAL_ERROR "Core component did not expose installed schema tooling")
endif()

foreach(_target IN LISTS _all_component_targets)
  if(_target IN_LIST _expected_targets)
    if(NOT TARGET ${_target})
      message(FATAL_ERROR "Installed component did not define ${_target}")
    endif()
  elseif(TARGET ${_target})
    message(FATAL_ERROR
        "Installed component ${HOLOSCAN_TEST_COMPONENT_MODE} imported non-closure target ${_target}")
  endif()
endforeach()

foreach(_target IN LISTS _expected_targets)
  if(_target MATCHES "^holoscan::config(_io)?$")
    get_target_property(_links ${_target} INTERFACE_LINK_LIBRARIES)
    if(_links MATCHES "CUDA|CCCL|Threads|yaml-cpp|reflectcpp|yyjson")
      message(FATAL_ERROR "${_target} leaked a private or Core-only dependency: ${_links}")
    endif()
  endif()
endforeach()

set(_metadata_dir "${HOLOSCAN_TEST_INSTALL_PREFIX}/share/holoscan/tooling/v1")
foreach(_metadata IN LISTS _all_metadata)
  set(_metadata_path "${_metadata_dir}/${_metadata}")
  if(_metadata IN_LIST _expected_metadata)
    if(NOT EXISTS "${_metadata_path}")
      message(FATAL_ERROR "Installed component metadata is missing: ${_metadata}")
    endif()
    file(READ "${_metadata_path}" _metadata_content)
    if(NOT _metadata_content MATCHES "\"schema\":\"holoscan.tooling-component/v1\"")
      message(FATAL_ERROR "Installed component metadata has the wrong schema: ${_metadata}")
    endif()
    if(_metadata STREQUAL "holoscan-config.json")
      set(_metadata_target holoscan::config)
    elseif(_metadata STREQUAL "holoscan-config-io.json")
      set(_metadata_target holoscan::config_io)
    else()
      set(_metadata_target holoscan::experimental_params_web_tuner)
    endif()
    if(NOT _metadata_content MATCHES "\"target\":\"${_metadata_target}\"")
      message(FATAL_ERROR "Installed component metadata has the wrong target: ${_metadata}")
    endif()
  elseif(EXISTS "${_metadata_path}")
    message(FATAL_ERROR "Isolated component install leaked non-closure metadata: ${_metadata}")
  endif()
endforeach()

add_executable(config_installed_consumer main.cpp)
target_compile_features(config_installed_consumer PRIVATE cxx_std_20)
set_target_properties(config_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(config_installed_consumer PRIVATE ${_link_target})

enable_testing()
add_test(NAME config_installed_consumer COMMAND config_installed_consumer)
