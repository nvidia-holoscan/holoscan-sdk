# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Register an isolated installed-package consumer using the SDK's generator and toolchain.
# SOURCE_DIRECTORY and WORKING_DIRECTORY are required. WORKING_DIRECTORY contains the
# isolated build and install trees. RUNNER defaults to the shared Core consumer runner.
# CMAKE_ARGS carries runner-specific -D arguments (quote values containing lists); LABELS
# augments installed-package. These tests build/install from one SDK tree, so they run
# serially. TIMEOUT is optional.
function(holoscan_add_installed_consumer_test test_name)
  set(options "")
  set(one_value_args SOURCE_DIRECTORY WORKING_DIRECTORY RUNNER TIMEOUT)
  set(multi_value_args CMAKE_ARGS LABELS)
  cmake_parse_arguments(PARSE_ARGV 1 _consumer
      "${options}" "${one_value_args}" "${multi_value_args}")
  if(_consumer_UNPARSED_ARGUMENTS OR _consumer_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid installed-consumer arguments for ${test_name}: "
        "${_consumer_UNPARSED_ARGUMENTS};${_consumer_KEYWORDS_MISSING_VALUES}")
  endif()
  if("${test_name}" STREQUAL "")
    message(FATAL_ERROR "Installed consumers require a test name")
  endif()
  foreach(_required IN ITEMS SOURCE_DIRECTORY WORKING_DIRECTORY)
    if(NOT DEFINED _consumer_${_required} OR "${_consumer_${_required}}" STREQUAL "")
      message(FATAL_ERROR "Installed consumer ${test_name} requires ${_required}")
    endif()
  endforeach()
  if(NOT DEFINED _consumer_RUNNER)
    set(_consumer_RUNNER "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../system/run_time_installed_consumer.cmake")
  endif()
  get_property(_consumer_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  add_test(NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
          "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
          "-DHOLOSCAN_SDK_SOURCE_DIR=${holoscan_SOURCE_DIR}"
          "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${_consumer_SOURCE_DIRECTORY}"
          "-DHOLOSCAN_CONSUMER_WORK_DIR=${_consumer_WORKING_DIRECTORY}"
          "-DCMAKE_BUILD_TYPE=$<CONFIG>"
          "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
          "-DHOLOSCAN_GENERATOR_IS_MULTI_CONFIG=${_consumer_multi_config}"
          "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
          "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
          "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
          "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
          "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
          "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
          "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
          ${_consumer_CMAKE_ARGS}
          -P "${_consumer_RUNNER}")
  set(_consumer_labels installed-package ${_consumer_LABELS})
  list(REMOVE_DUPLICATES _consumer_labels)
  set_tests_properties("${test_name}" PROPERTIES
      RUN_SERIAL TRUE
      LABELS "${_consumer_labels}")
  if(DEFINED _consumer_TIMEOUT)
    set_tests_properties("${test_name}" PROPERTIES TIMEOUT "${_consumer_TIMEOUT}")
  endif()
endfunction()
