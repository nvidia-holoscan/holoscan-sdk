# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Reuse the parent's launcher without changing compiler flags or requiring caching.
# Initial-cache entries preserve semicolon arguments and allow explicit overrides.
function(holoscan_consumer_compiler_launchers output_variable)
  set(settings "")
  foreach(language IN ITEMS C CXX CUDA)
    set(variable "CMAKE_${language}_COMPILER_LAUNCHER")
    if(DEFINED ${variable})
      string(APPEND settings "set(${variable} [==[${${variable}}]==] CACHE STRING \"\")\n")
    endif()
  endforeach()
  set("${output_variable}" "${settings}" PARENT_SCOPE)
endfunction()

# Order all selected preparations before consumers so RUN_SERIAL setup does not
# repeatedly interrupt their parallel work. DEPENDS orders tests already selected;
# unlike FIXTURES_REQUIRED it does not pull unrelated preparations into focused runs.
function(holoscan_finish_consumer_ordering)
  get_property(consumers GLOBAL PROPERTY HOLOSCAN_PREPARED_CONSUMERS)
  get_property(preparations GLOBAL PROPERTY HOLOSCAN_CONSUMER_PREPARATIONS)
  list(REMOVE_DUPLICATES preparations)
  foreach(consumer IN LISTS consumers)
    get_property(directory GLOBAL PROPERTY "HOLOSCAN_CONSUMER_DIRECTORY_${consumer}")
    set_tests_properties("${consumer}" DIRECTORY "${directory}" PROPERTIES DEPENDS "${preparations}")
  endforeach()
endfunction()

function(holoscan_order_prepared_consumer test_name preparation)
  get_property(scheduled GLOBAL PROPERTY HOLOSCAN_CONSUMER_ORDERING_SCHEDULED)
  if(NOT scheduled)
    set_property(GLOBAL PROPERTY HOLOSCAN_CONSUMER_ORDERING_SCHEDULED TRUE)
    cmake_language(DEFER DIRECTORY "${holoscan_SOURCE_DIR}" CALL holoscan_finish_consumer_ordering)
  endif()
  set_property(GLOBAL APPEND PROPERTY HOLOSCAN_PREPARED_CONSUMERS "${test_name}")
  set_property(GLOBAL APPEND PROPERTY HOLOSCAN_CONSUMER_PREPARATIONS "${preparation}")
  set_property(GLOBAL PROPERTY "HOLOSCAN_CONSUMER_DIRECTORY_${test_name}" "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

# Called within holoscan_add_installed_consumer_test to preserve list-valued CMAKE_ARGS
# through one expansion, including semicolons in optional target/component lists.
function(_holoscan_add_consumer_phase test_name phase)
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
          "-DHOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR=${HOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR}"
          ${_consumer_CMAKE_ARGS}
          "-DHOLOSCAN_COMPILER_LAUNCHER_CACHE=${_consumer_launcher_cache}"
          "-DHOLOSCAN_CONSUMER_PHASE=${phase}"
          -P "${_consumer_RUNNER}")
endfunction()

# Register an isolated installed-package consumer using the SDK's generator and toolchain.
# SOURCE_DIRECTORY and WORKING_DIRECTORY are required. WORKING_DIRECTORY contains the
# isolated build and install trees. RUNNER defaults to the shared Core consumer runner.
# CMAKE_ARGS carries runner-specific -D arguments (quote values containing lists); LABELS
# augments installed-package. The default runner separates serialized SDK build/install
# preparation from private consumer work. Custom runners remain serial. TIMEOUT is optional.
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
  set(_consumer_split_preparation FALSE)
  if(NOT DEFINED _consumer_RUNNER)
    set(_consumer_split_preparation TRUE)
    set(_consumer_RUNNER "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../system/run_time_installed_consumer.cmake")
    # The common SDK exports also reference the configuration and parameter libraries.
    set(_consumer_build_targets core holoscan_schema sensor_io config config_io params params_io)
    # iceoryx2-lib installs both C++ variants, regardless of the SDK's selected linkage.
    # Imported dependencies have no local build rule and must not be passed to --target.
    foreach(_dependency IN ITEMS spdlog shared-lib-cxx static-lib-cxx)
      if(TARGET ${_dependency})
        get_target_property(_imported ${_dependency} IMPORTED)
        if(NOT _imported)
          list(APPEND _consumer_build_targets ${_dependency})
        endif()
      endif()
    endforeach()

    if(BUILD_SHARED_LIBS)
      set(_iceoryx2_target iceoryx2-cxx::shared-lib-cxx)
    else()
      set(_iceoryx2_target iceoryx2-cxx::static-lib-cxx)
    endif()
    set(_imported FALSE)
    if(TARGET ${_iceoryx2_target})
      get_target_property(_imported ${_iceoryx2_target} IMPORTED)
    endif()
    set(_consumer_package_args "")
    if(_imported)
      # Carry the selected prebuilt packages into the separate consumer configuration.
      # Source builds instead discover the packages installed in the fresh test prefix.
      foreach(_package IN ITEMS iceoryx2-cxx iceoryx2-c iceoryx2-bb-cxx)
        if(DEFINED ${_package}_DIR AND NOT "${${_package}_DIR}" STREQUAL "")
          list(APPEND _consumer_package_args "-D${_package}_DIR=${${_package}_DIR}")
        endif()
      endforeach()
    endif()

    string(REPLACE ";" "\\;" _consumer_build_targets_arg "${_consumer_build_targets}")
    list(APPEND _consumer_CMAKE_ARGS
        "-DHOLOSCAN_BUILD_TARGETS=${_consumer_build_targets_arg}" ${_consumer_package_args})
  endif()
  set(_consumer_launcher_cache "")
  if(_consumer_split_preparation)
    set(_consumer_launcher_cache
        "${_consumer_WORKING_DIRECTORY}/compiler-launchers.cmake")
    holoscan_consumer_compiler_launchers(_consumer_launcher_settings)
    # Keep generator expressions literal until the consumer evaluates them in
    # its own configuration and target context.
    file(WRITE "${_consumer_launcher_cache}" "${_consumer_launcher_settings}")
    _holoscan_add_consumer_phase("${test_name}.prepare" prepare)
    set_tests_properties("${test_name}.prepare" PROPERTIES
        RUN_SERIAL TRUE FIXTURES_SETUP "${test_name}.prepared"
        LABELS "installed-package;installed-package-setup")
    _holoscan_add_consumer_phase("${test_name}" run)
    set_tests_properties("${test_name}" PROPERTIES
        FIXTURES_REQUIRED "${test_name}.prepared" PROCESSORS 2)
    holoscan_order_prepared_consumer("${test_name}" "${test_name}.prepare")
  else()
    _holoscan_add_consumer_phase("${test_name}" all)
    set_tests_properties("${test_name}" PROPERTIES RUN_SERIAL TRUE)
  endif()
  set(_consumer_labels installed-package ${_consumer_LABELS})
  list(REMOVE_DUPLICATES _consumer_labels)
  set_tests_properties("${test_name}" PROPERTIES
      LABELS "${_consumer_labels}")
  if(DEFINED _consumer_TIMEOUT)
    set_tests_properties("${test_name}" PROPERTIES TIMEOUT "${_consumer_TIMEOUT}")
    if(_consumer_split_preparation)
      set_tests_properties("${test_name}.prepare" PROPERTIES TIMEOUT "${_consumer_TIMEOUT}")
    endif()
  endif()
endfunction()
