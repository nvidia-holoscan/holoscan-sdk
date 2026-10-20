# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# REQ: OPERATOR-064
foreach(required IN ITEMS SDK_SETTINGS SDK_CASE)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()
include("${SDK_SETTINGS}")
set(root "${SDK_ROOT}")
set(prefix "${root}/prefix")
set(package_dir "${prefix}/${SDK_LIB_DIR}/cmake/holoscan")
if(SDK_CASE STREQUAL "consumer" AND DEFINED SDK_CELL)
    if(NOT SDK_DISCOVERY MATCHES "^(prefix|dir)$" OR NOT SDK_CELL MATCHES "^(core-(OFF|ON)|testing)$")
        message(FATAL_ERROR "Invalid installed-consumer matrix cell: ${SDK_DISCOVERY}/${SDK_CELL}")
    endif()
    set(root "${root}/consumer-${SDK_DISCOVERY}-${SDK_CELL}")
endif()
file(MAKE_DIRECTORY "${root}/logs")
if(WIN32)
    set(ENV{PATH} "${prefix}/${SDK_BIN_DIR};${prefix}/${SDK_LIB_DIR};$ENV{PATH}")
endif()
set(build_config_args)
set(ctest_config_args)
if(NOT SDK_CONFIG STREQUAL "")
    list(APPEND build_config_args --config "${SDK_CONFIG}")
    list(APPEND ctest_config_args --build-config "${SDK_CONFIG}")
endif()

function(require_success label)
    set(log_prefix "${root}/logs/${SDK_CASE}-${label}")
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result
        OUTPUT_FILE "${log_prefix}.stdout.log" ERROR_FILE "${log_prefix}.stderr.log"
        TIMEOUT 540)
    if(NOT result EQUAL 0)
        file(READ "${log_prefix}.stdout.log" command_stdout)
        file(READ "${log_prefix}.stderr.log" command_stderr)
        message(FATAL_ERROR
            "${label} failed (${result}); evidence: ${log_prefix}.{stdout,stderr}.log\n"
            "stdout:\n${command_stdout}\nstderr:\n${command_stderr}")
    endif()
endfunction()

function(install_component component)
    require_success("install-${component}" "${CMAKE_COMMAND}" --install "${SDK_BINARY}"
        ${build_config_args} --prefix "${prefix}" --component "${component}")
endfunction()

function(configure_consumer label source_dir build_dir discovery)
    # Every case reuses one bounded build path and starts without an old CMakeCache.
    file(REMOVE_RECURSE "${build_dir}")
    set(command "${CMAKE_COMMAND}" -S "${source_dir}" -B "${build_dir}"
        -G "${SDK_GENERATOR}" -C "${SDK_SETTINGS}"
        -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF)
    if(NOT SDK_GENERATOR_PLATFORM STREQUAL "")
        list(APPEND command -A "${SDK_GENERATOR_PLATFORM}")
    endif()
    if(NOT SDK_GENERATOR_TOOLSET STREQUAL "")
        list(APPEND command -T "${SDK_GENERATOR_TOOLSET}")
    endif()
    if(discovery STREQUAL "prefix")
        list(APPEND command "-DCMAKE_PREFIX_PATH:PATH=${prefix}")
    elseif(discovery STREQUAL "dir")
        list(APPEND command "-Dholoscan_DIR:PATH=${package_dir}")
    else()
        message(FATAL_ERROR "Unknown package discovery route: ${discovery}")
    endif()
    require_success("${label}-configure" ${command} ${ARGN})
endfunction()

if(SDK_CASE STREQUAL "setup")
    file(REMOVE_RECURSE "${prefix}")
    require_success(build-sdk "${CMAKE_COMMAND}" --build "${SDK_BINARY}"
        ${build_config_args} --target operator_testing_installed_dependencies --parallel 2)
    foreach(component IN ITEMS
            holoscan holoscan-dependencies iceoryx2-lib dev holoscan-holomq holoscan-core
            holoscan-sensor-io holoscan-config holoscan-config-io)
        install_component("${component}")
    endforeach()
    if(WIN32 AND SDK_SHARED)
        install_component(bin)
    endif()
    if(SDK_TESTING_ENABLED)
        install_component(holoscan-testing)
    else()
        foreach(testing_path IN ITEMS
                "${prefix}/${SDK_INCLUDE_DIR}/holoscan/testing"
                "${prefix}/${SDK_INCLUDE_DIR}/holoscan/holoscan_testing_export.h"
                "${package_dir}/holoscan-testing-targets.cmake")
            if(EXISTS "${testing_path}")
                message(FATAL_ERROR "Testing-disabled installation contains ${testing_path}")
            endif()
        endforeach()
        file(GLOB testing_libraries
            "${prefix}/${SDK_LIB_DIR}/*holoscan_testing*"
            "${prefix}/${SDK_BIN_DIR}/*holoscan_testing*")
        if(testing_libraries)
            message(FATAL_ERROR "Testing-disabled installation contains ${testing_libraries}")
        endif()
    endif()
    message(STATUS "Installed consumer fixture prepared: ${prefix}")
    return()
endif()
if(NOT EXISTS "${package_dir}/holoscan-config.cmake")
    message(FATAL_ERROR "Installed fixture is missing; run OPERATOR_TESTING.installed_setup")
endif()

if(SDK_CASE STREQUAL "consumer")
    # Core must remain usable without importing testing, including in an SDK built with testing OFF.
    # Check both states of the caller-owned package-root variable on success and dependency failure.
    file(MAKE_DIRECTORY "${root}/core-source")
    file(WRITE "${root}/core-source/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.24)
project(core_only_consumer LANGUAGES CXX)
if(CONSUMER_SET_PACKAGE_ROOT)
    set(iceoryx2-cxx_ROOT "caller-owned-transport-root")
else()
    unset(iceoryx2-cxx_ROOT)
    unset(iceoryx2-cxx_ROOT CACHE)
endif()
if(CONSUMER_DISABLE_TRANSPORT)
    set(CMAKE_DISABLE_FIND_PACKAGE_iceoryx2-cxx TRUE)
    find_package(holoscan QUIET COMPONENTS core)
elseif(CONSUMER_REQUEST_TESTING)
    find_package(holoscan QUIET COMPONENTS testing)
else()
    find_package(holoscan COMPONENTS core REQUIRED)
endif()
if(CONSUMER_SET_PACKAGE_ROOT)
    if(NOT "${iceoryx2-cxx_ROOT}" STREQUAL "caller-owned-transport-root")
        message(FATAL_ERROR "find_package changed the caller's package root")
    endif()
elseif(DEFINED iceoryx2-cxx_ROOT)
    message(FATAL_ERROR "find_package leaked a package root into the caller")
endif()
if(CONSUMER_DISABLE_TRANSPORT OR CONSUMER_REQUEST_TESTING)
    if(holoscan_FOUND)
        message(FATAL_ERROR "The unavailable dependency/component was reported as found")
    endif()
    return()
endif()
if(TARGET holoscan::testing)
    message(FATAL_ERROR "Core-only find imported optional testing support")
endif()
add_executable(core_only main.cpp)
target_link_libraries(core_only PRIVATE holoscan::core)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/core-path-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:core_only>")
]=])
    file(WRITE "${root}/core-source/main.cpp"
        "#include <holoscan/core/graph.hpp>\nint main() { holoscan::Graph graph{\"core-only\"}; }\n")
    set(discoveries prefix dir)
    set(root_states OFF ON)
    if(DEFINED SDK_CELL)
        set(discoveries "${SDK_DISCOVERY}")
        if(SDK_CELL STREQUAL "testing")
            set(root_states)
        else()
            string(REPLACE "core-" "" root_states "${SDK_CELL}")
        endif()
    endif()
    foreach(discovery IN LISTS discoveries)
        foreach(root_defined IN LISTS root_states)
            set(label "core-${discovery}-${root_defined}")
            set(build_dir "${root}/${label}")
            configure_consumer("${label}" "${root}/core-source" "${build_dir}" "${discovery}"
                "-DCONSUMER_SET_PACKAGE_ROOT=${root_defined}")
            require_success("${label}-build" "${CMAKE_COMMAND}" --build "${build_dir}"
                ${build_config_args} --parallel 2)
            file(READ "${build_dir}/core-path-${SDK_CONFIG}.txt" executable)
            require_success("${label}-run" ${CMAKE_CROSSCOMPILING_EMULATOR} "${executable}")
            configure_consumer("${label}-missing-transport" "${root}/core-source"
                "${root}/${label}-missing-transport" "${discovery}"
                "-DCONSUMER_SET_PACKAGE_ROOT=${root_defined}" -DCONSUMER_DISABLE_TRANSPORT=ON)
        endforeach()
        if(DEFINED SDK_CELL AND NOT SDK_CELL STREQUAL "testing")
            continue()
        endif()
        if(SDK_TESTING_ENABLED)
            set(label "testing-${discovery}")
            set(build_dir "${root}/${label}")
            configure_consumer("${label}" "${SDK_SOURCE}/tests/testing/installed_consumer"
                "${build_dir}" "${discovery}")
            require_success("${label}-build" "${CMAKE_COMMAND}" --build "${build_dir}"
                ${build_config_args} --parallel 2)
            file(READ "${build_dir}/consumer-path-${SDK_CONFIG}.txt" executable)
            require_success("${label}-run" ${CMAKE_CROSSCOMPILING_EMULATOR} "${executable}")
        else()
            configure_consumer("missing-testing-${discovery}" "${root}/core-source"
                "${root}/missing-testing-${discovery}" "${discovery}"
                -DCONSUMER_REQUEST_TESTING=ON)
        endif()
    endforeach()
elseif(SDK_CASE STREQUAL "examples" OR SDK_CASE STREQUAL "gtest")
    if(SDK_CASE STREQUAL "gtest" AND NOT EXISTS "${GTEST_SOURCE}/CMakeLists.txt")
        message(STATUS "SKIP: configured GoogleTest source is unavailable: ${GTEST_SOURCE}")
        return()
    endif()
    # Example installation is an independent case; the basic DSO consumer has no dependency on it.
    require_success(build-examples "${CMAKE_COMMAND}" --build "${SDK_BINARY}"
        ${build_config_args} --target sensor_io
        operator_testing operator_testing_tensor operator_testing_image operator_testing_device
        operator_testing_schema operator_testing_lifecycle operator_testing_clock
        operator_testing_restart --parallel 2)
    install_component(holoscan-sensor-io)
    install_component(holoscan-testing-examples)
    set(example_source "${prefix}/${SDK_EXAMPLES_DIR}/operator_testing/cpp")
    if(NOT EXISTS "${example_source}/CMakeLists.txt")
        message(FATAL_ERROR "Installed operator testing example sources are missing: ${example_source}")
    endif()
    set(example_args -DOPERATOR_TESTING_WITH_GTEST=OFF)
    if(SDK_CASE STREQUAL "gtest")
        configure_consumer(gtest "${GTEST_SOURCE}" "${root}/gtest-build" dir
            "-DCMAKE_INSTALL_PREFIX=${root}/gtest" -DINSTALL_GTEST=ON -DBUILD_GMOCK=OFF)
        require_success(gtest-build "${CMAKE_COMMAND}" --build "${root}/gtest-build"
            ${build_config_args} --parallel 2)
        require_success(gtest-install "${CMAKE_COMMAND}" --install "${root}/gtest-build"
            ${build_config_args})
        set(example_args -DOPERATOR_TESTING_WITH_GTEST=ON "-DGTest_ROOT=${root}/gtest")
    endif()
    set(build_dir "${root}/${SDK_CASE}-build")
    configure_consumer("${SDK_CASE}" "${example_source}" "${build_dir}" dir ${example_args})
    if(SDK_CASE STREQUAL "gtest")
        set(example_targets --target operator_testing_gtest)
        set(example_regex "^OPERATOR_TESTING_GTEST\\.")
    else()
        set(example_targets)
        set(example_regex "^EXAMPLE_CPP_OPERATOR_TESTING")
    endif()
    require_success(examples-build "${CMAKE_COMMAND}" --build "${build_dir}"
        ${build_config_args} ${example_targets} --parallel 2)
    require_success(examples-run "${SDK_CTEST_COMMAND}" --test-dir "${build_dir}"
        ${ctest_config_args} -R "${example_regex}" --no-tests=error --output-on-failure)
else()
    message(FATAL_ERROR "Unknown installed consumer case: ${SDK_CASE}")
endif()
message(STATUS "Installed ${SDK_CASE} checks passed; evidence: ${root}/logs")
