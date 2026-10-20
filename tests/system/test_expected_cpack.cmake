# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required IN ITEMS HOLOSCAN_SDK_SOURCE_DIR HOLOSCAN_TEST_WORK_DIR
        HOLOSCAN_TEST_RAPIDS_CMAKE_DIR HOLOSCAN_TEST_CPM_FILE HOLOSCAN_TEST_EXPECTED_SOURCE_DIR)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

function(run_checked working_directory)
    execute_process(COMMAND ${ARGN} WORKING_DIRECTORY "${working_directory}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Command failed: ${ARGN}\n${output}\n${error}")
    endif()
endfunction()

get_filename_component(cmake_bin_dir "${CMAKE_COMMAND}" DIRECTORY)
set(compiler_args)
if(HOLOSCAN_TEST_CXX_COMPILER)
    list(APPEND compiler_args "-DCMAKE_CXX_COMPILER=${HOLOSCAN_TEST_CXX_COMPILER}")
endif()

foreach(triplet IN ITEMS x86_64-linux-gnu aarch64-linux-gnu)
    set(work_dir "${HOLOSCAN_TEST_WORK_DIR}/${triplet}")
    set(libdir "lib/${triplet}")
    file(REMOVE_RECURSE "${work_dir}")
    file(MAKE_DIRECTORY "${work_dir}/extracted" "${work_dir}/consumer")
    run_checked("${work_dir}" "${CMAKE_COMMAND}"
        -S "${HOLOSCAN_SDK_SOURCE_DIR}/tests/system/dependency_package_config"
        -B "${work_dir}/build"
        "-DHOLOSCAN_SDK_SOURCE_DIR=${HOLOSCAN_SDK_SOURCE_DIR}"
        "-DHOLOSCAN_TEST_RAPIDS_CMAKE_DIR=${HOLOSCAN_TEST_RAPIDS_CMAKE_DIR}"
        "-DHOLOSCAN_TEST_CPM_FILE=${HOLOSCAN_TEST_CPM_FILE}"
        "-DHOLOSCAN_TEST_EXPECTED_SOURCE_DIR=${HOLOSCAN_TEST_EXPECTED_SOURCE_DIR}"
        -DCMAKE_INSTALL_PREFIX=/usr "-DCMAKE_INSTALL_LIBDIR=${libdir}"
        ${compiler_args})
    run_checked("${work_dir}/build" "${cmake_bin_dir}/cpack" -G TGZ
        -D CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF)
    run_checked("${work_dir}/extracted" "${CMAKE_COMMAND}" -E tar xzf
        "${work_dir}/build/expected-core.tar.gz")
    file(RENAME "${work_dir}/extracted/usr" "${work_dir}/relocated")
    if(EXISTS "${work_dir}/relocated/include/tl" OR
       EXISTS "${work_dir}/relocated/${libdir}/cmake/tl-expected")
        message(FATAL_ERROR "Grouped SDK archive bundled tl-expected")
    endif()
    file(GLOB_RECURSE archive_files RELATIVE "${work_dir}/relocated"
        "${work_dir}/relocated/*")
    if(NOT archive_files STREQUAL "share/holoscan/sentinel.txt")
        message(FATAL_ERROR "Unexpected dependency payload: ${archive_files}")
    endif()
    file(WRITE "${work_dir}/consumer/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.30.4)
project(external_expected_consumer LANGUAGES CXX)
find_package(tl-expected REQUIRED CONFIG)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE tl::expected)
target_compile_features(consumer PRIVATE cxx_std_17)
enable_testing()
add_test(NAME expected_value COMMAND consumer)
]=])
    file(WRITE "${work_dir}/consumer/main.cpp" [=[
#include <tl/expected.hpp>
int main() {
    tl::expected<int, int> value(42);
    return value.value() == 42 ? 0 : 1;
}
]=])
    run_checked("${work_dir}" "${CMAKE_COMMAND}"
        -S "${work_dir}/consumer" -B "${work_dir}/consumer-build"
        "-DHOLOSCAN_TEST_PREFIX=${work_dir}/relocated"
        "-DHOLOSCAN_TEST_LIBDIR=${libdir}" ${compiler_args})
    run_checked("${work_dir}" "${CMAKE_COMMAND}" --build "${work_dir}/consumer-build"
        --config Release)
    run_checked("${work_dir}" "${cmake_bin_dir}/ctest"
        --test-dir "${work_dir}/consumer-build" -C Release --output-on-failure)
endforeach()
