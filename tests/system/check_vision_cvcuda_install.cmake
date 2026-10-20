# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(run_checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Command failed: ${ARGV}\n${output}\n${error}")
    endif()
endfunction()

function(check_links directory soname)
    foreach(stem IN ITEMS libcvcuda libnvcv_types)
        set(link "${directory}/${stem}.so")
        if(NOT IS_SYMLINK "${link}" OR NOT EXISTS "${link}")
            message(FATAL_ERROR "Missing or broken CV-CUDA namelink: ${link}")
        endif()
        file(READ_SYMLINK "${link}" target)
        if(NOT target STREQUAL "${stem}.so.${soname}")
            message(FATAL_ERROR "Unexpected CV-CUDA namelink: ${link} -> ${target}")
        endif()
    endforeach()
endfunction()

file(MAKE_DIRECTORY "${HOLOSCAN_TEST_WORK_DIR}/source")
file(WRITE "${HOLOSCAN_TEST_WORK_DIR}/source/runtime.cpp" "int fixture() { return 0; }\n")
file(WRITE "${HOLOSCAN_TEST_WORK_DIR}/source/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.24)
project(vision_cvcuda_install_fixture VERSION 1.0.0 LANGUAGES CXX)
include("${HOLOSCAN_SOURCE_DIR}/cmake/modules/HoloscanImportedInstall.cmake")
foreach(target IN ITEMS cvcuda nvcv_types)
    add_library(${target}_provider SHARED runtime.cpp)
    set_target_properties(${target}_provider PROPERTIES
        OUTPUT_NAME ${target} SOVERSION ${FIXTURE_SONAME} VERSION ${FIXTURE_SONAME}.42.3
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/provider")
    add_library(bundled_${target} SHARED IMPORTED)
    set_target_properties(bundled_${target} PROPERTIES
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/provider/lib${target}.so.${FIXTURE_SONAME}.42.3"
        IMPORTED_SONAME "lib${target}.so.${FIXTURE_SONAME}")
    holoscan_imported_install_helper(bundled_${target} "lib${target}.so"
        "${FIXTURE_LIB_DIR}/holoscan/cvcuda" holoscan-vision)
endforeach()
set(CPACK_COMPONENTS_ALL holoscan-vision)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_PACKAGING_INSTALL_PREFIX /usr)
include(CPack)
]=])

get_filename_component(cmake_bin_dir "${CMAKE_COMMAND}" DIRECTORY)
find_program(cpack_command cpack HINTS "${cmake_bin_dir}" REQUIRED)
set(layout lib)
set(soname 7)
set(work "${HOLOSCAN_TEST_WORK_DIR}")
set(build "${work}/build")
run_checked("${CMAKE_COMMAND}" -S "${work}/source" -B "${build}"
    "-DHOLOSCAN_SOURCE_DIR=${HOLOSCAN_SOURCE_DIR}" "-DFIXTURE_LIB_DIR=${layout}"
    "-DFIXTURE_SONAME=${soname}" -DCMAKE_BUILD_TYPE=Release)
run_checked("${CMAKE_COMMAND}" --build "${build}" --config Release)
run_checked("${CMAKE_COMMAND}" --install "${build}" --config Release
    --prefix "${work}/unrelated" --component unrelated)
if(EXISTS "${work}/unrelated/${layout}/holoscan/cvcuda")
    message(FATAL_ERROR "CV-CUDA payload leaked into an unrelated component")
endif()
run_checked("${CMAKE_COMMAND}" --install "${build}" --config Release
    --prefix "${work}/install" --component holoscan-vision)
check_links("${work}/install/${layout}/holoscan/cvcuda" "${soname}")
run_checked("${CMAKE_COMMAND}" -E env "DESTDIR=${work}/destdir"
    "${CMAKE_COMMAND}" --install "${build}" --config Release
    --prefix /opt/relocated --component holoscan-vision)
check_links("${work}/destdir/opt/relocated/${layout}/holoscan/cvcuda" "${soname}")
run_checked("${cpack_command}" --config "${build}/CPackConfig.cmake" -C Release
    -G TGZ -B "${work}/tgz")
file(GLOB archives "${work}/tgz/*.tar.gz")
list(GET archives 0 archive)
file(MAKE_DIRECTORY "${work}/tgz-extract")
run_checked("${CMAKE_COMMAND}" -E chdir "${work}/tgz-extract"
    "${CMAKE_COMMAND}" -E tar xzf "${archive}")
check_links("${work}/tgz-extract/usr/${layout}/holoscan/cvcuda" "${soname}")
