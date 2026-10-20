# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_CPACK_COMMAND
        HOLOSCAN_CPACK_CONFIG
        HOLOSCAN_DPKG_DEB_COMMAND
        HOLOSCAN_META_DEB_SCRIPT
        HOLOSCAN_INSTALL_LIB_DIR
        HOLOSCAN_THIRD_PARTY_INCLUDE_DIR
        HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR
        HOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR
        HOLOSCAN_PACKAGE_VERSION
        HOLOSCAN_CPACK_WORK_DIR)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} must be provided")
    endif()
endforeach()

if(NOT EXISTS "${HOLOSCAN_CPACK_CONFIG}")
    message(FATAL_ERROR "Missing CPack configuration: ${HOLOSCAN_CPACK_CONFIG}")
endif()
function(get_debian_field package_file field output_variable)
    execute_process(
        COMMAND "${HOLOSCAN_DPKG_DEB_COMMAND}" --field "${package_file}" "${field}"
        RESULT_VARIABLE field_result
        OUTPUT_VARIABLE field_value
        ERROR_VARIABLE field_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT field_result EQUAL 0)
        message(FATAL_ERROR "Unable to read ${field} from ${package_file}: ${field_error}")
    endif()
    set(${output_variable} "${field_value}" PARENT_SCOPE)
endfunction()

function(require_package_path package_contents package_name required_path)
    string(FIND "${package_contents}" "${required_path}" required_path_index)
    if(required_path_index EQUAL -1)
        message(FATAL_ERROR "${package_name} is missing required path: ${required_path}")
    endif()
endfunction()

function(forbid_package_path package_contents package_name forbidden_path)
    string(FIND "${package_contents}" "${forbidden_path}" forbidden_path_index)
    if(NOT forbidden_path_index EQUAL -1)
        message(FATAL_ERROR "${package_name} must not contain: ${forbidden_path}")
    endif()
endfunction()

file(REMOVE_RECURSE "${HOLOSCAN_CPACK_WORK_DIR}")
set(payload_dir "${HOLOSCAN_CPACK_WORK_DIR}/payloads")
execute_process(
    COMMAND "${HOLOSCAN_CPACK_COMMAND}"
        --config "${HOLOSCAN_CPACK_CONFIG}"
        -G DEB
        -B "${payload_dir}"
    RESULT_VARIABLE cpack_result
    OUTPUT_VARIABLE cpack_output
    ERROR_VARIABLE cpack_error
)
if(NOT cpack_result EQUAL 0)
    message(FATAL_ERROR
        "Granular CPack Debian generation failed with exit code ${cpack_result}:\n"
        "${cpack_output}\n${cpack_error}")
endif()

set(expected_package_names holoscan-core)
if(HOLOSCAN_BUILD_VISION)
    list(APPEND expected_package_names holoscan-vision)
endif()
if(HOLOSCAN_BUILD_INFERENCE)
    list(APPEND expected_package_names holoscan-inference)
endif()
if(HOLOSCAN_BUILD_VISUALIZATION)
    list(APPEND expected_package_names holoscan-visualization)
endif()

file(GLOB payload_packages "${payload_dir}/*.deb")
list(LENGTH payload_packages payload_package_count)
list(LENGTH expected_package_names expected_package_count)
if(NOT payload_package_count EQUAL expected_package_count)
    message(FATAL_ERROR
        "Expected ${expected_package_count} granular Debian packages, found "
        "${payload_package_count}: ${payload_packages}")
endif()

set(found_package_names)
foreach(payload_package IN LISTS payload_packages)
    get_debian_field("${payload_package}" Package package_name)
    get_debian_field("${payload_package}" Version package_version)
    get_debian_field("${payload_package}" Depends package_dependencies)
    get_debian_field("${payload_package}" Recommends package_recommendations)

    list(FIND expected_package_names "${package_name}" expected_package_index)
    if(expected_package_index EQUAL -1)
        message(FATAL_ERROR "Unexpected granular package '${package_name}': ${payload_package}")
    endif()
    list(FIND found_package_names "${package_name}" duplicate_package_index)
    if(NOT duplicate_package_index EQUAL -1)
        message(FATAL_ERROR "Duplicate granular package: ${package_name}")
    endif()
    list(APPEND found_package_names "${package_name}")

    if(NOT package_version STREQUAL HOLOSCAN_PACKAGE_VERSION)
        message(FATAL_ERROR
            "${package_name} has version ${package_version}; expected ${HOLOSCAN_PACKAGE_VERSION}")
    endif()

    execute_process(
        COMMAND "${HOLOSCAN_DPKG_DEB_COMMAND}" --contents "${payload_package}"
        RESULT_VARIABLE contents_result
        OUTPUT_VARIABLE package_contents
        ERROR_VARIABLE contents_error
    )
    if(NOT contents_result EQUAL 0)
        message(FATAL_ERROR "Unable to inspect ${payload_package}: ${contents_error}")
    endif()
    if(package_contents MATCHES "\\./opt/nvidia/holoscan")
        message(FATAL_ERROR "${package_name} retains a legacy /opt/nvidia/holoscan payload")
    endif()
    if(package_contents MATCHES "\\./usr/lib/python[0-9][^/]*/(site|dist)-packages/holoscan")
        message(FATAL_ERROR "${package_name} includes version-specific Python bindings")
    endif()
    forbid_package_path("${package_contents}" "${package_name}" "/holoscan/examples")
    forbid_package_path("${package_contents}" "${package_name}" "./usr/LICENSE.txt")
    if(NOT package_name STREQUAL "holoscan-visualization" AND package_recommendations)
        message(FATAL_ERROR
            "${package_name} inherits unexpected recommendations: ${package_recommendations}")
    endif()

    if(package_name STREQUAL "holoscan-core")
        if(NOT package_dependencies MATCHES "libexpected-dev \\(>= 1\\.1\\.0\\)")
            message(FATAL_ERROR "Core lacks the compatible libexpected-dev dependency")
        endif()
        forbid_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/tl/")
        forbid_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/tl-expected/")
        foreach(private_path IN ITEMS
                "fmt/format.h"
                "dlpack/dlpack.h"
                "flatbuffers/flatbuffers.h"
                "nvtx3/nvtx3.hpp")
            require_package_path("${package_contents}" "${package_name}"
                "./usr/${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}/${private_path}")
        endforeach()
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR}/flatc")
        foreach(private_package IN ITEMS fmt dlpack flatbuffers nvtx3)
            require_package_path("${package_contents}" "${package_name}"
                "./usr/${HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR}/cmake/${private_package}/")
        endforeach()
        foreach(public_path IN ITEMS
                "./usr/include/fmt/"
                "./usr/include/dlpack/"
                "./usr/include/flatbuffers/"
                "./usr/include/nvtx3/"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/fmt/"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/dlpack/"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/flatbuffers/"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/nvtx3/"
                "./usr/bin/flatc")
            forbid_package_path("${package_contents}" "${package_name}" "${public_path}")
        endforeach()
        if(HOLOSCAN_BUILD_SHARED_LIBS)
            foreach(spdlog_path IN ITEMS
                    "./usr/include/spdlog/"
                    "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/spdlog/")
                forbid_package_path("${package_contents}" "${package_name}" "${spdlog_path}")
            endforeach()
        endif()
        foreach(testing_path IN ITEMS
                "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/holoscan/testing/operator_test_harness.hpp"
                "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/holoscan/holoscan_testing_export.h"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/libholoscan_testing"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-testing-targets.cmake")
            if(HOLOSCAN_BUILD_TESTING_LIBRARY)
                require_package_path("${package_contents}" "${package_name}" "${testing_path}")
            else()
                forbid_package_path("${package_contents}" "${package_name}" "${testing_path}")
            endif()
        endforeach()
        foreach(private_header IN ITEMS controlled_clock_run.hpp runtime_fixture_observation.hpp)
            forbid_package_path("${package_contents}" "${package_name}" "${private_header}")
        endforeach()
        if(package_dependencies MATCHES "holoscan-testing")
            message(FATAL_ERROR "Core must not depend on a separate testing package")
        endif()
        require_package_path("${package_contents}" "${package_name}"
            "./usr/bin/download_ngc_data")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/include/holoscan/holoscan.hpp")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-config.cmake")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/iceoryx2-cxx/iceoryx2-cxxConfig.cmake")
        forbid_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/libholoscan_op_holoviz.so")
    elseif(package_name STREQUAL "holoscan-vision")
        set(vision_extract_dir "${HOLOSCAN_CPACK_WORK_DIR}/vision-extract")
        execute_process(
            COMMAND "${HOLOSCAN_DPKG_DEB_COMMAND}" --extract "${payload_package}" "${vision_extract_dir}"
            RESULT_VARIABLE vision_extract_result
        )
        if(NOT vision_extract_result EQUAL 0)
            message(FATAL_ERROR "Cannot extract ${package_name} for CV-CUDA namelink validation")
        endif()
        foreach(stem IN ITEMS libcvcuda libnvcv_types)
            set(link "${vision_extract_dir}/usr/${HOLOSCAN_INSTALL_LIB_DIR}/holoscan/cvcuda/${stem}.so")
            if(NOT IS_SYMLINK "${link}" OR NOT EXISTS "${link}")
                message(FATAL_ERROR "${package_name} is missing a working CV-CUDA namelink: ${stem}.so")
            endif()
            file(READ_SYMLINK "${link}" link_target)
            if(NOT link_target STREQUAL "${stem}.so.0")
                message(FATAL_ERROR "${package_name} has an unexpected CV-CUDA namelink: ${link_target}")
            endif()
        endforeach()
        require_package_path("${package_contents}" "${package_name}"
            "./usr/include/holoscan/vision/vision_types.hpp")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-vision-targets.cmake")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-vision-config.cmake")
        string(FIND "${package_dependencies}"
            "holoscan-core (= ${HOLOSCAN_PACKAGE_VERSION})" core_dependency_index)
        if(core_dependency_index EQUAL -1)
            message(FATAL_ERROR "${package_name} lacks its exact core dependency: ${package_dependencies}")
        endif()
        if(package_dependencies MATCHES "libcvcuda0")
            message(FATAL_ERROR "${package_name} must bundle, not depend on, CV-CUDA: ${package_dependencies}")
        endif()
    elseif(package_name STREQUAL "holoscan-inference")
        string(FIND "${package_dependencies}"
            "holoscan-core (= ${HOLOSCAN_PACKAGE_VERSION})" core_dependency_index)
        if(core_dependency_index EQUAL -1)
            message(FATAL_ERROR "${package_name} lacks its exact core dependency: ${package_dependencies}")
        endif()
        foreach(inference_dependency IN ITEMS
                "libnvinfer10 (>= 10.13)" "libnvinfer-plugin10 (>= 10.13)"
                "libnvonnxparsers10 (>= 10.13)" "libnvrtc.so.")
            string(FIND "${package_dependencies}" "${inference_dependency}"
                inference_dependency_index)
            if(inference_dependency_index EQUAL -1)
                message(FATAL_ERROR
                    "${package_name} lacks the ${inference_dependency} dependency: "
                    "${package_dependencies}")
            endif()
        endforeach()
        if(HOLOSCAN_BUILD_SHARED_LIBS)
            set(inference_npp_dependency "libnpp.so.13")
        else()
            set(inference_npp_dependency "libnpp.so.13-dev")
        endif()
        if(NOT package_dependencies MATCHES "(^|, )${inference_npp_dependency}(,|$)")
            message(FATAL_ERROR
                "${package_name} lacks the ${inference_npp_dependency} dependency: "
                "${package_dependencies}")
        endif()
    elseif(package_name STREQUAL "holoscan-visualization")
        string(FIND "${package_dependencies}"
            "holoscan-core (= ${HOLOSCAN_PACKAGE_VERSION})" core_dependency_index)
        if(core_dependency_index EQUAL -1)
            message(FATAL_ERROR "${package_name} lacks its exact core dependency: ${package_dependencies}")
        endif()
        if(NOT package_dependencies MATCHES "(^|, )libvulkan1(,|$)")
            message(FATAL_ERROR "${package_name} lacks the libvulkan1 dependency: ${package_dependencies}")
        endif()
        if(NOT package_recommendations STREQUAL "libdrm2, libegl1, libx11-6, libxrandr2")
            message(FATAL_ERROR
                "${package_name} has unexpected recommendations: ${package_recommendations}")
        endif()
    endif()
endforeach()

if(HOLOSCAN_BUILD_VISION AND HOLOSCAN_BUILD_INFERENCE AND HOLOSCAN_BUILD_VISUALIZATION)
    set(expected_meta_dependencies)
    foreach(package_name IN ITEMS holoscan-core holoscan-vision holoscan-inference holoscan-visualization)
        if(expected_meta_dependencies)
            string(APPEND expected_meta_dependencies ", ")
        endif()
        string(APPEND expected_meta_dependencies "${package_name} (= ${HOLOSCAN_PACKAGE_VERSION})")
    endforeach()

    set(meta_args)
    foreach(payload_package IN LISTS payload_packages)
        list(APPEND meta_args --package "${payload_package}")
    endforeach()
    foreach(input_mode IN ITEMS payloads version)
        set(meta_dir "${HOLOSCAN_CPACK_WORK_DIR}/meta-${input_mode}")
        if(input_mode STREQUAL "payloads")
            set(input_args ${meta_args})
        else()
            set(input_args --version "${HOLOSCAN_PACKAGE_VERSION}")
        endif()
        execute_process(
            COMMAND "${HOLOSCAN_META_DEB_SCRIPT}" --output-dir "${meta_dir}" ${input_args}
            RESULT_VARIABLE meta_result ERROR_VARIABLE meta_error)
        if(NOT meta_result EQUAL 0)
            message(FATAL_ERROR "Meta-package generation (${input_mode}) failed: ${meta_error}")
        endif()
        file(GLOB meta_packages "${meta_dir}/holoscan_*_all.deb")
        list(LENGTH meta_packages meta_count)
        if(NOT meta_count EQUAL 1)
            message(FATAL_ERROR "Expected one ${input_mode} meta-package, found ${meta_count}")
        endif()
        list(GET meta_packages 0 meta_package)
        get_debian_field("${meta_package}" Package meta_name)
        get_debian_field("${meta_package}" Depends meta_dependencies)
        if(NOT meta_name STREQUAL "holoscan" OR
                NOT meta_dependencies STREQUAL expected_meta_dependencies)
            message(FATAL_ERROR
                "Wrong ${input_mode} meta-package contract: ${meta_name}: ${meta_dependencies}")
        endif()
    endforeach()

endif()

if(HOLOSCAN_BUILD_EXAMPLES)
    set(examples_dir "${HOLOSCAN_CPACK_WORK_DIR}/payloads-with-examples")
    execute_process(
        COMMAND "${HOLOSCAN_CPACK_COMMAND}"
            --config "${HOLOSCAN_CPACK_CONFIG}"
            -G DEB
            -D HOLOSCAN_CPACK_INCLUDE_EXAMPLES=ON
            -B "${examples_dir}"
        RESULT_VARIABLE cpack_result
        OUTPUT_VARIABLE cpack_output
        ERROR_VARIABLE cpack_error
    )
    if(NOT cpack_result EQUAL 0)
        message(FATAL_ERROR
            "Opt-in example CPack Debian generation failed with exit code ${cpack_result}:\n"
            "${cpack_output}\n${cpack_error}")
    endif()
    file(GLOB core_packages "${examples_dir}/holoscan-core*.deb")
    list(LENGTH core_packages core_package_count)
    if(NOT core_package_count EQUAL 1)
        message(FATAL_ERROR "Expected one opt-in example core package, found: ${core_packages}")
    endif()
    execute_process(
        COMMAND "${HOLOSCAN_DPKG_DEB_COMMAND}" --contents "${core_packages}"
        RESULT_VARIABLE contents_result
        OUTPUT_VARIABLE package_contents
        ERROR_VARIABLE contents_error
    )
    if(NOT contents_result EQUAL 0)
        message(FATAL_ERROR "Unable to inspect ${core_packages}: ${contents_error}")
    endif()
    set(examples_prefix "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/holoscan/examples")
    require_package_path("${package_contents}" holoscan-core
        "${examples_prefix}/download_example_data")
    if(HOLOSCAN_BUILD_TESTING_LIBRARY)
        require_package_path("${package_contents}" holoscan-core
            "${examples_prefix}/operator_testing")
    endif()
endif()
