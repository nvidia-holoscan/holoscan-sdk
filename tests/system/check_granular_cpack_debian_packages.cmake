# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_CPACK_COMMAND
        HOLOSCAN_CPACK_CONFIG
        HOLOSCAN_DPKG_DEB_COMMAND
        HOLOSCAN_INSTALL_LIB_DIR
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
if(HOLOSCAN_BUILD_TESTING_LIBRARY)
    list(APPEND expected_package_names holoscan-testing)
endif()
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

    if(package_name STREQUAL "holoscan-core")
        foreach(testing_path IN ITEMS
                "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/holoscan/testing"
                "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/holoscan/holoscan_testing_export.h"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/libholoscan_testing"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-testing-targets"
                "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/holoscan/examples/operator_testing")
            forbid_package_path("${package_contents}" "${package_name}" "${testing_path}")
        endforeach()
        if(package_dependencies MATCHES "holoscan-testing")
            message(FATAL_ERROR "Core must not depend on optional testing support")
        endif()
        require_package_path("${package_contents}" "${package_name}"
            "./usr/bin/download_ngc_data")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/include/holoscan/holoscan.hpp")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-config.cmake")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/iceoryx2-cxx/iceoryx2-cxxConfig.cmake")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/holoscan/examples/download_example_data")
        forbid_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/libholoscan_op_holoviz.so")
    elseif(package_name STREQUAL "holoscan-testing")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/holoscan/testing/operator_test_harness.hpp")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_INCLUDE_DIR}/holoscan/holoscan_testing_export.h")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan/holoscan-testing-targets.cmake")
        require_package_path("${package_contents}" "${package_name}"
            "./usr/${HOLOSCAN_INSTALL_LIB_DIR}/libholoscan_testing")
        string(FIND "${package_dependencies}"
            "holoscan-core (= ${HOLOSCAN_PACKAGE_VERSION})" core_dependency_index)
        if(core_dependency_index EQUAL -1)
            message(FATAL_ERROR "${package_name} lacks its exact core dependency: ${package_dependencies}")
        endif()
        foreach(private_header IN ITEMS controlled_clock_run.hpp runtime_fixture_observation.hpp)
            forbid_package_path("${package_contents}" "${package_name}" "${private_header}")
        endforeach()
    elseif(package_name STREQUAL "holoscan-vision")
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
        foreach(tensorrt_runtime_package IN ITEMS
                libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10)
            string(FIND "${package_recommendations}" "${tensorrt_runtime_package}"
                tensorrt_recommendation_index)
            if(tensorrt_recommendation_index EQUAL -1)
                message(FATAL_ERROR
                    "${package_name} lacks the ${tensorrt_runtime_package} recommendation: "
                    "${package_recommendations}")
            endif()
        endforeach()
    elseif(package_name STREQUAL "holoscan-visualization")
        string(FIND "${package_dependencies}"
            "holoscan-core (= ${HOLOSCAN_PACKAGE_VERSION})" core_dependency_index)
        if(core_dependency_index EQUAL -1)
            message(FATAL_ERROR "${package_name} lacks its exact core dependency: ${package_dependencies}")
        endif()
    endif()
endforeach()
