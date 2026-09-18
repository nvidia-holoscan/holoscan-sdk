# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_inference_installed_consumer LANGUAGES CXX)

set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_yaml-cpp TRUE)
if(HOLOSCAN_PACKAGE_FIND_MODE STREQUAL "module")
    if(NOT DEFINED HOLOSCAN_MODULE_PACKAGE_DIR OR HOLOSCAN_MODULE_PACKAGE_DIR STREQUAL "")
        message(FATAL_ERROR "HOLOSCAN_MODULE_PACKAGE_DIR is required for module package discovery")
    endif()
    set(holoscan-inference_DIR "${HOLOSCAN_MODULE_PACKAGE_DIR}")
    find_package(holoscan-inference CONFIG REQUIRED)
else()
    find_package(holoscan COMPONENTS inference REQUIRED
        PATHS "${CMAKE_PREFIX_PATH}"
        NO_DEFAULT_PATH
    )
endif()

if(NOT TARGET holoscan::infer OR NOT TARGET holoscan::op_inference)
    message(FATAL_ERROR "Installed package did not define the inference targets")
endif()

add_executable(holoscan_inference_installed_consumer
    main.cpp
    inference_header_smoke.cpp
)
target_compile_features(holoscan_inference_installed_consumer PRIVATE cxx_std_20)
set_target_properties(holoscan_inference_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(holoscan_inference_installed_consumer
    PRIVATE
        holoscan::infer
        holoscan::op_inference
)

include(CTest)
if(BUILD_TESTING)
    if(HOLOSCAN_PACKAGE_FIND_MODE STREQUAL "module")
        get_filename_component(_inference_install_prefix "${holoscan-inference_DIR}/../../.." ABSOLUTE)
    else()
        get_filename_component(_inference_install_prefix "${holoscan_DIR}/../../.." ABSOLUTE)
    endif()
    add_test(NAME holoscan_inference_installed_consumer.run
        COMMAND holoscan_inference_installed_consumer
    )
    if(WIN32)
        set_tests_properties(holoscan_inference_installed_consumer.run PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_inference_install_prefix}/bin")
    elseif(APPLE)
        set_tests_properties(holoscan_inference_installed_consumer.run PROPERTIES
            ENVIRONMENT_MODIFICATION
                "DYLD_LIBRARY_PATH=path_list_prepend:${_inference_install_prefix}/lib")
    else()
        set_tests_properties(holoscan_inference_installed_consumer.run PROPERTIES
            ENVIRONMENT_MODIFICATION
                "LD_LIBRARY_PATH=path_list_prepend:${_inference_install_prefix}/lib")
    endif()
    unset(_inference_install_prefix)
endif()
