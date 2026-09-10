# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_operator_installed_consumer LANGUAGES CXX)

set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)
find_package(holoscan REQUIRED CONFIG
    PATHS "${CMAKE_PREFIX_PATH}"
    NO_DEFAULT_PATH
)

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()
if(NOT TARGET holoscan::operators::holoviz)
    message(FATAL_ERROR "Installed package did not define holoscan::operators::holoviz")
endif()
get_filename_component(HOLOSCAN_TEST_INSTALL_PREFIX "${holoscan_DIR}/../../.." ABSOLUTE)
get_filename_component(HOLOSCAN_TEST_LIBRARY_DIR "${holoscan_DIR}/../.." ABSOLUTE)
file(WRITE "${CMAKE_BINARY_DIR}/holoscan-package-accepted" "${holoscan_DIR}\n")

add_executable(operator_installed_consumer
    main.cpp
    holoviz_operator_smoke.cpp
    parameter_compute_header_smoke.cpp
    runtime_header_smoke.cpp
)
target_compile_features(operator_installed_consumer PRIVATE cxx_std_20)
set_target_properties(operator_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(operator_installed_consumer PRIVATE
    holoscan::core
    holoscan::operators::holoviz
)

include(CTest)
add_test(NAME operator_installed_consumer.run COMMAND operator_installed_consumer)
if(WIN32)
    set_tests_properties(operator_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(operator_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_LIBRARY_DIR}")
else()
    set_tests_properties(operator_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_LIBRARY_DIR}")
endif()
