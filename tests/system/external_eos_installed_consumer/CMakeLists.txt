# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)
project(holoscan_external_eos_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")
find_package(holoscan PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH REQUIRED)
find_package(Threads REQUIRED)

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()

add_executable(external_eos_installed_consumer main.cpp)
target_compile_features(external_eos_installed_consumer PRIVATE cxx_std_20)
set_target_properties(external_eos_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(external_eos_installed_consumer PRIVATE holoscan::core Threads::Threads)

include(CTest)
add_test(NAME external_eos_installed_consumer.run COMMAND external_eos_installed_consumer)
if(WIN32)
    set_tests_properties(external_eos_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(external_eos_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
else()
    set_tests_properties(external_eos_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
endif()
