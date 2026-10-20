# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

# An out-of-tree constructor-handle provider built only from the installed SDK. It uses the plain
# Python C API, not nanobind or pybind11, to show that the capsule helper needs no binding library.
project(holoscan_installed_handle_provider LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

find_package(holoscan REQUIRED CONFIG PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.SABIModule)

set(capsule_header
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/python/constructor_handle_capsule.hpp")
if(NOT EXISTS "${capsule_header}")
    message(FATAL_ERROR "Installed package did not provide ${capsule_header}")
endif()

Python_add_library(installed_handle_provider MODULE USE_SABI 3.12 WITH_SOABI provider.cpp)
target_compile_features(installed_handle_provider PRIVATE cxx_std_20)
target_link_libraries(installed_handle_provider PRIVATE holoscan::core)

set(site_packages
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
string(APPEND site_packages "/site-packages")

enable_testing()
add_test(NAME installed_handle_provider
    COMMAND Python::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/check_provider.py")
set_tests_properties(installed_handle_provider PROPERTIES
    ENVIRONMENT
        "PYTHONPATH=$<TARGET_FILE_DIR:installed_handle_provider>:${site_packages};LD_LIBRARY_PATH=${HOLOSCAN_TEST_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}")
