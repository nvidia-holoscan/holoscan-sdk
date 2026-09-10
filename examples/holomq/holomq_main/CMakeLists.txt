# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holomq_main LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(holomq_main main.cpp)
target_include_directories(holomq_main PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(holomq_main PRIVATE ${HOLOMQ_EXAMPLE_LIBRARY} holomq_common CUDA::cuda_driver)

if(BUILD_TESTING)
    find_program(HOLOMQ_BASH_EXECUTABLE bash REQUIRED)
    add_test(
        NAME holomq_main_interprocess_test
        COMMAND
            "${HOLOMQ_BASH_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/test_holomq_main.sh"
            "$<TARGET_FILE:holomq_main>"
    )
    set_tests_properties(holomq_main_interprocess_test PROPERTIES
        LABELS "cuda;holomq;interprocess"
        RUN_SERIAL TRUE
        TIMEOUT 60
    )
endif()
