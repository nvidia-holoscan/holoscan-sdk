# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
endif()

project(holomq_examples LANGUAGES CXX CUDA)

include(CTest)

if(NOT TARGET holomq::holomq)
    find_package(holomq REQUIRED CONFIG
        PATHS
            "${CMAKE_CURRENT_LIST_DIR}/.."
    )
endif()

set(HOLOMQ_EXAMPLE_LIBRARY holomq::holomq)

add_subdirectory(common)
add_subdirectory(holomq_main)

include(GNUInstallDirs)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/holomq/examples
    COMPONENT holomq
    FILES_MATCHING
        PATTERN "CMakeLists.txt"
        PATTERN "*.cpp"
        PATTERN "*.cu"
        PATTERN "*.hpp"
        PATTERN "*.sh"
        PATTERN "README.md"
    PATTERN "__pycache__" EXCLUDE
)
