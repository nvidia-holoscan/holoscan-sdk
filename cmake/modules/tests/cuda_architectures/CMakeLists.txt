# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_cuda_architectures_test LANGUAGES CXX)

set(holoscan_SOURCE_DIR "${HOLOSCAN_SOURCE_DIR}")
set(CMAKE_CUDA_ARCHITECTURES "75;80")
include("${holoscan_SOURCE_DIR}/cmake/modules/SetupCUDA.cmake")

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75-real;80-real;80-virtual")
    message(FATAL_ERROR "Unexpected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
