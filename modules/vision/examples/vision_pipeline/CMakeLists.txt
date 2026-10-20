# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_vision_pipeline_example LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(holoscan COMPONENTS vision REQUIRED)

add_subdirectory(cpp)
