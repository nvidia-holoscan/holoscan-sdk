# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(raw_image_processor
    raw_image_processor.cpp
)

target_include_directories(op_raw_image_processor
    PUBLIC
        $<BUILD_INTERFACE:${HOLOSCAN_TOP}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(op_raw_image_processor
    PUBLIC
        holoscan::core
        holoscan::cuda_helper
        CUDA::cuda_driver
)
