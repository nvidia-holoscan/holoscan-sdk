# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(format_converter format_converter.cpp)

target_link_libraries(op_format_converter
    PUBLIC
        holoscan::core
        CUDA::nppidei
        CUDA::nppig
        CUDA::nppicc
)
