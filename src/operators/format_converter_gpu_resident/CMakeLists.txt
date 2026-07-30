# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(format_converter_gpu_resident format_converter_gpu_resident.cpp)

target_link_libraries(op_format_converter_gpu_resident
    PUBLIC
        holoscan::core
        CUDA::cudart
        CUDA::nppidei
        CUDA::nppig
        CUDA::nppicc
)
