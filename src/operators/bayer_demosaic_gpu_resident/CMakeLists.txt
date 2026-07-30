# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(bayer_demosaic_gpu_resident bayer_demosaic_gpu_resident.cpp)

target_link_libraries(op_bayer_demosaic_gpu_resident
    PUBLIC
        holoscan::core
        CUDA::cudart
        CUDA::nppicc
)
