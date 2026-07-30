# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(bayer_demosaic bayer_demosaic.cpp)

target_link_libraries(op_bayer_demosaic
    PUBLIC
        holoscan::core
        CUDA::cudart
        CUDA::nppicc
)
