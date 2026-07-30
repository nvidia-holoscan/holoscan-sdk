# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(gxf_codelet gxf_codelet.cpp)

target_link_libraries(op_gxf_codelet
    PUBLIC
        holoscan::core
)
