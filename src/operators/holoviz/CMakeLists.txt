# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(holoviz
    buffer_info.cpp
    conditions/present_done.cpp
    conditions/first_pixel_out.cpp
    holoviz.cpp
)

target_link_libraries(op_holoviz
    PUBLIC
        holoscan::core
        holoscan::viz
    PRIVATE
        CUDA::cudart
        GXF::multimedia
)
