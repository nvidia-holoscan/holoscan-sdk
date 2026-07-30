# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(v4l2 v4l2_video_capture.cpp)

target_link_libraries(op_v4l2
    PUBLIC
        holoscan::core
        op_video_io
    PRIVATE
        GXF::multimedia
        jpeg
)
