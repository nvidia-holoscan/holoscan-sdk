# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(video_stream_replayer video_stream_replayer.cpp)

target_link_libraries(op_video_stream_replayer
    PUBLIC
        holoscan::core
        GXF::serialization
)
