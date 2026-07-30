# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(video_stream_replayer
    video_stream_replayer.cpp
)
target_link_libraries(video_stream_replayer_python
    PUBLIC holoscan::ops::video_stream_replayer
)
