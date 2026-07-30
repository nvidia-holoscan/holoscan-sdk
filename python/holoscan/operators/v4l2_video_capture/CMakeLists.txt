# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(v4l2_video_capture
    v4l2_video_capture.cpp
)
target_link_libraries(v4l2_video_capture_python
    PUBLIC holoscan::ops::v4l2
)
