# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(video_io
    video_io_capability_convert.cpp
    video_io_registry.cpp
    video_acquisition_operator.cpp
    video_transmission_operator.cpp
)
