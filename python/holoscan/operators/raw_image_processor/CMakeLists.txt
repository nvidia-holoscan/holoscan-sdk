# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(raw_image_processor
    raw_image_processor.cpp
)
target_link_libraries(raw_image_processor_python
    PUBLIC holoscan::ops::raw_image_processor
)
