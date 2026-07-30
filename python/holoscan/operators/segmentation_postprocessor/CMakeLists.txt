# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(segmentation_postprocessor
    segmentation_postprocessor.cpp
)
target_link_libraries(segmentation_postprocessor_python
    PUBLIC holoscan::ops::segmentation_postprocessor
)
