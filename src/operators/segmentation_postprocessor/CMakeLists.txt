# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(segmentation_postprocessor
    segmentation_postprocessor.cpp
    segmentation_postprocessor.cu
)

target_link_libraries(op_segmentation_postprocessor
    PUBLIC holoscan::core
)
