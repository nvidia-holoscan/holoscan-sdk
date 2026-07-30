# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(gpu_resident_inference gpu_resident_inference.cpp)

target_link_libraries(op_gpu_resident_inference
    PUBLIC
        holoscan::core
        holoscan::infer
    PRIVATE
        holoscan::infer_utils
)
