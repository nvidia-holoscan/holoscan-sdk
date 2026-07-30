# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(inference inference.cpp)

target_link_libraries(op_inference
    PUBLIC
        holoscan::core
        holoscan::infer
    PRIVATE
        holoscan::holoinfer_utils
        holoscan::infer_utils
)
