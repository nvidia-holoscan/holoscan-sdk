# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(inference_processor inference_processor.cpp)

target_link_libraries(op_inference_processor
    PUBLIC
        holoscan::core
        holoscan::infer
        holoscan::holoinfer_utils
    PRIVATE
        holoscan::infer_utils
)
