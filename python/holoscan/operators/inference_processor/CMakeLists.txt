# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(inference_processor
    inference_processor.cpp
)

target_link_libraries(inference_processor_python
    PUBLIC holoscan::ops::inference_processor
)
