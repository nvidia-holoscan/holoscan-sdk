# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(test_ops
    test_ops.cpp
)
target_link_libraries(test_ops_python
    PUBLIC holoscan::ops::test_ops
)
