# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(test_ops
  pose_tree_manager_lookup.cpp
  rx_dtype_test.cpp
  tx_dtype_test.cpp
)

target_link_libraries(op_test_ops
    PUBLIC
        holoscan::core
        holoscan::pose_tree
)
