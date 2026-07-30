# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(pose_tree
    pose_tree.cpp
)

# Link the pose_tree library to the python module
target_link_libraries(pose_tree_python
    PUBLIC
        holoscan::pose_tree
)
