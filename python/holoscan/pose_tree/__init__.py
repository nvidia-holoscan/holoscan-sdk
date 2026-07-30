# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module provides a Python interface to the Holoscan SDK logger.

.. autosummary::

    holoscan.pose_tree.Pose2
    holoscan.pose_tree.Pose3
    holoscan.pose_tree.PoseTree
    holoscan.pose_tree.PoseTreeAccessMethod
    holoscan.pose_tree.SO2
    holoscan.pose_tree.SO3
    holoscan.pose_tree.PoseTreeUCXClient
    holoscan.pose_tree.PoseTreeUCXClientConfig
    holoscan.pose_tree.PoseTreeUCXServer
    holoscan.pose_tree.PoseTreeUCXServerConfig
"""

import holoscan.core  # noqa: F401

from ._pose_tree import (
    SO2,
    SO3,
    Pose2,
    Pose3,
    PoseTree,
    PoseTreeAccessMethod,
    PoseTreeManager,
    PoseTreeUCXClient,
    PoseTreeUCXClientConfig,
    PoseTreeUCXServer,
    PoseTreeUCXServerConfig,
)

__all__ = [
    "Pose2",
    "Pose3",
    "PoseTree",
    "PoseTreeAccessMethod",
    "PoseTreeManager",
    "SO2",
    "SO3",
    "PoseTreeUCXClient",
    "PoseTreeUCXClientConfig",
    "PoseTreeUCXServer",
    "PoseTreeUCXServerConfig",
]

PoseTreeManager.__doc__ = """Manage a shared PoseTree instance as a FragmentService.

This resource creates and holds a ``holoscan.pose_tree.PoseTree`` instance, making it
accessible to multiple components within the same fragment. When used in a distributed
application, it can synchronize the pose tree across driver and worker processes.

Typical usage:

    # In Application.compose()
    pose_tree_manager = PoseTreeManager(self, name="pose_tree_manager")
    self.register_service(pose_tree_manager)

    # In Operator.__init__()
    self.pose_tree = self.service(PoseTreeManager, "pose_tree_manager").tree

Notes
-----
**Experimental Feature**. The Pose Tree API may change in future releases.
"""
