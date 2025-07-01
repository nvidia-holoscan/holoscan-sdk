# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module provides a Python interface to the Holoscan SDK logger.

.. autosummary::

    holoscan.pose_tree.Pose2
    holoscan.pose_tree.Pose3
    holoscan.pose_tree.PoseTree
    holoscan.pose_tree.PoseTreeAccessMethod
    holoscan.pose_tree.SO2
    holoscan.pose_tree.SO3
"""

from ._pose_tree import (
    SO2,
    SO3,
    Pose2,
    Pose3,
    PoseTree,
    PoseTreeAccessMethod,
)

__all__ = [
    "Pose2",
    "Pose3",
    "PoseTree",
    "PoseTreeAccessMethod",
    "PoseTreeManager",
    "SO2",
    "SO3",
]

from holoscan.core import FragmentService, Resource


class PoseTreeManager(Resource, FragmentService):
    """Manage a shared PoseTree instance as a FragmentService.

    This resource creates and holds a `holoscan.pose_tree.PoseTree` instance, making it
    accessible to multiple components (like operators) within the same fragment. It simplifies
    the management of pose data by providing a centralized, configurable point of access.

    To use it, register an instance of `PoseTreeManager` with the fragment in your
    application's `compose` method:

        # In Application.compose()
        pose_tree_manager = PoseTreeManager(
            self, name="pose_tree", **self.kwargs("my_pose_tree_config")
        )
        self.register_service(pose_tree_manager)

    Then, operators can access the underlying `PoseTree` instance via the `service()` method:

        # In Operator.__init__()
        self.pose_tree = self.service(PoseTreeManager, "pose_tree").tree

    The parameters for the underlying `PoseTree` can be configured via the application's YAML
    configuration file or directly when creating the resource.

    Parameters
    ----------
    fragment : holoscan.core.Fragment
        The fragment that the resource belongs to.
    number_frames : int, optional
        Maximum number of coordinate frames to support. Defaults to 1024.
    number_edges : int, optional
        Maximum number of edges (direct transformations) to support. Defaults to 16384.
    history_length : int, optional
        Total capacity for storing historical pose data across all edges. Defaults to 1048576.
    default_number_edges : int, optional
        Default number of edges allocated per new frame. Defaults to 16.
    default_history_length : int, optional
        Default history capacity allocated per new edge. Defaults to 1024.
    edges_chunk_size : int, optional
        Allocation chunk size for a frame's edge list. Defaults to 4.
    history_chunk_size : int, optional
        Allocation chunk size for an edge's history buffer. Defaults to 64.
    name : str, optional
        The name of the resource.

    Notes
    -----
    **Experimental Feature**
    The Pose Tree feature, including this manager, is experimental. The API may change in
    future releases.
    """

    def __init__(
        self,
        fragment,
        *args,
        number_frames=1024,
        number_edges=16384,
        history_length=1048576,
        default_number_edges=16,
        default_history_length=1024,
        edges_chunk_size=4,
        history_chunk_size=64,
        **kwargs,
    ):
        Resource.__init__(self, fragment, *args, **kwargs)
        FragmentService.__init__(self)
        self._tree = PoseTree()
        self._tree.init(
            number_frames=number_frames,
            number_edges=number_edges,
            history_length=history_length,
            default_number_edges=default_number_edges,
            default_history_length=default_history_length,
            edges_chunk_size=edges_chunk_size,
            history_chunk_size=history_chunk_size,
        )

    # FragmentService interface – return ourself as the resource
    def resource(self, _new_resource=None):
        return self

    @property
    def tree(self):
        """Get the managed `PoseTree` instance.

        This is the primary method for accessing the pose tree from other components.

        Returns
        -------
        holoscan.pose_tree.PoseTree
            The underlying pose tree instance managed by this resource.
        """
        return self._tree
