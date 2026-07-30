"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators.test_ops import PoseTreeManagerLookupOp
from holoscan.pose_tree import PoseTreeManager


class PoseTreeManagerServiceLookupApp(Application):
    def compose(self):
        pose_tree_manager = PoseTreeManager(self, name="pose_tree_manager")
        self.register_service(pose_tree_manager)

        lookup = PoseTreeManagerLookupOp(self, CountCondition(self, count=1), name="lookup")
        self.add_operator(lookup)


def test_pose_tree_manager_service_lookup_from_cpp_operator():
    app = PoseTreeManagerServiceLookupApp()
    app.run()
