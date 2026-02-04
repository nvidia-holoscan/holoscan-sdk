"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
