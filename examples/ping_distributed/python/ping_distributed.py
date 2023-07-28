# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment
from holoscan.operators import PingRxOp, PingTxOp

# Now define a simple application using the operators defined above


class Fragment1(Fragment):
    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")

        # Add the operator (tx) to the fragment
        self.add_operator(tx)


class Fragment2(Fragment):
    def compose(self):
        rx = PingRxOp(self, name="rx")

        # Add the operator (rx) to the fragment
        self.add_operator(rx)


class MyPingApp(Application):
    def compose(self):
        fragment1 = Fragment1(self, name="fragment1")
        fragment2 = Fragment2(self, name="fragment2")

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(fragment1, fragment2, {("tx", "rx")})

        # self.resource(self.from_config("resources.fragments"))


if __name__ == "__main__":
    app = MyPingApp()
    app.run()
