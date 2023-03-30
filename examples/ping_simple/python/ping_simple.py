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
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


if __name__ == "__main__":
    app = MyPingApp()
    app.run()
