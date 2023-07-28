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

import threading

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp

# Now define a simple application using the operators defined above


class MyPingApp(Application):
    def __init__(self):
        self.target_op = None
        super().__init__()

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = PingRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)

        # Save a reference to the tx operator so we can access it later
        self.target_op = tx


if __name__ == "__main__":
    app = MyPingApp()
    future = app.run_async()
    print("# Application is running asynchronously.")
    # Executing `future.result()` will block until the application is done

    def print_status():
        if future.done():
            print("# Application finished")
            return
        else:
            print(
                "# Application still running... PingTxOp index: "
                f"{app.target_op.index if app.target_op else 'N/A'}"
            )
        # start a new thread to print status and start it
        threading.Thread(target=print_status).start()

    print_status()  # print status while application is running

    future.result()
    print("# Application has finished running.")
