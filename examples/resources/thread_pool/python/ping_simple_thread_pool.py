"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.schedulers import EventBasedScheduler


class SampleThreadPoolApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        tx1 = PingTxOp(self, CountCondition(self, 10), name="tx1")
        tx2 = PingTxOp(self, CountCondition(self, 15), name="tx2")
        rx1 = PingRxOp(self, name="rx1")
        rx2 = PingRxOp(self, name="rx2")

        # Create a thread pool with two threads and pin two operators to these threads.
        pool1 = self.make_thread_pool("pool1", 2)
        pool1.add(tx1, True)
        pool1.add(rx1, True)

        # Create a second thread pool with two threads. We use two separate pools in this example
        # purely for demonstration purposes. In practice, all operators can typically be added to
        # the same thread pool. The one exception to this is that all operators in a thread pool
        # using a GPU-based allocator like like BlockMemoryPool, CudaStreamPool, RMMAllocator or
        # StreamOrderedAllocator must be using a common CUDA Device ID ("dev id" parameter). If
        # operators involving different devices exist, these should be assigned to separate thread
        # pools.
        pool2 = self.make_thread_pool("pool2", 2)
        pool2.add([tx2, rx2], True)

        # Define the workflow:  tx1 -> rx1 and tx2 -> rx2
        self.add_flow(tx1, rx1)
        self.add_flow(tx2, rx2)


def main():
    app = SampleThreadPoolApp()
    scheduler = EventBasedScheduler(app, worker_thread_number=3, name="ebs")
    app.scheduler(scheduler)
    app.run()


if __name__ == "__main__":
    main()
