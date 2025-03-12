/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

class SampleThreadPoolApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    // Define the tx, forward and rx operators, allowing the tx operator to execute 10 times
    auto tx1 = make_operator<ops::PingTxOp>("tx1", make_condition<CountCondition>(10));
    auto rx1 = make_operator<ops::PingRxOp>("rx1");

    auto tx2 = make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(15));
    auto rx2 = make_operator<ops::PingRxOp>("rx2");

    // Create a thread pool with two threads
    auto pool1 = make_thread_pool("pool1", 2);
    // can assign operators individually to this thread pool (setting pinning to true)
    pool1->add(tx1, true);
    pool1->add(rx1, true);

    // Create a second thread pool with two threads. We use two separate pools in this example
    // purely for demonstration purposes. In practice, all operators can typically be added to the
    // same thread pool. The one exception to this is that all operators in a thread pool using a
    // GPU-based allocator like like BlockMemoryPool, CudaStreamPool, RMMAllocator or
    // StreamOrderedAllocator must be using a common CUDA Device ID ("dev id" parameter). If
    // operators involving different devices exist, these should be assigned to separate thread
    // pools.
    auto pool2 = make_thread_pool("pool2", 2);
    // Assign multiple operators to the pool in a single call
    pool2->add({tx2, rx2}, true);

    // Define the workflow:  tx1 -> rx1 and tx2 -> rx2
    add_flow(tx1, rx1);
    add_flow(tx2, rx2);
  }
};

int main() {
  auto app = holoscan::make_application<SampleThreadPoolApp>();

  // The default greedy scheduler is single threaded, so the ThreadPool would not be utilised.
  // Instead we configure the app to use EventBasedScheduler.
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));
  app->run();

  return 0;
}
