/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "work_queue.hpp"

#include <memory>

namespace holoscan {
namespace inference {

WorkQueue::WorkQueue(uint32_t threads) {
  threads_.resize(threads);
  for (uint32_t i = 0; i < threads; ++i) { add_thread(i); }
}

WorkQueue::~WorkQueue() {
  stop();
}

void WorkQueue::add_thread(int i) {
  auto f = [this, i]() {
    std::optional<std::unique_ptr<std::function<void()>>> func;

    func = queue_.pop();

    while (true) {
      while (func) {
        (**func)();
        if (done_) { return; }
        func = queue_.pop();
      }

      // queue is empty, wait
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this, &func]() {
        func = queue_.pop();
        return func || done_;
      });
      // if there is not function to execute then `done_` was true, exit.
      if (!func) {
        return;
      }
    }
  };
  threads_[i].reset(new std::thread(f));
}

void WorkQueue::stop() {
  if (done_) { return; }

  // signal the threads to finish
  done_ = true;

  // wake up all threads
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.notify_all();
  }

  // wait for the threads to finish
  for (size_t i = 0; i < threads_.size(); ++i) {
    if (threads_[i]->joinable()) { threads_[i]->join(); }
  }

  // clear the queue
  while (queue_.pop()) {}

  threads_.clear();
}

}  // namespace inference
}  // namespace holoscan
