/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "work_queue.hpp"

#include <memory>

namespace holoscan {
namespace inference {

WorkQueue::WorkQueue(uint32_t threads) {
  threads_.resize(threads);
  for (uint32_t i = 0; i < threads; ++i) {
    add_thread(i);
  }
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
        // `done_` is std::atomic<bool>; explicit relaxed load documents intent
        // (the mutex-guarded condition_variable below provides the wake-up ordering).
        if (done_.load(std::memory_order_relaxed)) {
          return;
        }
        func = queue_.pop();
      }

      // queue is empty, wait
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this, &func]() {
        func = queue_.pop();
        return func || done_.load(std::memory_order_relaxed);
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
  // Single-executor guard: the check-then-set pattern (separate load + store) let two
  // concurrent stop() callers both observe `done_ == false`, both transition it to true,
  // and both race into the teardown block below — double-joining threads_ / double-clearing
  // the container. A single atomic RMW makes exactly one caller the winner; losing callers
  // observe the old value `true` and return without touching threads_.
  if (done_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  // wake up all threads
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.notify_all();
  }

  // wait for the threads to finish
  for (size_t i = 0; i < threads_.size(); ++i) {
    if (threads_[i]->joinable()) {
      threads_[i]->join();
    }
  }

  // clear the queue
  while (queue_.pop()) {
  }

  threads_.clear();
}

}  // namespace inference
}  // namespace holoscan
