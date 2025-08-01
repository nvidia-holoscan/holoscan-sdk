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
#ifndef MODULES_HOLOINFER_SRC_UTILS_WORK_QUEUE_HPP
#define MODULES_HOLOINFER_SRC_UTILS_WORK_QUEUE_HPP

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

/**
 * A thread safe queue
 *
 * @tparam T type of object contained in the queue
 */
template <typename T>
class ThreadSafeQueue {
 public:
  /**
   * Add an object to the queue
   *
   * @param value object to add
   */
  void push(T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
  }

  /**
   * Get an object of the queue
   *
   * @return std::optional<T> object that had been unqueued or null if queue was empty
   */
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return {};
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return std::optional<T>(std::move(item));
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
};

/**
 * The WorkQueue class maintains a pool of threads which execute jobs added with the `async()`
 * function.
 */
class WorkQueue {
 public:
  /**
   * Construct a new work queue
   *
   * @param threads number of worker threads
   */
  explicit WorkQueue(uint32_t threads);
  WorkQueue() = delete;

  /**
   * Destroy the work queue
   */
  ~WorkQueue();

  /**
   * Enqueue a function to be executed asynchronously by a thread of the pool
   *
   * @tparam F function type
   * @tparam Args argument types
   * @param f function (can be any callable object)
   * @param args function arguments
   * @return std::shared_pointer with std::packed_task
   */
  template <class F, class... Args>
  auto async(F&& f, Args&&... args) -> std::shared_ptr<
      std::packaged_task<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>()>> {
    auto packed_task = std::make_shared<
        std::packaged_task<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto function = std::make_unique<std::function<void()>>([packed_task]() { (*packed_task)(); });
    queue_.push(function);

    std::unique_lock<std::mutex> lock(mutex_);
    condition_.notify_one();
    return packed_task;
  }

  /**
   * Stop all threads.
   */
  void stop();

 private:
  void add_thread(int i);

  std::vector<std::unique_ptr<std::thread>> threads_;
  ThreadSafeQueue<std::unique_ptr<std::function<void()>>> queue_;
  std::atomic<bool> done_ = false;
  std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace inference
}  // namespace holoscan

#endif /* MODULES_HOLOINFER_SRC_UTILS_WORK_QUEUE_HPP */
