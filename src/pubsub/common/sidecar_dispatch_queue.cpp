/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/pubsub/common/sidecar_dispatch_queue.hpp>

#include <utility>
#include <vector>

namespace holoscan {

SidecarDispatchQueue::SidecarDispatchQueue(CallbackProvider get_callback)
    : get_callback_(std::move(get_callback)) {}

SidecarDispatchQueue::~SidecarDispatchQueue() {
  stop();
}

void SidecarDispatchQueue::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_)
    return;
  running_ = true;
  thread_ = std::thread([this]() { dispatch_loop(); });
}

void SidecarDispatchQueue::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ && !thread_.joinable())
      return;
    running_ = false;
  }
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.clear();
}

void SidecarDispatchQueue::enqueue(nvidia::gxf::Gid publisher_gid, std::vector<uint8_t>&& payload,
                                   nvidia::gxf::MessageMetadata metadata) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(Item{publisher_gid, std::move(payload), std::move(metadata)});
  }
  cv_.notify_one();
}

void SidecarDispatchQueue::dispatch_loop() {
  while (true) {
    Item item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return !running_ || !queue_.empty(); });
      if (!running_ && queue_.empty())
        return;
      item = std::move(queue_.front());
      queue_.pop_front();
    }
    auto callback = get_callback_();
    if (callback) {
      callback(item.publisher_gid, std::move(item.payload), item.metadata);
    }
  }
}

}  // namespace holoscan
