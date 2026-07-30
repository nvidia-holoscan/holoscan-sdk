/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  std::thread thread_to_join;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ && !thread_.joinable())
      return;
    running_ = false;
    if (thread_.joinable()) {
      thread_to_join = std::move(thread_);
    }
  }
  cv_.notify_all();
  if (thread_to_join.joinable()) {
    thread_to_join.join();
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
