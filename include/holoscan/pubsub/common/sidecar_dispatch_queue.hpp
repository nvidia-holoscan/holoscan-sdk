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

#ifndef PUBSUB_COMMON_INCLUDE_PUBSUB_SIDECAR_DISPATCH_QUEUE_HPP
#define PUBSUB_COMMON_INCLUDE_PUBSUB_SIDECAR_DISPATCH_QUEUE_HPP

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <gxf/pubsub/gid.hpp>
#include <gxf/pubsub/pubsub_transport.hpp>

namespace holoscan {

/// Thread-safe queue + dispatch thread for deferred native-descriptor sidecar receives.
///
/// Both the DDS and Zenoh transports receive native-descriptor messages on a
/// transport callback thread that must not block. This queue decouples receive
/// from dispatch: the transport callback enqueues items, and a dedicated thread
/// dequeues and delivers them to the PubSubContext receive callback.
///
/// Usage:
///   // In transport constructor:
///   sidecar_queue_ = std::make_unique<SidecarDispatchQueue>([this]() {
///     std::lock_guard<std::mutex> lock(my_mutex_);
///     return receive_callback_;
///   });
///
///   // When subscribing:
///   sidecar_queue_->start();
///
///   // On transport callback thread:
///   sidecar_queue_->enqueue(gid, std::move(payload), metadata);
///
///   // On shutdown:
///   sidecar_queue_->stop();
class SidecarDispatchQueue {
 public:
  using ReceiveCallback = nvidia::gxf::PubSubTransport::ReceiveCallback;

  /// Function that returns the current receive callback (thread-safe copy).
  /// Called on the dispatch thread before each delivery.
  using CallbackProvider = std::function<ReceiveCallback()>;

  explicit SidecarDispatchQueue(CallbackProvider get_callback);
  ~SidecarDispatchQueue();

  SidecarDispatchQueue(const SidecarDispatchQueue&) = delete;
  SidecarDispatchQueue& operator=(const SidecarDispatchQueue&) = delete;

  /// Start the dispatch thread. No-op if already running.
  void start();

  /// Stop the dispatch thread and drain the queue.
  void stop();

  /// Enqueue a received sidecar message for dispatch. Thread-safe.
  void enqueue(nvidia::gxf::Gid publisher_gid, std::vector<uint8_t>&& payload,
               nvidia::gxf::MessageMetadata metadata);

 private:
  void dispatch_loop();

  struct Item {
    nvidia::gxf::Gid publisher_gid;
    std::vector<uint8_t> payload;
    nvidia::gxf::MessageMetadata metadata;
  };

  CallbackProvider get_callback_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  bool running_ = false;
  std::thread thread_;
};

}  // namespace holoscan

#endif /* PUBSUB_COMMON_INCLUDE_PUBSUB_SIDECAR_DISPATCH_QUEUE_HPP */
