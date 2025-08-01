/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/pose_tree/pose_tree_ucx_client.hpp"

#include <ucxx/api.h>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/logger/logger.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_common.hpp"

namespace holoscan {

template <typename T>
class ThreadSafeQueue {
 public:
  ThreadSafeQueue() = default;
  ThreadSafeQueue(const ThreadSafeQueue&) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

  void push(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
    cond_.notify_one();
  }

  void push(T&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cond_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  bool pop(T& value, std::chrono::microseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
      return false;  // timeout
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool try_pop(T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

struct PoseTreeUCXClient::ClientImpl {
  std::shared_ptr<ucxx::Context> context;
  std::shared_ptr<ucxx::Worker> worker;
  std::shared_ptr<ucxx::Endpoint> endpoint;
  std::shared_ptr<holoscan::PoseTree> pose_tree;
  ThreadSafeQueue<DeltaMessage> send_queue;
  std::atomic<bool> server_initiated_shutdown{false};

  // Maps remote frame IDs (from server) to local frame IDs
  std::unordered_map<holoscan::PoseTree::frame_t, holoscan::PoseTree::frame_t>
      remote_to_local_frame_id;
  std::mutex frame_id_map_mutex;

  // Store callback IDs for cleanup
  holoscan::PoseTree::uid_t create_frame_callback_id = 0;
  holoscan::PoseTree::uid_t set_edge_callback_id = 0;
};

void waitSingleRequest(std::shared_ptr<ucxx::Worker> worker, std::shared_ptr<ucxx::Request> request,
                       int64_t timeout_ms, int64_t poll_sleep_us) {
  auto start_time = std::chrono::steady_clock::now();
  while (!request->isCompleted()) {
    worker->progress();
    if (std::chrono::steady_clock::now() - start_time > std::chrono::milliseconds(timeout_ms)) {
      throw std::runtime_error("Request timed out");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(poll_sleep_us));
  }
  request->checkError();
}

PoseTreeUCXClient::PoseTreeUCXClient(std::shared_ptr<PoseTree> pose_tree,
                                     PoseTreeUCXClientConfig config)
    : impl_(std::make_unique<ClientImpl>()),
      pose_tree_(std::move(pose_tree)),
      config_(std::move(config)) {
  impl_->pose_tree = pose_tree_;
}
PoseTreeUCXClient::~PoseTreeUCXClient() {
  auto result = disconnect();
  if (!result) {
    try {
      HOLOSCAN_LOG_ERROR("PoseTreeUCXClient destructor: disconnect failed: {}",
                         error_to_str(result.error()));
    } catch (...) {
      // Suppress any exceptions from logging in destructor
    }
  }
}

PoseTreeUCXClient::expected<void> PoseTreeUCXClient::connect(std::string_view host, uint16_t port,
                                                             bool request_snapshot) {
  if (running_) {
    return unexpected(Error::kAlreadyConnected);
  }

  if (host.empty()) {
    return unexpected(Error::kInvalidArgument);
  }

  if (port == 0) {
    return unexpected(Error::kInvalidArgument);
  }

  host_ = host;
  port_ = port;
  request_snapshot_ = request_snapshot;
  running_ = true;
  ready_ = false;
  connect_failed_ = false;

  try {
    client_thread_ = std::thread(&PoseTreeUCXClient::run, this);
  } catch (const std::exception& e) {
    running_ = false;
    HOLOSCAN_LOG_ERROR("Failed to start client thread: {}", e.what());
    return unexpected(Error::kThreadError);
  }

  std::unique_lock<std::mutex> lk(ready_mutex_);
  ready_cv_.wait(lk, [this] { return ready_.load() || connect_failed_.load(); });

  if (connect_failed_) {
    if (client_thread_.joinable()) {
      client_thread_.join();
    }
    running_ = false;
    HOLOSCAN_LOG_ERROR("Unable to connect to PoseTreeUCXServer at {}:{}", host_, port_);
    return unexpected(Error::kConnectionFailed);
  }

  HOLOSCAN_LOG_DEBUG("PoseTreeUCXClient: connected to server ({}:{})", host_, port_);
  return {};
}

PoseTreeUCXClient::expected<void> PoseTreeUCXClient::disconnect() {
  if (!running_) {
    return {};  // Already disconnected, this is not an error
  }

  running_ = false;
  ready_cv_.notify_all();  // a safety measure to unblock a waiting connect()

  if (client_thread_.joinable()) {
    try {
      client_thread_.join();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to join client thread during disconnect: {}", e.what());
      return unexpected(Error::kShutdownError);
    }
  }

  return {};
}

const char* PoseTreeUCXClient::error_to_str(Error error) {
  switch (error) {
    case Error::kAlreadyConnected:
      return "Client is already connected";
    case Error::kInvalidArgument:
      return "Invalid argument provided";
    case Error::kConnectionFailed:
      return "Failed to connect to server";
    case Error::kNotConnected:
      return "Client is not connected";
    case Error::kThreadError:
      return "Thread operation failed";
    case Error::kShutdownError:
      return "Error during shutdown";
    case Error::kInternalError:
      return "Internal error";
    default:
      return "Unknown error";
  }
}

void PoseTreeUCXClient::run() {
  try {
    impl_->context = ucxx::createContext({}, UCP_FEATURE_AM);
    impl_->worker = impl_->context->createWorker();

    // Register AM receiver callbacks
    ucxx::AmReceiverCallbackInfo delta_callback_info("AMClient", MSG_DELTA);
    impl_->worker->registerAmReceiverCallback(
        delta_callback_info, [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
          HOLOSCAN_LOG_TRACE("PoseTreeUCXClient: Received delta message");
          if (!running_)
            return;
          try {
            req->checkError();
            auto buffer = req->getRecvBuffer();
            auto data = static_cast<const uint8_t*>(buffer->data());
            auto size = buffer->getSize();
            if (size == sizeof(DeltaMessage)) {
              DeltaMessage delta_msg;
              std::memcpy(&delta_msg, data, sizeof(DeltaMessage));
              switch (delta_msg.delta_type) {
                case DELTA_FRAME_CREATED: {
                  is_external_pose_tree_update_ = true;
                  auto result =
                      impl_->pose_tree->find_or_create_frame(delta_msg.data.frame_data.name);
                  if (result) {
                    if (result.value() != delta_msg.data.frame_data.frame_id) {
                      HOLOSCAN_LOG_DEBUG("Frame ID mismatch, mapping remote {} to local {}",
                                         delta_msg.data.frame_data.frame_id,
                                         result.value());
                    }
                    std::lock_guard<std::mutex> lock(impl_->frame_id_map_mutex);
                    impl_->remote_to_local_frame_id[delta_msg.data.frame_data.frame_id] =
                        result.value();
                  }
                  is_external_pose_tree_update_ = false;
                  break;
                }
                case DELTA_EDGE_SET: {
                  is_external_pose_tree_update_ = true;
                  holoscan::Pose3d pose = deserialize_pose3d(delta_msg.data.edge_data);

                  holoscan::PoseTree::frame_t local_lhs, local_rhs;
                  {
                    std::lock_guard<std::mutex> lock(impl_->frame_id_map_mutex);
                    auto lhs_it =
                        impl_->remote_to_local_frame_id.find(delta_msg.data.edge_data.lhs_frame);
                    auto rhs_it =
                        impl_->remote_to_local_frame_id.find(delta_msg.data.edge_data.rhs_frame);

                    if (lhs_it == impl_->remote_to_local_frame_id.end() ||
                        rhs_it == impl_->remote_to_local_frame_id.end()) {
                      HOLOSCAN_LOG_WARN(
                          "PoseTreeUCXClient: Could not find mapping for edge frames {} -> {}",
                          delta_msg.data.edge_data.lhs_frame,
                          delta_msg.data.edge_data.rhs_frame);
                      is_external_pose_tree_update_ = false;
                      break;
                    }
                    local_lhs = lhs_it->second;
                    local_rhs = rhs_it->second;
                  }

                  impl_->pose_tree->set(local_lhs, local_rhs, delta_msg.data.edge_data.time, pose);
                  is_external_pose_tree_update_ = false;
                  break;
                }
              }
            }
          } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Error in delta callback: {}", e.what());
            running_ = false;
          }
        });
    ucxx::AmReceiverCallbackInfo snapshot_callback_info("AMClient", MSG_SNAPSHOT_DATA);
    impl_->worker->registerAmReceiverCallback(
        snapshot_callback_info, [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
          HOLOSCAN_LOG_TRACE("PoseTreeUCXClient: Received snapshot message");
          if (!running_)
            return;
          try {
            req->checkError();
            auto buffer = req->getRecvBuffer();
            std::vector<FrameInfo> frames;
            std::vector<EdgeData> edges;
            deserialize_snapshot(
                static_cast<const uint8_t*>(buffer->data()), buffer->getSize(), frames, edges);
            HOLOSCAN_LOG_TRACE("PoseTreeUCXClient: Received snapshot message: frames.size() = {}",
                               frames.size());
            is_external_pose_tree_update_ = true;

            {
              std::lock_guard<std::mutex> lock(impl_->frame_id_map_mutex);
              impl_->remote_to_local_frame_id.clear();
            }

            for (const auto& frame : frames) {
              auto result = impl_->pose_tree->find_or_create_frame(frame.name);
              if (result) {
                std::lock_guard<std::mutex> lock(impl_->frame_id_map_mutex);
                impl_->remote_to_local_frame_id[frame.frame_id] = result.value();
              }
            }
            for (const auto& edge : edges) {
              holoscan::Pose3d pose = deserialize_pose3d(edge);

              holoscan::PoseTree::frame_t local_lhs, local_rhs;
              {
                std::lock_guard<std::mutex> lock(impl_->frame_id_map_mutex);
                auto lhs_it = impl_->remote_to_local_frame_id.find(edge.lhs_frame);
                auto rhs_it = impl_->remote_to_local_frame_id.find(edge.rhs_frame);

                if (lhs_it == impl_->remote_to_local_frame_id.end() ||
                    rhs_it == impl_->remote_to_local_frame_id.end()) {
                  HOLOSCAN_LOG_WARN(
                      "PoseTreeUCXClient: Could not find mapping for edge frames {} -> {} in "
                      "snapshot",
                      edge.lhs_frame,
                      edge.rhs_frame);
                  continue;
                }
                local_lhs = lhs_it->second;
                local_rhs = rhs_it->second;
              }
              impl_->pose_tree->set(local_lhs, local_rhs, edge.time, pose);
            }
            is_external_pose_tree_update_ = false;
          } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Error in snapshot callback: {}", e.what());
            running_ = false;
          }
        });

    ucxx::AmReceiverCallbackInfo close_callback_info("AMClient", MSG_CLOSE);
    impl_->worker->registerAmReceiverCallback(
        close_callback_info, [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
          if (!running_)
            return;
          try {
            req->checkError();
            HOLOSCAN_LOG_DEBUG("PoseTreeUCXClient: Received shutdown from server.");
            impl_->server_initiated_shutdown = true;
            running_ = false;
          } catch (const std::exception&) {
            if (running_) {
              running_ = false;
            }
          }
        });

    impl_->endpoint = impl_->worker->createEndpointFromHostname(host_, port_, true);

    SubscribeMessage subscribe_msg{};
    subscribe_msg.request_snapshot = request_snapshot_ ? 1 : 0;
    ucxx::AmReceiverCallbackInfo subscribe_info("AMServer", MSG_SUBSCRIBE);
    auto subscribe_request = impl_->endpoint->amSend(
        &subscribe_msg, sizeof(subscribe_msg), UCS_MEMORY_TYPE_HOST, subscribe_info);
    waitSingleRequest(impl_->worker,
                      subscribe_request,
                      config_.request_timeout_ms,
                      config_.request_poll_sleep_us);

    HOLOSCAN_LOG_TRACE("PoseTreeUCXClient: Adding frame create callback: pose_tree_ = {}",
                       reinterpret_cast<uint64_t>(impl_->pose_tree.get()));

    // Store callback ID for cleanup
    auto create_frame_result = impl_->pose_tree->add_create_frame_callback([&](const holoscan::
                                                                                   PoseTree::frame_t
                                                                                       frame_id) {
      HOLOSCAN_LOG_TRACE(
          "PoseTreeUCXClient: Received frame create callback: is_external_pose_tree_update_ = {} ",
          is_external_pose_tree_update_.load());
      if (!is_external_pose_tree_update_) {
        auto frame_name = impl_->pose_tree->get_frame_name(frame_id).value();
        // Use the helper method for safe initialization
        DeltaMessage frame_msg = create_pose_tree_frame_delta(frame_id, frame_name.data());
        impl_->send_queue.push(frame_msg);
      }
    });

    if (create_frame_result) {
      impl_->create_frame_callback_id = create_frame_result.value();
    } else {
      HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Failed to register create frame callback");
    }

    // Store callback ID for cleanup
    auto set_edge_result =
        impl_->pose_tree->add_set_edge_callback([&](holoscan::PoseTree::frame_t lhs,
                                                    holoscan::PoseTree::frame_t rhs,
                                                    double time,
                                                    const holoscan::Pose3d& lhs_T_rhs) {
          HOLOSCAN_LOG_TRACE(
              "PoseTreeUCXClient: Received edge set callback: is_external_pose_tree_update_ = {}",
              is_external_pose_tree_update_.load());
          if (!is_external_pose_tree_update_) {
            // Create EdgeData first, then use helper method
            EdgeData edge_data;
            edge_data.lhs_frame = lhs;
            edge_data.rhs_frame = rhs;
            edge_data.time = time;
            serialize_pose3d(lhs_T_rhs, edge_data);

            DeltaMessage edge_msg = create_pose_tree_edge_delta(edge_data);
            impl_->send_queue.push(edge_msg);
          }
        });

    if (set_edge_result) {
      impl_->set_edge_callback_id = set_edge_result.value();
    } else {
      HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Failed to register set edge callback");
    }

    {
      std::lock_guard<std::mutex> lk(ready_mutex_);
      ready_ = true;
    }
    ready_cv_.notify_one();

    while (running_) {
      DeltaMessage msg;
      while (
          impl_->send_queue.pop(msg, std::chrono::microseconds(config_.worker_progress_sleep_us))) {
        ucxx::AmReceiverCallbackInfo delta_info("AMServer", MSG_DELTA);
        auto send_request =
            impl_->endpoint->amSend(&msg, sizeof(msg), UCS_MEMORY_TYPE_HOST, delta_info);
        waitSingleRequest(
            impl_->worker, send_request, config_.request_timeout_ms, config_.request_poll_sleep_us);
      }
      impl_->worker->progress();  // always keep UCX moving
    }
  } catch (const std::exception& e) {
    if (running_) {
      HOLOSCAN_LOG_WARN("Exception in PoseTreeUCXClient thread, shutting down: {}", e.what());
    }
    {
      std::lock_guard<std::mutex> lk(ready_mutex_);
      connect_failed_ = true;
    }
    ready_cv_.notify_one();
    running_ = false;
  }

  // Cleanup
  // Deregister callbacks before destroying the client
  if (impl_ && impl_->pose_tree) {
    if (impl_->create_frame_callback_id != 0) {
      auto result = impl_->pose_tree->remove_create_frame_callback(impl_->create_frame_callback_id);
      if (!result) {
        HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Failed to remove create frame callback: {}",
                           PoseTree::error_to_str(result.error()));
      }
    }
    if (impl_->set_edge_callback_id != 0) {
      auto result = impl_->pose_tree->remove_set_edge_callback(impl_->set_edge_callback_id);
      if (!result) {
        HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Failed to remove set edge callback: {}",
                           PoseTree::error_to_str(result.error()));
      }
    }
  }

  try {
    if (impl_ && impl_->endpoint && !impl_->server_initiated_shutdown) {
      ucxx::AmReceiverCallbackInfo close_info("AMServer", MSG_CLOSE);
      auto close_request = impl_->endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST, close_info);
      waitSingleRequest(
          impl_->worker, close_request, config_.request_timeout_ms, config_.request_poll_sleep_us);
    }
  } catch (const std::exception& e) {
    // This can happen if the server is already down, log and ignore
    HOLOSCAN_LOG_ERROR("PoseTreeUCXClient: Unable to send close message to server: {}", e.what());
  }

  impl_->endpoint.reset();
  impl_->worker.reset();
  impl_->context.reset();
}

}  // namespace holoscan
