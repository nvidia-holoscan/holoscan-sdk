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
#include "holoscan/pose_tree/pose_tree_manager.hpp"

#include <memory>
#include <stdexcept>

#include "holoscan/pose_tree/pose_tree.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_client.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_server.hpp"

namespace holoscan {

std::shared_ptr<Resource> PoseTreeManager::resource() const {
  return resource_.lock();
}

void PoseTreeManager::resource(const std::shared_ptr<Resource>& resource) {
  resource_ = resource;
}

void PoseTreeManager::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("PoseTreeManager is already initialized. Skipping...");
    return;
  }

  Resource::initialize();  // Call base class initialize

  const auto port = port_.get();
  if (port < 1024 || port > 65535) {
    throw std::out_of_range(
        fmt::format("Port must be in the range [1024, 65535], but got {}.", port));
  }

  pose_tree_instance_ = std::make_shared<PoseTree>();
  // Initialize the underlying PoseTree with configured parameters
  pose_tree_instance_->init(number_frames_.get(),
                            number_edges_.get(),
                            history_length_.get(),
                            default_number_edges_.get(),
                            default_history_length_.get(),
                            edges_chunk_size_.get(),
                            history_chunk_size_.get());
  HOLOSCAN_LOG_DEBUG("PoseTree (0x{:x}) initialized with the following parameters:",
                     reinterpret_cast<uintptr_t>(pose_tree_instance_.get()));
  HOLOSCAN_LOG_DEBUG("  number_frames: {}", number_frames_.get());
  HOLOSCAN_LOG_DEBUG("  number_edges: {}", number_edges_.get());
  HOLOSCAN_LOG_DEBUG("  history_length: {}", history_length_.get());
  HOLOSCAN_LOG_DEBUG("  default_number_edges: {}", default_number_edges_.get());
  HOLOSCAN_LOG_DEBUG("  default_history_length: {}", default_history_length_.get());
  HOLOSCAN_LOG_DEBUG("  edges_chunk_size: {}", edges_chunk_size_.get());
  HOLOSCAN_LOG_DEBUG("  history_chunk_size: {}", history_chunk_size_.get());
}

void PoseTreeManager::setup(holoscan::ComponentSpec& spec) {
  spec.param(port_,
             "port",
             "Port",
             "Port to listen on when using the PoseTreeManager as a distributed service",
             13337);
  spec.param(number_frames_,
             "number_frames",
             "Maximum Number of Frames",
             "Maximum number of frames to support.",
             1024);
  spec.param(number_edges_,
             "number_edges",
             "Maximum Number of Edges",
             "Maximum number of edges to support.",
             16384);
  spec.param(history_length_,
             "history_length",
             "Maximum History Length",
             "Maximum history length.",
             1048576);
  spec.param(default_number_edges_,
             "default_number_edges",
             "Default Number of Edges per Frame",
             "Default number of edges per frame.",
             16);
  spec.param(default_history_length_,
             "default_history_length",
             "Default History Length per Edge",
             "Default history length per edge.",
             1024);
  spec.param(edges_chunk_size_,
             "edges_chunk_size",
             "Edges Chunk Size",
             "Chunk size for edge allocation.",
             4);
  spec.param(history_chunk_size_,
             "history_chunk_size",
             "History Chunk Size",
             "Chunk size for history allocation.",
             64);

  spec.param(request_timeout_ms_,
             "request_timeout_ms",
             "Request Timeout (ms)",
             "Timeout in milliseconds for a single UCX request.",
             5000L);
  spec.param(request_poll_sleep_us_,
             "request_poll_sleep_us",
             "Request Poll Sleep (us)",
             "Microseconds to sleep while polling for a request completion.",
             10L);
  spec.param(worker_progress_sleep_us_,
             "worker_progress_sleep_us",
             "Worker Progress Loop Sleep (us)",
             "Microseconds to sleep in the main progress loop for client and server.",
             10L);
  spec.param(server_shutdown_timeout_ms_,
             "server_shutdown_timeout_ms",
             "Server Shutdown Timeout (ms)",
             "Timeout in milliseconds for server shutdown.",
             1000L);
  spec.param(server_shutdown_poll_sleep_ms_,
             "server_shutdown_poll_sleep_ms",
             "Server Shutdown Poll Sleep (ms)",
             "Milliseconds to sleep while polling during server shutdown.",
             10L);
  spec.param(maximum_clients_,
             "maximum_clients",
             "Maximum number of PoseTree clients",
             "Maximum number of PoseTree clients.",
             1024L);
}

std::shared_ptr<PoseTree> PoseTreeManager::tree() {
  return pose_tree_instance_;
}

std::shared_ptr<PoseTree> PoseTreeManager::tree() const {
  return pose_tree_instance_;
}

void PoseTreeManager::driver_start(std::string_view driver_ip) {
  auto result = driver_start_impl(driver_ip);
  if (!result) {
    HOLOSCAN_LOG_ERROR("driver_start failed: {}", error_to_str(result.error()));
  }
}

void PoseTreeManager::driver_shutdown() {
  auto result = driver_shutdown_impl();
  if (!result) {
    HOLOSCAN_LOG_ERROR("driver_shutdown failed: {}", error_to_str(result.error()));
  }
}

void PoseTreeManager::worker_connect(std::string_view driver_ip) {
  auto result = worker_connect_impl(driver_ip);
  if (!result) {
    HOLOSCAN_LOG_ERROR("worker_connect failed: {}", error_to_str(result.error()));
  }
}

void PoseTreeManager::worker_disconnect() {
  auto result = worker_disconnect_impl();
  if (!result) {
    HOLOSCAN_LOG_ERROR("worker_disconnect failed: {}", error_to_str(result.error()));
  }
}

PoseTreeManager::expected<void> PoseTreeManager::driver_start_impl(std::string_view driver_ip) {
  HOLOSCAN_LOG_TRACE("PoseTreeManager::driver_start_impl({}) called", driver_ip);

  if (!pose_tree_instance_) {
    return unexpected(Error::kNotInitialized);
  }

  if (!server_) {
    PoseTreeUCXServerConfig server_config;
    server_config.worker_progress_sleep_us = worker_progress_sleep_us_.get();
    server_config.shutdown_timeout_ms = server_shutdown_timeout_ms_.get();
    server_config.shutdown_poll_sleep_ms = server_shutdown_poll_sleep_ms_.get();
    server_config.maximum_clients = maximum_clients_.get();

    try {
      server_ = std::make_unique<PoseTreeUCXServer>(pose_tree_instance_, server_config);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to create PoseTreeUCXServer: {}", e.what());
      return unexpected(Error::kServerError);
    }

    auto result = server_->start(port_.get());
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to start PoseTreeUCXServer: {}",
                         PoseTreeUCXServer::error_to_str(result.error()));
      server_.reset();
      return unexpected(Error::kServerError);
    }

    HOLOSCAN_LOG_DEBUG("PoseTreeUCXServer started on port {}", port_.get());
  } else {
    HOLOSCAN_LOG_DEBUG("PoseTreeUCXServer already started");
    return unexpected(Error::kAlreadyStarted);
  }

  return {};
}

PoseTreeManager::expected<void> PoseTreeManager::driver_shutdown_impl() {
  HOLOSCAN_LOG_TRACE("PoseTreeManager::driver_shutdown_impl()");

  if (server_) {
    auto result = server_->stop();
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to stop PoseTreeUCXServer: {}",
                         PoseTreeUCXServer::error_to_str(result.error()));
      // Continue with cleanup even if stop had issues
    }
    server_.reset();
    HOLOSCAN_LOG_DEBUG("PoseTreeUCXServer stopped");
  } else {
    return unexpected(Error::kNotStarted);
  }

  return {};
}

PoseTreeManager::expected<void> PoseTreeManager::worker_connect_impl(std::string_view driver_ip) {
  HOLOSCAN_LOG_TRACE("PoseTreeManager::worker_connect_impl({})", driver_ip);

  if (!pose_tree_instance_) {
    return unexpected(Error::kNotInitialized);
  }

  if (!client_) {
    PoseTreeUCXClientConfig client_config;
    client_config.request_timeout_ms = request_timeout_ms_.get();
    client_config.request_poll_sleep_us = request_poll_sleep_us_.get();
    client_config.worker_progress_sleep_us = worker_progress_sleep_us_.get();

    try {
      client_ = std::make_unique<PoseTreeUCXClient>(pose_tree_instance_, client_config);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to create PoseTreeUCXClient: {}", e.what());
      return unexpected(Error::kClientError);
    }

    // The worker connects to the driver. Assuming driver_ip is the host.
    // The snapshot flag could be a parameter in the future.
    auto result = client_->connect(driver_ip, port_.get(), true);
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to connect PoseTreeUCXClient: {}",
                         PoseTreeUCXClient::error_to_str(result.error()));
      client_.reset();  // Clean up the client if connection failed
      return unexpected(Error::kClientError);
    }

    HOLOSCAN_LOG_DEBUG("PoseTreeUCXClient connected to {}:{}", driver_ip, port_.get());
  } else {
    return unexpected(Error::kAlreadyStarted);
  }

  return {};
}

PoseTreeManager::expected<void> PoseTreeManager::worker_disconnect_impl() {
  HOLOSCAN_LOG_TRACE("PoseTreeManager::worker_disconnect_impl()");

  if (client_) {
    auto result = client_->disconnect();
    if (!result) {
      HOLOSCAN_LOG_ERROR("Error disconnecting PoseTreeUCXClient: {}",
                         PoseTreeUCXClient::error_to_str(result.error()));
      // Continue with cleanup even if disconnect had issues
    } else {
      HOLOSCAN_LOG_DEBUG("PoseTreeUCXClient disconnected");
    }
    client_.reset();
  } else {
    return unexpected(Error::kNotStarted);
  }

  return {};
}

const char* PoseTreeManager::error_to_str(Error error) {
  switch (error) {
    case Error::kNotInitialized:
      return "PoseTreeManager not initialized";
    case Error::kServerError:
      return "Server operation failed";
    case Error::kClientError:
      return "Client operation failed";
    case Error::kAlreadyStarted:
      return "Service already started";
    case Error::kNotStarted:
      return "Service not started";
    case Error::kInternalError:
      return "Internal error";
    default:
      return "Unknown error";
  }
}

}  // namespace holoscan
