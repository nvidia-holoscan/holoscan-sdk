/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/services/app_worker/server.hpp"

#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/app_worker.hpp"
#include "holoscan/core/cli_options.hpp"
#include "holoscan/core/services/common/network_constants.hpp"
#include "holoscan/core/system/network_utils.hpp"
#include "holoscan/core/system/system_resource_manager.hpp"
#include "holoscan/logger/logger.hpp"

#include "../app_driver/client.hpp"
#include "service_impl.hpp"

namespace holoscan::service {

AppWorkerServer::AppWorkerServer(holoscan::AppWorker* app_worker) : app_worker_(app_worker) {}

AppWorkerServer::~AppWorkerServer() = default;

void AppWorkerServer::start() {
  server_thread_ = std::make_unique<std::thread>(&AppWorkerServer::run, this);
}

void AppWorkerServer::stop() {
  should_stop_ = true;
  cv_.notify_all();

  if (fragment_executors_future_.valid()) { fragment_executors_future_.wait(); }
}

void AppWorkerServer::wait() {
  std::unique_lock<std::mutex> lock(join_mutex_);
  if (server_thread_ && server_thread_->joinable()) { server_thread_->join(); }
}

void AppWorkerServer::notify() {
  cv_.notify_all();
}

bool AppWorkerServer::connect_to_driver(int32_t max_connection_retry_count,
                                        int32_t connection_attempt_interval_ms) {
  const auto& driver_address = app_worker_->options()->driver_address;

  driver_client_ = std::make_unique<AppDriverClient>(
      driver_address, grpc::CreateChannel(driver_address, grpc::InsecureChannelCredentials()));

  // Get the target fragments from CLI Option
  const auto& target_fragments = app_worker_->target_fragments();

  // Get current system resource information
  SystemResourceManager system_resource_manager;
  auto cpuinfo = system_resource_manager.cpu_monitor()->update(
      CPUMetricFlag::AVAILABLE_PROCESSOR_COUNT | CPUMetricFlag::MEMORY_USAGE |
      CPUMetricFlag::SHARED_MEMORY_USAGE);
  auto gpuinfo = system_resource_manager.gpu_monitor()->gpu_info(GPUMetricFlag::MEMORY_USAGE);

  const auto& worker_address = app_worker_->options()->worker_address;
  auto worker_port = CLIOptions::parse_port(worker_address);

  // Connect to driver and send fragment allocation request
  bool connection_result = false;
  int attempt_count = 0;
  while (attempt_count < max_connection_retry_count && should_stop_ == false) {
    connection_result =
        driver_client_->fragment_allocation(worker_port, target_fragments, cpuinfo, gpuinfo);
    if (connection_result) {
      break;
    } else {
      attempt_count++;
      HOLOSCAN_LOG_ERROR("Failed to connect to driver at {} (trial {}/{})",
                         driver_address,
                         attempt_count,
                         max_connection_retry_count);
      std::this_thread::sleep_for(std::chrono::milliseconds(connection_attempt_interval_ms));
    }
  }
  return connection_result;
}

void AppWorkerServer::run() {
  // Get IP address and port
  auto& server_address = app_worker_->options()->worker_address;

  // Set default IP address and port if not specified
  auto split = server_address.find(':');
  if (split == std::string::npos) {
    std::string ip;
    if (server_address.empty() || server_address == "") {
      ip = "0.0.0.0";
    } else {
      ip = server_address;
    }
    auto unused_ports = get_unused_network_ports(1, kMinNetworkPort, kMaxNetworkPort);
    if (unused_ports.empty()) {
      HOLOSCAN_LOG_ERROR("No unused ports found");
      return;
    }
    int32_t unused_port = unused_ports[0];
    server_address.assign(fmt::format("{}:{}", ip, unused_port));
  } else if (split == 0) {
    std::string port = server_address.substr(1);
    server_address.assign(fmt::format("0.0.0.0:{}", port));
  }

  AppWorkerServiceImpl app_worker_service(app_worker_);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&app_worker_service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (server) {
    HOLOSCAN_LOG_INFO("AppWorkerServer listening on {}", server_address);
  } else {
    HOLOSCAN_LOG_ERROR("Failed to start server on {}", server_address);
    return;
  }

  // Wait until we should stop the server
  std::unique_lock<std::mutex> lock(mutex_);
  while (!should_stop_) {
    cv_.wait(lock);
    // Process message queue if there is any message
    app_worker_->process_message_queue();
  }
  server->Shutdown();
}

std::shared_future<void>& AppWorkerServer::fragment_executors_future() {
  return fragment_executors_future_;
}

void AppWorkerServer::fragment_executors_future(std::future<void>& future) {
  fragment_executors_future_ = future.share();
}

void AppWorkerServer::notify_worker_execution_finished(holoscan::AppWorkerTerminationCode code) {
  auto& server_address = app_worker_->options()->worker_address;

  driver_client_->worker_execution_finished(CLIOptions::parse_port(server_address), code);
}

}  // namespace holoscan::service
