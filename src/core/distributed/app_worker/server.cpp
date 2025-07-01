/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/distributed/app_worker/server.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../app_driver/client.hpp"
#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/app_worker.hpp"
#include "holoscan/core/cli_options.hpp"
#include "holoscan/core/distributed/common/network_constants.hpp"
#include "holoscan/core/system/network_utils.hpp"
#include "holoscan/core/system/system_resource_manager.hpp"
#include "holoscan/logger/logger.hpp"

#include "service_impl.hpp"

namespace holoscan::distributed {

AppWorkerServer::AppWorkerServer(holoscan::AppWorker* app_worker, bool need_health_check)
    : app_worker_(app_worker), need_health_check_(need_health_check) {}

AppWorkerServer::~AppWorkerServer() = default;

constexpr int kFragmentExecutorsShutdownTimeoutSeconds = 6;

void AppWorkerServer::start() {
  // Update server address
  auto& server_address = app_worker_->options()->worker_address;

  // Parse the server address using the parse_address method.
  auto [server_ip, server_port] = CLIOptions::parse_address(
      server_address,
      "0.0.0.0",  // default IP address
      "");        // We will determine the default port using get_unused_network_ports

  // If server_port is empty, find an unused network port and use it as the default port.
  if (server_port.empty()) {
    // Get the driver port to exclude it from the list of unused ports.
    std::string driver_address = app_worker_->options()->driver_address;
    auto driver_port_str = holoscan::CLIOptions::parse_port(driver_address);
    auto driver_port = kDefaultAppDriverPort;
    if (!driver_port_str.empty()) { driver_port = std::stoi(driver_port_str); }
    const std::vector<int> exclude_ports = {driver_port};
    auto unused_ports =
        get_unused_network_ports(1, kMinNetworkPort, kMaxNetworkPort, exclude_ports);
    if (unused_ports.empty()) {
      HOLOSCAN_LOG_ERROR("No unused ports found");
      return;
    }
    server_port = std::to_string(unused_ports[0]);
  }

  // Enclose IPv6 address in brackets if port is not empty and it's an IPv6 address.
  if (server_ip.find(':') != std::string::npos && !server_port.empty()) {
    server_ip = fmt::format("[{}]", server_ip);
  }

  // Reassemble server_address using the parsed IP and port.
  server_address = fmt::format("{}:{}", server_ip, server_port);

  // Set HOLOSCAN_UCX_SOURCE_ADDRESS environment variable if the `--worker-address` CLI option is
  // specified with a value other than '0.0.0.0' (default) for IPv4 or '::' for IPv6.
  // (See issue 4233845)
  if (server_ip != "0.0.0.0" && server_ip != "::") {
    const char* source_address = std::getenv("HOLOSCAN_UCX_SOURCE_ADDRESS");
    if (source_address == nullptr || source_address[0] == '\0') {
      auto associated_local_ip = get_associated_local_ip(server_ip);
      if (!associated_local_ip.empty()) {
        HOLOSCAN_LOG_DEBUG("Setting HOLOSCAN_UCX_SOURCE_ADDRESS to {}", associated_local_ip);
        setenv("HOLOSCAN_UCX_SOURCE_ADDRESS", server_ip.c_str(), 0);
      }
    }
  }

  server_thread_ = std::make_unique<std::thread>(&AppWorkerServer::run, this);
}

void AppWorkerServer::stop() {
  should_stop_ = true;
  cv_.notify_all();

  if (fragment_executors_future_.valid()) {
    // Add timeout to avoid potential deadlock
    auto status = fragment_executors_future_.wait_for(
        std::chrono::seconds(kFragmentExecutorsShutdownTimeoutSeconds));
    if (status == std::future_status::timeout) {
      HOLOSCAN_LOG_WARN("Timeout waiting for fragment executors to complete");
    }
  }
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

  driver_client_ = std::make_shared<AppDriverClient>(
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
  auto [worker_ip, worker_port] = CLIOptions::parse_address(worker_address, "0.0.0.0", "0", true);

  // Connect to driver and send fragment allocation request
  bool connection_result = false;
  int attempt_count = 0;
  while (attempt_count < max_connection_retry_count && should_stop_ == false) {
    connection_result = driver_client_->fragment_allocation(
        worker_ip, worker_port, target_fragments, cpuinfo, gpuinfo);
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
  // Always launch gRPC service on "0.0.0.0"
  std::string server_address = app_worker_->options()->worker_address;
  auto server_ip = "0.0.0.0";
  auto server_port = holoscan::CLIOptions::parse_port(server_address);
  server_address = fmt::format("{}:{}", server_ip, server_port);

  AppWorkerServiceImpl app_worker_service(app_worker_);

  // Start health checking server if needed
  if (need_health_check_) { grpc::EnableDefaultHealthCheckService(true); }
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&app_worker_service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (server) {
    if (need_health_check_) {
      app_worker_service.set_health_check_service(server->GetHealthCheckService());
    }
    HOLOSCAN_LOG_INFO("AppWorkerServer listening on {}", server_address);
    HOLOSCAN_LOG_INFO("AppWorkerServer/Health checking server listening on {}", server_address);
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
  auto& worker_address = app_worker_->options()->worker_address;

  auto [worker_ip, worker_port] = CLIOptions::parse_address(worker_address, "0.0.0.0", "0", true);

  driver_client_->worker_execution_finished(worker_ip, worker_port, code);
}

std::shared_ptr<distributed::AppDriverClient> AppWorkerServer::app_driver_client() const {
  return driver_client_;
}

}  // namespace holoscan::distributed
