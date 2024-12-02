/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/services/app_driver/server.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/cli_options.hpp"
#include "holoscan/core/system/network_utils.hpp"
#include "holoscan/logger/logger.hpp"

#include "../app_worker/client.hpp"
#include "../health_checking/service_impl.hpp"
#include "service_impl.hpp"

namespace holoscan::service {

AppDriverServer::AppDriverServer(holoscan::AppDriver* app_driver, bool need_driver,
                                 bool need_health_check)
    : app_driver_(app_driver), need_driver_(need_driver), need_health_check_(need_health_check) {}

AppDriverServer::~AppDriverServer() = default;

void AppDriverServer::start() {
  server_thread_ = std::make_unique<std::thread>(&AppDriverServer::run, this);
}

void AppDriverServer::stop() {
  should_stop_ = true;
  cv_.notify_all();
}

void AppDriverServer::wait() {
  std::unique_lock<std::mutex> lock(join_mutex_);
  if (server_thread_ && server_thread_->joinable()) { server_thread_->join(); }
}

void AppDriverServer::run() {
  // Always launch gRPC service on "0.0.0.0" (all interfaces)
  std::string server_address = app_driver_->options()->driver_address;
  auto server_ip = "0.0.0.0";
  auto server_port = holoscan::CLIOptions::parse_port(server_address);
  server_address = fmt::format("{}:{}", server_ip, server_port);

  // Start health checking server if needed
  std::unique_ptr<grpc::Server> health_check_server;
  grpc::health::v1::HealthImpl health_checking_service(app_driver_);
  if (need_health_check_) {
    int32_t health_check_port = kDefaultHealthCheckingPort;

    // Check the environment variable for the port (`HOLOSCAN_HEALTH_CHECK_PORT`)
    const char* env_health_check_port = std::getenv("HOLOSCAN_HEALTH_CHECK_PORT");
    if (env_health_check_port != nullptr && env_health_check_port[0] != '\0') {
      HOLOSCAN_LOG_DEBUG("Using environment variable HOLOSCAN_HEALTH_CHECK_PORT={}",
                         env_health_check_port);
      try {
        health_check_port = std::stoi(env_health_check_port);
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR(
            "Failed to parse HOLOSCAN_HEALTH_CHECK_PORT={}: {}", env_health_check_port, e.what());
      }
    }

    grpc::ServerBuilder health_check_builder;
    health_check_builder.AddListeningPort(fmt::format("0.0.0.0:{}", health_check_port),
                                          grpc::InsecureServerCredentials());
    health_check_builder.RegisterService(&health_checking_service);
    health_check_server = health_check_builder.BuildAndStart();
    if (health_check_server) {
      HOLOSCAN_LOG_INFO("Health checking server listening on 0.0.0.0:{}", health_check_port);
    } else {
      HOLOSCAN_LOG_ERROR("Failed to start health checking server on 0.0.0.0:{}", health_check_port);
    }
  }

  // Start AppDriver server if needed
  std::unique_ptr<grpc::Server> server;
  AppDriverServiceImpl app_driver_service(app_driver_);
  if (need_driver_) {
    // Check if the listening port is already in use
    int server_port_int = std::stoi(server_port);
    auto unused_ports =
        get_unused_network_ports(1, server_port_int, server_port_int, {}, {server_port_int});
    if (unused_ports.empty() || unused_ports[0] != server_port_int) {
      HOLOSCAN_LOG_ERROR("Port {} is already in use", server_port_int);
      should_stop_ = true;
    } else {
      // Start the gRPC server
      grpc::ServerBuilder builder;
      builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
      builder.RegisterService(&app_driver_service);
      server = builder.BuildAndStart();
      if (server) {
        HOLOSCAN_LOG_INFO("AppDriverServer listening on {}", server_address);
      } else {
        HOLOSCAN_LOG_ERROR("Failed to start AppDriverServer on {}", server_address);
        should_stop_ = true;
      }
    }
  }

  // Wait until we should stop the server
  std::unique_lock<std::mutex> lock(mutex_);
  while (!should_stop_) {
    cv_.wait(lock);
    // Process message queue if there is any message
    app_driver_->process_message_queue();
  }

  if (server) { server->Shutdown(); }
  if (health_check_server) { health_check_server->Shutdown(); }
}

void AppDriverServer::notify() {
  cv_.notify_all();
}

std::unique_ptr<AppWorkerClient>& AppDriverServer::connect_to_worker(
    const std::string& worker_address) {
  auto it = worker_clients_.find(worker_address);
  if (it == worker_clients_.end()) {
    auto client = std::make_unique<AppWorkerClient>(
        worker_address, grpc::CreateChannel(worker_address, grpc::InsecureChannelCredentials()));
    worker_clients_.emplace(worker_address, std::move(client));
    return worker_clients_.at(worker_address);
  } else {
    return it->second;
  }
}

bool AppDriverServer::close_worker_connection(const std::string& worker_address) {
  // Close the connection by removing the client
  std::size_t num_erased = worker_clients_.erase(worker_address);
  if (num_erased > 0) {
    HOLOSCAN_LOG_INFO("Closed connection to worker {}", worker_address);
    return true;
  } else {
    HOLOSCAN_LOG_DEBUG("No worker client available for {}", worker_address);
    return false;
  }
}

std::vector<std::string> AppDriverServer::get_worker_addresses() const {
  std::vector<std::string> worker_addresses;
  for (const auto& [worker_address, _] : worker_clients_) {
    worker_addresses.emplace_back(worker_address);
  }
  return worker_addresses;
}

std::size_t AppDriverServer::num_worker_connections() const {
  return worker_clients_.size();
}

}  // namespace holoscan::service
