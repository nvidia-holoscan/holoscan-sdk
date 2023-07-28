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

#include "service_impl.hpp"

#include <cstdint>
#include <string>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::service {

static int32_t float_string_to_int32(const std::string& str_value) {
  float value = std::stof(str_value);
  return static_cast<int32_t>(value);
}

AppDriverServiceImpl::AppDriverServiceImpl(holoscan::AppDriver* app_driver)
    : app_driver_(app_driver) {}

grpc::Status AppDriverServiceImpl::AllocateFragments(
    grpc::ServerContext* context, const holoscan::service::FragmentAllocationRequest* request,
    holoscan::service::FragmentAllocationResponse* response) {
  (void)context;

  // If the app is already running, reject the request.
  auto app_status = app_driver_->status();
  if (app_status != holoscan::AppDriver::AppStatus::kNotStarted) {
    holoscan::service::Result* result = new holoscan::service::Result();
    result->set_code(holoscan::service::ErrorCode::ALREADY_STARTED);

    switch (app_status) {
      case holoscan::AppDriver::AppStatus::kRunning:
        result->set_message("The app is already running");
        break;
      case holoscan::AppDriver::AppStatus::kError:
        result->set_message("The app is in the error state");
        break;
      case holoscan::AppDriver::AppStatus::kFinished:
        result->set_message("The app is already finished");
        break;
      default:
        result->set_message("The app is not in the kNotStarted state: {}",
                            static_cast<int>(app_status));
        break;
    }
    response->set_allocated_result(result);
    return grpc::Status::OK;
  }

  std::string client_address = context->peer();
  auto& worker_port = request->worker_port();

  // Construct worker_address using client_address's IP and the app worker's port
  std::string worker_address = fmt::format("{}:{}", parse_ip(client_address), worker_port);

  // Get fragment names from the request and simply log them.
  auto& fragment_names = request->fragment_names();
  const auto& resource = request->available_system_resource();

  // Print the client's IP address (<protocol>:<ip>:<port>)
  HOLOSCAN_LOG_INFO(
      "Connected from the client IP: {} (worker port: {})", client_address, worker_port);

  if (Logger::level() <= LogLevel::DEBUG) {
    HOLOSCAN_LOG_DEBUG("  Worker address: {}", worker_address);
    for (const auto& fragment_name : fragment_names) {
      HOLOSCAN_LOG_DEBUG("  Fragment name: {}", fragment_name);
    }
    HOLOSCAN_LOG_DEBUG(
        "  System resources\n"
        "    CPU: {}\n"
        "    GPU: {}\n"
        "    Mem: {}\n"
        "    Shm: {}\n"
        "    GPU mem: {}\n",
        resource.cpu(),
        resource.gpu(),
        resource.memory(),
        resource.shared_memory(),
        resource.gpu_memory());
  }

  // Store the worker address, the fragment names, and resources in the app_driver_.
  store_worker_info(worker_address, fragment_names, resource);

  // Construct a response.
  holoscan::service::Result* result = new holoscan::service::Result();
  result->set_code(holoscan::service::SUCCESS);
  result->set_message("Fragment allocation request is accepted.");

  response->set_allocated_result(result);

  // Request checking the fragment scheduler.
  app_driver_->submit_message(holoscan::AppDriver::DriverMessage{
      holoscan::AppDriver::DriverMessageCode::kCheckFragmentSchedule, worker_address});

  return grpc::Status::OK;
}

std::string AppDriverServiceImpl::parse_ip(const std::string& peer) {
  // Parse the IP address from a peer address ("[protocol]:[ip]:[port]")
  auto pivot = peer.find_first_of(':');
  if (pivot == std::string::npos) { return ""; }

  std::string ip = peer.substr(pivot + 1, peer.find_last_of(':') - pivot - 1);
  return ip;
}

std::string AppDriverServiceImpl::parse_port(const std::string& peer) {
  // Parse the port number from a peer address ("[protocol]:[ip]:[port]")
  std::string port = peer.substr(peer.find_last_of(':') + 1);
  return port;
}

void AppDriverServiceImpl::store_worker_info(
    const std::string& worker_address,
    const google::protobuf::RepeatedPtrField<std::string>& fragment_names,
    const holoscan::service::AvailableSystemResource& resource) {
  // Store the client address, the fragment names, and resources in the app_driver_.
  FragmentScheduler* fragment_scheduler = app_driver_->fragment_scheduler();

  if (fragment_scheduler == nullptr) {
    HOLOSCAN_LOG_ERROR("Fragment scheduler is not initialized.");
    return;
  }

  holoscan::AvailableSystemResource available_system_resource{};

  available_system_resource.app_worker_id = worker_address;
  // Assume that the string in resource.cpu() is valid
  // TODO(gbae): validate the content.
  available_system_resource.cpu = resource.has_cpu() ? float_string_to_int32(resource.cpu()) : 0;
  available_system_resource.gpu = resource.has_gpu() ? float_string_to_int32(resource.gpu()) : 0;
  available_system_resource.memory =
      resource.has_memory() ? holoscan::AppDriver::parse_memory_size(resource.memory()) : 0;
  available_system_resource.shared_memory =
      resource.has_shared_memory()
          ? holoscan::AppDriver::parse_memory_size(resource.shared_memory())
          : 0;
  available_system_resource.gpu_memory =
      resource.has_gpu_memory() ? holoscan::AppDriver::parse_memory_size(resource.gpu_memory()) : 0;

  for (const auto& fragment_name : fragment_names) {
    available_system_resource.target_fragments.emplace(fragment_name);
  }

  // Add worker info to the fragment scheduler.
  fragment_scheduler->add_available_resource(available_system_resource);
}

grpc::Status AppDriverServiceImpl::ReportWorkerExecutionFinished(
    grpc::ServerContext* context, const holoscan::service::WorkerExecutionFinishedRequest* request,
    holoscan::service::WorkerExecutionFinishedResponse* response) {
  (void)context;
  std::string client_address = context->peer();
  auto& worker_port = request->worker_port();

  // Construct worker_id using client_address's IP and the app worker's port
  std::string worker_id = fmt::format("{}:{}", parse_ip(client_address), worker_port);

  AppWorkerTerminationCode worker_termination_code = AppWorkerTerminationCode::kSuccess;
  auto& worker_termination_status = request->status();

  // Construct a response.
  holoscan::service::Result* result = new holoscan::service::Result();

  std::string message;
  switch (worker_termination_status.code()) {
    case holoscan::service::ErrorCode::SUCCESS:
      worker_termination_code = AppWorkerTerminationCode::kSuccess;
      message = fmt::format("Worker execution succeeded (worker id: {})", worker_id);
      break;
    case holoscan::service::ErrorCode::CANCELLED:
      worker_termination_code = AppWorkerTerminationCode::kCancelled;
      message = fmt::format("Worker execution cancelled (worker id: {})", worker_id);
      break;
    default:
      worker_termination_code = AppWorkerTerminationCode::kFailure;
      result->set_code(holoscan::service::FAILURE);
      message = fmt::format("Invalid worker termination status (worker id: {}, code: {})",
                            worker_id,
                            static_cast<int>(worker_termination_status.code()));

      break;
  }

  result->set_message(message);
  HOLOSCAN_LOG_INFO(message);
  response->set_allocated_result(result);

  // Request checking the fragment scheduler.
  app_driver_->submit_message(holoscan::AppDriver::DriverMessage{
      holoscan::AppDriver::DriverMessageCode::kWorkerExecutionFinished,
      AppWorkerTerminationStatus{worker_id, worker_termination_code}});

  return grpc::Status::OK;
}

}  // namespace holoscan::service
