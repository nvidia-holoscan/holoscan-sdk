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

#include "client.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../generated/error_code.pb.h"
#include "holoscan/core/app_worker.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::service {

AppDriverClient::AppDriverClient(const std::string& driver_address,
                                 std::shared_ptr<grpc::Channel> channel)
    : driver_address_(driver_address),
      stub_(holoscan::service::AppDriverService::NewStub(channel)) {}

bool AppDriverClient::fragment_allocation(const std::string& worker_ip,
                                          const std::string& worker_port,
                                          const std::vector<FragmentNodeType>& target_fragments,
                                          const CPUInfo& cpuinfo,
                                          const std::vector<GPUInfo>& gpuinfo) {
  holoscan::service::FragmentAllocationRequest request;

  request.set_worker_ip(worker_ip);
  request.set_worker_port(worker_port);

  // Adding fragment names
  for (const auto& fragment : target_fragments) { request.add_fragment_names(fragment->name()); }

  // Creating AvailableSystemResource and adding it to the request

  float cpu_memory = cpuinfo.memory_total / 1024 / 1024 / 1024;         /// convert to GiB
  float cpu_shared_memory = cpuinfo.shared_memory_total / 1024 / 1024;  /// convert to MiB
  float gpu_memory = std::numeric_limits<float>::max();
  // Calculate the minimum GPU memory among all GPUs
  for (const auto& gpu : gpuinfo) {
    gpu_memory = std::min(gpu_memory, static_cast<float>(gpu.memory_total / 1024 / 1024 / 1024));
  }

  // Format float value with one decimal place and convert it to string
  auto cpu_memory_str = fmt::format("{:.1f}Gi", cpu_memory);
  auto cpu_shared_memory_str = fmt::format("{:.1f}Mi", cpu_shared_memory);
  auto gpu_memory_str = fmt::format("{:.1f}Gi", gpu_memory);

  auto available_system_resource = request.mutable_available_system_resource();
  available_system_resource->set_cpu(fmt::format("{}", cpuinfo.num_processors));
  available_system_resource->set_gpu(fmt::format("{}", gpuinfo.size()));
  available_system_resource->set_memory(cpu_memory_str);
  available_system_resource->set_shared_memory(cpu_shared_memory_str);
  available_system_resource->set_gpu_memory(gpu_memory_str);

  holoscan::service::FragmentAllocationResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->AllocateFragments(&context, request, &response);

  if (status.ok()) {
    HOLOSCAN_LOG_INFO(
        "FragmentAllocation response ({}): {}", driver_address_, response.result().message());
    if (response.result().code() != holoscan::service::ErrorCode::SUCCESS) {
      HOLOSCAN_LOG_ERROR(
          "FragmentAllocation failed ({}): {}", driver_address_, response.result().message());
      return false;
    }
    return true;
  } else {
    HOLOSCAN_LOG_INFO(
        "FragmentAllocation rpc failed ({}): {}", driver_address_, status.error_message());
    return false;
  }
}

bool AppDriverClient::worker_execution_finished(const std::string& worker_ip,
                                                const std::string& worker_port,
                                                AppWorkerTerminationCode code) {
  holoscan::service::WorkerExecutionFinishedRequest request;
  request.set_worker_ip(worker_ip);
  request.set_worker_port(worker_port);

  holoscan::service::Result* worker_termination_status = new holoscan::service::Result();
  switch (code) {
    case AppWorkerTerminationCode::kSuccess:
      worker_termination_status->set_code(holoscan::service::ErrorCode::SUCCESS);
      break;
    case AppWorkerTerminationCode::kCancelled:
      worker_termination_status->set_code(holoscan::service::ErrorCode::CANCELLED);
      break;
    case AppWorkerTerminationCode::kFailure:
      worker_termination_status->set_code(holoscan::service::ErrorCode::FAILURE);
      break;
  }
  request.set_allocated_status(worker_termination_status);

  // Construct a response.
  holoscan::service::WorkerExecutionFinishedResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->ReportWorkerExecutionFinished(&context, request, &response);

  if (status.ok()) {
    HOLOSCAN_LOG_INFO(
        "WorkerExecutionFinished response ({}): {}", driver_address_, response.result().message());
    if (response.result().code() != holoscan::service::ErrorCode::SUCCESS) {
      HOLOSCAN_LOG_ERROR(
          "WorkerExecutionFinished failed ({}): {}", driver_address_, response.result().message());
      return false;
    }
    return true;
  } else {
    HOLOSCAN_LOG_INFO(
        "WorkerExecutionFinished rpc failed ({}): {}", driver_address_, status.error_message());
    return false;
  }
}

}  // namespace holoscan::service
