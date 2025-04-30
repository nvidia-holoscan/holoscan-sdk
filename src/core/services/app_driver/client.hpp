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

#ifndef CORE_SERVICES_APP_DRIVER_CLIENT_HPP
#define CORE_SERVICES_APP_DRIVER_CLIENT_HPP

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <vector>

#include "../generated/app_driver.grpc.pb.h"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/system/cpu_info.hpp"
#include "holoscan/core/system/gpu_info.hpp"

namespace holoscan {

// Forward declarations
enum class AppWorkerTerminationCode;
namespace service {

class AppDriverClient {
 public:
  AppDriverClient(const std::string& driver_address, std::shared_ptr<grpc::Channel> channel);

  bool fragment_allocation(const std::string& worker_ip, const std::string& worker_port,
                           const std::vector<FragmentNodeType>& target_fragments,
                           const CPUInfo& cpuinfo, const std::vector<GPUInfo>& gpuinfo);

  bool worker_execution_finished(const std::string& worker_ip, const std::string& worker_port,
                                 AppWorkerTerminationCode code);

  // Request the AppDriver to initiate a clean shutdown
  bool initiate_shutdown(const std::string& fragment_name);

 private:
  std::string driver_address_;
  std::unique_ptr<holoscan::service::AppDriverService::Stub> stub_;
};
}  // namespace service
}  // namespace holoscan

#endif /* CORE_SERVICES_APP_DRIVER_CLIENT_HPP */
