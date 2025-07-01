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

#ifndef CORE_DISTRIBUTED_APP_WORKER_CLIENT_HPP
#define CORE_DISTRIBUTED_APP_WORKER_CLIENT_HPP

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../generated/app_worker.grpc.pb.h"

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/distributed/common/network_constants.hpp"
#include "holoscan/core/forward_def.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan::distributed {

class AppWorkerClient {
 public:
  AppWorkerClient(const std::string& worker_address, std::shared_ptr<grpc::Channel> channel);

  const std::string& ip_address() const;

  std::vector<int32_t> available_ports(uint32_t number_of_ports,
                                       uint32_t min_port = kMinNetworkPort,
                                       uint32_t max_port = kMaxNetworkPort,
                                       const std::vector<uint32_t>& used_ports = {});

  MultipleFragmentsPortMap fragment_port_info(const std::vector<std::string>& fragment_names);

  bool fragment_execution(
      const std::vector<std::shared_ptr<Fragment>>& fragments,
      const std::unordered_map<std::shared_ptr<Fragment>,
                               std::vector<std::shared_ptr<holoscan::ConnectionItem>>>&
          connection_map);

  bool terminate_worker(AppWorkerTerminationCode code);

 private:
  std::string worker_address_;
  std::string worker_ip_;
  std::unique_ptr<holoscan::distributed::AppWorkerService::Stub> stub_;
};
}  // namespace holoscan::distributed

#endif /* CORE_DISTRIBUTED_APP_WORKER_CLIENT_HPP */
