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

#ifndef CORE_SERVICES_APP_DRIVER_SERVICE_IMPL_HPP
#define CORE_SERVICES_APP_DRIVER_SERVICE_IMPL_HPP

#include <grpcpp/grpcpp.h>

#include <string>
#include <vector>

#include "../generated/app_driver.grpc.pb.h"

// Forward declarations
namespace holoscan {}

namespace holoscan {

// Forward declarations
class AppDriver;

namespace service {

class AppDriverServiceImpl final : public AppDriverService::Service {
 public:
  explicit AppDriverServiceImpl(holoscan::AppDriver* app_driver);

  grpc::Status AllocateFragments(grpc::ServerContext* context,
                                 const holoscan::service::FragmentAllocationRequest* request,
                                 holoscan::service::FragmentAllocationResponse* response) override;

  grpc::Status ReportWorkerExecutionFinished(
      grpc::ServerContext* context,
      const holoscan::service::WorkerExecutionFinishedRequest* request,
      holoscan::service::WorkerExecutionFinishedResponse* response) override;

 private:
  /// Parse the IP address from a peer address ("[protocol]:[ip]:[port]").
  static std::string parse_ip(const std::string& peer);

  /// Parse the port number from a peer address ("[protocol]:[ip]:[port]").
  static std::string parse_port(const std::string& peer);

  void store_worker_info(const std::string& client_address,
                         const google::protobuf::RepeatedPtrField<std::string>& fragment_names,
                         const holoscan::service::AvailableSystemResource& resource);

  holoscan::AppDriver* app_driver_ = nullptr;
};

}  // namespace service
}  // namespace holoscan

#endif /* CORE_SERVICES_APP_DRIVER_SERVICE_IMPL_HPP */
