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

#ifndef CORE_DISTRIBUTED_APP_DRIVER_SERVICE_IMPL_HPP
#define CORE_DISTRIBUTED_APP_DRIVER_SERVICE_IMPL_HPP

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <string>
#include <vector>

#include "../generated/app_driver.grpc.pb.h"

// Forward declarations
namespace holoscan {}

namespace holoscan {

// Forward declarations
class AppDriver;

namespace distributed {

class AppDriverServiceImpl final : public AppDriverService::Service {
 public:
  explicit AppDriverServiceImpl(holoscan::AppDriver* app_driver);

  grpc::Status AllocateFragments(
      grpc::ServerContext* context, const holoscan::distributed::FragmentAllocationRequest* request,
      holoscan::distributed::FragmentAllocationResponse* response) override;

  grpc::Status ReportWorkerExecutionFinished(
      grpc::ServerContext* context,
      const holoscan::distributed::WorkerExecutionFinishedRequest* request,
      holoscan::distributed::WorkerExecutionFinishedResponse* response) override;

  grpc::Status InitiateShutdown(grpc::ServerContext* context,
                                const holoscan::distributed::InitiateShutdownRequest* request,
                                holoscan::distributed::InitiateShutdownResponse* response) override;

  void set_health_check_service(grpc::HealthCheckServiceInterface* health_check_service);

 private:
  /// Decode URI-encoded characters from the source string.
  static std::string uri_decode(const std::string& src);

  /// Parse the IP address from a peer address ("[protocol]:[ip]:[port]"). Enclose the IP address
  /// with square brackets if it is an IPv6 address. Decode URI-encoded characters from the peer
  /// string.
  static std::string parse_ip_from_peer(const std::string& peer);

  /// Parse the port number from a peer address ("[protocol]:[ip]:[port]"). Decode URI-encoded
  /// characters from the peer string.
  static std::string parse_port_from_peer(const std::string& peer);

  void store_worker_info(const std::string& client_address,
                         const google::protobuf::RepeatedPtrField<std::string>& fragment_names,
                         const holoscan::distributed::AvailableSystemResource& resource);

  holoscan::AppDriver* app_driver_ = nullptr;
  grpc::HealthCheckServiceInterface* health_check_service_ = nullptr;
};

}  // namespace distributed
}  // namespace holoscan

#endif /* CORE_DISTRIBUTED_APP_DRIVER_SERVICE_IMPL_HPP */
