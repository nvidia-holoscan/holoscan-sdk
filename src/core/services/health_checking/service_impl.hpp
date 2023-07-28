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

#ifndef CORE_SERVICES_HEALTH_CHECKING_SERVICE_IMPL_HPP
#define CORE_SERVICES_HEALTH_CHECKING_SERVICE_IMPL_HPP

#include <grpcpp/grpcpp.h>

#include "../generated/health_checking.grpc.pb.h"

// Forward declarations
namespace holoscan {
class AppDriver;
}

namespace grpc::health::v1 {

class HealthImpl final : public Health::Service {
 public:
  HealthImpl(holoscan::AppDriver* app_driver);
  Status Check(ServerContext* context, const HealthCheckRequest* request,
               HealthCheckResponse* response) override;

 private:
  holoscan::AppDriver* app_driver_ = nullptr;
};

}  // namespace grpc::health::v1

#endif /* CORE_SERVICES_HEALTH_CHECKING_SERVICE_IMPL_HPP */
