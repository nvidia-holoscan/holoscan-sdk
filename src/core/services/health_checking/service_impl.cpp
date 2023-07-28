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

#include "holoscan/core/app_driver.hpp"
#include "holoscan/logger/logger.hpp"

namespace grpc::health::v1 {

HealthImpl::HealthImpl(holoscan::AppDriver* app_driver) : app_driver_(app_driver) {}

Status HealthImpl::Check(ServerContext* context, const HealthCheckRequest* request,
                         HealthCheckResponse* response) {
  (void)context;
  (void)request;
  // TODO(gbae): Can provide different status based on the app_driver status
  response->set_status(HealthCheckResponse::SERVING);
  return Status::OK;
}

Status Watch(ServerContext* context, const HealthCheckRequest* request,
             ServerWriter<HealthCheckResponse>* writer) {
  (void)context;
  (void)request;
  HealthCheckResponse response;
  response.set_status(HealthCheckResponse::SERVING);
  writer->Write(response);
  return Status::OK;
}

}  // namespace grpc::health::v1
