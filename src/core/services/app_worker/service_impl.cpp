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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/system/network_utils.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::service {

AppWorkerServiceImpl::AppWorkerServiceImpl(holoscan::AppWorker* app_worker)
    : app_worker_(app_worker) {}

grpc::Status AppWorkerServiceImpl::GetAvailablePorts(
    grpc::ServerContext* context, const holoscan::service::AvailablePortsRequest* request,
    holoscan::service::AvailablePortsResponse* response) {
  (void)context;

  HOLOSCAN_LOG_INFO("Number of ports requested: {}", request->number_of_ports());

  // Create a vector of uint32_t from the used_ports
  std::vector<int> used_ports;
  for (const auto& port : request->used_ports()) { used_ports.push_back(port); }

  std::vector<int> unused_ports = get_unused_network_ports(
      request->number_of_ports(), request->min_port(), request->max_port(), used_ports);

  for (int port : unused_ports) { response->add_unused_ports(port); }

  return grpc::Status::OK;
}

grpc::Status AppWorkerServiceImpl::ExecuteFragments(
    grpc::ServerContext* context, const holoscan::service::FragmentExecutionRequest* request,
    holoscan::service::FragmentExecutionResponse* response) {
  (void)context;

  // Reconstructing connection_map (but std::string as a key for fragment name)
  std::unordered_map<std::string, std::vector<std::shared_ptr<holoscan::ConnectionItem>>>
      connection_map;

  const auto& fragment_connections_map = request->fragment_connections_map();
  for (const auto& fragment_connection : fragment_connections_map) {
    const auto& fragment_id = fragment_connection.first;
    const auto& connection_item_list = fragment_connection.second;

    if (connection_map.find(fragment_id) == connection_map.end()) {
      connection_map[fragment_id] = std::vector<std::shared_ptr<holoscan::ConnectionItem>>();
    }
    auto& connection_vector = connection_map[fragment_id];

    HOLOSCAN_LOG_DEBUG("Fragment ID: {}", fragment_id);
    for (const auto& connection_item : connection_item_list.connections()) {
      HOLOSCAN_LOG_DEBUG("Connection Name: {}", connection_item.name());
      HOLOSCAN_LOG_DEBUG("Connection IO Type: {}", connection_item.io_type());
      HOLOSCAN_LOG_DEBUG("Connection Connector Type: {}", connection_item.connector_type());

      IOSpec::IOType io_type = IOSpec::IOType::kInput;
      switch (connection_item.io_type()) {
        case holoscan::service::IOType::INPUT:
          io_type = IOSpec::IOType::kInput;
          break;
        case holoscan::service::IOType::OUTPUT:
          io_type = IOSpec::IOType::kOutput;
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unsupported connection IO type: {}", static_cast<int>(io_type));
          return grpc::Status::CANCELLED;
      }

      IOSpec::ConnectorType connector_type;
      switch (connection_item.connector_type()) {
        case holoscan::service::ConnectorType::DEFAULT:
          connector_type = IOSpec::ConnectorType::kDefault;
          break;
        case holoscan::service::ConnectorType::DOUBLE_BUFFER:
          connector_type = IOSpec::ConnectorType::kDoubleBuffer;
          break;
        case holoscan::service::ConnectorType::UCX:
          connector_type = IOSpec::ConnectorType::kUCX;
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unsupported connector type: {}", connection_item.connector_type());
          return grpc::Status::CANCELLED;
      }
      holoscan::ArgList arg_list;

      for (const auto& arg : connection_item.args()) {
        HOLOSCAN_LOG_DEBUG("Arg Key: {}", arg.key());

        switch (arg.value_case()) {
          case holoscan::service::ConnectorArg::ValueCase::kStrValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {}", arg.str_value());
            arg_list.add(holoscan::Arg(arg.key(), arg.str_value()));
            break;
          case holoscan::service::ConnectorArg::ValueCase::kIntValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {}", arg.int_value());
            arg_list.add(holoscan::Arg(arg.key(), arg.int_value()));
            break;
          case holoscan::service::ConnectorArg::ValueCase::kDoubleValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {}", arg.double_value());
            arg_list.add(holoscan::Arg(arg.key(), arg.double_value()));
            break;
          default:
            HOLOSCAN_LOG_ERROR("Arg Value: Unsupported type");
            break;
        }
      }

      auto connection_elem = std::make_shared<holoscan::ConnectionItem>(
          connection_item.name(), io_type, connector_type, arg_list);

      connection_vector.push_back(connection_elem);
    }
    HOLOSCAN_LOG_DEBUG("");
  }

  // Setting a response
  auto result = response->mutable_result();
  result->set_code(holoscan::service::ErrorCode::SUCCESS);
  result->set_message("Fragment Execution request processed successfully");

  // Executing the fragment
  app_worker_->submit_message(holoscan::AppWorker::WorkerMessage{
      holoscan::AppWorker::WorkerMessageCode::kExecuteFragments, connection_map});

  return grpc::Status::OK;
}

grpc::Status AppWorkerServiceImpl::TerminateWorker(
    grpc::ServerContext* context, const holoscan::service::TerminateWorkerRequest* request,
    holoscan::service::TerminateWorkerResponse* response) {
  (void)context;

  auto code = request->code();

  AppWorkerTerminationCode termination_code;

  switch (code) {
    case holoscan::service::ErrorCode::SUCCESS:
      termination_code = AppWorkerTerminationCode::kSuccess;
      break;
    case holoscan::service::ErrorCode::CANCELLED:
      termination_code = AppWorkerTerminationCode::kCancelled;
      break;
    default:
      termination_code = AppWorkerTerminationCode::kFailure;
      break;
  }

  // Terminating the worker
  app_worker_->submit_message(holoscan::AppWorker::WorkerMessage{
      holoscan::AppWorker::WorkerMessageCode::kTerminateWorker, termination_code});

  // Setting a response
  auto result = response->mutable_result();
  result->set_code(holoscan::service::ErrorCode::SUCCESS);
  result->set_message("Terminate Worker request processed successfully");

  return grpc::Status::OK;
}

}  // namespace holoscan::service
