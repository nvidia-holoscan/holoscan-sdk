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

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::service {

AppWorkerClient::AppWorkerClient(const std::string& worker_address,
                                 std::shared_ptr<grpc::Channel> channel)
    : worker_address_(worker_address),
      stub_(holoscan::service::AppWorkerService::NewStub(channel)) {
  // Extract IP address part
  auto colon_pos = worker_address.find(':');
  worker_ip_ =
      colon_pos != std::string::npos ? worker_address.substr(0, colon_pos) : worker_address;
}

const std::string& AppWorkerClient::ip_address() const {
  return worker_ip_;
}

std::vector<int32_t> AppWorkerClient::available_ports(uint32_t number_of_ports, uint32_t min_port,
                                                      uint32_t max_port,
                                                      const std::vector<uint32_t>& used_ports) {
  holoscan::service::AvailablePortsRequest request;

  request.set_number_of_ports(number_of_ports);
  request.set_min_port(min_port);
  request.set_max_port(max_port);
  if (!used_ports.empty()) {
    for (const auto& port : used_ports) { request.add_used_ports(port); }
  }

  holoscan::service::AvailablePortsResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->GetAvailablePorts(&context, request, &response);

  std::vector<int32_t> ports;
  if (status.ok()) {
    ports.reserve(response.unused_ports_size());
    for (const auto& port : response.unused_ports()) { ports.push_back(port); }
    HOLOSCAN_LOG_DEBUG("AvailablePorts response from worker '{}': {}",
                       worker_address_,
                       fmt::join(response.unused_ports(), ","));

  } else {
    HOLOSCAN_LOG_ERROR("Unable to get available ports from worker '{}' : {}",
                       worker_address_,
                       status.error_message());
  }
  return ports;
}

bool AppWorkerClient::fragment_execution(
    const std::vector<std::shared_ptr<Fragment>>& fragments,
    const std::unordered_map<std::shared_ptr<Fragment>,
                             std::vector<std::shared_ptr<holoscan::ConnectionItem>>>&
        connection_map) {
  holoscan::service::FragmentExecutionRequest request;

  for (const auto& fragment : fragments) {
    if (connection_map.find(fragment) != connection_map.end()) {
      auto& connections = connection_map.at(fragment);
      holoscan::service::ConnectionItemList connection_item_list;
      for (const auto& connection : connections) {
        holoscan::service::ConnectionItem* connection_item = connection_item_list.add_connections();
        connection_item->set_name(connection->name);

        switch (connection->io_type) {
          case IOSpec::IOType::kInput:
            connection_item->set_io_type(holoscan::service::IOType::INPUT);
            break;
          case IOSpec::IOType::kOutput:
            connection_item->set_io_type(holoscan::service::IOType::OUTPUT);
            break;
        }

        switch (connection->connector_type) {
          case IOSpec::ConnectorType::kDefault:
            connection_item->set_connector_type(holoscan::service::ConnectorType::DEFAULT);
            break;
          case IOSpec::ConnectorType::kDoubleBuffer:
            connection_item->set_connector_type(holoscan::service::ConnectorType::DOUBLE_BUFFER);
            break;
          case IOSpec::ConnectorType::kUCX:
            connection_item->set_connector_type(holoscan::service::ConnectorType::UCX);
            break;
        }

        // Currently supporting only arguments for UCX connector (rx_address, address, port)
        for (auto& arg : connection->args) {
          holoscan::service::ConnectorArg* connector_arg = connection_item->add_args();

          connector_arg->set_key(arg.name());
          try {
            switch (arg.arg_type().element_type()) {
              case holoscan::ArgElementType::kString:
                connector_arg->set_str_value(std::any_cast<std::string>(arg.value()));
                break;
              case holoscan::ArgElementType::kInt32:
                connector_arg->set_int_value(std::any_cast<int32_t>(arg.value()));
                break;
              case holoscan::ArgElementType::kFloat64:
                connector_arg->set_double_value(std::any_cast<double>(arg.value()));
                break;
              default:
                HOLOSCAN_LOG_ERROR("Unsupported arg type: {}",
                                   static_cast<int>(arg.arg_type().element_type()));
                break;
            }
          } catch (const std::bad_any_cast& e) {
            HOLOSCAN_LOG_ERROR("Unable to cast arg '{}' to string: {}", arg.name(), e.what());
          }
        }
      }

      auto& fragment_id = fragment->name();
      (*request.mutable_fragment_connections_map())[fragment_id] = connection_item_list;
    }
  }

  holoscan::service::FragmentExecutionResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ExecuteFragments(&context, request, &response);

  if (status.ok()) {
    HOLOSCAN_LOG_INFO(
        "FragmentExecution response ({}): {}", worker_address_, response.result().message());
    return true;
  } else {
    HOLOSCAN_LOG_INFO(
        "FragmentExecution rpc failed ({}): {}", worker_address_, status.error_message());
    return false;
  }
}

bool AppWorkerClient::terminate_worker(AppWorkerTerminationCode code) {
  holoscan::service::TerminateWorkerRequest request;

  switch (code) {
    case AppWorkerTerminationCode::kSuccess:
      request.set_code(holoscan::service::ErrorCode::SUCCESS);
      break;
    case AppWorkerTerminationCode::kCancelled:
      request.set_code(holoscan::service::ErrorCode::CANCELLED);
      break;
    case AppWorkerTerminationCode::kFailure:
      request.set_code(holoscan::service::ErrorCode::FAILURE);
      break;
  }

  holoscan::service::TerminateWorkerResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->TerminateWorker(&context, request, &response);

  if (status.ok()) {
    HOLOSCAN_LOG_INFO(
        "TerminateWorker response ({}): {}", worker_address_, response.result().message());
    return true;
  } else {
    HOLOSCAN_LOG_INFO(
        "TerminateWorker rpc failed ({}) : {}", worker_address_, status.error_message());
    return false;
  }
}

}  // namespace holoscan::service
