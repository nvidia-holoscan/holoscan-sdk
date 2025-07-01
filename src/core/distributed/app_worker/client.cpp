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

#include "client.hpp"

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::distributed {

AppWorkerClient::AppWorkerClient(const std::string& worker_address,
                                 std::shared_ptr<grpc::Channel> channel)
    : worker_address_(worker_address),
      stub_(holoscan::distributed::AppWorkerService::NewStub(channel)) {
  // Call parse_address to handle the parsing of the worker address.
  auto [extracted_ip, _] =
      CLIOptions::parse_address(worker_address,
                                "0.0.0.0",  // default IP address, if needed.
                                "");        // default port is empty, since we only want the IP.

  // Assign the extracted IP to worker_ip_. We don't need to enclose IPv6 in brackets for this use
  // case.
  worker_ip_ = std::move(extracted_ip);
}

const std::string& AppWorkerClient::ip_address() const {
  return worker_ip_;
}

std::vector<int32_t> AppWorkerClient::available_ports(uint32_t number_of_ports, uint32_t min_port,
                                                      uint32_t max_port,
                                                      const std::vector<uint32_t>& used_ports) {
  holoscan::distributed::AvailablePortsRequest request;

  request.set_number_of_ports(number_of_ports);
  request.set_min_port(min_port);
  request.set_max_port(max_port);
  if (!used_ports.empty()) {
    for (const auto& port : used_ports) { request.add_used_ports(port); }
  }

  holoscan::distributed::AvailablePortsResponse response;
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

MultipleFragmentsPortMap AppWorkerClient::fragment_port_info(
    const std::vector<std::string>& fragment_names) {
  holoscan::distributed::FragmentInfoRequest request;
  for (const auto& frag_name : fragment_names) { request.add_fragment_names(frag_name); }

  MultipleFragmentsPortMap target_fragments_port_map;
  target_fragments_port_map.reserve(fragment_names.size());

  holoscan::distributed::FragmentInfoResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->GetFragmentInfo(&context, request, &response);

  if (status.ok()) {
    HOLOSCAN_LOG_INFO("GetFragmentInfo response ({}) received", worker_address_);

    const auto& fragments_port_info = response.multi_fragment_port_info();

    for (const auto& target_info : fragments_port_info.targets_info()) {
      FragmentPortMap fragment_port_map;
      std::string frag_name;
      if (target_info.has_fragment_name()) {
        frag_name = target_info.fragment_name();
      } else {
        HOLOSCAN_LOG_ERROR(
            "MultiFragmentPortInfo in FragmentAllocationRequest did not contain a fragment name");
        continue;
      }
      for (const auto& op_info : target_info.operators_info()) {
        std::string op_name;
        if (op_info.has_operator_name()) {
          op_name = op_info.operator_name();
        } else {
          HOLOSCAN_LOG_ERROR(
              "MultiFragmentPortInfo_FragmentPortInfo in FragmentAllocationRequest did not contain "
              "an operator name");
          continue;
        }

        std::unordered_set<std::string> input_names;
        input_names.reserve(op_info.input_names_size());
        for (auto& in : op_info.input_names()) { input_names.insert(in); }

        std::unordered_set<std::string> output_names;
        output_names.reserve(op_info.output_names_size());
        for (auto& out : op_info.output_names()) { output_names.insert(out); }

        std::unordered_set<std::string> receiver_names;
        receiver_names.reserve(op_info.receiver_names_size());
        for (auto& recv : op_info.receiver_names()) { receiver_names.insert(recv); }

        fragment_port_map.try_emplace(std::move(op_name),
                                      std::move(input_names),
                                      std::move(output_names),
                                      std::move(receiver_names));
      }
      target_fragments_port_map.try_emplace(std::move(frag_name), std::move(fragment_port_map));
    }
  } else {
    HOLOSCAN_LOG_INFO(
        "GetFragmentInfo rpc failed ({}): {}", worker_address_, status.error_message());
  }
  return target_fragments_port_map;
}

bool AppWorkerClient::fragment_execution(
    const std::vector<std::shared_ptr<Fragment>>& fragments,
    const std::unordered_map<std::shared_ptr<Fragment>,
                             std::vector<std::shared_ptr<holoscan::ConnectionItem>>>&
        connection_map) {
  holoscan::distributed::FragmentExecutionRequest request;

  for (const auto& fragment : fragments) {
    if (connection_map.find(fragment) != connection_map.end()) {
      auto& connections = connection_map.at(fragment);
      holoscan::distributed::ConnectionItemList connection_item_list;
      for (const auto& connection : connections) {
        holoscan::distributed::ConnectionItem* connection_item =
            connection_item_list.add_connections();
        connection_item->set_name(connection->name);

        switch (connection->io_type) {
          case IOSpec::IOType::kInput:
            connection_item->set_io_type(holoscan::distributed::IOType::INPUT);
            break;
          case IOSpec::IOType::kOutput:
            connection_item->set_io_type(holoscan::distributed::IOType::OUTPUT);
            break;
        }

        switch (connection->connector_type) {
          case IOSpec::ConnectorType::kDefault:
            connection_item->set_connector_type(holoscan::distributed::ConnectorType::DEFAULT);
            break;
          case IOSpec::ConnectorType::kDoubleBuffer:
            connection_item->set_connector_type(
                holoscan::distributed::ConnectorType::DOUBLE_BUFFER);
            break;
          case IOSpec::ConnectorType::kUCX:
            connection_item->set_connector_type(holoscan::distributed::ConnectorType::UCX);
            break;
        }

        // Currently supporting only arguments for UCX connector (rx_address, address, port)
        for (auto& arg : connection->args) {
          holoscan::distributed::ConnectorArg* connector_arg = connection_item->add_args();

          connector_arg->set_key(arg.name());
          try {
            switch (arg.arg_type().element_type()) {
              case holoscan::ArgElementType::kString:
                connector_arg->set_str_value(std::any_cast<std::string>(arg.value()));
                break;
              case holoscan::ArgElementType::kInt32:
                connector_arg->set_int_value(std::any_cast<int32_t>(arg.value()));
                break;
              case holoscan::ArgElementType::kUnsigned32:
                connector_arg->set_uint_value(std::any_cast<uint32_t>(arg.value()));
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

  holoscan::distributed::FragmentExecutionResponse response;
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
  holoscan::distributed::TerminateWorkerRequest request;

  switch (code) {
    case AppWorkerTerminationCode::kSuccess:
      request.set_code(holoscan::distributed::ErrorCode::SUCCESS);
      break;
    case AppWorkerTerminationCode::kCancelled:
      request.set_code(holoscan::distributed::ErrorCode::CANCELLED);
      break;
    case AppWorkerTerminationCode::kFailure:
      request.set_code(holoscan::distributed::ErrorCode::FAILURE);
      break;
  }

  holoscan::distributed::TerminateWorkerResponse response;
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

}  // namespace holoscan::distributed
