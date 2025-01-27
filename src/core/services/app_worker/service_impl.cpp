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

#include "service_impl.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/fragment.hpp"
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

  // Get preferred network ports from environment variable
  auto prefer_ports = get_preferred_network_ports("HOLOSCAN_UCX_PORTS");
  std::vector<int> unused_ports = get_unused_network_ports(request->number_of_ports(),
                                                           request->min_port(),
                                                           request->max_port(),
                                                           used_ports,
                                                           prefer_ports);

  for (int port : unused_ports) { response->add_unused_ports(port); }

  return grpc::Status::OK;
}

grpc::Status AppWorkerServiceImpl::GetFragmentInfo(
    grpc::ServerContext* context, const holoscan::service::FragmentInfoRequest* request,
    holoscan::service::FragmentInfoResponse* response) {
  (void)context;

  // create MultipleFragmentsPortMap with fragment port info for the requested fragments
  MultipleFragmentsPortMap scheduled_fragments_map;
  scheduled_fragments_map.reserve(request->fragment_names_size());

  auto& frag_graph = app_worker_->fragment_graph();
  for (const auto& fragment_name : request->fragment_names()) {
    auto fragment = frag_graph.find_node(fragment_name);
    if (!fragment) {
      HOLOSCAN_LOG_ERROR("Did not find fragment {} in the worker's fragment graph", fragment_name);
      continue;
    }
    try {
      fragment->compose_graph();
    } catch (const std::exception& exception) {
      HOLOSCAN_LOG_ERROR("GetFragmentInfo failed: {}", exception.what());
      return grpc::Status::CANCELLED;
    }
    scheduled_fragments_map.try_emplace(fragment->name(), fragment->port_info());
  }

  // convert scheduled_fragments_map to the form needed for the response
  auto scheduled_fragments_info = response->mutable_multi_fragment_port_info();
  for (const auto& [frag_name, op_info_map] : scheduled_fragments_map) {
    holoscan::service::MultiFragmentPortInfo_FragmentPortInfo frag_info;

    frag_info.set_fragment_name(frag_name);
    for (const auto& [op_name, port_tuple] : op_info_map) {
      holoscan::service::MultiFragmentPortInfo_OperatorPortInfo op_info;
      op_info.set_operator_name(op_name);
      const auto& [in_names, out_names, receiver_names] = port_tuple;
      for (const auto& in_name : in_names) { op_info.add_input_names(in_name); }
      for (const auto& out_name : out_names) { op_info.add_output_names(out_name); }
      for (const auto& recv_name : receiver_names) { op_info.add_receiver_names(recv_name); }
      frag_info.add_operators_info()->CopyFrom(op_info);
    }

    scheduled_fragments_info->add_targets_info()->CopyFrom(frag_info);
  }

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
      HOLOSCAN_LOG_DEBUG("Connection IO Type: {}", static_cast<int>(connection_item.io_type()));
      HOLOSCAN_LOG_DEBUG("Connection Connector Type: {}",
                         static_cast<int>(connection_item.connector_type()));

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
          HOLOSCAN_LOG_ERROR("Unsupported connector type: {}",
                             static_cast<int>(connection_item.connector_type()));
          return grpc::Status::CANCELLED;
      }
      holoscan::ArgList arg_list;

      for (const auto& arg : connection_item.args()) {
        HOLOSCAN_LOG_DEBUG("Arg Key: {}", arg.key());

        switch (arg.value_case()) {
          case holoscan::service::ConnectorArg::ValueCase::kStrValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {} (string)", arg.str_value());
            arg_list.add(holoscan::Arg(arg.key(), arg.str_value()));
            break;
          case holoscan::service::ConnectorArg::ValueCase::kIntValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {} (int32)", arg.int_value());
            arg_list.add(holoscan::Arg(arg.key(), arg.int_value()));
            break;
          case holoscan::service::ConnectorArg::ValueCase::kUintValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {} (uint32)", arg.uint_value());
            arg_list.add(holoscan::Arg(arg.key(), arg.uint_value()));
            break;
          case holoscan::service::ConnectorArg::ValueCase::kDoubleValue:
            HOLOSCAN_LOG_DEBUG("Arg Value: {} (double)", arg.double_value());
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

void AppWorkerServiceImpl::set_health_check_service(
    grpc::HealthCheckServiceInterface* health_check_service) {
  health_check_service_ = health_check_service;
}
}  // namespace holoscan::service
