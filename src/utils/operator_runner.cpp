/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/utils/operator_runner.hpp"

#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::ops {

OperatorRunner::OperatorRunner(const std::shared_ptr<holoscan::Operator>& op) : op_(op) {
  auto frag = op_->fragment();
  if (frag) {
    gxf_context_ = frag->executor().context();
  } else {
    HOLOSCAN_LOG_ERROR("Failed to get fragment from operator");
    throw std::runtime_error("Failed to get fragment from operator");
  }

  // Append AsynchronousCondition to the operator to block the operator from running (using
  // AsynchronousEventState::WAIT). We cannot use BooleanCondition (with false) or
  // AsynchronousEventState::EVENT_NEVER because it will cause the operator to stop running
  // immediately.
  async_condition_ = frag->make_condition<holoscan::AsynchronousCondition>("async_condition");
  op_->add_arg(async_condition_);

  // For each output port of the operator, set condition type to 'None' to prevent adding any
  // default conditions like DownstreamAffordableCondition that would be added otherwise.
  for (const auto& [port_name, output_spec] : op_->spec()->outputs()) {
    output_spec->condition(holoscan::ConditionType::kNone);
  }

  auto& executor = op_->fragment()->executor();

  auto gxf_executor = dynamic_cast<holoscan::gxf::GXFExecutor*>(&executor);

  gxf_uid_t old_op_eid = 0;
  gxf_uid_t old_op_cid = 0;

  if (gxf_executor) {
    // If the GXF context is not owned by the Holoscan executor, meaning this operator is
    // initialized by OperatorWrapper, we need to explicitly set the GXFExecutor's op_eid and op_cid
    // to 0. This ensures that a new entity and component can be created for the operator used by
    // OperatorRunner, unlike the operator created by OperatorWrapper.
    if (!executor.owns_context()) {
      HOLOSCAN_LOG_DEBUG("Setting op_eid and op_cid of GXFExecutor to 0");
      old_op_eid = gxf_executor->op_eid();
      old_op_cid = gxf_executor->op_cid();
      gxf_executor->op_eid(0);
      gxf_executor->op_cid(0);
    }
  } else {
    HOLOSCAN_LOG_ERROR("Failed to get GXFExecutor from executor");
    throw std::runtime_error("Failed to get GXFExecutor from executor");
  }

  // Initialize the operator to create the components which initialize AsynchronousCondition as
  // well.
  op_->initialize();

  // Restore the GXFExecutor's op_eid and op_cid
  if (old_op_eid != 0) {
    gxf_executor->op_eid(old_op_eid);
    gxf_executor->op_cid(old_op_cid);
  }

  // If the GXF context is not owned by the Holoscan executor, meaning this operator is initialized
  // by OperatorWrapper, we need to explicitly activate the operator's entity since it is created
  // after the entity graph has been activated (by gxe).
  if (!executor.owns_context()) {
    GxfEntityActivate(executor.context(), op_->graph_entity()->eid());
  }

  // Set the AsynchronousCondition to 'WAIT' to block the operator from running.
  async_condition_->event_state(holoscan::AsynchronousEventState::WAIT);

  // Collect underlying DoubleBufferReceiver and DoubleBufferTransmitter to the operator.
  for (const auto& [port_name, input_spec] : op_->spec()->inputs()) {
    auto connector = input_spec->connector();
    if (connector) {
      auto double_buffer_receiver =
          std::dynamic_pointer_cast<holoscan::DoubleBufferReceiver>(connector);
      if (double_buffer_receiver) {
        auto gxf_receiver = double_buffer_receiver->get();
        if (gxf_receiver) {
          double_buffer_receivers_[port_name] = gxf_receiver;
          HOLOSCAN_LOG_DEBUG("DoubleBufferReceiver {} added to OperatorRunner", port_name);
        } else {
          HOLOSCAN_LOG_ERROR("Failed to get DoubleBufferReceiver for port {}", port_name);
          throw std::runtime_error("Failed to get DoubleBufferReceiver");
        }
      } else {
        HOLOSCAN_LOG_ERROR("Connector is not a DoubleBufferReceiver for port {}", port_name);
        throw std::runtime_error("Connector is not a DoubleBufferReceiver");
      }
    }
  }
  for (const auto& [port_name, output_spec] : op_->spec()->outputs()) {
    auto connector = output_spec->connector();
    if (connector) {
      auto double_buffer_transmitter =
          std::dynamic_pointer_cast<holoscan::DoubleBufferTransmitter>(connector);
      if (double_buffer_transmitter) {
        auto gxf_transmitter = double_buffer_transmitter->get();
        if (gxf_transmitter) {
          double_buffer_transmitters_[port_name] = gxf_transmitter;
          HOLOSCAN_LOG_DEBUG("DoubleBufferTransmitter {} added to OperatorRunner", port_name);
        } else {
          HOLOSCAN_LOG_ERROR("Failed to get DoubleBufferTransmitter for port {}", port_name);
          throw std::runtime_error("Failed to get DoubleBufferTransmitter");
        }
      } else {
        HOLOSCAN_LOG_ERROR("Connector is not a DoubleBufferTransmitter for port {}", port_name);
        throw std::runtime_error("Connector is not a DoubleBufferTransmitter");
      }
    }
  }

  // Operator's id is set to the GXF Codelet id (holoscan::gxf::GXFWrapper) so we can get the
  // pointer from the component ID.
  // Note: The GXFWrapper codelet is created in Operator::add_codelet_to_graph_entity() by
  // GXFExecutor::initialize_operator().

  // Check if operator is properly initialized. If not, the operator's id will be -1.
  if (op_->id() == -1) {
    auto message = fmt::format(
        "The Operator (name: {}) provided to OperatorRunner is not properly initialized. Ensure "
        "that the OperatorRunner object is created within the initialize() method of the umbrella "
        "operator, and that Operator::initialize() has been called beforehand.",
        op_->name());
    HOLOSCAN_LOG_ERROR(message);
    throw std::runtime_error(message);
  }

  // Get GXF wrapper component for the operator
  auto gxf_graph_entity = op_->graph_entity();
  if (gxf_graph_entity == nullptr) {
    throw std::runtime_error("GXF graph entity corresponding to the operator is not initialized");
  }
  auto codelet_handle = gxf_graph_entity->get_codelet();

  // Store wrapper pointer and set the operator
  gxf_wrapper_ = dynamic_cast<holoscan::gxf::GXFWrapper*>(codelet_handle.get());
  gxf_wrapper_->set_operator(op_.get());
}

const std::shared_ptr<holoscan::Operator>& OperatorRunner::op() const {
  return op_;
}

holoscan::expected<void, holoscan::RuntimeError> OperatorRunner::push_input(
    const std::string& port_name, nvidia::gxf::Entity& entity) {
  auto it = double_buffer_receivers_.find(port_name);
  if (it != double_buffer_receivers_.end()) {
    auto result = it->second->push(entity);
    if (!result) {
      HOLOSCAN_LOG_ERROR(
          "Failed to push input to operator {} - {}", op_->name(), result.get_error_message());
      return holoscan::unexpected<holoscan::RuntimeError>(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure, result.get_error_message()));
    }

    // Sync the input port with the DoubleBufferReceiver
    result = it->second->sync();
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to sync input port {} with DoubleBufferReceiver in operator {}",
                         port_name,
                         op_->name());
      return holoscan::unexpected<holoscan::RuntimeError>(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure, result.get_error_message()));
    }
    return {};  // return success
  }

  auto message =
      fmt::format("The input port ('{}') with DoubleBufferReceiver not found in the operator '{}'",
                  port_name,
                  op_->name());
  HOLOSCAN_LOG_DEBUG(message);
  return holoscan::unexpected<holoscan::RuntimeError>(
      holoscan::RuntimeError(holoscan::ErrorCode::kNotFound, message.c_str()));
}

holoscan::expected<void, holoscan::RuntimeError> OperatorRunner::push_input(
    const std::string& port_name, nvidia::gxf::Entity&& entity) {
  auto it = double_buffer_receivers_.find(port_name);
  if (it != double_buffer_receivers_.end()) {
    auto result = it->second->push(std::move(entity));
    if (!result) {
      HOLOSCAN_LOG_ERROR(
          "Failed to push input to operator {} - {}", op_->name(), result.get_error_message());
      return holoscan::unexpected<holoscan::RuntimeError>(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure, result.get_error_message()));
    }

    // Sync the input port with the DoubleBufferReceiver
    result = it->second->sync();
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to sync input port {} with DoubleBufferReceiver in operator {}",
                         port_name,
                         op_->name());
      return holoscan::unexpected<holoscan::RuntimeError>(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure, result.get_error_message()));
    }
    return {};  // return success
  }

  auto message =
      fmt::format("The input port ('{}') with DoubleBufferReceiver not found in the operator '{}'",
                  port_name,
                  op_->name());
  HOLOSCAN_LOG_DEBUG(message);
  return holoscan::unexpected<holoscan::RuntimeError>(
      holoscan::RuntimeError(holoscan::ErrorCode::kNotFound, message.c_str()));
}

holoscan::expected<void, holoscan::RuntimeError> OperatorRunner::push_input(
    const std::string& port_name, const holoscan::TensorMap& data) {
  auto out_message = holoscan::gxf::Entity::New(gxf_wrapper_->execution_context());
  for (auto& [key, tensor] : data) {
    out_message.add(tensor, key.c_str());
  }
  return push_input(port_name, out_message);
}

holoscan::expected<holoscan::gxf::Entity, holoscan::RuntimeError> OperatorRunner::pop_output(
    const std::string& port_name) {
  auto it = double_buffer_transmitters_.find(port_name);
  if (it != double_buffer_transmitters_.end()) {
    // Sync the output port with the DoubleBufferTransmitter
    it->second->sync();

    auto maybe_message = it->second->pop();
    if (maybe_message) {
      return holoscan::gxf::Entity(std::move(maybe_message.value()));
    } else {
      auto message = fmt::format(
          "Failed to pop message from output port '{}' using DoubleBufferTransmitter in the "
          "operator '{}'",
          port_name,
          op_->name());
      HOLOSCAN_LOG_DEBUG(message);
      return holoscan::unexpected<holoscan::RuntimeError>(
          holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, message.c_str()));
    }
  }
  auto message = fmt::format(
      "The output port ('{}') with DoubleBufferTransmitter not found in the operator '{}'",
      port_name,
      op_->name());
  HOLOSCAN_LOG_DEBUG(message);
  return holoscan::unexpected<holoscan::RuntimeError>(
      holoscan::RuntimeError(holoscan::ErrorCode::kNotFound, message.c_str()));
}

void OperatorRunner::run() {
  gxf_wrapper_->tick();
}

bool OperatorRunner::populate_tensor_map(const holoscan::gxf::Entity& gxf_entity,
                                         holoscan::TensorMap& tensor_map) {
  auto tensor_components_expected = gxf_entity.findAllHeap<nvidia::gxf::Tensor>();
  for (const auto& gxf_tensor : tensor_components_expected.value()) {
    // Do zero-copy conversion to holoscan::Tensor (as in gxf_entity.get<holoscan::Tensor>())
    auto maybe_dl_ctx = (*gxf_tensor->get()).toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR(
          "Failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
      return false;
    }
    auto holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());
    tensor_map.insert({gxf_tensor->name(), holoscan_tensor});
  }
  return true;
}

holoscan::expected<holoscan::TensorMap, holoscan::RuntimeError>
OperatorRunner::handle_tensor_map_output(const holoscan::gxf::Entity& entity,
                                         const std::string& port_name) {
  TensorMap tensor_map;
  if (!populate_tensor_map(entity, tensor_map)) {
    return create_error(
        holoscan::ErrorCode::kReceiveError,
        "Unable to populate the TensorMap from the received GXF Entity for output '{}'",
        port_name);
  }
  return tensor_map;
}

}  // namespace holoscan::ops
