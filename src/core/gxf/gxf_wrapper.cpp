/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/gxf_wrapper.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "holoscan/core/common.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/io_context.hpp"

#include "gxf/std/transmitter.hpp"

namespace holoscan::gxf {

gxf_result_t GXFWrapper::initialize() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::initialize()");
  PROF_REGISTER_CATEGORY(op_->id(), op_->name().c_str());
  return GXF_SUCCESS;
}
gxf_result_t GXFWrapper::deinitialize() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::deinitialize()");
  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::registerInterface(nvidia::gxf::Registrar* registrar) {
  HOLOSCAN_LOG_TRACE("GXFWrapper::registerInterface()");
  (void)registrar;
  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::start() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::start()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFWrapper::start() - Operator is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Starting operator: {}", op_->name());

  try {
    PROF_SCOPED_EVENT(op_->id(), event_start);
    op_->start();
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR(
        "Exception occurred when starting operator: '{}' - {}", op_->name(), e.what());
    return GXF_FAILURE;
  }

  initialize_contexts();
  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::tick() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::tick()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFWrapper::tick() - Operator is not set");
    return GXF_FAILURE;
  }
  // Ensure the contexts are initialized when used by `OperatorWrapper` as part of the GXF app
  if (exec_context_ == nullptr) { initialize_contexts(); }

  // clear any existing values from a previous compute call
  op_->metadata()->clear();

  // clear any received streams from previous compute call
  exec_context_->clear_received_streams();

  // Handle the signal if the operator has input execution port
  if (op_->input_exec_spec()) {
    HOLOSCAN_LOG_TRACE("Handling input execution port for operator: '{}'", op_->name());
    auto connector = op_->input_exec_spec()->connector();
    if (connector) {
      auto gxf_receiver =
          std::dynamic_pointer_cast<holoscan::DoubleBufferReceiver>(connector)->get();
      if (gxf_receiver) {
        auto queue_size = gxf_receiver->size();
        for (size_t i = 0; i < queue_size; ++i) {
          // Pop the entity from the input execution port so that it doesn't get processed by
          // GXF, and applying metadata to the operator.
          op_input_->receive<holoscan::gxf::Entity>(Operator::kInputExecPortName);
          // Note: We may want to introduce 'reducer' concept and apply the reducer logic here
          //       in the future.
        }
      }
    }
  }
  // Clear if dynamic flows are present
  if (op_->dynamic_flows()) { op_->dynamic_flows()->clear(); }

  HOLOSCAN_LOG_TRACE("Calling operator: {}", op_->name());
  try {
    PROF_SCOPED_EVENT(op_->id(), event_compute);
    op_->compute(*op_input_, *op_output_, *exec_context_);
  } catch (const std::exception& e) {
    // Note: Rethrowing the exception (using `throw;`) would cause the Python interpreter to exit.
    //       To avoid this, we store the exception and return GXF_FAILURE.
    //       The exception is then rethrown in GXFExecutor::run_gxf_graph().
    store_exception();
    HOLOSCAN_LOG_ERROR("Exception occurred for operator: '{}' - {}", op_->name(), e.what());
    return GXF_FAILURE;
  }

  // Call the dynamic flow function if it is set
  const auto& dynamic_flow_func = op_->dynamic_flow_func();
  if (dynamic_flow_func) {
    auto op_shared = op_->self_shared();
    dynamic_flow_func(op_shared);
  }

  if (dynamic_flow_func || op_->dynamic_flows()) {
    holoscan::gxf::Entity entity;

    std::unordered_map<std::string, holoscan::gxf::Entity> output_entity_map;

    // Conditionally set the entity to the downstream operators
    for (const auto& flow_info : *op_->dynamic_flows()) {
      auto it = output_entity_map.find(flow_info->output_port_name);
      if (it != output_entity_map.end()) {
        entity = it->second;
      } else {
        auto op_output_connector = flow_info->output_port_spec->connector();
        if (op_output_connector) {
          auto gxf_output_transmitter =
              std::dynamic_pointer_cast<holoscan::DoubleBufferTransmitter>(op_output_connector)
                  ->get();
          if (gxf_output_transmitter) {
            gxf_output_transmitter->sync();
            if (gxf_output_transmitter->size() == 0) {
              if (flow_info->output_port_name == Operator::kOutputExecPortName) {
                // Create a new entity if no data is available for the output execution port
                auto signal_entity = holoscan::gxf::Entity(
                    nvidia::gxf::Entity::New(op_->fragment()->executor().context()).value());

                // Push the output data (signal) to the output execution port
                op_output_->emit(signal_entity, Operator::kOutputExecPortName);
                // Sync the transmitter again to make the data pushed available for the output
                // execution port
                gxf_output_transmitter->sync();
              } else {
                HOLOSCAN_LOG_ERROR("No data available for the output port '{}' of {}",
                                   flow_info->output_port_name,
                                   op_->name());
                return GXF_FAILURE;
              }
            }
            // Overwrite the entity with the value from the output execution port.
            entity = holoscan::gxf::Entity(gxf_output_transmitter->pop().value());
            // Store the entity in the map
            output_entity_map[flow_info->output_port_name] = entity;
          } else {
            HOLOSCAN_LOG_ERROR("Unable to cast the output connector to DoubleBufferTransmitter for "
                                "the output port '{}' of {}",
                                flow_info->output_port_name,
                                op_->name());
            return GXF_FAILURE;
          }
        } else {
          HOLOSCAN_LOG_ERROR("Invalid connector for the output port '{}' of {}",
                             flow_info->output_port_name,
                             op_->name());
          return GXF_FAILURE;
        }
      }

      // Process for the downstream operators
      auto connector = flow_info->input_port_spec->connector();
      if (connector) {
        auto gxf_receiver =
            std::dynamic_pointer_cast<holoscan::DoubleBufferReceiver>(connector)->get();
        if (gxf_receiver) {
          gxf_receiver->push(entity);
          // Send event to trigger execution of downstream entities.
          GXF_LOG_VERBOSE("Notifying downstream receiver with eid '%ld'.", gxf_receiver->eid());
          GxfEntityNotifyEventType(context(), gxf_receiver->eid(), GXF_EVENT_MESSAGE_SYNC);
        }
      }
    }

    // Pop the entity from the output execution port if there is no dynamic flow for the output
    // execution port
    auto& op_outputs = op_->spec()->outputs();

    for (const auto& [output_port_name, output_port_spec] : op_outputs) {
      if (output_entity_map.find(output_port_name) == output_entity_map.end()) {
        auto op_output_connector = output_port_spec->connector();
        if (op_output_connector) {
          auto gxf_output_transmitter =
              std::dynamic_pointer_cast<holoscan::DoubleBufferTransmitter>(op_output_connector)
                  ->get();
          if (gxf_output_transmitter) {
            gxf_output_transmitter->sync();
            auto queue_size = gxf_output_transmitter->size();
            for (size_t i = 0; i < queue_size; ++i) {
              // Pop the entity from the output execution port so that it doesn't get processed by
              // the GXF.
              gxf_output_transmitter->pop();
            }
          }
        }
      }
    }
  } else {
    // If no dynamic flow function is set, signal to the downstream operators if the operator
    // has output execution port
    if (op_->output_exec_spec()) {
      // Push the output data (signal) to the output execution port
      auto entity = holoscan::gxf::Entity(
          nvidia::gxf::Entity::New(op_->fragment()->executor().context()).value());
      op_output_->emit(entity, Operator::kOutputExecPortName);
    }
  }

  // Note: output metadata is inserted via op_output.emit() rather than here
  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::stop() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::stop()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFWrapper::stop() - Operator is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Stopping operator: {}", op_->name());

  try {
    PROF_SCOPED_EVENT(op_->id(), event_stop);
    op_->stop();
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR(
        "Exception occurred when stopping operator: '{}' - {}", op_->name(), e.what());
    return GXF_FAILURE;
  }

  exec_context_->release_internal_cuda_streams();

  return GXF_SUCCESS;
}

void GXFWrapper::store_exception() {
  auto stored_exception = std::current_exception();
  if (stored_exception != nullptr) { op_->fragment()->executor().exception(stored_exception); }
}

void GXFWrapper::initialize_contexts() {
  if (!exec_context_) {
    // Initialize the execution context
    exec_context_ = std::make_unique<GXFExecutionContext>(context(), op_);
    exec_context_->init_cuda_object_handler(op_);
    HOLOSCAN_LOG_TRACE("GXFWrapper: exec_context_->cuda_object_handler() for op '{}' is {}null",
                       op_->name(),
                       exec_context_->cuda_object_handler() == nullptr ? "" : "not ");
    op_input_ = exec_context_->input();
    op_input_->cuda_object_handler(exec_context_->cuda_object_handler());
    op_output_ = exec_context_->output();
    op_output_->cuda_object_handler(exec_context_->cuda_object_handler());
  }
}
}  // namespace holoscan::gxf
