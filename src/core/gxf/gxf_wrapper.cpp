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

#include <fmt/format.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "holoscan/core/common.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/profiler/nvtx3.hpp"
#include "holoscan/profiler/profiler.hpp"

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
  PROF_SCOPED_EVENT(op_->id(), event_tick);

  // Ensure the contexts are initialized when used by `OperatorWrapper` as part of the GXF app
  if (exec_context_ == nullptr) {
    initialize_contexts();
  }

  // clear any existing values from a previous compute call
  {
    PROF_SCOPED_EVENT(op_->id(), event_metadata_clear);
    op_->metadata()->clear();
  }

  // clear any received streams from previous compute call
  {
    PROF_SCOPED_EVENT(op_->id(), event_clear_streams);
    exec_context_->clear_received_streams();
  }

  // reset acquisition timestamps for all ports to std::nullopt
  {
    PROF_SCOPED_EVENT(op_->id(), event_clear_acquisition_timestamps);
    op_input_->reset_acquisition_timestamps();
  }

  // Handle the signal if the operator has input execution port
  if (auto& input_exec_spec = op_->input_exec_spec()) {
    HOLOSCAN_LOG_TRACE("Handling input execution port for operator: '{}'", op_->name());
    auto connector = input_exec_spec->connector();
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
  if (op_->dynamic_flows()) {
    op_->dynamic_flows()->clear();
  }

  HOLOSCAN_LOG_TRACE("Calling operator: {}", op_->name());
  try {
    // Only create custom NVTX events when both features are enabled
    if (op_->fragment()->data_flow_tracker() && holoscan::profiler::trace_enabled()) {
      // Create custom NVTX event - just "compute" without frame info
      std::string message_str = event_compute::message;

      {
        holoscan::profiler::scoped_range p{
            PROF_CATEGORY(op_->id()), nvtx3::message{message_str.c_str()}, event_compute::color};

        op_->compute(*op_input_, *op_output_, *exec_context_);

        // Create post-compute NVTX child range with frame information
        create_post_compute_nvtx_range();
      }
    } else {
      // Fall back to simple compute call without NVTX frame tracking
      op_->compute(*op_input_, *op_output_, *exec_context_);
    }
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
            HOLOSCAN_LOG_ERROR(
                "Unable to cast the output connector to DoubleBufferTransmitter for "
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
  op_->release_internal_resources();

  return GXF_SUCCESS;
}

void GXFWrapper::set_operator(Operator* op) {
  op_ = op;
}

Operator* GXFWrapper::op() const {
  return op_;
}

GXFExecutionContext* GXFWrapper::execution_context() const {
  return static_cast<GXFExecutionContext*>(exec_context_);
}

InputContext* GXFWrapper::input_context() const {
  return op_input_;
}

OutputContext* GXFWrapper::output_context() const {
  return op_output_;
}

void GXFWrapper::store_exception() {
  auto stored_exception = std::current_exception();
  if (stored_exception != nullptr) {
    op_->fragment()->executor().exception(stored_exception);
  }
}

void GXFWrapper::initialize_contexts() {
  if (!exec_context_) {
    // Ensure the contexts for the operator
    op_->ensure_contexts();

    // Initialize the execution context
    exec_context_ = static_cast<GXFExecutionContext*>(op_->execution_context().get());
    op_input_ = exec_context_->input().get();
    op_output_ = exec_context_->output().get();
  }
}

void GXFWrapper::create_post_compute_nvtx_range() {
  // Add post-compute child NVTX range to show actual frame numbers after processing
  // Conditions are already checked before calling this function
  HOLOSCAN_LOG_DEBUG("POST-COMPUTE DEBUG: Starting post-compute frame collection for operator '{}'",
                     op_->name());

  // Determine operator type and get frame information
  bool is_current_op_root = is_root_operator();

  std::string processed_frame_info =
      is_current_op_root ? get_root_operator_frame_info() : get_non_root_operator_frame_info();

  // Create NVTX child range if we have frame info
  if (!processed_frame_info.empty()) {
    std::string post_compute_marker = fmt::format("{}{}", op_->name(), processed_frame_info);

    // Create a child range inside the main compute range
    {
      holoscan::profiler::scoped_range post_range{PROF_CATEGORY(op_->id()),
                                                  nvtx3::message{post_compute_marker.c_str()},
                                                  event_compute::color};
      // Range duration depends on debug logging and natural overhead
    }

    HOLOSCAN_LOG_DEBUG("POST-COMPUTE SUCCESS: Created NVTX child range '{}' for operator '{}'",
                       post_compute_marker,
                       op_->name());
  } else {
    HOLOSCAN_LOG_DEBUG(
        "POST-COMPUTE DEBUG: Operator '{}' -> no frame numbers available after compute",
        op_->name());
  }
}


bool GXFWrapper::is_root_operator() const {
  return op_->is_root() || op_->is_user_defined_root() ||
         holoscan::Operator::is_all_operator_predecessor_virtual(
             std::shared_ptr<holoscan::Operator>(op_, [](Operator*) {}), op_->fragment()->graph());
}

std::string GXFWrapper::get_root_operator_frame_info() const {
  // For root operators: get port-specific frame numbers from DataFlowTracker
  auto data_flow_tracker = op_->fragment()->data_flow_tracker();
  auto port_frame_numbers = data_flow_tracker->get_port_frame_numbers(op_->qualified_name());

  HOLOSCAN_LOG_DEBUG("POST-COMPUTE DEBUG: Root operator '{}' has {} port frame numbers",
                     op_->name(),
                     port_frame_numbers.size());

  if (port_frame_numbers.empty()) {
    return "";
  }

  // port_frame_numbers already contains operator-port keys, use directly
  fmt::memory_buffer processed_buffer;
  fmt::format_to(std::back_inserter(processed_buffer), ": frames ");
  bool processed_first = true;
  for (const auto& [operator_port_key, frame_number] : port_frame_numbers) {
    HOLOSCAN_LOG_DEBUG("POST-COMPUTE DEBUG: Operator '{}' post-compute frame from '{}': {}",
                       op_->name(),
                       operator_port_key,
                       frame_number);
    fmt::format_to(std::back_inserter(processed_buffer),
                   "{}{}:{}",
                   processed_first ? "" : " ",
                   operator_port_key,
                   frame_number);
    processed_first = false;
  }
  return fmt::to_string(processed_buffer);
}

std::string GXFWrapper::get_non_root_operator_frame_info() const {
  // For non-root operators: get frame numbers from input messages
  try {
    const auto& processed_frame_numbers = op_->get_consolidated_input_label().get_frame_numbers();
    HOLOSCAN_LOG_DEBUG("POST-COMPUTE DEBUG: Operator '{}' has {} post-compute frame numbers",
                       op_->name(),
                       processed_frame_numbers.size());

    if (processed_frame_numbers.empty()) {
      return "";
    }

    // Show all frames with their operator-port keys (handles both single and multiple)
    fmt::memory_buffer processed_buffer;
    fmt::format_to(std::back_inserter(processed_buffer), ": frames ");
    bool processed_first = true;
    for (const auto& [operator_port_key, frame_number] : processed_frame_numbers) {
      HOLOSCAN_LOG_DEBUG("POST-COMPUTE DEBUG: Operator '{}' post-compute frame from '{}': {}",
                         op_->name(),
                         operator_port_key,
                         frame_number);
      fmt::format_to(std::back_inserter(processed_buffer),
                     "{}{}:{}",
                     processed_first ? "" : " ",
                     operator_port_key,
                     frame_number);
      processed_first = false;
    }
    return fmt::to_string(processed_buffer);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_DEBUG(
        "POST-COMPUTE ERROR: Failed to get post-compute frame numbers for operator "
        "'{}': {}",
        op_->name(),
        e.what());
    return "";
  }
}

}  // namespace holoscan::gxf
