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

#include "holoscan/core/io_context.hpp"

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gxf/core/entity.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/data_logger.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/profiler/profiler.hpp"

namespace holoscan {

namespace {

// Static helper to safely get inputs reference
static std::unordered_map<std::string, std::shared_ptr<IOSpec>>& get_safe_inputs(Operator* op) {
  static std::unordered_map<std::string, std::shared_ptr<IOSpec>> empty_inputs;
  if (op && op->spec()) {
    return op->spec()->inputs();
  }
  return empty_inputs;
}

}  // namespace

InputContext::InputContext(ExecutionContext* execution_context, Operator* op,
                           std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs)
    : execution_context_(execution_context), op_(op), inputs_(inputs) {
  prepopulate_acquisition_timestamp_map();
}

/**
 * @brief Construct a new InputContext object.
 *
 * inputs for the InputContext will be set to op->spec()->inputs()
 *
 * @param execution_context The pointer to GXF execution runtime
 * @param op The pointer to the operator that this context is associated with.
 */
InputContext::InputContext(ExecutionContext* execution_context, Operator* op)
    : execution_context_(execution_context), op_(op), inputs_(get_safe_inputs(op)) {
  // Pre-initialize acquisition timestamp map with all input port names
  // This avoids map insertions/deletions during runtime execution
  prepopulate_acquisition_timestamp_map();
}

/**
 * @brief Return whether the input port has any data.
 *
 * For parameters with std::vector<IOSpec*> type, if all the inputs are empty, it will return
 * true. Otherwise, it will return false.
 *
 * @param name The name of the input port to check.
 * @return True, if it has no data, otherwise false.
 */
bool InputContext::empty(const char* name) {
  // First see if the name could be found in the inputs
  auto& inputs = op_->spec()->inputs();
  auto it = inputs.find(std::string(name));
  if (it != inputs.end()) {
    return empty_impl(name);
  }

  // Then see if it is in the parameters
  auto& params = op_->spec()->params();
  auto it2 = params.find(std::string(name));
  if (it2 != params.end()) {
    auto& param_wrapper = it2->second;
    auto& arg_type = param_wrapper.arg_type();
    if ((arg_type.element_type() != ArgElementType::kIOSpec) ||
        (arg_type.container_type() != ArgContainerType::kVector)) {
      HOLOSCAN_LOG_ERROR("Input parameter with name '{}' is not of type 'std::vector<IOSpec*>'",
                         name);
      return true;
    }
    std::any& any_param = param_wrapper.value();
    // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
    auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(any_param);
    int num_inputs = param.get().size();
    for (int i = 0; i < num_inputs; ++i) {
      // if any of them is not empty return false
      if (!empty_impl(fmt::format("{}:{}", name, i).c_str())) {
        return false;
      }
    }
    return true;  // all of them are empty, so return true.
  }

  HOLOSCAN_LOG_ERROR("Input port '{}' not found", name);
  return true;
}

/**
 * @brief Reset acquisition timestamp map values to std::nullopt for all input ports.
 *
 * This method should be called before each compute call to reset the timestamp values
 * while preserving the pre-initialized map structure for performance.
 */
void InputContext::reset_acquisition_timestamps() {
  for (auto& [_, timestamps] : acquisition_timestamp_map_) {
    timestamps.clear();
  }
}

/**
 * @brief Get the acquisition timestamp for a given input port.
 *
 * @param input_port_name The name of the input port.
 * @return The acquisition timestamp. If no timestamp was published by the upstream operator,
 * std::nullopt is returned.
 */
std::optional<int64_t> InputContext::get_acquisition_timestamp(const char* input_port_name) {
  PROF_SCOPED_EVENT(op_->id(), event_get_acquisition_timestamp);
  std::string input_name = holoscan::get_well_formed_name(input_port_name, inputs_);
  if (inputs_.find(input_name) == inputs_.end()) {
    std::string err_msg = fmt::format("An input port with name '{}' is not found", input_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    return std::nullopt;
  }
  auto& timestamps = acquisition_timestamp_map_[input_name];
  if (timestamps.empty()) {
    return std::nullopt;
  }
  if (timestamps.size() > 1) {
    HOLOSCAN_LOG_INFO(
        "Input port '{}' has multiple timestamps, returning only the first one."
        "Use get_acquisition_timestamps() to get all timestamps.",
        input_name);
  }
  return timestamps.front();
}

std::vector<std::optional<int64_t>> InputContext::get_acquisition_timestamps(
    const char* input_port_name) {
  PROF_SCOPED_EVENT(op_->id(), event_get_acquisition_timestamps);
  std::string input_name = holoscan::get_well_formed_name(input_port_name, inputs_);
  if (inputs_.find(input_name) == inputs_.end()) {
    std::string err_msg = fmt::format("An input port with name '{}' is not found", input_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    return std::vector<std::optional<int64_t>>{};
  }
  return acquisition_timestamp_map_[input_name];
}

// Get unique_id for an input port
std::string InputContext::get_unique_id(Operator* op, const std::string& port_name) {
  if (!op->spec()) {
    throw std::runtime_error("Operator spec is not available");
  }

  auto& ports = op->spec()->inputs();
  auto it = ports.find(port_name);
  if (it != ports.end()) {
    return it->second->unique_id();
  }

  HOLOSCAN_LOG_WARN("Input port '{}' not found", port_name);
  return fmt::format("{}.{}", op->qualified_name(), port_name);
}

void InputContext::prepopulate_acquisition_timestamp_map() {
  // Pre-initialize acquisition timestamp map with all input port names
  // This avoids map insertions/deletions during runtime execution
  acquisition_timestamp_map_.clear();

  // only populate if operator and spec are valid
  if (!op_ || !op_->spec()) {
    return;
  }
  for (const auto& [input_port_name, _] : inputs_) {
    acquisition_timestamp_map_[input_port_name] = std::vector<std::optional<int64_t>>{};
  }
}

void OutputContext::emit(holoscan::TensorMap& data, const char* name, const int64_t acq_timestamp) {
  HOLOSCAN_LOG_TRACE("OutputContext::emit (TensorMap) for op: {}, name: {}",
                     op_->name(),
                     name == nullptr ? "nullptr" : name);
  std::string output_name = holoscan::get_well_formed_name(name, outputs_);
  auto output_it = outputs_.find(output_name);
  std::string unique_id;
  if (output_it != outputs_.end()) {
    unique_id = output_it->second->unique_id();
  } else {
    unique_id = fmt::format("{}.{}", op_->name(), name == nullptr ? "<unknown>" : name);
  }
  PROF_SCOPED_PORT_EVENT(op_->id(), unique_id, event_emit::color);

  if (!op_->fragment()->data_loggers().empty()) {
    log_tensormap(data, unique_id, output_name.c_str());
  }

  auto out_message = holoscan::gxf::Entity::New(execution_context_);
  for (auto& [key, tensor] : data) {
    out_message.add(tensor, key.c_str());
  }
  emit_impl(nvidia::gxf::Entity(out_message), name, OutputType::kGXFEntity, acq_timestamp, true);
}

void OutputContext::emit(std::shared_ptr<holoscan::Tensor> data, const char* name,
                         const int64_t acq_timestamp) {
  HOLOSCAN_LOG_TRACE("OutputContext::emit (std::shared_ptr<holoscan::Tensor>) for op: {}, name: {}",
                     op_->name(),
                     name == nullptr ? "nullptr" : name);
  std::string output_name = holoscan::get_well_formed_name(name, outputs_);
  auto output_it = outputs_.find(output_name);
  std::string unique_id;
  if (output_it != outputs_.end()) {
    unique_id = output_it->second->unique_id();
  } else {
    unique_id = fmt::format("{}.{}", op_->name(), name == nullptr ? "<unknown>" : name);
  }
  PROF_SCOPED_PORT_EVENT(op_->id(), unique_id, event_emit::color);

  if (!op_->fragment()->data_loggers().empty()) {
    log_tensor(data, unique_id, output_name.c_str());
  }

  auto out_message = holoscan::gxf::Entity::New(execution_context_);
  out_message.add(data, "");
  emit_impl(nvidia::gxf::Entity(out_message), name, OutputType::kGXFEntity, acq_timestamp, true);
}

bool OutputContext::log_tensor(const std::shared_ptr<Tensor>& tensor, const std::string& unique_id,
                               const char* port_name) {
  PROF_SCOPED_EVENT(op_->id(), event_data_logging);
  auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;

  // Check if a CUDA stream is being emitted on this output port
  auto stream_for_logging = stream_to_emit(port_name);

  for (auto& data_logger : op_->fragment()->data_loggers()) {
    if (data_logger->should_log_output()) {
      PROF_SCOPED_EVENT(op_->id(), event_log_tensor);
      data_logger->log_tensor_data(
          tensor, unique_id, -1, metadata_ptr, IOSpec::IOType::kOutput, stream_for_logging);
    }
  }
  return true;
}

bool OutputContext::log_tensormap(const holoscan::TensorMap& tensor_map,
                                  const std::string& unique_id, const char* port_name) {
  PROF_SCOPED_EVENT(op_->id(), event_data_logging);
  auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;

  // Check if a CUDA stream is being emitted on this output port
  auto stream_for_logging = stream_to_emit(port_name);

  for (auto& data_logger : op_->fragment()->data_loggers()) {
    if (data_logger->should_log_output()) {
      PROF_SCOPED_EVENT(op_->id(), event_log_tensormap);
      data_logger->log_tensormap_data(
          tensor_map, unique_id, -1, metadata_ptr, IOSpec::IOType::kOutput, stream_for_logging);
    }
  }
  return true;
}

}  // namespace holoscan
