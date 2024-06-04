/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/gxf_component_info.hpp"

#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::gxf {

ComponentInfo::ComponentInfo(gxf_context_t context, gxf_tid_t tid)
    : gxf_context_(context), component_tid_(tid) {
  // Initialize component info
  component_info_.parameters = new const char*[MAX_PARAM_COUNT];
  component_info_.num_parameters = MAX_PARAM_COUNT;
  // This call will set the true number of parameters (will error if it exceeds MAX_PARAM_COUNT)
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentInfo(gxf_context_, component_tid_, &component_info_));

  // Get the number of parameters
  const auto num_parameters = component_info_.num_parameters;

  // Allocate space for parameter info
  parameter_infos_.resize(num_parameters);
  parameter_info_map_.reserve(num_parameters);
  transmitter_parameters_.reserve(num_parameters);
  receiver_parameters_.reserve(num_parameters);
  normal_parameters_.reserve(num_parameters);

  // Process each parameter
  for (size_t i = 0; i < num_parameters; ++i) {
    // Add the parameter key to the list
    parameter_keys_.push_back(component_info_.parameters[i]);

    // Get the parameter info and add it to the map
    HOLOSCAN_GXF_CALL_FATAL(GxfGetParameterInfo(
        gxf_context_, component_tid_, component_info_.parameters[i], &parameter_infos_[i]));
    parameter_info_map_[std::string(component_info_.parameters[i])] = parameter_infos_[i];

    // Determine if the parameter is a transmitter or receiver
    bool is_transmitter = false;
    bool is_receiver = false;
    if (parameter_infos_[i].type == GXF_PARAMETER_TYPE_HANDLE) {
      // Check if the parameter is a transmitter
      if (parameter_infos_[i].handle_tid == transmitter_tid()) {
        is_transmitter = true;
      } else {
        HOLOSCAN_GXF_CALL_FATAL(GxfComponentIsBase(
            gxf_context_, parameter_infos_[i].handle_tid, transmitter_tid(), &is_transmitter));
      }

      // If not a transmitter, check if the parameter is a receiver
      if (!is_transmitter && parameter_infos_[i].handle_tid == receiver_tid()) {
        is_receiver = true;
      } else if (!is_transmitter) {
        HOLOSCAN_GXF_CALL_FATAL(GxfComponentIsBase(
            gxf_context_, parameter_infos_[i].handle_tid, receiver_tid(), &is_receiver));
      }
    }

    // Add the parameter to the appropriate list
    if (is_transmitter) {
      transmitter_parameters_.push_back(component_info_.parameters[i]);
    } else if (is_receiver) {
      receiver_parameters_.push_back(component_info_.parameters[i]);
    } else {
      normal_parameters_.push_back(component_info_.parameters[i]);
    }
  }
}

ComponentInfo::~ComponentInfo() {
  if (component_info_.parameters) { delete[] component_info_.parameters; }
}

ArgType ComponentInfo::get_arg_type(const gxf_parameter_info_t& param) {
  ArgElementType element_type = ArgElementType::kCustom;

  switch (param.type) {
    case GXF_PARAMETER_TYPE_CUSTOM:
      element_type = ArgElementType::kCustom;
      break;
    case GXF_PARAMETER_TYPE_HANDLE:
      element_type = ArgElementType::kResource;
      break;
    case GXF_PARAMETER_TYPE_STRING:
      element_type = ArgElementType::kString;
      break;
    case GXF_PARAMETER_TYPE_INT64:
      element_type = ArgElementType::kInt64;
      break;
    case GXF_PARAMETER_TYPE_UINT64:
      element_type = ArgElementType::kUnsigned64;
      break;
    case GXF_PARAMETER_TYPE_FLOAT64:
      element_type = ArgElementType::kFloat64;
      break;
    case GXF_PARAMETER_TYPE_BOOL:
      element_type = ArgElementType::kBoolean;
      break;
    case GXF_PARAMETER_TYPE_INT32:
      element_type = ArgElementType::kInt32;
      break;
    case GXF_PARAMETER_TYPE_FILE:
      element_type = ArgElementType::kString;
      break;
    case GXF_PARAMETER_TYPE_INT8:
      element_type = ArgElementType::kInt8;
      break;
    case GXF_PARAMETER_TYPE_INT16:
      element_type = ArgElementType::kInt16;
      break;
    case GXF_PARAMETER_TYPE_UINT8:
      element_type = ArgElementType::kUnsigned8;
      break;
    case GXF_PARAMETER_TYPE_UINT16:
      element_type = ArgElementType::kUnsigned16;
      break;
    case GXF_PARAMETER_TYPE_UINT32:
      element_type = ArgElementType::kUnsigned32;
      break;
    case GXF_PARAMETER_TYPE_FLOAT32:
      element_type = ArgElementType::kFloat32;
      break;
    case GXF_PARAMETER_TYPE_COMPLEX64:
      element_type = ArgElementType::kComplex64;
      break;
    case GXF_PARAMETER_TYPE_COMPLEX128:
      element_type = ArgElementType::kComplex128;
      break;
    default:
      element_type = ArgElementType::kCustom;
      break;
  }
  ArgContainerType container_type = ArgContainerType::kNative;
  int32_t dimension = param.rank;
  if (dimension > 0) {
    if (param.shape[0] != -1) {
      container_type = ArgContainerType::kArray;
    } else {
      container_type = ArgContainerType::kVector;
    }
  }
  return ArgType{element_type, container_type, dimension};
}

gxf_tid_t ComponentInfo::receiver_tid() const {
  static gxf_tid_t receiver_tid = GxfTidNull();
  if (receiver_tid == GxfTidNull()) {
    HOLOSCAN_GXF_CALL_FATAL(
        GxfComponentTypeId(gxf_context_, "nvidia::gxf::Receiver", &receiver_tid));
  }

  return receiver_tid;
}

gxf_tid_t ComponentInfo::transmitter_tid() const {
  static gxf_tid_t transmitter_tid = GxfTidNull();
  if (transmitter_tid == GxfTidNull()) {
    HOLOSCAN_GXF_CALL_FATAL(
        GxfComponentTypeId(gxf_context_, "nvidia::gxf::Transmitter", &transmitter_tid));
  }

  return transmitter_tid;
}

const gxf_component_info_t& ComponentInfo::component_info() const {
  return component_info_;
}

const std::vector<const char*>& ComponentInfo::parameter_keys() const {
  return parameter_keys_;
}

const std::vector<gxf_parameter_info_t>& ComponentInfo::parameter_infos() const {
  return parameter_infos_;
}

const std::unordered_map<std::string, gxf_parameter_info_t>& ComponentInfo::parameter_info_map()
    const {
  return parameter_info_map_;
}

const std::vector<const char*>& ComponentInfo::receiver_parameters() const {
  return receiver_parameters_;
}

const std::vector<const char*>& ComponentInfo::transmitter_parameters() const {
  return transmitter_parameters_;
}

const std::vector<const char*>& ComponentInfo::normal_parameters() const {
  return normal_parameters_;
}

}  // namespace holoscan::gxf
