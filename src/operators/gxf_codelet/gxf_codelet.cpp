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

#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"

#include <vector>
#include <memory>

#include "gxf/core/common_expected_macro.hpp"
#include "gxf/core/expected_macro.hpp"
#include "gxf/core/parameter_registrar.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::ops {

const char* GXFCodeletOp::gxf_typename() const {
  return gxf_typename_.c_str();
}

void GXFCodeletOp::setup(OperatorSpec& spec) {
  // Get the GXF context from the executor
  gxf_context_t gxf_context = fragment()->executor().context();

  // Get the type ID of the codelet
  gxf_tid_t codelet_tid;
  gxf_result_t result = GxfComponentTypeId(gxf_context, gxf_typename(), &codelet_tid);
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to find the GXF type name ('{}') for the codelet", gxf_typename());
    throw std::runtime_error(
        fmt::format("Unable to find the GXF type name ('{}') for the codelet", gxf_typename()));
  }

  // Create a new ComponentInfo object for the codelet
  gxf_component_info_ = std::make_shared<gxf::ComponentInfo>(gxf_context, codelet_tid);

  // Specify the input and output ports of the operator
  for (const auto& param : gxf_component_info_->receiver_parameters()) {
    spec.input<gxf::Entity>(param);
  }
  for (const auto& param : gxf_component_info_->transmitter_parameters()) {
    spec.output<gxf::Entity>(param);
  }

  // Set fake parameters for the codelet to be able to show the operator description
  // through the `OperatorSpec::description()` method
  auto& params = spec.params();
  auto& gxf_parameter_map = gxf_component_info_->parameter_info_map();

  for (const auto& key : gxf_component_info_->normal_parameters()) {
    const auto& gxf_param = gxf_parameter_map.at(key);
    ParameterFlag flag = static_cast<ParameterFlag>(gxf_param.flags);

    parameters_.emplace_back(nullptr, key, gxf_param.headline, gxf_param.description, flag);
    auto& parameter = parameters_.back();

    ParameterWrapper&& parameter_wrapper{
        &parameter, &typeid(void), gxf::ComponentInfo::get_arg_type(gxf_param), &parameter};
    params.try_emplace(key, parameter_wrapper);
  }
}

void GXFCodeletOp::set_parameters() {
  // Here, we don't call update_params_from_args().
  // Instead, we set the parameters manually using the GXF API.

  // Set the handle values for the receiver parameters
  auto& inputs = spec_->inputs();
  for (const auto& param_key : gxf_component_info_->receiver_parameters()) {
    // Get the parameter info and the receiver handle
    auto param_info = codelet_handle_->getParameterInfo(param_key).value();
    auto connector = inputs[param_key]->connector();
    auto gxf_connector = std::dynamic_pointer_cast<holoscan::gxf::GXFResource>(connector);
    auto receiver_handle = nvidia::gxf::Handle<nvidia::gxf::Component>::Create(
        gxf_connector->gxf_context(), gxf_connector->gxf_cid());

    // Set the parameter with the handle value
    codelet_handle_->setParameter(param_key, receiver_handle.value());
  }

  // Set the handle values for the transmitter parameters
  auto& outputs = spec_->outputs();
  for (const auto& param_key : gxf_component_info_->transmitter_parameters()) {
    // Get the parameter info and the transmitter handle
    auto param_info = codelet_handle_->getParameterInfo(param_key).value();
    auto connector = outputs[param_key]->connector();
    auto gxf_connector = std::dynamic_pointer_cast<holoscan::gxf::GXFResource>(connector);
    auto transmitter_handle = nvidia::gxf::Handle<nvidia::gxf::Component>::Create(
        gxf_connector->gxf_context(), gxf_connector->gxf_cid());

    // Set the parameter with the handle value
    codelet_handle_->setParameter(param_key, transmitter_handle.value());
  }

  // Set parameter values if they are specified in the arguments
  auto& parameter_map = gxf_component_info_->parameter_info_map();
  for (auto& arg : args_) {
    if (parameter_map.find(arg.name()) != parameter_map.end()) {
      holoscan::gxf::GXFParameterAdaptor::set_param(
          gxf_context(), gxf_cid(), arg.name().c_str(), arg.arg_type(), arg.value());
    }
  }
}

}  // namespace holoscan::ops
