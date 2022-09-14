/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/gxf_operator.hpp"

#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_condition.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/double_buffer_transmitter.hpp"

namespace holoscan::ops {

void GXFOperator::initialize() {
  holoscan::Operator::initialize();
  gxf_context_ = fragment()->executor().context();

  if (!spec_) {
    HOLOSCAN_LOG_ERROR("No operator spec for GXFOperator '{}'", name());
    return;
  }

  auto& spec = *spec_;

  // Create Entity for this Operator
  gxf_uid_t eid;
  gxf_result_t code;
  const GxfEntityCreateInfo entity_create_info = {name().c_str(), GXF_ENTITY_CREATE_PROGRAM_BIT};
  code = GxfCreateEntity(gxf_context_, &entity_create_info, &eid);

  // Create Codelet component
  gxf_tid_t codelet_tid;
  gxf_uid_t codelet_cid;
  code = GxfComponentTypeId(gxf_context_, gxf_typename(), &codelet_tid);
  code = GxfComponentAdd(gxf_context_, eid, codelet_tid, name().c_str(), &codelet_cid);

  // Create Components for input
  const auto& inputs = spec.inputs();

  for (const auto& [name, io_spec] : inputs) {
    // Create Receiver component for this input
    const char* rx_name = name.c_str();

    auto rx_resource = std::make_shared<DoubleBufferReceiver>();
    rx_resource->name(rx_name);
    rx_resource->fragment(fragment());
    auto rx_spec = std::make_unique<ComponentSpec>(fragment());
    rx_resource->setup(*rx_spec.get());
    rx_resource->spec(std::move(rx_spec));

    rx_resource->gxf_eid(eid);
    rx_resource->initialize();

    gxf_uid_t rx_cid = rx_resource->gxf_cid();
    io_spec->resource(rx_resource);

    // Create SchedulingTerm component for this input
    if (io_spec->conditions().empty()) {
      // Default scheduling term for input:
      //   .condition(ConditionType::kMessageAvailable, Arg("min_size") = 1);
      gxf_tid_t term_tid;
      code = GxfComponentTypeId(
          gxf_context_, "nvidia::gxf::MessageAvailableSchedulingTerm", &term_tid);
      gxf_uid_t term_cid;
      code = GxfComponentAdd(gxf_context_, eid, term_tid, "__condition_input", &term_cid);
      code = GxfParameterSetHandle(gxf_context_, term_cid, "receiver", rx_cid);
      code = GxfParameterSetUInt64(gxf_context_, term_cid, "min_size", 1);
    } else {
      int condition_index = 0;
      for (const auto& [condition_type, condition] : io_spec->conditions()) {
        ++condition_index;
        if (condition_type == ConditionType::kMessageAvailable) {
          std::shared_ptr<MessageAvailableCondition> message_available_condition =
              std::dynamic_pointer_cast<MessageAvailableCondition>(condition);

          message_available_condition->receiver(rx_resource);
          message_available_condition->name(
              ::holoscan::gxf::create_name("__condition_input_", condition_index).c_str());
          message_available_condition->fragment(fragment());
          auto rx_condition_spec = std::make_unique<ComponentSpec>(fragment());
          message_available_condition->setup(*rx_condition_spec.get());
          message_available_condition->spec(std::move(rx_condition_spec));

          message_available_condition->gxf_eid(eid);
          message_available_condition->initialize();
        } else {
          throw std::runtime_error("Unsupported condition type");  // TODO: use std::expected
        }
      }
    }
  }

  // Create Components for output
  const auto& outputs = spec.outputs();

  for (const auto& [name, io_spec] : outputs) {
    // Create Transmitter component for this output
    const char* tx_name = name.c_str();

    auto tx_resource = std::make_shared<DoubleBufferTransmitter>();
    tx_resource->name(tx_name);
    tx_resource->fragment(fragment());
    auto tx_spec = std::make_unique<ComponentSpec>(fragment());
    tx_resource->setup(*tx_spec.get());
    tx_resource->spec(std::move(tx_spec));

    tx_resource->gxf_eid(eid);
    tx_resource->initialize();

    gxf_uid_t tx_cid = tx_resource->gxf_cid();
    io_spec->resource(tx_resource);

    // Create SchedulingTerm component for this output
    if (io_spec->conditions().empty()) {
      // Default scheduling term for output:
      //   .condition(ConditionType::kDownstreamMessageAffordable, Arg("min_size") = 1);
      gxf_tid_t term_tid;
      code = GxfComponentTypeId(
          gxf_context_, "nvidia::gxf::DownstreamReceptiveSchedulingTerm", &term_tid);
      gxf_uid_t term_cid;
      code = GxfComponentAdd(gxf_context_, eid, term_tid, "__condition_output", &term_cid);
      code = GxfParameterSetHandle(gxf_context_, term_cid, "transmitter", tx_cid);
      code = GxfParameterSetUInt64(gxf_context_, term_cid, "min_size", 1);
    } else {
      int condition_index = 0;
      for (const auto& [condition_type, condition] : io_spec->conditions()) {
        ++condition_index;
        if (condition_type == ConditionType::kDownstreamMessageAffordable) {
          std::shared_ptr<DownstreamMessageAffordableCondition>
              downstream_msg_affordable_condition =
                  std::dynamic_pointer_cast<DownstreamMessageAffordableCondition>(condition);

          downstream_msg_affordable_condition->transmitter(tx_resource);
          downstream_msg_affordable_condition->name(
              ::holoscan::gxf::create_name("__condition_input_", condition_index).c_str());
          downstream_msg_affordable_condition->fragment(fragment());
          auto tx_condition_spec = std::make_unique<ComponentSpec>(fragment());
          downstream_msg_affordable_condition->setup(*tx_condition_spec.get());
          downstream_msg_affordable_condition->spec(std::move(tx_condition_spec));

          downstream_msg_affordable_condition->gxf_eid(eid);
          downstream_msg_affordable_condition->initialize();
        } else {
          throw std::runtime_error("Unsupported condition type");  // TODO: use std::expected
        }
      }
    }
  }

  // Create Components for condition
  for (const auto& [name, condition] : conditions()) {
    auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
    // Initialize GXF component if it is not already initialized.
    if (gxf_condition->gxf_context() == nullptr) {
      gxf_condition->fragment(fragment());

      gxf_condition->gxf_eid(eid);  // set GXF entity id
      gxf_condition->initialize();
    }
  }

  // Create Components for resource
  for (const auto& [name, resource] : resources()) {
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // Initialize GXF component if it is not already initialized.
    if (gxf_resource->gxf_context() == nullptr) {
      gxf_resource->fragment(fragment());

      gxf_resource->gxf_eid(eid);  // set GXF entity id
      gxf_resource->initialize();
    }
  }

  // Set arguments
  auto& params = spec.params();
  for (auto& arg : args_) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' is not defined in spec", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting argument '{}'", name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters
  for (auto& [key, param_wrap] : params) {
    code = ::holoscan::gxf::GXFParameterAdaptor::set_param(
        gxf_context_, codelet_cid, key.c_str(), param_wrap);
    // TODO: handle error
    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting GXF parameter '{}'", name(), key);
  }
  (void)code;
}

}  // namespace holoscan::ops