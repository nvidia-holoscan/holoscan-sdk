/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

void GXFOperator::initialize() {
  // Call base class initialize function.
  Operator::initialize();
}

gxf_uid_t GXFOperator::add_codelet_to_graph_entity() {
  HOLOSCAN_LOG_TRACE("calling graph_entity()->addCodelet for {}", name_);
  if (!graph_entity_) { throw std::runtime_error("graph entity is not initialized"); }
  auto codelet_handle = graph_entity_->addCodelet(gxf_typename(), name_.c_str());
  if (!codelet_handle) {
    throw std::runtime_error("Failed to add codelet of type " + std::string(gxf_typename()));
  }
  gxf_uid_t codelet_cid = codelet_handle->cid();
  gxf_eid_ = graph_entity_->eid();
  gxf_cid_ = codelet_cid;
  gxf_context_ = graph_entity_->context();
  HOLOSCAN_LOG_TRACE("\tadded codelet with cid = {}", codelet_handle->cid());
  return codelet_cid;
}

void GXFOperator::set_parameters() {
  update_params_from_args();

  // Set Handler parameters
  for (auto& [key, param_wrap] : spec_->params()) {
    HOLOSCAN_GXF_CALL_WARN_MSG(::holoscan::gxf::GXFParameterAdaptor::set_param(
                                   gxf_context_, gxf_cid_, key.c_str(), param_wrap),
                               "GXFOperator '{}':: failed to set GXF parameter '{}'",
                               name_,
                               key);
    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting GXF parameter '{}'", name_, key);
  }
}

}  // namespace holoscan::ops
