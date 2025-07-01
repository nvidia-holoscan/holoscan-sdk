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

#include "holoscan/core/distributed/common/virtual_operator.hpp"
#include "holoscan/core/gxf/entity.hpp"

namespace holoscan::ops {

void VirtualOperator::initialize() {
  // We do not call the base class initialize function.
  // Operator::initialize();
}

IOSpec* VirtualOperator::input_spec() {
  if (spec_ == nullptr) { return nullptr; }
  if (input_spec_ == nullptr && io_type_ == IOSpec::IOType::kInput) {
    input_spec_ = spec_->inputs()[port_name_].get();
  }
  return input_spec_;
}

IOSpec* VirtualOperator::output_spec() {
  if (spec_ == nullptr) { return nullptr; }
  if (output_spec_ == nullptr && io_type_ == IOSpec::IOType::kOutput) {
    output_spec_ = spec_->outputs()[port_name_].get();
  }
  return output_spec_;
}

// VirtualTransmitterOp methods

void VirtualTransmitterOp::setup(OperatorSpec& spec) {
  spec.input<gxf::Entity>(port_name_);
}

// VirtualReceiverOp methods

void VirtualReceiverOp::setup(OperatorSpec& spec) {
  spec.output<gxf::Entity>(port_name_);
}

}  // namespace holoscan::ops
