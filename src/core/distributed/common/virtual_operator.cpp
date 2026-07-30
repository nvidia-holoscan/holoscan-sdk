/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/distributed/common/virtual_operator.hpp>
#include <holoscan/core/gxf/entity.hpp>

namespace holoscan::ops {

void VirtualOperator::initialize() {
  // We do not call the base class initialize function.
  // Operator::initialize();
}

IOSpec* VirtualOperator::input_spec() {
  if (spec_ == nullptr) {
    return nullptr;
  }
  if (input_spec_ == nullptr && io_type_ == IOSpec::IOType::kInput) {
    input_spec_ = spec_->inputs()[port_name_].get();
  }
  return input_spec_;
}

IOSpec* VirtualOperator::output_spec() {
  if (spec_ == nullptr) {
    return nullptr;
  }
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
