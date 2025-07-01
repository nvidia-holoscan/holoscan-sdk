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

#ifndef HOLOSCAN_CORE_DISTRIBUTED_COMMON_FORWARD_OP_HPP
#define HOLOSCAN_CORE_DISTRIBUTED_COMMON_FORWARD_OP_HPP

#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

/**
 * @brief Forwarding operator.
 *
 * Due to the nature of the GXF UCX extension, a GXF entity cannot have multiple UCX Receivers.
 * This means that an operator cannot have multiple input ports that receive data using UCX.
 *
 * To solve this problem, based on the virtual operator concept of the Holoscan framework, this
 * class is used to add an operator called "forwarding operator" to the fragment.
 * For each VirtualReceiverOp, a ForwardOp is added to the fragment graph.
 *
 * The ForwardOp receives data (from the 'in' input port) and forwards it to the next
 * operator (through the 'out' output port).
 *
 * ==Named Inputs==
 *
 * - **in** : gxf::Entity
 *   - The input data to forward.
 *
 * ==Named Outputs==
 *
 * - **out** : gxf::Entity
 *   - The forwarded data.
 */
class ForwardOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardOp)

  ForwardOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_CORE_DISTRIBUTED_COMMON_FORWARD_OP_HPP */
