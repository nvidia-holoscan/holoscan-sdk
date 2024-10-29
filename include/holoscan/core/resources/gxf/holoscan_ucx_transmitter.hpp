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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_UCX_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_UCX_TRANSMITTER_HPP

#include <string>

#include <gxf/ucx/ucx_transmitter.hpp>

#include <gxf/core/component.hpp>
#include <gxf/core/entity.hpp>
#include <gxf/core/handle.hpp>

namespace holoscan {

// Forward declarations
class Operator;

/**
 * @brief HoloscanUcxTransmitter class optionally adds a MessageLabel timestamp to every published
 * message if data flow tracking is enabled
 *
 */
class HoloscanUcxTransmitter : public nvidia::gxf::UcxTransmitter {
 public:
  HoloscanUcxTransmitter() = default;

  /**
   * @brief This function overrides the UcxTransmitter::publish_abi() function. It first calls
   * annotates the message with a MessageLabel timestamp if data flow tracking is enabled. It then
   * calls the base class' publish_abi() function. Finally, if data flow tracking is enabled, it
   * updates the Operator's number of published messages.
   */
  gxf_result_t publish_abi(gxf_uid_t uid);

  holoscan::Operator* op() { return op_; }

  /**
   * @brief Sets the associated operator for this HoloscanUcxTransmitter. It is set at
   * the @see create_input_port() function.
   *
   * @param op The operator that this transmitter is attached to.
   */
  void op(holoscan::Operator* op) { this->op_ = op; }
  void track() { tracking_ = true; }

 private:
  holoscan::Operator* op_ = nullptr;  ///< The operator that this transmitter is attached to.
  bool tracking_ = false;             ///< Used to decide whether to use data flow tracking or not.
  /// The concatenated name of the operator and this transmitter.
  std::string op_transmitter_name_pair_;
  int is_op_root = -1;  ///< Indicates whether associated op is a root operator.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_UCX_TRANSMITTER_HPP */
