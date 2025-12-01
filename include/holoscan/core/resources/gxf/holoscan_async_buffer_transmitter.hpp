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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_ASYNC_BUFFER_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_ASYNC_BUFFER_TRANSMITTER_HPP

#include <string>

#include <gxf/std/async_buffer_transmitter.hpp>

#include <gxf/core/component.hpp>
#include <gxf/core/entity.hpp>
#include <gxf/core/handle.hpp>

namespace holoscan {

// Forward declarations
class Operator;

/**
 * @brief HoloscanAsyncBufferTransmitter class tracks every transmitted message with a MessageLabel
 * timestamp if data flow tracking is enabled.
 *
 * Application authors are not expected to use this class directly.
 *
 */
class HoloscanAsyncBufferTransmitter : public nvidia::gxf::AsyncBufferTransmitter {
 public:
  HoloscanAsyncBufferTransmitter() = default;

  /**
   * @brief This function overrides the AsyncBufferTransmitter::publish_abi() function. It first
   * adds a new OperatorTimestampLabel to the MessageLabel of the message being published and then
   * calls the base class' publish_abi() function.
   */
  gxf_result_t publish_abi(gxf_uid_t uid) override;

  holoscan::Operator* op() const { return op_; }

  /**
   * @brief Sets the associated operator for this HoloscanAsyncBufferTransmitter. It is set at
   * the @see create_output_port() function.
   *
   * @param op The operator that this transmitter is attached to.
   */
  void op(holoscan::Operator* op) { this->op_ = op; }
  void track();

 private:
  holoscan::Operator* op_ = nullptr;  ///< The operator that this transmitter is attached to.
  bool tracking_ = false;             ///< Used to decide whether to use data flow tracking or not.
  /// The concatenated name of the operator and this transmitter.
  std::string op_transmitter_name_pair_ = "";
  int is_op_root_ = -1;  ///< Indicates whether associated operator is a root operator.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_ASYNC_BUFFER_TRANSMITTER_HPP */
