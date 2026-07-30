/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_ANNOTATED_DOUBLE_BUFFER_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_ANNOTATED_DOUBLE_BUFFER_TRANSMITTER_HPP

#include <gxf/core/component.hpp>
#include <gxf/core/entity.hpp>
#include <gxf/core/handle.hpp>

#include <gxf/std/double_buffer_transmitter.hpp>

namespace holoscan {

// Forward declarations
class Operator;

/**
 * @brief AnnotatedDoubleBufferTransmitter class tracks every published message by attaching a
 * MessageLabel that has a timestamp.
 *
 * Application authors are not expected to use this class directly. It will be automatically
 * configured for output ports specified via `Operator::setup` when data flow tracking is enabled.
 *
 * ==Parameters==
 *
 * See DoubleBufferTransmitter for more details.
 */
class AnnotatedDoubleBufferTransmitter : public nvidia::gxf::DoubleBufferTransmitter {
 public:
  AnnotatedDoubleBufferTransmitter() = default;

  /**
   * @brief Override the DoubleBufferTransmitter::publish_abi() function. It adds a
   * MessageLabel to the publish GXF Entity and finally calls the base class' publish_abi()
   * function. It gets the input message labels of every operator and then adds that consolidated
   * message label to the published message.
   *
   * For root operators, it also updates the number of published messages.
   */
  gxf_result_t publish_abi(gxf_uid_t uid);

  holoscan::Operator* op() { return op_; }

  /**
   * @brief Set the associated operator for this AnnotatedDoubleBufferTransmitter. It is set at
   * the @see create_input_port() function.
   *
   * @param op The operator that this transmitter is attached to.
   */
  void op(holoscan::Operator* op) { this->op_ = op; }

 private:
  holoscan::Operator* op_ = nullptr;  ///< The operator that this transmitter is attached to.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_ANNOTATED_DOUBLE_BUFFER_TRANSMITTER_HPP */
