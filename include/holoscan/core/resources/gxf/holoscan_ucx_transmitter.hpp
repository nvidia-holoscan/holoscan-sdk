/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_UCX_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_UCX_TRANSMITTER_HPP

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
 * Application authors are not expected to use this class directly. It will be automatically
 * configured for output ports specified via `Operator::setup` when `Application::add_flow` has been
 * used to make a connection across fragments of a distributed application and data flow tracking
 * has been enabled.
 *
 * ==Parameters==
 *
 * See UcxTransmitter for parameter descriptions.
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
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_HOLOSCAN_UCX_TRANSMITTER_HPP */
