/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP

#include <string>

#include <gxf/std/double_buffer_transmitter.hpp>

#include "./transmitter.hpp"

namespace holoscan {

// Forward declarations
class AnnotatedDoubleBufferTransmitter;

/**
 * @brief Double buffer transmitter class.
 *
 * The DoubleBufferTransmitter class is used to emit messages to another operator within a
 * fragment. This class uses a double buffer queue where messages are initially pushed to a "back
 * stage". When `OutputContext::emit` is called from an Operator's `compute` method, the message is
 * published to the back stage of the queue. After the `compute` call finishes, all messages from
 * the back stage are pushed to the main stage.
 *
 * Application authors are not expected to use this class directly. It will be automatically
 * configured for output ports specified via `Operator::setup`.
 *
 * ==Parameters==
 *
 * - **capacity** (uint64_t, optional): The capacity of the double-buffer queue used by the
 * transmitter. Defaults to 1.
 * - **policy** (uint64_t, optional): The policy to use when a message arrives, but there is no
 * space in the transmitter. The possible values are 0: pop, 1: reject, 2: fault (Default: 2).
 */
class DoubleBufferTransmitter : public Transmitter {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(DoubleBufferTransmitter, Transmitter)
  DoubleBufferTransmitter() = default;
  DoubleBufferTransmitter(const std::string& name, nvidia::gxf::DoubleBufferTransmitter* component);

  DoubleBufferTransmitter(const std::string& name, AnnotatedDoubleBufferTransmitter* component);

  const char* gxf_typename() const override;

  void setup(ComponentSpec& spec) override;

  /**
   * @brief Track the data flow of the receiver and use holoscan::AnnotatedDoubleBufferTransmitter
   * as the GXF Component.
   */
  void track();

  nvidia::gxf::DoubleBufferTransmitter* get() const;

  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

 private:
  bool tracking_ = false;  ///< Used to decide whether to use data flow tracking or not.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP */
