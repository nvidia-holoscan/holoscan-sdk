/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_ASYNC_BUFFER_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_ASYNC_BUFFER_TRANSMITTER_HPP

#include <string>

#include <gxf/std/async_buffer_transmitter.hpp>

#include "./transmitter.hpp"

namespace holoscan {

/**
 * @brief Async buffer transmitter class.
 *
 * The AsyncBufferTransmitter class is used to emit messages to another operator within a
 * fragment. This class uses a Simpson's four-slot buffer to enable lockless and asynchronous
 * communication.
 *
 */
class AsyncBufferTransmitter : public Transmitter {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsyncBufferTransmitter, Transmitter)
  AsyncBufferTransmitter() = default;
  AsyncBufferTransmitter(const std::string& name, nvidia::gxf::Transmitter* component);

  const char* gxf_typename() const override { return "holoscan::HoloscanAsyncBufferTransmitter"; }

  /**
   * @brief Track the data flow of the transmitter and use holoscan::HoloscanAsyncBufferTransmitter
   * as the GXF Component.
   */
  void track();

  nvidia::gxf::AsyncBufferTransmitter* get() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_ASYNC_BUFFER_TRANSMITTER_HPP */
