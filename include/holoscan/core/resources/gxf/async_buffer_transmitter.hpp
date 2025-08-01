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
