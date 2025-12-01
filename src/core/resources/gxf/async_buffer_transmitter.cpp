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

#include "holoscan/core/resources/gxf/async_buffer_transmitter.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/resources/gxf/holoscan_async_buffer_transmitter.hpp"

namespace holoscan {

AsyncBufferTransmitter::AsyncBufferTransmitter(const std::string& name,
                                               nvidia::gxf::Transmitter* component)
    : Transmitter(name, component) {}

nvidia::gxf::AsyncBufferTransmitter* AsyncBufferTransmitter::get() const {
  return static_cast<nvidia::gxf::AsyncBufferTransmitter*>(gxf_cptr_);
}

void AsyncBufferTransmitter::track() {
  auto transmitter_ptr = static_cast<holoscan::HoloscanAsyncBufferTransmitter*>(gxf_cptr_);
  if (transmitter_ptr) {
    transmitter_ptr->track();
  } else {
    throw std::runtime_error(
        "gxf_cptr_ not found. Cannot enable data flow tracking for asynchronous buffer "
        "transmitter.");
  }
}

}  // namespace holoscan
