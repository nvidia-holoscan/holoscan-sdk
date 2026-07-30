/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/async_buffer_receiver.hpp>

#include <string>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <holoscan/core/resources/gxf/holoscan_async_buffer_receiver.hpp>

namespace holoscan {

AsyncBufferReceiver::AsyncBufferReceiver(const std::string& name, nvidia::gxf::Receiver* component)
    : Receiver(name, component) {}

nvidia::gxf::AsyncBufferReceiver* AsyncBufferReceiver::get() const {
  return static_cast<nvidia::gxf::AsyncBufferReceiver*>(gxf_cptr_);
}

void AsyncBufferReceiver::track() {
  auto receiver_ptr = static_cast<holoscan::HoloscanAsyncBufferReceiver*>(gxf_cptr_);
  if (receiver_ptr) {
    receiver_ptr->track();
  } else {
    throw std::runtime_error(
        "gxf_cptr_ not found. Cannot enable data flow tracking for asynchronous buffer receiver.");
  }
}

}  // namespace holoscan
