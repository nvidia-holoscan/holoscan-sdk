/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <utility>

#include <holoscan/core/flow_tracking_annotation.hpp>
#include <holoscan/core/resources/gxf/holoscan_async_buffer_transmitter.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

gxf_result_t HoloscanAsyncBufferTransmitter::publish_abi(gxf_uid_t uid) {
  if (tracking_) {
    auto code = annotate_message(uid, context(), op(), name());
    if (code != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to annotate message");
      return code;
    }
  }

  // Call the Base class' publish_abi now
  auto code = nvidia::gxf::AsyncBufferTransmitter::publish_abi(uid);

  return code;
}

void HoloscanAsyncBufferTransmitter::track() {
  tracking_ = true;
}

}  // namespace holoscan
