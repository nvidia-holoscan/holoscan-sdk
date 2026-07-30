/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/flow_tracking_annotation.hpp>
#include <holoscan/core/resources/gxf/holoscan_async_buffer_receiver.hpp>
#include <holoscan/logger/logger.hpp>

#include <gxf/std/async_buffer_receiver.hpp>

namespace holoscan {

gxf_result_t HoloscanAsyncBufferReceiver::receive_abi(gxf_uid_t* uid) {
  gxf_result_t code = nvidia::gxf::AsyncBufferReceiver::receive_abi(uid);

  if (tracking_) {
    if (code == GXF_SUCCESS) {
      if (uid == nullptr) {
        HOLOSCAN_LOG_ERROR("Received message with null UID pointer");
        deannotate_message(nullptr, context(), op(), name());
        return GXF_FAILURE;
      }

      // Receive succeeded - deannotate the message
      // last argument tells message is old or not
      const bool is_old_message = (*uid == last_received_uid_);
      HOLOSCAN_LOG_DEBUG("Receiving message with UID: {}", *uid);
      deannotate_message(uid, context(), op(), name(), is_old_message);
      last_received_uid_ = *uid;
    } else {
      // Receive failed. Clear any stale input_message_label.
      deannotate_message(nullptr, context(), op(), name());
    }
  }

  return code;
}

}  // namespace holoscan
