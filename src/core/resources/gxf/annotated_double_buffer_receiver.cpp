/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/flow_tracking_annotation.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <holoscan/core/message.hpp>
#include <holoscan/core/messagelabel.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resources/gxf/annotated_double_buffer_receiver.hpp>
#include <holoscan/logger/logger.hpp>

#include <gxf/std/double_buffer_receiver.hpp>

namespace holoscan {

gxf_result_t AnnotatedDoubleBufferReceiver::receive_abi(gxf_uid_t* uid) {
  gxf_result_t code = nvidia::gxf::DoubleBufferReceiver::receive_abi(uid);

  if (code == GXF_SUCCESS) {
    deannotate_message(uid, context(), op(), name());
  } else {
    // Receive failed. Clear any stale input_message_label.
    deannotate_message(nullptr, context(), op(), name());
  }

  return code;
}

}  // namespace holoscan
