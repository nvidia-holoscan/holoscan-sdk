/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/flow_tracking_annotation.hpp>
#include <holoscan/core/resources/gxf/holoscan_ucx_transmitter.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
gxf_result_t HoloscanUcxTransmitter::publish_abi(gxf_uid_t uid) {
  if (tracking_) {
    auto code = annotate_message(uid, context(), op(), name());
    if (code != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to annotate message");
      return code;
    }
  }

  // Call the Base class' publish_abi now
  auto code = nvidia::gxf::UcxTransmitter::publish_abi(uid);

  return code;
}
}  // namespace holoscan
