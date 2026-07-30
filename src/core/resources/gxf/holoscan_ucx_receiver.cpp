/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/flow_tracking_annotation.hpp>
#include <holoscan/core/resources/gxf/holoscan_ucx_receiver.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

gxf_result_t HoloscanUcxReceiver::receive_abi(gxf_uid_t* uid) {
  gxf_result_t code = nvidia::gxf::UcxReceiver::receive_abi(uid);

  if (tracking_) {
    deannotate_message(uid, context(), op(), name());
  }

  return code;
}

}  // namespace holoscan
