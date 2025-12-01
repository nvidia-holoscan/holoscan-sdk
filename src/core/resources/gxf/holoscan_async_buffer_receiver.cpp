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

#include "holoscan/core/resources/gxf/holoscan_async_buffer_receiver.hpp"
#include "holoscan/core/flow_tracking_annotation.hpp"
#include "holoscan/logger/logger.hpp"

#include <gxf/std/async_buffer_receiver.hpp>

namespace holoscan {

gxf_result_t HoloscanAsyncBufferReceiver::receive_abi(gxf_uid_t* uid) {
  gxf_result_t code = nvidia::gxf::AsyncBufferReceiver::receive_abi(uid);

  if (tracking_ && code == GXF_SUCCESS) {
    HOLOSCAN_LOG_DEBUG("Receiving message with UID: {}", *uid);
    if (*uid == kNullUid) {
      HOLOSCAN_LOG_WARN("Invalid message received. Ignoring data flow tracking annotation.");
      return code;
    }
    // last argument tells message is old or not
    deannotate_message(uid, context(), op(), name(), (*uid == last_received_uid_));
    last_received_uid_ = *uid;
  }

  return code;
}

}  // namespace holoscan
