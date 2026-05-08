/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/resources/gxf/annotated_double_buffer_transmitter.hpp>

#include <gxf/core/gxf.h>

#include <holoscan/core/flow_tracking_annotation.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

gxf_result_t AnnotatedDoubleBufferTransmitter::publish_abi(gxf_uid_t uid) {
  auto code = annotate_message(uid, context(), op(), name());
  if (code != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to annotate message");
    return code;
  }

  // Call the Base class' publish_abi now
  code = nvidia::gxf::DoubleBufferTransmitter::publish_abi(uid);

  return code;
}

}  // namespace holoscan
