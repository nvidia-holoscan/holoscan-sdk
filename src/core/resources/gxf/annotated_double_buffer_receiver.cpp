/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/annotated_double_buffer_receiver.hpp"
#include "holoscan/core/flow_tracking_annotation.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/logger/logger.hpp"

#include <gxf/std/double_buffer_receiver.hpp>

namespace holoscan {

gxf_result_t AnnotatedDoubleBufferReceiver::receive_abi(gxf_uid_t* uid) {
  gxf_result_t code = nvidia::gxf::DoubleBufferReceiver::receive_abi(uid);

  deannotate_message(uid, context(), op(), name());

  return code;
}

}  // namespace holoscan
