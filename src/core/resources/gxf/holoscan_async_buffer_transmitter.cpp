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

#include <memory>
#include <utility>

#include "holoscan/core/resources/gxf/holoscan_async_buffer_transmitter.hpp"
#include "holoscan/core/flow_tracking_annotation.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"

#include <gxf/std/async_buffer_transmitter.hpp>

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

  if (tracking_) {
    // Check whether the associated operator is a root operator for the first time.
    if (is_op_root_ == -1) {
      std::shared_ptr<holoscan::Operator> op_shared_ptr(op(), [](Operator*) {});
      is_op_root_ = op()->is_root() || op()->is_user_defined_root() ||
                    Operator::is_all_operator_predecessor_virtual(std::move(op_shared_ptr),
                                                                  op()->fragment()->graph());
    }

    // After the first time, only update number of published messages for a root operator only.
    if (is_op_root_) {
      if (!op_transmitter_name_pair_.size())
        op_transmitter_name_pair_ = fmt::format("{}->{}", op()->qualified_name(), name());
      op()->update_published_messages(op_transmitter_name_pair_);
    }
  }

  return code;
}

}  // namespace holoscan
