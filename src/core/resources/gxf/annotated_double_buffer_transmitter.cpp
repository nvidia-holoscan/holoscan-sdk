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

#include "holoscan/core/resources/gxf/annotated_double_buffer_transmitter.hpp"
#include <gxf/core/gxf.h>
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

gxf_result_t AnnotatedDoubleBufferTransmitter::publish_abi(gxf_uid_t uid) {
  if (!this->op()) {
    HOLOSCAN_LOG_ERROR("Operator is nullptr.");
    return GXF_FAILURE;
  } else {
    auto gxf_entity = nvidia::gxf::Entity::Shared(context(), uid);
    gxf_entity->deactivate();  // GXF Entity might be activated by the caller; so deactivate it to
                               // add MessageLabel
    MessageLabel m;
    m = op()->get_consolidated_input_label();
    m.update_last_op_publish();

    // Check if a message_label component already exists in the entity
    static gxf_tid_t message_label_tid = GxfTidNull();
    if (message_label_tid == GxfTidNull()) {
      GxfComponentTypeId(context(), "holoscan::MessageLabel", &message_label_tid);
    }
    // If a message_label component already exists in the entity, just update the value of the
    // MessageLabel
    if (gxf::has_component(context(), uid, message_label_tid, "message_label")) {
      HOLOSCAN_LOG_DEBUG(
          "Found a message label already inside the entity. Replacing the original with a new "
          "one with timestamp.");
      auto maybe_buffer = gxf_entity.value().get<MessageLabel>("message_label");
      if (!maybe_buffer) {
        // Fail early if we cannot add the MessageLabel
        HOLOSCAN_LOG_ERROR(GxfResultStr(maybe_buffer.error()));
        return maybe_buffer.error();
      }
      *maybe_buffer.value() = m;
    } else {  // if no message_label component exists in the entity, add a new one
      auto maybe_buffer = gxf_entity.value().add<MessageLabel>("message_label");
      if (!maybe_buffer) {
        // Fail early if we cannot add the MessageLabel
        HOLOSCAN_LOG_ERROR(GxfResultStr(maybe_buffer.error()));
        return maybe_buffer.error();
      }
      *maybe_buffer.value() = m;
    }

    // We do not activate the GXF Entity because these message entities are not supposed to be
    // activated by default.
  }

  // Call the Base class' publish_abi now
  gxf_result_t code = nvidia::gxf::DoubleBufferTransmitter::publish_abi(uid);

  if (op()->is_root() || op()->is_user_defined_root()) {
    if (!op_transmitter_name_pair_.size())
      op_transmitter_name_pair_ = fmt::format("{}->{}", op()->name(), name());
    op()->update_published_messages(op_transmitter_name_pair_);
  }

  return code;
}

}  // namespace holoscan
