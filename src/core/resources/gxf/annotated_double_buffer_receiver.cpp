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

  static gxf_tid_t message_label_tid = GxfTidNull();
  if (message_label_tid == GxfTidNull()) {
    HOLOSCAN_GXF_CALL(GxfComponentTypeId(context(), "holoscan::MessageLabel", &message_label_tid));
  }

  if (gxf::has_component(context(), *uid, message_label_tid, "message_label")) {
    auto gxf_entity = nvidia::gxf::Entity::Shared(context(), *uid);
    auto buffer = gxf_entity.value().get<MessageLabel>("message_label");
    MessageLabel m = *(buffer.value());

    if (!this->op()) {
      HOLOSCAN_LOG_ERROR("AnnotatedDoubleBufferReceiver: {} - Operator* is nullptr", name());
    } else {
      // Create a new operator timestamp with only receive timestamp
      OperatorTimestampLabel cur_op_timestamp(op());
      // Find whether current operator is already in the paths of message label m
      auto cyclic_path_indices = m.has_operator(op()->name());
      if (cyclic_path_indices.empty()) {  // No cyclic paths
        m.add_new_op_timestamp(cur_op_timestamp);
        op()->update_input_message_label(name(), m);
      } else {
        // Update the publish timestamp of current operator where the cycle ends, to be the same as
        // the receive timestamp. For cycles, we don't want to include the last operator's
        // execution time. And the end-to-end latency for cycles is the difference of the start of
        // the first operator and the *start* of the last operator. For others, the end-to-end
        // latency is the start of the first operator and the *end* of the last operator.
        cur_op_timestamp.pub_timestamp = cur_op_timestamp.rec_timestamp;
        m.add_new_op_timestamp(cur_op_timestamp);
        MessageLabel label_wo_cycles;
        // For each cyclic path in m, update the flow tracker
        // For all non-cyclic paths, add to the label_wo_cycles
        int cycle_index = 0;
        for (int i = 0; i < m.num_paths(); i++) {
          if (cycle_index < (int)cyclic_path_indices.size() &&
              i == cyclic_path_indices[cycle_index]) {
            // Update flow tracker here for cyclic paths
            op()->fragment()->data_flow_tracker()->update_latency(m.get_path_name(i),
                                                                  m.get_e2e_latency_ms(i));
            op()->fragment()->data_flow_tracker()->write_to_logfile(
                MessageLabel::to_string(m.get_path(i)));
            cycle_index++;
          } else {
            // For non-cyclic paths, prepare the label_wo_cycles to propagate to the next operator
            label_wo_cycles.add_new_path(m.get_path(i));
          }
        }
        if (!label_wo_cycles.num_paths()) {
          // Since there are no paths in label_wo_cycles, add the current operator in a new path
          label_wo_cycles.add_new_op_timestamp(cur_op_timestamp);
        }
        op()->update_input_message_label(name(), label_wo_cycles);
      }
    }
  } else {
    HOLOSCAN_LOG_DEBUG("AnnotatedDoubleBufferReceiver: {} - No message label found", name());
    op()->delete_input_message_label(name());
  }

  return code;
}

}  // namespace holoscan
