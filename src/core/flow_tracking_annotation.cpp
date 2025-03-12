/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/flow_tracking_annotation.hpp"

#include <memory>
#include <utility>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

gxf_result_t annotate_message(gxf_uid_t uid, const gxf_context_t& context, Operator* op,
                              const char* transmitter_name) {
  HOLOSCAN_LOG_DEBUG("annotate_message");
  if (!op) {
    HOLOSCAN_LOG_ERROR("Operator is nullptr. Transmitter: {}", transmitter_name);
    return GXF_FAILURE;
  } else if (op->operator_type() == Operator::OperatorType::kVirtual) {
    HOLOSCAN_LOG_DEBUG("Virtual Operators are not timestamped.");
    return GXF_SUCCESS;
  } else {
    auto gxf_entity = nvidia::gxf::Entity::Shared(context, uid);
    if (!gxf_entity) {
      HOLOSCAN_LOG_ERROR("Failed to get GXF Entity with uid: {}", uid);
      return gxf_entity.error();
    }
    // GXF Entity is activated by CreateTensorMap function but it generally is not activated for
    // message Entity. Therefore, we should not need to deactivate the received GXF Entity.
    // Previously, it was done to resolve the issue in CreateTensorMap. It is being commented out in
    // expectation of removal of the Entity activation in CreateTensorMap.
    // gxf_entity->deactivate();
    MessageLabel m;
    m = std::move(op->get_consolidated_input_label());

    std::shared_ptr<holoscan::Operator> op_shared_ptr(op, [](Operator*) {});

    bool is_current_op_root = op->is_root() || op->is_user_defined_root() ||
                              holoscan::Operator::is_all_operator_predecessor_virtual(
                                  std::move(op_shared_ptr), op->fragment()->graph());
    if (!op->fragment()->data_flow_tracker()->limited_tracking() ||
        (op->fragment()->data_flow_tracker()->limited_tracking() &&
         is_current_op_root)) {  // update the last timestamp if limited tracking is not enabled
      m.update_last_op_publish();
    }

    HOLOSCAN_LOG_DEBUG("annotate_message: MessageLabel: {}", m.to_string());

    static gxf_tid_t message_label_tid = GxfTidNull();
    if (message_label_tid == GxfTidNull()) {
      auto gxf_result = GxfComponentTypeId(context, "holoscan::MessageLabel", &message_label_tid);
      if (gxf_result != GXF_SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to get the component type id for MessageLabel: {}",
                           GxfResultStr(gxf_result));
        return gxf_result;
      }
    }

    // Check if a message_label component already exists in the entity
    // If a message_label component already exists in the entity, just update the value of the
    // MessageLabel
    if (gxf::has_component(context, uid, message_label_tid, "message_label")) {
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
  }
  return GXF_SUCCESS;
}

gxf_result_t deannotate_message(gxf_uid_t* uid, const gxf_context_t& context, Operator* op,
                                const char* receiver_name) {
  HOLOSCAN_LOG_DEBUG("deannotate_message");
  if (!op) {
    HOLOSCAN_LOG_ERROR("Operator is nullptr. Receiver: {}", receiver_name);
    return GXF_FAILURE;
  } else if (op->operator_type() == Operator::OperatorType::kVirtual) {
    HOLOSCAN_LOG_DEBUG("Virtual Operators are not timestamped.");
    return GXF_SUCCESS;
  }
  static gxf_tid_t message_label_tid = GxfTidNull();
  if (message_label_tid == GxfTidNull()) {
    HOLOSCAN_GXF_CALL(GxfComponentTypeId(context, "holoscan::MessageLabel", &message_label_tid));
  }

  if (gxf::has_component(context, *uid, message_label_tid, "message_label")) {
    auto gxf_entity = nvidia::gxf::Entity::Shared(context, *uid);
    auto buffer = gxf_entity.value().get<MessageLabel>("message_label");
    MessageLabel m = *(buffer.value());

    // Create a new operator timestamp with only receive timestamp
    OperatorTimestampLabel cur_op_timestamp(op->qualified_name());
    // Find whether current operator is already in the paths of message label m
    auto cyclic_path_indices = m.has_operator(op->qualified_name());
    if (cyclic_path_indices.empty()) {  // No cyclic paths
      std::shared_ptr<holoscan::Operator> op_shared_ptr(op, [](Operator*) {});
      bool is_current_op_leaf =
          op->is_leaf() || holoscan::Operator::is_all_operator_successor_virtual(
                               std::move(op_shared_ptr), op->fragment()->graph());
      if (!op->fragment()->data_flow_tracker()->limited_tracking() ||
          (op->fragment()->data_flow_tracker()->limited_tracking() &&
           is_current_op_leaf)) {  // add a new timestamp if limited tracking is not enabled
        m.add_new_op_timestamp(cur_op_timestamp);
      }
      HOLOSCAN_LOG_DEBUG("deannotate_message: MessageLabel: {}", m.to_string());
      op->update_input_message_label(receiver_name, m);
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
          op->fragment()->data_flow_tracker()->update_latency(m.get_path_name(i),
                                                              m.get_e2e_latency_ms(i));
          op->fragment()->data_flow_tracker()->write_to_logfile(
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
      op->update_input_message_label(receiver_name, label_wo_cycles);
    }
  } else {
    HOLOSCAN_LOG_DEBUG("{} - {} - No message label found", op->qualified_name(), receiver_name);
    op->delete_input_message_label(receiver_name);
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

}  // namespace holoscan
