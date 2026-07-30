/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/dfft_collector.hpp>

#include <utility>

#include <gxf/std/codelet.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

gxf_result_t DFFTCollector::on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) {
  if (!data_flow_tracker_) {
    HOLOSCAN_LOG_ERROR("data_flow_tracker_ is null in DFFTCollector.");
    return GXF_FAILURE;
  }

  // Get handle to entity
  auto entity = nvidia::gxf::Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }
  (void)timestamp;
  (void)code;

  auto codelet = entity->get<nvidia::gxf::Codelet>();

  gxf_uid_t codelet_id = -1;

  if (codelet.value()) {
    codelet_id = codelet.value()->cid();
  }

  if (codelet_id < 0) {
    HOLOSCAN_LOG_ERROR("codelet_id is less than 0 in DFFTCollector.");
    return GXF_FAILURE;
  }

  // Sometimes, Entity Monitor is called in GXF without tick, start or stop but just to check
  // scheduling condition and abort doing anything. getExecutionCount() is tested to check whether a
  // tick really happened for a leaf operator

  // Lazily initialize execution counters
  if (leaf_last_execution_count_.find(codelet_id) == leaf_last_execution_count_.end()) {
    leaf_last_execution_count_[codelet_id] = 0;
  }
  if (probe_last_execution_count_.find(codelet_id) == probe_last_execution_count_.end()) {
    probe_last_execution_count_[codelet_id] = 0;
  }

  auto leaf_op_opt = data_flow_tracker_->is_leaf_codelet(codelet_id);
  auto root_op_opt = data_flow_tracker_->is_root_codelet(codelet_id);
  if (leaf_op_opt && *leaf_op_opt &&
      ((*leaf_op_opt)->has_input_message_labels() || (root_op_opt && *root_op_opt)) &&
      codelet.value()->getExecutionCount() > leaf_last_execution_count_[codelet_id]) {
    leaf_last_execution_count_[codelet_id] = codelet.value()->getExecutionCount();

    holoscan::Operator* op_ptr = *leaf_op_opt;

    MessageLabel m = op_ptr->get_consolidated_input_label();
    op_ptr->reset_input_message_labels();

    if (m.num_paths()) {
      auto all_path_names = m.get_all_path_names();
      m.update_last_op_publish();
      for (int i = 0; i < m.num_paths(); i++) {
        data_flow_tracker_->update_latency(all_path_names[i], m.get_e2e_latency_ms(i));
      }
      data_flow_tracker_->write_to_logfile(m.to_string());
    }
  } else {
    auto probe_op_opt = data_flow_tracker_->is_probe_codelet(codelet_id);
    if (probe_op_opt && *probe_op_opt && (*probe_op_opt)->has_input_message_labels() &&
        codelet.value()->getExecutionCount() > probe_last_execution_count_[codelet_id]) {
      probe_last_execution_count_[codelet_id] = codelet.value()->getExecutionCount();

      holoscan::Operator* op_ptr = *probe_op_opt;
      MessageLabel m = op_ptr->get_consolidated_input_label();

      // we don't want to reset the input message labels, as that could block further data flow
      // tracking

      if (m.num_paths()) {
        auto all_path_names = m.get_all_path_names();
        m.update_last_op_publish();
        for (int i = 0; i < m.num_paths(); i++) {
          data_flow_tracker_->update_latency(all_path_names[i], m.get_e2e_latency_ms(i));
        }
        // we don't write to logfile because logging is for end-to-end application performance
        // In the future, we can have separate logging for probe operators
      }
    }
  }
  // leaf can also be root, especially for distributed app
  if (root_op_opt) {
    holoscan::Operator* cur_op = *root_op_opt;
    for (auto& it : cur_op->num_published_messages_map()) {
      data_flow_tracker_->update_source_messages_number(it.first, it.second);
    }
  } else if (auto probe_op_opt2 = data_flow_tracker_->is_probe_codelet(codelet_id)) {
    holoscan::Operator* cur_op = *probe_op_opt2;
    for (auto& it : cur_op->num_published_messages_map()) {
      data_flow_tracker_->update_source_messages_number(it.first, it.second);
    }
  }
  return GXF_SUCCESS;
}

void DFFTCollector::data_flow_tracker(holoscan::DataFlowTracker* d) {
  data_flow_tracker_ = d;
}

}  // namespace holoscan
