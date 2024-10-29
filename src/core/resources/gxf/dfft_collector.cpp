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

#include "holoscan/core/resources/gxf/dfft_collector.hpp"

#include <iostream>
#include <utility>

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

gxf_result_t DFFTCollector::on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) {
  if (!data_flow_tracker_) {
    HOLOSCAN_LOG_ERROR("data_flow_tracker_ is null in DFFTCollector.");
    return GXF_FAILURE;
  }

  // Get handle to entity
  auto entity = nvidia::gxf::Entity::Shared(context(), eid);
  if (!entity) { return ToResultCode(entity); }
  (void)timestamp;
  (void)code;

  auto codelet = entity->get<nvidia::gxf::Codelet>();

  int64_t codelet_id = -1;

  if (codelet.value()) { codelet_id = codelet.value()->cid(); }

  if (codelet_id < 0) {
    HOLOSCAN_LOG_ERROR("codelet_id is less than 0 in DFFTCollector.");
    return GXF_FAILURE;
  }

  // Sometimes, Entity Monitor is called in GXF without tick, start or stop but just to check
  // scheduling condition and abort doing anything. getExecutionCount() is tested to check whether a
  // tick really happened for a leaf operator
  if (leaf_ops_.find(codelet_id) != leaf_ops_.end() &&
      codelet.value()->getExecutionCount() > leaf_last_execution_count_[codelet_id]) {
    leaf_last_execution_count_[codelet_id] = codelet.value()->getExecutionCount();
    MessageLabel m = std::move(leaf_ops_[codelet_id]->get_consolidated_input_label());
    leaf_ops_[codelet_id]->reset_input_message_labels();

    if (m.num_paths()) {
      auto all_path_names = m.get_all_path_names();
      m.update_last_op_publish();
      for (int i = 0; i < m.num_paths(); i++) {
        data_flow_tracker_->update_latency(all_path_names[i], m.get_e2e_latency_ms(i));
      }
      data_flow_tracker_->write_to_logfile(m.to_string());
    }
  }
  // leaf can also be root, especially for distributed app
  if (root_ops_.find(codelet_id) != root_ops_.end()) {
    holoscan::Operator* cur_op = root_ops_[codelet_id];
    for (auto& it : cur_op->num_published_messages_map()) {
      data_flow_tracker_->update_source_messages_number(it.first, it.second);
    }
  }
  return GXF_SUCCESS;
}

void DFFTCollector::add_leaf_op(holoscan::Operator* op) {
  leaf_ops_[op->id()] = op;
  leaf_last_execution_count_[op->id()] = 0;
}

void DFFTCollector::add_root_op(holoscan::Operator* op) {
  root_ops_[op->id()] = op;
}

void DFFTCollector::data_flow_tracker(holoscan::DataFlowTracker* d) {
  data_flow_tracker_ = d;
}

}  // namespace holoscan
