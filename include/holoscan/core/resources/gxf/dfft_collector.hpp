/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_DFFT_COLLECTOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_DFFT_COLLECTOR_HPP

#include <map>
#include <set>
#include <utility>

#include "gxf/std/clock.hpp"
#include "gxf/std/monitor.hpp"
#include "holoscan/core/dataflow_tracker.hpp"

namespace holoscan {

/**
 * @brief DFFTCollector class (Data Frame Flow Tracking - DFFT) collects the metrics data at the end
 * of the execution of the leaf operators. It updates the DataFlowTracker objects with the final
 * result after every execution of a leaf operator. It also updates the receive timestamp of the
 * root operators to the tick start time of the operators.
 */
class DFFTCollector : public nvidia::gxf::Monitor {
 public:
  /**
   * @brief Called after the execution of an entity. Whenever a Holoscan operator's entity or any
   * other Entity (e.g., entity containing the Broadcast extension component) is finished executing,
   * this function is called.
   *
   * @param eid GXF entity id of the entity that has finished executing.
   */
  gxf_result_t on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) override;

  /**
   * @brief Add an operator as a leaf operator.
   *
   * @param op The operator to be added as a leaf operator.
   */
  void add_leaf_op(holoscan::Operator* op);

  /**
   * @brief Add an operator as a root operator.
   *
   * @param op The operator to be added as a root operator.
   */
  void add_root_op(holoscan::Operator* op);

  /**
   * @brief Set the DataFlowTracker object for this DFFTCollector object.
   *
   * @param d The dataflow tracker object to be set.
   */
  void data_flow_tracker(holoscan::DataFlowTracker* d);

 private:
  /// Pointer to the DataFlowTracker object to update the DataFlowTracker object with the final
  /// results at the end of the execution of a tick of a leaf operator.
  holoscan::DataFlowTracker* data_flow_tracker_ = nullptr;

  /// A map of codelet id and the operator pointers for the leaf operators.
  std::map<int64_t, holoscan::Operator*> leaf_ops_;

  /// A map of codelet id and the operator pointers for the root operators.
  std::map<int64_t, holoscan::Operator*> root_ops_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_DFFT_COLLECTOR_HPP */
