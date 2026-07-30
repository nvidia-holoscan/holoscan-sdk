/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_DFFT_COLLECTOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_DFFT_COLLECTOR_HPP

#include <map>

#include <gxf/std/monitor.hpp>
#include <holoscan/core/dataflow_tracker.hpp>

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
   * @param timestamp Timestamp of the execution.
   * @param code Result code of the execution.
   */
  gxf_result_t on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) override;

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

  /// A map of codelet id and the last execution count number from the nvidia::gxf::Codelet
  std::map<gxf_uid_t, int64_t> leaf_last_execution_count_;

  /// A map of codelet id and the last execution count number from the nvidia::gxf::Codelet
  std::map<gxf_uid_t, int64_t> probe_last_execution_count_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_DFFT_COLLECTOR_HPP */
