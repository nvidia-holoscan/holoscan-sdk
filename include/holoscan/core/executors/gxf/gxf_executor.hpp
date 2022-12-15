/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_EXECUTORS_GXF_GXF_EXECUTOR_HPP
#define HOLOSCAN_CORE_EXECUTORS_GXF_GXF_EXECUTOR_HPP

#include <gxf/core/gxf.h>

#include <string>
#include <memory>
#include <set>
#include <vector>
#include <functional>

#include "../../executor.hpp"
namespace holoscan::gxf {

/**
 * @brief Executor for GXF.
 */
class GXFExecutor : public holoscan::Executor {
 public:
  GXFExecutor() = delete;
  explicit GXFExecutor(holoscan::Fragment* app);

  ~GXFExecutor() override;

  /**
   * @brief Initialize the graph and run the graph.
   *
   * This method calls `compose()` to compose the graph, and runs the graph.
   */
  void run(Graph& graph) override;

  /**
   * @brief Create and setup GXF components for input port.
   *
   * For a given input port specification, create a GXF Receiver component for the port and
   * create a GXF SchedulingTerm component that is corresponding to the Condition of the port.
   *
   * If there is no condition specified for the port, a default condition
   * (MessageAvailableCondition) is created.
   * It currently supports ConditionType::kMessageAvailable and ConditionType::kNone condition
   * types.
   *
   * This function is a static function so that it can be called from other classes without
   * dependency on this class.
   *
   * @param fragment The fragment that this operator belongs to.
   * @param gxf_context The GXF context.
   * @param eid The GXF entity ID.
   * @param io_spec The input port specification.
   */
  static void create_input_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                IOSpec* io_spec);

  /**
   * @brief Create and setup GXF components for output port.
   *
   * For a given output port specification, create a GXF Receiver component for the port and
   * create a GXF SchedulingTerm component that is corresponding to the Condition of the port.
   *
   * If there is no condition specified for the port, a default condition
   * (DownstreamMessageAffordableCondition) is created.
   * It currently supports ConditionType::kDownstreamMessageAffordable and ConditionType::kNone
   * condition types.
   *
   * This function is a static function so that it can be called from other classes without
   * dependency on on this class.
   *
   * @param fragment The fragment that this operator belongs to.
   * @param gxf_context The GXF context.
   * @param eid The GXF entity ID.
   * @param io_spec The output port specification.
   */
  static void create_output_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                 IOSpec* io_spec);

 protected:
  bool initialize_operator(Operator* op) override;
  bool add_receivers(const std::shared_ptr<Operator>& op, const std::string& receivers_name,
                     std::set<std::string, std::less<>>& input_labels,
                     std::vector<holoscan::IOSpec*>& iospec_vector) override;

 private:
  void register_extensions();
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_EXECUTORS_GXF_GXF_EXECUTOR_HPP */
