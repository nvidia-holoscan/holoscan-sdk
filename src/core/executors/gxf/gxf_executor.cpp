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

#include "holoscan/core/executors/gxf/gxf_executor.hpp"

#include <deque>
#include <unordered_set>

#include <common/assert.hpp>
#include <common/logger.hpp>

#include <gxf/core/gxf.h>

#include "holoscan/core/operator.hpp"

#include "holoscan/core/config.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::gxf {

GXFExecutor::GXFExecutor(holoscan::Fragment* app) : Executor(app) {
  gxf_result_t code;
  GXF_LOG_INFO("Creating context");
  code = GxfContextCreate(&context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("GxfContextCreate Error: %s", GxfResultStr(code));
    return;
  }
};

GXFExecutor::~GXFExecutor() {
  gxf_result_t code;
  GXF_LOG_INFO("Destroying context");
  code = GxfContextDestroy(context_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("GxfContextDestroy Error: %s", GxfResultStr(code));
    return;
  }
};

void GXFExecutor::run(Graph& graph) {
  auto context = context_;

  // Load extensions
  const char* manifest_filename = fragment_->config().config_file().c_str();

  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &manifest_filename, 1, nullptr};
  HOLOSCAN_LOG_INFO("Loading extensions...");
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_ext_info));

  // Compose the graph
  fragment_->compose();

  // Additional setup for GXF Application
  gxf_uid_t eid;
  const GxfEntityCreateInfo entity_create_info = {"_holoscan_util_entity",
                                                  GXF_ENTITY_CREATE_PROGRAM_BIT};
  gxf_result_t code;
  code = GxfCreateEntity(context, &entity_create_info, &eid);

  gxf_tid_t clock_tid;
  code = GxfComponentTypeId(context, "nvidia::gxf::RealtimeClock", &clock_tid);
  gxf_uid_t clock_cid;
  code = GxfComponentAdd(context, eid, clock_tid, "clock", &clock_cid);

  gxf_tid_t sched_tid;
  code = GxfComponentTypeId(context, "nvidia::gxf::GreedyScheduler", &sched_tid);
  gxf_uid_t sched_cid;
  code = GxfComponentAdd(context, eid, sched_tid, nullptr, &sched_cid);

  code = GxfParameterSetHandle(context, sched_cid, "clock", clock_cid);

  // Add connections
  std::deque<holoscan::Graph::NodeType> worklist;
  for (auto& node : graph.get_root_operators()) { worklist.push_back(std::move(node)); }

  auto operators = graph.get_operators();
  std::unordered_set<holoscan::Graph::NodeType> visited_nodes;
  visited_nodes.reserve(operators.size());

  while (!worklist.empty()) {
    const auto& op = worklist.front();
    worklist.pop_front();

    // Check if we have already visited this node
    if (visited_nodes.find(op) != visited_nodes.end()) { continue; }
    visited_nodes.insert(op);

    auto op_spec = op->spec();
    auto& op_name = op->name();
    HOLOSCAN_LOG_DEBUG("Operator: {}", op_name);
    auto next_operators = graph.get_next_operators(op);

    // Collect connections
    std::unordered_map<gxf_uid_t, std::set<gxf_uid_t>> connections;
    for (auto& next_op : next_operators) {
      auto next_op_spec = next_op->spec();
      auto& next_op_name = next_op->name();
      HOLOSCAN_LOG_DEBUG("  Next operator: {}", next_op_name);
      auto port_map_opt = graph.get_port_map(op, next_op);
      if (!port_map_opt.has_value()) {
        HOLOSCAN_LOG_ERROR("Could not find port map for {} -> {}", op_name, next_op_name);
        continue;
      }
      const auto& port_map = port_map_opt.value();

      for (const auto& [source_port, target_ports] : *port_map) {
        for (const auto& target_port : target_ports) {
          HOLOSCAN_LOG_DEBUG("    Port: {} -> {}", source_port, target_port);
          auto source_gxf_resource =
              std::dynamic_pointer_cast<GXFResource>(op_spec->outputs()[source_port]->resource());
          auto target_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
              next_op_spec->inputs()[target_port]->resource());
          gxf_uid_t source_cid = source_gxf_resource->gxf_cid();
          gxf_uid_t target_cid = target_gxf_resource->gxf_cid();

          if (connections.find(source_cid) == connections.end()) {
            connections[source_cid] = std::set<gxf_uid_t>();
          }
          connections[source_cid].insert(target_cid);
        }
      }

      // Add next operator to worklist
      worklist.push_back(std::move(next_op));
    }

    gxf_tid_t broadcast_tid = GxfTidNull();
    gxf_tid_t rx_term_tid = GxfTidNull();
    gxf_tid_t tx_term_tid = GxfTidNull();
    gxf_tid_t rx_tid = GxfTidNull();
    gxf_tid_t tx_tid = GxfTidNull();

    // Create Connection components
    for (const auto& [source_cid, target_cids] : connections) {
      if (target_cids.empty()) {
        HOLOSCAN_LOG_ERROR("No target component found for source_id: {}", source_cid);
        continue;
      }

      // Insert GXF's Broadcast component if source port is connected to multiple targets
      if (target_cids.size() > 1) {
        gxf_uid_t broadcast_eid;
        const GxfEntityCreateInfo broadcast_entity_create_info = {"",
                                                                  GXF_ENTITY_CREATE_PROGRAM_BIT};
        code = GxfCreateEntity(context, &broadcast_entity_create_info, &broadcast_eid);

        if (rx_tid == GxfTidNull()) {
          code = GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferReceiver", &rx_tid);
        }
        gxf_uid_t rx_cid;
        code = GxfComponentAdd(context, broadcast_eid, rx_tid, "", &rx_cid);

        if (rx_term_tid == GxfTidNull()) {
          code = GxfComponentTypeId(
              context, "nvidia::gxf::MessageAvailableSchedulingTerm", &rx_term_tid);
        }
        gxf_uid_t rx_term_cid;
        code = GxfComponentAdd(context, broadcast_eid, rx_term_tid, "", &rx_term_cid);
        code = GxfParameterSetHandle(context, rx_term_cid, "receiver", rx_cid);
        code = GxfParameterSetUInt64(context, rx_term_cid, "min_size", 1);

        std::vector<gxf_uid_t> tx_cids;
        tx_cids.reserve(target_cids.size());
        for (size_t i = target_cids.size(); i > 0; --i) {
          if (tx_tid == GxfTidNull()) {
            code = GxfComponentTypeId(context, "nvidia::gxf::DoubleBufferTransmitter", &tx_tid);
          }
          gxf_uid_t tx_cid;
          code = GxfComponentAdd(context, broadcast_eid, tx_tid, "", &tx_cid);

          if (tx_term_tid == GxfTidNull()) {
            code = GxfComponentTypeId(
                context, "nvidia::gxf::DownstreamReceptiveSchedulingTerm", &tx_term_tid);
          }
          gxf_uid_t tx_term_cid;
          code = GxfComponentAdd(context, broadcast_eid, tx_term_tid, "", &tx_term_cid);
          code = GxfParameterSetHandle(context, tx_term_cid, "transmitter", tx_cid);
          code = GxfParameterSetUInt64(context, tx_term_cid, "min_size", 1);

          tx_cids.push_back(tx_cid);
        }

        if (broadcast_tid == GxfTidNull()) {
          code = GxfComponentTypeId(context, "nvidia::gxf::Broadcast", &broadcast_tid);
        }
        gxf_uid_t broadcast_cid;
        code = GxfComponentAdd(context, broadcast_eid, broadcast_tid, "", &broadcast_cid);
        code = GxfParameterSetHandle(context, broadcast_cid, "source", rx_cid);

        // Insert GXF's Connection components for Broadcast component
        ::holoscan::gxf::add_connection(context, source_cid, rx_cid);
        size_t tx_cid_index = 0;
        for (auto& target_cid : target_cids) {
          ::holoscan::gxf::add_connection(context, tx_cids[tx_cid_index], target_cid);
          ++tx_cid_index;
        }
      } else {
        ::holoscan::gxf::add_connection(context, source_cid, *target_cids.begin());
      }
    }
  }
  (void)code;

  // Run the graph
  HOLOSCAN_LOG_INFO("Activating Graph...");
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  HOLOSCAN_LOG_INFO("Running Graph...");
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  HOLOSCAN_LOG_INFO("Waiting for completion...");
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  HOLOSCAN_LOG_INFO("Deactivating Graph...");
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
}

}  // namespace holoscan::gxf