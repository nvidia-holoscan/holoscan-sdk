/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <signal.h>

#include <deque>
#include <unordered_set>

#include <common/assert.hpp>
#include <common/logger.hpp>

#include "holoscan/core/operator.hpp"

#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/gxf/gxf_extension_registrar.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_tensor.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/double_buffer_transmitter.hpp"

#include "gxf/std/default_extension.hpp"

namespace holoscan::gxf {

static const std::vector<std::string> kDefaultGXFExtensions{
    "libgxf_std.so",
    "libgxf_cuda.so",
    "libgxf_multimedia.so",
    "libgxf_serialization.so",
};

static const std::vector<std::string> kDefaultHoloscanGXFExtensions{
    "libgxf_bayer_demosaic.so",
    "libgxf_stream_playback.so",  // keep for use of VideoStreamSerializer
    "libgxf_tensor_rt.so",
};

/// Global context for signal() to interrupt with Ctrl+C
gxf_context_t s_signal_context;

GXFExecutor::GXFExecutor(holoscan::Fragment* fragment, bool create_gxf_context)
    : Executor(fragment) {
  if (create_gxf_context) {
    gxf_result_t code;
    GXF_LOG_INFO("Creating context");
    code = GxfContextCreate(&context_);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfContextCreate Error: %s", GxfResultStr(code));
      return;
    }
    own_gxf_context_ = true;
    gxf_extension_manager_ = std::make_shared<GXFExtensionManager>(context_);
    // Register extensions for holoscan (GXFWrapper codelet)
    register_extensions();
  }
}

GXFExecutor::~GXFExecutor() {
  // Deinitialize GXF context only if `own_gxf_context_` is true
  if (own_gxf_context_) {
    gxf_result_t code;
    GXF_LOG_INFO("Destroying context");
    code = GxfContextDestroy(context_);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfContextDestroy Error: %s", GxfResultStr(code));
      return;
    }
  }
}

void GXFExecutor::run(Graph& graph) {
  auto context = context_;

  HOLOSCAN_LOG_INFO("Loading extensions from configs...");
  // Load extensions from config file if exists.
  for (const auto& yaml_node : fragment_->config().yaml_nodes()) {
    gxf_extension_manager_->load_extensions_from_yaml(yaml_node);
  }

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

  while (true) {
    if (worklist.empty()) {
      // If the worklist is empty, we check if we have visited all nodes.
      if (visited_nodes.size() == operators.size()) {
        // If we have visited all nodes, we are done.
        break;
      } else {
        HOLOSCAN_LOG_TRACE("Worklist is empty, but not all nodes have been visited");
        // If we have not visited all nodes, we have a cycle in the graph.
        // Add unvisited nodes to the worklist.
        for (auto& node : operators) {
          if (visited_nodes.find(node) != visited_nodes.end()) { continue; }
          HOLOSCAN_LOG_TRACE("Adding node {} to worklist", node->name());
          worklist.push_back(node);
        }
      }
    }
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
        const char* source_cname = "";
        code = GxfComponentName(context, source_cid, &source_cname);
        gxf_uid_t broadcast_eid;
        auto broadcast_entity_name = fmt::format("_broadcast_{}_{}", op_name, source_cname);
        const GxfEntityCreateInfo broadcast_entity_create_info = {broadcast_entity_name.c_str(),
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

  // Install signal handler
  s_signal_context = context;
  signal(SIGINT, [](int signum) {
    (void)signum;
    HOLOSCAN_LOG_ERROR("Interrupted by user");
    gxf_result_t code = GxfGraphInterrupt(s_signal_context);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfGraphInterrupt Error: %s", GxfResultStr(code));
      GXF_LOG_ERROR("Send interrupt once more to terminate immediately");
      signal(SIGINT, SIG_DFL);
    }
  });

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

void GXFExecutor::context(void* context) {
  context_ = context;
  gxf_extension_manager_ = std::make_shared<GXFExtensionManager>(context_);
}

std::shared_ptr<ExtensionManager> GXFExecutor::extension_manager() {
  return gxf_extension_manager_;
}

void GXFExecutor::create_input_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                    IOSpec* io_spec, bool bind_port) {
  const char* rx_name = io_spec->name().c_str();  // input port name

  // If this executor is used by OperatorWrapper (bind_port == true) to wrap Native Operator,
  // then we need to call `io_spec->resource(...)` to set the existing GXF Receiver for this input.
  if (bind_port) {
    const char* entity_name = "";
    GxfComponentName(gxf_context, eid, &entity_name);

    gxf_tid_t receiver_find_tid{};
    GxfComponentTypeId(gxf_context, "nvidia::gxf::Receiver", &receiver_find_tid);

    gxf_uid_t receiver_cid = 0;
    GxfComponentFind(gxf_context, eid, receiver_find_tid, rx_name, nullptr, &receiver_cid);

    gxf_tid_t receiver_tid{};
    GxfComponentType(gxf_context, receiver_cid, &receiver_tid);

    gxf_tid_t double_buffer_receiver_tid{};
    GxfComponentTypeId(
        gxf_context, "nvidia::gxf::DoubleBufferReceiver", &double_buffer_receiver_tid);

    if (receiver_tid == double_buffer_receiver_tid) {
      nvidia::gxf::DoubleBufferReceiver* double_buffer_receiver_ptr = nullptr;
      GxfComponentPointer(gxf_context,
                          receiver_cid,
                          receiver_tid,
                          reinterpret_cast<void**>(&double_buffer_receiver_ptr));

      if (double_buffer_receiver_ptr) {
        auto receiver =
            std::make_shared<holoscan::DoubleBufferReceiver>(rx_name, double_buffer_receiver_ptr);
        // Set the existing DoubleBufferReceiver for this input
        io_spec->resource(receiver);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to get DoubleBufferReceiver pointer for the handle: '{}' in '{}' entity",
            rx_name,
            entity_name);
      }
    } else {
      HOLOSCAN_LOG_ERROR("Unsupported GXF receiver type for the handle: '{}' in '{}' entity",
                         rx_name,
                         entity_name);
    }
    return;
  }

  gxf_result_t code;
  // Create Receiver component for this input
  auto rx_resource = std::make_shared<DoubleBufferReceiver>();
  rx_resource->name(rx_name);
  rx_resource->fragment(fragment);
  auto rx_spec = std::make_shared<ComponentSpec>(fragment);
  rx_resource->setup(*rx_spec.get());
  rx_resource->spec(std::move(rx_spec));

  rx_resource->gxf_eid(eid);
  rx_resource->initialize();

  gxf_uid_t rx_cid = rx_resource->gxf_cid();
  io_spec->resource(rx_resource);

  // Create SchedulingTerm component for this input
  if (io_spec->conditions().empty()) {
    // Default scheduling term for input:
    //   .condition(ConditionType::kMessageAvailable, Arg("min_size") = 1);
    gxf_tid_t term_tid;
    code =
        GxfComponentTypeId(gxf_context, "nvidia::gxf::MessageAvailableSchedulingTerm", &term_tid);
    gxf_uid_t term_cid;
    code = GxfComponentAdd(gxf_context, eid, term_tid, "__condition_input", &term_cid);
    code = GxfParameterSetHandle(gxf_context, term_cid, "receiver", rx_cid);
    code = GxfParameterSetUInt64(gxf_context, term_cid, "min_size", 1);
  } else {
    int condition_index = 0;
    for (const auto& [condition_type, condition] : io_spec->conditions()) {
      ++condition_index;
      switch (condition_type) {
        case ConditionType::kMessageAvailable: {
          std::shared_ptr<MessageAvailableCondition> message_available_condition =
              std::dynamic_pointer_cast<MessageAvailableCondition>(condition);

          message_available_condition->receiver(rx_resource);
          message_available_condition->name(
              ::holoscan::gxf::create_name("__condition_input_", condition_index).c_str());
          message_available_condition->fragment(fragment);
          auto rx_condition_spec = std::make_shared<ComponentSpec>(fragment);
          message_available_condition->setup(*rx_condition_spec.get());
          message_available_condition->spec(std::move(rx_condition_spec));

          message_available_condition->gxf_eid(eid);
          message_available_condition->initialize();
          break;
        }
        case ConditionType::kNone:
          // No condition
          break;
        default:
          throw std::runtime_error("Unsupported condition type");  // TODO: use std::expected
      }
    }
  }
  (void)code;
}

void GXFExecutor::create_output_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                     IOSpec* io_spec, bool bind_port) {
  const char* tx_name = io_spec->name().c_str();

  // If this executor is used by OperatorWrapper (bind_port == true) to wrap Native Operator,
  // then we need to call `io_spec->resource(...)` to set the existing GXF Transmitter for this
  // output.
  if (bind_port) {
    const char* entity_name = "";
    GxfComponentName(gxf_context, eid, &entity_name);

    gxf_tid_t transmitter_find_tid{};
    GxfComponentTypeId(gxf_context, "nvidia::gxf::Transmitter", &transmitter_find_tid);

    gxf_uid_t transmitter_cid = 0;
    GxfComponentFind(gxf_context, eid, transmitter_find_tid, tx_name, nullptr, &transmitter_cid);

    gxf_tid_t transmitter_tid{};
    GxfComponentType(gxf_context, transmitter_cid, &transmitter_tid);

    gxf_tid_t double_buffer_transmitter_tid{};
    GxfComponentTypeId(
        gxf_context, "nvidia::gxf::DoubleBufferTransmitter", &double_buffer_transmitter_tid);

    if (transmitter_tid == double_buffer_transmitter_tid) {
      nvidia::gxf::DoubleBufferTransmitter* double_buffer_transmitter_ptr = nullptr;
      GxfComponentPointer(gxf_context,
                          transmitter_cid,
                          transmitter_tid,
                          reinterpret_cast<void**>(&double_buffer_transmitter_ptr));

      if (double_buffer_transmitter_ptr) {
        auto transmitter = std::make_shared<holoscan::DoubleBufferTransmitter>(
            tx_name, double_buffer_transmitter_ptr);
        // Set the existing DoubleBufferTransmitter for this output
        io_spec->resource(transmitter);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to get DoubleBufferTransmitter pointer for the handle: '{}' in '{}' entity",
            tx_name,
            entity_name);
      }
    } else {
      HOLOSCAN_LOG_ERROR("Unsupported GXF transmitter type for the handle: '{}' in '{}' entity",
                         tx_name,
                         entity_name);
    }
    return;
  }

  gxf_result_t code;
  // Create Transmitter component for this output
  auto tx_resource = std::make_shared<DoubleBufferTransmitter>();
  tx_resource->name(tx_name);
  tx_resource->fragment(fragment);
  auto tx_spec = std::make_shared<ComponentSpec>(fragment);
  tx_resource->setup(*tx_spec.get());
  tx_resource->spec(std::move(tx_spec));

  tx_resource->gxf_eid(eid);
  tx_resource->initialize();

  gxf_uid_t tx_cid = tx_resource->gxf_cid();
  io_spec->resource(tx_resource);

  // Create SchedulingTerm component for this output
  if (io_spec->conditions().empty()) {
    // Default scheduling term for output:
    //   .condition(ConditionType::kDownstreamMessageAffordable, Arg("min_size") = 1);
    gxf_tid_t term_tid;
    code = GxfComponentTypeId(
        gxf_context, "nvidia::gxf::DownstreamReceptiveSchedulingTerm", &term_tid);
    gxf_uid_t term_cid;
    code = GxfComponentAdd(gxf_context, eid, term_tid, "__condition_output", &term_cid);
    code = GxfParameterSetHandle(gxf_context, term_cid, "transmitter", tx_cid);
    code = GxfParameterSetUInt64(gxf_context, term_cid, "min_size", 1);
  } else {
    int condition_index = 0;
    for (const auto& [condition_type, condition] : io_spec->conditions()) {
      ++condition_index;
      switch (condition_type) {
        case ConditionType::kDownstreamMessageAffordable: {
          std::shared_ptr<DownstreamMessageAffordableCondition>
              downstream_msg_affordable_condition =
                  std::dynamic_pointer_cast<DownstreamMessageAffordableCondition>(condition);

          downstream_msg_affordable_condition->transmitter(tx_resource);
          downstream_msg_affordable_condition->name(
              ::holoscan::gxf::create_name("__condition_output_", condition_index).c_str());
          downstream_msg_affordable_condition->fragment(fragment);
          auto tx_condition_spec = std::make_shared<ComponentSpec>(fragment);
          downstream_msg_affordable_condition->setup(*tx_condition_spec.get());
          downstream_msg_affordable_condition->spec(std::move(tx_condition_spec));

          downstream_msg_affordable_condition->gxf_eid(eid);
          downstream_msg_affordable_condition->initialize();
          break;
        }
        case ConditionType::kNone:
          // No condition
          break;
        default:
          throw std::runtime_error("Unsupported condition type");  // TODO: use std::expected
      }
    }
    (void)code;
  }
}

bool GXFExecutor::initialize_operator(Operator* op) {
  if (!op->spec()) {
    HOLOSCAN_LOG_ERROR("No operator spec for GXFOperator '{}'", op->name());
    return false;
  }

  // If the type name is not set, the operator is assumed to be a Holoscan native operator and use
  // `holoscan::gxf::GXFWrapper` as the GXF Codelet.
  const bool is_native_operator = (op->operator_type() == Operator::OperatorType::kNative);
  ops::GXFOperator* gxf_op = static_cast<ops::GXFOperator*>(op);

  const char* codelet_typename = nullptr;
  if (is_native_operator) {
    codelet_typename = "holoscan::gxf::GXFWrapper";
  } else {
    codelet_typename = gxf_op->gxf_typename();
  }

  auto& spec = *(op->spec());

  gxf_uid_t eid = 0;
  gxf_result_t code;

  // Create Entity for the operator if `op_eid_` is 0
  if (op_eid_ == 0) {
    const GxfEntityCreateInfo entity_create_info = {op->name().c_str(),
                                                    GXF_ENTITY_CREATE_PROGRAM_BIT};
    code = GxfCreateEntity(context_, &entity_create_info, &eid);
  } else {
    eid = op_eid_;
  }

  gxf_uid_t codelet_cid;
  // Create Codelet component if `op_cid_` is 0
  if (op_cid_ == 0) {
    gxf_tid_t codelet_tid;
    code = GxfComponentTypeId(context_, codelet_typename, &codelet_tid);
    code = GxfComponentAdd(context_, eid, codelet_tid, op->name().c_str(), &codelet_cid);

    // Set the operator to the GXFWrapper if it is a native operator
    if (is_native_operator) {
      holoscan::gxf::GXFWrapper* gxf_wrapper = nullptr;
      code = GxfComponentPointer(
          context_, codelet_cid, codelet_tid, reinterpret_cast<void**>(&gxf_wrapper));
      if (gxf_wrapper) {
        gxf_wrapper->set_operator(op);
      } else {
        HOLOSCAN_LOG_ERROR("Unable to get GXFWrapper for Operator '{}'", op->name());
      }
    } else {
      // Set the entity id
      gxf_op->gxf_eid(eid);
      // Set the codelet component id
      gxf_op->gxf_cid(codelet_cid);
    }
  } else {
    codelet_cid = op_cid_;
  }

  // Set GXF Codelet ID as the ID of the operator
  op->id(codelet_cid);

  // Create Components for input
  const auto& inputs = spec.inputs();
  for (const auto& [name, io_spec] : inputs) {
    gxf::GXFExecutor::create_input_port(fragment(), context_, eid, io_spec.get(), op_eid_ != 0);
  }

  // Create Components for output
  const auto& outputs = spec.outputs();
  for (const auto& [name, io_spec] : outputs) {
    gxf::GXFExecutor::create_output_port(fragment(), context_, eid, io_spec.get(), op_eid_ != 0);
  }

  // Create Components for condition
  for (const auto& [name, condition] : op->conditions()) {
    auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
    // Initialize GXF component if it is not already initialized.
    if (gxf_condition->gxf_context() == nullptr) {
      gxf_condition->fragment(fragment());

      gxf_condition->gxf_eid(eid);  // set GXF entity id
      gxf_condition->initialize();
    }
  }

  // Create Components for resource
  for (const auto& [name, resource] : op->resources()) {
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // Initialize GXF component if it is not already initialized.
    if (gxf_resource->gxf_context() == nullptr) {
      gxf_resource->fragment(fragment());

      gxf_resource->gxf_eid(eid);  // set GXF entity id
      gxf_resource->initialize();
    }
  }

  // Set arguments
  auto& params = spec.params();
  for (auto& arg : op->args()) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' is not defined in spec", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting argument '{}'", op->name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters if it is an operator that wraps an existing GXF Codelet.
  if (!is_native_operator) {
    // Set Handler parameters
    for (auto& [key, param_wrap] : params) {
      code = ::holoscan::gxf::GXFParameterAdaptor::set_param(
          context_, codelet_cid, key.c_str(), param_wrap);

      if (code != GXF_SUCCESS) {
        HOLOSCAN_LOG_WARN("GXFOperator '{}':: error {} setting GXF parameter '{}'",
                          op->name(),
                          GxfResultStr(code),
                          key);
        // TODO: handle error
      }

      HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting GXF parameter '{}'", op->name(), key);
    }
  } else {
    // Set only default parameter values
    for (auto& [key, param_wrap] : params) {
      // If no value is specified, the default value will be used by setting an empty argument.
      Arg empty_arg("");
      ArgumentSetter::set_param(param_wrap, empty_arg);
    }
  }
  (void)code;
  return true;
}

bool GXFExecutor::add_receivers(const std::shared_ptr<Operator>& op,
                                const std::string& receivers_name,
                                std::set<std::string, std::less<>>& input_labels,
                                std::vector<holoscan::IOSpec*>& iospec_vector) {
  const auto downstream_op_spec = op->spec();

  // 1. Create input port for the receivers parameter

  // Create a new input port label
  const std::string& new_input_label = fmt::format("{}:{}", receivers_name, iospec_vector.size());
  HOLOSCAN_LOG_TRACE("Creating new input port with label '{}'", new_input_label);
  auto& input_port = downstream_op_spec->input<holoscan::gxf::Entity>(new_input_label);
  // TODO: Currently, there is no convenient API to set the condition of the receivers (input ports)
  //       from the setup() method of the operator. We need to add a new API to set the condition
  //       of the receivers (input ports) from the setup() method of the operator.

  // Add the new input port to the vector.
  iospec_vector.push_back(&input_port);

  // 2. Initialize the new input port and update the port_map

  // In GXFExecutor, we use the ID of the operator as the ID of the GXF codelet.
  gxf_uid_t codelet_cid = op->id();
  if (codelet_cid < 0) {
    HOLOSCAN_LOG_ERROR("Invalid GXF Codelet ID for operator '{}': {}", op->name(), codelet_cid);
    return false;
  }

  gxf_context_t gxf_context = context();
  gxf_uid_t eid = get_component_eid(gxf_context, codelet_cid);

  create_input_port(fragment_, gxf_context, eid, &input_port);

  auto operator_type = op->operator_type();
  if (operator_type == Operator::OperatorType::kGXF) {
    // Reset the parameter of the receiver/transmitter vector in the downstream
    // operator (codelet).
    // NOTE: whenever we add a new input port to the vector parameter, we need to reset
    //       the parameter of the receiver/transmitter vector in the downstream operator
    //       (codelet), which maybe inefficient if we have a lot of input ports
    //       connected.
    YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
    for (const auto& io_spec : iospec_vector) {
      const auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(io_spec->resource());
      yaml_node.push_back(gxf_resource->gxf_cname());
    }
    GxfParameterSetFromYamlNode(gxf_context, codelet_cid, receivers_name.c_str(), &yaml_node, "");
  }

  // Update port_map to use the new input port label
  input_labels.erase(receivers_name);
  input_labels.insert(new_input_label);

  return true;
}

void GXFExecutor::register_extensions() {
  // Register the default GXF extensions
  for (auto& gxf_extension_file_name : kDefaultGXFExtensions) {
    gxf_extension_manager_->load_extension(gxf_extension_file_name);
  }

  // Register the default Holoscan GXF extensions
  for (auto& gxf_extension_file_name : kDefaultHoloscanGXFExtensions) {
    gxf_extension_manager_->load_extension(gxf_extension_file_name);
  }

  // Register the GXF extension that provides the native operators
  gxf_tid_t gxf_wrapper_tid{0xd4e7c16bcae741f8, 0xa5eb93731de9ccf6};

  if (!gxf_extension_manager_->is_extension_loaded(gxf_wrapper_tid)) {
    GXFExtensionRegistrar extension_factory(
        context_,
        "HoloscanSdkInternalExtension",
        "A runtime hidden extension used by Holoscan SDK to provide the native operators",
        gxf_wrapper_tid);

    extension_factory.add_component<holoscan::gxf::GXFWrapper, nvidia::gxf::Codelet>(
        "GXF wrapper to support Holoscan SDK native operators");
    extension_factory.add_type<holoscan::Message>("Holoscan message type",
                                                  {0x61510ca06aa9493b, 0x8a777d0bf87476b7});

    extension_factory.add_component<holoscan::gxf::GXFTensor, nvidia::gxf::Tensor>(
        "Holoscan's GXF Tensor type", {0xa02945eaf20e418c, 0x8e6992b68672ce40});
    extension_factory.add_type<holoscan::Tensor>("Holoscan's Tensor type",
                                                 {0xa5eb0ed57d7f4aa2, 0xb5865ccca0ef955c});

    if (!extension_factory.register_extension()) {
      HOLOSCAN_LOG_ERROR("Failed to register Holoscan SDK internal extension");
    }
  }
}

}  // namespace holoscan::gxf
