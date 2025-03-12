/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../app_driver.hpp"
#include "../../executor.hpp"
#include "../../graph.hpp"
#include "../../gxf/gxf_extension_manager.hpp"
#include "gxf/app/graph_entity.hpp"

namespace holoscan {

// Forward declarations
class Arg;
class Condition;
class GPUDevice;
class Resource;

}  // namespace holoscan

namespace holoscan::gxf {

/**
 * @brief Executor for GXF.
 */
class GXFExecutor : public holoscan::Executor {
 public:
  GXFExecutor() = delete;
  explicit GXFExecutor(holoscan::Fragment* app, bool create_gxf_context = true);

  ~GXFExecutor() override;

  /**
   * @brief Initialize the graph and run the graph.
   *
   * This method calls `compose()` to compose the graph, and runs the graph.
   *
   * @param graph The reference to the graph.
   */
  void run(OperatorGraph& graph) override;

  /**
   * @brief Initialize the graph and run the graph asynchronously.
   *
   * This method calls `compose()` to compose the graph, and runs the graph asynchronously.
   * The graph is executed in a separate thread and returns a future object.
   *
   * @param graph The reference to the graph.
   * @return The future object.
   */
  std::future<void> run_async(OperatorGraph& graph) override;

  /**
   * @brief Interrupt the execution.
   *
   * This method calls GxfGraphInterrupt() to interrupt the execution.
   */
  void interrupt() override;

  /**
   * @brief Set the context.
   *
   * For GXF, GXFExtensionManager(extension_manager_) is initialized with the context.
   *
   * @param context The context.
   */
  void context(void* context) override;

  // Inherit Executor::context().
  using Executor::context;

  /**
   * @brief Get GXF extension manager.
   *
   * @return The GXF extension manager.
   * @see GXFExtensionManager
   */
  std::shared_ptr<ExtensionManager> extension_manager() override;

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
   * @param eid The GXF entity ID. (Deprecated: now ignored. The eid is obtained from op instead)
   * @param io_spec The input port specification.
   * @param bind_port If true, bind the port to the existing GXF Receiver component. Otherwise,
   * @param op The operator to which this port is being added.
   * create a new GXF Receiver component.
   */
  static void create_input_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                IOSpec* io_spec, bool bind_port = false, Operator* op = nullptr);

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
   * @param eid The GXF entity ID. (Deprecated: now ignored. The eid is obtained from op instead)
   * @param io_spec The output port specification.
   * @param bind_port If true, bind the port to the existing GXF Transmitter component. Otherwise,
   * @param op The operator to which this port is being added.
   * create a new GXF Transmitter component.
   */
  static void create_output_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                 IOSpec* io_spec, bool bind_port = false, Operator* op = nullptr);

  /**
   * @brief Set the GXF entity ID of the operator initialized by this executor.
   *
   * If this is 0, a new entity is created for the operator.
   * Otherwise, the operator, as a codelet, will be added to the existing entity specified by this
   * ID.
   *
   * This is useful when initializing operators within an existing entity, e.g., when
   * initializing an operator from the `holoscan::gxf::OperatorWrapper` class.
   *
   * @param eid The GXF entity ID.
   */
  void op_eid(gxf_uid_t eid) { op_eid_ = eid; }

  /**
   * @brief Get the GXF entity ID of the operator initialized by this executor.
   *
   * Note: This method is used by OperatorRunner to get the GXF entity ID of the operator.
   * @return The GXF entity ID.
   */
  gxf_uid_t op_eid() { return op_eid_; }

  /**
   * @brief Set the GXF component ID of the operator initialized by this executor.
   *
   * If this is 0, a new component is created for the operator.
   *
   * This is useful when initializing operators using an existing component within an existing
   * entity, e.g., when initializing an operator from the `holoscan::gxf::OperatorWrapper` class.
   *
   * @param cid The GXF component ID.
   */
  void op_cid(gxf_uid_t cid) { op_cid_ = cid; }

  /**
   * @brief Get the GXF component ID of the operator initialized by this executor.
   *
   * @return The GXF component ID.
   */
  gxf_uid_t op_cid() { return op_cid_; }

  /**
   * @brief Get the entity prefix string.
   *
   * @return The entity prefix string.
   */
  const std::string& entity_prefix() { return entity_prefix_; }

 protected:
  bool initialize_fragment() override;
  bool initialize_operator(Operator* op) override;
  bool initialize_scheduler(Scheduler* sch) override;
  bool initialize_network_context(NetworkContext* network_context) override;
  bool add_receivers(const std::shared_ptr<Operator>& op, const std::string& receivers_name,
                     std::vector<std::string>& new_input_labels,
                     std::vector<holoscan::IOSpec*>& iospec_vector) override;
  bool add_control_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op) override;

  friend class holoscan::AppDriver;
  friend class holoscan::AppWorker;

  bool initialize_gxf_graph(OperatorGraph& graph);
  void activate_gxf_graph();
  void run_gxf_graph();
  bool connection_items(std::vector<std::shared_ptr<holoscan::ConnectionItem>>& connection_items);

  void add_operator_to_entity_group(gxf_context_t context, gxf_uid_t entity_group_gid,
                                    std::shared_ptr<Operator> op);

  void register_extensions();
  gxf_uid_t op_eid_ = 0;          ///< The GXF entity ID of the operator. Create new entity for
                                  ///< initializing a new operator if this is 0.
  gxf_uid_t op_cid_ = 0;          ///< The GXF component ID of the operator. Create new component
                                  ///< for initializing a new operator if this is 0.
  nvidia::gxf::Extension* gxf_holoscan_extension_ = nullptr;  ///< The GXF holoscan extension.

  /// The flag to indicate whether the GXF graph is initialized.
  bool is_gxf_graph_initialized_ = false;
  /// The flag to indicate whether the GXF graph is activated.
  bool is_gxf_graph_activated_ = false;

  /// The entity prefix for the fragment.
  std::string entity_prefix_;

  /// The connection items for virtual operators.
  std::vector<std::shared_ptr<holoscan::ConnectionItem>> connection_items_;

  /// The list of implicit broadcast entities to be added to the network entity group.
  std::list<std::shared_ptr<nvidia::gxf::GraphEntity>> implicit_broadcast_entities_;

  std::shared_ptr<nvidia::gxf::GraphEntity> util_entity_;
  std::shared_ptr<nvidia::gxf::GraphEntity> gpu_device_entity_;
  std::shared_ptr<nvidia::gxf::GraphEntity> scheduler_entity_;
  std::shared_ptr<nvidia::gxf::GraphEntity> network_context_entity_;
  std::shared_ptr<nvidia::gxf::GraphEntity> connections_entity_;

 private:
  // Map of connections indexed by source port uid and stores a pair of the target operator name
  // and target port name
  using TargetPort = std::pair<holoscan::OperatorGraph::NodeType, std::string>;
  using TargetsInfo = std::tuple<std::string, IOSpec::ConnectorType, std::set<TargetPort>>;
  using TargetConnectionsMapType = std::unordered_map<gxf_uid_t, TargetsInfo>;

  using BroadcastEntityMapType = std::unordered_map<
      holoscan::OperatorGraph::NodeType,
      std::unordered_map<std::string, std::shared_ptr<nvidia::gxf::GraphEntity>>>;

  /** @brief Initialize all GXF Resources in the map and assign them to graph_entity.
   *
   *  Utility function grouping common code across `initialize_network_context` and
   *  `intialize_scheduler`.
   *
   * @param resources Unordered map of GXF resources.
   * @param eid The entity to which the resources will be assigned.
   * @param graph_entity nvidia::gxf::GraphEntity pointer for the resources.
   *
   */
  void initialize_gxf_resources(
      std::unordered_map<std::string, std::shared_ptr<Resource>>& resources, gxf_uid_t eid,
      std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  /** @brief Create a GXF Connection component between a transmitter and receiver.
   *
   * The Connection object created will belong to connections_entity_.
   *
   * @param source_cid The GXF Transmitter component ID.
   * @param target_cid The GXF Receiver component ID.
   * @return The GXF status code.
   */
  gxf_result_t add_connection(gxf_uid_t source_cid, gxf_uid_t target_cid);

  /** @brief Create Broadcast components and add their nvidia::gxf::GraphEntity to
   * broadcast_entites.
   *
   * This is a helper method that gets called by initialize_fragment.
   *
   * Creates broadcast components for any output ports of `op` that connect to more than one
   * input port.
   *
   * Does not add any transmitter to the Broadcast entity. The transmitters will be added later
   * when the incoming edges to the respective operators are processed.
   *
   * Any connected ports of the operator are removed from port_map_val.
   *
   * @param op The operator to create broadcast components for.
   * @param broadcast_entities The mapping of broadcast graph entities.
   * @param connections TODO
   */
  void create_broadcast_components(holoscan::OperatorGraph::NodeType op,
                                   BroadcastEntityMapType& broadcast_entities,
                                   const TargetConnectionsMapType& connections);

  /** @brief Add connection between the prior Broadcast component and the current operator's input
   * port(s).
   *
   * Creates a transmitter on the broadcast component and connects it to the input port of `op`.
   *
   * Any connected ports of the operator are removed from port_map_val.
   *
   * @param broadcast_entities The mapping of broadcast graph entities.
   * @param op The broadcast entity's output will connect to the input port of this operator.
   * @param prev_op The operator connected to the input of the broadcast entity. The capacity
   * and policy of the transmitter added to the broadcast entity will be copied from the transmitter
   * on the broadcasted output port of this operator.
   * @param port_map_val The port mapping between prev_op and op.
   */
  void connect_broadcast_to_previous_op(const BroadcastEntityMapType& broadcast_entities,
                                        holoscan::OperatorGraph::NodeType op,
                                        holoscan::OperatorGraph::NodeType prev_op,
                                        holoscan::OperatorGraph::EdgeDataType port_map_val);

  /// Indicate whether this executor was created by a Holoscan Application.
  bool is_holoscan() const;

  /// Helper function that adds a GXF Condition to the specified graph entity
  bool add_condition_to_graph_entity(std::shared_ptr<Condition> condition,
                                     std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  /// Helper function that adds a GXF Resource to the specified graph entity.
  bool add_resource_to_graph_entity(std::shared_ptr<Resource> resource,
                                    std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  /* @brief Add an IOspec connector resource and any conditions to the graph entity.
   *
   * Helper function for add_component_arg_to_graph_entity.
   *
   * @param io_spec Pointer to the IOSpec object to update.
   * @param graph_entity The graph entity this IOSpec will be associated with.
   * @return true if the IOSpec's components were all successfully added to the graph entity.
   */
  bool add_iospec_to_graph_entity(IOSpec* io_spec,
                                  std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  /* @brief Add any GXF resources and conditions present in the arguments to the provided graph
   * entity.
   *
   * Handles Component, Resource and IOSpec arguments and vectors of each of these.
   *
   * @param io_spec Pointer to the IOSpec object to update.
   * @param graph_entity The graph entity this IOSpec will be associated with.
   * @return true if the IOSpec's components were all successfully added to the graph entity.
   */
  void add_component_args_to_graph_entity(std::vector<Arg>& args,
                                          std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  std::shared_ptr<GPUDevice> add_gpu_device_to_graph_entity(
      const std::string& device_name, std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity,
      std::optional<int32_t> device_id = std::nullopt);
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_EXECUTORS_GXF_GXF_EXECUTOR_HPP */
