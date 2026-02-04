/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_SUBGRAPH_HPP
#define HOLOSCAN_CORE_SUBGRAPH_HPP

#include <fmt/format.h>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "io_spec.hpp"

namespace holoscan {

// Forward declarations
class Config;
class DataLogger;
class Fragment;
class Operator;
class OperatorSpec;

/**
 * @brief Interface port that maps external subgraph port name to internal operator port(s)
 *
 * For most ports, there is a single mapping. However, input interface ports can have
 * multiple mappings, allowing a single external interface port to broadcast to multiple
 * internal operators.
 */
struct InterfacePort {
  /**
   * @brief Type of interface port
   */
  enum class PortType {
    kData,      ///< Regular data port (for data flow)
    kExecution  ///< Execution control port (for control flow)
  };

  /**
   * @brief A single mapping from external interface port to internal operator port
   */
  struct Mapping {
    std::shared_ptr<Operator> internal_operator;  ///< Internal operator that owns the port
    std::string internal_port_name;               ///< Port name on the internal operator
  };

  std::vector<Mapping> mappings;         ///< List of internal operator/port mappings
  bool is_input;                         ///< Whether this is an input port (vs output)
  PortType port_type = PortType::kData;  ///< Port type (data or execution)

  /// @brief Get number of mappings
  size_t size() const { return mappings.size(); }

  /// @brief Check if empty
  bool empty() const { return mappings.empty(); }

  /// @brief Add a mapping
  void add_mapping(std::shared_ptr<Operator> op, std::string port_name) {
    mappings.push_back({std::move(op), std::move(port_name)});
  }
};

/**
 * @brief A reusable subgraph that directly populates a Fragment's operator graph
 *
 * Subgraph receives Fragment* during construction and directly adds operators and flows to the
 * Fragment's main graph during compose().
 *
 * Usage example:
 *
 * ```cpp
 * class CameraSubgraph : public Subgraph {
 *  public:
 *   CameraSubgraph(Fragment* fragment, const std::string& name)
 *       : Subgraph(fragment, name) {}
 *
 *   void compose() override {
 *     auto source = make_operator<V4L2VideoOp>("source", from_config("v4l2"));
 *     auto converter = make_operator<FormatConverterOp>("converter",
 *                                                        from_config("format_converter"));
 *
 *     add_flow(source, converter);  // Directly added to Fragment's main graph
 *
 *     // Expose interface ports for external connections
 *     // The "tensor" output port of converter will be exposed as "video_out"
 *     add_output_interface_port("video_out", converter, "tensor");
 *   }
 * };
 *
 * // In Fragment::compose():
 *
 * // Note that for subgraph name "camera1" and "camera2", the operator names will become
 * // "camera1_source", "camera2_source", "camera1_converter", "camera2_converter".
 * auto camera1 = make_subgraph<CameraSubgraph>("camera1");
 * auto camera2 = make_subgraph<CameraSubgraph>("camera2");
 * auto visualizer = make_operator<HolovizOp>("visualizer", from_config("holoviz"));
 *
 * // Direct connection to other operators (or subgraphs) via interface ports
 * add_flow(camera1, visualizer, {{"video_out", "receivers"}});
 * add_flow(camera2, visualizer, {{"video_out", "receivers"}});
 * ```
 */
class Subgraph {
 public:
  /**
   * @brief Construct Subgraph with target Fragment
   * @param fragment Target Fragment to populate with operators
   * @param name Unique instance name for operator qualification
   * @param config_file Optional path to a YAML configuration file for this subgraph.
   *        If provided, the configuration is loaded before compose() is called,
   *        making from_config() available during composition.
   */
  Subgraph(Fragment* fragment, const std::string& name, const std::string& config_file = "");

  virtual ~Subgraph() = default;

  /**
   * @brief Define the internal structure of the subgraph
   *
   * This method should create operators and flows, which will be directly added to the Fragment's
   * main graph with qualified names.
   */
  virtual void compose() = 0;

  /**
   * @brief Get the name for this subgraph
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Get the instance name for this subgraph
   * @deprecated Use name() instead. This method will be removed in a future release.
   */
  const std::string& instance_name() const {
    static bool warned = false;
    if (!warned) {
      HOLOSCAN_LOG_WARN(
          "Subgraph::instance_name() is deprecated. Please use Subgraph::name() instead.");
      warned = true;
    }
    return name_;
  }

  /**
   * @brief Get the Fragment that this subgraph belongs to
   *
   * @return Pointer to the fragment this subgraph belongs to
   */
  Fragment* fragment() { return fragment_; }

  /**
   * @brief Get the Fragment that this subgraph belongs to (const version)
   *
   * @return Const pointer to the fragment this subgraph belongs to
   */
  const Fragment* fragment() const { return fragment_; }

  /**
   * @brief Create qualified operator name: subgraph_name + "_" + operator_name
   */
  std::string get_qualified_name(const std::string& object_name,
                                 const std::string& type_name = "operator") const {
    if (name_.empty()) {
      return object_name;
    }

    if (name_ == object_name) {
      auto err_msg = fmt::format(
          "Subgraph: child {} name '{}' is the same as the parent subgraph name '{}'. "
          "This is not allowed.",
          type_name,
          object_name,
          name_);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // Check if the operator name is already qualified with this instance name
    std::string expected_prefix = name_ + "_";
    if (object_name.size() >= expected_prefix.size() &&
        object_name.substr(0, expected_prefix.size()) == expected_prefix) {
      HOLOSCAN_LOG_TRACE(
          "Subgraph: {} name '{}' is already qualified with subgraph name '{}', returning "
          "as-is",
          type_name,
          object_name,
          name_);
      return object_name;  // Already qualified, return as-is
    }

    return fmt::format("{}_{}", name_, object_name);
  }

  // Fragment API compatibility - template method implementations in fragment.hpp
  template <typename OperatorT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<OperatorT> make_operator(StringT name, ArgsT&&... args);

  template <typename OperatorT, typename... ArgsT>
  std::shared_ptr<OperatorT> make_operator(ArgsT&&... args);

  template <typename ConditionT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<ConditionT> make_condition(StringT name, ArgsT&&... args);

  template <typename ConditionT, typename... ArgsT>
  std::shared_ptr<ConditionT> make_condition(ArgsT&&... args);

  template <typename ResourceT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<ResourceT> make_resource(StringT name, ArgsT&&... args);

  template <typename ResourceT, typename... ArgsT>
  std::shared_ptr<ResourceT> make_resource(ArgsT&&... args);

  /**
   * @brief Register an existing service instance with the fragment.
   *
   * Registers an already created service instance with the fragment.
   * This allows the service to be retrieved later using Fragment::service().
   *
   * @tparam ServiceT The type of the service (must inherit from Resource or FragmentService).
   * @param svc The shared pointer to the service instance to register.
   * @param id The identifier for the service registration. If empty, uses the service type or
   *           resource name as identifier.
   * @return true if the service was successfully registered, false otherwise.
   */
  template <typename ServiceT>
  bool register_service(const std::shared_ptr<ServiceT>& svc, std::string_view id = "");

  /**
   * @brief Create a nested subgraph within this subgraph
   *
   * This enables hierarchical Subgraph composition. The nested Subgraph will use
   * the same Fragment* and will have its operators added directly to the Fragment's
   * main graph with hierarchical qualified names (parent_name_child_name_operator).
   *
   * @tparam SubgraphT The nested subgraph class type
   * @param child_name Name for the nested subgraph
   * @param args Additional arguments for the nested subgraph constructor
   * @return Shared pointer to the nested subgraph
   */
  template <typename SubgraphT, typename... ArgsT>
  std::shared_ptr<SubgraphT> make_subgraph(const std::string& child_name, ArgsT&&... args) {
    // Check for duplicate nested subgraph names
    if (nested_subgraph_names_.find(child_name) != nested_subgraph_names_.end()) {
      throw std::runtime_error(
          fmt::format("Subgraph::make_subgraph: Duplicate nested subgraph name '{}' in subgraph "
                      "'{}'. Each nested subgraph must have a unique name within the same parent "
                      "subgraph.",
                      child_name,
                      name_));
    }

    // Create hierarchical name: parent_name + "_" + child_name
    std::string hierarchical_name = get_qualified_name(child_name, "subgraph");

    // Create nested subgraph with same Fragment* and hierarchical name
    auto nested_subgraph =
        std::make_shared<SubgraphT>(fragment_, hierarchical_name, std::forward<ArgsT>(args)...);

    // Compose immediately - operators added directly to Fragment's main graph
    if (!nested_subgraph->is_composed()) {
      nested_subgraph->compose();
      nested_subgraph->set_composed(true);
    }

    // Store for interface port resolution
    nested_subgraphs_.push_back(nested_subgraph);

    // Register the child name
    nested_subgraph_names_.insert(child_name);

    return nested_subgraph;
  }

  /**
   * @brief Add an operator to the Fragment's main graph with qualified name
   *
   * This directly calls fragment_->add_operator() with a qualified name,
   * eliminating the need for intermediate graph storage.
   */
  void add_operator(const std::shared_ptr<Operator>& op);

  /**
   * @brief Add a subgraph to the Fragment
   *
   * This method ensures the subgraph is composed and its operators are added to the fragment.
   * Use this method when a nested subgraph has no interface ports and doesn't need to be
   * connected to other operators or subgraphs via add_flow.
   *
   * @param subgraph The subgraph to be added.
   */
  void add_subgraph(const std::shared_ptr<Subgraph>& subgraph);

  /**
   * @brief Add a flow between two operators directly in the Fragment's main graph
   *
   * This directly calls fragment_->add_flow(), eliminating the need for
   * intermediate flow storage.
   */
  void add_flow(const std::shared_ptr<Operator>& upstream,
                const std::shared_ptr<Operator>& downstream,
                std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Operator to Subgraph within this Subgraph
   *
   * @param upstream_op The upstream operator
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_port, subgraph_interface_port}
   */
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Subgraph to Operator within this Subgraph
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_op The downstream operator
   * @param port_pairs Port connections: {subgraph_interface_port, downstream_port}
   */
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Subgraph to Subgraph within this Subgraph
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_interface_port, downstream_interface_port}
   */
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Operator to Subgraph with connector type
   *
   * @param upstream_op The upstream operator
   * @param downstream_subgraph The downstream subgraph
   * @param connector_type The connector type
   */
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Operator to Subgraph with port pairs and connector type
   *
   * @param upstream_op The upstream operator
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_port, subgraph_interface_port}
   * @param connector_type The connector type
   */
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs,
                const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Operator with connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_op The downstream operator
   * @param connector_type The connector type
   */
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Operator>& downstream_op,
                const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Operator with port pairs and connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_op The downstream operator
   * @param port_pairs Port connections: {subgraph_interface_port, downstream_port}
   * @param connector_type The connector type
   */
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> port_pairs,
                const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Subgraph with connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_subgraph The downstream subgraph
   * @param connector_type The connector type
   */
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Subgraph with port pairs and connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_interface_port, downstream_interface_port}
   * @param connector_type The connector type
   */
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs,
                const IOSpec::ConnectorType connector_type);

  /**
   * @brief Set a callback function to define dynamic flows for an operator at runtime.
   *
   * This method allows operators to modify their connections with other operators during execution.
   * The callback function is called after the operator executes and can add dynamic flows using
   * the operator's `add_dynamic_flow()` methods.
   *
   * @param op The operator to set dynamic flows for
   * @param dynamic_flow_func The callback function that defines the dynamic flows. Takes a shared
   *                          pointer to the operator as input and returns void.
   */
  virtual void set_dynamic_flows(
      const std::shared_ptr<Operator>& op,
      const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func);

  /**
   * @brief Add a data logger to the fragment.
   *
   * This method dispatches to the fragment's add_data_logger method.
   *
   * @param logger The data logger to add.
   */
  void add_data_logger(const std::shared_ptr<DataLogger>& logger);

  /**
   * @brief Add an interface port that can be connected from external Subgraphs/Operators
   *
   * Validates that the internal operator has the specified port. If is_input is not specified,
   * the method automatically detects whether the port is an input or output. If the port name
   * exists in both inputs and outputs, a runtime_error is thrown requiring explicit specification.
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_op The internal operator that owns the actual port
   * @param internal_port The port name on the internal operator (defaults to external_name if not
   *                      specified)
   * @param is_input Whether this is an input port (true) or output port (false). If not specified,
   *                 the port direction is auto-detected from the operator's port definitions.
   */
  void add_interface_port(const std::string& external_name,
                          const std::shared_ptr<Operator>& internal_op,
                          std::optional<std::string> internal_port = std::nullopt,
                          std::optional<bool> is_input = std::nullopt);

  /**
   * @brief Add an input interface port (convenience method)
   *
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_op The internal operator that owns the actual port
   * @param internal_port The port name on the internal operator (defaults to external_name if not
   *                      specified)
   */
  void add_input_interface_port(const std::string& external_name,
                                const std::shared_ptr<Operator>& internal_op,
                                std::optional<std::string> internal_port = std::nullopt);

  /**
   * @brief Add an output interface port (convenience method)
   *
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_op The internal operator that owns the actual port
   * @param internal_port The port name on the internal operator (defaults to external_name if not
   *                      specified)
   */
  void add_output_interface_port(const std::string& external_name,
                                 const std::shared_ptr<Operator>& internal_op,
                                 std::optional<std::string> internal_port = std::nullopt);

  /**
   * @brief Add an interface port that exposes a nested subgraph's interface port
   *
   * This method enables hierarchical port composition by exposing an interface port
   * from a nested subgraph as an external interface port of the current subgraph.
   * The nested subgraph's interface port is resolved to find the underlying operator
   * and port, which are then registered as the current subgraph's interface port.
   *
   * If is_input is not specified, the port direction is auto-detected from the nested
   * subgraph's interface port definition.
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_subgraph The nested subgraph whose interface port to expose
   * @param internal_interface_port The interface port name on the nested subgraph (defaults to
   *                                external_name if not specified)
   * @param is_input Whether this is an input port (true) or output port (false). If not specified,
   *                 the port direction is auto-detected from the nested subgraph's interface port.
   */
  void add_interface_port(const std::string& external_name,
                          const std::shared_ptr<Subgraph>& internal_subgraph,
                          std::optional<std::string> internal_interface_port = std::nullopt,
                          std::optional<bool> is_input = std::nullopt);

  /**
   * @brief Add an input interface port from a nested subgraph (convenience method)
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_subgraph The nested subgraph whose interface port to expose
   * @param internal_interface_port The interface port name on the nested subgraph (defaults to
   *                                external_name if not specified)
   */
  void add_input_interface_port(const std::string& external_name,
                                const std::shared_ptr<Subgraph>& internal_subgraph,
                                std::optional<std::string> internal_interface_port = std::nullopt);

  /**
   * @brief Add an output interface port from a nested subgraph (convenience method)
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_subgraph The nested subgraph whose interface port to expose
   * @param internal_interface_port The interface port name on the nested subgraph (defaults to
   *                                external_name if not specified)
   */
  void add_output_interface_port(const std::string& external_name,
                                 const std::shared_ptr<Subgraph>& internal_subgraph,
                                 std::optional<std::string> internal_interface_port = std::nullopt);

  // ========== Execution Interface Port Methods ==========

  /**
   * @brief Add an input execution interface port for control flow
   *
   * Exposes an execution port from an internal operator for control flow connections.
   * The internal operator must be a Native operator with an input execution spec.
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_op The internal operator that has the execution port
   */
  void add_input_exec_interface_port(const std::string& external_name,
                                     const std::shared_ptr<Operator>& internal_op);

  /**
   * @brief Add an output execution interface port for control flow
   *
   * Exposes an execution port from an internal operator for control flow connections.
   * The internal operator must be a Native operator with an output execution spec.
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_op The internal operator that has the execution port
   */
  void add_output_exec_interface_port(const std::string& external_name,
                                      const std::shared_ptr<Operator>& internal_op);

  /**
   * @brief Add an input execution interface port from a nested subgraph
   *
   * Exposes an execution interface port from a nested subgraph as this subgraph's
   * execution interface port, enabling hierarchical control flow composition.
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_subgraph The nested subgraph whose exec interface port to expose
   * @param internal_interface_port The exec interface port name on the nested subgraph (defaults to
   *                                external_name if not specified)
   */
  void add_input_exec_interface_port(
      const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
      std::optional<std::string> internal_interface_port = std::nullopt);

  /**
   * @brief Add an output execution interface port from a nested subgraph
   *
   * Exposes an execution interface port from a nested subgraph as this subgraph's
   * execution interface port, enabling hierarchical control flow composition.
   *
   * @param external_name The name of the interface port (used in add_flow calls)
   * @param internal_subgraph The nested subgraph whose exec interface port to expose
   * @param internal_interface_port The exec interface port name on the nested subgraph (defaults to
   *                                external_name if not specified)
   */
  void add_output_exec_interface_port(
      const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
      std::optional<std::string> internal_interface_port = std::nullopt);

  /**
   * @brief Get data interface ports
   *
   * Returns a map of interface port names to InterfacePort objects.
   * Each InterfacePort can contain multiple mappings for broadcast input ports.
   */
  const std::unordered_map<std::string, InterfacePort>& interface_ports() const {
    return interface_ports_;
  }

  /**
   * @brief Get execution interface ports
   */
  const std::unordered_map<std::string, InterfacePort>& exec_interface_ports() const {
    return exec_interface_ports_;
  }

  /**
   * @brief Get the first operator/port for a data interface port name
   *
   * This method first checks local interface ports, then recursively checks
   * nested subgraphs for hierarchical port resolution.
   *
   * For broadcast input ports that have multiple mappings, this returns
   * only the first mapping. Access the InterfacePort directly via interface_ports()
   * to get all mappings.
   *
   * @param port_name The interface port name
   * @return Pair of (operator, actual_port_name) or (nullptr, "") if not found
   */
  std::pair<std::shared_ptr<Operator>, std::string> get_interface_operator_port(
      const std::string& port_name) const;

  /**
   * @brief Get the operator/port for an execution interface port name
   *
   * This method first checks local exec interface ports, then recursively checks
   * nested subgraphs for hierarchical port resolution.
   *
   * @param port_name The exec interface port name
   * @return Pair of (operator, actual_port_name) or (nullptr, "") if not found
   */
  std::pair<std::shared_ptr<Operator>, std::string> get_exec_interface_operator_port(
      const std::string& port_name) const;

  /**
   * @brief Check if the subgraph has been composed
   */
  bool is_composed() const { return is_composed_; }

  /**
   * @brief Set the composed state of the subgraph
   */
  void set_composed(bool composed) { is_composed_ = composed; }

  /**
   * @brief Get all operators belonging to this subgraph and its nested subgraphs.
   *
   * This method returns all operators whose names are prefixed with this subgraph's
   * name followed by an underscore. This includes operators from nested subgraphs
   * since their names are also prefixed with the parent subgraph's name.
   *
   * @return Vector of shared pointers to the operators.
   */
  std::vector<std::shared_ptr<Operator>> operators() const;

  // ========== Configuration Methods (Getters) ==========

  /**
   * @brief Get the configuration of the subgraph.
   *
   * @return The reference to the configuration of the subgraph (`Config` object.)
   */
  Config& config();

  /**
   * @brief Get the shared pointer to the configuration of the subgraph.
   *
   * @return The shared pointer to the configuration of the subgraph.
   */
  std::shared_ptr<Config> config_shared();

  /**
   * @brief Get the value of a configuration key as an ArgList.
   *
   * This method retrieves the value from the subgraph's configuration for the given key.
   * You can use '.' (dot) to access nested fields. The returned ArgList can be passed
   * directly to make_operator() or other methods that accept configuration arguments.
   *
   * Example usage:
   * @code
   *   auto op = make_operator<MyOp>("my_op", from_config("my_op"));
   * @endcode
   *
   * @param key The key of the configuration.
   * @return The argument list of the configuration for the key.
   */
  ArgList from_config(const std::string& key);

  /**
   * @brief Determine the set of keys present in the subgraph's config.
   *
   * @return The set of valid keys.
   */
  std::unordered_set<std::string> config_keys();

 protected:
  // ========== Configuration Methods (Setters - Protected) ==========
  // These setters are protected because config must be set before compose() runs.
  // Pass config_file to the Subgraph constructor instead of calling these methods directly.

  /**
   * @brief Set the configuration of the subgraph from a file.
   *
   * The configuration file is a YAML file that contains parameter values that can be
   * accessed via from_config(). This is useful when a subgraph needs its own configuration
   * separate from the main application configuration.
   *
   * @note This method is protected because configuration must be set before compose() runs.
   * Pass the config_file to the Subgraph constructor instead.
   *
   * @note Loading GXF extensions is not supported from the subgraph config file. GXF extensions
   * should be loaded via the application-level configuration only.
   *
   * @param config_file The path to the configuration file.
   * @param prefix The prefix string that is prepended to the key of the configuration.
   * @throws RuntimeError if the config_file is non-empty and the file doesn't exist.
   */
  void config(const std::string& config_file, const std::string& prefix = "");

 private:
  std::unordered_map<std::string, InterfacePort> interface_ports_;  ///< Data interface ports
  std::unordered_map<std::string, InterfacePort>
      exec_interface_ports_;  ///< Execution interface ports
  std::vector<std::shared_ptr<Subgraph>>
      nested_subgraphs_;  ///< Nested Subgraphs for hierarchical composition
  std::unordered_set<std::string>
      nested_subgraph_names_;  ///< Track nested child names to detect duplicates
  bool is_composed_ = false;
  Fragment* fragment_;              ///< Target fragment for direct operator/flow addition
  const std::string name_;          ///< Name for this subgraph
  std::shared_ptr<Config> config_;  ///< Subgraph-specific configuration

  /**
   * @brief Efficiently format a port list for error messages
   *
   * Uses fmt::memory_buffer for O(n) string building instead of O(n²) string concatenation.
   * This provides better performance and avoids repeated memory reallocations.
   *
   * @param ports The port collection (inputs or outputs from OperatorSpec)
   * @return Formatted string with port names like "'port1', 'port2', 'port3'"
   */
  std::string format_port_list(
      const std::unordered_map<std::string, std::shared_ptr<IOSpec>>& ports);

  /**
   * @brief Validate that an operator has a port with the correct type
   *
   * @param op The operator to validate
   * @param port_name The name of the port to check
   * @param expect_input Whether we expect this to be an input port (true) or output port (false)
   * @return true if the port exists and has the correct type, false otherwise
   */
  bool validate_operator_port(const std::shared_ptr<Operator>& op, const std::string& port_name,
                              bool expect_input);

  /**
   * @brief Validate that an operator can be used for execution control flow
   *
   * Checks that the operator is a Native operator. Execution specs are created automatically
   * by the GXF executor during initialization.
   *
   * @param op The operator to validate
   * @return true if the operator can be used for execution control flow, false otherwise
   */
  bool validate_operator_exec_port(const std::shared_ptr<Operator>& op);

  /**
   * @brief Check if an exec interface port name is already in use
   *
   * Checks both data and exec interface port maps for name conflicts.
   *
   * @param external_name The name to check
   * @return true if the name is available (not already used)
   * @throws std::runtime_error if the name is already in use
   */
  bool check_exec_port_name_available(const std::string& external_name);

  /**
   * @brief Register an exec interface port
   *
   * Creates and stores an InterfacePort in exec_interface_ports_ with the given parameters.
   *
   * @param external_name The external interface port name
   * @param internal_op The internal operator
   * @param internal_port_name The internal port name (kInputExecPortName or kOutputExecPortName)
   * @param is_input Whether this is an input port
   */
  void register_exec_interface_port(const std::string& external_name,
                                    const std::shared_ptr<Operator>& internal_op,
                                    const std::string& internal_port_name, bool is_input);

  /**
   * @brief Resolve a nested subgraph's exec interface port to an operator and port
   *
   * Looks up the specified exec interface port in the nested subgraph and validates
   * that it has the expected direction (input vs output).
   *
   * @param external_name The name for this subgraph's interface port (for error messages)
   * @param internal_subgraph The nested subgraph to query
   * @param internal_interface_port The exec interface port name to look up
   * @param expect_input Whether we expect an input port (true) or output port (false)
   * @return Pair of (operator, port_name) or (nullptr, "") if resolution fails
   */
  std::pair<std::shared_ptr<Operator>, std::string> resolve_nested_exec_port(
      const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
      const std::string& internal_interface_port, bool expect_input);
};

// Template method implementations moved to fragment.hpp to resolve circular dependency

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SUBGRAPH_HPP */
