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

#ifndef HOLOSCAN_CORE_OPERATOR_HPP
#define HOLOSCAN_CORE_OPERATOR_HPP

#include <yaml-cpp/yaml.h>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./arg.hpp"
#include "./gxf/codec_registry.hpp"
#include "./common.hpp"
#include "./component.hpp"
#include "./condition.hpp"
#include "./forward_def.hpp"
#include "./graph.hpp"
#include "./io_spec.hpp"
#include "./messagelabel.hpp"
#include "./metadata.hpp"
#include "./operator_spec.hpp"
#include "./operator_status.hpp"
#include "./resource.hpp"
#include "./gxf/gxf_cuda.hpp"

#include "gxf/app/graph_entity.hpp"
#include "gxf/core/gxf.h"

#define HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                            \
  template <typename ArgT,                                                              \
            typename... ArgsT,                                                          \
            typename = std::enable_if_t<                                                \
                !std::is_base_of_v<holoscan::Operator, std::decay_t<ArgT>> &&           \
                (std::is_same_v<holoscan::Arg, std::decay_t<ArgT>> ||                   \
                 std::is_same_v<holoscan::ArgList, std::decay_t<ArgT>> ||               \
                 std::is_base_of_v<holoscan::Condition,                                 \
                                   typename holoscan::type_info<ArgT>::derived_type> || \
                 std::is_base_of_v<holoscan::Resource,                                  \
                                   typename holoscan::type_info<ArgT>::derived_type>)>>

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the operator class.
 *
 * Use this macro if the base class is a `holoscan::Operator`.
 *
 * Example:
 *
 * ```cpp
 * class GXFOperator : public holoscan::Operator {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS(GXFOperator)
 *
 *   GXFOperator() = default;
 *
 *   void initialize() override;
 *
 *   virtual const char* gxf_typename() const = 0;
 * };
 * ```
 *
 * @param class_name The name of the class.
 */
#define HOLOSCAN_OPERATOR_FORWARD_ARGS(class_name) \
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()             \
  explicit class_name(ArgT&& arg, ArgsT&&... args) \
      : Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the operator class.
 *
 * Use this macro if the class is derived from `holoscan::Operator` or the base class is derived
 * from `holoscan::Operator`.
 *
 * Example:
 *
 * ```cpp
 * class SourceOp : public holoscan::ops::GXFOperator {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SourceOp, holoscan::ops::GXFOperator)
 *
 *   SourceOp() = default;
 *
 *   const char* gxf_typename() const override { return "nvidia::holoscan::Source"; }
 *
 *   void setup(OperatorSpec& spec) override;
 *
 *   void initialize() override;
 * };
 * ```
 *
 * @param class_name The name of the class.
 * @param super_class_name The name of the super class.
 */
#define HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                     \
  explicit class_name(ArgT&& arg, ArgsT&&... args)                         \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

// Forward declarations
class ExecutionContext;
class InputContext;
class OutputContext;

namespace gxf {
class GXFExecutor;
}  // namespace gxf

/**
 * @brief Base class for all operators.
 *
 * An operator is the most basic unit of work in Holoscan SDK. An Operator receives
 * streaming data at an input port, processes it, and publishes it to one of its output ports.
 *
 * This class is the base class for all operators. It provides the basic functionality for all
 * operators.
 *
 * @note This class is not intended to be used directly. Inherit from this class to create a new
 * operator.
 */
class Operator : public ComponentBase {
 public:
  /**
   * @brief Operator type used by the executor.
   */
  enum class OperatorType {
    kNative,   ///< Native operator.
    kGXF,      ///< GXF operator.
    kVirtual,  ///< Virtual operator.
               ///< (for internal use, not intended for use by application authors)
  };

  /// Default input execution port name.
  static constexpr const char* kInputExecPortName = "__input_exec__";
  /// Default output execution port name.
  static constexpr const char* kOutputExecPortName = "__output_exec__";

  /**
   * @brief Construct a new Operator object.
   *
   * @param arg The first argument to be passed to the operator.
   * @param args The remaining arguments to be passed to the operator.
   */
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit Operator(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  Operator() = default;

  ~Operator() override = default;

  /**
   * @brief Get the operator type.
   *
   * @return The operator type.
   */
  OperatorType operator_type() const { return operator_type_; }

  using ComponentBase::id;
  /**
   * @brief Set the Operator ID.
   *
   * @param id The ID of the operator.
   * @return The reference to this operator.
   */
  Operator& id(int64_t id) {
    id_ = id;
    return *this;
  }

  using ComponentBase::name;
  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this operator.
   */
  Operator& name(const std::string& name) {
    // Operator::parse_port_name requires that "." is not allowed in the Operator name
    if (name.find(".") != std::string::npos) {
      throw std::invalid_argument(fmt::format(
          "The . character is reserved and cannot be used in the operator name ('{}').", name));
    }
    name_ = name;
    return *this;
  }

  using ComponentBase::fragment;
  /**
   * @brief Set the fragment of the operator.
   *
   * @param fragment The pointer to the fragment of the operator.
   * @return The reference to this operator.
   */
  Operator& fragment(Fragment* fragment) {
    fragment_ = fragment;
    return *this;
  }

  /**
   * @brief Set the operator spec.
   *
   * @param spec The operator spec.
   * @return The reference to this operator.
   */
  Operator& spec(const std::shared_ptr<OperatorSpec>& spec) {
    spec_ = spec;
    return *this;
  }
  /**
   * @brief Get the operator spec.
   *
   * @return The operator spec.
   */
  OperatorSpec* spec() {
    if (!spec_) {
      HOLOSCAN_LOG_WARN("OperatorSpec of Operator '{}' is not initialized, returning nullptr",
                        name_);
      return nullptr;
    }
    return spec_.get();
  }

  /**
   * @brief Get the shared pointer to the operator spec.
   *
   * @return The shared pointer to the operator spec.
   */
  std::shared_ptr<OperatorSpec> spec_shared() { return spec_; }

  template <typename ConditionT>
  /**
   * @brief Get a shared pointer to the Condition object.
   *
   * @param name The name of the condition.
   * @return The reference to the Condition object. If the condition does not exist, return the
   * nullptr.
   */
  std::shared_ptr<ConditionT> condition(const std::string& name) {
    if (auto condition = conditions_.find(name); condition != conditions_.end()) {
      return std::dynamic_pointer_cast<ConditionT>(condition->second);
    }
    return nullptr;
  }

  /**
   * @brief Get the conditions of the operator.
   *
   * @return The conditions of the operator.
   */
  std::unordered_map<std::string, std::shared_ptr<Condition>>& conditions() { return conditions_; }

  template <typename ResourceT>
  /**
   * @brief Get a shared pointer to the Resource object.
   *
   * @param name The name of the resource.
   * @return The reference to the Resource object. If the resource does not exist, returns the
   * nullptr.
   */
  std::shared_ptr<ResourceT> resource(const std::string& name) {
    if (auto resource = resources_.find(name); resource != resources_.end()) {
      return std::dynamic_pointer_cast<ResourceT>(resource->second);
    }
    return nullptr;
  }

  /**
   * @brief Get the resources of the operator.
   *
   * @return The resources of the operator.
   */
  std::unordered_map<std::string, std::shared_ptr<Resource>>& resources() { return resources_; }

  using ComponentBase::add_arg;

  /**
   * @brief Add a condition to the operator.
   *
   * @param arg The condition to add.
   */
  void add_arg(const std::shared_ptr<Condition>& arg) {
    if (conditions_.find(arg->name()) != conditions_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Condition '{}' already exists in the operator. Please specify a unique "
          "name when creating a Condition instance.",
          arg->name());
    } else {
      conditions_[arg->name()] = arg;
    }
  }

  /**
   * @brief Add a condition to the operator.
   *
   * @param arg The condition to add.
   */
  void add_arg(std::shared_ptr<Condition>&& arg) {
    if (conditions_.find(arg->name()) != conditions_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Condition '{}' already exists in the operator. Please specify a unique "
          "name when creating a Condition instance.",
          arg->name());
    } else {
      conditions_[arg->name()] = std::move(arg);
    }
  }

  /**
   * @brief Add a resource to the operator.
   *
   * @param arg The resource to add.
   */
  void add_arg(const std::shared_ptr<Resource>& arg) {
    if (resources_.find(arg->name()) != resources_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Resource '{}' already exists in the operator. Please specify a unique "
          "name when creating a Resource instance.",
          arg->name());
    } else {
      resources_[arg->name()] = arg;
    }
  }

  /**
   * @brief Add a resource to the operator.
   *
   * @param arg The resource to add.
   */
  void add_arg(std::shared_ptr<Resource>&& arg) {
    if (resources_.find(arg->name()) != resources_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Resource '{}' already exists in the operator. Please specify a unique "
          "name when creating a Resource instance.",
          arg->name());
    } else {
      resources_[arg->name()] = std::move(arg);
    }
  }

  /**
   * @brief Define the operator specification.
   *
   * @param spec The reference to the operator specification.
   */
  virtual void setup([[maybe_unused]] OperatorSpec& spec) {}

  /**
   * @brief Returns whether the operator is a root operator based on its fragment's graph
   *
   * @return True, if the operator is a root operator; false, otherwise
   */
  bool is_root();

  /**
   * @brief Returns whether the operator is a user-defined root operator i.e., the first operator
   * added to the graph.
   *
   * @return True, if the operator is a user-defined root operator; false, otherwise
   */

  bool is_user_defined_root();

  /**
   * @brief Returns whether the operator is a leaf operator based on its fragment's graph
   *
   * @return True, if the operator is a leaf operator; false, otherwise
   */
  bool is_leaf();

  /**
   * @brief Returns whether all the successors of an operator are virtual operators
   *
   * @param op The shared_ptr to the operator for which the check is to be performed
   * @param graph The graph of operators. fragment()->graph() can usually be used to get this graph.
   * @return true if the operator has all virtual operator successors, false otherwise
   */
  static bool is_all_operator_successor_virtual(OperatorNodeType op, OperatorGraph& graph);

  /**
   * @brief Returns whether all the predecessors of an operator are virtual operators
   *
   * @param op The shared_ptr to the operator for which the check is to be performed
   * @param graph The graph of operators. fragment()->graph() can usually be used to get this graph.
   * @return true if the operator has all virtual operator predecessors, false otherwise
   */
  static bool is_all_operator_predecessor_virtual(OperatorNodeType op, OperatorGraph& graph);

  /**
   * @brief Returns the fully qualified name of the operator including the name of the fragment.
   *
   * @return std::string fully qualified name of the operator in the format:
   * "<fragment_name>.<operator_name>"
   */
  std::string qualified_name();

  /**
   * @brief Initialize the operator.
   *
   * This function is called when the fragment is initialized by
   * Executor::initialize_fragment().
   */
  void initialize() override;

  /**
   * @brief Implement the startup logic of the operator.
   *
   * This method is called multiple times over the lifecycle of the operator according to the
   * order defined in the lifecycle, and used for heavy initialization tasks such as allocating
   * memory resources.
   */
  virtual void start() {
    // Empty default implementation
  }

  /**
   * @brief Implement the shutdown logic of the operator.
   *
   * This method is called multiple times over the lifecycle of the operator according to the
   * order defined in the lifecycle, and used for heavy deinitialization tasks such as deallocation
   * of all resources previously assigned in start.
   */
  virtual void stop() {
    // Empty default implementation
  }

  /**
   * @brief Implement the compute method.
   *
   * This method is called by the runtime multiple times. The runtime calls this method until
   * the operator is stopped.
   *
   * @param op_input The input context of the operator.
   * @param op_output The output context of the operator.
   * @param context The execution context of the operator.
   */
  virtual void compute([[maybe_unused]] InputContext& op_input,
                       [[maybe_unused]] OutputContext& op_output,
                       [[maybe_unused]] ExecutionContext& context) {}

  /// Return operator name and port name from a string in the format of "<op_name>[.<port_name>]".
  static std::pair<std::string, std::string> parse_port_name(const std::string& op_port_name);

  /**
   * @brief Register the codec for serialization/deserialization of a custom type.
   * @deprecated Use holoscan::gxf::GXFExecutor::register_codec instead.
   */
  template <typename typeT>
  [[deprecated(
      "Use holoscan::gxf::GXFExecutor::register_codec() instead of "
      "Operator::register_codec()")]] static void
  register_codec(const std::string& codec_name, bool overwrite = true) {
    HOLOSCAN_LOG_WARN(
        "Operator::register_codec is deprecated. Please use the static method "
        "holoscan::gxf::GXFExecutor::register_codec instead.");
    gxf::CodecRegistry::get_instance().add_codec<typeT>(codec_name, overwrite);
  }

  /**
   * @brief Get a YAML representation of the operator.
   *
   * @return YAML node including type, specs, conditions and resources of the operator in addition
   * to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

  /**
   * @brief Get the GXF GraphEntity object corresponding to this operator
   *
   * @return graph entity corresponding to the operator
   */
  std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity() { return graph_entity_; }

  /**
   * @brief Get a shared pointer to the dynamic metadata of this operator.
   *
   *
   * Note: currently this metadata dictionary is only active if explicitly enabled for the
   * application by setting `Fragment::is_metadata_enabled(true)`. When metadata is disabled
   * the dictionary will not be populated by receive calls and will not be emitted on emit calls.
   *
   * This metadata dictionary is always empty at the start of each compute call. It is populated
   * by metadata received on input ports during `InputContext::receive()` calls and can be
   * modified as desired by the operator during the compute call. Any metadata corresponding to this
   * object will be sent on the output ports by any `OutputContext::emit()` calls.
   *
   * @returns The metadata dictionary for this operator.
   */
  std::shared_ptr<MetadataDictionary> metadata() { return dynamic_metadata_; }

  /**
   * @brief Determine if metadata is enabled for this operator.
   *
   * @returns Boolean indicating if metadata is enabled (returns `fragment()->is_metadata_enabled()`
   * if `enable_metadata` was not explicitly called for the operator.
   */
  bool is_metadata_enabled() const;

  /**
   * @brief Enable or disable metadata for this operator.
   *
   * If this method has not been used to explicitly enable or disable metadata, the value for
   * `is_metadata_enabled()` will be determined by `Fragment::is_metadata_enabled()` when the
   * operator is initialized.
   *
   * @param enable Boolean indicating if metadata should be enabled.
   */
  void enable_metadata(bool enable) { is_metadata_enabled_ = enable; }

  /**
   * @brief Get the metadata update policy used by this operator.
   *
   * @returns The metadata update policy used by this operator.
   */
  MetadataPolicy metadata_policy() const { return dynamic_metadata_->policy(); }

  /**
   * @brief Set the metadata update policy used by this operator.
   *
   * The metadata policy determines how metadata is merged across multiple receive calls:
   *    - `MetadataPolicy::kUpdate`: Update the existing value when a key already exists.
   *    - `MetadataPolicy::kInplaceUpdate`: Update the existing MetadataObject's value in-place
   *    when a key already exists.
   *    - `MetadataPolicy::kReject`: Do not modify the existing value if a key already exists.
   *    - `MetadataPolicy::kRaise`: Raise an exception if a key already exists (default).
   *
   * @param policy The metadata update policy to be used by this operator.
   */
  void metadata_policy(MetadataPolicy policy) { dynamic_metadata_->policy(policy); }

  /**
   * @brief If no `CudaStreamPool` parameter or argument already exists, add a default one.
   *
   * This method is available to be called by derived classes to add a default CudaStreamPool in
   * This method is available to be called by derived classes to add a default `CudaStreamPool` in`
   * the case that the user did not pass one in as an argument to `make_operator`.
   *
   * This function will not add an additional CUDA stream pool if one was already passed in as an
   * argument to `make_operator` (i.e. it is in resources_) or if a "cuda_stream_pool" parameter
   * already exists in the operator spec.
   *
   * @param dev_id The device id for the CudaStreamPool.
   * @param stream_flags The stream flags for any streams allocated by the pool.
   * @param stream_priority The stream priority for any streams allocated by the pool.
   * @param reserved_size The number of initial streams to reserve in the pool.
   * @param max_size The maximum number of streams that can be allocated by the pool
   * (0 = unbounded).
   */
  [[deprecated(
      "Operator::add_cuda_stream_pool() is deprecated and will be removed in a future release. "
      "A default CUDA stream pool is automatically added. To provide a customized CUDA stream "
      "instead, just pass a std::shared_ptr<CudaStreamPool> as created via "
      "Fragment::make_resource<CudaStreamPool> directly as an unnamed positional argument to "
      "Fragment::make_operator.")]]
  void add_cuda_stream_pool(int32_t dev_id = 0, uint32_t stream_flags = 0,
                            int32_t stream_priority = 0, uint32_t reserved_size = 1,
                            uint32_t max_size = 0);

  /**@brief Return the Receiver corresponding to a specific input port.
   *
   * @param port_name The name of the input port.
   * @return The Receiver corresponding to the input port, if it exists. Otherwise, return nullopt.
   */
  std::optional<std::shared_ptr<Receiver>> receiver(const std::string& port_name);

  /**@brief Return the Transmitter corresponding to a specific output port.
   *
   * @param port_name The name of the output port.
   * @return The Transmitter corresponding to the output port, if it exists. Otherwise, return
   * nullopt.
   */
  std::optional<std::shared_ptr<Transmitter>> transmitter(const std::string& port_name);

  /**@brief Set the queue policy to be used by an input or output port.
   *
   * The following IOSpec::QueuePolicy values are supported:
   *
   * - QueuePolicy::kPop    - If the queue is full, pop the oldest item, then add the new one.
   * - QueuePolicy::kReject - If the queue is full, reject (discard) the new item.
   * - QueuePolicy::kFault  - If the queue is full, log a warning and reject the new item.
   *
   * @param port_name The name of the port.
   * @param port_type Enum flag indicating whether `port_name` specifies an input or output port.
   * @param policy The queue policy to set for the port.
   */
  void queue_policy(const std::string& port_name, IOSpec::IOType port_type = IOSpec::IOType::kInput,
                    IOSpec::QueuePolicy policy = IOSpec::QueuePolicy::kFault);
  const std::shared_ptr<IOSpec>& input_exec_spec();
  const std::shared_ptr<IOSpec>& output_exec_spec();
  const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func();
  std::shared_ptr<Operator> self_shared();

  /**
   * @brief Information about a flow connection between two operators.
   *
   * This struct encapsulates all the necessary information about a connection between two
   * operators, including the source and destination operators, their respective ports, and port
   * specifications.
   */
  struct FlowInfo {
    /**
     * @brief Construct a new FlowInfo object.
     *
     * @param curr_operator The source operator of the flow.
     * @param output_port_name The name of the output port on the source operator.
     * @param next_operator The destination operator of the flow.
     * @param input_port_name The name of the input port on the destination operator.
     */
    FlowInfo(const std::shared_ptr<Operator>& curr_operator, const std::string& output_port_name,
             const std::shared_ptr<Operator>& next_operator, const std::string& input_port_name)
        : curr_operator(curr_operator),
          output_port_name(output_port_name),
          output_port_spec(curr_operator->spec()->outputs()[output_port_name]),
          next_operator(next_operator),
          input_port_name(input_port_name),
          input_port_spec(next_operator->spec()->inputs()[input_port_name]) {}

    /// The source operator of the flow connection
    const std::shared_ptr<Operator> curr_operator;
    /// The name of the output port on the source operator
    const std::string output_port_name;
    /// The specification of the output port
    const std::shared_ptr<IOSpec> output_port_spec;
    /// The destination operator of the flow connection
    const std::shared_ptr<Operator> next_operator;
    /// The name of the input port on the destination operator
    const std::string input_port_name;
    /// The specification of the input port
    const std::shared_ptr<IOSpec> input_port_spec;
  };

  /**
   * @brief Get the list of next flows connected to this operator.
   *
   * @return A vector of FlowInfo objects representing the flows to downstream operators.
   */
  const std::vector<std::shared_ptr<FlowInfo>>& next_flows();

  /**
   * @brief Add a dynamic flow from this operator to another operator using a FlowInfo object.
   *
   * @param flow The flow information object describing the connection between operators.
   */
  void add_dynamic_flow(const std::shared_ptr<FlowInfo>& flow);

  /**
   * @brief Add multiple dynamic flows from this operator using a list of FlowInfo objects.
   *
   * @param flows List of flow information objects describing the connections between operators.
   */
  void add_dynamic_flow(const std::vector<std::shared_ptr<FlowInfo>>& flows);

  /**
   * @brief Add a dynamic flow from this operator to another operator with specified output port.
   *
   * @param curr_output_port_name The name of the output port on this operator to connect from.
   * @param next_op The downstream operator to connect to.
   * @param next_input_port_name The name of the input port on the downstream operator to connect
   * to. If not specified, the first available input port will be used.
   */
  void add_dynamic_flow(const std::string& curr_output_port_name,
                        const std::shared_ptr<Operator>& next_op,
                        const std::string& next_input_port_name = "");

  /**
   * @brief Add a dynamic flow from this operator to another operator using default output port.
   *
   * @param next_op The downstream operator to connect to.
   * @param next_input_port_name The name of the input port on the downstream operator to connect
   * to. If not specified, the first available input port will be used.
   */
  void add_dynamic_flow(const std::shared_ptr<Operator>& next_op,
                        const std::string& next_input_port_name = "");

  /**
   * @brief Get the list of dynamic flows that have been added to this operator.
   *
   * @return A shared pointer to a vector of FlowInfo objects representing the dynamic flows.
   */
  const std::shared_ptr<std::vector<std::shared_ptr<FlowInfo>>>& dynamic_flows();

  /**
   * @brief Locate a flow info in the operator's next flows based on a given predicate.
   *
   * @param predicate Lambda function that takes a FlowInfo shared pointer and returns a boolean.
   * @return Shared pointer to the matching FlowInfo, or nullptr if not found.
   */
  const std::shared_ptr<Operator::FlowInfo>& find_flow_info(
      const std::function<bool(const std::shared_ptr<Operator::FlowInfo>&)>& predicate);

  /**
   * @brief Find all FlowInfo objects in the operator's next flows that match a given condition.
   *
   * @param predicate A lambda function that takes a shared pointer to a FlowInfo object and returns
   * a boolean.
   * @return A vector of shared pointers to the matching FlowInfo objects.
   */
  std::vector<std::shared_ptr<Operator::FlowInfo>> find_all_flow_info(
      const std::function<bool(const std::shared_ptr<Operator::FlowInfo>&)>& predicate);

  /**
   * @brief Get the internal asynchronous condition for the operator.
   *
   * @return A shared pointer to the internal asynchronous condition.
   */
  std::shared_ptr<holoscan::AsynchronousCondition> async_condition();

  /**
   * @brief Stop the execution of the operator.
   *
   * This method is used to stop the execution of the operator by setting the internal async
   * condition to EVENT_NEVER state, which sets the scheduling condition to NEVER.
   * Once stopped, the operator will not be scheduled for execution
   * (the `compute()` method will not be called).
   *
   * Note that executing this method does not trigger the operator's `stop()` method.
   * The `stop()` method is called only when the scheduler deactivates all operators together.
   */
  void stop_execution();

  /**
   * @brief Get the ExecutionContext object.
   *
   * @return The shared pointer to the ExecutionContext object.
   */
  virtual std::shared_ptr<holoscan::ExecutionContext> execution_context() const;

  /**
   * @brief Ensure the contexts (input/output/execution) for the operator.
   *
   * This method is called by the GXFExecutor when the operator is initialized.
   */
  void ensure_contexts();

  /**
   * @brief Internal method to clean up operator resources and prevent circular references.
   *
   * This is an internal method called automatically by the GXFWrapper during operator shutdown
   * (in `GXFWrapper::stop()` method).
   * It resets std::shared_ptr fields and std::function objects such as `input_exec_spec_`,
   * `output_exec_spec_`, `next_flows_`, `dynamic_flows_`, and `dynamic_flow_func_` to break
   * potential circular references between connected Operator objects.
   *
   * @warning This is an internal method that should never be called directly by user code.
   *          Improper use can lead to undefined behavior.
   */
  virtual void release_internal_resources();

  /// Reset any backend-specific objects associated with this operator (e.g. GXF GraphEntity)
  void reset_backend_objects() override;

  /// Set the parameters based on defaults (sets GXF parameters for GXF operators)
  virtual void set_parameters();

 protected:
  // Making the following classes as friend classes to allow them to access
  // get_consolidated_input_label, num_published_messages_map, update_input_message_label,
  // reset_input_message_labels and update_published_messages functions, which should only be called
  // externally by them
  friend class AnnotatedDoubleBufferReceiver;
  friend class AnnotatedDoubleBufferTransmitter;
  friend class HoloscanAsyncBufferReceiver;
  friend class HoloscanAsyncBufferTransmitter;
  friend class HoloscanUcxTransmitter;
  friend class HoloscanUcxReceiver;
  friend class DFFTCollector;

  // Make GXFExecutor a friend class so it can call protected initialization methods
  friend class holoscan::gxf::GXFExecutor;
  // Fragment must be able to call set_self_shared
  friend class Fragment;

  friend gxf_result_t deannotate_message(gxf_uid_t* uid, const gxf_context_t& context, Operator* op,
                                         const char* receiver_name);
  friend gxf_result_t annotate_message(gxf_uid_t uid, const gxf_context_t& context, Operator* op,
                                       const char* transmitter_name);

  /**
   * @brief This function creates a GraphEntity corresponding to the operator
   * @param context The GXF context.
   * @param entity_prefix prefix to add to the operator's name when creating the GraphEntity.
   * @return The GXF entity eid corresponding to the graph entity.
   */
  gxf_uid_t initialize_graph_entity(void* context, const std::string& entity_prefix = "");

  /**
   * @brief Initialize the internal asynchronous condition to control the operator execution.
   *
   * This method is called by the GXFExecutor when the operator is initialized.
   */
  void initialize_async_condition();

  /**
   * @brief Add this operator as the codelet in the GXF GraphEntity
   *
   * @return The codelet component id corresponding to GXF codelet.
   */
  virtual gxf_uid_t add_codelet_to_graph_entity();

  /// Initialize conditions and add GXF conditions to graph_entity_
  void initialize_conditions();

  /// Initialize resources and add GXF resources to graph_entity_
  void initialize_resources();

  using ComponentBase::update_params_from_args;

  /// Update parameters based on the specified arguments
  void update_params_from_args();

  /** @brief Replace any "receiver" supplied as a string with the actual receiver of that name
   *
   * Can only be called after GXFExecutor::create_input_port so the input ports (receivers) exist.
   */
  void update_connector_arguments();

  /** @brief Determine ports whose transmitter or receiver are associated with an Arg for a
   * Condition.
   *
   * Should be called before GXFExecutor::create_input_port or GXFExecutor::create_output_port.
   */
  void find_ports_used_by_condition_args();

  /**
   * @brief This function returns a consolidated MessageLabel for all the input ports of an
   * Operator. If there is no input port (root Operator), then a new MessageLabel with the current
   * Operator and default receive timestamp is returned.
   *
   * @return The consolidated MessageLabel
   */

  MessageLabel get_consolidated_input_label();

  /**
   * @brief Update the input_message_labels map with the given MessageLabel a
   * corresponding input_name
   *
   * @param input_name The input port name for which the MessageLabel is updated
   * @param m The new MessageLabel that will be set for the input port
   */
  void update_input_message_label(std::string input_name, MessageLabel m) {
    input_message_labels[input_name] = m;
  }

  /**
   * @brief Delete the input_message_labels map entry for the given input_name
   *
   * @param input_name The input port name for which the MessageLabel is deleted
   */
  void delete_input_message_label(std::string input_name) {
    input_message_labels.erase(input_name);
  }

  /**
   * @brief Reset the input message labels to clear all its contents. This is done for a leaf
   * operator when it finishes its execution as it is assumed that all its inputs are processed.
   */
  void reset_input_message_labels() { input_message_labels.clear(); }

  /**
   * @brief Get the number of published messages for each output port indexed by the output port
   * name.
   *
   * The function is utilized by the DFFTCollector to update the DataFlowTracker with the number of
   * published messages for root operators.
   *
   * @return The map of the number of published messages for every output name.
   */
  std::map<std::string, uint64_t> num_published_messages_map() {
    return num_published_messages_map_;
  }

  /**
   * @brief This function updates the number of published messages for a given output port.
   *
   * @param output_name The name of the output port
   */
  void update_published_messages(std::string output_name);

  /// Initialize the next flows for the operator.
  void initialize_next_flows();

  std::vector<std::string>& non_default_input_ports() { return non_default_input_ports_; }
  std::vector<std::string>& non_default_output_ports() { return non_default_output_ports_; }

  void set_input_exec_spec(const std::shared_ptr<IOSpec>& input_exec_spec);
  void set_output_exec_spec(const std::shared_ptr<IOSpec>& output_exec_spec);
  void set_dynamic_flows(
      const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func);
  void set_self_shared(const std::shared_ptr<Operator>& this_op);

  bool is_initialized_ = false;                         ///< Whether the operator is initialized.
  OperatorType operator_type_ = OperatorType::kNative;  ///< The type of the operator.
  std::shared_ptr<OperatorSpec> spec_;                  ///< The operator spec of the operator.
  std::unordered_map<std::string, std::shared_ptr<Condition>>
      conditions_;  ///< The conditions of the operator.
  std::unordered_map<std::string, std::shared_ptr<Resource>>
      resources_;                                           ///< The resources used by the operator.
  std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity_;  ///< GXF graph entity corresponding to
                                                            ///< the Operator
  /// The asynchronous condition to control the operator execution.
  std::shared_ptr<holoscan::AsynchronousCondition> internal_async_condition_;
  /// The execution context for the operator.
  std::shared_ptr<ExecutionContext> execution_context_{};

  std::shared_ptr<MetadataDictionary> dynamic_metadata_ =
      std::make_shared<MetadataDictionary>();  ///< The metadata dictionary for the operator.
  std::optional<bool> is_metadata_enabled_ =
      std::nullopt;  ///< Flag to enable or disable metadata for the operator.
                     ///< If not set, the value from the Fragment is used.

  std::shared_ptr<IOSpec> input_exec_spec_;   ///< The input execution port specification.
  std::shared_ptr<IOSpec> output_exec_spec_;  ///< The output execution port specification.
  std::function<void(const std::shared_ptr<Operator>&)> dynamic_flow_func_ = nullptr;
  std::weak_ptr<Operator> self_shared_;
  std::shared_ptr<std::vector<std::shared_ptr<FlowInfo>>> next_flows_;
  std::shared_ptr<std::vector<std::shared_ptr<FlowInfo>>> dynamic_flows_;

 private:
  /// An empty shared pointer to FlowInfo.
  static inline const std::shared_ptr<FlowInfo> kEmptyFlowInfo{nullptr};

  ///  Set the operator codelet or any other backend codebase.
  void set_op_backend();

  bool has_ucx_connector();  ///< Check if the operator has any UCX connectors.

  /// The MessageLabel objects corresponding to the input ports indexed by the input port.
  std::unordered_map<std::string, MessageLabel> input_message_labels;

  /// The number of published messages for each output indexed by output names.
  std::map<std::string, uint64_t> num_published_messages_map_;

  // Keep track of which ports have a user-assigned condition involving its receiver or
  // transmitter (a default condition will NOT be added to any such ports).
  std::vector<std::string> non_default_input_ports_;
  std::vector<std::string> non_default_output_ports_;

  /// The backend Codelet or other codebase pointer. It is used for DFFT.
  void* op_backend_ptr = nullptr;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_HPP */
