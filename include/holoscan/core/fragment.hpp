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

#ifndef HOLOSCAN_CORE_FRAGMENT_HPP
#define HOLOSCAN_CORE_FRAGMENT_HPP

#include <future>       // for std::future
#include <iostream>     // for std::cout
#include <memory>       // for std::shared_ptr
#include <set>          // for std::set
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if_t, std::is_constructible
#include <utility>      // for std::pair

#include "common.hpp"
#include "config.hpp"
#include "dataflow_tracker.hpp"
#include "executor.hpp"
#include "graph.hpp"
#include "network_context.hpp"
#include "scheduler.hpp"

namespace holoscan {

/**
 * @brief The fragment of the application.
 *
 * A fragment is a building block of the Application. It is a Directed Acyclic Graph (DAG) of
 * operators. A fragment can be assigned to a physical node of a Holoscan cluster during execution.
 * The run-time execution manages communication across fragments. In a Fragment, Operators (Graph
 * Nodes) are connected to each other by flows (Graph Edges).
 */
class Fragment {
 public:
  Fragment() = default;
  virtual ~Fragment() = default;

  Fragment(Fragment&&) = default;

  Fragment& operator=(Fragment&&) = default;

  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this fragment (for chaining).
   */
  Fragment& name(const std::string& name) &;

  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this fragment (for chaining).
   */
  Fragment&& name(const std::string& name) &&;

  /**
   * @brief Get the name of the fragment.
   *
   * @return The name of the fragment.
   */
  const std::string& name() const;

  /**
   * @brief Set the application of the fragment.
   *
   * @param app The pointer to the application of the fragment.
   * @return The reference to this fragment (for chaining).
   */
  Fragment& application(Application* app);

  /**
   * @brief Get the application of the fragment.
   *
   * @return The pointer to the application of the fragment.
   */
  Application* application() const;

  /**
   * @brief Set the configuration of the fragment.
   *
   * The configuration file is a YAML file that has the information of GXF extension paths and some
   * parameter values for operators.
   *
   * The `extensions` field in the YAML configuration file is a list of GXF extension paths.
   * The paths can be absolute or relative to the current working directory, considering paths in
   * `LD_LIBRARY_PATH` environment variable.
   *
   * The paths can consist of the following parts:
   *
   * - GXF core extensions
   *   - built-in extensions such as `libgxf_std.so` and `libgxf_cuda.so`.
   *   - `libgxf_std.so`, `libgxf_cuda.so`, `libgxf_multimedia.so`, `libgxf_serialization.so` are
   *     always loaded by default.
   *   - GXF core extensions are copied to the `lib` directory of the build/installation directory.
   * - Other GXF extensions
   *   - GXF extensions that are required for operators that this fragment uses.
   *   - some core GXF extensions such as `libgxf_stream_playback.so` are always loaded by default.
   *   - these paths are usually relative to the build/installation directory.
   *
   * The extension paths are used to load dependent GXF extensions at runtime when
   * `::run()` method is called.
   *
   * For other fields in the YAML file, you can freely define the parameter values for
   * operators/fragments.
   *
   * For example:
   *
   * ```yaml
   * extensions:
   *   - libmy_recorder.so
   *
   * replayer:
   *   directory: "../data/endoscopy/video"
   *   basename: "surgical_video"
   *   frame_rate: 0   # as specified in timestamps
   *   repeat: false   # default: false
   *   realtime: true  # default: true
   *   count: 0        # default: 0 (no frame count restriction)
   *
   * recorder:
   *   out_directory: "/tmp"
   *   basename: "tensor_out"
   * ```
   *
   * You can get the value of this configuration file by calling `::from_config()` method.
   *
   * If the application is executed with `--config` option or HOLOSCAN_CONFIG_PATH environment,
   * the configuration file is overridden by the configuration file specified by the option or
   * environment variable.
   *
   * @param config_file The path to the configuration file.
   * @param prefix The prefix string that is prepended to the key of the configuration. (not
   * implemented yet)
   */
  void config(const std::string& config_file, const std::string& prefix = "");

  /**
   * @brief Set the configuration of the fragment.
   *
   * If you want to set the configuration of the fragment manually, you can use this method.
   * However, it is recommended to use `::config(const std::string&, const std::string&)` method
   * because once you set the configuration manually, you cannot get the configuration from the
   * override file (through `--config` option or HOLOSCAN_CONFIG_PATH environment variable).
   *
   * @param config The shared pointer to the configuration of the fragment (`Config` object).
   */
  void config(std::shared_ptr<Config>& config);

  /**
   * @brief Get the configuration of the fragment.
   *
   * @return The reference to the configuration of the fragment (`Config` object.)
   */
  Config& config();

  /**
   * @brief Get the graph of the fragment.
   *
   * @return The reference to the graph of the fragment (`Graph` object.)
   */
  OperatorGraph& graph();

  /**
   * @brief Get the executor of the fragment.
   *
   * @return The reference to the executor of the fragment (`Executor` object.)
   */
  Executor& executor();

  /**
   * @brief Get the scheduler used by the executor
   *
   * @return The reference to the scheduler of the fragment's executor (`Scheduler` object.)
   */
  std::shared_ptr<Scheduler> scheduler();

  // /**
  //  * @brief Set the scheduler used by the executor
  //  *
  //  * @param scheduler The scheduler to be added.
  //  */
  void scheduler(const std::shared_ptr<Scheduler>& scheduler);

  /**
   * @brief Get the network context used by the executor
   *
   * @return The reference to the network context of the fragment's executor (`NetworkContext`
   * object.)
   */
  std::shared_ptr<NetworkContext> network_context();

  // /**
  //  * @brief Set the network context used by the executor
  //  *
  //  * @param network_context The network context to be added.
  //  */
  void network_context(const std::shared_ptr<NetworkContext>& network_context);

  /**
   * @brief Get the Argument(s) from the configuration file.
   *
   * For the given key, this method returns the value of the configuration file.
   *
   * For example:
   *
   * ```yaml
   * source: "replayer"
   * do_record: false   # or 'true' if you want to record input video stream.
   *
   * aja:
   *   width: 1920
   *   height: 1080
   *   rdma: true
   * ```
   *
   * `from_config("aja")` returns an ArgList (vector-like) object that contains the following
   * items:
   *
   * - `Arg("width") = 1920`
   * - `Arg("height") = 1080`
   * - `Arg("rdma") = true`
   *
   * You can use '.' (dot) to access nested fields.
   *
   * `from_config("aja.rdma")` returns an ArgList object that contains only one item and it can be
   * converted to `bool` through `ArgList::as()` method:
   *
   * ```cpp
   * bool is_rdma = from_config("aja.rdma").as<bool>();
   * ```
   *
   * @param key The key of the configuration.
   * @return The argument list of the configuration for the key.
   */
  ArgList from_config(const std::string& key);

  /**
   * @brief Create a new operator.
   *
   * @tparam OperatorT The type of the operator.
   * @param name The name of the operator.
   * @param args The arguments for the operator.
   * @return The shared pointer to the operator.
   */
  template <typename OperatorT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<OperatorT> make_operator(const StringT& name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating operator '{}'", name);
    auto op = std::make_shared<OperatorT>(std::forward<ArgsT>(args)...);
    op->name(name);
    op->fragment(this);
    auto spec = std::make_shared<OperatorSpec>(this);
    op->setup(*spec.get());
    op->spec(spec);

    // We used to initialize operator here, but now it is initialized in initialize_fragment
    // function after a graph of a fragment has been composed.

    return op;
  }
  /**
   * @brief Create a new operator.
   *
   * @tparam OperatorT The type of the operator.
   * @param args The arguments for the operator.
   * @return The shared pointer to the operator.
   */
  template <typename OperatorT, typename... ArgsT>
  std::shared_ptr<OperatorT> make_operator(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating operator");
    auto op = make_operator<OperatorT>("noname_operator", std::forward<ArgsT>(args)...);
    return op;
  }

  /**
   * @brief Create a new (operator) resource.
   *
   * @tparam ResourceT The type of the resource.
   * @param name The name of the resource.
   * @param args The arguments for the resource.
   * @return The shared pointer to the resource.
   */
  template <typename ResourceT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<ResourceT> make_resource(const StringT& name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating resource '{}'", name);
    auto resource = std::make_shared<ResourceT>(std::forward<ArgsT>(args)...);
    resource->name(name);
    resource->fragment(this);
    auto spec = std::make_shared<ComponentSpec>(this);
    resource->setup(*spec.get());
    resource->spec(spec);

    // Skip initialization. `resource->initialize()` is done in GXFOperator::initialize()

    return resource;
  }
  /**
   * @brief Create a new (operator) resource.
   *
   * @tparam ResourceT The type of the resource.
   * @param args The arguments for the resource.
   * @return The shared pointer to the resource.
   */
  template <typename ResourceT, typename... ArgsT>
  std::shared_ptr<ResourceT> make_resource(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating resource");
    auto resource = make_resource<ResourceT>("noname_resource", std::forward<ArgsT>(args)...);
    return resource;
  }

  /**
   * @brief Create a new condition.
   *
   * @tparam ConditionT The type of the condition.
   * @param name The name of the condition.
   * @param args The arguments for the condition.
   * @return The shared pointer to the condition.
   */
  template <typename ConditionT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<ConditionT> make_condition(const StringT& name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating condition '{}'", name);
    auto condition = std::make_shared<ConditionT>(std::forward<ArgsT>(args)...);
    condition->name(name);
    condition->fragment(this);
    auto spec = std::make_shared<ComponentSpec>(this);
    condition->setup(*spec.get());
    condition->spec(spec);

    // Skip initialization. `condition->initialize()` is done in GXFOperator::initialize()

    return condition;
  }

  /**
   * @brief Create a new condition.
   *
   * @tparam ConditionT The type of the condition.
   * @param args The arguments for the condition.
   * @return The shared pointer to the condition.
   */
  template <typename ConditionT, typename... ArgsT>
  std::shared_ptr<ConditionT> make_condition(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating condition");
    auto condition = make_condition<ConditionT>("noname_condition", std::forward<ArgsT>(args)...);
    return condition;
  }

  /**
   * @brief Create a new scheduler.
   *
   * @tparam SchedulerT The type of the scheduler.
   * @param name The name of the scheduler.
   * @param args The arguments for the scheduler.
   * @return The shared pointer to the scheduler.
   */
  template <typename SchedulerT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<SchedulerT> make_scheduler(const StringT& name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating scheduler '{}'", name);
    auto scheduler = std::make_shared<SchedulerT>(std::forward<ArgsT>(args)...);
    scheduler->name(name);
    scheduler->fragment(this);
    auto spec = std::make_shared<ComponentSpec>(this);
    scheduler->setup(*spec.get());
    scheduler->spec(spec);

    // Skip initialization. `scheduler->initialize()` is done in GXFExecutor::run()

    return scheduler;
  }

  /**
   * @brief Create a new scheduler.
   *
   * @tparam SchedulerT The type of the scheduler.
   * @param args The arguments for the scheduler.
   * @return The shared pointer to the scheduler.
   */
  template <typename SchedulerT, typename... ArgsT>
  std::shared_ptr<SchedulerT> make_scheduler(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating scheduler");
    auto scheduler = make_scheduler<SchedulerT>("", std::forward<ArgsT>(args)...);
    return scheduler;
  }

  /**
   * @brief Create a new network context.
   *
   * @tparam NetworkContextT The type of the network context.
   * @param name The name of the network context.
   * @param args The arguments for the network context.
   * @return The shared pointer to the network context.
   */
  template <typename NetworkContextT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<NetworkContextT> make_network_context(const StringT& name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating network context '{}'", name);
    auto network_context = std::make_shared<NetworkContextT>(std::forward<ArgsT>(args)...);
    network_context->name(name);
    network_context->fragment(this);
    auto spec = std::make_shared<ComponentSpec>(this);
    network_context->setup(*spec.get());
    network_context->spec(spec);

    // Skip initialization. `network_context->initialize()` is done in GXFExecutor::run()

    return network_context;
  }

  /**
   * @brief Create a new network context.
   *
   * @tparam NetworkContextT The type of the network context.
   * @param args The arguments for the network context.
   * @return The shared pointer to the network context.
   */
  template <typename NetworkContextT, typename... ArgsT>
  std::shared_ptr<NetworkContextT> make_network_context(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating network_context");
    auto network_context = make_network_context<NetworkContextT>("", std::forward<ArgsT>(args)...);
    return network_context;
  }

  /**
   * @brief Add an operator to the graph.
   *
   * The information of the operator is stored in the Graph object.
   * If the operator is already added, this method does nothing.
   *
   * @param op The operator to be added.
   */
  virtual void add_operator(const std::shared_ptr<Operator>& op);

  /**
   * @brief Add a flow between two operators.
   *
   * An output port of the upstream operator is connected to an input port of the
   * downstream operator.
   * The information about the flow (edge) is stored in the Graph object.
   *
   * If the upstream operator or the downstream operator is not in the graph, it will be added to
   * the graph.
   *
   * If there are multiple output ports in the upstream operator or multiple input ports in the
   * downstream operator, it shows an error message.
   *
   * @param upstream_op The upstream operator.
   * @param downstream_op The downstream operator.
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op);

  /**
   * @brief Add a flow between two operators.
   *
   * An output port of the upstream operator is connected to an input port of the
   * downstream operator.
   * The information about the flow (edge) is stored in the Graph object.
   *
   * If the upstream operator or the downstream operator is not in the graph, it will be added to
   * the graph.
   *
   * In `port_pairs`, an empty port name ("") can be used for specifying a port name if the operator
   * has only one input/output port.
   *
   * If a non-existent port name is specified in `port_pairs`, it first checks if there is a
   * parameter with the same name but with a type of `std::vector<holoscan::IOSpec*>` in the
   * downstream operator.
   * If there is such a parameter (e.g., `receivers`), it creates a new input port with a specific
   * label (`<parameter name>:<index>`. e.g., `receivers:0`), otherwise it shows an error message.
   *
   * For example, if a parameter `receivers` want to have an arbitrary number of receivers,
   *
   *     class HolovizOp : public holoscan::ops::GXFOperator {
   *         ...
   *         private:
   *           Parameter<std::vector<holoscan::IOSpec*>> receivers_;
   *         ...
   *
   * Instead of creating a fixed number of input ports (e.g., `source_video` and `tensor`) and
   * assigning them to the parameter (`receivers`):
   *
   *     void HolovizOp::setup(OperatorSpec& spec) {
   *       ...
   *
   *       auto& in_source_video = spec.input<holoscan::gxf::Entity>("source_video");
   *       auto& in_tensor = spec.input<holoscan::gxf::Entity>("tensor");
   *
   *       spec.param(receivers_,
   *                  "receivers",
   *                  "Input Receivers",
   *                  "List of input receivers.",
   *                  {&in_source_video, &in_tensor});
   *       ...
   *
   * You can skip the creation of input ports and assign them to the parameter (`receivers`) as
   * follows:
   *
   *     void HolovizOp::setup(OperatorSpec& spec) {
   *       ...
   *       spec.param(receivers_,
   *                  "receivers",
   *                  "Input Receivers",
   *                  "List of input receivers.",
   *                  {&in_source_video, &in_tensor});
   *       ...
   *
   * This makes the following code possible in the Application's `compose()` method:
   *
   *     add_flow(source, visualizer_format_converter);
   *     add_flow(visualizer_format_converter, visualizer, {{"", "receivers"}});
   *
   *     add_flow(source, format_converter);
   *     add_flow(format_converter, inference);
   *     add_flow(inference, visualizer, {{"", "receivers"}});
   *
   * Instead of:
   *
   *     add_flow(source, visualizer_format_converter);
   *     add_flow(visualizer_format_converter, visualizer, {{"", "source_video"}});
   *
   *     add_flow(source, format_converter);
   *     add_flow(format_converter, inference);
   *     add_flow(inference, visualizer, {{"", "tensor"}});
   *
   * By using the parameter (`receivers`) with `std::vector<holoscan::IOSpec*>` type, the framework
   * creates input ports (`receivers:0` and `receivers:1`) implicitly and connects them (and adds
   * the references of the input ports to the `receivers` vector).
   *
   * @param upstream_op The upstream operator.
   * @param downstream_op The downstream operator.
   * @param port_pairs The port pairs. The first element of the pair is the port of the upstream
   * operator and the second element is the port of the downstream operator.
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs);

  /**
   * @brief Compose a graph.
   *
   * The graph is composed by adding operators and flows in this method.
   */
  virtual void compose();

  /**
   * @brief Initialize the graph and run the graph.
   *
   * This method calls `compose()` to compose the graph, and runs the graph.
   */
  virtual void run();

  /**
   * @brief Initialize the graph and run the graph asynchronously.
   *
   * This method calls `compose()` to compose the graph, and runs the graph asynchronously.
   *
   * @return The future object.
   */
  virtual std::future<void> run_async();

  /**
   * @brief Turn on data frame flow tracking.
   *
   * A reference to a DataFlowTracker object is returned rather than a pointer so that the
   * developers can use it as an object without unnecessary pointer dereferencing.
   *
   * @param num_start_messages_to_skip The number of messages to skip at the beginning.
   * @param num_last_messages_to_discard The number of messages to discard at the end.
   * @param latency_threshold The minimum end-to-end latency in milliseconds to account for
   * in the end-to-end latency metric calculations.
   * @return A reference to the DataFlowTracker object in which results will be
   * stored.
   */
  DataFlowTracker& track(uint64_t num_start_messages_to_skip = kDefaultNumStartMessagesToSkip,
                         uint64_t num_last_messages_to_discard = kDefaultNumLastMessagesToDiscard,
                         int latency_threshold = kDefaultLatencyThreshold);

  /**
   * @brief Get the DataFlowTracker object for this fragment.
   *
   * @return The pointer to the DataFlowTracker object.
   */
  DataFlowTracker* data_flow_tracker() { return data_flow_tracker_.get(); }

  /**
   * @brief Calls compose() if the graph is not composed yet.
   */
  void compose_graph();

 protected:
  friend class Application;  // to access 'scheduler_' in Application
  friend class AppDriver;

  template <typename ConfigT, typename... ArgsT>
  std::shared_ptr<Config> make_config(ArgsT&&... args) {
    return std::make_shared<ConfigT>(std::forward<ArgsT>(args)...);
  }

  template <typename GraphT>
  std::unique_ptr<GraphT> make_graph() {
    return std::make_unique<GraphT>();
  }

  template <typename ExecutorT>
  std::shared_ptr<Executor> make_executor() {
    return std::make_shared<ExecutorT>(this);
  }

  template <typename ExecutorT, typename... ArgsT>
  std::unique_ptr<Executor> make_executor(ArgsT&&... args) {
    return std::make_unique<ExecutorT>(std::forward<ArgsT>(args)...);
  }

  std::string name_;                      ///< The name of the fragment.
  Application* app_ = nullptr;            ///< The application that this fragment belongs to.
  std::shared_ptr<Config> config_;        ///< The configuration of the fragment.
  std::unique_ptr<OperatorGraph> graph_;  ///< The graph of the fragment.
  std::shared_ptr<Executor> executor_;    ///< The executor for the fragment.
  std::shared_ptr<Scheduler> scheduler_;  ///< The scheduler used by the executor
  std::shared_ptr<NetworkContext> network_context_;  ///< The network_context used by the executor
  std::shared_ptr<DataFlowTracker> data_flow_tracker_;  ///< The DataFlowTracker for the fragment
  bool is_composed_ = false;                            ///< Whether the graph is composed or not.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_FRAGMENT_HPP */
