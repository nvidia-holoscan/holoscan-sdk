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

#ifndef HOLOSCAN_CORE_APPLICATION_HPP
#define HOLOSCAN_CORE_APPLICATION_HPP

#include <iostream>       // for std::cout
#include <memory>         // for std::shared_ptr
#include <set>            // for std::set
#include <string>         // for std::string
#include <type_traits>    // for std::enable_if_t, std::is_constructible
#include <unordered_map>  // for std::unordered_map
#include <utility>        // for std::pair
#include <vector>         // for std::vector

#include "./fragment.hpp"
#include "dataflow_tracker.hpp"

#include "./app_worker.hpp"
#include "./cli_parser.hpp"

namespace holoscan {

// forward declaration
class AppDriver;
class AppDriverClient;

/**
 * @brief Utility function to create an application.
 *
 * @tparam AppT The type of the application to create.
 * @param args The arguments to pass to the application constructor.
 * @return The shared pointer to the created application.
 */
template <typename AppT, typename... ArgsT>
std::shared_ptr<AppT> make_application(ArgsT&&... args) {
  return std::make_shared<AppT>(std::forward<ArgsT>(args)...);
}

/**
 * @brief Application class.
 *
 * An application acquires and processes streaming data. An application is a collection of fragments
 * where each fragment can be allocated to execute on a physical node of a Holoscan cluster.
 */
class Application : public Fragment {
 public:
  /**
   * @brief Construct a new Application object.
   *
   * This constructor parses the command line for flags that are recognized by App Driver/Worker,
   * and removes all recognized flags so users can use the remaining flags for their own purposes.
   *
   * The command line arguments are retrieved from /proc/self/cmdline so that the single-fragment
   * application works as expected without any command line arguments.
   *
   * The arguments after processing arguments are stored in the `argv_` member variable and
   * the reference to the vector of arguments can be accessed through the `argv()` method.
   *
   * Example:
   *
   * ```cpp
   * #include <holoscan/holoscan.hpp>
   *
   * class MyPingApp : public holoscan::Application {
   * // ...
   * };
   *
   * int main(int argc, char** argv) {
   *   auto my_argv =
   *       holoscan::Application({"myapp", "--driver", "my_arg1", "--address=10.0.0.1"}).argv();
   *   HOLOSCAN_LOG_INFO(" my_argv: {}", fmt::join(my_argv, " "));
   *
   *   HOLOSCAN_LOG_INFO(
   *       "    argv: {} (argc: {}) ",
   *       fmt::join(std::vector<std::string>(argv, argv + argc), " "),
   *       argc);
   *
   *   auto app_argv = holoscan::Application().argv();  // do not use reference ('auto&') here
   *   HOLOSCAN_LOG_INFO("app_argv: {} (size: {})", fmt::join(app_argv, " "), app_argv.size());
   *
   *   auto app = holoscan::make_application<MyPingApp>();
   *   HOLOSCAN_LOG_INFO("app->argv() == app_argv: {}", app->argv() == app_argv);
   *
   *   app->run();
   *   return 0;
   * }
   *
   * // $ ./myapp --driver --input image.dat --address 10.0.0.20
   *
   * //  my_argv: myapp my_arg1
   * //     argv: ./myapp --driver --input image.dat --address 10.0.0.20 (argc: 6)
   * // app_argv: ./myapp --input image.dat (size: 3)
   * // app->argv() == app_argv: true
   * ```
   *
   * @param argv The command line arguments.
   */
  explicit Application(const std::vector<std::string>& argv = {});

  ~Application() override = default;

  /**
   * @brief Create a new fragment.
   *
   * @tparam FragmentT The type of the fragment to create.
   * @param name The name of the fragment.
   * @param args The arguments to pass to the fragment constructor.
   * @return The shared pointer to the created fragment.
   */
  template <typename FragmentT = Fragment, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<Fragment> make_fragment(StringT name, ArgsT&&... args) {
    auto fragment = std::make_shared<FragmentT>(std::forward<ArgsT>(args)...);
    fragment->name(name);
    fragment->application(this);
    // Set the fragment config to the application config.
    fragment->config(config_);
    return fragment;
  }

  /**
   * @brief Create a new fragment.
   *
   * @tparam FragmentT The type of the fragment to create.
   * @param args The arguments to pass to the fragment constructor.
   * @return The shared pointer to the created fragment.
   */
  template <typename FragmentT, typename... ArgsT>
  std::shared_ptr<FragmentT> make_fragment(ArgsT&&... args) {
    auto fragment = std::make_shared<FragmentT>(std::forward<ArgsT>(args)...);
    fragment->application(this);
    // Set the fragment config to the application config.
    fragment->config(config_);
    return fragment;
  }

  /**
   * @brief Get the application description.
   *
   * @return The application description.
   */
  std::string& description();

  /**
   * @brief Set the application description
   *
   * @param desc The application description
   * @return The reference to this application (for chaining).
   */
  Application& description(const std::string& desc) &;

  /**
   * @brief Set the application description
   *
   * @param desc The application description
   * @return The reference to this application (for chaining).
   */
  Application&& description(const std::string& desc) &&;

  /**
   * @brief Get the application version.
   *
   * @return The application version.
   */
  std::string& version();

  /**
   * @brief Set the application version
   *
   * @param version The application version
   * @return The reference to this application (for chaining).
   */
  Application& version(const std::string& version) &;

  /**
   * @brief Set the application version
   *
   * @param version The application version
   * @return The reference to this application (for chaining).
   */
  Application&& version(const std::string& version) &&;

  /**
   * @brief Get the reference to the command line arguments after processing flags.
   *
   * The returned vector includes the executable name as the first element.
   *
   * @return The reference to the command line arguments after processing flags.
   */
  std::vector<std::string>& argv();

  /**
   * @brief Get the reference to the CLI options.
   *
   * @return The reference to the CLI options.
   */
  CLIOptions& options();

  /**
   * @brief Get the fragment connection graph.
   *
   * When two operators are connected through `add_flow(Fragment, Fragment)`, the fragment
   * connection graph is automatically updated. The fragment connection graph is used to assign
   * transmitters and receivers to the corresponding Operator instances in the fragment so that
   * the application can be executed in a distributed manner.
   *
   * @return The reference to the fragment connection graph (`Graph` object.)
   */
  FragmentGraph& fragment_graph();

  /**
   * @brief Add a fragment to the graph.
   *
   * The information of the fragment is stored in the Graph object.
   * If the fragment is already added, this method does nothing.
   *
   * @param frag The fragment to be added.
   */
  virtual void add_fragment(const std::shared_ptr<Fragment>& frag);

  // Inherit Fragment's add_flow methods (for Operator) in addition to the overloads below
  using Fragment::add_flow;

  /**
   * @brief Add a flow between two fragments.
   *
   * It takes two fragments and a vector of string pairs as arguments. The vector of string pairs is
   * used to connect the output ports of the first fragment to the input ports of the second
   * fragment. The input and output ports of the operators are specified as a string in the format
   * of `<operator name>.<port name>`. If the operator has only one input or output port, the port
   * name can be omitted.
   *
   * ```cpp
   * class App : public holoscan::Application {
   *  public:
   *   void compose() override {
   *     using namespace holoscan;
   *
   *     auto fragment1 = make_fragment<Fragment1>("fragment1");
   *     auto fragment2 = make_fragment<Fragment2>("fragment2");
   *
   *     add_flow(fragment1, fragment2, {{"blur_image", "sharpen_image"}});
   *   }
   * };
   * ```
   *
   * In the above example, the output port of the `blur_image` operator in `fragment1` is
   * connected to the input port of the `sharpen_image` operator in `fragment2`.
   * Since `blur_image` and `sharpen_image` operators have only one output/input port, the port
   * names are omitted.
   *
   * The information about the flow (edge) is stored in the Graph object and can be accessed through
   * the `fragment_graph()` method.
   *
   * If the upstream fragment or the downstream fragment is not in the graph, it will be added to
   * the graph.
   *
   * @param upstream_frag The upstream fragment.
   * @param downstream_frag The downstream fragment.
   * @param port_pairs The port pairs. The first element of the pair is the output port of the
   * operator in the upstream fragment and the second element is the input port of the operator in
   * the downstream fragment.
   */
  virtual void add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                        const std::shared_ptr<Fragment>& downstream_frag,
                        std::set<std::pair<std::string, std::string>> port_pairs);

  /**
   * @brief Calls compose() if the fragment graph is not composed yet.
   */
  void compose_graph() override;

  void run() override;

  std::future<void> run_async() override;

  /**
   * @brief Returns a map of fragment names to DataFlowTracker* corresponding to respective
   * fragments. The trackers will store cumulatively progressive timestamps, meaning a fragment
   * tracker will store the timestamps of operators in the previous fragments as well.
   *
   * @param num_start_messages_to_skip The number of start messages to skip.
   * @param num_last_messages_to_discard The number of last messages to discard.
   * @param latency_threshold The latency threshold.
   * @return std::unordered_map<std::string, DataFlowTracker*> Fragment name to DataFlowTracker*
   * mapping.
   */
  std::unordered_map<std::string, DataFlowTracker*> track_distributed(
      uint64_t num_start_messages_to_skip = kDefaultNumStartMessagesToSkip,
      uint64_t num_last_messages_to_discard = kDefaultNumLastMessagesToDiscard,
      int latency_threshold = kDefaultLatencyThreshold, bool is_limited_tracking = false);

  /**
   * @brief Determine whether metadata is enabled by default for operators in this application.
   *
   * Note that individual fragments may override the application-level default via
   * `Fragment::enable_metadata` and individual operators may override their fragment's default
   * via `Operator::enable_metadata`.
   *
   * @param enable Boolean indicating whether metadata is enabled.
   */
  bool is_metadata_enabled() const override;

  /**
   * @brief Deprecated method to enable or disable metadata for the application.
   *
   * Please use `enable_metadata` instead.
   *
   * @param enable Boolean indicating whether metadata should be enabled.
   */
  void is_metadata_enabled(bool) override;

  /**
   * @brief Enable or disable metadata for the application.
   *
   * Controls whether metadata is enabled or disabled by default for fragments in this application.
   * Individual fragments of a distributed application can override this setting using the
   * `Fragment::enable_metadata()` method. Similarly, individual operators can override the
   * default for their fragment by using the `Operator::enable_metadata()` method.
   *
   * @param enable Boolean indicating whether metadata should be enabled.
   */
  void enable_metadata(bool enable) override;

  /**
   * @brief Set the default metadata update policy to be used for fragments within this
   * application.
   *
   * The metadata policy determines how metadata is merged across multiple receive calls:
   *    - `MetadataPolicy::kUpdate`: Update the existing value when a key already exists.
   *    - `MetadataPolicy::kInplaceUpdate`: Update the existing MetadataObject's value in-place
   *    when a key already exists.
   *    - `MetadataPolicy::kReject`: Do not modify the existing value if a key already exists.
   *    - `MetadataPolicy::kRaise`: Raise an exception if a key already exists (default).
   *
   * For distributed applications, individual fragments can override the application-level
   * default via `Fragment::metadata_policy`.
   *
   * @param policy The metadata update policy to be used by this operator.
   */
  void metadata_policy(MetadataPolicy policy) override;

  /**
   * @brief Get the default metadata update policy used for fragments within this application.
   *
   * If a value was set for a specific fragment via `Fragment::metadata_policy` that value will
   * take precedence over this application default. Similarly individual operators can override
   * the default policy for their fragment via `Operator::metadata_policy`.
   *
   * @returns The default metadata update policy used by operators in this fragment.
   */
  MetadataPolicy metadata_policy() const override;

  /**
   * @brief Get the AppDriverClient for the application.
   *
   * @return The shared pointer to the AppDriverClient.
   */
  std::shared_ptr<service::AppDriverClient> app_driver_client() const;

  /**
   * @brief Initiate shutdown of the distributed application.
   *
   * This method initiates a shutdown of the distributed application by calling the
   * `initiate_shutdown` method on the AppDriverClient.
   *
   * @param fragment_name The name of the fragment initiating the shutdown.
   */
  void initiate_distributed_app_shutdown(const std::string& fragment_name);

 protected:
  friend class AppDriver;
  friend class AppWorker;

  /**
   * @brief Get the application driver.
   *
   * @return The reference to the application driver.
   */
  AppDriver& driver();

  /**
   * @brief Get the application worker.
   *
   * @return The reference to the application worker.
   */
  AppWorker& worker();

  void process_arguments();

  static expected<SchedulerType, ErrorCode> get_distributed_app_scheduler_env();
  static expected<bool, ErrorCode> get_stop_on_deadlock_env();
  static expected<int64_t, ErrorCode> get_stop_on_deadlock_timeout_env();
  static expected<int64_t, ErrorCode> get_max_duration_ms_env();
  static expected<double, ErrorCode> get_check_recession_period_ms_env();

  /**
   * @brief Set the scheduler for fragments object.
   *
   * Set scheduler for each fragment to use multi-thread scheduler by default because
   * UcxTransmitter/UcxReceiver doesn't work with GreedyScheduler with the following graph.
   *
   * - Fragment (fragment1)
   *   - Operator (op1)
   *     - Output port: out
   *   - Operator (op2)
   *     - Output port: out
   * - Fragment (fragment2)
   *   - Operator (op3)
   *     - Input ports
   *       - in1
   *       - in2
   *
   * With the following graph connections, due to how UcxTransmitter/UcxReceiver works,
   * UCX connections between op1 and op3 and between op2 and op3 are not established
   * (resulting in a deadlock).
   *
   * - op1.out -> op3.in1
   * - op2.out -> op3.in2
   *
   * @param target_fragments The fragments to set the scheduler.
   */
  static void set_scheduler_for_fragments(std::vector<FragmentNodeType>& target_fragments);

  std::string app_description_{};     ///< The description of the application.
  std::string app_version_{"0.0.0"};  ///< The version of the application.

  CLIParser cli_parser_;           ///< The command line parser.
  std::vector<std::string> argv_;  ///< The command line arguments after processing flags.
  std::shared_ptr<FragmentGraph> fragment_graph_;  ///< The fragment connection graph.

  std::shared_ptr<AppDriver> app_driver_;  ///< The application driver.
  std::shared_ptr<AppWorker> app_worker_;  ///< The application worker.

 private:
  /**
   * @brief Configure UCX environment variables
   */
  void set_ucx_env();
  void set_v4l2_env();
  MetadataPolicy metadata_policy_ = kDefaultMetadataPolicy;
  bool is_metadata_enabled_ = kDefaultMetadataEnabled;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_APPLICATION_HPP */
