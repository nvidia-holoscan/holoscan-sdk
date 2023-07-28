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

#ifndef HOLOSCAN_CORE_APP_DRIVER_HPP
#define HOLOSCAN_CORE_APP_DRIVER_HPP

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>  // for std::pair
#include <vector>

#include "holoscan/core/common.hpp"

#include "holoscan/core/application.hpp"
#include "holoscan/core/fragment_scheduler.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/io_spec.hpp"

namespace holoscan {

// Forward declarations
struct AppWorkerTerminationStatus;
enum class AppWorkerTerminationCode;

struct ConnectionItem {
  ConnectionItem(std::string name, IOSpec::IOType io_type, IOSpec::ConnectorType connector_type,
                 ArgList args)
      : name(std::move(name)),
        io_type(io_type),
        connector_type(connector_type),
        args(std::move(args)) {}

  std::string name;
  IOSpec::IOType io_type;
  IOSpec::ConnectorType connector_type;
  ArgList args;
};

class AppDriver {
 public:
  explicit AppDriver(Application* app);
  virtual ~AppDriver();

  /**
   * @brief Retrieves a boolean value from an environment variable.
   *
   * This function fetches the value of a named environment variable and converts it to a boolean.
   * The conversion is case-insensitive and accepts "true", "1", or "on" as true, any other values
   * are considered false. If the environment variable is not set or its value is not recognized as
   * 'true', the function returns a default value.
   *
   * This function uses std::getenv() to access environment variables. The environment variable
   * to look up is specified by the 'name' parameter. The value of the environment variable is
   * converted to a lower-case string, and compared to the known 'true' strings.
   *
   * The function does not throw an exception if the environment variable is not found or if
   * the value does not match any of the expected 'true' strings.
   *
   * @param name The name of the environment variable to look up.
   * @param default_value The value to return if the environment variable is not set or its
   *                      value does not match any of the known 'true' strings. The default is
   *                      'false'.
   *
   * @return true if the environment variable is set and its value is recognized as 'true',
   *         and false otherwise. If the environment variable is not set, the function returns
   *         'default_value'.
   */
  static bool get_bool_env_var(const char* name, bool default_value = false);

  /**
   * @brief Retrieves an integer value from an environment variable.
   *
   * @param name The name of the environment variable to look up.
   * @param default_value The value to return if the environment variable is not set or its
   *                     value is not a valid integer. The default is 0.
   * @return The value of the environment variable, converted to an integer. If the environment
   *        variable is not set or its value is not a valid integer, the function returns
   *       'default_value'.
   */
  static int64_t get_int_env_var(const char* name, int64_t default_value = 0);

  /**
   * Parses a string representing a memory size and returns its value in bytes.
   *
   * This method takes a string in the format of "XGi" or "XMi", where X is a numerical value,
   * and Mi/Gi represents Mebibytes/Gibibytes. The method converts this string into an
   * equivalent memory size in bytes.
   *
   * The string is case-insensitive, meaning that "1gi" is considered equivalent to "1Gi".
   *
   * The string is expected to represent a positive integer followed by either "M" or "G",
   * signifying mebibytes or gibibytes, respectively. If the string does not follow this
   * format, the behavior is undefined.
   *
   * @param size_str A string representing a memory size in mebibytes or gibibytes.
   *                 For example, "1Gi" represents one gibibyte and "500Mi" represents 500
   * mebibytes.
   *
   * @return The memory size represented by size_str, converted into bytes. For example, if
   *         size_str is "1Gi", the return value will be 1,073,741,824 (1024 * 1024 * 1024).
   *         If size_str is "500Mi", the return value will be 524,288,000 (500 * 1024 * 1024).
   */
  static uint64_t parse_memory_size(const std::string& size_str);

  /**
   * Set UCX_TLS to disable cuda_ipc transport
   *
   * Will check the UCX_TLS environment variable. If it is not already set, it sets
   * UCX_TLS=^cuda_ipc to exclude this transport method. If it is already set and cuda_ipc is not
   * excluded, a warning will be logged.
   *
   */
  static void set_ucx_to_exclude_cuda_ipc();

  /**
   * Disable CUDA interprocess communication (IPC) if the fragment is running on an iGPU.
   *
   * Tegra devices do not support CUDA IPC.
   *
   * Calls set_ucx_tls_to_exclude_cuda_ipc if we are running on iGPU.
   *
   */
  static void exclude_cuda_ipc_transport_on_igpu();

  enum class AppStatus {
    kNotInitialized,
    kNotStarted,
    kRunning,
    kFinished,
    kError,
  };

  enum class DriverMessageCode {
    kCheckFragmentSchedule,
    kWorkerExecutionFinished,
    kTerminateServer,
  };

  struct DriverMessage {
    DriverMessageCode code;
    std::any data;
  };

  void run();

  std::future<void> run_async();

  CLIOptions* options();

  /// Note that the application status is not updated when the application is running locally.
  AppStatus status();

  FragmentScheduler* fragment_scheduler();

  void submit_message(DriverMessage&& message);

  void process_message_queue();

 private:
  friend class service::AppDriverServer;  ///< Allow AppDriverServer to access private members.

  /// Correct the port names of the given fragment graph edge.
  bool update_port_names(holoscan::FragmentNodeType src_frag,
                         holoscan::FragmentNodeType target_frag,
                         std::shared_ptr<holoscan::FragmentEdgeDataElementType>& port_map);

  /// Collect fragment connections.
  bool collect_connections(holoscan::FragmentGraph& fragment_graph);

  /// Correct connection map.
  /// `connection_map_` is initialized with the default IP (0.0.0.0) and port (zero-based index).
  /// This function corrects the connection map by replacing the default IP and port with the
  /// real IP and port (using index_to_ip_map_ and index_to_port_map_).
  void correct_connection_map();

  /// Connect target fragments with UCX connector.
  void connect_fragments(holoscan::FragmentGraph& fragment_graph,
                         std::vector<holoscan::FragmentNodeType>& target_fragments);

  /// Check the configuration of the application.
  /// This function checks if the application is running locally or remotely and sets the
  /// corresponding flags.
  /// This also calls the application's compose() function to compose the application.
  bool check_configuration();

  /// Get the system resource requirement of the fragment.
  void collect_resource_requirements(const Config& app_config,
                                     holoscan::FragmentGraph& fragment_graph);

  /// Parse the system resource requirement from the given YAML node.
  SystemResourceRequirement parse_resource_requirement(const YAML::Node& node);

  /// Parse the system resource requirement from the given fragment name, the YAML node and the base
  /// requirement.
  SystemResourceRequirement parse_resource_requirement(
      const std::string& fragment_name, const YAML::Node& node,
      const SystemResourceRequirement& base_requirement);

  /// Return the system resource requirement from the base requirement with the fragment name
  /// replaced.
  SystemResourceRequirement parse_resource_requirement(
      const std::string& fragment_name, const SystemResourceRequirement& base_requirement);

  /// Check the fragment schedule to ensure that all fragments are scheduled to run.
  void check_fragment_schedule(const std::string& worker_address = "");

  /// Check if the all workers have finished execution.
  void check_worker_execution(const AppWorkerTerminationStatus& termination_status);

  /// Terminate all worker and close all worker connections.
  void terminate_all_workers(AppWorkerTerminationCode error_code);

  /// Run the application in driver mode.
  /// Even if the application is running locally, we launch the driver to provide the health check
  /// service if `need_health_check_` is true.
  void launch_app_driver();

  /// Run the application in worker mode.
  void launch_app_worker();

  /// Launch fragments asynchronously.
  std::future<void> launch_fragments_async(std::vector<FragmentNodeType>& target_fragments);

  Application* app_ = nullptr;      ///< The application to run.
  CLIOptions* options_ = nullptr;   ///< The command line options.
  bool need_health_check_ = false;  ///< Whether to check the health of the application.
  bool need_driver_ = false;        ///< Whether to run the application in driver mode.
  bool need_worker_ = false;        ///< Whether to run the application in worker mode.
  bool is_local_ = false;  ///< Whether the application is running locally without a server.
  AppStatus app_status_ = AppStatus::kNotInitialized;  ///< The status of the application.

  /// The map that associates a fragment with a list of connection items.
  std::unordered_map<std::shared_ptr<Fragment>, std::vector<std::shared_ptr<ConnectionItem>>>
      connection_map_;

  /// The map that associates a fragment name with a list of pairs. Each pair contains an index that
  /// represents a port and a real port number that is associated with that index.
  std::unordered_map<std::string, std::vector<std::pair<int32_t, int32_t>>> receiver_port_map_;

  /// Maps port indices to their IP addresses (initially set to the fragment name).
  std::unordered_map<int32_t, std::string> index_to_ip_map_;

  /// Maps port indices to real port numbers (initially set to -1).
  std::unordered_map<int32_t, int32_t> index_to_port_map_;

  std::unique_ptr<service::AppDriverServer> driver_server_;

  std::unique_ptr<FragmentScheduler> fragment_scheduler_;
  std::mutex message_mutex_;                 ///< Mutex for the message queue.
  std::queue<DriverMessage> message_queue_;  ///< Queue of messages to be processed.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_APP_DRIVER_HPP */
