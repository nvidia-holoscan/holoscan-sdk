/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_POSE_TREE_POSE_TREE_MANAGER_HPP
#define HOLOSCAN_POSE_TREE_POSE_TREE_MANAGER_HPP

#include <memory>
#include <utility>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/fragment_service.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/pose_tree/pose_tree.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_client.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_server.hpp"

namespace holoscan {

class PoseTree;  // Forward declaration
class ComponentSpec;

/**
 * @brief Manage a shared PoseTree instance as a FragmentService.
 *
 * This resource creates and holds a `holoscan::PoseTree` instance, making it accessible to
 * multiple components (like operators) within the same fragment. It simplifies the management of
 * pose data by providing a centralized, configurable point of access.
 *
 * In distributed applications, PoseTreeManager also provides automatic network synchronization
 * of the PoseTree across fragments using UCX (Unified Communication X) when used as an
 * DistributedAppService.
 *
 * ## Usage as a Resource (Single Fragment)
 *
 * To use it as a resource within a single fragment, register an instance with the fragment:
 *
 * ```cpp
 * // In Application::compose()
 * auto pose_tree_manager = make_resource<PoseTreeManager>("pose_tree_manager",
 *     from_config("my_pose_tree"));
 * register_service(pose_tree_manager);
 * ```
 *
 * Then, operators can access the underlying `PoseTree` instance via the `service()` method:
 *
 * ```cpp
 * // In Operator::initialize()
 * pose_tree_ = service<holoscan::PoseTreeManager>("pose_tree_manager")->tree();
 * ```
 *
 * ## Usage as an DistributedAppService (Multi-Fragment)
 *
 * For distributed applications, register as a service to enable automatic synchronization:
 *
 * ```cpp
 * // In Application::compose()
 * auto pose_tree_manager = make_resource<PoseTreeManager>("pose_tree_manager",
 *     from_config("my_pose_tree"));
 * register_service(pose_tree_manager);
 * ```
 *
 * The framework will automatically call the appropriate driver/worker methods to establish
 * network connections between fragments.
 *
 * ## Configuration
 *
 * The parameters for the underlying `PoseTree` can be configured via the application's YAML
 * configuration file or directly when creating the resource.
 *
 * **YAML-based configuration:**
 * ```yaml
 * my_pose_tree:
 *   # PoseTree capacity parameters
 *   number_frames: 256
 *   number_edges: 4096
 *   history_length: 16384
 *   default_number_edges: 32
 *   default_history_length: 1024
 *   edges_chunk_size: 16
 *   history_chunk_size: 128
 *
 *   # Network synchronization parameters (for distributed apps)
 *   port: 13337                      # UCX server port
 *   request_timeout_ms: 5000         # Timeout for UCX requests
 *   request_poll_sleep_us: 10        # Sleep between request polls
 *   worker_progress_sleep_us: 100    # Sleep between worker progress calls
 *   server_shutdown_timeout_ms: 1000 # Timeout for server shutdown
 *   server_shutdown_poll_sleep_ms: 10 # Sleep between shutdown polls
 * ```
 *
 * **Experimental Feature**
 * The Pose Tree feature, including this manager, is experimental. The API may change in future
 * releases.
 */
class PoseTreeManager : public holoscan::Resource, public holoscan::DistributedAppService {
 public:
  /**
   * @brief Error codes used by this class.
   */
  enum class Error {
    /// kNotInitialized is returned when operations are performed before initialization
    kNotInitialized = 0,
    /// kServerError is returned when server operations fail
    kServerError = 1,
    /// kClientError is returned when client operations fail
    kClientError = 2,
    /// kAlreadyStarted is returned when trying to start a service that's already running
    kAlreadyStarted = 3,
    /// kNotStarted is returned when trying to stop a service that's not running
    kNotStarted = 4,
    /// kInternalError is returned for unexpected internal errors
    kInternalError = 5,
  };

  /// Expected type used by this class.
  template <typename T>
  using expected = holoscan::expected<T, Error>;

  /// Unexpected type used by this class.
  using unexpected = holoscan::unexpected<Error>;

  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(PoseTreeManager, holoscan::Resource)

  PoseTreeManager() = default;

  /**
   * @brief Return a shared pointer to this resource, part of the FragmentService interface.
   * @return A `std::shared_ptr<Resource>` pointing to this instance.
   */
  std::shared_ptr<Resource> resource() const override;

  /**
   * @brief Set the internal weak pointer to this resource, part of the FragmentService interface.
   * @param resource A `std::shared_ptr<Resource>` that must point to this instance.
   */
  void resource(const std::shared_ptr<Resource>& resource) override;

  /**
   * @brief Initialize the resource and creates the underlying `PoseTree` instance.
   *
   * This method is called by the framework after the resource is created and its parameters
   * have been set. It allocates and initializes the `PoseTree` with the configured capacity
   * parameters.
   */
  void initialize() override;

  /**
   * @brief Define the parameters for configuring the `PoseTree` instance.
   *
   * This method registers the following parameters:
   * - `port`: Port number for UCX server (default: 13337)
   * - `number_frames`: Maximum number of coordinate frames.
   * - `number_edges`: Maximum number of edges (direct transformations) between frames.
   * - `history_length`: Total capacity for storing historical pose data across all edges.
   * - `default_number_edges`: Default number of edges allocated per new frame.
   * - `default_history_length`: Default history capacity allocated per new edge.
   * - `edges_chunk_size`: Allocation chunk size for a frame's edge list.
   * - `history_chunk_size`: Allocation chunk size for an edge's history buffer.
   * - `request_timeout_ms`: Timeout for UCX requests in milliseconds.
   * - `request_poll_sleep_us`: Sleep duration between request polls in microseconds.
   * - `worker_progress_sleep_us`: Sleep duration between worker progress calls in microseconds.
   * - `server_shutdown_timeout_ms`: Timeout for server shutdown in milliseconds.
   * - `server_shutdown_poll_sleep_ms`: Sleep duration between shutdown polls in milliseconds.
   *
   * @param spec The component specification to which the parameters are added.
   */
  void setup(holoscan::ComponentSpec& spec) override;

  /**
   * @brief Get a shared pointer to the managed `PoseTree` instance.
   *
   * This is the primary method for accessing the pose tree from other components.
   *
   * @return A `std::shared_ptr<PoseTree>` to the underlying pose tree.
   */
  std::shared_ptr<PoseTree> tree();

  /**
   * @brief Get a shared pointer to the managed `PoseTree` instance from a const context.
   *
   * @return A `std::shared_ptr<PoseTree>` to the underlying pose tree.
   */
  std::shared_ptr<PoseTree> tree() const;

  // AppService interface methods

  /**
   * @brief Start the UCX server on the driver fragment.
   *
   * Called by the framework on the driver fragment to start the PoseTreeUCXServer.
   * The server will listen on the configured port for client connections.
   *
   * @param driver_ip The IP address of the driver (unused in current implementation)
   * @note Errors are logged but not thrown to satisfy the DistributedAppService interface
   */
  void driver_start(std::string_view driver_ip) override;

  /**
   * @brief Shutdown the UCX server on the driver fragment.
   *
   * Called by the framework to stop the PoseTreeUCXServer and clean up resources.
   *
   * @note Errors are logged but not thrown to satisfy the DistributedAppService interface
   */
  void driver_shutdown() override;

  /**
   * @brief Connect a worker fragment to the driver's UCX server.
   *
   * Called by the framework on worker fragments to establish a connection
   * to the driver's PoseTreeUCXServer using a PoseTreeUCXClient.
   *
   * @param driver_ip The IP address of the driver fragment
   * @note Errors are logged but not thrown to satisfy the DistributedAppService interface
   */
  void worker_connect(std::string_view driver_ip) override;

  /**
   * @brief Disconnect a worker fragment from the driver's UCX server.
   *
   * Called by the framework to disconnect the PoseTreeUCXClient and clean up resources.
   *
   * @note Errors are logged but not thrown to satisfy the DistributedAppService interface
   */
  void worker_disconnect() override;

  /**
   * @brief Convert an error code to a human readable error string.
   *
   * @param error Error code to convert.
   * @return Human-readable error string.
   */
  static const char* error_to_str(Error error);

 protected:
  // Internal methods that use expected for error handling

  /**
   * @brief Internal implementation of driver_start with error handling
   *
   * @param driver_ip The IP address of the driver
   * @return Success (void) or error status
   */
  expected<void> driver_start_impl(std::string_view driver_ip);

  /**
   * @brief Internal implementation of driver_shutdown with error handling
   *
   * @return Success (void) or error status
   */
  expected<void> driver_shutdown_impl();

  /**
   * @brief Internal implementation of worker_connect with error handling
   *
   * @param driver_ip The IP address of the driver fragment
   * @return Success (void) or error status
   */
  expected<void> worker_connect_impl(std::string_view driver_ip);

  /**
   * @brief Internal implementation of worker_disconnect with error handling
   *
   * @return Success (void) or error status
   */
  expected<void> worker_disconnect_impl();

 private:
  // Parameters for UCX connections
  Parameter<int32_t> port_;  ///< Port number for UCX server

  // Parameters for PoseTree::init
  Parameter<int32_t> number_frames_;           ///< Maximum number of coordinate frames
  Parameter<int32_t> number_edges_;            ///< Maximum number of edges between frames
  Parameter<int32_t> history_length_;          ///< Total capacity for historical pose data
  Parameter<int32_t> default_number_edges_;    ///< Default edges allocated per new frame
  Parameter<int32_t> default_history_length_;  ///< Default history capacity per new edge
  Parameter<int32_t> edges_chunk_size_;        ///< Allocation chunk size for frame's edge list
  Parameter<int32_t> history_chunk_size_;      ///< Allocation chunk size for edge's history buffer

  // Parameters for UCX timings
  Parameter<int64_t> request_timeout_ms_;  ///< Timeout for UCX requests in milliseconds
  Parameter<int64_t>
      request_poll_sleep_us_;  ///< Sleep duration between request polls in microseconds
  Parameter<int64_t>
      worker_progress_sleep_us_;  ///< Sleep duration between worker progress calls in microseconds
  Parameter<int64_t> server_shutdown_timeout_ms_;  ///< Timeout for server shutdown in milliseconds
  Parameter<int64_t>
      server_shutdown_poll_sleep_ms_;  ///< Sleep duration between shutdown polls in milliseconds

  std::shared_ptr<PoseTree> pose_tree_instance_;  ///< The managed PoseTree instance
  std::unique_ptr<PoseTreeUCXServer> server_;     ///< UCX server (only on driver fragment)
  std::unique_ptr<PoseTreeUCXClient> client_;     ///< UCX client (only on worker fragments)

  std::weak_ptr<Resource> resource_;  ///< Weak reference to the managed resource (self)
};

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_POSE_TREE_MANAGER_HPP */
