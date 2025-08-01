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

#ifndef HOLOSCAN_POSE_TREE_UCX_SERVER_HPP
#define HOLOSCAN_POSE_TREE_UCX_SERVER_HPP

#include <ucp/api/ucp.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include "holoscan/core/expected.hpp"
#include "holoscan/pose_tree/pose_tree.hpp"

namespace holoscan {

class PoseTree;

/**
 * @brief Configuration for PoseTreeUCXServer
 *
 * This struct holds configuration parameters that control the behavior of the PoseTreeUCXServer,
 * including timing parameters for worker thread operations and shutdown behavior.
 */
struct PoseTreeUCXServerConfig {
  int64_t worker_progress_sleep_us{
      100};  ///< Sleep duration in microseconds between worker progress calls
  int64_t shutdown_timeout_ms{1000};   ///< Maximum time in milliseconds to wait for clean shutdown
  int64_t shutdown_poll_sleep_ms{10};  ///< Sleep duration in milliseconds between shutdown polls
};

/**
 * @brief UCX-based server for remote PoseTree synchronization
 *
 * This class provides a server that listens for connections from PoseTreeUCXClient instances
 * to synchronize PoseTree updates across distributed systems using UCX (Unified Communication X).
 * The server maintains its own PoseTree instance and broadcasts updates to all connected clients.
 *
 * @warning This class is NOT thread-safe for its public methods. All public methods
 * must be called from the same thread.
 *
 * ## Lifetime and Thread Management
 *
 * The PoseTreeUCXServer manages an internal worker thread that handles UCX communication.
 * The class ensures proper cleanup through its destructor:
 *
 * - The destructor automatically calls stop() if still running
 * - The worker thread is always joined before destruction completes
 * - All UCX resources (listener, worker, context) are properly released
 * - Connected clients are notified of shutdown before the server terminates
 *
 * Users do NOT need to manually call stop() before destruction, though doing so
 * is safe and allows for error handling.
 *
 * The class is non-copyable and non-movable to ensure clean ownership semantics and
 * prevent resource management issues.
 *
 * @note If stop() fails during destruction, the error is logged but the destructor
 * continues to ensure resource cleanup (no-throw guarantee).
 */
class PoseTreeUCXServer {
 public:
  /**
   * @brief Error codes used by this class.
   */
  enum class Error {
    /// kAlreadyRunning is returned when trying to start while already running
    kAlreadyRunning = 0,
    /// kInvalidArgument is returned when invalid parameters are provided
    kInvalidArgument = 1,
    /// kStartupFailed is returned when server fails to start
    kStartupFailed = 2,
    /// kNotRunning is returned when trying to perform operations that require the server to be
    /// running
    kNotRunning = 3,
    /// kShutdownTimeout is returned when shutdown takes too long
    kShutdownTimeout = 4,
    /// kInternalError is returned for unexpected internal errors
    kInternalError = 5,
  };

  /// Expected type used by this class.
  template <typename T>
  using expected = holoscan::expected<T, Error>;

  /// Unexpected type used by this class.
  using unexpected = holoscan::unexpected<Error>;

  /**
   * @brief Construct a new PoseTreeUCXServer object
   *
   * Creates a server that will synchronize the provided PoseTree with connected clients.
   * The server creates its own internal copy of the PoseTree to avoid conflicts with
   * the worker thread.
   *
   * @param pose_tree The PoseTree instance to synchronize. Must be initialized.
   * @param config Configuration parameters for the server.
   * @throw std::runtime_error if pose_tree is null or not initialized.
   */
  explicit PoseTreeUCXServer(std::shared_ptr<PoseTree> pose_tree,
                             PoseTreeUCXServerConfig config = PoseTreeUCXServerConfig{});

  /**
   * @brief Destructor - ensures clean shutdown
   *
   * Automatically stops the server if still running, joins the worker thread,
   * and releases all resources. Any errors during stop are logged
   * but do not throw exceptions (no-throw guarantee).
   */
  ~PoseTreeUCXServer();

  // Deleted copy/move operations ensure clean ownership
  PoseTreeUCXServer(const PoseTreeUCXServer&) = delete;
  PoseTreeUCXServer& operator=(const PoseTreeUCXServer&) = delete;
  PoseTreeUCXServer(PoseTreeUCXServer&&) = delete;             // Delete move constructor
  PoseTreeUCXServer& operator=(PoseTreeUCXServer&&) = delete;  // Delete move assignment

  /**
   * @brief Start the server on the specified port
   *
   * Starts an internal worker thread that listens for client connections and handles
   * UCX communication. The thread runs until stop() is called or the destructor runs.
   *
   * @param port The port number to listen on (must be non-zero)
   * @return Success (void) or error status
   * @note This method blocks until the server is fully started or fails to start
   * @note Only one server can listen on a given port at a time
   */
  expected<void> start(uint16_t port);

  /**
   * @brief Stop the server
   *
   * Signals the worker thread to stop, notifies all connected clients of shutdown,
   * waits for the thread to finish (join), and cleans up all UCX resources.
   * This method is automatically called by the destructor if needed.
   *
   * @return Success (void) or error status
   * @note This method is idempotent - calling it when already stopped returns success
   * @note This method blocks until the worker thread has fully stopped
   * @note Connected clients are given time to disconnect cleanly (controlled by
   * shutdown_timeout_ms)
   */
  expected<void> stop();

  /**
   * @brief Check if the server is running
   *
   * @return true if the server is running, false otherwise
   */
  bool is_running() const { return running_.load(); }

  /**
   * @brief Convert an error code to a human readable error string.
   *
   * @param error Error code to convert.
   * @return Human-readable error string.
   */
  static const char* error_to_str(Error error);

  friend void connection_callback(ucp_conn_request_h req, void* arg);

 private:
  /**
   * @brief Main worker thread function
   *
   * Handles UCX worker progress, processes client requests, and manages connections.
   */
  void run();

  struct ServerImpl;
  std::unique_ptr<ServerImpl> impl_;  ///< Implementation details (PIMPL pattern)

  std::shared_ptr<PoseTree> pose_tree_;  ///< The PoseTree instance being synchronized
  PoseTree::InitParameters
      pose_tree_init_params_;         ///< Initialization parameters from the original PoseTree
  uint16_t port_;                     ///< The port number the server is listening on
  std::atomic<bool> running_{false};  ///< Flag indicating if the server is running
  std::thread server_thread_;         ///< The worker thread handling UCX communication

  std::mutex ready_mutex_;            ///< Mutex for synchronizing server startup
  std::condition_variable ready_cv_;  ///< Condition variable for signaling server readiness
  std::atomic<bool> ready_{
      false};  ///< Flag indicating if the server is ready to accept connections
  PoseTreeUCXServerConfig config_;  ///< Configuration parameters for the server
};

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_UCX_SERVER_HPP */
