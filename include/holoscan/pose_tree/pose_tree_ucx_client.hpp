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

#ifndef HOLOSCAN_POSE_TREE_UCX_CLIENT_HPP
#define HOLOSCAN_POSE_TREE_UCX_CLIENT_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "holoscan/core/expected.hpp"

namespace holoscan {

class PoseTree;

/**
 * @brief Configuration for PoseTreeUCXClient
 *
 * This struct holds configuration parameters that control the behavior of the PoseTreeUCXClient,
 * including timeout and polling intervals for UCX operations.
 */
struct PoseTreeUCXClientConfig {
  int64_t request_timeout_ms{5000};  ///< Timeout in milliseconds for UCX requests
  int64_t request_poll_sleep_us{
      10};  ///< Sleep duration in microseconds between request status polls
  int64_t worker_progress_sleep_us{
      100};  ///< Sleep duration in microseconds between worker progress calls
};

/**
 * @brief UCX-based client for remote PoseTree synchronization
 *
 * This class provides a client that connects to a PoseTreeUCXServer to synchronize
 * PoseTree updates across distributed systems using UCX (Unified Communication X).
 *
 * @warning This class is NOT thread-safe for its public methods. All public methods
 * must be called from the same thread.
 *
 * ## Lifetime and Thread Management
 *
 * The PoseTreeUCXClient manages an internal worker thread that handles UCX communication.
 * The class ensures proper cleanup through its destructor:
 *
 * - The destructor automatically calls disconnect() if still connected
 * - The worker thread is always joined before destruction completes
 * - All UCX resources (endpoint, worker, context) are properly released
 * - Outstanding requests are handled (a close message is sent to the server)
 * - PoseTree callbacks are deregistered
 *
 * Users do NOT need to manually call disconnect() before destruction, though doing so
 * is safe and allows for error handling.
 *
 * The class is non-copyable and non-movable to ensure clean ownership semantics and
 * prevent resource management issues.
 *
 * @note If disconnect() fails during destruction, the error is logged but the destructor
 * continues to ensure resource cleanup.
 */
class PoseTreeUCXClient {
 public:
  /**
   * @brief Error codes used by this class.
   */
  enum class Error {
    /// kAlreadyConnected is returned when trying to connect while already connected
    kAlreadyConnected = 0,
    /// kInvalidArgument is returned when invalid parameters are provided (e.g., empty host, invalid
    /// port)
    kInvalidArgument = 1,
    /// kConnectionFailed is returned when connection to the server fails
    kConnectionFailed = 2,
    /// kNotConnected is returned when trying to perform operations that require a connection
    kNotConnected = 3,
    /// kThreadError is returned when thread operations fail
    kThreadError = 4,
    /// kShutdownError is returned when errors occur during shutdown/disconnect
    kShutdownError = 5,
    /// kInternalError is returned for unexpected internal errors
    kInternalError = 6,
  };

  /// Expected type used by this class.
  template <typename T>
  using expected = holoscan::expected<T, Error>;

  /// Unexpected type used by this class.
  using unexpected = holoscan::unexpected<Error>;

  explicit PoseTreeUCXClient(std::shared_ptr<PoseTree> pose_tree,
                             PoseTreeUCXClientConfig config = PoseTreeUCXClientConfig{});

  /**
   * @brief Destructor - ensures clean shutdown
   *
   * Automatically disconnects if still connected, joins the worker thread,
   * and releases all resources. Any errors during disconnect are logged
   * but do not throw exceptions (no-throw guarantee).
   */
  ~PoseTreeUCXClient();

  // Deleted copy/move operations ensure clean ownership
  PoseTreeUCXClient(const PoseTreeUCXClient&) = delete;
  PoseTreeUCXClient& operator=(const PoseTreeUCXClient&) = delete;
  PoseTreeUCXClient(PoseTreeUCXClient&&) = delete;             // Delete move constructor
  PoseTreeUCXClient& operator=(PoseTreeUCXClient&&) = delete;  // Delete move assignment

  /**
   * @brief Connect to a PoseTreeUCXServer
   *
   * Starts an internal worker thread that handles UCX communication with the server.
   * The thread runs until disconnect() is called or the destructor runs.
   *
   * @param host The hostname or IP address of the server
   * @param port The port number of the server
   * @param request_snapshot Whether to request a full snapshot of the pose tree upon connection
   *
   * @return Success (void) or error status
   * @note This method blocks until the connection is established or fails
   */
  expected<void> connect(std::string_view host, uint16_t port, bool request_snapshot);

  /**
   * @brief Disconnect from the server
   *
   * Signals the worker thread to stop, waits for it to finish (join), and cleans up
   * all UCX resources. This method is automatically called by the destructor if needed.
   *
   * @return Success (void) or error status
   * @note This method is idempotent - calling it when already disconnected returns success
   * @note This method blocks until the worker thread has fully stopped
   */
  expected<void> disconnect();

  bool is_running() const { return running_.load(); }

  /**
   * @brief Convert an error code to a human readable error string.
   *
   * @param error Error code to convert.
   * @return Human-readable error string.
   */
  static const char* error_to_str(Error error);

 private:
  /**
   * @brief Main worker thread function
   *
   * Handles UCX worker progress, processes server messages, and manages the connection.
   * Runs in a separate thread started by connect() and stopped by disconnect().
   */
  void run();

  struct ClientImpl;
  std::unique_ptr<ClientImpl>
      impl_;  ///< Implementation details (PIMPL pattern) containing UCX objects

  std::shared_ptr<PoseTree> pose_tree_;  ///< The local PoseTree instance to synchronize
  std::string host_;                     ///< Hostname or IP address of the server
  uint16_t port_;                        ///< Port number of the server
  bool request_snapshot_;                ///< Whether to request a full snapshot on connection
  std::atomic<bool> running_{false};     ///< Flag indicating if the client is running
  std::thread client_thread_;            ///< The worker thread handling UCX communication
  std::atomic<bool> is_external_pose_tree_update_{
      false};  ///< Flag to prevent feedback loops during updates

  std::mutex ready_mutex_;            ///< Mutex for synchronizing connection startup
  std::condition_variable ready_cv_;  ///< Condition variable for signaling connection readiness
  std::atomic<bool> ready_{false};    ///< Flag indicating if the connection is established
  std::atomic<bool> connect_failed_{false};  ///< Flag indicating if connection attempt failed
  PoseTreeUCXClientConfig config_;           ///< Configuration parameters for the client
};

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_UCX_CLIENT_HPP */
