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

#ifndef HOLOSCAN_CORE_SERVICES_APP_WORKER_SERVER_HPP
#define HOLOSCAN_CORE_SERVICES_APP_WORKER_SERVER_HPP

#include <condition_variable>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "holoscan/core/forward_def.hpp"

#include "holoscan/core/graph.hpp"

// Forward declaration of gRPC server
namespace grpc {
class Server;
}  // namespace grpc

namespace holoscan {

// Forward declarations
enum class AppWorkerTerminationCode;

namespace service {

constexpr int32_t kDefaultMaxConnectionRetryCount = 10;
constexpr int32_t kDefaultConnectionRetryIntervalMs = 1000;

class AppWorkerServer {
 public:
  explicit AppWorkerServer(holoscan::AppWorker* app_worker);
  virtual ~AppWorkerServer();

  void start();
  void stop();
  void wait();

  void notify();

  bool connect_to_driver(int32_t max_connection_retry_count = kDefaultMaxConnectionRetryCount,
                         int32_t connection_retry_interval_ms = kDefaultConnectionRetryIntervalMs);

  std::shared_future<void>& fragment_executors_future();

  void fragment_executors_future(std::future<void>& future);

  void notify_worker_execution_finished(holoscan::AppWorkerTerminationCode code);

 private:
  /// The thread function for the server thread.
  void run();

  std::unique_ptr<grpc::Server> server_;        ///< Pointer to the gRPC server.
  std::unique_ptr<std::thread> server_thread_;  ///< Pointer to the server thread.
  std::condition_variable cv_;                  ///< Condition variable for the server thread.
  std::mutex mutex_;                            ///< Mutex for the server thread.
  std::mutex join_mutex_;                       ///< Mutex for the join function.
  bool should_stop_ = false;                    ///< Whether the server should stop.

  holoscan::AppWorker* app_worker_ = nullptr;  ///< Pointer to the application worker.
  std::unique_ptr<AppDriverClient> driver_client_;

  std::shared_future<void> fragment_executors_future_;  ///< Future for the fragment executors.
};

}  // namespace service
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SERVICES_APP_WORKER_SERVER_HPP */
