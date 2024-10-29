/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/app_worker.hpp"

#include <stdlib.h>  // POSIX setenv

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/cli_options.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/network_contexts/gxf/ucx_context.hpp"
#include "holoscan/core/schedulers/gxf/multithread_scheduler.hpp"
#include "holoscan/core/services/app_worker/server.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include "holoscan/logger/logger.hpp"

namespace holoscan {

AppWorker::AppWorker(Application* app) : app_(app) {
  if (app_) {
    options_ = &app_->cli_parser_.options();
    auto& fragment_graph = app_->fragment_graph();
    fragment_graph_ = &fragment_graph;
    target_fragments_ = get_target_fragments(fragment_graph);
  }
}

AppWorker::~AppWorker() = default;

CLIOptions* AppWorker::options() {
  if (app_ == nullptr) { return nullptr; }
  return options_;
}

std::vector<FragmentNodeType>& AppWorker::target_fragments() {
  return target_fragments_;
}

FragmentGraph& AppWorker::fragment_graph() {
  return *fragment_graph_;
}

service::AppWorkerServer* AppWorker::server(std::unique_ptr<service::AppWorkerServer>&& server) {
  worker_server_ = std::move(server);
  return worker_server_.get();
}

service::AppWorkerServer* AppWorker::server() {
  return worker_server_.get();
}

bool AppWorker::execute_fragments(
    std::unordered_map<std::string, std::vector<std::shared_ptr<holoscan::ConnectionItem>>>&
        name_connection_list_map) {
  if (!worker_server_) {
    HOLOSCAN_LOG_ERROR("AppWorkerServer is not initialized");
    return false;
  }

  // // exclude CUDA Interprocess Communication if we are on iGPU
  // AppDriver::exclude_cuda_ipc_transport_on_igpu();
  // Disable CUDA Interprocess Communication (issue 4318442)
  AppDriver::set_ucx_to_exclude_cuda_ipc();

  // Initialize scheduled fragments
  auto& scheduled_fragments = scheduled_fragments_;
  scheduled_fragments_.clear();
  scheduled_fragments.reserve(name_connection_list_map.size());

  // Construct connection_map
  std::unordered_map<FragmentNodeType, std::vector<std::shared_ptr<holoscan::ConnectionItem>>>
      connection_map;
  connection_map.reserve(name_connection_list_map.size());

  for (auto& [fragment_name, connection_list] : name_connection_list_map) {
    auto fragment = fragment_graph_->find_node(fragment_name);
    if (!fragment) {
      HOLOSCAN_LOG_ERROR("Fragment {} not found in the fragment graph", fragment_name);
      continue;
    }
    scheduled_fragments.push_back(fragment);
    connection_map[fragment] = connection_list;
  }

  // Compose scheduled fragments
  for (auto& fragment : scheduled_fragments) {
    try {
      fragment->compose_graph();
    } catch (const std::exception& exception) {
      HOLOSCAN_LOG_ERROR(
          "Failed to compose fragment graph '{}': {}", fragment->name(), exception.what());
      // Notify the worker server that the worker execution is finished with failure
      termination_code_ = AppWorkerTerminationCode::kFailure;
      submit_message(
          WorkerMessage{AppWorker::WorkerMessageCode::kNotifyWorkerExecutionFinished, {}});
      return false;
    }
  }

  int gpu_count = 0;
  cudaError_t cuda_err = HOLOSCAN_CUDA_CALL_WARN_MSG(
      cudaGetDeviceCount(&gpu_count), "Initializing UcxContext with support for CPU data only");
  if (cuda_err == cudaSuccess) {
    HOLOSCAN_LOG_DEBUG("Detected {} GPU(s), initializing UcxContext with GPU support", gpu_count);
  }

  // Add the UCX network context
  bool enable_async = AppDriver::get_bool_env_var("HOLOSCAN_UCX_ASYNCHRONOUS", true);
  for (auto& fragment : scheduled_fragments) {
    auto network_context = fragment->make_network_context<holoscan::UcxContext>(
        "ucx_context", Arg("cpu_data_only", gpu_count == 0), Arg("enable_async", enable_async));
    fragment->network_context(network_context);
  }

  // Set scheduler for each fragment
  // Should be called before GXFExecutor::initialize_gxf_graph()
  Application::set_scheduler_for_fragments(scheduled_fragments);

  // Initialize fragment graphs
  for (auto& fragment : scheduled_fragments) {
    auto gxf_executor = dynamic_cast<gxf::GXFExecutor*>(&fragment->executor());
    if (gxf_executor == nullptr) {
      HOLOSCAN_LOG_ERROR("Cannot cast executor to GXFExecutor");
      return false;
    }
    // Set the connection items
    if (connection_map.find(fragment) != connection_map.end()) {
      gxf_executor->connection_items(connection_map[fragment]);
    }
    // Initialize the operator graph
    gxf_executor->initialize_gxf_graph(fragment->graph());
  }

  // Launch fragments
  need_notify_execution_finished_ = true;  // Set the flag to true
  std::vector<std::future<void>> futures;
  futures.reserve(scheduled_fragments.size());
  for (auto& fragment : scheduled_fragments) {
    HOLOSCAN_LOG_INFO("Launching fragment: {}", fragment->name());
    futures.push_back(fragment->executor().run_async(fragment->graph()));
  }

  auto future = std::async(
      std::launch::async,
      [this, futures = std::move(futures), &notify_flag = need_notify_execution_finished_]() {
        for (auto& future_obj : futures) { future_obj.wait(); }

        if (notify_flag) {
          submit_message(
              WorkerMessage{AppWorker::WorkerMessageCode::kNotifyWorkerExecutionFinished, {}});
        }
      });

  // Register the future to the worker server
  worker_server_->fragment_executors_future(future);

  return true;
}

void AppWorker::submit_message(WorkerMessage&& message) {
  std::lock_guard<std::mutex> lock(message_mutex_);
  message_queue_.push(std::move(message));

  // Notify the worker server to process the message queue
  if (worker_server_) {
    worker_server_->notify();
  } else {
    HOLOSCAN_LOG_ERROR("No app worker server available");
  }
}

void AppWorker::process_message_queue() {
  std::lock_guard<std::mutex> lock(message_mutex_);

  while (!message_queue_.empty()) {
    auto message = std::move(message_queue_.front());
    message_queue_.pop();

    // Process message based on message code
    auto message_code = message.code;
    switch (message_code) {
      case WorkerMessageCode::kExecuteFragments: {
        try {
          auto connection_map = std::any_cast<
              std::unordered_map<std::string,
                                 std::vector<std::shared_ptr<holoscan::ConnectionItem>>>>(
              message.data);
          execute_fragments(connection_map);
        } catch (const std::bad_any_cast& e) {
          HOLOSCAN_LOG_ERROR("Failed to cast message data to connection map: {}", e.what());
        }
      } break;
      case WorkerMessageCode::kNotifyWorkerExecutionFinished: {
        HOLOSCAN_LOG_INFO("Worker execution finished");
        if (worker_server_) {
          worker_server_->notify_worker_execution_finished(termination_code_);
          worker_server_->stop();
          // Do not call 'worker_server_->wait()' as current thread is the worker server thread
        }
      } break;
      case WorkerMessageCode::kTerminateWorker: {
        try {
          termination_code_ = std::any_cast<AppWorkerTerminationCode>(message.data);
          HOLOSCAN_LOG_INFO(
              "Terminating worker because other worker/driver is terminated (code: {})",
              static_cast<int>(termination_code_));
        } catch (const std::bad_any_cast& e) {
          HOLOSCAN_LOG_ERROR("Failed to cast message data to termination code: {}", e.what());
        }
        // Set the flag to false because the app driver already knows the worker has been
        // terminated.
        need_notify_execution_finished_ = false;
        if (app_ && app_->app_driver_) {
          auto& app_driver = app_->app_driver_;
          // Submit driver message and process the message queue to terminate the app driver server.
          app_driver->submit_message(
              AppDriver::DriverMessage{AppDriver::DriverMessageCode::kTerminateServer, {}});
          app_driver->process_message_queue();
        }
        terminate_scheduled_fragments();

        if (worker_server_) {
          worker_server_->stop();
          // Do not call 'worker_server_->wait()' as current thread is the worker server thread
        }
      } break;
      default:
        HOLOSCAN_LOG_WARN("Unknown message code: {}", static_cast<int>(message_code));
        break;
    }
  }
}

bool AppWorker::terminate_scheduled_fragments() {
  // Terminate all fragments
  for (auto& fragment : scheduled_fragments_) {
    HOLOSCAN_LOG_INFO("Terminating fragment: {}", fragment->name());
    fragment->executor().interrupt();
  }
  return true;
}

std::vector<FragmentNodeType> AppWorker::get_target_fragments(FragmentGraph& fragment_graph) {
  std::vector<FragmentNodeType> target_fragments;

  auto& app_options = *options_;
  auto& targets = app_options.worker_targets;

  // If the target is "all", we run the entire graph.
  if (targets.size() == 1 && targets[0] == "all") {
    auto all_fragments = fragment_graph.get_nodes();
    target_fragments.swap(all_fragments);
  } else {  // Otherwise, we run each fragment separately.
    // Collect fragments to run
    for (auto& target : targets) {
      auto fragment =
          fragment_graph.find_node([&target](const auto& node) { return node->name() == target; });
      if (fragment) {
        target_fragments.push_back(fragment);
      } else {
        HOLOSCAN_LOG_ERROR("Cannot find fragment: {}", target);
      }
    }
  }
  return target_fragments;
}

}  // namespace holoscan
