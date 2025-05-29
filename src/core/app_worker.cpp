/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <csignal>   // Add this line for signal handling functions

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/cli_options.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/network_contexts/gxf/ucx_context.hpp"
#include "holoscan/core/schedulers/gxf/multithread_scheduler.hpp"
#include "holoscan/core/services/app_worker/server.hpp"
#include "holoscan/core/signal_handler.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Constants
constexpr int kWorkerTerminationGracePeriodMs = 500;
constexpr int kFragmentShutdownGracePeriodMs = 250;
constexpr int kWorkerShutdownTimeoutSeconds = 10;

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
  bool enable_async = AppDriver::get_bool_env_var("HOLOSCAN_UCX_ASYNCHRONOUS", false);
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

std::shared_ptr<service::AppDriverClient> AppWorker::app_driver_client() const {
  return worker_server_->app_driver_client();
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
        terminate_scheduled_fragments();
        // Set the flag to false because the app driver already knows the worker has been
        // terminated.
        need_notify_execution_finished_ = false;
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
  std::unordered_set<std::string> terminated_fragments;
  if (!fragment_graph_) {
    HOLOSCAN_LOG_ERROR("Fragment graph is not initialized");
    return false;
  }
  auto& fragment_graph = *fragment_graph_;
  while (!scheduled_fragments_.empty()) {
    // Find current root fragments scheduled on this worker
    std::vector<FragmentNodeType> current_roots;

    HOLOSCAN_LOG_DEBUG("Remaining scheduled fragments to shut down:");
    for (auto& fragment : scheduled_fragments_) { HOLOSCAN_LOG_DEBUG("\t{}", fragment->name()); }

    for (auto& fragment : scheduled_fragments_) {
      auto node = fragment_graph.find_node(fragment->name());
      if (!node) {
        // unexpected case: schedule fragment for shutdown if it is not in the graph
        HOLOSCAN_LOG_WARN("Fragment with name '{}' not found in graph.", fragment->name());
        current_roots.push_back(fragment);
      } else if (fragment_graph.is_root(node)) {
        HOLOSCAN_LOG_DEBUG(
            "Fragment with name '{}' is currently a root fragment (for this worker).",
            fragment->name());
        current_roots.push_back(fragment);
      }
    }

    // If there were no root fragments, then just shutdown any remaining fragments
    // to avoid deadlock.
    if (current_roots.size() == 0) {
      for (auto& fragment : scheduled_fragments_) {
        HOLOSCAN_LOG_DEBUG("Scheduling non-root fragment '{}' for shutdown", fragment->name());
        HOLOSCAN_LOG_DEBUG(
            "Fragment with name '{}' is currently a root fragment (for this worker).",
            fragment->name());
        current_roots.push_back(fragment);
      }
      // wait some time for other workers that may have root fragments to terminate
      std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerTerminationGracePeriodMs));
    }

    // Terminate current root fragments
    for (auto& root_fragment : current_roots) {
      HOLOSCAN_LOG_INFO("Terminating fragment '{}' via stop_execution()", root_fragment->name());

      // Important: use `root_fragment->stop_execution()` instead of
      // `root_fragment->executor().interrupt()`. With interrupt, any already queued UCX messages
      // will not have a chance to be sent, resulting in several errors being logged.
      // TODO (grelee): May need to check if stop_on_deadlock() is set to false and fallback to
      // interrupt in that case.
      root_fragment->stop_execution();

      // clang-format off
      // Wait for 250 ms to give the fragment time to properly shut down.
      // The exact wait time needed is likely hardware/network dependent. 30 ms was sufficient on
      // an x86_64 test system where both nodes were on the same machine.
      // Without any wait, for example, any UCX messages that had not yet been sent will be lost,
      // resulting in some errors logged at the GXF level:
      //   [error] [ucx_transmitter.cpp:275] unable to send UCX message (Connection reset by remote peer)  // NOLINT
      //   [error] [ucx_transmitter.cpp:306] Failed to send entity [error]
      //   [entity_executor.cpp:624] Failed to sync outbox for entity: fragment1__replayer code: GXF_FAILURE  // NOLINT
      //
      // The UCX transmit errors propagate up to multiple additional errors logged later during shutdown  // NOLINT
      //   [error] [program.cpp:580] wait failed. Deactivating...
      //   [warning] [entity_warden.cpp:575] Component of type nvidia::gxf::EventBasedScheduler, cid 5 failed to deinitialize with code GXF_FAILURE  // NOLINT
      //   [error] [runtime.cpp:791] Could not deinitialize entity 'fragment1__event-based-scheduler' (E4): GXF_FAILURE  // NOLINT
      //   [error] [program.cpp:582] Deactivation failed.
      //   [error] [runtime.cpp:1649] Graph wait failed with error: GXF_FAILURE
      //   [warning] [gxf_executor.cpp:2429] GXF call GxfGraphWait(context) in line 2429 of file /workspace/holoscan-sdk/src/core/executors/gxf/gxf_executor.cpp failed with 'GXF_FAILURE'  // NOLINT
      //   [info] [gxf_executor.cpp:2439] [fragment1] Graph execution finished.
      //   [error] [gxf_executor.cpp:2447] [fragment1] Graph execution error: GXF_FAILURE
      // clang-format on
      std::this_thread::sleep_for(std::chrono::milliseconds(kFragmentShutdownGracePeriodMs));

      terminated_fragments.insert(root_fragment->name());

      // Remove from scheduled fragments
      scheduled_fragments_.erase(
          std::remove(scheduled_fragments_.begin(), scheduled_fragments_.end(), root_fragment),
          scheduled_fragments_.end());
    }

    // Update graph by removing terminated fragments
    for (const auto& terminated : terminated_fragments) {
      auto node = fragment_graph.find_node(terminated);
      if (node) { fragment_graph.remove_node(node); }
    }
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

void AppWorker::setup_signal_handlers() {
  auto sig_handler = [this]([[maybe_unused]] void* context, int signum) {
    // Set a static flag instead of logging in the signal handler
    static std::atomic<bool> shutdown_in_progress{false};
    bool expected = false;

    // Only one thread should handle the shutdown
    if (!shutdown_in_progress.compare_exchange_strong(expected, true)) {
      return;  // Another thread is already handling shutdown
    }

    // Now that we're outside the immediate signal handler context,
    // it's safer to log (though still not ideal)
    std::thread([this, signum]() {
      // Run cleanup in a separate thread to avoid signal handler issues
      HOLOSCAN_LOG_WARN("Received interrupt signal (Ctrl+C). Initiating worker shutdown...");

      if (worker_server_) {
        // Terminate fragments
        terminate_scheduled_fragments();

        // Set the flag to false to avoid notification race
        need_notify_execution_finished_ = false;

        // Notify the driver that the worker is cancelled
        worker_server_->notify_worker_execution_finished(AppWorkerTerminationCode::kCancelled);
        // Stop the worker server
        worker_server_->stop();
      }

      // Create a watchdog thread to ensure we exit even if clean shutdown hangs
      std::thread([signum]() {
        // Wait for a reasonable time for clean shutdown
        std::this_thread::sleep_for(std::chrono::seconds(kWorkerShutdownTimeoutSeconds));

        HOLOSCAN_LOG_ERROR("Worker clean shutdown timed out after {} seconds. Forcing exit...",
                           kWorkerShutdownTimeoutSeconds);
        std::signal(signum, SIG_DFL);
        std::raise(signum);
      }).detach();
    }).detach();
  };

  SignalHandler::register_signal_handler(app_->executor().context(), SIGINT, sig_handler);
  SignalHandler::register_signal_handler(app_->executor().context(), SIGTERM, sig_handler);
}

}  // namespace holoscan
