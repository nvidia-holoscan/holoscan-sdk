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
#include <atomic>
#include <chrono>
#include <future>
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
#include "holoscan/core/distributed/app_worker/server.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/network_contexts/gxf/ucx_context.hpp"
#include "holoscan/core/schedulers/gxf/event_based_scheduler.hpp"
#include "holoscan/core/schedulers/gxf/greedy_scheduler.hpp"
#include "holoscan/core/schedulers/gxf/multithread_scheduler.hpp"
#include "holoscan/core/signal_handler.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Constants
constexpr int kDefaultFragmentServiceDriverPort = 13337;

AppWorker::AppWorker(Application* app) : app_(app) {
  if (app_) {
    options_ = &app_->cli_parser_.options();
    auto& fragment_graph = app_->fragment_graph();
    fragment_graph_ = &fragment_graph;
    target_fragments_ = get_target_fragments(fragment_graph);
  }
}

AppWorker::~AppWorker() {
  // Signal watchdog thread to cancel (if running) since we're shutting down cleanly
  if (shutdown_complete_) {
    shutdown_complete_->store(true);
  }

  // Unregister signal handlers to prevent dangling 'this' pointer issues
  // if signals are raised after this AppWorker is destroyed
  if (app_) {
    void* context = app_->executor().context();
    if (context) {
      SignalHandler::unregister_signal_handler(context, SIGINT);
      SignalHandler::unregister_signal_handler(context, SIGTERM);
    }
  }
}

CLIOptions* AppWorker::options() {
  if (app_ == nullptr) {
    return nullptr;
  }
  return options_;
}

std::vector<FragmentNodeType>& AppWorker::target_fragments() {
  return target_fragments_;
}

FragmentGraph& AppWorker::fragment_graph() {
  return *fragment_graph_;
}

distributed::AppWorkerServer* AppWorker::server(
    std::unique_ptr<distributed::AppWorkerServer>&& server) {
  worker_server_ = std::move(server);
  return worker_server_.get();
}

distributed::AppWorkerServer* AppWorker::server() {
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

  // Initialize distributed fragment service worker endpoints
  const auto& driver_address = options()->driver_address;
  auto [driver_ip, driver_port_str] = CLIOptions::parse_address(
      driver_address, "0.0.0.0", std::to_string(kDefaultFragmentServiceDriverPort));
  (void)driver_port_str;  // unused

  // Initialize fragment services
  app_->executor().initialize_fragment_services();

  handle_worker_connect(driver_ip);

  // Compose scheduled fragments
  for (auto& fragment : scheduled_fragments) {
    try {
      fragment->compose_graph();

      // Attach application services to the current fragment
      app_->attach_services_to_fragment(fragment);
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
  if (enable_async) {
    HOLOSCAN_LOG_WARN(
        "HOLOSCAN_UCX_ASYNCHRONOUS mode is deprecated as of Holoscan v3.7 and will be removed in "
        "v4.0");
  }
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
  need_notify_execution_finished_.store(true);  // Set the flag to true (atomic)
  std::vector<std::shared_future<void>> futures;
  futures.reserve(scheduled_fragments.size());

  // Store individual fragment futures for checking completion status
  fragment_futures_.clear();

  for (auto& fragment : scheduled_fragments) {
    HOLOSCAN_LOG_INFO("Launching fragment: {}", fragment->name());
    auto frag_future = fragment->executor().run_async(fragment->graph());

    // Convert to shared_future immediately so we can store copies in multiple places
    auto shared_future = frag_future.share();

    // Store in both places
    fragment_futures_[fragment->name()] = shared_future;
    futures.push_back(shared_future);
  }

  // Capture notify_flag by pointer (not reference) so we can safely check it even if there's
  // a race with the main thread setting it to false. We use an atomic load to read it safely.
  // Note: We capture the address rather than a reference to make the capture semantics clearer
  // and to allow for potential future use of atomic operations.
  auto* notify_flag_ptr = &need_notify_execution_finished_;
  auto future =
      std::async(std::launch::async, [this, futures = std::move(futures), notify_flag_ptr]() {
        for (auto& future_obj : futures) {
          future_obj.wait();
        }

        // Check the flag atomically - if it's still true, notify the server that execution
        // finished. The main thread may set this to false during forced termination to prevent
        // duplicate notifications.
        if (notify_flag_ptr->load()) {
          submit_message(
              WorkerMessage{AppWorker::WorkerMessageCode::kNotifyWorkerExecutionFinished, {}});
        }
      });

  // Register the future to the worker server
  worker_server_->fragment_executors_future(future);

  return true;
}

std::shared_ptr<distributed::AppDriverClient> AppWorker::app_driver_client() const {
  if (!worker_server_) {
    HOLOSCAN_LOG_DEBUG(
        "AppWorker::app_driver_client() called but worker_server_ is null "
        "(expected if the distributed application is running in local mode)");
    return nullptr;
  }
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
        // terminated. Use atomic store to synchronize with the async executor task.
        need_notify_execution_finished_.store(false);
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

  // Calculate maximum fragment shutdown time from schedulers
  // This is used when waiting for other workers to terminate their root fragments
  int64_t max_stop_on_deadlock_timeout = 5000L;  // Default fallback
  for (auto& fragment : scheduled_fragments_) {
    auto scheduler = fragment->scheduler();
    int64_t timeout = 5000L;

    try {
      if (auto* ebs = dynamic_cast<EventBasedScheduler*>(scheduler.get())) {
        timeout = ebs->stop_on_deadlock_timeout();
        HOLOSCAN_LOG_DEBUG(
            "Fragment '{}': EventBasedScheduler::stop_on_deadlock_timeout() returned {} ms for "
            "watchdog calculation",
            fragment->name(),
            timeout);
      } else if (auto* mts = dynamic_cast<MultiThreadScheduler*>(scheduler.get())) {
        timeout = mts->stop_on_deadlock_timeout();
        HOLOSCAN_LOG_DEBUG(
            "Fragment '{}': MultiThreadScheduler::stop_on_deadlock_timeout() returned {} ms for "
            "watchdog calculation",
            fragment->name(),
            timeout);
      } else if (auto* gs = dynamic_cast<GreedyScheduler*>(scheduler.get())) {
        timeout = gs->stop_on_deadlock_timeout();
        HOLOSCAN_LOG_DEBUG(
            "Fragment '{}': GreedyScheduler::stop_on_deadlock_timeout() returned {} ms for "
            "watchdog "
            "calculation",
            fragment->name(),
            timeout);
      }
    } catch (const std::runtime_error& e) {
      // Parameter not set on scheduler, try environment variable
      HOLOSCAN_LOG_WARN(
          "Fragment '{}': Exception when reading stop_on_deadlock_timeout for watchdog "
          "calculation: "
          "{}. Trying environment variable or using default {} ms",
          fragment->name(),
          e.what(),
          timeout);
      if (app_) {
        auto env_result = app_->get_stop_on_deadlock_timeout_env();
        if (env_result.has_value()) {
          timeout = env_result.value();
        }
        // else: keep default 5000ms
      }
    }
    max_stop_on_deadlock_timeout = std::max(max_stop_on_deadlock_timeout, timeout);
  }
  int64_t worker_termination_grace_period_ms = max_stop_on_deadlock_timeout + 500;

  // Calculate watchdog timeout for the entire shutdown process
  // Worst case: fragments are in a linear dependency chain, so we wait for each sequentially
  // Add extra buffer of 1 second for safety and overhead
  int64_t fragment_shutdown_time_ms = max_stop_on_deadlock_timeout + 250;
  int64_t total_shutdown_timeout_ms =
      scheduled_fragments_.size() * fragment_shutdown_time_ms + 1000;
  worker_shutdown_timeout_ms_ = total_shutdown_timeout_ms;

  HOLOSCAN_LOG_DEBUG(
      "Worker shutdown watchdog timeout: {} ms (based on {} fragments with max timeout {} ms)",
      worker_shutdown_timeout_ms_,
      scheduled_fragments_.size(),
      max_stop_on_deadlock_timeout);

  // Initiate UCX shutdown on all scheduled fragments before stopping execution.
  // This signals the UCX threads to exit gracefully and causes connection errors
  // during shutdown to be treated as expected (not fatal errors).
  for (auto& fragment : scheduled_fragments_) {
    auto network_context = fragment->network_context();
    if (network_context) {
      if (auto* ucx_context = dynamic_cast<UcxContext*>(network_context.get())) {
        ucx_context->initiate_shutdown();
      }
    }
  }

  while (!scheduled_fragments_.empty()) {
    // Find current root fragments scheduled on this worker
    std::vector<FragmentNodeType> current_roots;

    HOLOSCAN_LOG_DEBUG("Remaining scheduled fragments to shut down:");
    for (auto& fragment : scheduled_fragments_) {
      HOLOSCAN_LOG_DEBUG("\t{}", fragment->name());
    }

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
      // Wait for other workers that may have root fragments to terminate them
      HOLOSCAN_LOG_DEBUG(
          "No root fragments on this worker, waiting {} ms for other workers to terminate theirs",
          worker_termination_grace_period_ms);
      std::this_thread::sleep_for(std::chrono::milliseconds(worker_termination_grace_period_ms));
    }

    // Terminate current root fragments
    for (auto& root_fragment : current_roots) {
      // Check if this fragment has already finished execution
      bool fragment_already_done = false;
      auto it = fragment_futures_.find(root_fragment->name());
      if (it != fragment_futures_.end()) {
        auto status = it->second.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
          fragment_already_done = true;
          HOLOSCAN_LOG_DEBUG(
              "Fragment '{}' has already finished execution, skipping stop_execution() and wait",
              root_fragment->name());
        }
      }

      if (!fragment_already_done) {
        HOLOSCAN_LOG_INFO("Terminating fragment '{}' via stop_execution()", root_fragment->name());

        // Get the stop_on_deadlock_timeout from this fragment's scheduler
        // This is the time the scheduler waits before confirming a deadlock and exiting
        // when all operators have been stopped via stop_execution()
        int64_t stop_on_deadlock_timeout = 5000L;  // Default fallback value
        auto scheduler = root_fragment->scheduler();

        // Try to cast to known scheduler types that have stop_on_deadlock_timeout()
        try {
          if (auto* ebs = dynamic_cast<EventBasedScheduler*>(scheduler.get())) {
            stop_on_deadlock_timeout = ebs->stop_on_deadlock_timeout();
            HOLOSCAN_LOG_DEBUG(
                "Fragment '{}': EventBasedScheduler::stop_on_deadlock_timeout() returned {} ms",
                root_fragment->name(),
                stop_on_deadlock_timeout);
          } else if (auto* mts = dynamic_cast<MultiThreadScheduler*>(scheduler.get())) {
            stop_on_deadlock_timeout = mts->stop_on_deadlock_timeout();
            HOLOSCAN_LOG_DEBUG(
                "Fragment '{}': MultiThreadScheduler::stop_on_deadlock_timeout() returned {} ms",
                root_fragment->name(),
                stop_on_deadlock_timeout);
          } else if (auto* gs = dynamic_cast<GreedyScheduler*>(scheduler.get())) {
            stop_on_deadlock_timeout = gs->stop_on_deadlock_timeout();
            HOLOSCAN_LOG_DEBUG(
                "Fragment '{}': GreedyScheduler::stop_on_deadlock_timeout() returned {} ms",
                root_fragment->name(),
                stop_on_deadlock_timeout);
          }
        } catch (const std::runtime_error& e) {
          // Parameter not set on scheduler, try environment variable
          HOLOSCAN_LOG_WARN(
              "Fragment '{}': Exception when reading stop_on_deadlock_timeout from scheduler: {}. "
              "Trying environment variable or using default {} ms",
              root_fragment->name(),
              e.what(),
              stop_on_deadlock_timeout);
          if (app_) {
            auto env_result = app_->get_stop_on_deadlock_timeout_env();
            if (env_result.has_value()) {
              stop_on_deadlock_timeout = env_result.value();
            }
            // else: keep default 5000ms
          }
        }

        // Add extra margin (250ms) to account for any processing overhead
        int64_t fragment_shutdown_grace_period_ms = stop_on_deadlock_timeout + 250;

        HOLOSCAN_LOG_DEBUG(
            "Fragment '{}': using shutdown grace period of {} ms (stop_on_deadlock_timeout={} ms)",
            root_fragment->name(),
            fragment_shutdown_grace_period_ms,
            stop_on_deadlock_timeout);

        // Important: use `root_fragment->stop_execution()` instead of
        // `root_fragment->executor().interrupt()`. With interrupt, any already queued UCX messages
        // will not have a chance to be sent, resulting in several errors being logged.
        // TODO (grelee): May need to check if stop_on_deadlock() is set to false and fallback to
        // interrupt in that case.
        root_fragment->stop_execution();

        // clang-format off
      // Wait to give the fragment time to properly shut down based on its scheduler's timeout.
      // The exact wait time needed is hardware/network dependent. Without sufficient wait time,
      // any UCX messages that had not yet been sent will be lost, resulting in errors logged at
      // the GXF level:
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

        // Wait for the fragment to shut down gracefully
        std::this_thread::sleep_for(std::chrono::milliseconds(fragment_shutdown_grace_period_ms));
      }  // end if (!fragment_already_done)

      terminated_fragments.insert(root_fragment->name());

      // Remove from scheduled fragments
      scheduled_fragments_.erase(
          std::remove(scheduled_fragments_.begin(), scheduled_fragments_.end(), root_fragment),
          scheduled_fragments_.end());
    }

    // Update graph by removing terminated fragments
    for (const auto& terminated : terminated_fragments) {
      auto node = fragment_graph.find_node(terminated);
      if (node) {
        fragment_graph.remove_node(node);
      }
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

void AppWorker::handle_worker_connect(const std::string_view& driver_ip) noexcept {
  try {
    std::unordered_set<std::shared_ptr<FragmentService>> executed_services;
    std::vector<std::string> failed_services;  // Track failures

    for (const auto& [service_key, service] : app_->fragment_services_by_key()) {
      if (executed_services.find(service) != executed_services.end()) {
        continue;
      }
      executed_services.insert(service);

      try {
        auto worker_endpoint =
            std::dynamic_pointer_cast<distributed::ServiceWorkerEndpoint>(service);
        HOLOSCAN_LOG_DEBUG(
            "handle_worker_connect: checking '{}' ('{}')", service_key.id, service_key.type.name());
        if (worker_endpoint) {
          HOLOSCAN_LOG_DEBUG("handle_worker_connect: Starting worker endpoint for service '{}'",
                             service_key.id);
          worker_endpoint->worker_connect(driver_ip);
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR(
            "handle_worker_connect: Failed to connect service '{}': {}", service_key.id, e.what());
        failed_services.push_back(service_key.id);
      }
    }

    if (!failed_services.empty()) {
      throw std::runtime_error(
          fmt::format("Failed to connect services: {}", fmt::join(failed_services, ", ")));
    }
  } catch (const std::exception& e) {
    // Wrap error logging in try-catch since we're in the outermost catch
    try {
      HOLOSCAN_LOG_ERROR("handle_worker_connect: exception caught: {}", e.what());
    } catch (...) {
    }
  } catch (...) {
    // Wrap error logging in try-catch since we're in the outermost catch
    try {
      HOLOSCAN_LOG_ERROR("handle_worker_connect: unknown exception caught");
    } catch (...) {
    }
  }
}

void AppWorker::handle_worker_disconnect() noexcept {
  try {
    std::unordered_set<std::shared_ptr<FragmentService>> executed_services;
    std::vector<std::string> failed_services;  // Track failures

    for (const auto& [service_key, service] : app_->fragment_services_by_key()) {
      if (executed_services.find(service) != executed_services.end()) {
        continue;
      }
      executed_services.insert(service);

      try {
        HOLOSCAN_LOG_DEBUG("handle_worker_disconnect: checking '{}' ('{}')",
                           service_key.id,
                           service_key.type.name());
        auto worker_endpoint =
            std::dynamic_pointer_cast<distributed::ServiceWorkerEndpoint>(service);
        if (worker_endpoint) {
          HOLOSCAN_LOG_DEBUG(
              "handle_worker_disconnect: Shutting down worker endpoint for service '{}'",
              service_key.id);
          worker_endpoint->worker_disconnect();
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("handle_worker_disconnect: Failed to disconnect service '{}': {}",
                           service_key.id,
                           e.what());
        failed_services.push_back(service_key.id);
      }
    }

    if (!failed_services.empty()) {
      throw std::runtime_error(
          fmt::format("Failed to disconnect services: {}", fmt::join(failed_services, ", ")));
    }
  } catch (const std::exception& e) {
    // Wrap error logging in try-catch since we're in the outermost catch
    try {
      HOLOSCAN_LOG_ERROR("handle_worker_disconnect: exception caught: {}", e.what());
    } catch (...) {
    }
  } catch (...) {
    // Wrap error logging in try-catch since we're in the outermost catch
    try {
      HOLOSCAN_LOG_ERROR("handle_worker_disconnect: unknown exception caught");
    } catch (...) {
    }
  }
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

        // Set the flag to false to avoid notification race.
        // Use atomic store to synchronize with the async executor task.
        need_notify_execution_finished_.store(false);

        // Notify the driver that the worker is cancelled
        worker_server_->notify_worker_execution_finished(AppWorkerTerminationCode::kCancelled);
        // Stop the worker server
        worker_server_->stop();
      }

      // Shutdown data loggers on all scheduled fragments BEFORE starting watchdog countdown
      // to ensure async loggers have time to drain their queues
      for (const auto& fragment : scheduled_fragments_) {
        if (fragment && !fragment->data_loggers().empty()) {
          HOLOSCAN_LOG_INFO(
              "Fragment '{}': Shutting down {} data logger(s) before watchdog countdown...",
              fragment->name(),
              fragment->data_loggers().size());
          fragment->shutdown_data_loggers();
          HOLOSCAN_LOG_INFO("Fragment '{}': Data logger shutdown complete.", fragment->name());
        }
      }

      // Create a watchdog thread to ensure we exit even if clean shutdown hangs
      // Capture shutdown_complete_ by value (shared_ptr) so it remains valid after AppWorker
      // destruction
      auto shutdown_flag = shutdown_complete_;
      auto timeout_ms = worker_shutdown_timeout_ms_;
      std::thread([shutdown_flag, timeout_ms, signum]() {
        // Wait for a reasonable time for clean shutdown
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));

        // Check if shutdown completed successfully before forcing exit
        if (shutdown_flag && shutdown_flag->load()) {
          return;  // Clean shutdown completed, don't force exit
        }

        HOLOSCAN_LOG_ERROR("Worker clean shutdown timed out after {} ms. Forcing exit...",
                           timeout_ms);
        std::signal(signum, SIG_DFL);
        std::raise(signum);
      }).detach();
    }).detach();
  };

  // Initialize shutdown flag for watchdog cancellation
  shutdown_complete_ = std::make_shared<std::atomic<bool>>(false);

  SignalHandler::register_signal_handler(app_->executor().context(), SIGINT, sig_handler);
  SignalHandler::register_signal_handler(app_->executor().context(), SIGTERM, sig_handler);
}

}  // namespace holoscan
