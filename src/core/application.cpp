/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/application.hpp"

#include <sys/resource.h>  // for getrlimit (stack size check)
#include <ucs/config/global_opts.h>
#include <ucs/type/status.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>  // for std::call_once
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <magic_enum.hpp>

#include "./distributed/app_driver/client.hpp"
#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/app_worker.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/dataflow_tracker.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/flow_graphs/flow_graph_impl.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/resources/gxf/synthetic_clock.hpp"
#include "holoscan/core/schedulers/gxf/event_based_scheduler.hpp"
#include "holoscan/core/schedulers/gxf/greedy_scheduler.hpp"
#include "holoscan/core/schedulers/gxf/multithread_scheduler.hpp"

namespace CLI {
////////////////////////////////////////////////////////////////////////////////////////////////////
// The following snippet from CLI11 that is under the BSD-3-Clause license:
//     https://github.com/CLIUtils/CLI11/blob/89601ee/include/CLI/impl/Argv_inl.hpp
// We have modified it to use Linux specific functions to get the command line arguments from
// /proc/self/cmdline. Please see https://github.com/CLIUtils/CLI11/pull/804.
// Once the CLI11 library releases a new version, we can remove this snippet.
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
static const std::vector<const char*>& args() {
  // This function uses initialization via lambdas extensively to take advantage of the thread
  // safety of static variable initialization [stmt.dcl.3]
  static const std::vector<const char*> static_args = [] {
    static const std::vector<char> static_cmdline = [] {
      // On posix, retrieve arguments from /proc/self/cmdline, separated by null terminators.
      std::vector<char> cmdline;

      auto deleter = [](FILE* f) { (void)std::fclose(f); };
      std::unique_ptr<FILE, decltype(deleter)> fp_unique(std::fopen("/proc/self/cmdline", "re"),
                                                         deleter);
      FILE* fp = fp_unique.get();
      if (!fp) {
        throw std::runtime_error(
            "could not open /proc/self/cmdline for reading");  // LCOV_EXCL_LINE
      }

      size_t size = 0;
      while (std::feof(fp) == 0) {
        cmdline.resize(size + 128);
        size += std::fread(cmdline.data() + size, 1, 128, fp);

        if (std::ferror(fp) != 0) {
          throw std::runtime_error("error during reading /proc/self/cmdline");  // LCOV_EXCL_LINE
        }
      }
      cmdline.resize(size);

      return cmdline;
    }();

    std::size_t argc =
        static_cast<std::size_t>(std::count(static_cmdline.begin(), static_cmdline.end(), '\0'));
    std::vector<const char*> static_args_result;
    static_args_result.reserve(argc);

    for (auto it = static_cmdline.begin(); it != static_cmdline.end();
         it = std::find(it, static_cmdline.end(), '\0') + 1) {
      static_args_result.push_back(static_cmdline.data() + (it - static_cmdline.begin()));
    }

    return static_args_result;
  }();

  return static_args;
}
}  // namespace detail

inline const char* const* argv() {
  return detail::args().data();
}
inline int argc() {
  return static_cast<int>(detail::args().size());
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace CLI

namespace holoscan {

namespace {
// Default values for scheduler configuration
constexpr bool kDefaultStopOnDeadlock = true;
constexpr int64_t kDefaultStopOnDeadlockTimeout = 1000L;
constexpr int64_t kDefaultUcxNetworkConnectionTimeout = 5000L;
constexpr double kDefaultCheckRecessionPeriodMs = 0.0;

// ============================================================================
// Helper types and functions for set_scheduler_for_fragments
// ============================================================================

/// @brief Holds environment variable overrides for scheduler arguments.
struct SchedulerEnvOverrides {
  expected<bool, ErrorCode> stop_on_deadlock;
  expected<int64_t, ErrorCode> stop_on_deadlock_timeout;
  expected<int64_t, ErrorCode> ucx_network_connection_timeout;
  expected<int64_t, ErrorCode> max_duration_ms;
  expected<double, ErrorCode> check_recession_period_ms;
};

/**
 * @brief Detect the scheduler type from a scheduler pointer.
 * @param scheduler The scheduler to detect the type of.
 * @param fragment_name The name of the fragment (for logging).
 * @return The detected SchedulerType, or kDefault if not recognized. A warning is logged
 *         when an unrecognized scheduler type is encountered.
 */
SchedulerType detect_scheduler_type(const std::shared_ptr<Scheduler>& scheduler,
                                    const std::string& fragment_name) {
  if (!scheduler) {
    return SchedulerType::kDefault;
  }

  if (std::dynamic_pointer_cast<holoscan::EventBasedScheduler>(scheduler)) {
    return SchedulerType::kEventBased;
  } else if (std::dynamic_pointer_cast<holoscan::MultiThreadScheduler>(scheduler)) {
    return SchedulerType::kMultiThread;
  } else if (std::dynamic_pointer_cast<holoscan::GreedyScheduler>(scheduler)) {
    return SchedulerType::kGreedy;
  } else {
    // Scheduler type not recognized (custom scheduler or unknown type)
    HOLOSCAN_LOG_WARN(
        "Fragment '{}': Unrecognized scheduler type '{}'. The user-defined scheduler will be "
        "kept, but default arguments will not be applied.",
        fragment_name,
        scheduler->name());
    return SchedulerType::kDefault;
  }
}

/**
 * @brief Resolve which scheduler type to use for a fragment.
 * @param user_set_scheduler Whether the user explicitly set a scheduler on the fragment.
 * @param use_app_scheduler_type Whether to use the app scheduler's type.
 * @param user_scheduler_type The detected user scheduler type.
 * @param env_scheduler_type The scheduler type from environment variable (kDefault if not set).
 * @return The resolved SchedulerType to use.
 */
SchedulerType resolve_scheduler_type(bool user_set_scheduler, bool use_app_scheduler_type,
                                     SchedulerType user_scheduler_type,
                                     SchedulerType env_scheduler_type) {
  // If environment variable specifies a scheduler type, use it
  if (env_scheduler_type != SchedulerType::kDefault) {
    return env_scheduler_type;
  }

  // If user explicitly set a recognized scheduler type, use it
  if ((user_set_scheduler || use_app_scheduler_type) &&
      user_scheduler_type != SchedulerType::kDefault) {
    return user_scheduler_type;
  }

  // No recognized scheduler type detected. If we need to create a new scheduler,
  // we'll use EventBasedScheduler as it works better with UCX connections for
  // distributed apps. However, if the user explicitly set an unrecognized scheduler
  // on the fragment, we'll respect their choice and keep it (see create_scheduler_for_fragment).
  return SchedulerType::kEventBased;
}

/**
 * @brief Create a scheduler for a fragment if needed, or return the existing one.
 *
 * Creates a new scheduler if:
 * - Using app scheduler type (can't share instances between fragments)
 * - User didn't set a scheduler on the fragment
 * - Environment variable forces a different scheduler type
 *
 * @param fragment The fragment to create the scheduler for.
 * @param existing_scheduler The current scheduler (may be nullptr).
 * @param resolved_scheduler_type The resolved target scheduler type.
 * @param user_set_scheduler Whether the user explicitly set a scheduler on the fragment.
 * @param use_app_scheduler_type Whether to use the app scheduler's type.
 * @param user_scheduler_type The detected user scheduler type.
 * @param env_scheduler_type The scheduler type from environment variable (kDefault if not set).
 * @return The new scheduler if created, or the existing scheduler if kept.
 */
std::shared_ptr<Scheduler> create_scheduler_for_fragment(
    const std::shared_ptr<Fragment>& fragment, const std::shared_ptr<Scheduler>& existing_scheduler,
    SchedulerType resolved_scheduler_type, bool user_set_scheduler, bool use_app_scheduler_type,
    SchedulerType user_scheduler_type, SchedulerType env_scheduler_type) {
  // Determine if we need to create a new scheduler:
  // 1. If using app scheduler type, we always create a new scheduler instance
  //    (can't share scheduler instances between fragments due to GXF entity naming)
  // 2. User didn't set one on the fragment, OR
  // 3. Environment variable forces a different scheduler type than what user set
  bool should_create = use_app_scheduler_type || !user_set_scheduler ||
                       (env_scheduler_type != SchedulerType::kDefault &&
                        user_scheduler_type != resolved_scheduler_type);

  if (!should_create) {
    return existing_scheduler;
  }

  const auto& frag_name = fragment->name();

  switch (resolved_scheduler_type) {
    case SchedulerType::kDefault:
      // This case shouldn't occur given the resolve logic, but handle gracefully
      // by creating an EventBasedScheduler
      [[fallthrough]];
    case SchedulerType::kEventBased:
      return fragment->make_scheduler<holoscan::EventBasedScheduler>(
          fmt::format("{}-event-based-scheduler", frag_name));
    case SchedulerType::kGreedy:
      return fragment->make_scheduler<holoscan::GreedyScheduler>(
          fmt::format("{}-greedy-scheduler", frag_name));
    case SchedulerType::kMultiThread:
      return fragment->make_scheduler<holoscan::MultiThreadScheduler>(
          fmt::format("{}-multithread-scheduler", frag_name));
  }

  // Should never reach here, but satisfy compiler
  return fragment->make_scheduler<holoscan::EventBasedScheduler>(
      fmt::format("{}-event-based-scheduler", frag_name));
}

/**
 * @brief Add default arguments to a scheduler based on its type.
 * @param scheduler The scheduler to add arguments to.
 * @param scheduler_type The type of the scheduler.
 * @param fragment The fragment (used to determine worker_thread_number).
 */
void add_default_scheduler_args(const std::shared_ptr<Scheduler>& scheduler,
                                SchedulerType scheduler_type,
                                const std::shared_ptr<Fragment>& fragment) {
  // Common args for all scheduler types
  scheduler->add_arg(holoscan::Arg("stop_on_deadlock", kDefaultStopOnDeadlock));
  scheduler->add_arg(holoscan::Arg("stop_on_deadlock_timeout", kDefaultStopOnDeadlockTimeout));
  scheduler->add_arg(
      holoscan::Arg("network_connection_timeout", kDefaultUcxNetworkConnectionTimeout));

  // Scheduler-type-specific args
  switch (scheduler_type) {
    case SchedulerType::kEventBased: {
      // hardware_concurrency() can return 0 if not computable; default to 1 in that case
      unsigned int num_processors = std::max(1u, std::thread::hardware_concurrency());
      int64_t worker_thread_number =
          std::min(fragment->graph().get_nodes().size(), static_cast<size_t>(num_processors));
      scheduler->add_arg(holoscan::Arg("worker_thread_number", worker_thread_number));
    } break;
    case SchedulerType::kGreedy:
      scheduler->add_arg(
          holoscan::Arg("check_recession_period_ms", kDefaultCheckRecessionPeriodMs));
      break;
    case SchedulerType::kMultiThread: {
      // hardware_concurrency() can return 0 if not computable; default to 1 in that case
      unsigned int num_processors = std::max(1u, std::thread::hardware_concurrency());
      int64_t worker_thread_number =
          std::min(fragment->graph().get_nodes().size(), static_cast<size_t>(num_processors));
      scheduler->add_arg(
          holoscan::Arg("check_recession_period_ms", kDefaultCheckRecessionPeriodMs));
      scheduler->add_arg(holoscan::Arg("worker_thread_number", worker_thread_number));
    } break;
    default:
      break;
  }
}

/**
 * @brief Apply environment variable overrides to a scheduler.
 * @param scheduler The scheduler to apply overrides to.
 * @param env_overrides The environment variable overrides to apply.
 */
void apply_scheduler_env_overrides(const std::shared_ptr<Scheduler>& scheduler,
                                   const SchedulerEnvOverrides& env_overrides) {
  // Override arguments using environment variables (always apply if env vars are set).
  // This works because calling `add_arg()` more than once for the same argument will overwrite
  // the previous value.
  if (env_overrides.stop_on_deadlock) {
    scheduler->add_arg(holoscan::Arg("stop_on_deadlock", env_overrides.stop_on_deadlock.value()));
  }
  if (env_overrides.stop_on_deadlock_timeout) {
    scheduler->add_arg(
        holoscan::Arg("stop_on_deadlock_timeout", env_overrides.stop_on_deadlock_timeout.value()));
  }
  if (env_overrides.ucx_network_connection_timeout) {
    scheduler->add_arg(holoscan::Arg("network_connection_timeout",
                                     env_overrides.ucx_network_connection_timeout.value()));
  }
  if (env_overrides.max_duration_ms) {
    scheduler->add_arg(holoscan::Arg("max_duration_ms", env_overrides.max_duration_ms.value()));
  }
  if (env_overrides.check_recession_period_ms) {
    scheduler->add_arg(holoscan::Arg("check_recession_period_ms",
                                     env_overrides.check_recession_period_ms.value()));
  }
}

/**
 * @brief Clone a scheduler clock resource for a fragment.
 *
 * Creates a new clock resource of the same concrete type, copies its arguments,
 * and binds it to the target fragment. Returns nullptr if the clock is unsupported
 * or cannot be cloned.
 */
std::shared_ptr<Resource> clone_scheduler_clock_for_fragment(
    const std::shared_ptr<Fragment>& fragment, const std::shared_ptr<Resource>& clock_resource,
    const std::string& scheduler_name) {
  if (!fragment || !clock_resource) {
    return nullptr;
  }

  auto gxf_clock = std::dynamic_pointer_cast<holoscan::gxf::Clock>(clock_resource);
  if (!gxf_clock) {
    return nullptr;
  }

  const std::string cloned_name = fmt::format("{}__{}", scheduler_name, clock_resource->name());
  std::shared_ptr<Resource> cloned_clock;

  if (std::dynamic_pointer_cast<RealtimeClock>(clock_resource)) {
    cloned_clock = fragment->make_resource<RealtimeClock>(cloned_name);
  } else if (std::dynamic_pointer_cast<ManualClock>(clock_resource)) {
    cloned_clock = fragment->make_resource<ManualClock>(cloned_name);
  } else if (std::dynamic_pointer_cast<SyntheticClock>(clock_resource)) {
    cloned_clock = fragment->make_resource<SyntheticClock>(cloned_name);
  } else {
    HOLOSCAN_LOG_WARN(
        "Fragment '{}': Unsupported clock type '{}' for scheduler '{}'; using default clock",
        fragment->name(),
        gxf_clock->gxf_typename(),
        scheduler_name);
    return nullptr;
  }

  for (const auto& arg : clock_resource->args()) {
    cloned_clock->add_arg(arg);
  }

  return cloned_clock;
}

}  // namespace

Application::Application(const std::vector<std::string>& argv) : Fragment(), argv_(argv) {
  // Set the log level from the environment variable if it exists.
  // Or, set the default log level to INFO if it hasn't been set by the user.
  if (!Logger::log_level_set_by_user) {
    holoscan::set_log_level(LogLevel::INFO);
  } else {
    // Allow log level to be reset from the environment variable if overridden.
    holoscan::set_log_level(holoscan::log_level());
  }
  // Set the log format from the environment variable if it exists.
  // Or, set the default log format depending on the log level if it hasn't been set by the user.
  holoscan::set_log_pattern();

  // Set the application pointer to this
  app_ = this;
  process_arguments();
}

std::string& Application::description() {
  return app_description_;
}

Application& Application::description(const std::string& desc) & {
  app_description_ = desc;
  return *this;
}

Application&& Application::description(const std::string& desc) && {
  app_description_ = desc;
  return std::move(*this);
}

std::string& Application::version() {
  return app_version_;
}

Application& Application::version(const std::string& version) & {
  app_version_ = version;
  return *this;
}

Application&& Application::version(const std::string& version) && {
  app_version_ = version;
  return std::move(*this);
}

std::vector<std::string>& Application::argv() {
  return argv_;
}

CLIOptions& Application::options() {
  return cli_parser_.options();
}

FragmentFlowGraph& Application::fragment_graph() {
  if (!fragment_graph_) {
    fragment_graph_ = make_graph<FragmentFlowGraphImpl>();
  }
  return *fragment_graph_;
}

void Application::add_fragment(const std::shared_ptr<Fragment>& frag) {
  // check if any existing fragment in the fragment_graph is GPU-resident
  if (frag->is_gpu_resident() && is_any_fragment_gpu_resident()) {
    auto err_msg = fmt::format(
        "Fragment ({}) is a GPU-resident fragment but the application already has a GPU-resident "
        "fragment. An application cannot have multiple GPU-resident fragments.",
        frag->name());
    throw std::runtime_error(err_msg);
  }
  fragment_graph().add_node(frag);
}

void Application::add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                           const std::shared_ptr<Fragment>& downstream_frag,
                           const std::set<std::pair<std::string, std::string>>& port_pairs) {
  // If port_pairs is empty, fail fast.
  if (port_pairs.empty()) {
    auto err_msg = std::string("Unable to add fragment flow with empty port_pairs");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // if any of the fragments is GPU-resident, then this function should throw an error
  if (upstream_frag->is_gpu_resident() || downstream_frag->is_gpu_resident()) {
    auto err_msg = fmt::format(
        "Connections between GPU-resident fragments are not allowed. "
        "Upstream fragment: {}, Downstream fragment: {}",
        upstream_frag->name(),
        downstream_frag->name());
    throw std::runtime_error(err_msg);
  }

  auto port_map = std::make_shared<FragmentFlowGraph::EdgeDataElementType>();

  // Convert the port name pairs to port map
  // (set<pair<string, string>> -> map<string, set<string>>)
  for (const auto& [key, value] : port_pairs) {
    if (port_map->find(key) == port_map->end()) {
      (*port_map)[key] = std::set<std::string, std::less<>>();
    }
    (*port_map)[key].insert(value);
  }

  // Check if both the fragments have the same executor type
  auto& upstream_executor = *upstream_frag->executor_shared();
  auto& downstream_executor = *downstream_frag->executor_shared();
  if (typeid(upstream_executor) != typeid(downstream_executor)) {
    throw std::runtime_error(
        fmt::format("Fragments have different executor types: upstream fragment "
                    "executor type - {} "
                    "vs downstream fragment executor type - {}.",
                    std::string(typeid(upstream_executor).name()),
                    std::string(typeid(downstream_executor).name())));
  }

  // Add the flow to the fragment graph
  // Note that we don't check if the operator names are valid or not.
  // It will be checked when the graph is run.
  fragment_graph().add_flow(upstream_frag, downstream_frag, port_map);
}

void Application::set_ucx_env() {
  // If UCX_PROTO_ENABLE is not already set, set it to enable UCX Protocols v2
  setenv("UCX_PROTO_ENABLE", "y", 0);
  // Reuse address
  // (see https://github.com/openucx/ucx/issues/8585 and https://github.com/rapidsai/ucxx#c-1)
  setenv("UCX_TCP_CM_REUSEADDR", "y", 0);
  // Disable UCX CM to use all devices
  // (see issue 4233845)
  setenv("UCX_CM_USE_ALL_DEVICES", "n", 0);

  // Disable UCX memory type cache
  // (see https://ucx-py.readthedocs.io/en/latest/configuration.html#ucx-memtype-cache and
  //  https://openucx.readthedocs.io/en/master/faq.html#i-m-running-ucx-with-gpu-memory-and-geting-a-segfault-why)

  // Set memtype cache to disabled unless the user has already set it
  if (std::getenv("UCX_MEMTYPE_CACHE") == nullptr) {
    // UCX uses `__attribute__((constructor))` to load 'global' UCS environment variables even
    // before Holoscan's main() is called, so cannot set this one via `setenv`.
    // We need to set it via a call to`ucs_global_opts_set_value` instead.

    ucs_status_t status = ucs_global_opts_set_value("MEMTYPE_CACHE", "no");
    if (status != UCS_OK) {
      HOLOSCAN_LOG_WARN("Failed to set UCX_MEMTYPE_CACHE=no with status: {}",
                        ucs_status_string(status));
    }
  }
}

void Application::set_v4l2_env() {
  const char* env_value = std::getenv("HOLOSCAN_DISABLE_V4L2_RTLD_NODELETE");
  // Workaround to avoid v4l2 seg fault https://nvbugs/4210082
  if (env_value == nullptr) {
    HOLOSCAN_LOG_DEBUG(
        "Enable the libnvv4l2 workaround by setting the "
        "`LIBV4L2_ENABLE_RTLD_NODELETE` environment variable.");
    setenv("LIBV4L2_ENABLE_RTLD_NODELETE", "1", 0);
  }
}

void Application::check_stack_size() {
  struct rlimit rl;
  if (getrlimit(RLIMIT_STACK, &rl) == 0) {
    constexpr rlim_t min_stack_size = 33554432;  // 32 MB in bytes (32 * 1024 * 1024)

    // Check if the current limit is less than the minimum
    // (RLIM_INFINITY means unlimited, which is fine)
    if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < min_stack_size) {
      HOLOSCAN_LOG_WARN(
          "Current stack size limit ({} bytes / {} KB) is below the recommended minimum "
          "({} bytes / {} KB). Consider increasing it with 'ulimit -s {}'. "
          "For Docker, use '--ulimit stack={}'",
          rl.rlim_cur,
          rl.rlim_cur / 1024,
          min_stack_size,
          min_stack_size / 1024,
          min_stack_size / 1024,
          min_stack_size);
    }
  } else {
    HOLOSCAN_LOG_DEBUG("Failed to query stack size limit using getrlimit()");
  }
}

void Application::reset_state() {
  if (!is_run_called_) {
    HOLOSCAN_LOG_DEBUG(
        "skipping application state reset since run() or run_async() was not called yet");
    return;
  }

  HOLOSCAN_LOG_DEBUG("Resetting Application state to prepare for subsequent runs");
  // Reset the fragment state
  Fragment::reset_state();

  if (is_run_called_) {
    // Since the fragment_graph_ is used by the `Application::track_distributed()` method in
    // distributed applications, when the fragment graph is already composed but the operator graph
    // requires recomposition, we must preserve the data_flow_tracker_ pointers from the previous
    // fragment graph.
    if (is_fragment_graph_composed_ && !is_composed_) {
      // Reset the fragment graph but keep the existing trackers
      auto old_fragment_graph = std::move(fragment_graph_);
      // Redundant for std::shared_ptr after move, but kept to clarify intent:
      // lazy re-initialization via fragment_graph()
      fragment_graph_.reset();

      // Recompose the main graph and fragment graph
      compose_graph();

      // Move the tracker object to the new fragment graph
      for (const auto& each_fragment : old_fragment_graph->get_nodes()) {
        auto new_fragment = fragment_graph().find_node(each_fragment->name());
        if (new_fragment) {
          new_fragment->data_flow_tracker_ = std::move(each_fragment->data_flow_tracker_);
        }
      }
      // Reset the old fragment graph to release the memory
      old_fragment_graph.reset();
    }

    // Reset the application status
    app_driver_.reset();

    // Reset the application status
    app_worker_.reset();
  }
}

void Application::run() {
  // Debug log to show that the run() function is executed
  // (with the logging function pointer info to check if the logging function pointer address is
  // the same as the one set in the Python side).
  // This message is checked by the test_app_log_function in test_application_minimal.py.
  HOLOSCAN_LOG_DEBUG("Executing Application::run()... (log_func_ptr=0x{:x})",
                     reinterpret_cast<uint64_t>(&nvidia::LoggingFunction));
  if (cli_parser_.has_error()) {
    auto err_msg = std::string(
        "Application::run() failed to run because of CLI parser errors. "
        "Please check the CLI arguments and try again.");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  set_ucx_env();
  set_v4l2_env();
  check_stack_size();

  // Initialize clean state to ensure proper execution and support multiple consecutive runs
  reset_state();

  driver().run();
  is_run_called_ = true;
}

std::future<void> Application::run_async() {
  if (cli_parser_.has_error()) {
    auto err_msg = std::string(
        "Application::run_async() failed to run because of CLI parser errors. "
        "Please check the CLI arguments and try again.");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  set_ucx_env();
  set_v4l2_env();
  check_stack_size();

  // Initialize clean state to ensure proper execution and support multiple consecutive runs
  reset_state();

  auto future = driver().run_async();
  is_run_called_ = true;
  return future;
}

bool Application::is_metadata_enabled() const {
  return is_metadata_enabled_;
}

void Application::is_metadata_enabled(bool enabled) {
  static std::once_flag warn_flag;
  std::call_once(warn_flag, []() {
    HOLOSCAN_LOG_WARN(
        "The Application::is_metadata_enabled(bool) setter is deprecated. Please use "
        "Application::(bool) instead.");
  });
  is_metadata_enabled_ = enabled;
}

void Application::enable_metadata(bool enabled) {
  is_metadata_enabled_ = enabled;
}

void Application::add_data_logger(const std::shared_ptr<DataLogger>& logger) {
  if (fragment_graph().is_empty()) {
    // single-fragment application
    Fragment::add_data_logger(logger);
  } else {
    // add the data logger to each fragment in the fragment graph
    for (const auto& fragment : fragment_graph().get_nodes()) {
      fragment->add_data_logger(logger);
    }
  }
}

MetadataPolicy Application::metadata_policy() const {
  return metadata_policy_;
}

void Application::metadata_policy(MetadataPolicy policy) {
  metadata_policy_ = policy;
}

std::unordered_map<std::string, DataFlowTracker*> Application::track_distributed(
    uint64_t num_start_messages_to_skip, uint64_t num_last_messages_to_discard,
    int latency_threshold, bool is_limited_tracking) {
  if (!is_composed_) {
    compose_graph();
  }
  std::unordered_map<std::string, DataFlowTracker*> trackers;
  auto& frag_graph = fragment_graph();
  // iterate over all nodes in frag_graph
  for (const auto& each_fragment : frag_graph.get_nodes()) {
    // if track has not been called on the fragment, then call the tracker
    if (!each_fragment->data_flow_tracker()) {
      each_fragment->track(num_start_messages_to_skip,
                           num_last_messages_to_discard,
                           latency_threshold,
                           is_limited_tracking);
    }
    trackers[each_fragment->name()] = each_fragment->data_flow_tracker();
  }
  return trackers;
}

AppDriver& Application::driver() {
  if (!app_driver_) {
    app_driver_ = std::make_shared<AppDriver>(this);
  }
  return *app_driver_;
}

AppWorker& Application::worker() {
  if (!app_worker_) {
    app_worker_ = std::make_shared<AppWorker>(this);
  }
  return *app_worker_;
}

void Application::process_arguments() {
  // If the user has not provided any arguments, we will use the arguments from the command line.
  if (argv_.empty()) {
    auto args = CLI::detail::args();
    argv_.assign(args.begin(), args.end());
  }

  cli_parser_.initialize(app_description_, app_version_);

  // Parse the arguments.
  cli_parser_.parse(argv_);
}

expected<SchedulerType, ErrorCode> Application::get_distributed_app_scheduler_env() {
  const char* env_value = std::getenv("HOLOSCAN_DISTRIBUTED_APP_SCHEDULER");
  if (env_value != nullptr && env_value[0] != '\0') {
    if (std::strcmp(env_value, "greedy") == 0) {
      return SchedulerType::kGreedy;
    } else if (std::strcmp(env_value, "multithread") == 0 ||
               std::strcmp(env_value, "multi_thread") == 0) {
      return SchedulerType::kMultiThread;
    } else if (std::strcmp(env_value, "event_based") == 0) {
      return SchedulerType::kEventBased;
    } else {
      HOLOSCAN_LOG_ERROR("Invalid value for HOLOSCAN_DISTRIBUTED_APP_SCHEDULER: {}", env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    }
  } else {
    return SchedulerType::kDefault;
  }
}

expected<bool, ErrorCode> Application::get_stop_on_deadlock_env() {
  const char* env_value = std::getenv("HOLOSCAN_STOP_ON_DEADLOCK");
  if (env_value != nullptr && env_value[0] != '\0') {
    bool value = AppDriver::get_bool_env_var("HOLOSCAN_STOP_ON_DEADLOCK", true);
    return value;
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

expected<int64_t, ErrorCode> Application::get_stop_on_deadlock_timeout_env() {
  const char* env_value = std::getenv("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT");
  if (env_value != nullptr && env_value[0] != '\0') {
    try {
      return std::stoll(env_value);
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid value for HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT: {}", env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Value for HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT is out of range: {}",
                         env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    }
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

expected<int64_t, ErrorCode> Application::get_ucx_network_connection_timeout_env() {
  const char* env_value = std::getenv("HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT");
  if (env_value != nullptr && env_value[0] != '\0') {
    try {
      return std::stoll(env_value);
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid value for HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT: {}",
                         env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Value for HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT is out of range: {}",
                         env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    }
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

expected<int64_t, ErrorCode> Application::get_max_duration_ms_env() {
  const char* env_value = std::getenv("HOLOSCAN_MAX_DURATION_MS");
  if (env_value != nullptr && env_value[0] != '\0') {
    try {
      return std::stoll(env_value);
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid value for HOLOSCAN_MAX_DURATION_MS: {}", env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Value for HOLOSCAN_MAX_DURATION_MS is out of range: {}", env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    }
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

expected<double, ErrorCode> Application::get_check_recession_period_ms_env() {
  const char* env_value = std::getenv("HOLOSCAN_CHECK_RECESSION_PERIOD_MS");
  if (env_value != nullptr && env_value[0] != '\0') {
    try {
      return std::stod(env_value);
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid value for HOLOSCAN_CHECK_RECESSION_PERIOD_MS: {}", env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Value for HOLOSCAN_CHECK_RECESSION_PERIOD_MS is out of range: {}",
                         env_value);
      return make_unexpected(ErrorCode::kInvalidArgument);
    }
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

void Application::compose_graph() {
  if (is_composed_) {
    HOLOSCAN_LOG_DEBUG("The application({}) has already been composed. Skipping...", name());
    return;
  }

  // Load extensions from the config file before composing the graph.
  // (The GXFCodeletOp and GXFComponentResource classes are required to access the underlying GXF
  //  types in the setup() method when composing a graph.)
  load_extensions_from_config();

  compose();
  is_composed_ = true;
  is_fragment_graph_composed_ = true;
}

void Application::set_scheduler_for_fragments(std::vector<FragmentNodeType>& target_fragments,
                                              const std::shared_ptr<Scheduler>& app_scheduler) {
  // Collect environment variable overrides once
  // TODO(grelee): switch to designated initializers once C++20 is supported)
  SchedulerEnvOverrides env_overrides{Application::get_stop_on_deadlock_env(),
                                      Application::get_stop_on_deadlock_timeout_env(),
                                      Application::get_ucx_network_connection_timeout_env(),
                                      Application::get_max_duration_ms_env(),
                                      Application::get_check_recession_period_ms_env()};

  // Get scheduler type from environment variable (kDefault if not set)
  auto scheduler_type_env = Application::get_distributed_app_scheduler_env();
  SchedulerType env_scheduler_type =
      scheduler_type_env ? scheduler_type_env.value() : SchedulerType::kDefault;

  for (auto& fragment : target_fragments) {
    if (fragment->is_gpu_resident()) {
      HOLOSCAN_LOG_DEBUG("Scheduler is not set for GPU-resident fragment ({}).", fragment->name());
      continue;
    }
    std::shared_ptr<Scheduler>& scheduler = fragment->scheduler_;
    const auto& frag_name = fragment->name();

    // Check if the user has explicitly set a scheduler on the fragment
    bool user_set_scheduler = (scheduler != nullptr);

    // If fragment doesn't have a scheduler set, but the application does,
    // we'll create a new scheduler of the same type (can't share the instance
    // because GXF entities must have unique names per fragment)
    bool use_app_scheduler_type = (!scheduler && app_scheduler);
    if (use_app_scheduler_type) {
      HOLOSCAN_LOG_DEBUG("Fragment '{}': Using scheduler type from Application", frag_name);
    }

    // Detect the type of scheduler to use (from fragment scheduler or app scheduler)
    const auto& scheduler_to_check = scheduler ? scheduler : app_scheduler;
    SchedulerType user_scheduler_type = detect_scheduler_type(scheduler_to_check, frag_name);

    // Determine the target scheduler type
    SchedulerType resolved_scheduler_type = resolve_scheduler_type(
        user_set_scheduler, use_app_scheduler_type, user_scheduler_type, env_scheduler_type);

    // GreedyScheduler is not supported for multi-fragment apps as it causes deadlock.
    // Override to EventBasedScheduler and warn the user.
    if (resolved_scheduler_type == SchedulerType::kGreedy) {
      HOLOSCAN_LOG_WARN(
          "Fragment '{}': GreedyScheduler is not supported for multi-fragment applications "
          "(causes deadlock). Using EventBasedScheduler instead.",
          frag_name);
      resolved_scheduler_type = SchedulerType::kEventBased;
      // Reset the fragment scheduler and user_set_scheduler flag to force creation of a new
      // EventBasedScheduler. Keep use_app_scheduler_type so we still copy args (including clock).
      scheduler.reset();
      user_set_scheduler = false;
    }

    // Warn if user's scheduler is being overridden by environment variable
    if (user_set_scheduler && env_scheduler_type != SchedulerType::kDefault &&
        user_scheduler_type != resolved_scheduler_type) {
      HOLOSCAN_LOG_INFO(
          "Fragment '{}': User-defined scheduler ({}) is being overridden by "
          "HOLOSCAN_DISTRIBUTED_APP_SCHEDULER environment variable to {}",
          frag_name,
          magic_enum::enum_name(user_scheduler_type),
          magic_enum::enum_name(resolved_scheduler_type));
    }

    // Create a new scheduler if needed, or keep the existing one
    auto previous_scheduler = scheduler;
    scheduler = create_scheduler_for_fragment(fragment,
                                              scheduler,
                                              resolved_scheduler_type,
                                              user_set_scheduler,
                                              use_app_scheduler_type,
                                              user_scheduler_type,
                                              env_scheduler_type);

    // If a new scheduler was created, configure its arguments
    if (scheduler != previous_scheduler) {
      if (use_app_scheduler_type && app_scheduler) {
        // Copy arguments from the app scheduler
        for (const auto& arg : app_scheduler->args()) {
          if (arg.arg_type().element_type() == ArgElementType::kResource &&
              arg.arg_type().container_type() == ArgContainerType::kNative &&
              arg.name() == "clock") {
            auto clock_resource = std::any_cast<std::shared_ptr<Resource>>(arg.value());
            if (clock_resource && clock_resource->fragment() != fragment.get()) {
              auto cloned_clock =
                  clone_scheduler_clock_for_fragment(fragment, clock_resource, scheduler->name());
              if (cloned_clock) {
                HOLOSCAN_LOG_DEBUG("Fragment '{}': Cloned clock resource '{}' for scheduler '{}'",
                                   frag_name,
                                   clock_resource->name(),
                                   scheduler->name());
                scheduler->add_arg(holoscan::Arg("clock", cloned_clock));
              } else {
                HOLOSCAN_LOG_DEBUG(
                    "Fragment '{}': Skipping app scheduler clock '{}' to use default clock",
                    frag_name,
                    clock_resource->name());
              }
              continue;
            }
          }
          // Note: This message text is relied on by tests in distributed_app_scheduler_test.cpp.
          HOLOSCAN_LOG_DEBUG("Fragment '{}': Copying argument '{}' from Application scheduler",
                             frag_name,
                             arg.name());
          scheduler->add_arg(arg);
        }
      } else {
        // No app scheduler - use default arguments based on scheduler type
        add_default_scheduler_args(scheduler, resolved_scheduler_type, fragment);
      }
    }
    // else: User set a scheduler and we're keeping it - respect their configuration entirely.
    // Environment variable overrides below will still apply if set.

    // Apply environment variable overrides
    apply_scheduler_env_overrides(scheduler, env_overrides);

    fragment->scheduler(scheduler);
  }
}

std::shared_ptr<distributed::AppDriverClient> Application::app_driver_client() const {
  if (!app_worker_) {
    HOLOSCAN_LOG_ERROR("Cannot get AppDriverClient for this fragment: app_worker_ is null");
    return nullptr;
  }
  return app_worker_->app_driver_client();
}

void Application::attach_services_to_fragment(const std::shared_ptr<Fragment>& fragment) {
  std::unordered_set<std::string> registered_service_ids;
  for (const auto& [service_key, service] : fragment_services_by_key()) {
    if (registered_service_ids.find(service_key.id) == registered_service_ids.end()) {
      HOLOSCAN_LOG_DEBUG(
          "Registering service '{}' with fragment '{}'", service_key.id, fragment->name());
      // Register service from the application to the fragment
      fragment->register_service_from(this, service_key.id);
      registered_service_ids.insert(service_key.id);
    }
  }
}

void Application::initiate_distributed_app_shutdown(const std::string& fragment_name) {
  // Get access to the AppDriverClient
  auto driver_client = app_driver_client();

  if (driver_client) {
    HOLOSCAN_LOG_INFO("Application::initiate_distributed_app_shutdown started");
    // Initiate shutdown via RPC
    driver_client->initiate_shutdown(fragment_name);
  } else if (!fragment_graph().is_empty()) {
    HOLOSCAN_LOG_DEBUG("Initiating local multi-fragment app shutdown");
    initiate_local_app_shutdown(fragment_name);
  } else {
    HOLOSCAN_LOG_WARN(
        "Cannot initiate distributed app shutdown: fragment graph is empty indicating that this is "
        "a single-fragment application, not a distributed one.");
  }
  return;
}

void Application::initiate_local_app_shutdown(const std::string& fragment_name) {
  (void)fragment_name;  // Kept for API compatibility; local shutdown stops all fragments.

  auto& frag_graph = fragment_graph();
  if (frag_graph.is_empty()) {
    HOLOSCAN_LOG_DEBUG("Cannot initiate local shutdown: fragment graph is empty");
    return;
  }

  HOLOSCAN_LOG_INFO("Initiating orderly shutdown of local multi-fragment application");

  // Get all fragments - we'll remove them from a copy of the graph as we shut them down
  auto remaining_fragments = frag_graph.get_nodes();
  std::unordered_set<std::string> terminated_fragments;

  while (!remaining_fragments.empty()) {
    // Find current root fragments
    std::vector<FragmentNodeType> current_roots;

    HOLOSCAN_LOG_DEBUG("Remaining fragments to shut down:");
    for (const auto& fragment : remaining_fragments) {
      HOLOSCAN_LOG_DEBUG("\t{}", fragment->name());
    }

    for (const auto& fragment : remaining_fragments) {
      // Need to find the node in the graph first, since the graph may have been modified
      auto node = frag_graph.find_node(fragment->name());
      if (!node) {
        HOLOSCAN_LOG_WARN("Fragment '{}' not found in graph - treating as root", fragment->name());
        current_roots.push_back(fragment);
      } else if (frag_graph.is_root(node)) {
        HOLOSCAN_LOG_DEBUG("Fragment '{}' is currently a root fragment", fragment->name());
        current_roots.push_back(fragment);
      } else {
        HOLOSCAN_LOG_DEBUG("Fragment '{}' is NOT a root (has {} upstream fragments)",
                           fragment->name(),
                           frag_graph.get_previous_nodes(node).size());
      }
    }

    // If there were no root fragments, just shutdown any remaining fragments to avoid deadlock
    if (current_roots.empty()) {
      HOLOSCAN_LOG_WARN(
          "No root fragments found in remaining fragments - shutting down all remaining");
      current_roots = remaining_fragments;
    }

    // Terminate current root fragments
    for (const auto& root_fragment : current_roots) {
      HOLOSCAN_LOG_INFO("Terminating fragment '{}' via stop_execution()", root_fragment->name());

      // Get the stop_on_deadlock_timeout from this fragment's scheduler
      // This is the time the scheduler waits before confirming a deadlock and exiting
      // when all operators have been stopped via stop_execution()
      int64_t stop_on_deadlock_timeout = kDefaultStopOnDeadlockTimeout;
      auto scheduler = root_fragment->scheduler();

      // Try to cast to known scheduler types that have stop_on_deadlock_timeout()
      try {
        if (auto* ebs = dynamic_cast<EventBasedScheduler*>(scheduler.get())) {
          stop_on_deadlock_timeout = ebs->stop_on_deadlock_timeout();
        } else if (auto* mts = dynamic_cast<MultiThreadScheduler*>(scheduler.get())) {
          stop_on_deadlock_timeout = mts->stop_on_deadlock_timeout();
        } else if (auto* gs = dynamic_cast<GreedyScheduler*>(scheduler.get())) {
          stop_on_deadlock_timeout = gs->stop_on_deadlock_timeout();
        }
      } catch (const std::runtime_error&) {
        // Parameter not set on scheduler, try environment variable
        auto env_result = get_stop_on_deadlock_timeout_env();
        if (env_result.has_value()) {
          stop_on_deadlock_timeout = env_result.value();
        }
        // else: keep default 1000ms
      }

      // Add extra margin (250ms) to account for any processing overhead
      int64_t fragment_shutdown_grace_period_ms = stop_on_deadlock_timeout + 250;

      HOLOSCAN_LOG_DEBUG(
          "Fragment '{}': using shutdown grace period of {} ms (stop_on_deadlock_timeout={} ms)",
          root_fragment->name(),
          fragment_shutdown_grace_period_ms,
          stop_on_deadlock_timeout);

      // Use stop_execution() instead of executor().interrupt() to allow queued UCX messages
      // to be sent, avoiding connection errors
      root_fragment->stop_execution();

      // Wait to give the fragment time to properly shut down
      // This prevents UCX messages that haven't been sent yet from being lost
      std::this_thread::sleep_for(std::chrono::milliseconds(fragment_shutdown_grace_period_ms));

      // Remove this fragment from the graph so we can find the next layer of roots
      frag_graph.remove_node(root_fragment);
      terminated_fragments.insert(root_fragment->name());
    }

    // Update remaining fragments
    remaining_fragments.erase(std::remove_if(remaining_fragments.begin(),
                                             remaining_fragments.end(),
                                             [&terminated_fragments](const FragmentNodeType& frag) {
                                               return terminated_fragments.find(frag->name()) !=
                                                      terminated_fragments.end();
                                             }),
                              remaining_fragments.end());
  }

  HOLOSCAN_LOG_INFO("Local multi-fragment application shutdown complete");
}

bool Application::is_any_fragment_gpu_resident() {
  for (const auto& fragment : fragment_graph().get_nodes()) {
    if (fragment->is_gpu_resident()) {
      return true;
    }
  }
  return false;
}

}  // namespace holoscan
