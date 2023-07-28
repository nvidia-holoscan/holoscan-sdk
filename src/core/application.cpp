/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/operator.hpp"
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

      auto deleter = [](FILE* f) { std::fclose(f); };
      std::unique_ptr<FILE, decltype(deleter)> fp_unique(std::fopen("/proc/self/cmdline", "r"),
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

Application::Application(const std::vector<std::string>& argv) : Fragment(), argv_(argv) {
  // Set the log level from the environment variable if it exists.
  // Or, set the default log level to INFO if it hasn't been set by the user.
  if (!Logger::log_level_set_by_user) { holoscan::set_log_level(LogLevel::INFO); }
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

FragmentGraph& Application::fragment_graph() {
  if (!fragment_graph_) { fragment_graph_ = make_graph<FragmentFlowGraph>(); }
  return *fragment_graph_;
}

void Application::add_fragment(const std::shared_ptr<Fragment>& frag) {
  fragment_graph().add_node(frag);
}

void Application::add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                           const std::shared_ptr<Fragment>& downstream_frag,
                           std::set<std::pair<std::string, std::string>> port_pairs) {
  // If port_pairs is empty, print an error message and return.
  if (port_pairs.empty()) {
    HOLOSCAN_LOG_ERROR("Unable to add fragment flow with empty port_pairs");
    return;
  }

  auto port_map = std::make_shared<FragmentGraph::EdgeDataElementType>();

  // Convert the port name pairs to port map
  // (set<pair<string, string>> -> map<string, set<string>>)
  for (const auto& [key, value] : port_pairs) {
    if (port_map->find(key) == port_map->end()) {
      (*port_map)[key] = std::set<std::string, std::less<>>();
    }
    (*port_map)[key].insert(value);
  }

  // Add the flow to the fragment graph
  // Note that we don't check if the operator names are valid or not.
  // It will be checked when the graph is run.
  fragment_graph().add_flow(upstream_frag, downstream_frag, port_map);
}

void Application::run() {
  if (cli_parser_.has_error()) { return; }

  driver().run();
}

std::future<void> Application::run_async() {
  if (cli_parser_.has_error()) { return {}; }

  return driver().run_async();
}

AppDriver& Application::driver() {
  if (!app_driver_) { app_driver_ = std::make_shared<AppDriver>(this); }
  return *app_driver_;
}

AppWorker& Application::worker() {
  if (!app_worker_) { app_worker_ = std::make_shared<AppWorker>(this); }
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
    } else if (std::strcmp(env_value, "multithread") == 0) {
      return SchedulerType::kMultiThread;
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

void Application::set_scheduler_for_fragments(std::vector<FragmentNodeType>& target_fragments) {
  constexpr bool kDefaultStopOnDeadlock = true;
  constexpr int64_t kDefaultStopOnDeadlockTimeout = 5000L;
  constexpr int64_t kDefaultMaxDurationMs = -1L;  // optional value
  constexpr double kDefaultCheckRecessionPeriodMs = 0.0;

  auto scheduler_type_env = Application::get_distributed_app_scheduler_env();
  auto stop_on_deadlock_env = Application::get_stop_on_deadlock_env();
  bool stop_on_deadlock =
      stop_on_deadlock_env ? stop_on_deadlock_env.value() : kDefaultStopOnDeadlock;
  auto stop_on_deadlock_timeout_env = Application::get_stop_on_deadlock_timeout_env();
  int64_t stop_on_deadlock_timeout = stop_on_deadlock_timeout_env
                                         ? stop_on_deadlock_timeout_env.value()
                                         : kDefaultStopOnDeadlockTimeout;
  auto max_duration_ms_env = Application::get_max_duration_ms_env();
  int64_t max_duration_ms =
      max_duration_ms_env ? max_duration_ms_env.value() : kDefaultMaxDurationMs;
  auto check_recession_period_ms_env = Application::get_check_recession_period_ms_env();
  double check_recession_period_ms = check_recession_period_ms_env
                                         ? check_recession_period_ms_env.value()
                                         : kDefaultCheckRecessionPeriodMs;

  SchedulerType scheduler_type = SchedulerType::kDefault;
  if (scheduler_type_env) { scheduler_type = scheduler_type_env.value(); }

  for (auto& fragment : target_fragments) {
    std::shared_ptr<Scheduler>& scheduler = fragment->scheduler_;
    SchedulerType scheduler_setting = scheduler_type;

    // Make sure that multi-thread scheduler is used for the fragment
    if (scheduler_setting == SchedulerType::kDefault) {
      // Check if holoscan::MultiThreadScheduler is already set to the fragment.
      // If it is, then we should use the default scheduler.
      // Otherwise, we should set new multi-thread scheduler.
      auto multi_thread_scheduler =
          std::dynamic_pointer_cast<holoscan::MultiThreadScheduler>(scheduler);
      if (!multi_thread_scheduler) { scheduler_setting = SchedulerType::kMultiThread; }
    }

    switch (scheduler_setting) {
      case SchedulerType::kDefault:
        // Override the existing scheduler to use the proper deadlock timeout value so that
        // the scheduler allows some time for the operator having UCXReceiver to receive input
        // messages from the remote operators. This is necessary because the scheduler would stop
        // immediately when no input messages from UCXReceiver are received.
        scheduler->add_arg(holoscan::Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout));
        break;
      case SchedulerType::kGreedy:
        scheduler = fragment->make_scheduler<holoscan::GreedyScheduler>("greedy-scheduler");
        scheduler->add_arg(holoscan::Arg("stop_on_deadlock", stop_on_deadlock));
        scheduler->add_arg(holoscan::Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout));
        if (max_duration_ms >= 0) {
          scheduler->add_arg(holoscan::Arg("max_duration_ms", max_duration_ms));
        }
        scheduler->add_arg(holoscan::Arg("check_recession_period_ms", check_recession_period_ms));
        break;
      case SchedulerType::kMultiThread: {
        scheduler =
            fragment->make_scheduler<holoscan::MultiThreadScheduler>("multithread-scheduler");
        unsigned int num_processors = std::thread::hardware_concurrency();
        int64_t worker_thread_number =
            std::min(fragment->graph().get_nodes().size(), static_cast<size_t>(num_processors));
        scheduler->add_arg(holoscan::Arg("stop_on_deadlock", stop_on_deadlock));
        scheduler->add_arg(holoscan::Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout));
        if (max_duration_ms >= 0) {
          scheduler->add_arg(holoscan::Arg("max_duration_ms", max_duration_ms));
        }
        scheduler->add_arg(holoscan::Arg("check_recession_period_ms", check_recession_period_ms));
        scheduler->add_arg(holoscan::Arg("worker_thread_number", worker_thread_number));
      } break;
    }

    // Override arguments from environment variables
    if (stop_on_deadlock_env) {
      scheduler->add_arg(holoscan::Arg("stop_on_deadlock", stop_on_deadlock_env.value()));
    }
    if (stop_on_deadlock_timeout_env) {
      scheduler->add_arg(
          holoscan::Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout_env.value()));
    }
    if (max_duration_ms_env) {
      scheduler->add_arg(holoscan::Arg("max_duration_ms", max_duration_ms_env.value()));
    }
    if (check_recession_period_ms_env) {
      scheduler->add_arg(
          holoscan::Arg("check_recession_period_ms", check_recession_period_ms_env.value()));
    }
    fragment->scheduler(scheduler);
  }
}

}  // namespace holoscan
