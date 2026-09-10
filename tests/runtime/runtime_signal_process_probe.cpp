/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <dirent.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>

#include "runtime/managed_shutdown_signals.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

extern char** environ;

// REQ: RUNTIME-047
// REQ: RUNTIME-049
namespace {

using holoscan::detail::RuntimeManagedSignalDiagnostic;

volatile sig_atomic_t g_forwarded_signal{};
volatile sig_atomic_t g_forwarded_fd{-1};
volatile sig_atomic_t g_forwarded_count{};
volatile sig_atomic_t g_forwarded_tid{};
std::array<volatile sig_atomic_t, 4> g_forwarded_history{};

extern "C" void record_forwarded_signal(const int signal) {
  g_forwarded_tid = static_cast<sig_atomic_t>(syscall(SYS_gettid));
  const sig_atomic_t index = g_forwarded_count;
  if (index >= 0 && index < static_cast<sig_atomic_t>(std::size(g_forwarded_history))) {
    g_forwarded_history[index] = signal;
    g_forwarded_count = index + 1;
  }
  if (g_forwarded_fd >= 0) {
    const std::uint8_t value{1U};
    [[maybe_unused]] const ssize_t written =
        write(static_cast<int>(g_forwarded_fd), &value, sizeof(value));
  }
  // Publish the signal identity last; polling tests use it as the completion flag for the
  // accompanying handler evidence.
  g_forwarded_signal = signal;
}

void reset_forwarded_signal_history() noexcept {
  g_forwarded_signal = 0;
  g_forwarded_count = 0;
  g_forwarded_tid = 0;
  for (auto& signal : g_forwarded_history) {
    signal = 0;
  }
}

struct Pipe {
  std::array<int, 2> fd{{-1, -1}};

  [[nodiscard]] bool open() noexcept { return pipe(fd.data()) == 0; }

  void close_read() noexcept {
    if (fd[0] >= 0) {
      static_cast<void>(close(fd[0]));
      fd[0] = -1;
    }
  }

  void close_write() noexcept {
    if (fd[1] >= 0) {
      static_cast<void>(close(fd[1]));
      fd[1] = -1;
    }
  }

  ~Pipe() {
    close_read();
    close_write();
  }
};

[[nodiscard]] std::int64_t monotonic_time_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] bool write_byte(const int fd, const std::uint8_t value = 1U) noexcept {
  ssize_t result{};
  do {
    result = write(fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  return result == sizeof(value);
}

[[nodiscard]] bool read_byte(const int fd) noexcept {
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  int polled{};
  do {
    polled = poll(&descriptor, 1U, 10'000);
  } while (polled < 0 && errno == EINTR);
  if (polled != 1 || (descriptor.revents & (POLLIN | POLLHUP)) == 0) {
    return false;
  }
  std::uint8_t value{};
  ssize_t result{};
  do {
    result = read(fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  return result == sizeof(value);
}

[[nodiscard]] bool byte_ready_for(const int fd, const std::chrono::milliseconds timeout) noexcept {
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  int polled{};
  do {
    polled = poll(&descriptor, 1U, static_cast<int>(timeout.count()));
  } while (polled < 0 && errno == EINTR);
  return polled == 1 && (descriptor.revents & (POLLIN | POLLHUP)) != 0;
}

struct ManagedDescriptorCounts {
  std::size_t signal{};
  std::size_t event{};
};

[[nodiscard]] ManagedDescriptorCounts managed_descriptor_counts() noexcept {
  DIR* const directory = opendir("/proc/self/fd");
  if (directory == nullptr) {
    return {};
  }
  const int directory_fd = dirfd(directory);
  ManagedDescriptorCounts counts;
  while (const dirent* const entry = readdir(directory)) {
    if (entry->d_name[0] == '.' || std::atoi(entry->d_name) == directory_fd) {
      continue;
    }
    std::array<char, 128> target{};
    const ssize_t size = readlinkat(directory_fd, entry->d_name, target.data(), target.size() - 1U);
    if (size <= 0) {
      continue;
    }
    const std::string_view link(target.data(), static_cast<std::size_t>(size));
    if (link == "anon_inode:[signalfd]") {
      ++counts.signal;
    } else if (link == "anon_inode:[eventfd]") {
      ++counts.event;
    }
  }
  static_cast<void>(closedir(directory));
  return counts;
}

struct LivenessEvidence {
  std::int64_t received_ns{};
  std::int64_t stop_observed_ns{};
  std::int64_t callback_return_ns{};
  std::int64_t wait_return_ns{};
};

[[nodiscard]] bool write_liveness_evidence(const int fd,
                                           const LivenessEvidence& evidence) noexcept {
  const auto* cursor = reinterpret_cast<const std::byte*>(&evidence);
  std::size_t remaining = sizeof(evidence);
  while (remaining != 0U) {
    const ssize_t result = write(fd, cursor, remaining);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      return false;
    }
    cursor += result;
    remaining -= static_cast<std::size_t>(result);
  }
  return true;
}

[[nodiscard]] bool read_liveness_evidence(const int fd, LivenessEvidence& evidence) noexcept {
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  int polled{};
  do {
    polled = poll(&descriptor, 1U, 10'000);
  } while (polled < 0 && errno == EINTR);
  if (polled != 1 || (descriptor.revents & (POLLIN | POLLHUP)) == 0) {
    return false;
  }
  auto* cursor = reinterpret_cast<std::byte*>(&evidence);
  std::size_t remaining = sizeof(evidence);
  while (remaining != 0U) {
    const ssize_t result = read(fd, cursor, remaining);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      return false;
    }
    cursor += result;
    remaining -= static_cast<std::size_t>(result);
  }
  return true;
}

[[nodiscard]] bool wait_for_child_exit(const pid_t child, int& status) noexcept {
  pid_t waited{};
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  return waited == child;
}

[[nodiscard]] bool wait_for_child_exit_for(const pid_t child, int& status,
                                           const std::chrono::seconds timeout) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const pid_t waited = waitpid(child, &status, WNOHANG);
    if (waited == child) {
      return true;
    }
    if (waited < 0 && errno != EINTR) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

class ProcessSignalSink final : public holoscan::detail::RuntimeStructuralDiagnosticSink,
                                public holoscan::detail::RuntimeTerminalSummarySink {
 public:
  struct ManagedEvent {
    std::string token;
    RuntimeManagedSignalDiagnostic detail;
    std::int64_t timestamp_ns{};
  };

  explicit ProcessSignalSink(const int received_fd = -1, const int escalated_fd = -1) noexcept
      : received_fd_(received_fd), escalated_fd_(escalated_fd) {}

  void publish(std::span<const holoscan::Diagnostic> diagnostics) override {
    const std::lock_guard lock(mutex_);
    for (const auto& diagnostic : diagnostics) {
      tokens_.push_back(diagnostic.token);
    }
  }

  void publish_managed_signal_event(const holoscan::Diagnostic& diagnostic,
                                    const RuntimeManagedSignalDiagnostic& detail) override {
    {
      const std::lock_guard lock(mutex_);
      tokens_.push_back(diagnostic.token);
      managed_events_.push_back(ManagedEvent{diagnostic.token, detail, monotonic_time_ns()});
    }
    if (diagnostic.token == "RUNTIME_MANAGED_SIGNAL_RECEIVED" &&
        !received_written_.exchange(true, std::memory_order_acq_rel) && received_fd_ >= 0) {
      static_cast<void>(write_byte(received_fd_));
    }
    if (diagnostic.token == "RUNTIME_MANAGED_SIGNAL_ESCALATED" &&
        !escalated_written_.exchange(true, std::memory_order_acq_rel) && escalated_fd_ >= 0) {
      static_cast<void>(write_byte(escalated_fd_));
    }
  }

  void publish_runtime_details(
      std::span<const holoscan::detail::RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock(mutex_);
    for (const auto& detail : details) {
      if (detail.present && detail.managed_signal.present) {
        terminal_managed_details_.push_back(detail.managed_signal);
      }
    }
  }

  void publish(const holoscan::detail::RuntimeTerminalSummary& summary) override {
    const std::lock_guard lock(mutex_);
    if (summary.category == holoscan::detail::RuntimeTerminalCategory::kFailed) {
      ++failed_summary_count_;
    }
  }

  [[nodiscard]] bool contains(const std::string_view token) const {
    const std::lock_guard lock(mutex_);
    for (const auto& candidate : tokens_) {
      if (candidate == token) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool contains_managed_signal_record() const {
    const std::lock_guard lock(mutex_);
    for (const auto& token : tokens_) {
      if (token.starts_with("RUNTIME_MANAGED_SIGNAL_")) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::size_t event_count(const std::string_view token) const {
    const std::lock_guard lock(mutex_);
    return static_cast<std::size_t>(std::count_if(
        managed_events_.begin(), managed_events_.end(), [token](const ManagedEvent& event) {
          return event.token == token;
        }));
  }

  [[nodiscard]] std::size_t token_count(const std::string_view token) const {
    const std::lock_guard lock(mutex_);
    return static_cast<std::size_t>(std::count(tokens_.begin(), tokens_.end(), std::string{token}));
  }

  [[nodiscard]] std::size_t failed_summary_count() const {
    const std::lock_guard lock(mutex_);
    return failed_summary_count_;
  }

  [[nodiscard]] RuntimeManagedSignalDiagnostic event_detail(const std::string_view token) const {
    const std::lock_guard lock(mutex_);
    for (const auto& event : managed_events_) {
      if (event.token == token) {
        return event.detail;
      }
    }
    return {};
  }

  [[nodiscard]] std::int64_t event_time_ns(const std::string_view token) const {
    const std::lock_guard lock(mutex_);
    for (const auto& event : managed_events_) {
      if (event.token == token) {
        return event.timestamp_ns;
      }
    }
    return 0;
  }

  [[nodiscard]] bool contains_terminal_detail(
      const holoscan::detail::RuntimeManagedSignalPhase phase, const int os_error,
      const holoscan::detail::RuntimeManagedSignalCensusReason census_reason,
      const bool require_binding_generation = true, const bool require_domain_generation = true,
      const std::uint32_t minimum_observed_threads = 0U) const {
    const std::lock_guard lock(mutex_);
    return std::any_of(terminal_managed_details_.begin(),
                       terminal_managed_details_.end(),
                       [&](const RuntimeManagedSignalDiagnostic& detail) {
                         return detail.present && detail.phase == phase &&
                                detail.os_error == os_error &&
                                detail.census_reason == census_reason &&
                                (!require_domain_generation || detail.domain_generation != 0U) &&
                                (!require_binding_generation || detail.binding_generation != 0U) &&
                                detail.observed_thread_count >= minimum_observed_threads;
                       });
  }

 private:
  int received_fd_{-1};
  int escalated_fd_{-1};
  std::atomic_bool received_written_{};
  std::atomic_bool escalated_written_{};
  mutable std::mutex mutex_;
  std::vector<std::string> tokens_;
  std::vector<ManagedEvent> managed_events_;
  std::vector<RuntimeManagedSignalDiagnostic> terminal_managed_details_;
  std::size_t failed_summary_count_{};
};

[[nodiscard]] holoscan::detail::RuntimeManagedSignal runtime_signal(const int signal) noexcept {
  return signal == SIGINT ? holoscan::detail::RuntimeManagedSignal::kSigInt
                          : holoscan::detail::RuntimeManagedSignal::kSigTerm;
}

[[nodiscard]] holoscan::ExecutionPlan make_plan(const std::shared_ptr<ProcessSignalSink>& sink = {},
                                                const std::uint32_t activation_count = 1U) {
  auto plan = holoscan::test::enable_direct_runtime(holoscan::test::make_runtime_test_plan(),
                                                    activation_count);
  if (sink != nullptr) {
    plan = holoscan::detail::RuntimeTestPlanAccess::with_diagnostic_sink(plan, sink);
  }
  return plan;
}

// REQ: RUNTIME-041
struct AutomaticSignalOwnerProbeState {
  static void reset(const int gate_fd) noexcept {
    active.store(0U, std::memory_order_relaxed);
    max_active.store(0U, std::memory_order_relaxed);
    stop_gate_fd.store(gate_fd, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_two_active() {
    std::unique_lock lock{mutex};
    return cv.wait_for(
        lock, std::chrono::seconds{5}, [] { return active.load(std::memory_order_acquire) >= 2U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::uint32_t> active{};
  static inline std::atomic<std::uint32_t> max_active{};
  static inline std::atomic<int> stop_gate_fd{-1};
};

class AutomaticSignalOwnerOperator final : public holoscan::Operator<> {
 public:
  explicit AutomaticSignalOwnerOperator(const bool hold_terminal) : hold_terminal_(hold_terminal) {}

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t current =
        AutomaticSignalOwnerProbeState::active.fetch_add(1U, std::memory_order_acq_rel) + 1U;
    auto maximum = AutomaticSignalOwnerProbeState::max_active.load(std::memory_order_relaxed);
    while (maximum < current && !AutomaticSignalOwnerProbeState::max_active.compare_exchange_weak(
                                    maximum, current, std::memory_order_relaxed)) {
    }
    AutomaticSignalOwnerProbeState::cv.notify_all();
    while (!context.cancellation().stop_requested()) {
      std::this_thread::yield();
    }
    if (hold_terminal_) {
      const int gate_fd =
          AutomaticSignalOwnerProbeState::stop_gate_fd.exchange(-1, std::memory_order_acq_rel);
      if (gate_fd >= 0) {
        std::uint8_t release{};
        [[maybe_unused]] const ssize_t read_count = read(gate_fd, &release, sizeof(release));
      }
    }
    AutomaticSignalOwnerProbeState::active.fetch_sub(1U, std::memory_order_acq_rel);
    return {};
  }

 private:
  bool hold_terminal_{};
};

[[nodiscard]] holoscan::ExecutionPlan make_automatic_multi_owner_signal_plan(
    const std::shared_ptr<ProcessSignalSink>& sink) {
  holoscan::Graph graph{"runtime-managed-signal-automatic-multi-owner"};
  (void)graph.op<AutomaticSignalOwnerOperator>("first", true);
  (void)graph.op<AutomaticSignalOwnerOperator>("second", false);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-managed-signal-clock"));
  auto plan = holoscan::compile(graph);
  return holoscan::detail::RuntimeTestPlanAccess::with_diagnostic_sink(plan, sink);
}

[[nodiscard]] std::size_t count_directory_entries(const char* path) noexcept {
  DIR* const directory = opendir(path);
  if (directory == nullptr) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t count{};
  while (const dirent* const entry = readdir(directory)) {
    if (entry->d_name[0] == '.' &&
        (entry->d_name[1] == '\0' || (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) {
      continue;
    }
    ++count;
  }
  static_cast<void>(closedir(directory));
  return count;
}

[[nodiscard]] bool wait_for_process_thread_count(const std::size_t expected) noexcept {
  // Match the bounded procfs retirement handoff used by managed signal broker cleanup.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
  do {
    if (count_directory_entries("/proc/self/task") == expected) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

[[nodiscard]] bool signal_sets_equal(const sigset_t& left, const sigset_t& right) noexcept {
  for (int signal = 1; signal < NSIG; ++signal) {
    if (sigismember(&left, signal) != sigismember(&right, signal)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool signal_actions_equal(const struct sigaction& left,
                                        const struct sigaction& right) noexcept {
  if (left.sa_flags != right.sa_flags || !signal_sets_equal(left.sa_mask, right.sa_mask)) {
    return false;
  }
  return (left.sa_flags & SA_SIGINFO) != 0 ? left.sa_sigaction == right.sa_sigaction
                                           : left.sa_handler == right.sa_handler;
}

struct ProcessResourceSnapshot {
  std::size_t descriptors{};
  std::size_t threads{};
  sigset_t mask{};
  std::array<struct sigaction, 2> actions{};
  bool valid{};
};

[[nodiscard]] ProcessResourceSnapshot capture_process_resources() noexcept {
  ProcessResourceSnapshot snapshot{
      .descriptors = count_directory_entries("/proc/self/fd"),
      .threads = count_directory_entries("/proc/self/task"),
  };
  snapshot.valid = snapshot.descriptors != std::numeric_limits<std::size_t>::max() &&
                   snapshot.threads != std::numeric_limits<std::size_t>::max() &&
                   pthread_sigmask(SIG_BLOCK, nullptr, &snapshot.mask) == 0 &&
                   sigaction(SIGINT, nullptr, &snapshot.actions[0]) == 0 &&
                   sigaction(SIGTERM, nullptr, &snapshot.actions[1]) == 0;
  return snapshot;
}

[[nodiscard]] bool process_resources_equal(const ProcessResourceSnapshot& expected,
                                           const ProcessResourceSnapshot& actual) noexcept {
  return expected.valid && actual.valid && expected.descriptors == actual.descriptors &&
         expected.threads == actual.threads && signal_sets_equal(expected.mask, actual.mask) &&
         signal_actions_equal(expected.actions[0], actual.actions[0]) &&
         signal_actions_equal(expected.actions[1], actual.actions[1]);
}

[[nodiscard]] bool process_resources_match(const ProcessResourceSnapshot& expected) noexcept {
  if (!expected.valid) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{100};
  for (;;) {
    const ProcessResourceSnapshot actual = capture_process_resources();
    const bool stable_resources = actual.valid && expected.descriptors == actual.descriptors &&
                                  signal_sets_equal(expected.mask, actual.mask) &&
                                  signal_actions_equal(expected.actions[0], actual.actions[0]) &&
                                  signal_actions_equal(expected.actions[1], actual.actions[1]);
    if (!stable_resources) {
      return false;
    }
    if (expected.threads == actual.threads) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
}

[[nodiscard]] bool set_managed_signal_mask(const int how) noexcept {
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  return pthread_sigmask(how, &signals, nullptr) == 0;
}

[[nodiscard]] bool managed_signal_mask_matches(const bool blocked) noexcept {
  sigset_t current;
  if (pthread_sigmask(SIG_BLOCK, nullptr, &current) != 0) {
    return false;
  }
  const int expected = blocked ? 1 : 0;
  return sigismember(&current, SIGINT) == expected && sigismember(&current, SIGTERM) == expected;
}

[[nodiscard]] bool managed_signal_mask_matches(const bool interrupt_blocked,
                                               const bool terminate_blocked) noexcept {
  sigset_t current;
  if (pthread_sigmask(SIG_BLOCK, nullptr, &current) != 0) {
    return false;
  }
  return sigismember(&current, SIGINT) == (interrupt_blocked ? 1 : 0) &&
         sigismember(&current, SIGTERM) == (terminate_blocked ? 1 : 0);
}

enum class ManagedChildMode : std::uint8_t {
  kTransient,
  kAdoptedBlocking,
  kExplicitAsync,
};

[[nodiscard]] int run_managed_child(const ManagedChildMode mode, const int ready_fd,
                                    const bool initially_blocked, const int expected_signal) {
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  if (initially_blocked && !set_managed_signal_mask(SIG_BLOCK)) {
    return 1;
  }
  const ProcessResourceSnapshot resources_before = capture_process_resources();

  const auto sink = std::make_shared<ProcessSignalSink>();
  if (mode == ManagedChildMode::kTransient) {
    const int result = holoscan::run(make_plan(sink));
    const auto detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
    return result != 0 && managed_signal_mask_matches(initially_blocked) &&
                   process_resources_match(resources_before) &&
                   sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U && detail.present &&
                   detail.phase == holoscan::detail::RuntimeManagedSignalPhase::kAcceptSignal &&
                   detail.signal == runtime_signal(expected_signal) &&
                   detail.occurrence == holoscan::detail::RuntimeManagedSignalOccurrence::kFirst &&
                   detail.domain_generation != 0U && detail.binding_generation != 0U
               ? 0
               : 1;
  }

  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  bool stopped{};
  if (mode == ManagedChildMode::kAdoptedBlocking) {
    stopped = holoscan::run(make_plan(sink)) != 0;
  } else {
    {
      auto session = holoscan::run_async(make_plan(sink), signals);
      session.wait();
      stopped = session.termination() == holoscan::RunTermination::kStopped;
    }
  }
  const auto closed = signals.close();
  const auto detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  return stopped && closed && managed_signal_mask_matches(initially_blocked) &&
                 process_resources_match(resources_before) &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U && detail.present &&
                 detail.signal == runtime_signal(expected_signal) &&
                 detail.occurrence == holoscan::detail::RuntimeManagedSignalOccurrence::kFirst &&
                 detail.domain_generation != 0U && detail.binding_generation != 0U
             ? 0
             : 1;
}

[[nodiscard]] int supervise_first_signal(const ManagedChildMode mode, const int signal,
                                         const bool initially_blocked = false) {
  Pipe ready;
  if (!ready.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    const int result = run_managed_child(mode, ready.fd[1], initially_blocked, signal);
    _exit(result);
  }
  ready.close_write();
  if (!read_byte(ready.fd[0]) || kill(child, signal) != 0) {
    static_cast<void>(kill(child, SIGKILL));
  }
  int status{};
  return wait_for_child_exit(child, status) && WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0
                                                                                             : 1;
}

enum class PriorPolicy : std::uint8_t {
  kDefault,
  kIgnored,
  kCustom,
};

[[nodiscard]] bool install_prior_policy(const PriorPolicy policy) noexcept {
  if (policy == PriorPolicy::kDefault) {
    return true;
  }
  struct sigaction action{};
  sigemptyset(&action.sa_mask);
  action.sa_handler = policy == PriorPolicy::kIgnored ? SIG_IGN : &record_forwarded_signal;
  return sigaction(SIGINT, &action, nullptr) == 0 && sigaction(SIGTERM, &action, nullptr) == 0;
}

[[nodiscard]] bool prior_policy_is_unchanged(const PriorPolicy policy) noexcept {
  if (policy == PriorPolicy::kDefault) {
    return true;
  }
  struct sigaction interrupt_action{};
  struct sigaction terminate_action{};
  if (sigaction(SIGINT, nullptr, &interrupt_action) != 0 ||
      sigaction(SIGTERM, nullptr, &terminate_action) != 0) {
    return false;
  }
  const auto expected = policy == PriorPolicy::kIgnored ? SIG_IGN : &record_forwarded_signal;
  return interrupt_action.sa_handler == expected && terminate_action.sa_handler == expected;
}

[[nodiscard]] int run_second_signal_child(const int first, const int second,
                                          const PriorPolicy policy, const int ready_fd,
                                          const int received_fd, const int escalated_fd,
                                          const int stop_gate_fd, const bool persistent_domain,
                                          const bool initially_blocked) {
  if (!install_prior_policy(policy)) {
    return 1;
  }
  if (initially_blocked) {
    sigset_t blocked_signal{};
    sigemptyset(&blocked_signal);
    sigaddset(&blocked_signal, second);
    if (pthread_sigmask(SIG_BLOCK, &blocked_signal, nullptr) != 0) {
      return 1;
    }
  }
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  reset_forwarded_signal_history();
  const pid_t creator_tid = static_cast<pid_t>(syscall(SYS_gettid));
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(stop_gate_fd);
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd, escalated_fd);
  bool stopped{};
  bool closed{true};
  if (persistent_domain) {
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
      return 1;
    }
    auto signals = std::move(created).value();
    {
      auto session = holoscan::run_async(make_plan(sink), signals);
      session.wait();
      stopped = session.termination() == holoscan::RunTermination::kStopped;
    }
    closed = static_cast<bool>(signals.close());
  } else {
    stopped = holoscan::run(make_plan(sink)) != 0;
  }
  bool blocked_signal_was_pending{};
  bool blocked_signal_consumed{};
  if (initially_blocked && closed) {
    sigset_t pending{};
    sigset_t blocked_signal{};
    sigemptyset(&blocked_signal);
    sigaddset(&blocked_signal, second);
    blocked_signal_was_pending =
        sigpending(&pending) == 0 && sigismember(&pending, second) == 1 && g_forwarded_count == 0;
    const timespec timeout{};
    blocked_signal_consumed = sigtimedwait(&blocked_signal, nullptr, &timeout) == second;
  }
  if (policy == PriorPolicy::kDefault) {
    return 1;
  }
  const auto expected_disposition =
      policy == PriorPolicy::kIgnored ? holoscan::detail::RuntimeManagedSignalDisposition::kIgnored
                                      : holoscan::detail::RuntimeManagedSignalDisposition::kCustom;
  const bool policy_effect =
      policy == PriorPolicy::kIgnored ||
      (initially_blocked
           ? blocked_signal_was_pending && blocked_signal_consumed && g_forwarded_count == 0
           : g_forwarded_signal == second && g_forwarded_tid != 0 &&
                 (!persistent_domain || g_forwarded_tid != creator_tid));
  const auto received_detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const auto escalated_detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_ESCALATED");
  const bool passed =
      stopped && closed && policy_effect && prior_policy_is_unchanged(policy) &&
      process_resources_match(resources_before) &&
      sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U &&
      sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED") == 1U &&
      received_detail.signal == runtime_signal(first) &&
      received_detail.occurrence == holoscan::detail::RuntimeManagedSignalOccurrence::kFirst &&
      received_detail.phase == holoscan::detail::RuntimeManagedSignalPhase::kAcceptSignal &&
      received_detail.prior_disposition == expected_disposition &&
      !received_detail.originally_blocked && escalated_detail.signal == runtime_signal(second) &&
      escalated_detail.occurrence == holoscan::detail::RuntimeManagedSignalOccurrence::kSecond &&
      escalated_detail.phase == holoscan::detail::RuntimeManagedSignalPhase::kDelegateSignal &&
      escalated_detail.prior_disposition == expected_disposition &&
      escalated_detail.originally_blocked == initially_blocked &&
      received_detail.domain_generation == escalated_detail.domain_generation &&
      received_detail.binding_generation == escalated_detail.binding_generation &&
      received_detail.domain_generation != 0U && received_detail.binding_generation != 0U &&
      sink->token_count("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED") == 0U &&
      sink->token_count("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED") == 0U;
  if (!passed) {
    std::fprintf(stderr,
                 "second-signal mismatch: stopped=%d closed=%d policy_effect=%d policy=%u "
                 "forwarded=%d count=%d tid=%d creator=%d pending=%d consumed=%d "
                 "received=%zu escalated=%zu received_blocked=%d escalated_blocked=%d "
                 "received_prior=%u escalated_prior=%u wait=%zu restore=%zu resources=%d\n",
                 static_cast<int>(stopped),
                 static_cast<int>(closed),
                 static_cast<int>(policy_effect),
                 static_cast<unsigned>(policy),
                 static_cast<int>(g_forwarded_signal),
                 static_cast<int>(g_forwarded_count),
                 static_cast<int>(g_forwarded_tid),
                 static_cast<int>(creator_tid),
                 static_cast<int>(blocked_signal_was_pending),
                 static_cast<int>(blocked_signal_consumed),
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED"),
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED"),
                 static_cast<int>(received_detail.originally_blocked),
                 static_cast<int>(escalated_detail.originally_blocked),
                 static_cast<unsigned>(received_detail.prior_disposition),
                 static_cast<unsigned>(escalated_detail.prior_disposition),
                 sink->token_count("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED"),
                 sink->token_count("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED"),
                 static_cast<int>(process_resources_match(resources_before)));
  }
  return passed ? 0 : 1;
}

[[nodiscard]] int supervise_second_signal(const int first, const int second,
                                          const PriorPolicy policy,
                                          const bool persistent_domain = false,
                                          const bool initially_blocked = false) {
  Pipe ready;
  Pipe received;
  Pipe escalated;
  Pipe stop_gate;
  if (!ready.open() || !received.open() || !escalated.open() || !stop_gate.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    escalated.close_read();
    stop_gate.close_write();
    const int result = run_second_signal_child(first,
                                               second,
                                               policy,
                                               ready.fd[1],
                                               received.fd[1],
                                               escalated.fd[1],
                                               stop_gate.fd[0],
                                               persistent_domain,
                                               initially_blocked);
    _exit(result);
  }
  ready.close_write();
  received.close_write();
  escalated.close_write();
  stop_gate.close_read();
  bool protocol_ok = read_byte(ready.fd[0]) && kill(child, first) == 0 &&
                     read_byte(received.fd[0]) && kill(child, second) == 0 &&
                     read_byte(escalated.fd[0]);
  if (policy != PriorPolicy::kDefault) {
    protocol_ok = write_byte(stop_gate.fd[1]) && protocol_ok;
  }
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  if (policy == PriorPolicy::kDefault) {
    return WIFSIGNALED(status) && WTERMSIG(status) == second ? 0 : 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_automatic_multi_owner_signal_child(const int ready_fd, const int received_fd,
                                                         const int escalated_fd,
                                                         const int stop_gate_fd) {
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  reset_forwarded_signal_history();
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    std::fprintf(stderr,
                 "automatic multi-owner signal domain creation failed: error=%u\n",
                 static_cast<unsigned>(created.error().code));
    return 1;
  }
  auto signals = std::move(created).value();
  AutomaticSignalOwnerProbeState::reset(stop_gate_fd);
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd, escalated_fd);
  const holoscan::ExecutionPlan plan = make_automatic_multi_owner_signal_plan(sink);
  if (!plan.ok() || plan.partitioning_plan().mode != holoscan::PartitioningMode::kAutomatic ||
      plan.partitioning_plan().authored_partition_count != 0U ||
      plan.partitioning_plan().automatic_partition_count != 2U ||
      plan.execution_partitions().size() != 2U || plan.execution_lanes().size() != 2U) {
    std::fprintf(stderr,
                 "automatic multi-owner plan invalid: ok=%d mode=%u authored=%u automatic=%u "
                 "partitions=%zu lanes=%zu diagnostic=%s\n",
                 plan.ok(),
                 static_cast<unsigned>(plan.partitioning_plan().mode),
                 plan.partitioning_plan().authored_partition_count,
                 plan.partitioning_plan().automatic_partition_count,
                 plan.execution_partitions().size(),
                 plan.execution_lanes().size(),
                 plan.diagnostics().empty() ? "none" : plan.diagnostics().front().token.c_str());
    return 1;
  }

  bool stopped{};
  bool two_owners_executed{};
  {
    auto session = holoscan::run_async(plan, signals);
    if (!AutomaticSignalOwnerProbeState::wait_for_two_active() || !write_byte(ready_fd)) {
      std::fprintf(stderr,
                   "automatic multi-owner callbacks did not become active: active=%u max=%u\n",
                   AutomaticSignalOwnerProbeState::active.load(std::memory_order_acquire),
                   AutomaticSignalOwnerProbeState::max_active.load(std::memory_order_acquire));
      return 1;
    }
    session.wait();
    stopped = session.termination() == holoscan::RunTermination::kStopped;
    two_owners_executed =
        AutomaticSignalOwnerProbeState::max_active.load(std::memory_order_acquire) == 2U;
  }
  const auto received_detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const auto escalated_detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_ESCALATED");
  const bool closed = static_cast<bool>(signals.close());
  const bool policy_restored = prior_policy_is_unchanged(PriorPolicy::kCustom);
  const bool resources_restored = process_resources_match(resources_before);
  const std::size_t received_count = sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const std::size_t escalated_count = sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED");
  const bool passed =
      stopped && two_owners_executed && closed && g_forwarded_signal == SIGTERM &&
      policy_restored && resources_restored && received_count == 1U && escalated_count == 1U &&
      received_detail.present && escalated_detail.present &&
      received_detail.signal == holoscan::detail::RuntimeManagedSignal::kSigInt &&
      received_detail.occurrence == holoscan::detail::RuntimeManagedSignalOccurrence::kFirst &&
      received_detail.phase == holoscan::detail::RuntimeManagedSignalPhase::kAcceptSignal &&
      escalated_detail.signal == holoscan::detail::RuntimeManagedSignal::kSigTerm &&
      escalated_detail.occurrence == holoscan::detail::RuntimeManagedSignalOccurrence::kSecond &&
      escalated_detail.phase == holoscan::detail::RuntimeManagedSignalPhase::kDelegateSignal &&
      received_detail.domain_generation == escalated_detail.domain_generation &&
      received_detail.binding_generation == escalated_detail.binding_generation &&
      received_detail.domain_generation != 0U && received_detail.binding_generation != 0U;
  if (!passed) {
    std::fprintf(stderr,
                 "automatic multi-owner signal evidence failed: stopped=%d owners=%d closed=%d "
                 "forwarded=%d policy=%d resources=%d received=%zu escalated=%zu "
                 "received_present=%d escalated_present=%d\n",
                 stopped,
                 two_owners_executed,
                 closed,
                 static_cast<int>(g_forwarded_signal),
                 policy_restored,
                 resources_restored,
                 received_count,
                 escalated_count,
                 received_detail.present,
                 escalated_detail.present);
  }
  return passed ? 0 : 1;
}

enum class AutomaticMultiOwnerSignalExpectation : std::uint8_t {
  kEscalated,
  kChildFailureAfterFirstSignal,
};

[[nodiscard]] int supervise_automatic_multi_owner_signals(
    const AutomaticMultiOwnerSignalExpectation expectation) {
  Pipe ready;
  Pipe received;
  Pipe escalated;
  Pipe stop_gate;
  if (!ready.open() || !received.open() || !escalated.open() || !stop_gate.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    escalated.close_read();
    stop_gate.close_write();
    const int result = run_automatic_multi_owner_signal_child(
        ready.fd[1], received.fd[1], escalated.fd[1], stop_gate.fd[0]);
    _exit(result);
  }
  ready.close_write();
  received.close_write();
  escalated.close_write();
  stop_gate.close_read();
  bool protocol_ok =
      read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 && read_byte(received.fd[0]);
  if (protocol_ok && expectation == AutomaticMultiOwnerSignalExpectation::kEscalated) {
    protocol_ok = kill(child, SIGTERM) == 0 && read_byte(escalated.fd[0]);
  }

  // Closing the write end releases the held terminal callback on both the
  // successful protocol and every parent-side failure path. A byte is not
  // required: the child treats either data or EOF as the release boundary.
  stop_gate.close_write();
  int status{};
  if (!wait_for_child_exit_for(child, status, std::chrono::seconds{10})) {
    std::fprintf(stderr, "automatic multi-owner signal child did not exit after gate release\n");
    static_cast<void>(kill(child, SIGKILL));
    static_cast<void>(wait_for_child_exit_for(child, status, std::chrono::seconds{1}));
    return 1;
  }
  if (!protocol_ok) {
    std::fprintf(stderr, "automatic multi-owner parent protocol failed before child exit\n");
    return 1;
  }
  if (!WIFEXITED(status)) {
    std::fprintf(
        stderr, "automatic multi-owner signal child terminated abnormally: status=%d\n", status);
    return 1;
  }
  const int child_result = WEXITSTATUS(status);
  if (expectation == AutomaticMultiOwnerSignalExpectation::kChildFailureAfterFirstSignal) {
    if (child_result != 1) {
      std::fprintf(stderr,
                   "automatic multi-owner signal child failure was not explicit: result=%d\n",
                   child_result);
      return 1;
    }
    return 0;
  }
  if (child_result != 0) {
    std::fprintf(stderr, "automatic multi-owner signal child failed: result=%d\n", child_result);
    return 1;
  }
  return 0;
}

[[nodiscard]] int run_third_signal_after_escalation_child(const int ready_fd, const int received_fd,
                                                          const int escalated_fd,
                                                          const int forwarded_fd,
                                                          const int stop_gate_fd) {
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  reset_forwarded_signal_history();
  g_forwarded_fd = forwarded_fd;
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    g_forwarded_fd = -1;
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(stop_gate_fd);
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd, escalated_fd);
  auto session = holoscan::run_async(make_plan(sink), signals);
  session.wait();
  const auto closed = signals.close();
  g_forwarded_fd = -1;
  return session.termination() == holoscan::RunTermination::kStopped && closed &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED") == 1U &&
                 g_forwarded_count == 2 && g_forwarded_history[0] == SIGTERM &&
                 g_forwarded_history[1] == SIGINT && prior_policy_is_unchanged(PriorPolicy::kCustom)
             ? 0
             : 1;
}

[[nodiscard]] int supervise_third_signal_after_escalation() {
  Pipe ready;
  Pipe received;
  Pipe escalated;
  Pipe forwarded;
  Pipe stop_gate;
  if (!ready.open() || !received.open() || !escalated.open() || !forwarded.open() ||
      !stop_gate.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    escalated.close_read();
    forwarded.close_read();
    stop_gate.close_write();
    _exit(run_third_signal_after_escalation_child(
        ready.fd[1], received.fd[1], escalated.fd[1], forwarded.fd[1], stop_gate.fd[0]));
  }
  ready.close_write();
  received.close_write();
  escalated.close_write();
  forwarded.close_write();
  stop_gate.close_read();
  const bool protocol_ok =
      read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 && read_byte(received.fd[0]) &&
      kill(child, SIGTERM) == 0 && read_byte(escalated.fd[0]) && read_byte(forwarded.fd[0]) &&
      kill(child, SIGINT) == 0 && read_byte(forwarded.fd[0]) && write_byte(stop_gate.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_asymmetric_mask_child(const int ready_fd) {
  sigset_t interrupt;
  sigemptyset(&interrupt);
  sigaddset(&interrupt, SIGINT);
  if (pthread_sigmask(SIG_BLOCK, &interrupt, nullptr) != 0) {
    return 1;
  }
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  return holoscan::run(make_plan()) != 0 && managed_signal_mask_matches(true, false) ? 0 : 1;
}

[[nodiscard]] int run_first_custom_child(const int ready_fd) {
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  g_forwarded_signal = 0;
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  const int result = holoscan::run(make_plan());
  return result != 0 && g_forwarded_signal == 0 && prior_policy_is_unchanged(PriorPolicy::kCustom)
             ? 0
             : 1;
}

[[nodiscard]] int supervise_one_signal_child(const int signal, int (*child_main)(int)) {
  Pipe ready;
  if (!ready.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    _exit(child_main(ready.fd[1]));
  }
  ready.close_write();
  const bool protocol_ok = read_byte(ready.fd[0]) && kill(child, signal) == 0;
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_domain_required() {
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto plan = make_plan(sink);
  std::atomic_bool release{};
  std::thread peer([&] {
    while (!release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  });
  const int result = holoscan::run(plan);
  release.store(true, std::memory_order_release);
  peer.join();
  return result != 0 && !holoscan::test::RuntimeProbeState::compute_entered.load() &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED") &&
                 sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kThreadCensus,
                     0,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kMultipleThreads,
                     false)
             ? 0
             : 1;
}

[[nodiscard]] int run_domain_adopt_spent_retry() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const int adopted_result = holoscan::run(make_plan());

  holoscan::test::RuntimeProbeState::reset();
  const auto spent_sink = std::make_shared<ProcessSignalSink>();
  const int spent_result = holoscan::run(make_plan(spent_sink));
  const bool spent_rejected =
      spent_result != 0 && !holoscan::test::RuntimeProbeState::compute_entered.load() &&
      spent_sink->contains("RUNTIME_MANAGED_SIGNAL_OWNER_BUSY") &&
      spent_sink->contains_terminal_detail(
          holoscan::detail::RuntimeManagedSignalPhase::kAcquireDomain,
          0,
          holoscan::detail::RuntimeManagedSignalCensusReason::kIncompatibleDomain,
          false,
          true);
  const auto closed = signals.close();

  auto recreated = holoscan::ManagedShutdownSignals::create();
  if (!recreated) {
    return 1;
  }
  auto fresh = std::move(recreated).value();
  holoscan::test::RuntimeProbeState::reset();
  const int fresh_result = holoscan::run(make_plan());
  const auto fresh_closed = fresh.close();
  return adopted_result == 0 && spent_rejected && closed && fresh_result == 0 && fresh_closed &&
                 process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int run_caller_managed_with_peer() {
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto plan = make_plan(sink);
  std::atomic_bool release{};
  std::thread peer([&] {
    while (!release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  });
  const int result = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
  release.store(true, std::memory_order_release);
  peer.join();
  return result == 0 && holoscan::test::RuntimeProbeState::compute_entered.load() &&
                 !sink->contains_managed_signal_record()
             ? 0
             : 1;
}

[[nodiscard]] int run_plain_async_neutrality() {
  struct sigaction action{};
  struct sigaction prior{};
  sigemptyset(&action.sa_mask);
  action.sa_handler = &record_forwarded_signal;
  if (sigaction(SIGINT, &action, &prior) != 0) {
    return 1;
  }
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    static_cast<void>(sigaction(SIGINT, &prior, nullptr));
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  bool neutral{};
  {
    auto session = holoscan::run_async(make_plan());
    if (holoscan::test::RuntimeProbeState::wait_until_compute_enters() &&
        kill(getpid(), SIGINT) == 0) {
      for (std::size_t spin = 0U; spin < 10'000'000U && g_forwarded_signal != SIGINT; ++spin) {
        std::this_thread::yield();
      }
      neutral = g_forwarded_signal == SIGINT && session.is_running();
    }
    session.request_stop();
    session.wait();
  }
  const auto closed = signals.close();
  const bool restored =
      sigaction(SIGINT, nullptr, &action) == 0 && action.sa_handler == &record_forwarded_signal;
  static_cast<void>(sigaction(SIGINT, &prior, nullptr));
  return neutral && closed && restored ? 0 : 1;
}

[[nodiscard]] bool session_has_diagnostic(const holoscan::RunSession& session,
                                          const std::string_view token) {
  for (const auto& diagnostic : session.diagnostics()) {
    if (diagnostic.token == token) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] holoscan::ExecutionPlan make_failure_plan(
    const std::shared_ptr<ProcessSignalSink>& sink,
    const std::shared_ptr<holoscan::detail::RuntimeManagedSignalTestControl>& control);

[[nodiscard]] int run_close_after_wait() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  auto session = holoscan::run_async(make_plan(), signals);
  session.wait();
  return session.termination() == holoscan::RunTermination::kCompleted && signals.close() ? 0 : 1;
}

[[nodiscard]] bool wait_for_control_hits(
    const std::shared_ptr<holoscan::detail::RuntimeManagedSignalTestControl>& control,
    const std::uint32_t expected) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (control->hits() < expected) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

[[nodiscard]] int run_bound_event_driven_idle_child(const int ready_fd, const int release_fd) {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(
      holoscan::test::RuntimeProbeMode::kWaitForExternalRelease);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(release_fd);
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kBoundWaitForActivity, 0);
  auto session = holoscan::run_async(make_failure_plan({}, control), signals);
  const bool idle_wait_entered = holoscan::test::RuntimeProbeState::wait_until_compute_enters() &&
                                 wait_for_control_hits(control, 1U) && write_byte(ready_fd);
  session.wait();
  const auto closed = signals.close();
  return idle_wait_entered && session.termination() == holoscan::RunTermination::kCompleted &&
                 control->hits() == 1U && closed && process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int supervise_bound_event_driven_idle() {
  Pipe ready;
  Pipe release;
  if (!ready.open() || !release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    release.close_write();
    _exit(run_bound_event_driven_idle_child(ready.fd[1], release.fd[0]));
  }
  ready.close_write();
  release.close_read();
  const bool entered = read_byte(ready.fd[0]);
  std::this_thread::sleep_for(std::chrono::milliseconds{350});
  const bool released = entered && write_byte(release.fd[1]);
  int status{};
  if (!released || !wait_for_child_exit_for(child, status, std::chrono::seconds{10})) {
    static_cast<void>(kill(child, SIGKILL));
    static_cast<void>(wait_for_child_exit(child, status));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_unbound_event_driven_idle() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kUnboundWaitForActivity, 0);
  auto session = holoscan::run_async(make_failure_plan({}, control), signals);
  session.wait();
  const bool entered_unbound_wait = wait_for_control_hits(control, 1U);
  const std::uint32_t visits = control->hits();
  std::this_thread::sleep_for(std::chrono::milliseconds{350});
  const bool wait_stable = control->hits() == visits;
  const auto closed = signals.close();
  return session.termination() == holoscan::RunTermination::kCompleted && entered_unbound_wait &&
                 visits == 1U && wait_stable && closed && process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int run_bound_wait_failure_no_spin() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  Pipe release;
  if (!release.open()) {
    return 1;
  }
  holoscan::test::RuntimeProbeState::reset(
      holoscan::test::RuntimeProbeMode::kIgnoreStopUntilReleased);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(release.fd[0]);
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kBoundWaitForActivity, EIO);
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  const bool failed_once = wait_for_control_hits(control, 1U);
  std::this_thread::sleep_for(std::chrono::milliseconds{350});
  const bool stable = control->hits() == 1U;
  const bool released = write_byte(release.fd[1]);
  session.wait();
  const auto closed = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  release.close_read();
  release.close_write();
  const bool passed =
      failed_once && stable && released &&
      session.termination() == holoscan::RunTermination::kFailed && !closed &&
      closed.error().code == holoscan::ErrorCode::kFailure && !competing_create &&
      competing_create.error().code == holoscan::ErrorCode::kResourceExhausted &&
      sink->token_count("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED") == 1U &&
      sink->contains_terminal_detail(holoscan::detail::RuntimeManagedSignalPhase::kPoll,
                                     EIO,
                                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
      process_resources_match(resources_before);
  if (!passed) {
    std::fprintf(stderr,
                 "bound wait mismatch: failed_once=%d stable=%d released=%d hits=%u term=%u "
                 "closed=%d close_code=%u wait=%zu detail=%d resources=%d\n",
                 static_cast<int>(failed_once),
                 static_cast<int>(stable),
                 static_cast<int>(released),
                 control->hits(),
                 static_cast<unsigned>(session.termination()),
                 static_cast<int>(static_cast<bool>(closed)),
                 closed ? 0U : static_cast<unsigned>(closed.error().code),
                 sink->token_count("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED"),
                 static_cast<int>(sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kPoll,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone)),
                 static_cast<int>(process_resources_match(resources_before)));
  }
  return passed ? 0 : 1;
}

[[nodiscard]] int run_unbound_wait_failure_retires() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kUnboundWaitForActivity, EIO);
  auto session = holoscan::run_async(make_failure_plan({}, control), signals);
  session.wait();
  const bool failed_once = wait_for_control_hits(control, 1U);
  std::this_thread::sleep_for(std::chrono::milliseconds{350});
  const bool stable = control->hits() == 1U;
  const auto closed = signals.close();
  return session.termination() == holoscan::RunTermination::kCompleted && failed_once && stable &&
                 !closed && closed.error().code == holoscan::ErrorCode::kFailure &&
                 process_resources_match(resources_before)
             ? 0
             : 1;
}

struct PollCheckpointState {
  int reached_fd{-1};
  int release_fd{-1};
};

void block_before_poll(const holoscan::detail::RuntimeManagedSignalTestPoint,
                       void* state) noexcept {
  auto* const checkpoint = static_cast<PollCheckpointState*>(state);
  if (checkpoint == nullptr || !write_byte(checkpoint->reached_fd)) {
    return;
  }
  static_cast<void>(read_byte(checkpoint->release_fd));
}

[[nodiscard]] int run_terminal_wake_before_close_child(const int candidate_reached_fd,
                                                       const int candidate_release_fd,
                                                       const int close_reached_fd,
                                                       const int close_release_fd) {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kComplete);
  PollCheckpointState candidate_checkpoint{.reached_fd = candidate_reached_fd,
                                           .release_fd = candidate_release_fd};
  PollCheckpointState close_checkpoint{.reached_fd = close_reached_fd,
                                       .release_fd = close_release_fd};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kCandidatePublishedBeforeWake,
      &block_before_poll,
      &candidate_checkpoint,
      holoscan::detail::RuntimeManagedSignalTestPoint::kBeforeControlFdClose,
      &block_before_poll,
      &close_checkpoint);
  const auto sink = std::make_shared<ProcessSignalSink>();
  const int result = holoscan::run(make_failure_plan(sink, control));
  return result != 0 && sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U &&
                 !sink->contains("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED") && control->hits() == 2U &&
                 process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int supervise_terminal_wake_before_close() {
  Pipe candidate_reached;
  Pipe candidate_release;
  Pipe close_reached;
  Pipe close_release;
  if (!candidate_reached.open() || !candidate_release.open() || !close_reached.open() ||
      !close_release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    candidate_reached.close_read();
    candidate_release.close_write();
    close_reached.close_read();
    close_release.close_write();
    _exit(run_terminal_wake_before_close_child(candidate_reached.fd[1],
                                               candidate_release.fd[0],
                                               close_reached.fd[1],
                                               close_release.fd[0]));
  }
  candidate_reached.close_write();
  candidate_release.close_read();
  close_reached.close_write();
  close_release.close_read();
  const auto kill_and_reap = [&]() {
    int status{};
    static_cast<void>(kill(child, SIGKILL));
    static_cast<void>(wait_for_child_exit(child, status));
  };
  if (!read_byte(candidate_reached.fd[0]) || kill(child, SIGINT) != 0) {
    kill_and_reap();
    return 1;
  }

  // Before the fix, the controller can observe the candidate and start closing the eventfd while
  // the worker is still blocked before its wake. The fixed ordering prevents that early close.
  const bool close_started_early =
      byte_ready_for(close_reached.fd[0], std::chrono::milliseconds{300});
  if (close_started_early) {
    static_cast<void>(read_byte(close_reached.fd[0]));
    static_cast<void>(write_byte(close_release.fd[1]));
    static_cast<void>(write_byte(candidate_release.fd[1]));
  } else if (!write_byte(candidate_release.fd[1]) || !read_byte(close_reached.fd[0]) ||
             !write_byte(close_release.fd[1])) {
    kill_and_reap();
    return 1;
  }

  int status{};
  if (!wait_for_child_exit_for(child, status, std::chrono::seconds{5})) {
    kill_and_reap();
    return 1;
  }
  return !close_started_early && WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_blocked_notify_failure_child(const int ready_fd, const int compute_release_fd,
                                                   const int poll_reached_fd,
                                                   const int poll_release_fd) {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(
      holoscan::test::RuntimeProbeMode::kWaitForExternalRelease);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(compute_release_fd);
  PollCheckpointState checkpoint{.reached_fd = poll_reached_fd, .release_fd = poll_release_fd};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kNotifyController,
      EIO,
      3U,
      holoscan::detail::RuntimeManagedSignalTestPoint::kBoundWaitForActivity,
      &block_before_poll,
      &checkpoint);
  const auto sink = std::make_shared<ProcessSignalSink>();
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto closed = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return session.termination() == holoscan::RunTermination::kFailed && !closed &&
                 closed.error().code == holoscan::ErrorCode::kFailure && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted &&
                 sink->token_count("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED") == 1U &&
                 sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kNotifyController,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
                 process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int supervise_blocked_notify_failure() {
  Pipe ready;
  Pipe compute_release;
  Pipe poll_reached;
  Pipe poll_release;
  if (!ready.open() || !compute_release.open() || !poll_reached.open() || !poll_release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    compute_release.close_write();
    poll_reached.close_read();
    poll_release.close_write();
    _exit(run_blocked_notify_failure_child(
        ready.fd[1], compute_release.fd[0], poll_reached.fd[1], poll_release.fd[0]));
  }
  ready.close_write();
  compute_release.close_read();
  poll_reached.close_write();
  poll_release.close_read();
  const auto kill_and_reap = [&]() {
    int status{};
    static_cast<void>(kill(child, SIGKILL));
    static_cast<void>(wait_for_child_exit(child, status));
  };
  if (!read_byte(poll_reached.fd[0])) {
    kill_and_reap();
    return 1;
  }
  if (!read_byte(ready.fd[0]) || !write_byte(compute_release.fd[1])) {
    kill_and_reap();
    return 1;
  }
  const bool released_poll = write_byte(poll_release.fd[1]);
  int status{};
  if (!released_poll || !wait_for_child_exit_for(child, status, std::chrono::seconds{5})) {
    kill_and_reap();
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_wrong_thread_close() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::ErrorCode peer_code{holoscan::ErrorCode::kSuccess};
  std::thread peer([&] {
    const auto closed = signals.close();
    peer_code = closed ? holoscan::ErrorCode::kSuccess : closed.error().code;
  });
  peer.join();
  const auto closed = signals.close();
  return peer_code == holoscan::ErrorCode::kInvalidArgument && closed ? 0 : 1;
}

[[nodiscard]] int run_managed_owner_busy() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  auto first = holoscan::run_async(make_plan(), signals);
  if (!holoscan::test::RuntimeProbeState::wait_until_compute_enters()) {
    first.request_stop();
    first.wait();
    return 1;
  }
  auto competing = holoscan::run_async(make_plan(), signals);
  competing.wait();
  const bool competing_rejected =
      competing.termination() == holoscan::RunTermination::kFailed &&
      session_has_diagnostic(competing, "RUNTIME_MANAGED_SIGNAL_OWNER_BUSY");
  first.request_stop();
  first.wait();
  auto reused = holoscan::run_async(make_plan(), signals);
  reused.wait();
  const bool one_shot_rejected =
      reused.termination() == holoscan::RunTermination::kFailed &&
      session_has_diagnostic(reused, "RUNTIME_MANAGED_SIGNAL_OWNER_BUSY");
  return competing_rejected && one_shot_rejected && signals.close() ? 0 : 1;
}

[[nodiscard]] int run_caller_managed_inside_domain_child(const int ready_fd, const int forwarded_fd,
                                                         const int release_fd) {
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  g_forwarded_signal = 0;
  g_forwarded_fd = forwarded_fd;
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    g_forwarded_fd = -1;
    return 1;
  }
  auto signals = std::move(created).value();
  const auto sink = std::make_shared<ProcessSignalSink>();
  holoscan::test::RuntimeProbeState::reset(
      holoscan::test::RuntimeProbeMode::kWaitForExternalRelease);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(release_fd);
  const int result = holoscan::run(make_plan(sink), holoscan::RunSignalPolicy::kCallerManaged);
  const bool forwarded = g_forwarded_signal == SIGINT;
  g_forwarded_fd = -1;
  return result == 0 && forwarded && !sink->contains_managed_signal_record() && signals.close() ? 0
                                                                                                : 1;
}

[[nodiscard]] int supervise_caller_managed_inside_domain() {
  Pipe ready;
  Pipe forwarded;
  Pipe release;
  if (!ready.open() || !forwarded.open() || !release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    forwarded.close_read();
    release.close_write();
    _exit(run_caller_managed_inside_domain_child(ready.fd[1], forwarded.fd[1], release.fd[0]));
  }
  ready.close_write();
  forwarded.close_write();
  release.close_read();
  const bool protocol_ok = read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 &&
                           read_byte(forwarded.fd[0]) && write_byte(release.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_wrong_thread_bind() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  std::atomic_bool rejected{};
  std::thread peer([&] {
    holoscan::test::RuntimeProbeState::reset();
    const auto sink = std::make_shared<ProcessSignalSink>();
    auto session = holoscan::run_async(make_plan(sink), signals);
    session.wait();
    rejected.store(session.termination() == holoscan::RunTermination::kFailed &&
                       session_has_diagnostic(session, "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") &&
                       sink->contains_terminal_detail(
                           holoscan::detail::RuntimeManagedSignalPhase::kAcquireDomain,
                           0,
                           holoscan::detail::RuntimeManagedSignalCensusReason::kWrongThread,
                           false),
                   std::memory_order_release);
  });
  peer.join();
  return rejected.load(std::memory_order_acquire) && signals.close() ? 0 : 1;
}

[[nodiscard]] int run_fork_rejection() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    holoscan::test::RuntimeProbeState::reset();
    const auto sink = std::make_shared<ProcessSignalSink>();
    auto session = holoscan::run_async(make_plan(sink), signals);
    session.wait();
    _exit(session.termination() == holoscan::RunTermination::kFailed &&
                  session_has_diagnostic(session, "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") &&
                  sink->contains_terminal_detail(
                      holoscan::detail::RuntimeManagedSignalPhase::kAcquireDomain,
                      0,
                      holoscan::detail::RuntimeManagedSignalCensusReason::kForkedProcess,
                      false,
                      false)
              ? 0
              : 1);
  }
  int status{};
  const bool child_rejected =
      wait_for_child_exit(child, status) && WIFEXITED(status) && WEXITSTATUS(status) == 0;
  return child_rejected && signals.close() ? 0 : 1;
}

[[nodiscard]] int run_exec_mask_observer(const bool expected_blocked,
                                         const ManagedDescriptorCounts expected_counts) {
  sigset_t current{};
  if (pthread_sigmask(SIG_BLOCK, nullptr, &current) != 0) {
    return 1;
  }
  const bool int_blocked = sigismember(&current, SIGINT) == 1;
  const bool term_blocked = sigismember(&current, SIGTERM) == 1;
  const ManagedDescriptorCounts counts = managed_descriptor_counts();
  return int_blocked == expected_blocked && term_blocked == expected_blocked &&
                 counts.signal == expected_counts.signal && counts.event == expected_counts.event
             ? 0
             : 1;
}

[[nodiscard]] bool wait_for_successful_child(const pid_t child) noexcept {
  int status{};
  return child > 0 && wait_for_child_exit_for(child, status, std::chrono::seconds{5}) &&
         WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

[[nodiscard]] pid_t fork_exec_mask_observer(
    const bool reset_mask, const ManagedDescriptorCounts expected_counts) noexcept {
  sigset_t managed{};
  sigemptyset(&managed);
  sigaddset(&managed, SIGINT);
  sigaddset(&managed, SIGTERM);
  std::array<char, 32> signal_count{};
  std::array<char, 32> event_count{};
  static_cast<void>(
      std::snprintf(signal_count.data(), signal_count.size(), "%zu", expected_counts.signal));
  static_cast<void>(
      std::snprintf(event_count.data(), event_count.size(), "%zu", expected_counts.event));
  const pid_t child = fork();
  if (child != 0) {
    return child;
  }
  if (reset_mask) {
    if (sigprocmask(SIG_UNBLOCK, &managed, nullptr) != 0) {
      _exit(1);
    }
  }
  execl("/proc/self/exe",
        "runtime_signal_process_probe",
        reset_mask ? "exec-mask-observer-unblocked" : "exec-mask-observer-blocked",
        signal_count.data(),
        event_count.data(),
        nullptr);
  _exit(1);
}

[[nodiscard]] pid_t spawn_exec_mask_observer_with_empty_mask(
    const ManagedDescriptorCounts expected_counts) noexcept {
  posix_spawnattr_t attributes{};
  sigset_t empty{};
  sigemptyset(&empty);
  if (posix_spawnattr_init(&attributes) != 0) {
    return -1;
  }
  if (posix_spawnattr_setsigmask(&attributes, &empty) != 0 ||
      posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK) != 0) {
    static_cast<void>(posix_spawnattr_destroy(&attributes));
    return -1;
  }
  auto executable = std::to_array("/proc/self/exe");
  auto scenario = std::to_array("exec-mask-observer-unblocked");
  std::array<char, 32> signal_count{};
  std::array<char, 32> event_count{};
  static_cast<void>(
      std::snprintf(signal_count.data(), signal_count.size(), "%zu", expected_counts.signal));
  static_cast<void>(
      std::snprintf(event_count.data(), event_count.size(), "%zu", expected_counts.event));
  const std::array<char*, 5> arguments{
      executable.data(), scenario.data(), signal_count.data(), event_count.data(), nullptr};
  pid_t child{-1};
  const int result =
      posix_spawn(&child, executable.data(), nullptr, &attributes, arguments.data(), environ);
  static_cast<void>(posix_spawnattr_destroy(&attributes));
  return result == 0 ? child : -1;
}

[[nodiscard]] int run_exec_mask_inheritance_and_remedies() {
  sigset_t managed{};
  sigset_t prior{};
  sigemptyset(&managed);
  sigaddset(&managed, SIGINT);
  sigaddset(&managed, SIGTERM);
  if (pthread_sigmask(SIG_UNBLOCK, &managed, &prior) != 0) {
    return 1;
  }
  const ManagedDescriptorCounts baseline_counts = managed_descriptor_counts();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    static_cast<void>(pthread_sigmask(SIG_SETMASK, &prior, nullptr));
    return 1;
  }
  auto signals = std::move(created).value();
  const ManagedDescriptorCounts domain_counts = managed_descriptor_counts();
  const bool descriptors_created = domain_counts.signal == baseline_counts.signal + 1U &&
                                   domain_counts.event == baseline_counts.event + 1U;
  const bool inherited = wait_for_successful_child(fork_exec_mask_observer(false, baseline_counts));
  const bool fork_reset = wait_for_successful_child(fork_exec_mask_observer(true, baseline_counts));
  const bool spawn_reset =
      wait_for_successful_child(spawn_exec_mask_observer_with_empty_mask(baseline_counts));
  const auto closed = signals.close();
  const bool restored = pthread_sigmask(SIG_SETMASK, &prior, nullptr) == 0;
  return descriptors_created && inherited && fork_reset && spawn_reset && closed && restored ? 0
                                                                                             : 1;
}

enum class ForkedRegistryOperation : std::uint8_t {
  kClose,
  kDestroy,
  kCreate,
  kDefaultRun,
};

struct ForkedRegistryState {
  ForkedRegistryOperation operation{ForkedRegistryOperation::kClose};
  holoscan::ManagedShutdownSignals* signals{};
  const holoscan::ExecutionPlan* plan{};
  std::shared_ptr<ProcessSignalSink> sink;
  pid_t child{-1};
};

[[noreturn]] void run_forked_registry_child(ForkedRegistryState& state) noexcept {
  switch (state.operation) {
    case ForkedRegistryOperation::kClose: {
      const auto closed = state.signals->close();
      _exit(!closed && closed.error().code == holoscan::ErrorCode::kNotSupported ? 0 : 1);
    }
    case ForkedRegistryOperation::kDestroy: {
      {
        auto inherited = std::move(*state.signals);
      }
      _exit(0);
    }
    case ForkedRegistryOperation::kCreate: {
      const auto created = holoscan::ManagedShutdownSignals::create();
      _exit(!created && created.error().code == holoscan::ErrorCode::kNotSupported ? 0 : 1);
    }
    case ForkedRegistryOperation::kDefaultRun: {
      const int result = holoscan::run(*state.plan);
      _exit(result != 0 && state.sink->contains("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") &&
                    state.sink->contains_terminal_detail(
                        holoscan::detail::RuntimeManagedSignalPhase::kAcquireDomain,
                        0,
                        holoscan::detail::RuntimeManagedSignalCensusReason::kForkedProcess,
                        false,
                        false)
                ? 0
                : 1);
    }
  }
  _exit(1);
}

void fork_while_registry_locked(const holoscan::detail::RuntimeManagedSignalTestPoint,
                                void* opaque_state) noexcept {
  auto* const state = static_cast<ForkedRegistryState*>(opaque_state);
  if (state == nullptr || state->signals == nullptr || state->plan == nullptr) {
    return;
  }
  state->child = fork();
  if (state->child == 0) {
    run_forked_registry_child(*state);
  }
}

[[nodiscard]] int run_forked_registry_rejection(const ForkedRegistryOperation operation) {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto child_plan = make_plan(sink);
  ForkedRegistryState state{
      .operation = operation, .signals = &signals, .plan = &child_plan, .sink = sink};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kRegistryLockedBeforeBinding,
      &fork_while_registry_locked,
      &state);
  auto session = holoscan::run_async(make_failure_plan({}, control), signals);
  session.wait();
  const auto closed = signals.close();
  int status{};
  if (state.child <= 0 || !closed ||
      !wait_for_child_exit_for(state.child, status, std::chrono::seconds{5})) {
    if (state.child > 0) {
      static_cast<void>(kill(state.child, SIGKILL));
      static_cast<void>(wait_for_child_exit(state.child, status));
    }
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_resource_loop() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  for (std::size_t index = 0U; index < 20U; ++index) {
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
      std::fprintf(stderr,
                   "resource-loop create failed at %zu: code=%u fds=%zu threads=%zu\n",
                   index,
                   static_cast<unsigned>(created.error().code),
                   count_directory_entries("/proc/self/fd"),
                   count_directory_entries("/proc/self/task"));
      return 1;
    }
    auto signals = std::move(created).value();
    const auto closed = signals.close();
    if (!closed) {
      std::fprintf(stderr,
                   "resource-loop close failed at %zu: code=%u\n",
                   index,
                   static_cast<unsigned>(closed.error().code));
      return 1;
    }
    const ProcessResourceSnapshot resources_after = capture_process_resources();
    if (!process_resources_equal(resources_before, resources_after)) {
      std::fprintf(stderr,
                   "resource-loop mismatch at %zu: fds=%zu->%zu threads=%zu->%zu\n",
                   index,
                   resources_before.descriptors,
                   resources_after.descriptors,
                   resources_before.threads,
                   resources_after.threads);
      return 1;
    }
  }
  return 0;
}

[[nodiscard]] int run_worker_start_failure() {
  const std::size_t descriptors_before = count_directory_entries("/proc/self/fd");
  const std::size_t threads_before = count_directory_entries("/proc/self/task");
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kWorkerStart, EIO);
  const auto plan = holoscan::detail::RuntimeTestPlanAccess::with_managed_signal_test_control(
      make_plan(sink), control);
  auto session = holoscan::run_async(plan, signals);
  session.wait();
  const bool failed = session.termination() == holoscan::RunTermination::kFailed &&
                      session_has_diagnostic(session, "RUNTIME_WORKER_START_FAILED") &&
                      control->hits() == 1U;
  if (!signals.close()) {
    return 1;
  }
  auto recreated = holoscan::ManagedShutdownSignals::create();
  if (!recreated) {
    return 1;
  }
  auto replacement = std::move(recreated).value();
  const bool replacement_closed = static_cast<bool>(replacement.close());
  return failed && replacement_closed &&
                 count_directory_entries("/proc/self/fd") == descriptors_before &&
                 count_directory_entries("/proc/self/task") == threads_before
             ? 0
             : 1;
}

[[nodiscard]] int run_close_while_bound() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  auto session = holoscan::run_async(make_plan(), signals);
  if (!holoscan::test::RuntimeProbeState::wait_until_compute_enters()) {
    session.request_stop();
    session.wait();
    return 1;
  }
  const auto premature_close = signals.close();
  session.request_stop();
  session.wait();
  const auto final_close = signals.close();
  return !premature_close && premature_close.error().code == holoscan::ErrorCode::kNotReady &&
                 session.termination() == holoscan::RunTermination::kStopped && final_close
             ? 0
             : 1;
}

[[nodiscard]] int run_unbound_destructor_releases_domain() {
  const std::size_t descriptors_before = count_directory_entries("/proc/self/fd");
  const std::size_t threads_before = count_directory_entries("/proc/self/task");
  {
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
      return 1;
    }
    auto signals = std::move(created).value();
    static_cast<void>(signals);
  }
  auto recreated = holoscan::ManagedShutdownSignals::create();
  if (!recreated) {
    return 1;
  }
  auto signals = std::move(recreated).value();
  const bool closed = static_cast<bool>(signals.close());
  return closed && count_directory_entries("/proc/self/fd") == descriptors_before &&
                 count_directory_entries("/proc/self/task") == threads_before
             ? 0
             : 1;
}

[[nodiscard]] int run_generation_monotonic_child(const int ready_fd, const int received_fd) {
  std::uint64_t prior_domain{};
  std::uint64_t prior_binding{};
  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
      return 1;
    }
    auto signals = std::move(created).value();
    holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
    holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
    const auto sink = std::make_shared<ProcessSignalSink>(received_fd);
    auto session = holoscan::run_async(make_plan(sink), signals);
    session.wait();
    const auto detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
    const bool monotonic =
        detail.domain_generation > prior_domain && detail.binding_generation > prior_binding;
    prior_domain = detail.domain_generation;
    prior_binding = detail.binding_generation;
    if (session.termination() != holoscan::RunTermination::kStopped || !monotonic ||
        !signals.close()) {
      return 1;
    }
  }
  return 0;
}

[[nodiscard]] int supervise_generation_monotonic() {
  Pipe ready;
  Pipe received;
  if (!ready.open() || !received.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    _exit(run_generation_monotonic_child(ready.fd[1], received.fd[1]));
  }
  ready.close_write();
  received.close_write();
  bool protocol_ok{true};
  for (std::size_t iteration = 0U; iteration < 2U && protocol_ok; ++iteration) {
    protocol_ok = read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 && read_byte(received.fd[0]);
  }
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_invalid_policy() {
  holoscan::test::RuntimeProbeState::reset();
  // Exercise the documented invalid-representation path without invoking an
  // undefined enum value.
  const int result =
      holoscan::run(make_plan(),
                    static_cast<holoscan::RunSignalPolicy>(
                        0xffU));  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  return result != 0 && !holoscan::test::RuntimeProbeState::compute_entered.load() ? 0 : 1;
}

[[nodiscard]] holoscan::ExecutionPlan make_failure_plan(
    const std::shared_ptr<ProcessSignalSink>& sink,
    const std::shared_ptr<holoscan::detail::RuntimeManagedSignalTestControl>& control) {
  auto plan = holoscan::detail::RuntimeTestPlanAccess::with_managed_signal_test_control(
      make_plan(sink), control);
  return holoscan::detail::RuntimeTestPlanAccess::with_terminal_summary_sink(plan, sink);
}

struct OwnerBusyEstablishingState {
  const holoscan::ExecutionPlan* competing_plan{};
  std::size_t expected_thread_count{};
  int result{};
  bool compute_entered{};
  bool thread_joined{};
  bool thread_retired{};
  bool thread_failed{};
};

void owner_busy_establishing_checkpoint(const holoscan::detail::RuntimeManagedSignalTestPoint,
                                        void* state) noexcept {
  auto* const checkpoint = static_cast<OwnerBusyEstablishingState*>(state);
  if (checkpoint == nullptr || checkpoint->competing_plan == nullptr) {
    return;
  }
  try {
    std::thread competing([&] {
      checkpoint->result = holoscan::run(*checkpoint->competing_plan);
      checkpoint->compute_entered =
          holoscan::test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire);
    });
    competing.join();
    checkpoint->thread_joined = true;
    checkpoint->thread_retired = wait_for_process_thread_count(checkpoint->expected_thread_count);
  } catch (...) {
    checkpoint->thread_failed = true;
  }
}

[[nodiscard]] int run_owner_busy_while_establishing() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  holoscan::test::RuntimeProbeState::reset();
  const auto competing_sink = std::make_shared<ProcessSignalSink>();
  const auto competing_plan = make_plan(competing_sink);
  OwnerBusyEstablishingState checkpoint{.competing_plan = &competing_plan,
                                        .expected_thread_count = resources_before.threads};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kCensusBeforeMask,
      &owner_busy_establishing_checkpoint,
      &checkpoint);
  const int outer_result = holoscan::run(make_failure_plan({}, control));
  const bool passed = outer_result == 0 && checkpoint.result != 0 && !checkpoint.compute_entered &&
                      checkpoint.thread_joined && checkpoint.thread_retired &&
                      !checkpoint.thread_failed && control->hits() == 1U &&
                      competing_sink->contains("RUNTIME_MANAGED_SIGNAL_OWNER_BUSY") &&
                      competing_sink->contains_terminal_detail(
                          holoscan::detail::RuntimeManagedSignalPhase::kAcquireDomain,
                          0,
                          holoscan::detail::RuntimeManagedSignalCensusReason::kIncompatibleDomain,
                          false,
                          true) &&
                      process_resources_match(resources_before);
  if (!passed) {
    std::fprintf(stderr,
                 "owner-busy-establishing mismatch: outer=%d competing=%d compute=%d joined=%d "
                 "retired=%d thread_failed=%d hits=%u threads=%zu->%zu\n",
                 outer_result,
                 checkpoint.result,
                 static_cast<int>(checkpoint.compute_entered),
                 static_cast<int>(checkpoint.thread_joined),
                 static_cast<int>(checkpoint.thread_retired),
                 static_cast<int>(checkpoint.thread_failed),
                 control->hits(),
                 checkpoint.expected_thread_count,
                 count_directory_entries("/proc/self/task"));
  }
  return passed ? 0 : 1;
}

[[nodiscard]] int run_setup_failure_matrix() {
  using Point = holoscan::detail::RuntimeManagedSignalTestPoint;
  using Phase = holoscan::detail::RuntimeManagedSignalPhase;
  using Census = holoscan::detail::RuntimeManagedSignalCensusReason;
  struct SetupFailureCase {
    Point point;
    Phase phase;
    Census census_reason;
    bool domain_generation;
    bool binding_generation;
    std::uint32_t fail_on_hit{1U};
  };
  constexpr std::array cases{
      SetupFailureCase{
          Point::kReserveDomainGeneration, Phase::kAcquireDomain, Census::kNone, false, false},
      SetupFailureCase{
          Point::kCensusBeforeMask, Phase::kThreadCensus, Census::kCensusUnavailable, true, false},
      SetupFailureCase{Point::kSnapshotMask, Phase::kSnapshotMask, Census::kNone, true, false},
      SetupFailureCase{Point::kBlockMask, Phase::kBlockMask, Census::kNone, true, false},
      SetupFailureCase{
          Point::kCensusAfterMask, Phase::kThreadCensus, Census::kCensusUnavailable, true, false},
      SetupFailureCase{
          Point::kQueryDisposition, Phase::kQueryDisposition, Census::kNone, true, false},
      SetupFailureCase{
          Point::kQueryDisposition, Phase::kQueryDisposition, Census::kNone, true, false, 2U},
      SetupFailureCase{Point::kCreateSignalFd, Phase::kCreateSignalFd, Census::kNone, true, false},
      SetupFailureCase{
          Point::kCreateNotification, Phase::kCreateNotification, Census::kNone, true, false},
      SetupFailureCase{
          Point::kReserveBindingGeneration, Phase::kAcquireBinding, Census::kNone, true, false},
      SetupFailureCase{Point::kAcquireBinding, Phase::kAcquireBinding, Census::kNone, true, true},
  };
  for (const auto& failure : cases) {
    holoscan::test::RuntimeProbeState::reset();
    const auto sink = std::make_shared<ProcessSignalSink>();
    const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
        failure.point, EIO, failure.fail_on_hit);
    const ProcessResourceSnapshot resources_before = capture_process_resources();
    const int result = holoscan::run(make_failure_plan(sink, control));
    if (result == 0 || control->hits() != failure.fail_on_hit ||
        holoscan::test::RuntimeProbeState::compute_entered.load() ||
        !sink->contains("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") ||
        !sink->contains_terminal_detail(failure.phase,
                                        EIO,
                                        failure.census_reason,
                                        failure.binding_generation,
                                        failure.domain_generation) ||
        !process_resources_match(resources_before)) {
      return 1;
    }
    holoscan::test::RuntimeProbeState::reset();
    if (holoscan::run(make_plan()) != 0 || !process_resources_match(resources_before)) {
      return 1;
    }
  }
  return 0;
}

[[nodiscard]] int run_start_broker_failure() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kStartBroker, EAGAIN);
  auto created = holoscan::detail::ManagedShutdownSignalsAccess::create_for_test(control);
  if (created || created.error().token != "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED" ||
      created.error().error.code != holoscan::ErrorCode::kResourceExhausted ||
      created.error().detail.phase != holoscan::detail::RuntimeManagedSignalPhase::kStartBroker ||
      created.error().detail.os_error != EAGAIN || created.error().detail.domain_generation == 0U ||
      created.error().detail.binding_generation != 0U ||
      created.error().detail.census_reason !=
          holoscan::detail::RuntimeManagedSignalCensusReason::kNone ||
      control->hits() != 1U || !process_resources_match(resources_before)) {
    return 1;
  }
  auto recovered = holoscan::ManagedShutdownSignals::create();
  if (!recovered) {
    return 1;
  }
  auto signals = std::move(recovered).value();
  return signals.close() && process_resources_match(resources_before) ? 0 : 1;
}

[[nodiscard]] int run_broker_task_open_failure() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kOpenBrokerTask, EIO);
  auto created = holoscan::detail::ManagedShutdownSignalsAccess::create_for_test(control);
  if (created || created.error().token != "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED" ||
      created.error().error.code != holoscan::ErrorCode::kResourceExhausted ||
      created.error().detail.phase != holoscan::detail::RuntimeManagedSignalPhase::kStartBroker ||
      created.error().detail.os_error != EIO || created.error().detail.domain_generation == 0U ||
      created.error().detail.binding_generation != 0U || !created.error().cleanup_present ||
      created.error().cleanup_token != "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED" ||
      created.error().cleanup_detail.phase !=
          holoscan::detail::RuntimeManagedSignalPhase::kBindingCleanup ||
      created.error().cleanup_detail.os_error != EIO || !created.error().poison_process ||
      control->hits() != 1U || !process_resources_match(resources_before)) {
    return 1;
  }
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted
             ? 0
             : 1;
}

[[nodiscard]] int run_setup_primary_and_cleanup_failure() {
  using Point = holoscan::detail::RuntimeManagedSignalTestPoint;
  using Phase = holoscan::detail::RuntimeManagedSignalPhase;
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      Point::kCreateNotification, EIO, Point::kCloseSignalFd, EBADF);
  const int result = holoscan::run(make_failure_plan(sink, control));
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return result != 0 && !holoscan::test::RuntimeProbeState::compute_entered.load() &&
                 control->hits() == 2U && sink->contains("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED") &&
                 sink->contains_terminal_detail(
                     Phase::kCreateNotification,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone,
                     false,
                     true) &&
                 sink->contains_terminal_detail(
                     Phase::kBindingCleanup,
                     EBADF,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone,
                     false,
                     true) &&
                 process_resources_match(resources_before) && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted
             ? 0
             : 1;
}

[[nodiscard]] int run_wait_primary_and_cleanup_failure() {
  using Point = holoscan::detail::RuntimeManagedSignalTestPoint;
  using Phase = holoscan::detail::RuntimeManagedSignalPhase;
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      Point::kReadSignal, EIO, Point::kRestoreMask, EBUSY);
  const int result = holoscan::run(make_failure_plan(sink, control));
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return result != 0 && control->hits() >= 2U &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED") &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED") &&
                 sink->contains_terminal_detail(
                     Phase::kRead,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
                 sink->contains_terminal_detail(
                     Phase::kRestoreMask,
                     EBUSY,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
                 process_resources_match(resources_before) && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted
             ? 0
             : 1;
}

[[nodiscard]] int run_transient_failure(
    const holoscan::detail::RuntimeManagedSignalTestPoint point,
    const holoscan::test::RuntimeProbeMode mode, const std::string_view expected_token,
    const holoscan::detail::RuntimeManagedSignalPhase expected_phase) {
  holoscan::test::RuntimeProbeState::reset(mode);
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control =
      std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(point, EIO);
  const int result = holoscan::run(make_failure_plan(sink, control));
  const bool token_present = sink->contains(expected_token);
  const bool detail_present = sink->contains_terminal_detail(
      expected_phase, EIO, holoscan::detail::RuntimeManagedSignalCensusReason::kNone);
  const bool resources_restored = process_resources_match(resources_before);
  if (result == 0 || control->hits() < 1U || !token_present || !detail_present ||
      !resources_restored) {
    std::fprintf(stderr,
                 "transient failure mismatch: result=%d hits=%u token=%d detail=%d resources=%d\n",
                 result,
                 control->hits(),
                 static_cast<int>(token_present),
                 static_cast<int>(detail_present),
                 static_cast<int>(resources_restored));
    return 1;
  }
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return process_resources_match(resources_before) && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted
             ? 0
             : 1;
}

[[nodiscard]] int run_close_domain_failure() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kCloseDomain, EIO);
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto closed = signals.close();
  const auto retried_close = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return session.termination() == holoscan::RunTermination::kCompleted && !closed &&
                 closed.error().code == holoscan::ErrorCode::kFailure && !retried_close &&
                 retried_close.error().code == holoscan::ErrorCode::kClosed && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted &&
                 control->hits() == 1U
             ? 0
             : 1;
}

[[nodiscard]] int run_broker_join_failure() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kJoinBroker, EIO);
  auto session = holoscan::run_async(make_failure_plan({}, control), signals);
  session.wait();
  const auto closed = signals.close();
  const auto retried_close = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return session.termination() == holoscan::RunTermination::kCompleted && !closed &&
                 closed.error().code == holoscan::ErrorCode::kFailure && !retried_close &&
                 retried_close.error().code == holoscan::ErrorCode::kClosed && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted &&
                 control->hits() == 1U && process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int run_broker_retirement_failure() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kWaitBrokerRetirement, EIO);
  auto created = holoscan::detail::ManagedShutdownSignalsAccess::create_for_test(control);
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  const auto closed = signals.close();
  const auto retried_close = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return !closed && closed.error().code == holoscan::ErrorCode::kFailure && !retried_close &&
                 retried_close.error().code == holoscan::ErrorCode::kClosed && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted &&
                 control->hits() == 1U &&
                 process_resources_equal(resources_before, capture_process_resources())
             ? 0
             : 1;
}

[[nodiscard]] int run_persistent_drain_failure() {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kDrainNotification, EIO);
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto closed = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return session.termination() == holoscan::RunTermination::kFailed &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_WAIT_FAILED") &&
                 sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kDrainNotification,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
                 control->hits() >= 1U && !closed &&
                 closed.error().code == holoscan::ErrorCode::kFailure && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted
             ? 0
             : 1;
}

[[nodiscard]] int run_delegate_failure_child(const int ready_fd, const int received_fd,
                                             const int escalated_fd, const int stop_gate_fd) {
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  g_forwarded_signal = 0;
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(stop_gate_fd);
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd, escalated_fd);
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kDelegateSignal, EIO);
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto closed = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  return session.termination() == holoscan::RunTermination::kFailed && !closed &&
                 closed.error().code == holoscan::ErrorCode::kFailure && !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted &&
                 control->hits() >= 1U && g_forwarded_signal == SIGTERM &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED") == 1U &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED") &&
                 sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kDelegateSignal,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
                 prior_policy_is_unchanged(PriorPolicy::kCustom)
             ? 0
             : 1;
}

struct RestoreMaskCheckpoint {
  int reached_fd{-1};
};

void notify_restore_mask_checkpoint(const holoscan::detail::RuntimeManagedSignalTestPoint,
                                    void* state) noexcept {
  auto* const checkpoint = static_cast<RestoreMaskCheckpoint*>(state);
  if (checkpoint != nullptr) {
    static_cast<void>(write_byte(checkpoint->reached_fd));
  }
}

[[nodiscard]] int run_second_restore_failure_child(const bool persistent_domain, const int ready_fd,
                                                   const int received_fd,
                                                   const int restore_reached_fd,
                                                   const int stop_gate_fd) {
  using Point = holoscan::detail::RuntimeManagedSignalTestPoint;
  using Phase = holoscan::detail::RuntimeManagedSignalPhase;
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  reset_forwarded_signal_history();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(stop_gate_fd);
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd);
  RestoreMaskCheckpoint checkpoint{.reached_fd = restore_reached_fd};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      Point::kRestoreMask, EIO, Point::kRestoreMask, &notify_restore_mask_checkpoint, &checkpoint);

  bool failed{};
  bool close_failed{true};
  if (persistent_domain) {
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
      return 1;
    }
    auto signals = std::move(created).value();
    auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
    session.wait();
    failed = session.termination() == holoscan::RunTermination::kFailed;
    const auto closed = signals.close();
    close_failed = !closed && closed.error().code == holoscan::ErrorCode::kFailure;
  } else {
    failed = holoscan::run(make_failure_plan(sink, control)) != 0;
  }

  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  const std::size_t received_events = sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const std::size_t escalated_events = sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED");
  const std::size_t restore_failures = sink->token_count("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED");
  const std::size_t failed_summaries = sink->failed_summary_count();
  const bool detail_present = sink->contains_terminal_detail(
      Phase::kRestoreMask, EIO, holoscan::detail::RuntimeManagedSignalCensusReason::kNone);
  const bool policy_unchanged = prior_policy_is_unchanged(PriorPolicy::kCustom);
  const bool resources_restored = process_resources_match(resources_before);
  const bool poisoned =
      !competing_create && competing_create.error().code == holoscan::ErrorCode::kResourceExhausted;
  const std::uint32_t expected_hits = persistent_domain ? 3U : 2U;
  if (!failed || !close_failed || control->hits() != expected_hits || received_events != 1U ||
      escalated_events != 0U || restore_failures != 1U || failed_summaries != 1U ||
      !detail_present || g_forwarded_signal != 0 || g_forwarded_count != 0 || !policy_unchanged ||
      !resources_restored || !poisoned) {
    std::fprintf(stderr,
                 "second restore failure mismatch: persistent=%d failed=%d close=%d hits=%u "
                 "received=%zu escalated=%zu restore=%zu summaries=%zu detail=%d forwarded=%d "
                 "count=%d policy=%d resources=%d poison=%d\n",
                 static_cast<int>(persistent_domain),
                 static_cast<int>(failed),
                 static_cast<int>(close_failed),
                 control->hits(),
                 received_events,
                 escalated_events,
                 restore_failures,
                 failed_summaries,
                 static_cast<int>(detail_present),
                 static_cast<int>(g_forwarded_signal),
                 static_cast<int>(g_forwarded_count),
                 static_cast<int>(policy_unchanged),
                 static_cast<int>(resources_restored),
                 static_cast<int>(poisoned));
    return 1;
  }
  return 0;
}

[[nodiscard]] int supervise_second_restore_failure(const bool persistent_domain) {
  Pipe ready;
  Pipe received;
  Pipe restore_reached;
  Pipe stop_gate;
  if (!ready.open() || !received.open() || !restore_reached.open() || !stop_gate.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    restore_reached.close_read();
    stop_gate.close_write();
    _exit(run_second_restore_failure_child(
        persistent_domain, ready.fd[1], received.fd[1], restore_reached.fd[1], stop_gate.fd[0]));
  }
  ready.close_write();
  received.close_write();
  restore_reached.close_write();
  stop_gate.close_read();
  const bool protocol_ok = read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 &&
                           read_byte(received.fd[0]) && kill(child, SIGTERM) == 0 &&
                           read_byte(restore_reached.fd[0]) && write_byte(stop_gate.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int supervise_delegate_failure() {
  Pipe ready;
  Pipe received;
  Pipe escalated;
  Pipe stop_gate;
  if (!ready.open() || !received.open() || !escalated.open() || !stop_gate.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    escalated.close_read();
    stop_gate.close_write();
    _exit(
        run_delegate_failure_child(ready.fd[1], received.fd[1], escalated.fd[1], stop_gate.fd[0]));
  }
  ready.close_write();
  received.close_write();
  escalated.close_write();
  stop_gate.close_read();
  const bool protocol_ok = read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 &&
                           read_byte(received.fd[0]) && kill(child, SIGTERM) == 0 &&
                           read_byte(escalated.fd[0]) && write_byte(stop_gate.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_disposition_drift() {
  struct sigaction custom{};
  struct sigaction ignored{};
  struct sigaction prior{};
  sigemptyset(&custom.sa_mask);
  sigemptyset(&ignored.sa_mask);
  custom.sa_handler = &record_forwarded_signal;
  ignored.sa_handler = SIG_IGN;
  if (sigaction(SIGINT, &custom, &prior) != 0) {
    return 1;
  }
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    static_cast<void>(sigaction(SIGINT, &prior, nullptr));
    return 1;
  }
  auto signals = std::move(created).value();
  const auto sink = std::make_shared<ProcessSignalSink>();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  auto session = holoscan::run_async(make_plan(sink), signals);
  const bool entered = holoscan::test::RuntimeProbeState::wait_until_compute_enters();
  const bool mutated = entered && sigaction(SIGINT, &ignored, nullptr) == 0;
  session.request_stop();
  session.wait();
  const auto closed = signals.close();
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  static_cast<void>(sigaction(SIGINT, &prior, nullptr));
  return mutated && session.termination() == holoscan::RunTermination::kFailed &&
                 sink->token_count("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED") == 1U &&
                 sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kQueryDisposition,
                     EPROTO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
                 !closed && closed.error().code == holoscan::ErrorCode::kIncompatible &&
                 !competing_create &&
                 competing_create.error().code == holoscan::ErrorCode::kResourceExhausted
             ? 0
             : 1;
}

struct DispositionDriftCheckpoint {
  int error{};
};

void mutate_sigint_disposition(const holoscan::detail::RuntimeManagedSignalTestPoint,
                               void* opaque_state) noexcept {
  auto* const state = static_cast<DispositionDriftCheckpoint*>(opaque_state);
  if (state == nullptr) {
    return;
  }
  struct sigaction ignored{};
  sigemptyset(&ignored.sa_mask);
  ignored.sa_handler = SIG_IGN;
  state->error = sigaction(SIGINT, &ignored, nullptr) == 0 ? 0 : errno;
}

[[nodiscard]] int run_transient_disposition_drift() {
  struct sigaction custom{};
  struct sigaction prior{};
  sigemptyset(&custom.sa_mask);
  custom.sa_handler = &record_forwarded_signal;
  if (sigaction(SIGINT, &custom, &prior) != 0) {
    return 1;
  }
  DispositionDriftCheckpoint checkpoint;
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kCandidateReadyBeforeFinalDrain,
      &mutate_sigint_disposition,
      &checkpoint);
  const auto sink = std::make_shared<ProcessSignalSink>();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kComplete);
  const int result = holoscan::run(make_failure_plan(sink, control));
  const auto competing_create = holoscan::ManagedShutdownSignals::create();
  static_cast<void>(sigaction(SIGINT, &prior, nullptr));
  const bool passed =
      result != 0 && checkpoint.error == 0 && control->hits() == 1U &&
      sink->token_count("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED") == 1U &&
      sink->contains_terminal_detail(holoscan::detail::RuntimeManagedSignalPhase::kQueryDisposition,
                                     EPROTO,
                                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone) &&
      !competing_create && competing_create.error().code == holoscan::ErrorCode::kResourceExhausted;
  if (!passed) {
    std::fprintf(stderr,
                 "transient disposition mismatch: result=%d checkpoint=%d hits=%u restore=%zu "
                 "detail=%d competing=%d competing_code=%u\n",
                 result,
                 checkpoint.error,
                 control->hits(),
                 sink->token_count("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED"),
                 static_cast<int>(sink->contains_terminal_detail(
                     holoscan::detail::RuntimeManagedSignalPhase::kQueryDisposition,
                     EPROTO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone)),
                 static_cast<int>(static_cast<bool>(competing_create)),
                 competing_create ? 0U : static_cast<unsigned>(competing_create.error().code));
  }
  return passed ? 0 : 1;
}

[[nodiscard]] int run_terminal_precedence_child(const holoscan::test::RuntimeProbeMode mode,
                                                const holoscan::RunTermination expected,
                                                const int ready_fd, const int received_fd,
                                                const int stop_gate_fd) {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(mode);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(stop_gate_fd);
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd);
  auto session = holoscan::run_async(make_plan(sink), signals);
  session.wait();
  const bool received = sink->contains("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const auto closed = signals.close();
  if (session.termination() != expected || !received || !closed) {
    std::fprintf(stderr,
                 "terminal precedence mismatch: expected=%u actual=%u received=%d close=%d "
                 "close_code=%u\n",
                 static_cast<unsigned>(expected),
                 static_cast<unsigned>(session.termination()),
                 static_cast<int>(received),
                 static_cast<int>(static_cast<bool>(closed)),
                 closed ? 0U : static_cast<unsigned>(closed.error().code));
    return 1;
  }
  return 0;
}

[[nodiscard]] int supervise_terminal_precedence(const holoscan::test::RuntimeProbeMode mode,
                                                const holoscan::RunTermination expected) {
  Pipe ready;
  Pipe received;
  Pipe stop_gate;
  if (!ready.open() || !received.open() || !stop_gate.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    stop_gate.close_write();
    _exit(run_terminal_precedence_child(
        mode, expected, ready.fd[1], received.fd[1], stop_gate.fd[0]));
  }
  ready.close_write();
  received.close_write();
  stop_gate.close_read();
  const bool protocol_ok = read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 &&
                           read_byte(received.fd[0]) && write_byte(stop_gate.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

struct ExternalCheckpoint {
  int reached_fd{-1};
  int release_fd{-1};
};

void external_checkpoint(const holoscan::detail::RuntimeManagedSignalTestPoint,
                         void* state) noexcept {
  auto* const checkpoint = static_cast<ExternalCheckpoint*>(state);
  if (checkpoint == nullptr || !write_byte(checkpoint->reached_fd)) {
    return;
  }
  static_cast<void>(read_byte(checkpoint->release_fd));
}

struct CensusCheckpointState {
  pthread_t thread{};
  std::atomic_bool started{};
  std::atomic_bool release{};
  int create_error{};
  bool created{};
};

void* census_checkpoint_peer(void* state) noexcept {
  auto* const checkpoint = static_cast<CensusCheckpointState*>(state);
  checkpoint->started.store(true, std::memory_order_release);
  while (!checkpoint->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  return nullptr;
}

void census_changed_checkpoint(const holoscan::detail::RuntimeManagedSignalTestPoint,
                               void* state) noexcept {
  auto* const checkpoint = static_cast<CensusCheckpointState*>(state);
  if (checkpoint == nullptr) {
    return;
  }
  checkpoint->create_error =
      pthread_create(&checkpoint->thread, nullptr, &census_checkpoint_peer, checkpoint);
  checkpoint->created = checkpoint->create_error == 0;
  while (checkpoint->created && !checkpoint->started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

struct PendingEstablishmentSignal {
  int signal{};
  int send_error{};
};

void pending_establishment_signal_checkpoint(const holoscan::detail::RuntimeManagedSignalTestPoint,
                                             void* state) noexcept {
  auto* const pending = static_cast<PendingEstablishmentSignal*>(state);
  if (pending != nullptr && kill(getpid(), pending->signal) != 0) {
    pending->send_error = errno;
  }
}

[[nodiscard]] int run_census_changed_during_setup() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  CensusCheckpointState checkpoint;
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kCensusAfterMask,
      &census_changed_checkpoint,
      &checkpoint);
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const int result = holoscan::run(make_failure_plan(sink, control));
  checkpoint.release.store(true, std::memory_order_release);
  const int join_error = checkpoint.created ? pthread_join(checkpoint.thread, nullptr) : 0;
  const bool rejected = result != 0 && checkpoint.created && checkpoint.create_error == 0 &&
                        join_error == 0 && control->hits() == 1U &&
                        !holoscan::test::RuntimeProbeState::compute_entered.load() &&
                        sink->contains("RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED") &&
                        sink->contains_terminal_detail(
                            holoscan::detail::RuntimeManagedSignalPhase::kThreadCensus,
                            0,
                            holoscan::detail::RuntimeManagedSignalCensusReason::kCensusChanged,
                            false,
                            true,
                            2U) &&
                        process_resources_match(resources_before);
  holoscan::test::RuntimeProbeState::reset();
  return rejected && holoscan::run(make_plan()) == 0 && process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int run_pending_during_establishment() {
  for (const int signal : std::array<int, 2>{SIGINT, SIGTERM}) {
    const ProcessResourceSnapshot resources_before = capture_process_resources();
    PendingEstablishmentSignal pending{.signal = signal};
    const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
        holoscan::detail::RuntimeManagedSignalTestPoint::kMaskBlockedBeforeSignalFd,
        &pending_establishment_signal_checkpoint,
        &pending);
    holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
    const auto sink = std::make_shared<ProcessSignalSink>();
    const int result = holoscan::run(make_failure_plan(sink, control));
    const auto detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
    if (result == 0 || pending.send_error != 0 || control->hits() != 1U ||
        sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") != 1U || !detail.present ||
        detail.phase != holoscan::detail::RuntimeManagedSignalPhase::kAcceptSignal ||
        detail.signal != runtime_signal(signal) ||
        detail.occurrence != holoscan::detail::RuntimeManagedSignalOccurrence::kFirst ||
        detail.domain_generation == 0U || detail.binding_generation == 0U ||
        !process_resources_match(resources_before)) {
      return 1;
    }
  }
  return 0;
}

[[nodiscard]] int run_pending_establishment_rollback() {
  using Point = holoscan::detail::RuntimeManagedSignalTestPoint;
  using Phase = holoscan::detail::RuntimeManagedSignalPhase;
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  reset_forwarded_signal_history();
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  PendingEstablishmentSignal pending{.signal = SIGINT};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      Point::kCreateNotification,
      EIO,
      Point::kMaskBlockedBeforeSignalFd,
      &pending_establishment_signal_checkpoint,
      &pending);
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const int result = holoscan::run(make_failure_plan(sink, control));
  for (std::size_t spin = 0U; spin < 10'000'000U && g_forwarded_signal != SIGINT; ++spin) {
    std::this_thread::yield();
  }
  const bool failed_before_compute =
      !holoscan::test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire);
  holoscan::test::RuntimeProbeState::reset();
  const bool recovered = holoscan::run(make_plan()) == 0;
  return result != 0 && pending.send_error == 0 && control->hits() == 2U && failed_before_compute &&
                 sink->contains("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") &&
                 sink->contains_terminal_detail(
                     Phase::kCreateNotification,
                     EIO,
                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone,
                     false,
                     true) &&
                 g_forwarded_signal == SIGINT && recovered &&
                 process_resources_match(resources_before)
             ? 0
             : 1;
}

[[nodiscard]] int run_adopted_binding_failure() {
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kAcquireBinding, EIO);
  const int rejected_result = holoscan::run(make_failure_plan(sink, control));
  const bool rejected =
      rejected_result != 0 && control->hits() == 1U &&
      !holoscan::test::RuntimeProbeState::compute_entered.load() &&
      sink->contains("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") &&
      sink->contains_terminal_detail(holoscan::detail::RuntimeManagedSignalPhase::kAcquireBinding,
                                     EIO,
                                     holoscan::detail::RuntimeManagedSignalCensusReason::kNone);
  holoscan::test::RuntimeProbeState::reset();
  const bool recovered = holoscan::run(make_plan()) == 0;
  const bool closed = static_cast<bool>(signals.close());
  return rejected && recovered && closed && process_resources_match(resources_before) ? 0 : 1;
}

[[nodiscard]] int run_generation_exhaustion_child(const int ready_fd, const int received_fd) {
  using Phase = holoscan::detail::RuntimeManagedSignalPhase;
  constexpr std::uint64_t kDomainGeneration = std::numeric_limits<std::uint64_t>::max() - 2U;
  constexpr std::uint64_t kBindingGeneration = std::numeric_limits<std::uint64_t>::max() - 1U;
  const ProcessResourceSnapshot resources_before = capture_process_resources();
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd);
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kNone, 0);
  control->seed_next_generation(kDomainGeneration);
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  const int first_result = holoscan::run(make_failure_plan(sink, control));
  const auto received = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  if (first_result == 0 || received.domain_generation != kDomainGeneration ||
      received.binding_generation != kBindingGeneration ||
      !process_resources_match(resources_before)) {
    return 1;
  }
  for (std::size_t attempt = 0U; attempt < 2U; ++attempt) {
    holoscan::test::RuntimeProbeState::reset();
    const auto exhausted_sink = std::make_shared<ProcessSignalSink>();
    if (holoscan::run(make_plan(exhausted_sink)) == 0 ||
        holoscan::test::RuntimeProbeState::compute_entered.load() ||
        !exhausted_sink->contains("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED") ||
        !exhausted_sink->contains_terminal_detail(
            Phase::kAcquireDomain,
            0,
            holoscan::detail::RuntimeManagedSignalCensusReason::kNone,
            false,
            false) ||
        !process_resources_match(resources_before)) {
      return 1;
    }
  }
  return 0;
}

[[nodiscard]] int supervise_generation_exhaustion() {
  Pipe ready;
  Pipe received;
  if (!ready.open() || !received.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    _exit(run_generation_exhaustion_child(ready.fd[1], received.fd[1]));
  }
  ready.close_write();
  received.close_write();
  const bool protocol_ok =
      read_byte(ready.fd[0]) && kill(child, SIGINT) == 0 && read_byte(received.fd[0]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

void broker_tid_checkpoint(const holoscan::detail::RuntimeManagedSignalTestPoint,
                           void* state) noexcept {
  auto* const checkpoint = static_cast<ExternalCheckpoint*>(state);
  if (checkpoint == nullptr) {
    return;
  }
  const pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  [[maybe_unused]] const ssize_t written = write(checkpoint->reached_fd, &tid, sizeof(tid));
  static_cast<void>(read_byte(checkpoint->release_fd));
}

[[nodiscard]] bool read_pid(const int fd, pid_t& value) noexcept {
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  int polled{};
  do {
    polled = poll(&descriptor, 1U, 10'000);
  } while (polled < 0 && errno == EINTR);
  if (polled != 1 || (descriptor.revents & (POLLIN | POLLHUP)) == 0) {
    return false;
  }
  ssize_t result{};
  do {
    result = read(fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  return result == sizeof(value);
}

enum class CheckpointScenario : std::uint8_t {
  kBeforeAttach,
  kCompletedCandidate,
  kFailedCandidate,
};

[[nodiscard]] int run_checkpoint_signal_child(const CheckpointScenario scenario,
                                              const int reached_fd, const int release_fd) {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  const auto mode = scenario == CheckpointScenario::kFailedCandidate
                        ? holoscan::test::RuntimeProbeMode::kFail
                        : (scenario == CheckpointScenario::kCompletedCandidate
                               ? holoscan::test::RuntimeProbeMode::kComplete
                               : holoscan::test::RuntimeProbeMode::kWaitForStop);
  holoscan::test::RuntimeProbeState::reset(mode);
  ExternalCheckpoint checkpoint{.reached_fd = reached_fd, .release_fd = release_fd};
  const auto point =
      scenario == CheckpointScenario::kBeforeAttach
          ? holoscan::detail::RuntimeManagedSignalTestPoint::kBindingPublishedBeforeAttach
          : holoscan::detail::RuntimeManagedSignalTestPoint::kCandidateReadyBeforeFinalDrain;
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      point, &external_checkpoint, &checkpoint);
  const auto sink = std::make_shared<ProcessSignalSink>();
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto closed = signals.close();
  const auto expected = scenario == CheckpointScenario::kFailedCandidate
                            ? holoscan::RunTermination::kFailed
                            : holoscan::RunTermination::kStopped;
  return session.termination() == expected &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U &&
                 control->hits() == 1U && closed
             ? 0
             : 1;
}

[[nodiscard]] int supervise_checkpoint_signal(const CheckpointScenario scenario) {
  Pipe reached;
  Pipe release;
  if (!reached.open() || !release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    reached.close_read();
    release.close_write();
    _exit(run_checkpoint_signal_child(scenario, reached.fd[1], release.fd[0]));
  }
  reached.close_write();
  release.close_read();
  const bool protocol_ok =
      read_byte(reached.fd[0]) && kill(child, SIGINT) == 0 && write_byte(release.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_broker_directed_child(const int reached_fd, const int release_fd) {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  ExternalCheckpoint checkpoint{.reached_fd = reached_fd, .release_fd = release_fd};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kBrokerReadyForDirectedSignal,
      &broker_tid_checkpoint,
      &checkpoint);
  const auto sink = std::make_shared<ProcessSignalSink>();
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto detail = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const auto closed = signals.close();
  return session.termination() == holoscan::RunTermination::kStopped &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U &&
                 detail.signal == holoscan::detail::RuntimeManagedSignal::kSigInt &&
                 control->hits() == 1U && closed
             ? 0
             : 1;
}

[[nodiscard]] int supervise_broker_directed_signal() {
  Pipe reached;
  Pipe release;
  if (!reached.open() || !release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    reached.close_read();
    release.close_write();
    _exit(run_broker_directed_child(reached.fd[1], release.fd[0]));
  }
  reached.close_write();
  release.close_read();
  pid_t broker_tid{};
  const bool protocol_ok = read_pid(reached.fd[0], broker_tid) && broker_tid > 0 &&
                           syscall(SYS_tgkill, child, broker_tid, SIGINT) == 0 &&
                           write_byte(release.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

[[nodiscard]] int run_pending_batch_child(const bool mixed, const int reached_fd,
                                          const int release_fd) {
  if (!install_prior_policy(PriorPolicy::kCustom)) {
    return 1;
  }
  g_forwarded_signal = 0;
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(holoscan::test::RuntimeProbeMode::kWaitForStop);
  ExternalCheckpoint checkpoint{.reached_fd = reached_fd, .release_fd = release_fd};
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kBrokerReadyForDirectedSignal,
      &external_checkpoint,
      &checkpoint);
  const auto sink = std::make_shared<ProcessSignalSink>();
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const auto received = sink->event_detail("RUNTIME_MANAGED_SIGNAL_RECEIVED");
  const auto escalated = sink->event_detail("RUNTIME_MANAGED_SIGNAL_ESCALATED");
  const auto closed = signals.close();
  const bool common = session.termination() == holoscan::RunTermination::kStopped && closed &&
                      sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U;
  if (!mixed) {
    return common && sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED") == 0U &&
                   received.signal == holoscan::detail::RuntimeManagedSignal::kSigInt &&
                   g_forwarded_signal == 0
               ? 0
               : 1;
  }
  const bool distinct = received.signal != escalated.signal &&
                        received.signal != holoscan::detail::RuntimeManagedSignal::kNone &&
                        escalated.signal != holoscan::detail::RuntimeManagedSignal::kNone;
  const bool forwarded_actual_second =
      (g_forwarded_signal == SIGINT &&
       escalated.signal == holoscan::detail::RuntimeManagedSignal::kSigInt) ||
      (g_forwarded_signal == SIGTERM &&
       escalated.signal == holoscan::detail::RuntimeManagedSignal::kSigTerm);
  return common && sink->event_count("RUNTIME_MANAGED_SIGNAL_ESCALATED") == 1U && distinct &&
                 forwarded_actual_second
             ? 0
             : 1;
}

[[nodiscard]] int supervise_pending_batch(const bool mixed) {
  Pipe reached;
  Pipe release;
  if (!reached.open() || !release.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    reached.close_read();
    release.close_write();
    _exit(run_pending_batch_child(mixed, reached.fd[1], release.fd[0]));
  }
  reached.close_write();
  release.close_read();
  bool protocol_ok = read_byte(reached.fd[0]) && kill(child, SIGINT) == 0;
  protocol_ok =
      protocol_ok && kill(child, mixed ? SIGTERM : SIGINT) == 0 && write_byte(release.fd[1]);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

struct ThreadIdCheckpoint {
  std::atomic<pid_t> tid{};
};

[[nodiscard]] bool wait_for_thread_id(const ThreadIdCheckpoint& checkpoint) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
  while (checkpoint.tid.load(std::memory_order_acquire) <= 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

void capture_thread_id(const holoscan::detail::RuntimeManagedSignalTestPoint,
                       void* opaque_state) noexcept {
  auto* const state = static_cast<ThreadIdCheckpoint*>(opaque_state);
  if (state != nullptr) {
    state->tid.store(static_cast<pid_t>(syscall(SYS_gettid)), std::memory_order_release);
  }
}

[[nodiscard]] int run_signal_after_freeze_forwarded() {
  struct sigaction custom{};
  struct sigaction prior{};
  sigemptyset(&custom.sa_mask);
  custom.sa_handler = &record_forwarded_signal;
  if (sigaction(SIGINT, &custom, &prior) != 0) {
    return 1;
  }
  reset_forwarded_signal_history();
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    static_cast<void>(sigaction(SIGINT, &prior, nullptr));
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  ThreadIdCheckpoint broker;
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kUnboundWaitForActivity,
      &capture_thread_id,
      &broker);
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const bool broker_waiting = wait_for_control_hits(control, 1U);
  if (broker_waiting && kill(getpid(), SIGINT) == 0) {
    for (std::size_t spin = 0U; spin < 10'000'000U && g_forwarded_signal != SIGINT; ++spin) {
      std::this_thread::yield();
    }
  }
  const bool unchanged = session.termination() == holoscan::RunTermination::kCompleted &&
                         sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 0U &&
                         g_forwarded_signal == SIGINT && broker.tid.load() > 0 &&
                         g_forwarded_tid == broker.tid.load();
  const auto closed = signals.close();
  static_cast<void>(sigaction(SIGINT, &prior, nullptr));
  return unchanged && closed ? 0 : 1;
}

[[nodiscard]] int run_blocked_signal_after_freeze_pending() {
  struct sigaction custom{};
  struct sigaction prior_action{};
  sigset_t managed{};
  sigset_t prior_mask{};
  sigemptyset(&custom.sa_mask);
  sigemptyset(&managed);
  sigaddset(&managed, SIGINT);
  custom.sa_handler = &record_forwarded_signal;
  if (pthread_sigmask(SIG_BLOCK, &managed, &prior_mask) != 0 ||
      sigaction(SIGINT, &custom, &prior_action) != 0) {
    return 1;
  }
  reset_forwarded_signal_history();
  const pid_t creator_tid = static_cast<pid_t>(syscall(SYS_gettid));
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    static_cast<void>(sigaction(SIGINT, &prior_action, nullptr));
    static_cast<void>(pthread_sigmask(SIG_SETMASK, &prior_mask, nullptr));
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<ProcessSignalSink>();
  ThreadIdCheckpoint broker;
  const auto control = std::make_shared<holoscan::detail::RuntimeManagedSignalTestControl>(
      holoscan::detail::RuntimeManagedSignalTestPoint::kUnboundWaitForActivity,
      &capture_thread_id,
      &broker);
  auto session = holoscan::run_async(make_failure_plan(sink, control), signals);
  session.wait();
  const bool broker_waiting = wait_for_thread_id(broker);
  const pid_t broker_tid = broker.tid.load(std::memory_order_acquire);
  const bool sent = broker_waiting && syscall(SYS_tgkill, getpid(), broker_tid, SIGINT) == 0;
  bool pending{};
  for (std::size_t spin = 0U; sent && spin < 10'000'000U && !pending; ++spin) {
    sigset_t observed{};
    pending = sigpending(&observed) == 0 && sigismember(&observed, SIGINT) == 1;
    if (!pending) {
      std::this_thread::yield();
    }
  }
  const bool quiet_while_blocked = g_forwarded_signal == 0;
  const auto closed = signals.close();
  const bool unblocked = pthread_sigmask(SIG_UNBLOCK, &managed, nullptr) == 0;
  for (std::size_t spin = 0U; unblocked && spin < 10'000'000U && g_forwarded_signal != SIGINT;
       ++spin) {
    std::this_thread::yield();
  }
  const bool delivered_on_creator = g_forwarded_signal == SIGINT && g_forwarded_tid == creator_tid;
  static_cast<void>(sigaction(SIGINT, &prior_action, nullptr));
  static_cast<void>(pthread_sigmask(SIG_SETMASK, &prior_mask, nullptr));
  return session.termination() == holoscan::RunTermination::kCompleted && sent && pending &&
                 quiet_while_blocked && closed && delivered_on_creator &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 0U
             ? 0
             : 1;
}

[[nodiscard]] int run_liveness_child(const holoscan::test::RuntimeProbeMode mode,
                                     const int ready_fd, const int received_fd,
                                     const int release_fd, const int evidence_fd) {
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    return 1;
  }
  auto signals = std::move(created).value();
  holoscan::test::RuntimeProbeState::reset(mode);
  holoscan::test::RuntimeProbeState::bind_external_ready_fd(ready_fd);
  if (mode == holoscan::test::RuntimeProbeMode::kIgnoreStopUntilReleased) {
    holoscan::test::RuntimeProbeState::bind_external_stop_gate_fd(release_fd);
  }
  const auto sink = std::make_shared<ProcessSignalSink>(received_fd);
  auto session = holoscan::run_async(make_plan(sink), signals);
  session.wait();
  const LivenessEvidence evidence{
      .received_ns = sink->event_time_ns("RUNTIME_MANAGED_SIGNAL_RECEIVED"),
      .stop_observed_ns =
          holoscan::test::RuntimeProbeState::stop_observed_time_ns.load(std::memory_order_acquire),
      .callback_return_ns = holoscan::test::RuntimeProbeState::callback_return_time_ns.load(
          std::memory_order_acquire),
      .wait_return_ns = monotonic_time_ns(),
  };
  const bool wrote = write_liveness_evidence(evidence_fd, evidence);
  const auto closed = signals.close();
  return session.termination() == holoscan::RunTermination::kStopped &&
                 sink->event_count("RUNTIME_MANAGED_SIGNAL_RECEIVED") == 1U && wrote && closed
             ? 0
             : 1;
}

[[nodiscard]] int supervise_liveness(const holoscan::test::RuntimeProbeMode mode) {
  Pipe ready;
  Pipe received;
  Pipe release;
  Pipe evidence_pipe;
  if (!ready.open() || !received.open() || !release.open() || !evidence_pipe.open()) {
    return 1;
  }
  const pid_t child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    ready.close_read();
    received.close_read();
    release.close_write();
    evidence_pipe.close_read();
    _exit(
        run_liveness_child(mode, ready.fd[1], received.fd[1], release.fd[0], evidence_pipe.fd[1]));
  }
  ready.close_write();
  received.close_write();
  release.close_read();
  evidence_pipe.close_write();
  if (!read_byte(ready.fd[0])) {
    static_cast<void>(kill(child, SIGKILL));
    int ignored_status{};
    static_cast<void>(wait_for_child_exit(child, ignored_status));
    return 1;
  }
  const std::int64_t sent_ns = monotonic_time_ns();
  bool protocol_ok = kill(child, SIGINT) == 0 && read_byte(received.fd[0]);
  if (protocol_ok && mode == holoscan::test::RuntimeProbeMode::kIgnoreStopUntilReleased) {
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    protocol_ok = write_byte(release.fd[1]);
  }
  LivenessEvidence evidence{};
  protocol_ok = protocol_ok && read_liveness_evidence(evidence_pipe.fd[0], evidence);
  int status{};
  if (!protocol_ok || !wait_for_child_exit(child, status)) {
    static_cast<void>(kill(child, SIGKILL));
    return 1;
  }
  const std::int64_t signal_to_publication = evidence.received_ns - sent_ns;
  const std::int64_t publication_to_observation = evidence.stop_observed_ns - evidence.received_ns;
  const std::int64_t callback_return_to_join =
      evidence.wait_return_ns - evidence.callback_return_ns;
  std::fprintf(stderr,
               "managed-signal liveness ns: signal_to_publication=%lld "
               "publication_to_observation=%lld callback_return_to_join=%lld\n",
               static_cast<long long>(signal_to_publication),
               static_cast<long long>(publication_to_observation),
               static_cast<long long>(callback_return_to_join));
  constexpr std::int64_t kStructuralTestCeilingNs = 5'000'000'000LL;
  const bool ordered =
      signal_to_publication >= 0 && publication_to_observation >= 0 && callback_return_to_join >= 0;
  const bool finite_structural_intervals = signal_to_publication < kStructuralTestCeilingNs &&
                                           callback_return_to_join < kStructuralTestCeilingNs;
  const bool uncooperative_delay_observed =
      mode != holoscan::test::RuntimeProbeMode::kIgnoreStopUntilReleased ||
      publication_to_observation >= 25'000'000LL;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 && ordered && finite_structural_intervals &&
                 uncooperative_delay_observed
             ? 0
             : 1;
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc != 2 && argc != 4) {
    return 64;
  }
  const std::string_view scenario{argv[1]};
  if (scenario == "exec-mask-observer-blocked" || scenario == "exec-mask-observer-unblocked") {
    if (argc != 4) {
      return 64;
    }
    const ManagedDescriptorCounts expected_counts{
        .signal = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)),
        .event = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)),
    };
    return run_exec_mask_observer(scenario == "exec-mask-observer-blocked", expected_counts);
  }
  if (argc != 2) {
    return 64;
  }
  if (scenario == "complete") {
    return holoscan::run(make_plan());
  }
  if (scenario == "sigint") {
    return supervise_first_signal(ManagedChildMode::kTransient, SIGINT);
  }
  if (scenario == "sigint-blocked") {
    return supervise_first_signal(ManagedChildMode::kTransient, SIGINT, true);
  }
  if (scenario == "sigterm") {
    return supervise_first_signal(ManagedChildMode::kTransient, SIGTERM);
  }
  if (scenario == "domain-adopt") {
    return supervise_first_signal(ManagedChildMode::kAdoptedBlocking, SIGINT);
  }
  if (scenario == "domain-adopt-spent-retry") {
    return run_domain_adopt_spent_retry();
  }
  if (scenario == "async-managed") {
    return supervise_first_signal(ManagedChildMode::kExplicitAsync, SIGTERM);
  }
  if (scenario == "domain-required") {
    return run_domain_required();
  }
  if (scenario == "caller-managed") {
    return run_caller_managed_with_peer();
  }
  if (scenario == "async-plain") {
    return run_plain_async_neutrality();
  }
  if (scenario == "int-int") {
    return supervise_second_signal(SIGINT, SIGINT, PriorPolicy::kDefault);
  }
  if (scenario == "term-term") {
    return supervise_second_signal(SIGTERM, SIGTERM, PriorPolicy::kDefault);
  }
  if (scenario == "int-term") {
    return supervise_second_signal(SIGINT, SIGTERM, PriorPolicy::kDefault);
  }
  if (scenario == "term-int") {
    return supervise_second_signal(SIGTERM, SIGINT, PriorPolicy::kDefault);
  }
  if (scenario == "second-ignore") {
    return supervise_second_signal(SIGINT, SIGTERM, PriorPolicy::kIgnored);
  }
  if (scenario == "second-custom") {
    return supervise_second_signal(SIGTERM, SIGINT, PriorPolicy::kCustom);
  }
  if (scenario == "persistent-second-default") {
    return supervise_second_signal(SIGINT, SIGTERM, PriorPolicy::kDefault, true);
  }
  if (scenario == "persistent-second-ignore") {
    return supervise_second_signal(SIGTERM, SIGINT, PriorPolicy::kIgnored, true);
  }
  if (scenario == "persistent-second-custom") {
    return supervise_second_signal(SIGINT, SIGTERM, PriorPolicy::kCustom, true);
  }
  if (scenario == "persistent-second-custom-originally-blocked") {
    return supervise_second_signal(SIGINT, SIGTERM, PriorPolicy::kCustom, true, true);
  }
  if (scenario == "automatic-multi-owner-signals") {
    return supervise_automatic_multi_owner_signals(
        AutomaticMultiOwnerSignalExpectation::kEscalated);
  }
  if (scenario == "automatic-multi-owner-child-failure") {
    return supervise_automatic_multi_owner_signals(
        AutomaticMultiOwnerSignalExpectation::kChildFailureAfterFirstSignal);
  }
  if (scenario == "third-after-escalation-custom") {
    return supervise_third_signal_after_escalation();
  }
  if (scenario == "asymmetric-mask") {
    return supervise_one_signal_child(SIGTERM, &run_asymmetric_mask_child);
  }
  if (scenario == "first-custom-preserved") {
    return supervise_one_signal_child(SIGINT, &run_first_custom_child);
  }
  if (scenario == "close-after-wait") {
    return run_close_after_wait();
  }
  if (scenario == "wrong-thread-close") {
    return run_wrong_thread_close();
  }
  if (scenario == "owner-busy") {
    return run_managed_owner_busy();
  }
  if (scenario == "caller-managed-domain") {
    return supervise_caller_managed_inside_domain();
  }
  if (scenario == "wrong-thread-bind") {
    return run_wrong_thread_bind();
  }
  if (scenario == "fork-rejected") {
    return run_fork_rejection();
  }
  if (scenario == "exec-mask-inheritance-and-remedies") {
    return run_exec_mask_inheritance_and_remedies();
  }
  if (scenario == "fork-locked-close-rejected") {
    return run_forked_registry_rejection(ForkedRegistryOperation::kClose);
  }
  if (scenario == "fork-locked-destroy-rejected") {
    return run_forked_registry_rejection(ForkedRegistryOperation::kDestroy);
  }
  if (scenario == "fork-locked-create-rejected") {
    return run_forked_registry_rejection(ForkedRegistryOperation::kCreate);
  }
  if (scenario == "fork-locked-default-run-rejected") {
    return run_forked_registry_rejection(ForkedRegistryOperation::kDefaultRun);
  }
  if (scenario == "resource-loop") {
    return run_resource_loop();
  }
  if (scenario == "owner-busy-establishing") {
    return run_owner_busy_while_establishing();
  }
  if (scenario == "census-changed-during-setup") {
    return run_census_changed_during_setup();
  }
  if (scenario == "pending-during-establishment") {
    return run_pending_during_establishment();
  }
  if (scenario == "pending-establishment-rollback") {
    return run_pending_establishment_rollback();
  }
  if (scenario == "failure-adopt-binding") {
    return run_adopted_binding_failure();
  }
  if (scenario == "worker-start-failure") {
    return run_worker_start_failure();
  }
  if (scenario == "close-while-bound") {
    return run_close_while_bound();
  }
  if (scenario == "unbound-destructor") {
    return run_unbound_destructor_releases_domain();
  }
  if (scenario == "generation-monotonic") {
    return supervise_generation_monotonic();
  }
  if (scenario == "generation-exhaustion") {
    return supervise_generation_exhaustion();
  }
  if (scenario == "invalid-policy") {
    return run_invalid_policy();
  }
  if (scenario == "failure-setup") {
    return run_setup_failure_matrix();
  }
  if (scenario == "failure-start-broker") {
    return run_start_broker_failure();
  }
  if (scenario == "failure-broker-task-open") {
    return run_broker_task_open_failure();
  }
  if (scenario == "failure-setup-cleanup") {
    return run_setup_primary_and_cleanup_failure();
  }
  if (scenario == "idle-bound-event-driven") {
    return supervise_bound_event_driven_idle();
  }
  if (scenario == "idle-unbound-event-driven") {
    return run_unbound_event_driven_idle();
  }
  if (scenario == "terminal-wake-before-close") {
    return supervise_terminal_wake_before_close();
  }
  if (scenario == "failure-bound-wait-no-spin") {
    return run_bound_wait_failure_no_spin();
  }
  if (scenario == "failure-unbound-wait-retires") {
    return run_unbound_wait_failure_retires();
  }
  if (scenario == "failure-notify") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kNotifyController,
                                 holoscan::test::RuntimeProbeMode::kWaitForStop,
                                 "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kNotifyController);
  }
  if (scenario == "notify-failure-blocked-controller") {
    return supervise_blocked_notify_failure();
  }
  if (scenario == "failure-drain") {
    return run_transient_failure(
        holoscan::detail::RuntimeManagedSignalTestPoint::kDrainNotification,
        holoscan::test::RuntimeProbeMode::kWaitForStop,
        "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
        holoscan::detail::RuntimeManagedSignalPhase::kDrainNotification);
  }
  if (scenario == "failure-fixed-state") {
    return run_transient_failure(
        holoscan::detail::RuntimeManagedSignalTestPoint::kConsumeFixedState,
        holoscan::test::RuntimeProbeMode::kWaitForStop,
        "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
        holoscan::detail::RuntimeManagedSignalPhase::kRead);
  }
  if (scenario == "failure-wait-cleanup") {
    return run_wait_primary_and_cleanup_failure();
  }
  if (scenario == "failure-persistent-drain") {
    return run_persistent_drain_failure();
  }
  if (scenario == "failure-poll") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kPoll,
                                 holoscan::test::RuntimeProbeMode::kWaitForStop,
                                 "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kPoll);
  }
  if (scenario == "failure-read-signal") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kReadSignal,
                                 holoscan::test::RuntimeProbeMode::kWaitForStop,
                                 "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kRead);
  }
  if (scenario == "failure-final-drain") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kFinalDrain,
                                 holoscan::test::RuntimeProbeMode::kComplete,
                                 "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kFinalDrain);
  }
  if (scenario == "failure-terminal-release") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kTerminalRelease,
                                 holoscan::test::RuntimeProbeMode::kComplete,
                                 "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kBindingCleanup);
  }
  if (scenario == "failure-close-signal") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kCloseSignalFd,
                                 holoscan::test::RuntimeProbeMode::kComplete,
                                 "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kBindingCleanup);
  }
  if (scenario == "failure-close-notification") {
    return run_transient_failure(
        holoscan::detail::RuntimeManagedSignalTestPoint::kCloseNotification,
        holoscan::test::RuntimeProbeMode::kComplete,
        "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
        holoscan::detail::RuntimeManagedSignalPhase::kBindingCleanup);
  }
  if (scenario == "failure-restore-mask") {
    return run_transient_failure(holoscan::detail::RuntimeManagedSignalTestPoint::kRestoreMask,
                                 holoscan::test::RuntimeProbeMode::kComplete,
                                 "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
                                 holoscan::detail::RuntimeManagedSignalPhase::kRestoreMask);
  }
  if (scenario == "failure-close-domain") {
    return run_close_domain_failure();
  }
  if (scenario == "failure-join-broker") {
    return run_broker_join_failure();
  }
  if (scenario == "failure-broker-retirement") {
    return run_broker_retirement_failure();
  }
  if (scenario == "failure-delegate-signal") {
    return supervise_delegate_failure();
  }
  if (scenario == "failure-second-restore-mask") {
    return supervise_second_restore_failure(false);
  }
  if (scenario == "failure-persistent-second-restore-mask") {
    return supervise_second_restore_failure(true);
  }
  if (scenario == "disposition-drift") {
    return run_disposition_drift();
  }
  if (scenario == "disposition-drift-transient") {
    return run_transient_disposition_drift();
  }
  if (scenario == "signal-vs-failure") {
    return supervise_terminal_precedence(holoscan::test::RuntimeProbeMode::kWaitForStopThenFail,
                                         holoscan::RunTermination::kFailed);
  }
  if (scenario == "signal-vs-requested-completion") {
    return supervise_terminal_precedence(
        holoscan::test::RuntimeProbeMode::kRequestCompletionThenWaitForStop,
        holoscan::RunTermination::kStopped);
  }
  if (scenario == "pending-before-attach") {
    return supervise_checkpoint_signal(CheckpointScenario::kBeforeAttach);
  }
  if (scenario == "broker-thread-directed") {
    return supervise_broker_directed_signal();
  }
  if (scenario == "same-signal-coalesced") {
    return supervise_pending_batch(false);
  }
  if (scenario == "mixed-signals-pending") {
    return supervise_pending_batch(true);
  }
  if (scenario == "signal-at-completed-candidate") {
    return supervise_checkpoint_signal(CheckpointScenario::kCompletedCandidate);
  }
  if (scenario == "signal-at-failed-candidate") {
    return supervise_checkpoint_signal(CheckpointScenario::kFailedCandidate);
  }
  if (scenario == "signal-after-freeze-forwarded") {
    return run_signal_after_freeze_forwarded();
  }
  if (scenario == "blocked-signal-after-freeze-pending") {
    return run_blocked_signal_after_freeze_pending();
  }
  if (scenario == "liveness-cooperative") {
    return supervise_liveness(holoscan::test::RuntimeProbeMode::kWaitForStop);
  }
  if (scenario == "liveness-polling") {
    return supervise_liveness(holoscan::test::RuntimeProbeMode::kPollForStop);
  }
  if (scenario == "liveness-uncooperative") {
    return supervise_liveness(holoscan::test::RuntimeProbeMode::kIgnoreStopUntilReleased);
  }
  return 64;
}
