/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// EA1 two-process runtime smoke. A parent Graph process exchanges scalar
// samples with a genuinely exec'd standalone child in both directions:
// a child ChannelPublisher feeds a Graph import, and a Graph export feeds a
// child ChannelSubscriber. The only shared contract is the serialized
// endpoint catalog; the child never inherits compiled state, C++ objects, or
// transport handles.

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/predicate.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Thread-safe sink used by ImportSink to accumulate received sample values and
// signal the parent test thread when enough samples have arrived.
struct ImportProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    values.fill(0U);
    count.store(0U, std::memory_order_release);
  }

  static void record(const std::uint32_t value) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < values.size()) {
      values[index] = value;
      count.store(index + 1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<std::uint32_t, 4U> values{};
};

// Snapshot of the values, capture timestamps, and predicate metadata produced
// by one temporal-semantic sink invocation; used for cross-locality comparison.
struct SemanticObservation {
  std::vector<std::uint32_t> values;
  std::vector<std::int64_t> capture_times;
  std::size_t predicate_evaluations{};
  std::uint32_t predicate_driver{};
  std::uint32_t predicate_additional{};

  friend bool operator==(const SemanticObservation&, const SemanticObservation&) = default;
};

// Thread-safe accumulator shared by temporal-semantic sink operators
// (DrainSink, WindowSink, PredicateSink). Records the selected batch and
// predicate metadata, then signals the parent test thread via `wait()`.
struct SemanticProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    observation = {};
    complete = false;
  }

  static void record_batch(const holoscan::SelectedBatchView<std::uint32_t>& selected) {
    const std::lock_guard lock{mutex};
    observation.values.clear();
    observation.capture_times.clear();
    observation.values.reserve(selected.size());
    observation.capture_times.reserve(selected.size());
    for (const auto& sample : selected) {
      observation.values.push_back(sample.data);
      observation.capture_times.push_back(sample.metadata.capture_timestamp_ns);
    }
    complete = true;
    ready.notify_all();
  }

  static void record_predicate_evaluation(const std::uint32_t driver,
                                          const std::uint32_t additional) {
    const std::lock_guard lock{mutex};
    ++observation.predicate_evaluations;
    observation.predicate_driver = driver;
    observation.predicate_additional = additional;
  }

  static void record_predicate_compute(const holoscan::Sample<std::uint32_t>& driver,
                                       const holoscan::Sample<std::uint32_t>& additional) {
    const std::lock_guard lock{mutex};
    observation.values = {driver.data, additional.data};
    observation.capture_times = {driver.metadata.capture_timestamp_ns,
                                 additional.metadata.capture_timestamp_ns};
    complete = true;
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return complete; });
  }

  [[nodiscard]] static SemanticObservation snapshot() {
    const std::lock_guard lock{mutex};
    return observation;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline SemanticObservation observation;
  static inline bool complete{};
};

// Sink operator used in child-publisher scenarios. Receives each cross-process
// sample on a LatestWithin input and records the value to ImportProbe so the
// parent test thread can verify count and order.
class ImportSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{})
        .input(input, holoscan::LatestWithin{.max_age = 1s})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample.error()));
    }
    ImportProbe::record(sample->data);
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

// Source operator used in the Graph-export scenario. Fires on a 20 ms clock
// and emits a fixed marker value to the export topic so the child subscriber
// can verify it receives at least one sample.
class ExportSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 20ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto emitted = output.emit(0xdecafbadU);
    // A lost or unproven publication consumes that sample's authority; the
    // next tick emits the next logical sample. Only structural errors fail.
    if (!emitted && emitted.error().code != holoscan::ErrorCode::kNotReady &&
        emitted.error().code != holoscan::ErrorCode::kBackpressured &&
        emitted.error().code != holoscan::ErrorCode::kPublicationIndeterminate) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    return {};
  }

  holoscan::Output<std::uint32_t> output;
};

// Temporal-semantic sink for the DrainAll scenario. Fires via OnEach{driver}
// and drains all accumulated `values` samples in one batch, recording them to
// SemanticProbe for cross-locality comparison.
class DrainSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(values, "values").queue_depth(8U);
    spec.input(driver, "driver").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{driver}).input(values, holoscan::DrainAll{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto selected = values.selected_batch();
    if (!selected) {
      return holoscan::make_unexpected(std::move(selected.error()));
    }
    auto trigger = driver.receive();
    if (!trigger) {
      return holoscan::make_unexpected(std::move(trigger.error()));
    }
    SemanticProbe::record_batch(*selected);
    return {};
  }

  holoscan::Input<std::uint32_t> values;
  holoscan::Input<std::uint32_t> driver;
};

// Temporal-semantic sink for the BracketedWindow scenario. Uses OnWindow with
// a ±1-sample lookahead/lookbehind around each anchor to verify that the
// runtime assembles the correct time-aligned window across the process boundary.
class WindowSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(anchor, "anchor").queue_depth(4U);
    spec.input(window, "window").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnWindow{.anchor = anchor, .max_wait = 1s, .max_pending_windows = 2U})
        .input(anchor, holoscan::Exact{})
        .input(window,
               holoscan::BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 3U})
        .on_missing(window, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto selected_anchor = anchor.receive();
    if (!selected_anchor) {
      return holoscan::make_unexpected(std::move(selected_anchor.error()));
    }
    auto selected = window.selected_batch();
    if (!selected) {
      return holoscan::make_unexpected(std::move(selected.error()));
    }
    SemanticProbe::record_batch(*selected);
    return {};
  }

  holoscan::Input<std::uint32_t> anchor;
  holoscan::Input<std::uint32_t> window;
};

// Temporal-semantic sink for the OnPredicate scenario. The evaluator fires only
// when driver==7 AND additional==70; both the predicate call count and the
// compute payload are recorded to SemanticProbe for cross-locality comparison.
class PredicateSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(driver, "driver").queue_depth(4U);
    spec.input(additional, "additional").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    auto evaluator = holoscan::predicate(
        "interprocess-input-predicate",
        holoscan::predicate_inputs(additional),
        [](const holoscan::PredicateEvaluationContext&,
           const holoscan::PredicateInputView<std::uint32_t>& driver_view,
           const holoscan::PredicateInputView<std::uint32_t>& additional_view) noexcept {
          SemanticProbe::record_predicate_evaluation(driver_view.sample().data,
                                                     additional_view.sample().data);
          return driver_view.sample().data == 7U && additional_view.sample().data == 70U;
        });
    holoscan::Contract result;
    result
        .trigger(holoscan::OnPredicate{
            .evaluate_on = holoscan::OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = holoscan::PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, holoscan::Exact{})
        .input(additional, holoscan::Exact{})
        .on_missing(additional, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto selected_driver = driver.receive();
    auto selected_additional = additional.receive();
    if (!selected_driver) {
      return holoscan::make_unexpected(std::move(selected_driver.error()));
    }
    if (!selected_additional) {
      return holoscan::make_unexpected(std::move(selected_additional.error()));
    }
    SemanticProbe::record_predicate_compute(*selected_driver, *selected_additional);
    return {};
  }

  holoscan::Input<std::uint32_t> driver;
  holoscan::Input<std::uint32_t> additional;
};

// RAII guard for a fork+exec'd child process. Holds the two pipe FDs:
//   start_fd — parent writes a byte to unblock the child from its wait loop.
//   result_fd — child writes status codes (ready=1, then exit=0) back to parent.
// The destructor kills and reaps the child if it is still alive.
struct ChildProcess {
  ChildProcess() = default;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  ~ChildProcess() { terminate_and_reap(); }

  void close_parent_pipes() noexcept {
    if (result_fd >= 0) {
      ::close(result_fd);
      result_fd = -1;
    }
    if (start_fd >= 0) {
      ::close(start_fd);
      start_fd = -1;
    }
  }

  void terminate_and_reap() noexcept {
    close_parent_pipes();
    if (pid <= 0) {
      return;
    }

    int status{};
    pid_t result{};
    do {
      result = ::waitpid(pid, &status, WNOHANG);
    } while (result < 0 && errno == EINTR);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      pid = -1;
      return;
    }

    static_cast<void>(::kill(pid, SIGKILL));
    do {
      result = ::waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    pid = -1;
  }

  pid_t pid{-1};
  int result_fd{-1};
  int start_fd{-1};
};

constexpr auto kChildIoTimeout = 30s;
constexpr auto kChildJoinTimeout = 5s;

void close_pipe(std::array<int, 2U>& pipe_fds) noexcept {
  for (int& fd : pipe_fds) {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
}

[[nodiscard]] bool wait_for_fd(int fd, short events,
                               std::chrono::steady_clock::time_point deadline) {
  while (true) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return false;
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const auto bounded =
        std::min<std::int64_t>(remaining.count() + 1, std::numeric_limits<int>::max());
    pollfd descriptor{.fd = fd, .events = events, .revents = 0};
    const int result = ::poll(&descriptor, 1U, static_cast<int>(bounded));
    if (result > 0) {
      if ((descriptor.revents & events) != 0) {
        return true;
      }
      if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return false;
      }
    } else if (result < 0 && errno != EINTR) {
      return false;
    }
  }
}

[[nodiscard]] bool read_code(const int fd, std::uint8_t& code) {
  const auto deadline = std::chrono::steady_clock::now() + kChildIoTimeout;
  if (!wait_for_fd(fd, POLLIN, deadline)) {
    return false;
  }
  while (true) {
    const ssize_t result = ::read(fd, &code, sizeof(code));
    if (result == static_cast<ssize_t>(sizeof(code))) {
      return true;
    }
    if (result < 0 && errno == EINTR && std::chrono::steady_clock::now() < deadline) {
      continue;
    }
    return false;
  }
}

[[nodiscard]] bool spawn_child(const char* role, const std::string& catalog_path,
                               const std::string& endpoint_name, ChildProcess& child,
                               const std::string& second_endpoint_name = {}) {
  const char* const child_exe = HOLOSCAN_EA1_INTERPROCESS_RUNTIME_CHILD_EXE;
  if (child_exe[0] == '\0') {
    return false;
  }
  std::array<int, 2U> result_pipe{};
  std::array<int, 2U> start_pipe{};
  result_pipe.fill(-1);
  start_pipe.fill(-1);
  if (::pipe(result_pipe.data()) != 0) {
    return false;
  }
  if (::pipe(start_pipe.data()) != 0) {
    close_pipe(result_pipe);
    return false;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    close_pipe(result_pipe);
    close_pipe(start_pipe);
    return false;
  }
  if (pid == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    static_cast<void>(::setenv("HOLOSCAN_IPR_SMOKE_ROLE", role, 1));
    static_cast<void>(::setenv("HOLOSCAN_IPR_SMOKE_CATALOG", catalog_path.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_IPR_SMOKE_ENDPOINT", endpoint_name.c_str(), 1));
    if (!second_endpoint_name.empty()) {
      static_cast<void>(
          ::setenv("HOLOSCAN_IPR_SMOKE_SECOND_ENDPOINT", second_endpoint_name.c_str(), 1));
    }
    static_cast<void>(::setenv("HOLOSCAN_IPR_SMOKE_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_IPR_SMOKE_START_FD", start_fd.c_str(), 1));
    ::execl(child_exe, "ea1_interprocess_runtime_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  child.pid = pid;
  child.result_fd = result_pipe[0];
  child.start_fd = start_pipe[1];
  return true;
}

[[nodiscard]] bool release_child(const ChildProcess& child) {
  constexpr std::uint8_t kStart = 1U;
  const auto deadline = std::chrono::steady_clock::now() + kChildIoTimeout;
  if (!wait_for_fd(child.start_fd, POLLOUT, deadline)) {
    return false;
  }
  while (true) {
    const ssize_t result = ::write(child.start_fd, &kStart, sizeof(kStart));
    if (result == static_cast<ssize_t>(sizeof(kStart))) {
      return true;
    }
    if (result < 0 && errno == EINTR && std::chrono::steady_clock::now() < deadline) {
      continue;
    }
    return false;
  }
}

[[nodiscard]] int join_child(ChildProcess& child) {
  child.close_parent_pipes();
  const auto deadline = std::chrono::steady_clock::now() + kChildJoinTimeout;
  int status{};
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t result = ::waitpid(child.pid, &status, WNOHANG);
    if (result == child.pid) {
      child.pid = -1;
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    if (result < 0 && errno != EINTR) {
      child.pid = -1;
      return -1;
    }
    std::this_thread::sleep_for(2ms);
  }
  child.terminate_and_reap();
  return -1;
}

[[nodiscard]] std::string unique_name(const char* stem) {
  return std::string(stem) + "_" + std::to_string(::getpid()) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}

[[nodiscard]] std::string write_catalog(const holoscan::ChannelEndpointCatalog& catalog,
                                        const std::string& stem) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / (stem + "-catalog.json");
  auto saved = catalog.save_json(path);
  if (!saved) {
    return {};
  }
  return path.string();
}

[[nodiscard]] int fail(const char* stage) {
  std::cerr << "ea1_interprocess_runtime_smoke failed: " << stage << '\n';
  return 1;
}

[[nodiscard]] bool observe_interprocess_plan(const holoscan::ExecutionPlan& plan, const char* role,
                                             const std::string& first_endpoint,
                                             const std::string& second_endpoint,
                                             SemanticObservation& observation, const char* stage) {
  if (!plan.ok()) {
    std::cerr << stage << " plan rejected:";
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << ' ' << diagnostic.token;
    }
    std::cerr << '\n';
    return false;
  }
  const std::string catalog_path = write_catalog(plan.endpoint_catalog(), first_endpoint);
  if (catalog_path.empty()) {
    std::cerr << stage << " catalog save failed\n";
    return false;
  }
  ChildProcess child;
  if (!spawn_child(role, catalog_path, first_endpoint, child, second_endpoint)) {
    std::filesystem::remove(catalog_path);
    std::cerr << stage << " child spawn failed\n";
    return false;
  }

  SemanticProbe::reset();
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool released = release_child(child);
  std::uint8_t ready = 0xffU;
  const bool child_ready = released && read_code(child.result_fd, ready) && ready == 1U;
  const bool observed = child_ready && SemanticProbe::wait();
  std::uint8_t child_code = 0xffU;
  const bool child_reported = child_ready && read_code(child.result_fd, child_code);
  session.request_stop();
  session.wait();
  const int child_exit = join_child(child);
  std::filesystem::remove(catalog_path);
  if (!released || !child_ready || !observed || !child_reported || child_code != 0U ||
      child_exit != 0 || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << stage << " released=" << released << " ready=" << child_ready
              << " observed=" << observed << " child_reported=" << child_reported
              << " child_code=" << static_cast<unsigned>(child_code) << " child_exit=" << child_exit
              << " termination=" << static_cast<int>(session.termination()) << " diagnostics=";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << ' ';
    }
    std::cerr << '\n';
    return false;
  }
  observation = SemanticProbe::snapshot();
  return true;
}

[[nodiscard]] bool wait_for_subscriber(holoscan::ChannelPublisher<std::uint32_t>& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!publisher.has_subscribers()) {
    if (!publisher.update_connections() || std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(2ms);
  }
  return publisher.update_connections().has_value();
}

[[nodiscard]] bool emit_captured_until(holoscan::ChannelPublisher<std::uint32_t>& publisher,
                                       const std::uint32_t value, const std::int64_t capture_ns) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (true) {
    auto emitted = publisher.emit_captured(value, capture_ns);
    if (emitted) {
      return true;
    }
    if ((emitted.error().code != holoscan::ErrorCode::kNotReady &&
         emitted.error().code != holoscan::ErrorCode::kBackpressured) ||
        std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(2ms);
  }
}

using TemporalPublisher = bool (*)(holoscan::ChannelRuntime&,
                                   const holoscan::ChannelEndpointCatalog&, const std::string&,
                                   const std::string&);

[[nodiscard]] bool publish_drain(holoscan::ChannelRuntime& runtime,
                                 const holoscan::ChannelEndpointCatalog& catalog,
                                 const std::string& values_endpoint,
                                 const std::string& driver_endpoint) {
  auto values = runtime.create_publisher<std::uint32_t>(catalog, values_endpoint);
  auto driver = runtime.create_publisher<std::uint32_t>(catalog, driver_endpoint);
  if (!values || !driver || !wait_for_subscriber(*values) || !wait_for_subscriber(*driver)) {
    return false;
  }
  for (std::uint32_t value = 1U; value <= 3U; ++value) {
    if (!emit_captured_until(*values, value, static_cast<std::int64_t>(value) * 10)) {
      return false;
    }
  }
  return emit_captured_until(*driver, 99U, 40);
}

[[nodiscard]] bool publish_window(holoscan::ChannelRuntime& runtime,
                                  const holoscan::ChannelEndpointCatalog& catalog,
                                  const std::string& anchor_endpoint,
                                  const std::string& window_endpoint) {
  auto anchor = runtime.create_publisher<std::uint32_t>(catalog, anchor_endpoint);
  auto window = runtime.create_publisher<std::uint32_t>(catalog, window_endpoint);
  return anchor && window && wait_for_subscriber(*anchor) && wait_for_subscriber(*window) &&
         emit_captured_until(*window, 5U, 5) && emit_captured_until(*anchor, 10U, 10) &&
         emit_captured_until(*window, 10U, 10) && emit_captured_until(*anchor, 11U, 11) &&
         emit_captured_until(*window, 15U, 15);
}

[[nodiscard]] bool publish_predicate(holoscan::ChannelRuntime& runtime,
                                     const holoscan::ChannelEndpointCatalog& catalog,
                                     const std::string& driver_endpoint,
                                     const std::string& additional_endpoint) {
  auto driver = runtime.create_publisher<std::uint32_t>(catalog, driver_endpoint);
  auto additional = runtime.create_publisher<std::uint32_t>(catalog, additional_endpoint);
  if (!driver || !additional || !wait_for_subscriber(*driver) ||
      !wait_for_subscriber(*additional) || !emit_captured_until(*additional, 70U, 70)) {
    return false;
  }
  std::this_thread::sleep_for(50ms);
  return emit_captured_until(*driver, 7U, 70);
}

[[nodiscard]] bool observe_same_process_plan(const holoscan::ExecutionPlan& plan,
                                             const std::string& first_endpoint,
                                             const std::string& second_endpoint,
                                             const TemporalPublisher publish,
                                             SemanticObservation& observation, const char* stage) {
  if (!plan.ok()) {
    std::cerr << stage << " plan rejected:";
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << ' ' << diagnostic.token;
    }
    std::cerr << '\n';
    return false;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << stage << " ChannelRuntime creation failed\n";
    return false;
  }
  SemanticProbe::reset();
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool published =
      publish(*runtime, plan.endpoint_catalog(), first_endpoint, second_endpoint);
  const bool observed = published && SemanticProbe::wait();
  session.request_stop();
  session.wait();
  if (!published || !observed || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << stage << " published=" << published << " observed=" << observed
              << " termination=" << static_cast<int>(session.termination()) << " diagnostics=";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << ' ';
    }
    std::cerr << '\n';
    return false;
  }
  observation = SemanticProbe::snapshot();
  return true;
}

[[nodiscard]] SemanticObservation expected_drain_observation() {
  return SemanticObservation{
      .values = {1U, 2U, 3U},
      .capture_times = {10, 20, 30},
  };
}

[[nodiscard]] SemanticObservation expected_window_observation() {
  return SemanticObservation{
      .values = {5U, 10U, 15U},
      .capture_times = {5, 10, 15},
  };
}

[[nodiscard]] SemanticObservation expected_predicate_observation() {
  return SemanticObservation{
      .values = {7U, 70U},
      .capture_times = {70, 70},
      .predicate_evaluations = 1U,
      .predicate_driver = 7U,
      .predicate_additional = 70U,
  };
}

[[nodiscard]] int run_temporal_locality_parity() {
  SemanticObservation local_drain;
  {
    const std::string values_name = unique_name("local_drain_values");
    const std::string driver_name = unique_name("local_drain_driver");
    holoscan::Graph graph{"ea1-local-drain-parity"};
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    holoscan::TopicOptions values_options;
    values_options.endpoint_name = values_name;
    values_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    holoscan::TopicOptions driver_options;
    driver_options.endpoint_name = driver_name;
    driver_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto values =
        graph.import_topic<std::uint32_t>("holomq://" + values_name, std::move(values_options));
    const auto driver =
        graph.import_topic<std::uint32_t>("holomq://" + driver_name, std::move(driver_options));
    const auto sink = graph.op<DrainSink>("sink");
    graph.add_flow(values.output(), sink->values, holoscan::ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(driver.output(), sink->driver, holoscan::ConnectionOptions{.queue_depth = 2U});
    if (!observe_same_process_plan(compile(graph),
                                   values_name,
                                   driver_name,
                                   publish_drain,
                                   local_drain,
                                   "local DrainAll")) {
      return fail("local DrainAll parity");
    }
  }

  SemanticObservation cross_drain;
  {
    const std::string values_name = unique_name("ipr_drain_values");
    const std::string driver_name = unique_name("ipr_drain_driver");
    holoscan::Graph graph{"ea1-cross-drain-parity"};
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    holoscan::TopicOptions values_options;
    values_options.endpoint_name = values_name;
    values_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    holoscan::TopicOptions driver_options;
    driver_options.endpoint_name = driver_name;
    driver_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto values =
        graph.import_topic<std::uint32_t>("holomq://" + values_name, std::move(values_options));
    const auto driver =
        graph.import_topic<std::uint32_t>("holomq://" + driver_name, std::move(driver_options));
    const auto sink = graph.op<DrainSink>("sink");
    graph.add_flow(values.output(), sink->values, holoscan::ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(driver.output(), sink->driver, holoscan::ConnectionOptions{.queue_depth = 2U});
    if (!observe_interprocess_plan(compile(graph),
                                   "temporal-drain-publisher",
                                   values_name,
                                   driver_name,
                                   cross_drain,
                                   "cross DrainAll")) {
      return fail("cross DrainAll parity");
    }
  }
  if (local_drain != expected_drain_observation() || cross_drain != local_drain) {
    return fail("DrainAll locality observations differ");
  }

  SemanticObservation local_window;
  {
    const std::string anchor_name = unique_name("local_window_anchor");
    const std::string window_name = unique_name("local_window_samples");
    holoscan::Graph graph{"ea1-local-window-parity"};
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    holoscan::TopicOptions anchor_options;
    anchor_options.endpoint_name = anchor_name;
    anchor_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    holoscan::TopicOptions window_options;
    window_options.endpoint_name = window_name;
    window_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto anchor =
        graph.import_topic<std::uint32_t>("holomq://" + anchor_name, std::move(anchor_options));
    const auto window =
        graph.import_topic<std::uint32_t>("holomq://" + window_name, std::move(window_options));
    const auto sink = graph.op<WindowSink>("sink");
    graph.add_flow(anchor.output(), sink->anchor, holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(window.output(), sink->window, holoscan::ConnectionOptions{.queue_depth = 8U});
    if (!observe_same_process_plan(compile(graph),
                                   anchor_name,
                                   window_name,
                                   publish_window,
                                   local_window,
                                   "local BracketedWindow")) {
      return fail("local BracketedWindow parity");
    }
  }

  SemanticObservation cross_window;
  {
    const std::string anchor_name = unique_name("ipr_window_anchor");
    const std::string window_name = unique_name("ipr_window_samples");
    holoscan::Graph graph{"ea1-cross-window-parity"};
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    holoscan::TopicOptions anchor_options;
    anchor_options.endpoint_name = anchor_name;
    anchor_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    holoscan::TopicOptions window_options;
    window_options.endpoint_name = window_name;
    window_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto anchor =
        graph.import_topic<std::uint32_t>("holomq://" + anchor_name, std::move(anchor_options));
    const auto window =
        graph.import_topic<std::uint32_t>("holomq://" + window_name, std::move(window_options));
    const auto sink = graph.op<WindowSink>("sink");
    graph.add_flow(anchor.output(), sink->anchor, holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(window.output(), sink->window, holoscan::ConnectionOptions{.queue_depth = 8U});
    if (!observe_interprocess_plan(compile(graph),
                                   "temporal-window-publisher",
                                   anchor_name,
                                   window_name,
                                   cross_window,
                                   "cross BracketedWindow")) {
      return fail("cross BracketedWindow parity");
    }
  }
  if (local_window != expected_window_observation() || cross_window != local_window) {
    return fail("BracketedWindow locality observations differ");
  }

  SemanticObservation local_predicate;
  {
    const std::string driver_name = unique_name("local_predicate_driver");
    const std::string additional_name = unique_name("local_predicate_additional");
    holoscan::Graph graph{"ea1-local-predicate-parity"};
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    holoscan::TopicOptions driver_options;
    driver_options.endpoint_name = driver_name;
    driver_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    holoscan::TopicOptions additional_options;
    additional_options.endpoint_name = additional_name;
    additional_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto driver =
        graph.import_topic<std::uint32_t>("holomq://" + driver_name, std::move(driver_options));
    const auto additional = graph.import_topic<std::uint32_t>("holomq://" + additional_name,
                                                              std::move(additional_options));
    const auto sink = graph.op<PredicateSink>("sink");
    graph.add_flow(driver.output(), sink->driver, holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(
        additional.output(), sink->additional, holoscan::ConnectionOptions{.queue_depth = 4U});
    if (!observe_same_process_plan(compile(graph),
                                   driver_name,
                                   additional_name,
                                   publish_predicate,
                                   local_predicate,
                                   "local input predicate")) {
      return fail("local predicate parity");
    }
  }

  SemanticObservation cross_predicate;
  {
    const std::string driver_name = unique_name("ipr_predicate_driver");
    const std::string additional_name = unique_name("ipr_predicate_additional");
    holoscan::Graph graph{"ea1-cross-predicate-parity"};
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    holoscan::TopicOptions driver_options;
    driver_options.endpoint_name = driver_name;
    driver_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    holoscan::TopicOptions additional_options;
    additional_options.endpoint_name = additional_name;
    additional_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto driver =
        graph.import_topic<std::uint32_t>("holomq://" + driver_name, std::move(driver_options));
    const auto additional = graph.import_topic<std::uint32_t>("holomq://" + additional_name,
                                                              std::move(additional_options));
    const auto sink = graph.op<PredicateSink>("sink");
    graph.add_flow(driver.output(), sink->driver, holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(
        additional.output(), sink->additional, holoscan::ConnectionOptions{.queue_depth = 4U});
    if (!observe_interprocess_plan(compile(graph),
                                   "temporal-predicate-publisher",
                                   driver_name,
                                   additional_name,
                                   cross_predicate,
                                   "cross input predicate")) {
      return fail("cross predicate parity");
    }
  }
  if (local_predicate != expected_predicate_observation() || cross_predicate != local_predicate) {
    return fail("input-predicate locality observations differ");
  }
  return 0;
}

[[nodiscard]] int run_child_publisher_feeds_graph_import() {
  ImportProbe::reset();
  const std::string endpoint_name = unique_name("ipr_import");

  holoscan::Graph graph{"ea1-interprocess-import"};
  const auto capture_clock =
      graph.add_clock<holoscan::ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});

  const holoscan::ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << "import diagnostic: " << diagnostic.token << '\n';
    }
    return fail("import plan rejected");
  }
  const std::string catalog_path = write_catalog(plan.endpoint_catalog(), endpoint_name);
  if (catalog_path.empty()) {
    return fail("import catalog save");
  }

  ChildProcess child;
  if (!spawn_child("publisher", catalog_path, endpoint_name, child)) {
    return fail("spawn publisher child");
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  if (!release_child(child)) {
    session.request_stop();
    session.wait();
    return fail("release publisher child");
  }
  std::uint8_t ready = 0xffU;
  if (!read_code(child.result_fd, ready) || ready != 1U) {
    session.request_stop();
    session.wait();
    static_cast<void>(join_child(child));
    return fail("publisher child ready");
  }
  const bool received = ImportProbe::wait_for_count(3U);
  session.request_stop();
  session.wait();
  std::uint8_t child_code = 0xffU;
  const bool child_reported = read_code(child.result_fd, child_code);
  const int child_exit = join_child(child);
  std::filesystem::remove(catalog_path);
  if (!received) {
    std::cerr << "import probe count=" << ImportProbe::count.load(std::memory_order_acquire)
              << " child_reported=" << child_reported
              << " child_code=" << static_cast<unsigned>(child_code) << " child_exit=" << child_exit
              << " termination=" << static_cast<int>(session.termination()) << " diagnostics="
              << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
              << '\n';
    return fail("graph import did not observe three child samples");
  }
  if (ImportProbe::values[0] != 41U || ImportProbe::values[1] != 42U ||
      ImportProbe::values[2] != 43U) {
    return fail("graph import observed wrong values or order");
  }
  if (!child_reported || child_code != 0U || child_exit != 0) {
    return fail("publisher child validation");
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return fail("import session termination");
  }
  return 0;
}

[[nodiscard]] int run_graph_export_feeds_child_subscriber() {
  const std::string endpoint_name = unique_name("ipr_export");

  holoscan::Graph graph{"ea1-interprocess-export"};
  const auto runtime_clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto source = graph.op<ExportSource>("source");
  graph.add_flow(source->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 4U});

  const holoscan::ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << "export diagnostic: " << diagnostic.token << '\n';
    }
    return fail("export plan rejected");
  }
  const std::string catalog_path = write_catalog(plan.endpoint_catalog(), endpoint_name);
  if (catalog_path.empty()) {
    return fail("export catalog save");
  }

  ChildProcess child;
  if (!spawn_child("subscriber", catalog_path, endpoint_name, child)) {
    return fail("spawn subscriber child");
  }
  if (!release_child(child)) {
    static_cast<void>(join_child(child));
    return fail("release subscriber child");
  }
  std::uint8_t ready = 0xffU;
  if (!read_code(child.result_fd, ready) || ready != 1U) {
    static_cast<void>(join_child(child));
    return fail("subscriber child ready");
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  std::uint8_t child_code = 0xffU;
  const bool child_reported = read_code(child.result_fd, child_code);
  session.request_stop();
  session.wait();
  const int child_exit = join_child(child);
  std::filesystem::remove(catalog_path);
  if (!child_reported || child_code != 0U || child_exit != 0) {
    std::cerr << "export child_reported=" << child_reported
              << " child_code=" << static_cast<unsigned>(child_code) << " child_exit=" << child_exit
              << " termination=" << static_cast<int>(session.termination()) << " diagnostics=";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << ' ';
    }
    std::cerr << '\n';
    return fail("subscriber child validation");
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return fail("export session termination");
  }
  return 0;
}

// The exact serialized catalog is the only cross-process contract; both
// processes must resolve identical canonical bytes and service bindings.
[[nodiscard]] int run_catalog_round_trip_identity() {
  const std::string endpoint_name = unique_name("ipr_standalone");

  holoscan::Graph graph{"ea1-interprocess-standalone-probe"};
  const auto capture_clock =
      graph.add_clock<holoscan::ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  const holoscan::ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    return fail("standalone probe plan rejected");
  }
  const std::string catalog_path = write_catalog(plan.endpoint_catalog(), endpoint_name);
  if (catalog_path.empty()) {
    return fail("standalone probe catalog save");
  }

  auto reloaded = holoscan::ChannelEndpointCatalog::load_json(catalog_path);
  if (!reloaded) {
    return fail("standalone probe reload");
  }
  if (reloaded->json() != plan.endpoint_catalog().json()) {
    std::cerr << "compiled catalog json:\n"
              << plan.endpoint_catalog().json() << '\n'
              << "reloaded catalog json:\n"
              << reloaded->json() << '\n';
    return fail("standalone probe catalog round-trip drift");
  }
  std::filesystem::remove(catalog_path);
  return 0;
}

// Q61-T12 cross-process arm: publisher-serialized clock-health order must be
// receiver-applied across the real process boundary. H1 precedes S1/S2
// eligibility; H2 makes S3 stale-unhealthy so the Standard-profile Skip
// disposition suppresses it without failing the run.
[[nodiscard]] int run_monitored_import_across_process() {
  ImportProbe::reset();
  const std::string endpoint_name = unique_name("ipr_monitored");

  holoscan::Graph graph{"ea1-interprocess-monitored-import"};
  const auto capture_clock =
      graph.add_clock<holoscan::ManualClock>("capture-clock", std::int64_t{1000});
  const auto age_clock = graph.add_clock<holoscan::RealtimeClock>("age-clock");
  graph.set_default_clock(age_clock);
  graph.add_clock_conversion(capture_clock, age_clock, holoscan::AffineClockConversion{});
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
  options.clock_health = holoscan::ClockHealthOptions{
      .capability = holoscan::ClockHealthCapability::kStatusReports,
      .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});

  const holoscan::ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << "monitored diagnostic: " << diagnostic.token << '\n';
    }
    return fail("monitored plan rejected");
  }
  const std::string catalog_path = write_catalog(plan.endpoint_catalog(), endpoint_name);
  if (catalog_path.empty()) {
    return fail("monitored catalog save");
  }

  ChildProcess child;
  if (!spawn_child("monitored-publisher", catalog_path, endpoint_name, child)) {
    return fail("spawn monitored child");
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  if (!release_child(child)) {
    session.request_stop();
    session.wait();
    return fail("release monitored child");
  }
  std::uint8_t ready = 0xffU;
  if (!read_code(child.result_fd, ready) || ready != 1U) {
    session.request_stop();
    session.wait();
    static_cast<void>(join_child(child));
    return fail("monitored child ready");
  }
  const bool received = ImportProbe::wait_for_count(2U);
  std::uint8_t child_code = 0xffU;
  const bool child_reported = read_code(child.result_fd, child_code);
  // Give the suppressed stale-unhealthy sample time to arrive before stopping.
  std::this_thread::sleep_for(300ms);
  const std::size_t final_count = ImportProbe::count.load(std::memory_order_acquire);
  session.request_stop();
  session.wait();
  const int child_exit = join_child(child);
  std::filesystem::remove(catalog_path);
  if (!child_reported || child_code != 0U || child_exit != 0) {
    std::cerr << "monitored child_reported=" << child_reported
              << " child_code=" << static_cast<unsigned>(child_code) << " child_exit=" << child_exit
              << '\n';
    return fail("monitored child validation");
  }
  if (!received || ImportProbe::values[0] != 41U || ImportProbe::values[1] != 42U) {
    return fail("monitored import did not observe the two healthy samples");
  }
  if (final_count != 2U) {
    return fail("stale-unhealthy sample was not suppressed across the process boundary");
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return fail("monitored session termination");
  }
  return 0;
}

// Child loss must not hang cooperative teardown, leak an unbounded claim, or
// silently change graph temporal semantics: the run continues and stops
// cleanly after the peer vanishes without transport-level cleanup.
[[nodiscard]] int run_child_loss_reclamation() {
  ImportProbe::reset();
  const std::string endpoint_name = unique_name("ipr_childloss");

  holoscan::Graph graph{"ea1-interprocess-child-loss"};
  const auto capture_clock =
      graph.add_clock<holoscan::ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});

  const holoscan::ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    return fail("child-loss plan rejected");
  }
  const std::string catalog_path = write_catalog(plan.endpoint_catalog(), endpoint_name);
  if (catalog_path.empty()) {
    return fail("child-loss catalog save");
  }

  ChildProcess child;
  if (!spawn_child("abrupt-publisher", catalog_path, endpoint_name, child)) {
    return fail("spawn abrupt child");
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  if (!release_child(child)) {
    session.request_stop();
    session.wait();
    return fail("release abrupt child");
  }
  std::uint8_t ready = 0xffU;
  if (!read_code(child.result_fd, ready) || ready != 1U) {
    session.request_stop();
    session.wait();
    static_cast<void>(join_child(child));
    return fail("abrupt child ready");
  }
  const bool received = ImportProbe::wait_for_count(1U);
  std::uint8_t child_code = 0xffU;
  const bool child_reported = read_code(child.result_fd, child_code);
  const int child_exit = join_child(child);
  if (!received || !child_reported || child_code != 0U || child_exit != 0) {
    session.request_stop();
    session.wait();
    std::filesystem::remove(catalog_path);
    return fail("abrupt child did not deliver its sample before exiting");
  }
  // The peer is gone with no transport cleanup. The session must keep running
  // and then stop cleanly within the ordinary cooperative path.
  std::this_thread::sleep_for(300ms);
  if (!session.is_running()) {
    std::filesystem::remove(catalog_path);
    return fail("session failed after abrupt peer loss");
  }
  session.request_stop();
  session.wait();
  std::filesystem::remove(catalog_path);
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return fail("child-loss session termination");
  }
  if (ImportProbe::values[0] != 77U) {
    return fail("child-loss sample value");
  }
  return 0;
}

}  // namespace

// REQ: CHAN-086
// REQ: CHAN-089
//
// Overall data-flow topology exercised by this smoke test:
//
//  Scenario A — child publisher → Graph import (both directions tested):
//    child:ChannelPublisher ──[holomq]──> import_topic ──> ImportSink
//    ExportSource ──> export_topic ──[holomq]──> child:ChannelSubscriber
//
//  Scenario B — temporal locality parity (same-process vs. cross-process):
//    publisher ──[holomq]──> import_topic ──> DrainSink
//    publisher ──[holomq]──> import_topic ──> WindowSink
//    publisher ──[holomq]──> import_topic ──> PredicateSink
//    Results must be bit-identical whether publisher is in-process or exec'd.
//
//  Scenario C — catalog round-trip identity:
//    compile(graph) → save JSON → reload JSON  (no transport, no child)
//
//  Scenario D — clock-health-monitored import across process boundary
//  Scenario E — abrupt child loss followed by clean cooperative teardown
int main() {
  // Pipe peers are independently failing processes. Convert peer exit into EPIPE
  // so the parent can execute its bounded kill/reap cleanup instead of dying.
  if (::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    return 1;  // exit 1: SIGPIPE disposition setup failed
  }
  const int temporal_parity = run_temporal_locality_parity();
  if (temporal_parity != 0) {
    // exit: same-process and cross-process temporal-semantic results diverged
    return temporal_parity;
  }
  const int standalone_probe = run_catalog_round_trip_identity();
  if (standalone_probe != 0) {
    // exit: serialized endpoint catalog does not survive a JSON round-trip
    return standalone_probe;
  }
  const int export_result = run_graph_export_feeds_child_subscriber();
  if (export_result != 0) {
    // exit: Graph export topic did not deliver samples to the child subscriber
    return export_result;
  }
  const int import_result = run_child_publisher_feeds_graph_import();
  if (import_result != 0) {
    // exit: child ChannelPublisher samples were not received by the Graph import
    return import_result;
  }
  const int monitored_result = run_monitored_import_across_process();
  if (monitored_result != 0) {
    // exit: clock-health filtering was not applied correctly across the boundary
    return monitored_result;
  }
  const int child_loss_result = run_child_loss_reclamation();
  if (child_loss_result != 0) {
    // exit: abrupt peer loss caused a hang, leak, or non-kStopped termination
    return child_loss_result;
  }
  std::cout << "ea1_interprocess_runtime_smoke passed\n";
  return 0;
}
