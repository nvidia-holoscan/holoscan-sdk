/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Bridge an application-owned Linux eventfd into one bounded NotificationSource.
//
// The Operator acquires its generation-bound NotificationSender at kArm,
// starts exactly one std::jthread at kStart, cooperatively joins it at kStop,
// and closes the descriptor at kRelease. The worker retries the documented
// pre-start kNotReady and in-flight kBackpressured outcomes; it never detaches.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stop_token>
#include <string_view>
#include <thread>
#include <utility>

#if defined(__linux__) && !defined(HOLOSCAN_NOTIFICATION_BRIDGE_FORCE_UNSUPPORTED)

#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kExpectedNotifications = 4U;
constexpr std::chrono::milliseconds kPollInterval = 100ms;
constexpr std::chrono::milliseconds kPostRetryInterval = 1ms;

struct BridgeObservation {
  static void reset() noexcept {
    const std::lock_guard lock(mutex);
    activations = 0U;
    pre_start_post = holoscan::ErrorCode::kFailure;
    lifecycle_error = holoscan::ErrorCode::kSuccess;
    worker_error = holoscan::ErrorCode::kSuccess;
    callback_error = holoscan::ErrorCode::kSuccess;
    worker_started = false;
    worker_joined = false;
    released = false;
  }

  static std::size_t record_activation() {
    std::size_t result = 0U;
    {
      const std::lock_guard lock(mutex);
      ++activations;
      result = activations;
    }
    ready.notify_all();
    return result;
  }

  static void record_pre_start(holoscan::ErrorCode code) noexcept {
    const std::lock_guard lock(mutex);
    pre_start_post = code;
  }

  static void record_lifecycle_error(holoscan::ErrorCode code) noexcept {
    const std::lock_guard lock(mutex);
    lifecycle_error = code;
    ready.notify_all();
  }

  static void record_worker_error(holoscan::ErrorCode code) noexcept {
    const std::lock_guard lock(mutex);
    worker_error = code;
    ready.notify_all();
  }

  static void record_callback_error(holoscan::ErrorCode code) noexcept {
    const std::lock_guard lock(mutex);
    callback_error = code;
    ready.notify_all();
  }

  static void record_worker_started() noexcept {
    const std::lock_guard lock(mutex);
    worker_started = true;
    ready.notify_all();
  }

  static void record_worker_joined() noexcept {
    const std::lock_guard lock(mutex);
    worker_joined = true;
    ready.notify_all();
  }

  static void record_released() noexcept {
    const std::lock_guard lock(mutex);
    released = true;
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_notifications(bool expect_failure) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expect_failure] {
      return (expect_failure ? callback_error != holoscan::ErrorCode::kSuccess
                             : activations >= kExpectedNotifications) ||
             lifecycle_error != holoscan::ErrorCode::kSuccess;
    });
  }

  [[nodiscard]] static bool valid_after_wait(bool expect_failure) noexcept {
    const std::lock_guard lock(mutex);
    const bool common = pre_start_post == holoscan::ErrorCode::kNotReady &&
                        lifecycle_error == holoscan::ErrorCode::kSuccess && worker_started &&
                        worker_joined && released;
    if (expect_failure) {
      return common && activations == 0U && worker_error == holoscan::ErrorCode::kFailure &&
             callback_error == holoscan::ErrorCode::kFailure;
    }
    return common && activations == kExpectedNotifications &&
           worker_error == holoscan::ErrorCode::kSuccess &&
           callback_error == holoscan::ErrorCode::kSuccess;
  }

  static void print_failure_state() {
    const std::lock_guard lock(mutex);
    std::cerr << " activations=" << activations
              << " pre_start_post=" << holoscan::error_code_name(pre_start_post)
              << " lifecycle_error=" << holoscan::error_code_name(lifecycle_error)
              << " worker_error=" << holoscan::error_code_name(worker_error)
              << " callback_error=" << holoscan::error_code_name(callback_error)
              << " worker_started=" << worker_started << " worker_joined=" << worker_joined
              << " released=" << released;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t activations{};
  static inline holoscan::ErrorCode pre_start_post{holoscan::ErrorCode::kFailure};
  static inline holoscan::ErrorCode lifecycle_error{holoscan::ErrorCode::kSuccess};
  static inline holoscan::ErrorCode worker_error{holoscan::ErrorCode::kSuccess};
  static inline holoscan::ErrorCode callback_error{holoscan::ErrorCode::kSuccess};
  static inline bool worker_started{};
  static inline bool worker_joined{};
  static inline bool released{};
};

class EventfdNotificationBridge final : public holoscan::Operator<> {
 public:
  explicit EventfdNotificationBridge(bool simulate_worker_failure)
      : simulate_worker_failure_(simulate_worker_failure) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.notification_source(notification_, "eventfd-notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm, &EventfdNotificationBridge::on_arm)
        .stage(holoscan::LifecycleStage::kStart, &EventfdNotificationBridge::on_start)
        .stage(holoscan::LifecycleStage::kStop, &EventfdNotificationBridge::on_stop)
        .stage(holoscan::LifecycleStage::kRelease, &EventfdNotificationBridge::on_release);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnNotified{.event = notification_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const holoscan::ErrorCode worker_error = worker_error_.load(std::memory_order_acquire);
    if (worker_error != holoscan::ErrorCode::kSuccess) {
      BridgeObservation::record_callback_error(worker_error);
      return holoscan::make_unexpected(holoscan::Error{worker_error});
    }

    static_cast<void>(BridgeObservation::record_activation());
    return {};
  }

 private:
  holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext& context) noexcept {
    fd_ = ::eventfd(0U, EFD_CLOEXEC | EFD_NONBLOCK | EFD_SEMAPHORE);
    if (fd_ < 0) {
      BridgeObservation::record_lifecycle_error(holoscan::ErrorCode::kResourceExhausted);
      return holoscan::LifecycleStatus::kFatalFailure;
    }

    auto sender = context.notification_sender(notification_);
    if (!sender) {
      BridgeObservation::record_lifecycle_error(sender.error().code);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    sender_ = std::move(*sender);

    const auto pre_start = sender_.post();
    const holoscan::ErrorCode pre_start_code =
        pre_start ? holoscan::ErrorCode::kSuccess : pre_start.error().code;
    BridgeObservation::record_pre_start(pre_start_code);
    if (pre_start_code != holoscan::ErrorCode::kNotReady) {
      BridgeObservation::record_lifecycle_error(holoscan::ErrorCode::kProtocolError);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept {
    // Seed a bounded burst before the worker starts. EFD_SEMAPHORE then exposes exactly one
    // counter unit per read, so each external event maps to one bounded NotificationSender post.
    for (std::size_t index = 0U; index < kExpectedNotifications; ++index) {
      if (!signal_eventfd()) {
        BridgeObservation::record_lifecycle_error(holoscan::ErrorCode::kFailure);
        return holoscan::LifecycleStatus::kFatalFailure;
      }
    }
    try {
      worker_ = std::jthread(
          [this](const std::stop_token& stop_token) noexcept { bridge_loop(stop_token); });
    } catch (...) {
      BridgeObservation::record_lifecycle_error(holoscan::ErrorCode::kResourceExhausted);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    BridgeObservation::record_worker_started();
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept {
    if (!stop_and_join_worker()) {
      BridgeObservation::record_lifecycle_error(holoscan::ErrorCode::kFailure);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_release(holoscan::LifecycleContext&) noexcept {
    sender_ = holoscan::NotificationSender{};
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
      fd_ = -1;
    }
    BridgeObservation::record_released();
    return holoscan::LifecycleStatus::kOk;
  }

  [[nodiscard]] bool signal_eventfd() const noexcept {
    constexpr std::uint64_t kOne = 1U;
    const ssize_t written = ::write(fd_, &kOne, sizeof(kOne));
    return written == static_cast<ssize_t>(sizeof(kOne));
  }

  [[nodiscard]] bool stop_and_join_worker() noexcept {
    if (!worker_.joinable()) {
      return true;
    }
    worker_.request_stop();
    static_cast<void>(signal_eventfd());
    try {
      worker_.join();
    } catch (...) {
      return false;
    }
    BridgeObservation::record_worker_joined();
    return true;
  }

  [[nodiscard]] bool post_until_accepted(const std::stop_token& stop_token) noexcept {
    while (!stop_token.stop_requested()) {
      const auto posted = sender_.post();
      if (posted) {
        return true;
      }
      const holoscan::ErrorCode code = posted.error().code;
      if (code == holoscan::ErrorCode::kNotReady || code == holoscan::ErrorCode::kBackpressured) {
        std::this_thread::sleep_for(kPostRetryInterval);
        continue;
      }
      if (stop_token.stop_requested() &&
          (code == holoscan::ErrorCode::kClosed || code == holoscan::ErrorCode::kStaleEndpoint)) {
        return false;
      }
      worker_error_.store(code, std::memory_order_release);
      BridgeObservation::record_worker_error(code);
      return false;
    }
    return false;
  }

  void bridge_loop(const std::stop_token& stop_token) noexcept {
    while (!stop_token.stop_requested()) {
      pollfd waitable{.fd = fd_, .events = POLLIN, .revents = 0};
      const int ready = ::poll(&waitable, 1U, static_cast<int>(kPollInterval.count()));
      if (ready < 0) {
        if (errno == EINTR) {
          continue;
        }
        fail_worker_and_wake(holoscan::ErrorCode::kFailure, stop_token);
        return;
      }
      if (ready == 0) {
        continue;
      }
      if ((waitable.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        fail_worker_and_wake(holoscan::ErrorCode::kFailure, stop_token);
        return;
      }

      std::uint64_t count = 0U;
      const ssize_t bytes = ::read(fd_, &count, sizeof(count));
      if (bytes < 0 && errno == EAGAIN) {
        continue;
      }
      if (bytes != static_cast<ssize_t>(sizeof(count))) {
        fail_worker_and_wake(holoscan::ErrorCode::kFailure, stop_token);
        return;
      }
      if (count != 1U) {
        fail_worker_and_wake(holoscan::ErrorCode::kProtocolError, stop_token);
        return;
      }
      if (stop_token.stop_requested()) {
        return;
      }
      if (simulate_worker_failure_) {
        fail_worker_and_wake(holoscan::ErrorCode::kFailure, stop_token);
        return;
      }
      if (!post_until_accepted(stop_token)) {
        return;
      }
    }
  }

  void fail_worker_and_wake(holoscan::ErrorCode code, const std::stop_token& stop_token) noexcept {
    worker_error_.store(code, std::memory_order_release);
    BridgeObservation::record_worker_error(code);
    static_cast<void>(post_until_accepted(stop_token));
  }

  holoscan::NotificationSource notification_;
  holoscan::NotificationSender sender_;
  std::jthread worker_;
  std::atomic<holoscan::ErrorCode> worker_error_{holoscan::ErrorCode::kSuccess};
  bool simulate_worker_failure_{};
  int fd_{-1};
};

}  // namespace

int main(int argc, char** argv) {
  const bool simulate_worker_failure =
      argc == 2 && std::string_view(argv[1]) == "--simulate-worker-failure";
  if (argc > 2 || (argc == 2 && !simulate_worker_failure)) {
    std::cerr << "usage: notification_bridge [--simulate-worker-failure]\n";
    return 2;
  }
  BridgeObservation::reset();

  holoscan::Graph graph{"notification-bridge"};
  const auto bridge =
      graph.op<EventfdNotificationBridge>("eventfd-bridge", simulate_worker_failure);
  graph.partition("notification-bridge-runtime").add(bridge);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "notification-bridge-runtime",
      holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kSleepingWaitset});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = BridgeObservation::wait_for_notifications(simulate_worker_failure);
  if (!simulate_worker_failure || !observed) {
    session.request_stop();
  }
  session.wait();

  // Structured RunSession state is authoritative. This human-readable line is
  // emitted once only after shutdown and is not a second terminal channel.
  const bool normal_result = !simulate_worker_failure &&
                             session.termination() == holoscan::RunTermination::kStopped &&
                             session.diagnostics().empty();
  const bool failure_result =
      simulate_worker_failure && session.termination() == holoscan::RunTermination::kFailed &&
      session.diagnostics().size() == 1U &&
      session.diagnostics().front().token == "RUNTIME_OPERATOR_CALLBACK_FAILED" &&
      session.diagnostics().front().service == "operator_compute";
  if (!observed || (!normal_result && !failure_result) ||
      !BridgeObservation::valid_after_wait(simulate_worker_failure)) {
    std::cerr << "notification_bridge failed: termination="
              << static_cast<int>(session.termination()) << " diagnostics=";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << ',';
    }
    BridgeObservation::print_failure_state();
    std::cerr << '\n';
    return 2;
  }

  if (simulate_worker_failure) {
    std::cout << "notification_bridge: worker failure propagated through "
                 "RUNTIME_OPERATOR_CALLBACK_FAILED; lifecycle-owned worker joined\n";
    return 0;
  }
  std::cout << "notification_bridge: eventfd bridged 4 notifications; "
               "lifecycle-owned worker joined; termination=stopped\n";
  return 0;
}

#else

int main() {
  std::cerr << "SKIP: notification_bridge requires Linux eventfd; this platform does not "
               "provide the required waitable-descriptor capability\n";
  return 77;
}

#endif
