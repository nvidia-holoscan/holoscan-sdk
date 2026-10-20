/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/managed_shutdown_signals.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

#include "managed_shutdown_signals.hpp"
#include "managed_shutdown_signals_test_control.hpp"

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#endif

namespace holoscan::detail {

#if defined(__linux__)

namespace {

enum class ManagedSignalRegistryPhase : std::uint8_t {
  kUnowned,
  kEstablishing,
  kAvailable,
  kBound,
  kForwarding,
  kSpent,
  kPoisoned,
};

class ManagedSignalDomainState;
class ManagedSignalBinding;

struct ManagedSignalRegistry {
  std::atomic<pid_t> process_id{getpid()};
  std::mutex mutex;
  std::uint64_t next_generation{1U};
  std::uint64_t owner_generation{};
  std::uint64_t binding_generation{};
  ManagedSignalRegistryPhase phase{ManagedSignalRegistryPhase::kUnowned};
  std::shared_ptr<ManagedSignalDomainState> domain;
};

static_assert(std::atomic<pid_t>::is_always_lock_free);

[[nodiscard]] ManagedSignalRegistry& managed_signal_registry() noexcept {
  static ManagedSignalRegistry registry;
  return registry;
}

[[nodiscard]] bool inherited_managed_signal_registry() noexcept {
  const auto& registry = managed_signal_registry();
  return registry.process_id.load(std::memory_order_acquire) != getpid();
}

[[nodiscard]] expected<std::uint64_t, Error> reserve_generation_locked(
    ManagedSignalRegistry& registry) noexcept {
  if (registry.next_generation == 0U ||
      registry.next_generation == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return registry.next_generation++;
}

[[nodiscard]] RuntimeManagedSignal to_runtime_signal(const int signal) noexcept {
  if (signal == SIGINT) {
    return RuntimeManagedSignal::kSigInt;
  }
  if (signal == SIGTERM) {
    return RuntimeManagedSignal::kSigTerm;
  }
  return RuntimeManagedSignal::kNone;
}

[[nodiscard]] std::size_t signal_index(const int signal) noexcept {
  return signal == SIGTERM ? 1U : 0U;
}

[[nodiscard]] RuntimeManagedSignalDisposition classify_disposition(
    const struct sigaction& action) noexcept {
  if (action.sa_handler == SIG_DFL) {
    return RuntimeManagedSignalDisposition::kDefault;
  }
  if (action.sa_handler == SIG_IGN) {
    return RuntimeManagedSignalDisposition::kIgnored;
  }
  return RuntimeManagedSignalDisposition::kCustom;
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
  if ((left.sa_flags & SA_SIGINFO) != 0) {
    return left.sa_sigaction == right.sa_sigaction;
  }
  return left.sa_handler == right.sa_handler;
}

struct ThreadCensus {
  std::uint32_t count{};
  int error{};
};

[[nodiscard]] ThreadCensus read_thread_census() noexcept {
  DIR* const directory = opendir("/proc/self/task");
  if (directory == nullptr) {
    return ThreadCensus{.error = errno};
  }
  std::uint32_t count{};
  int read_error{};
  for (;;) {
    errno = 0;
    const dirent* const entry = readdir(directory);
    if (entry == nullptr) {
      read_error = errno;
      break;
    }
    if (entry->d_name[0] == '.' &&
        (entry->d_name[1] == '\0' || (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) {
      continue;
    }
    if (count != std::numeric_limits<std::uint32_t>::max()) {
      ++count;
    }
  }
  const int close_error = closedir(directory) == 0 ? 0 : errno;
  return ThreadCensus{.count = count, .error = read_error != 0 ? read_error : close_error};
}

[[nodiscard]] ManagedSignalAdmissionFailure admission_failure(
    const std::string_view token, const RuntimeManagedSignalPhase phase, const ErrorCode code,
    const int os_error = 0,
    const RuntimeManagedSignalCensusReason census_reason = RuntimeManagedSignalCensusReason::kNone,
    const std::uint32_t observed_threads = 0U, const std::uint64_t domain_generation = 0U,
    const std::uint64_t binding_generation = 0U, const bool poison_process = false) {
  return ManagedSignalAdmissionFailure{
      .token = token,
      .detail =
          RuntimeManagedSignalDiagnostic{
              .present = true,
              .phase = phase,
              .census_reason = census_reason,
              .domain_generation = domain_generation,
              .binding_generation = binding_generation,
              .observed_thread_count = observed_threads,
              .os_error = os_error,
          },
      .error = Error{code},
      .poison_process = poison_process,
  };
}

[[nodiscard]] int close_fd(int& fd) noexcept {
  if (fd < 0) {
    return 0;
  }
  const int owned = std::exchange(fd, -1);
  return close(owned) == 0 ? 0 : errno;
}

[[nodiscard]] int inject_failure(const std::shared_ptr<RuntimeManagedSignalTestControl>& control,
                                 const RuntimeManagedSignalTestPoint point) noexcept {
  return control != nullptr ? control->consume(point) : 0;
}

struct BrokerTaskProbe {
  bool retired{};
  int error{};
};

[[nodiscard]] BrokerTaskProbe probe_broker_task_retirement(const int task_fd) noexcept {
  int stat_fd{};
  do {
    stat_fd = openat(task_fd, "stat", O_RDONLY | O_CLOEXEC);
  } while (stat_fd < 0 && errno == EINTR);
  if (stat_fd < 0) {
    const int stat_error = errno;
    return BrokerTaskProbe{
        .retired = stat_error == ENOENT || stat_error == ESRCH,
        .error = stat_error == ENOENT || stat_error == ESRCH ? 0 : stat_error,
    };
  }

  char value{};
  ssize_t read_result{};
  do {
    read_result = read(stat_fd, &value, sizeof(value));
  } while (read_result < 0 && errno == EINTR);
  const int read_error = read_result < 0 ? errno : 0;
  const int stat_close_error = close(stat_fd) == 0 ? 0 : errno;
  if (read_result < 0 && (read_error == ENOENT || read_error == ESRCH)) {
    return BrokerTaskProbe{.retired = true, .error = stat_close_error};
  }
  if (read_result != sizeof(value)) {
    return BrokerTaskProbe{.error = read_error != 0 ? read_error : EIO};
  }
  return BrokerTaskProbe{.error = stat_close_error};
}

class ManagedSignalBinding final : public RuntimeTerminalParticipant,
                                   public ManagedSignalBindingController,
                                   public std::enable_shared_from_this<ManagedSignalBinding> {
 public:
  ManagedSignalBinding(std::shared_ptr<ManagedSignalDomainState> domain,
                       const std::uint64_t generation, const bool caller_drives,
                       std::shared_ptr<RuntimeManagedSignalTestControl> test_control) noexcept
      : domain_(std::move(domain)),
        generation_(generation),
        caller_drives_(caller_drives),
        test_control_(std::move(test_control)) {}

  [[nodiscard]] std::shared_ptr<RuntimeTerminalParticipant> terminal_participant() override {
    return shared_from_this();
  }

  [[nodiscard]] bool caller_drives() const noexcept override { return caller_drives_; }

  void drive() noexcept override;

  void attach(const RuntimeTerminalParticipantControl control) noexcept override {
    if (test_control_ != nullptr) {
      test_control_->checkpoint(RuntimeManagedSignalTestPoint::kBindingPublishedBeforeAttach);
    }
    {
      const std::lock_guard lock(mutex_);
      control_ = control;
      attached_ = control.state != nullptr;
    }
    cv_.notify_all();
    wake_controller();
  }

  [[nodiscard]] bool before_terminal_freeze(const RunTermination) noexcept override {
    std::unique_lock lock(mutex_);
    candidate_ready_ = true;
    worker_waiting_ = true;
    if (test_control_ != nullptr) {
      test_control_->checkpoint(RuntimeManagedSignalTestPoint::kCandidatePublishedBeforeWake);
    }
    wake_controller_locked();
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
    return !force_failed_;
  }

  void after_terminal_freeze() noexcept override {
    {
      const std::lock_guard lock(mutex_);
      terminal_frozen_ = true;
    }
    cv_.notify_all();
  }

  void wait_for_retirement() noexcept override {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return retired_; });
  }

  void wait_until_attached_or_aborted() noexcept {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return attached_ || aborted_; });
  }

  void stage_start_failure() noexcept override {
    {
      const std::lock_guard lock(mutex_);
      candidate_ready_ = true;
      worker_waiting_ = false;
      force_failed_ = true;
      wake_controller_locked();
    }
    cv_.notify_all();
  }

  void abort_before_session() noexcept override {
    {
      const std::lock_guard lock(mutex_);
      aborted_ = true;
      candidate_ready_ = true;
      worker_waiting_ = false;
      force_failed_ = true;
      wake_controller_locked();
    }
    cv_.notify_all();
    // A persistent domain is driven by its broker, so a pre-session failure must not
    // return an already-terminal RunSession while the binding is still published.
    // The transient path is driven later by the invoking run() thread and would
    // deadlock if it waited here.
    if (!caller_drives_) {
      wait_for_retirement();
    }
  }

  void wake_controller() noexcept;

  void wait_until_candidate_ready() noexcept {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return candidate_ready_ || aborted_; });
  }

  struct Snapshot {
    RuntimeTerminalParticipantControl control;
    bool attached{};
    bool candidate_ready{};
    bool worker_waiting{};
    bool aborted{};
    int first_signal{};
    int second_signal{};
    bool first_dispatched{};
    bool second_dispatched{};
    bool second_escalation_published{};
    int notification_error{};
    bool notification_failure_dispatched{};
    bool disposition_failure_dispatched{};
    bool controller_wait_failed{};
  };

  [[nodiscard]] Snapshot snapshot() const noexcept {
    const std::lock_guard lock(mutex_);
    return Snapshot{
        .control = control_,
        .attached = attached_,
        .candidate_ready = candidate_ready_,
        .worker_waiting = worker_waiting_,
        .aborted = aborted_,
        .first_signal = first_signal_,
        .second_signal = second_signal_,
        .first_dispatched = first_dispatched_,
        .second_dispatched = second_dispatched_,
        .second_escalation_published = second_escalation_published_,
        .notification_error = notification_error_,
        .notification_failure_dispatched = notification_failure_dispatched_,
        .disposition_failure_dispatched = disposition_failure_dispatched_,
        .controller_wait_failed = controller_wait_failed_,
    };
  }

  [[nodiscard]] RuntimeManagedSignalOccurrence accept_signal(const int signal) noexcept {
    const std::lock_guard lock(mutex_);
    if (first_signal_ == 0) {
      first_signal_ = signal;
      return RuntimeManagedSignalOccurrence::kFirst;
    }
    if (second_signal_ == 0) {
      second_signal_ = signal;
      return RuntimeManagedSignalOccurrence::kSecond;
    }
    return RuntimeManagedSignalOccurrence::kNone;
  }

  void mark_first_dispatched() noexcept {
    const std::lock_guard lock(mutex_);
    first_dispatched_ = true;
  }

  void mark_second_dispatched() noexcept {
    const std::lock_guard lock(mutex_);
    second_dispatched_ = true;
  }

  void mark_second_escalation_published() noexcept {
    const std::lock_guard lock(mutex_);
    second_escalation_published_ = true;
  }

  void fail() noexcept {
    const std::lock_guard lock(mutex_);
    force_failed_ = true;
  }

  [[nodiscard]] bool mark_controller_wait_failed() noexcept {
    const std::lock_guard lock(mutex_);
    force_failed_ = true;
    if (controller_wait_failed_) {
      return false;
    }
    controller_wait_failed_ = true;
    return true;
  }

  void mark_notification_failure_dispatched() noexcept {
    const std::lock_guard lock(mutex_);
    notification_failure_dispatched_ = true;
  }

  void mark_disposition_failure_dispatched() noexcept {
    const std::lock_guard lock(mutex_);
    disposition_failure_dispatched_ = true;
  }

  void release() noexcept {
    {
      const std::lock_guard lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

  void finish_start_failure() noexcept {
    const Snapshot state = snapshot();
    if (state.attached && state.control.finish_start_failure != nullptr) {
      state.control.finish_start_failure(state.control.state);
    } else {
      const std::lock_guard lock(mutex_);
      terminal_frozen_ = true;
      cv_.notify_all();
    }
  }

  void wait_until_terminal_frozen() noexcept {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return terminal_frozen_ || aborted_; });
  }

  void mark_retired() noexcept {
    {
      const std::lock_guard lock(mutex_);
      retired_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

  [[nodiscard]] const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control()
      const noexcept {
    return test_control_;
  }

 private:
  // The caller holds mutex_. Keeping terminal-candidate publication, logical notification
  // failure, and the durable eventfd write in one critical section prevents cleanup from
  // observing a candidate and closing the descriptor before its wake completes.
  void wake_controller_locked() noexcept;

  std::shared_ptr<ManagedSignalDomainState> domain_;
  std::uint64_t generation_{};
  bool caller_drives_{};
  const std::shared_ptr<RuntimeManagedSignalTestControl> test_control_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  RuntimeTerminalParticipantControl control_;
  bool attached_{};
  bool candidate_ready_{};
  bool worker_waiting_{};
  bool aborted_{};
  bool released_{};
  bool terminal_frozen_{};
  bool retired_{};
  bool force_failed_{};
  int first_signal_{};
  int second_signal_{};
  bool first_dispatched_{};
  bool second_dispatched_{};
  bool second_escalation_published_{};
  int notification_error_{};
  bool notification_failure_dispatched_{};
  bool disposition_failure_dispatched_{};
  bool controller_wait_failed_{};
};

class ManagedSignalDomainState final
    : public std::enable_shared_from_this<ManagedSignalDomainState> {
 public:
  ManagedSignalDomainState(const bool persistent, const std::uint64_t generation,
                           std::shared_ptr<RuntimeManagedSignalTestControl> test_control) noexcept
      : persistent_(persistent),
        generation_(generation),
        creator_thread_(pthread_self()),
        creator_cpp_thread_(std::this_thread::get_id()),
        creator_pid_(getpid()),
        setup_test_control_(std::move(test_control)) {
    sigemptyset(&managed_mask_);
    sigaddset(&managed_mask_, SIGINT);
    sigaddset(&managed_mask_, SIGTERM);
  }

  ManagedSignalDomainState(const ManagedSignalDomainState&) = delete;
  ManagedSignalDomainState& operator=(const ManagedSignalDomainState&) = delete;

  ~ManagedSignalDomainState() {
    if (broker_.joinable()) {
      std::terminate();
    }
    static_cast<void>(close_fd(broker_task_fd_));
    static_cast<void>(close_fd(signal_fd_));
    static_cast<void>(close_fd(control_fd_));
  }

  [[nodiscard]] static expected<std::shared_ptr<ManagedSignalDomainState>,
                                ManagedSignalAdmissionFailure>
  // The ordered acquisition and rollback steps form one auditable transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  establish(const bool persistent, const std::uint64_t generation,
            std::shared_ptr<RuntimeManagedSignalTestControl> test_control) {
    if (test_control != nullptr) {
      test_control->checkpoint(RuntimeManagedSignalTestPoint::kCensusBeforeMask);
    }
    if (const int test_error =
            inject_failure(test_control, RuntimeManagedSignalTestPoint::kCensusBeforeMask);
        test_error != 0) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kThreadCensus,
                                               ErrorCode::kFailure,
                                               test_error,
                                               RuntimeManagedSignalCensusReason::kCensusUnavailable,
                                               0U,
                                               generation));
    }
    const ThreadCensus first = read_thread_census();
    if (first.error != 0) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kThreadCensus,
                                               ErrorCode::kFailure,
                                               first.error,
                                               RuntimeManagedSignalCensusReason::kCensusUnavailable,
                                               first.count,
                                               generation));
    }
    if (first.count != 1U) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED",
                                               RuntimeManagedSignalPhase::kThreadCensus,
                                               ErrorCode::kInvalidArgument,
                                               0,
                                               RuntimeManagedSignalCensusReason::kMultipleThreads,
                                               first.count,
                                               generation));
    }

    std::shared_ptr<ManagedSignalDomainState> domain;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      domain = std::shared_ptr<ManagedSignalDomainState>(
          new ManagedSignalDomainState(persistent, generation, std::move(test_control)));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kAcquireDomain,
                                               ErrorCode::kResourceExhausted,
                                               0,
                                               RuntimeManagedSignalCensusReason::kNone,
                                               first.count,
                                               generation));
    }
#endif
    int mask_error =
        inject_failure(domain->setup_test_control_, RuntimeManagedSignalTestPoint::kSnapshotMask);
    if (mask_error == 0) {
      mask_error = pthread_sigmask(SIG_BLOCK, nullptr, &domain->prior_mask_);
    }
    if (mask_error != 0) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kSnapshotMask,
                                               ErrorCode::kFailure,
                                               mask_error,
                                               RuntimeManagedSignalCensusReason::kNone,
                                               first.count,
                                               generation));
    }
    mask_error =
        inject_failure(domain->setup_test_control_, RuntimeManagedSignalTestPoint::kBlockMask);
    if (mask_error == 0) {
      mask_error = pthread_sigmask(SIG_BLOCK, &domain->managed_mask_, nullptr);
    }
    if (mask_error != 0) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kBlockMask,
                                               ErrorCode::kFailure,
                                               mask_error,
                                               RuntimeManagedSignalCensusReason::kNone,
                                               first.count,
                                               generation));
    }
    domain->mask_owned_ = true;

    const int second_census_error = inject_failure(domain->setup_test_control_,
                                                   RuntimeManagedSignalTestPoint::kCensusAfterMask);
    if (domain->setup_test_control_ != nullptr) {
      domain->setup_test_control_->checkpoint(RuntimeManagedSignalTestPoint::kCensusAfterMask);
    }
    const ThreadCensus second = second_census_error == 0
                                    ? read_thread_census()
                                    : ThreadCensus{.error = second_census_error};
    if (second.error != 0 || second.count != 1U) {
      return make_unexpected(domain->rollback_setup(
          admission_failure(second.error != 0 ? "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED"
                                              : "RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED",
                            RuntimeManagedSignalPhase::kThreadCensus,
                            second.error != 0 ? ErrorCode::kFailure : ErrorCode::kInvalidArgument,
                            second.error,
                            second.error != 0 ? RuntimeManagedSignalCensusReason::kCensusUnavailable
                                              : RuntimeManagedSignalCensusReason::kCensusChanged,
                            second.count,
                            generation)));
    }

    for (const int signal : std::array<int, 2>{SIGINT, SIGTERM}) {
      const int injected_error = inject_failure(domain->setup_test_control_,
                                                RuntimeManagedSignalTestPoint::kQueryDisposition);
      if (injected_error != 0 ||
          sigaction(signal, nullptr, &domain->prior_actions_[signal_index(signal)]) != 0) {
        const int action_error = injected_error != 0 ? injected_error : errno;
        return make_unexpected(
            domain->rollback_setup(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                     RuntimeManagedSignalPhase::kQueryDisposition,
                                                     ErrorCode::kFailure,
                                                     action_error,
                                                     RuntimeManagedSignalCensusReason::kNone,
                                                     second.count,
                                                     generation)));
      }
    }

    if (domain->setup_test_control_ != nullptr) {
      domain->setup_test_control_->checkpoint(
          RuntimeManagedSignalTestPoint::kMaskBlockedBeforeSignalFd);
    }

    const int signal_fd_injected =
        inject_failure(domain->setup_test_control_, RuntimeManagedSignalTestPoint::kCreateSignalFd);
    domain->signal_fd_ = signal_fd_injected == 0
                             ? signalfd(-1, &domain->managed_mask_, SFD_CLOEXEC | SFD_NONBLOCK)
                             : -1;
    if (domain->signal_fd_ < 0) {
      const int signal_fd_error = signal_fd_injected != 0 ? signal_fd_injected : errno;
      return make_unexpected(
          domain->rollback_setup(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                   RuntimeManagedSignalPhase::kCreateSignalFd,
                                                   ErrorCode::kFailure,
                                                   signal_fd_error,
                                                   RuntimeManagedSignalCensusReason::kNone,
                                                   second.count,
                                                   generation)));
    }
    const int notification_injected = inject_failure(
        domain->setup_test_control_, RuntimeManagedSignalTestPoint::kCreateNotification);
    domain->control_fd_ = notification_injected == 0 ? eventfd(0U, EFD_CLOEXEC | EFD_NONBLOCK) : -1;
    if (domain->control_fd_ < 0) {
      const int notification_error = notification_injected != 0 ? notification_injected : errno;
      return make_unexpected(
          domain->rollback_setup(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                   RuntimeManagedSignalPhase::kCreateNotification,
                                                   ErrorCode::kFailure,
                                                   notification_error,
                                                   RuntimeManagedSignalCensusReason::kNone,
                                                   second.count,
                                                   generation)));
    }

    if (persistent) {
      const int broker_error =
          inject_failure(domain->setup_test_control_, RuntimeManagedSignalTestPoint::kStartBroker);
      if (broker_error != 0) {
        return make_unexpected(
            domain->rollback_setup(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                     RuntimeManagedSignalPhase::kStartBroker,
                                                     ErrorCode::kResourceExhausted,
                                                     broker_error,
                                                     RuntimeManagedSignalCensusReason::kNone,
                                                     second.count,
                                                     generation)));
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        domain->broker_ = std::thread([retained = domain] { retained->broker_main(); });
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        return make_unexpected(
            domain->rollback_setup(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                     RuntimeManagedSignalPhase::kStartBroker,
                                                     ErrorCode::kResourceExhausted,
                                                     0,
                                                     RuntimeManagedSignalCensusReason::kNone,
                                                     second.count,
                                                     generation)));
      }
#endif
      int broker_task_error{};
      {
        std::unique_lock lock(domain->mutex_);
        domain->domain_cv_.wait(lock, [&domain] { return domain->broker_ready_; });
        broker_task_error = domain->broker_task_error_;
      }
      if (broker_task_error != 0) {
        if (domain->broker_.joinable()) {
          domain->broker_.join();
        }
        auto failure = admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                         RuntimeManagedSignalPhase::kStartBroker,
                                         ErrorCode::kResourceExhausted,
                                         broker_task_error,
                                         RuntimeManagedSignalCensusReason::kNone,
                                         second.count,
                                         generation);
        const int retirement_error =
            domain->broker_task_fd_ < 0
                ? broker_task_error
                : domain->wait_for_broker_task_retirement(domain->setup_test_control_);
        if (retirement_error != 0) {
          failure = domain->add_cleanup_failure(
              std::move(failure), RuntimeManagedSignalPhase::kBindingCleanup, retirement_error);
        }
        return make_unexpected(domain->rollback_setup(std::move(failure)));
      }
    }
    return domain;
  }

  [[nodiscard]] expected<std::shared_ptr<ManagedSignalBinding>, ManagedSignalAdmissionFailure>
  reserve_binding(const bool caller_drives,
                  std::shared_ptr<RuntimeManagedSignalTestControl> test_control = {}) {
    auto& registry = managed_signal_registry();
    std::uint64_t binding_generation{};
    std::shared_ptr<ManagedSignalBinding> binding;
    {
      std::unique_lock domain_lock(mutex_);
      domain_cv_.wait(domain_lock, [this] { return !unbound_draining_; });
      const std::lock_guard registry_lock(registry.mutex);
      if (registry.domain.get() != this ||
          (registry.phase != ManagedSignalRegistryPhase::kAvailable &&
           !(caller_drives && registry.phase == ManagedSignalRegistryPhase::kEstablishing)) ||
          binding_used_ || close_requested_ || poisoned_.load(std::memory_order_acquire)) {
        return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_OWNER_BUSY",
                                                 RuntimeManagedSignalPhase::kAcquireBinding,
                                                 ErrorCode::kResourceExhausted,
                                                 0,
                                                 RuntimeManagedSignalCensusReason::kNone,
                                                 0U,
                                                 generation_));
      }
      if (test_control != nullptr) {
        test_control->checkpoint(RuntimeManagedSignalTestPoint::kRegistryLockedBeforeBinding);
      }
      if (const int test_error = inject_failure(
              test_control, RuntimeManagedSignalTestPoint::kReserveBindingGeneration);
          test_error != 0) {
        return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                 RuntimeManagedSignalPhase::kAcquireBinding,
                                                 ErrorCode::kResourceExhausted,
                                                 test_error,
                                                 RuntimeManagedSignalCensusReason::kNone,
                                                 0U,
                                                 generation_));
      }
      auto reserved = reserve_generation_locked(registry);
      if (!reserved) {
        return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                 RuntimeManagedSignalPhase::kAcquireBinding,
                                                 reserved.error().code,
                                                 0,
                                                 RuntimeManagedSignalCensusReason::kNone,
                                                 0U,
                                                 generation_));
      }
      binding_generation = *reserved;

      if (const int test_error =
              inject_failure(test_control, RuntimeManagedSignalTestPoint::kAcquireBinding);
          test_error != 0) {
        return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                 RuntimeManagedSignalPhase::kAcquireBinding,
                                                 ErrorCode::kResourceExhausted,
                                                 test_error,
                                                 RuntimeManagedSignalCensusReason::kNone,
                                                 0U,
                                                 generation_,
                                                 binding_generation));
      }

      // An unbound custom policy may legitimately update its own one-shot disposition. Snapshot
      // the binding's exact policy after the broker has left its unbound-forwarding boundary.
      for (const int signal : std::array<int, 2>{SIGINT, SIGTERM}) {
        if (sigaction(signal, nullptr, &prior_actions_[signal_index(signal)]) != 0) {
          return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                   RuntimeManagedSignalPhase::kQueryDisposition,
                                                   ErrorCode::kFailure,
                                                   errno,
                                                   RuntimeManagedSignalCensusReason::kNone,
                                                   0U,
                                                   generation_,
                                                   binding_generation));
        }
      }

      // Publish the strong binding while holding the same domain lock the broker uses to choose
      // between binding delivery and unbound forwarding. The registry becomes Bound only after
      // that publication, so neither signals nor an admission-abort wake can fall through a weak
      // ownership gap.
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        binding = std::make_shared<ManagedSignalBinding>(
            shared_from_this(), binding_generation, caller_drives, test_control);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                                 RuntimeManagedSignalPhase::kAcquireBinding,
                                                 ErrorCode::kResourceExhausted,
                                                 0,
                                                 RuntimeManagedSignalCensusReason::kNone,
                                                 0U,
                                                 generation_,
                                                 binding_generation));
      }
#endif
      binding_ = binding;
      close_test_control_ = std::move(test_control);
      binding_used_ = true;
      registry.binding_generation = binding_generation;
      registry.phase = ManagedSignalRegistryPhase::kBound;
    }
    binding->wake_controller();
    return binding;
  }

  [[nodiscard]] ManagedSignalAdmissionFailure abandon_unbound_transient(
      ManagedSignalAdmissionFailure failure) noexcept {
    failure = rollback_setup(std::move(failure));
    auto& registry = managed_signal_registry();
    const std::lock_guard lock(registry.mutex);
    if (registry.domain.get() == this) {
      if (failure.poison_process) {
        poisoned_.store(true, std::memory_order_release);
        registry.domain.reset();
        registry.owner_generation = 0U;
        registry.binding_generation = 0U;
        registry.phase = ManagedSignalRegistryPhase::kPoisoned;
      } else {
        registry.domain.reset();
        registry.owner_generation = 0U;
        registry.binding_generation = 0U;
        registry.phase = ManagedSignalRegistryPhase::kUnowned;
      }
    }
    return failure;
  }

  [[nodiscard]] int wake() noexcept {
    const std::lock_guard lock(descriptor_mutex_);
    if (control_fd_ < 0) {
      return EBADF;
    }
    const std::uint64_t one{1U};
    ssize_t written{};
    do {
      written = write(control_fd_, &one, sizeof(one));
    } while (written < 0 && errno == EINTR);
    if (std::cmp_equal(written, sizeof(one)) || (written < 0 && errno == EAGAIN)) {
      return 0;
    }
    return written < 0 ? errno : EIO;
  }

  void drive_binding(const std::shared_ptr<ManagedSignalBinding>& binding) noexcept {
    while (true) {
      const auto before = binding->snapshot();
      if (!before.attached && !before.aborted) {
        binding->wait_until_attached_or_aborted();
        continue;
      }
      if (before.aborted) {
        complete_binding(binding);
        return;
      }
      if (before.controller_wait_failed && !before.candidate_ready) {
        binding->wait_until_candidate_ready();
        continue;
      }
      if (binding->test_control() != nullptr) {
        binding->test_control()->checkpoint(
            RuntimeManagedSignalTestPoint::kBrokerReadyForDirectedSignal);
      }
      if (const int control_error = drain_control(binding->test_control()); control_error != 0) {
        record_wait_failure(binding, RuntimeManagedSignalPhase::kDrainNotification, control_error);
      }
      if (const int state_error = inject_failure(binding->test_control(),
                                                 RuntimeManagedSignalTestPoint::kConsumeFixedState);
          state_error != 0) {
        record_wait_failure(binding, RuntimeManagedSignalPhase::kRead, state_error);
      }
      if (!forwarding_) {
        drain_signals(binding, RuntimeManagedSignalPhase::kRead);
      }
      dispatch_binding(binding);
      const auto state = binding->snapshot();
      if (finish_binding_if_ready(binding, state)) {
        return;
      }
      if (state.controller_wait_failed || !wait_for_activity(binding)) {
        binding->wait_until_candidate_ready();
      }
    }
  }

  // Keeping best-effort cleanup and poisoning in one transaction makes ordering reviewable.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> close_public() noexcept {
    if (closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (!persistent_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (getpid() != creator_pid_) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (std::this_thread::get_id() != creator_cpp_thread_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::shared_ptr<RuntimeManagedSignalTestControl> close_test_control;
    {
      const std::lock_guard lock(mutex_);
      if (binding_ != nullptr) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      close_requested_ = true;
      close_test_control = close_test_control_;
    }
    if (const int test_error =
            inject_failure(close_test_control, RuntimeManagedSignalTestPoint::kNotifyController);
        test_error != 0) {
      record_broker_failure(test_error);
    }
    if (const int wake_error = wake(); wake_error != 0) {
      record_broker_failure(wake_error);
    }
    int broker_join_error{};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      if (broker_.joinable()) {
        broker_.join();
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      broker_join_error = EIO;
      // Do not let a transient join failure skip safe cleanup. Retry once so an exited broker can
      // still be reaped. If it remains joinable, retain the poisoned domain: closing its
      // descriptors or destroying its state would race an ordinary broker thread.
      try {
        if (broker_.joinable()) {
          broker_.join();
        }
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // broker_join_error records the retry failure; joinable() below selects containment.
      }
    }
#endif
    if (broker_.joinable()) {
      poison();
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (const int test_error =
            inject_failure(close_test_control, RuntimeManagedSignalTestPoint::kJoinBroker);
        test_error != 0) {
      broker_join_error = test_error;
    }
    const auto& retirement_test_control =
        close_test_control != nullptr ? close_test_control : setup_test_control_;
    if (const int retirement_error = wait_for_broker_task_retirement(retirement_test_control);
        retirement_error != 0 && broker_join_error == 0) {
      broker_join_error = retirement_error;
    }
    const int signal_close_error = close_signal_fd(close_test_control);
    const int control_close_error = close_control_fd(close_test_control);
    const int disposition_error = forwarding_ ? 0 : validate_prior_actions();
    const int restored = restore_creator_mask(close_test_control);
    const int cleanup_error =
        restored != 0 ? restored
                      : (signal_close_error != 0 ? signal_close_error : control_close_error);
    int broker_failure_error{};
    {
      const std::lock_guard lock(mutex_);
      broker_failure_error = broker_failure_error_;
    }
    const int close_domain_error =
        inject_failure(close_test_control, RuntimeManagedSignalTestPoint::kCloseDomain);
    const bool close_failed = poisoned_.load(std::memory_order_acquire) || broker_join_error != 0 ||
                              cleanup_error != 0 || broker_failure_error != 0 ||
                              close_domain_error != 0 || disposition_error != 0;
    if (close_failed) {
      poisoned_.store(true, std::memory_order_release);
    }
    auto& registry = managed_signal_registry();
    const std::lock_guard registry_lock(registry.mutex);
    if (registry.domain.get() == this) {
      if (close_failed) {
        registry.domain.reset();
        registry.owner_generation = 0U;
        registry.binding_generation = 0U;
        registry.phase = ManagedSignalRegistryPhase::kPoisoned;
      } else {
        registry.domain.reset();
        registry.owner_generation = 0U;
        registry.binding_generation = 0U;
        registry.phase = ManagedSignalRegistryPhase::kUnowned;
      }
    }
    closed_ = true;
    if (close_failed) {
      return make_unexpected(
          Error{disposition_error != 0 ? ErrorCode::kIncompatible : ErrorCode::kFailure});
    }
    return {};
  }

  void record_broker_failure(const int os_error) noexcept {
    {
      const std::lock_guard lock(mutex_);
      if (broker_failure_error_ == 0) {
        broker_failure_error_ = os_error;
      }
    }
    poison();
  }

  void poison() noexcept {
    poisoned_.store(true, std::memory_order_release);
    auto& registry = managed_signal_registry();
    const std::lock_guard lock(registry.mutex);
    if (registry.domain.get() == this) {
      registry.phase = ManagedSignalRegistryPhase::kPoisoned;
    }
  }

  [[nodiscard]] bool is_creator_thread() const noexcept {
    return getpid() == creator_pid_ && std::this_thread::get_id() == creator_cpp_thread_;
  }

  [[nodiscard]] bool is_forked_process() const noexcept { return getpid() != creator_pid_; }

  [[nodiscard]] bool persistent() const noexcept { return persistent_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

  void enable_broker() noexcept {
    if (!persistent_) {
      return;
    }
    {
      const std::lock_guard lock(mutex_);
      broker_enabled_ = true;
    }
    domain_cv_.notify_all();
  }

  void best_effort_destroy() noexcept {
    if (closed_) {
      return;
    }
    if (is_forked_process()) {
      return;
    }
    if (is_creator_thread()) {
      std::shared_ptr<ManagedSignalBinding> binding;
      {
        const std::lock_guard lock(mutex_);
        binding = binding_;
      }
      if (binding != nullptr) {
        request_binding_stop(binding);
        binding->wait_for_retirement();
      }
      const auto closed = close_public();
      if (closed) {
        return;
      }
    }
    poison();
  }

 private:
  friend class ManagedSignalBinding;

  [[nodiscard]] bool finish_binding_if_ready(const std::shared_ptr<ManagedSignalBinding>& binding,
                                             const ManagedSignalBinding::Snapshot& state) noexcept {
    if (!state.candidate_ready) {
      return false;
    }
    if (binding->test_control() != nullptr) {
      binding->test_control()->checkpoint(
          RuntimeManagedSignalTestPoint::kCandidateReadyBeforeFinalDrain);
    }
    if (!forwarding_) {
      drain_signals(binding, RuntimeManagedSignalPhase::kFinalDrain);
      dispatch_binding(binding);
    }
    complete_binding(binding);
    return true;
  }

  [[nodiscard]] int restore_creator_mask(
      const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control) noexcept {
    if (!mask_owned_) {
      return 0;
    }
    const int result = pthread_sigmask(SIG_SETMASK, &prior_mask_, nullptr);
    if (result == 0) {
      mask_owned_ = false;
      if (test_control != nullptr) {
        test_control->checkpoint(RuntimeManagedSignalTestPoint::kRestoreMask);
      }
      return inject_failure(test_control, RuntimeManagedSignalTestPoint::kRestoreMask);
    }
    return result;
  }

  // pthread_join can complete before procfs removes the task entry used by the next admission
  // census. The broker-opened directory pins its identity across numeric TID reuse; bounded sleeps
  // avoid the retired implementation's busy-yield loop.
  [[nodiscard]] int wait_for_broker_task_retirement(
      const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control) noexcept {
    if (!persistent_) {
      return 0;
    }
    if (broker_task_fd_ < 0) {
      return EPROTO;
    }

    const int injected_error =
        inject_failure(test_control, RuntimeManagedSignalTestPoint::kWaitBrokerRetirement);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    auto retry_delay = std::chrono::microseconds{50};
    for (;;) {
      const BrokerTaskProbe probe = probe_broker_task_retirement(broker_task_fd_);
      if (probe.retired || probe.error != 0) {
        const int task_close_error = close_fd(broker_task_fd_);
        if (probe.error != 0) {
          return probe.error;
        }
        return task_close_error != 0 ? task_close_error : injected_error;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        static_cast<void>(close_fd(broker_task_fd_));
        return ETIMEDOUT;
      }
      std::this_thread::sleep_for(retry_delay);
      if (retry_delay < std::chrono::milliseconds{1}) {
        retry_delay *= 2;
      }
    }
  }

  [[nodiscard]] int validate_prior_actions() const noexcept {
    for (const int signal : std::array<int, 2>{SIGINT, SIGTERM}) {
      struct sigaction current{};
      if (sigaction(signal, nullptr, &current) != 0) {
        return errno;
      }
      if (!signal_actions_equal(current, prior_actions_[signal_index(signal)])) {
        return EPROTO;
      }
    }
    return 0;
  }

  void refresh_prior_actions_after_forwarding() noexcept {
    for (const int signal : std::array<int, 2>{SIGINT, SIGTERM}) {
      struct sigaction current{};
      if (sigaction(signal, nullptr, &current) != 0) {
        record_broker_failure(errno);
        return;
      }
      prior_actions_[signal_index(signal)] = current;
    }
  }

  [[nodiscard]] int close_signal_fd(
      const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control) noexcept {
    const int close_error = close_fd(signal_fd_);
    const int test_error =
        inject_failure(test_control, RuntimeManagedSignalTestPoint::kCloseSignalFd);
    return close_error != 0 ? close_error : test_error;
  }

  [[nodiscard]] int close_control_fd(
      const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control) noexcept {
    if (test_control != nullptr) {
      test_control->checkpoint(RuntimeManagedSignalTestPoint::kBeforeControlFdClose);
    }
    const std::lock_guard lock(descriptor_mutex_);
    const int close_error = close_fd(control_fd_);
    const int test_error =
        inject_failure(test_control, RuntimeManagedSignalTestPoint::kCloseNotification);
    return close_error != 0 ? close_error : test_error;
  }

  [[nodiscard]] ManagedSignalAdmissionFailure add_cleanup_failure(
      ManagedSignalAdmissionFailure failure, const RuntimeManagedSignalPhase phase,
      const int cleanup_error) const noexcept {
    if (failure.cleanup_present) {
      return failure;
    }
    const auto cleanup = admission_failure("RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
                                           phase,
                                           ErrorCode::kFailure,
                                           cleanup_error,
                                           RuntimeManagedSignalCensusReason::kNone,
                                           0U,
                                           generation_,
                                           0U,
                                           true);
    failure.cleanup_token = cleanup.token;
    failure.cleanup_detail = cleanup.detail;
    failure.cleanup_error = cleanup.error;
    failure.cleanup_present = true;
    failure.poison_process = true;
    return failure;
  }

  [[nodiscard]] ManagedSignalAdmissionFailure rollback_setup(
      ManagedSignalAdmissionFailure failure) noexcept {
    const int signal_close_error = close_signal_fd(setup_test_control_);
    const int control_close_error = close_control_fd(setup_test_control_);
    const int mask_error = restore_creator_mask(setup_test_control_);
    const int cleanup_error =
        mask_error != 0 ? mask_error
                        : (signal_close_error != 0 ? signal_close_error : control_close_error);
    if (cleanup_error == 0) {
      return failure;
    }
    return add_cleanup_failure(std::move(failure),
                               mask_error != 0 ? RuntimeManagedSignalPhase::kRestoreMask
                                               : RuntimeManagedSignalPhase::kBindingCleanup,
                               cleanup_error);
  }

  [[nodiscard]] int drain_control(
      const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control) noexcept {
    if (const int test_error =
            inject_failure(test_control, RuntimeManagedSignalTestPoint::kDrainNotification);
        test_error != 0) {
      return test_error;
    }
    if (control_fd_ < 0) {
      return EBADF;
    }
    std::uint64_t value{};
    for (;;) {
      const ssize_t read_result = read(control_fd_, &value, sizeof(value));
      if (read_result == sizeof(value)) {
        continue;
      }
      if (read_result < 0 && errno == EINTR) {
        continue;
      }
      if (read_result < 0 && errno == EAGAIN) {
        return 0;
      }
      return read_result < 0 ? errno : EIO;
    }
  }

  [[nodiscard]] RuntimeManagedSignalDiagnostic diagnostic_for(
      const std::shared_ptr<ManagedSignalBinding>& binding, const int signal,
      const RuntimeManagedSignalOccurrence occurrence,
      const RuntimeManagedSignalPhase phase) const noexcept {
    const std::size_t index = signal_index(signal);
    return RuntimeManagedSignalDiagnostic{
        .present = true,
        .phase = phase,
        .signal = to_runtime_signal(signal),
        .occurrence = occurrence,
        .prior_disposition = classify_disposition(prior_actions_[index]),
        .originally_blocked = sigismember(&prior_mask_, signal) == 1,
        .domain_generation = generation_,
        .binding_generation = binding != nullptr ? binding->generation() : 0U,
    };
  }

  void append_binding_diagnostic(const std::shared_ptr<ManagedSignalBinding>& binding,
                                 const std::string_view token,
                                 const RuntimeManagedSignalDiagnostic& detail,
                                 const bool event) noexcept {
    const auto state = binding->snapshot();
    if (state.attached && state.control.append_managed_diagnostic != nullptr) {
      state.control.append_managed_diagnostic(state.control.state, token, detail, event);
    }
  }

  void request_binding_stop(const std::shared_ptr<ManagedSignalBinding>& binding) noexcept {
    const auto state = binding->snapshot();
    if (state.attached && state.control.request_stop != nullptr) {
      state.control.request_stop(state.control.state);
    }
  }

  void record_wait_failure(const std::shared_ptr<ManagedSignalBinding>& binding,
                           const RuntimeManagedSignalPhase phase, const int os_error) noexcept {
    poison();
    if (!binding->mark_controller_wait_failed()) {
      return;
    }
    const RuntimeManagedSignalDiagnostic detail{
        .present = true,
        .phase = phase,
        .domain_generation = generation_,
        .binding_generation = binding->generation(),
        .os_error = os_error,
    };
    append_binding_diagnostic(binding, "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED", detail, false);
    request_binding_stop(binding);
  }

  // Keep the two-occurrence consumption loop together so its ordering remains explicit.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void drain_signals(const std::shared_ptr<ManagedSignalBinding>& binding,
                     const RuntimeManagedSignalPhase phase) noexcept {
    if (signal_fd_ < 0) {
      return;
    }
    if (binding != nullptr && binding->snapshot().second_signal != 0) {
      return;
    }
    const RuntimeManagedSignalTestPoint test_point =
        phase == RuntimeManagedSignalPhase::kFinalDrain
            ? RuntimeManagedSignalTestPoint::kFinalDrain
            : RuntimeManagedSignalTestPoint::kReadSignal;
    const auto& test_control = binding != nullptr ? binding->test_control() : setup_test_control_;
    if (const int test_error = inject_failure(test_control, test_point); test_error != 0) {
      if (binding != nullptr) {
        record_wait_failure(binding, phase, test_error);
      } else {
        record_broker_failure(test_error);
      }
      return;
    }
    for (;;) {
      signalfd_siginfo info{};
      const ssize_t count = read(signal_fd_, &info, sizeof(info));
      if (count == sizeof(info)) {
        const int signal = static_cast<int>(info.ssi_signo);
        if (signal != SIGINT && signal != SIGTERM) {
          continue;
        }
        if (binding != nullptr) {
          if (binding->accept_signal(signal) == RuntimeManagedSignalOccurrence::kSecond) {
            // Disarm before consuming any later occurrence. There is intentionally no HSDK third
            // escalation state; later pending signals belong to the unchanged prior policy.
            return;
          }
        } else {
          forward_unbound_signal(signal);
        }
        continue;
      }
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count < 0 && errno == EAGAIN) {
        return;
      }
      if (binding != nullptr) {
        record_wait_failure(binding, phase, count < 0 ? errno : EIO);
      } else {
        record_broker_failure(count < 0 ? errno : EIO);
      }
      return;
    }
  }

  void dispatch_binding(const std::shared_ptr<ManagedSignalBinding>& binding) noexcept {
    auto state = binding->snapshot();
    if (state.attached && (state.first_signal != 0 || state.candidate_ready) &&
        !state.disposition_failure_dispatched) {
      if (const int disposition_error = validate_prior_actions(); disposition_error != 0) {
        binding->mark_disposition_failure_dispatched();
        record_restore_failure(
            binding, RuntimeManagedSignalPhase::kQueryDisposition, disposition_error);
        state = binding->snapshot();
      }
    }
    if (state.attached && state.notification_error != 0 && !state.notification_failure_dispatched) {
      binding->mark_notification_failure_dispatched();
      record_wait_failure(
          binding, RuntimeManagedSignalPhase::kNotifyController, state.notification_error);
      state = binding->snapshot();
    }
    if (state.attached && state.first_signal != 0 && !state.first_dispatched) {
      const auto detail = diagnostic_for(binding,
                                         state.first_signal,
                                         RuntimeManagedSignalOccurrence::kFirst,
                                         RuntimeManagedSignalPhase::kAcceptSignal);
      append_binding_diagnostic(binding, "RUNTIME_MANAGED_SIGNAL_RECEIVED", detail, true);
      request_binding_stop(binding);
      binding->mark_first_dispatched();
      state = binding->snapshot();
    }
    if (state.attached && state.second_signal != 0 && !state.second_dispatched) {
      const auto detail = diagnostic_for(binding,
                                         state.second_signal,
                                         RuntimeManagedSignalOccurrence::kSecond,
                                         RuntimeManagedSignalPhase::kDelegateSignal);
      delegate_second_signal(binding, state.second_signal, detail);
    }
  }

  void record_restore_failure(const std::shared_ptr<ManagedSignalBinding>& binding,
                              const RuntimeManagedSignalPhase phase, const int os_error) noexcept {
    poison();
    binding->fail();
    const RuntimeManagedSignalDiagnostic detail{
        .present = true,
        .phase = phase,
        .domain_generation = generation_,
        .binding_generation = binding->generation(),
        .os_error = os_error,
    };
    append_binding_diagnostic(binding, "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED", detail, false);
    request_binding_stop(binding);
  }

  void delegate_second_signal(const std::shared_ptr<ManagedSignalBinding>& binding,
                              const int signal,
                              const RuntimeManagedSignalDiagnostic& detail) noexcept {
    forwarding_ = true;
    if (!persistent_) {
      const int close_error = close_signal_fd(binding->test_control());
      if (close_error != 0) {
        record_restore_failure(binding, RuntimeManagedSignalPhase::kBindingCleanup, close_error);
        return;
      }
      const int restore_error = restore_creator_mask(binding->test_control());
      if (restore_error != 0) {
        binding->mark_second_dispatched();
        record_restore_failure(binding, RuntimeManagedSignalPhase::kRestoreMask, restore_error);
        return;
      }
      if (!binding->snapshot().second_escalation_published) {
        append_binding_diagnostic(binding, "RUNTIME_MANAGED_SIGNAL_ESCALATED", detail, true);
        binding->mark_second_escalation_published();
      }
      const int injected_delegate =
          inject_failure(binding->test_control(), RuntimeManagedSignalTestPoint::kDelegateSignal);
      if (injected_delegate != 0 || raise(signal) != 0) {
        const int delegate_error = injected_delegate != 0 ? injected_delegate : errno;
        record_restore_failure(binding, RuntimeManagedSignalPhase::kDelegateSignal, delegate_error);
        return;
      }
      binding->mark_second_dispatched();
      return;
    }
    auto& registry = managed_signal_registry();
    {
      const std::lock_guard registry_lock(registry.mutex);
      if (registry.domain.get() == this) {
        registry.phase = ManagedSignalRegistryPhase::kForwarding;
      }
    }
    const bool was_blocked = sigismember(&prior_mask_, signal) == 1;
    int mask_result = pthread_sigmask(SIG_SETMASK, &prior_mask_, nullptr);
    if (mask_result == 0) {
      if (binding->test_control() != nullptr) {
        binding->test_control()->checkpoint(RuntimeManagedSignalTestPoint::kRestoreMask);
      }
      mask_result =
          inject_failure(binding->test_control(), RuntimeManagedSignalTestPoint::kRestoreMask);
    }
    if (mask_result != 0) {
      binding->mark_second_dispatched();
      record_restore_failure(binding, RuntimeManagedSignalPhase::kRestoreMask, mask_result);
      return;
    }
    if (!binding->snapshot().second_escalation_published) {
      append_binding_diagnostic(binding, "RUNTIME_MANAGED_SIGNAL_ESCALATED", detail, true);
      binding->mark_second_escalation_published();
    }
    int delegate_error =
        inject_failure(binding->test_control(), RuntimeManagedSignalTestPoint::kDelegateSignal);
    if (delegate_error == 0 && was_blocked) {
      delegate_error = pthread_kill(creator_thread_, signal);
    } else if (delegate_error == 0) {
      delegate_error = raise(signal) == 0 ? 0 : errno;
    }
    if (delegate_error != 0) {
      record_restore_failure(binding, RuntimeManagedSignalPhase::kDelegateSignal, delegate_error);
      return;
    }
    binding->mark_second_dispatched();
  }

  void forward_unbound_signal(const int signal) noexcept {
    const bool was_blocked = sigismember(&prior_mask_, signal) == 1;
    if (was_blocked) {
      if (const int delegate_error = pthread_kill(creator_thread_, signal); delegate_error != 0) {
        record_broker_failure(delegate_error);
      }
      return;
    }
    if (const int restore_error = pthread_sigmask(SIG_SETMASK, &prior_mask_, nullptr);
        restore_error != 0) {
      record_broker_failure(restore_error);
      return;
    }
    const int delegate_error = raise(signal) == 0 ? 0 : errno;
    const int reblock_error = pthread_sigmask(SIG_BLOCK, &managed_mask_, nullptr);
    if (delegate_error != 0) {
      record_broker_failure(delegate_error);
    }
    if (reblock_error != 0) {
      record_broker_failure(reblock_error);
    } else {
      refresh_prior_actions_after_forwarding();
    }
  }

  [[nodiscard]] int wait_for_descriptor_activity(
      const std::shared_ptr<RuntimeManagedSignalTestControl>& test_control,
      const RuntimeManagedSignalTestPoint wait_point) noexcept {
    std::array<pollfd, 2> descriptors{{
        {.fd = forwarding_ ? -1 : signal_fd_, .events = POLLIN, .revents = 0},
        {.fd = control_fd_, .events = POLLIN, .revents = 0},
    }};
    if (test_control != nullptr) {
      test_control->checkpoint(wait_point);
    }
    if (const int wait_error = inject_failure(test_control, wait_point); wait_error != 0) {
      return wait_error;
    }
    const int injected_poll = inject_failure(test_control, RuntimeManagedSignalTestPoint::kPoll);
    if (injected_poll != 0) {
      return injected_poll;
    }
    int result{};
    do {
      result = poll(descriptors.data(), descriptors.size(), -1);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      return errno;
    }
    if (result > 0 &&
        ((descriptors[0].revents | descriptors[1].revents) & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return EIO;
    }
    return 0;
  }

  [[nodiscard]] bool wait_for_activity(
      const std::shared_ptr<ManagedSignalBinding>& binding) noexcept {
    if (const int error = wait_for_descriptor_activity(
            binding->test_control(), RuntimeManagedSignalTestPoint::kBoundWaitForActivity);
        error != 0) {
      record_wait_failure(binding, RuntimeManagedSignalPhase::kPoll, error);
      return false;
    }
    return true;
  }

  // Terminal publication and registry retirement are one ordered handshake.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void complete_binding(const std::shared_ptr<ManagedSignalBinding>& binding) noexcept {
    const auto before = binding->snapshot();
    if (!persistent_) {
      const int signal_close_error = close_signal_fd(binding->test_control());
      const int control_close_error = close_control_fd(binding->test_control());
      const int restore_error = restore_creator_mask(binding->test_control());
      if (signal_close_error != 0 || control_close_error != 0) {
        record_restore_failure(binding,
                               RuntimeManagedSignalPhase::kBindingCleanup,
                               signal_close_error != 0 ? signal_close_error : control_close_error);
      }
      if (restore_error != 0) {
        record_restore_failure(binding, RuntimeManagedSignalPhase::kRestoreMask, restore_error);
      }
    }

    const auto after_cleanup = binding->snapshot();
    if (!after_cleanup.aborted && after_cleanup.second_signal != 0 &&
        !after_cleanup.second_dispatched) {
      const auto detail = diagnostic_for(binding,
                                         after_cleanup.second_signal,
                                         RuntimeManagedSignalOccurrence::kSecond,
                                         RuntimeManagedSignalPhase::kDelegateSignal);
      delegate_second_signal(binding, after_cleanup.second_signal, detail);
    }

    if (const int release_error = inject_failure(binding->test_control(),
                                                 RuntimeManagedSignalTestPoint::kTerminalRelease);
        release_error != 0) {
      record_restore_failure(binding, RuntimeManagedSignalPhase::kBindingCleanup, release_error);
    }
    binding->release();
    if (before.attached) {
      if (!before.worker_waiting) {
        binding->finish_start_failure();
      }
      binding->wait_until_terminal_frozen();
    }

    {
      const std::lock_guard lock(mutex_);
      binding_.reset();
    }

    {
      auto& registry = managed_signal_registry();
      const std::lock_guard registry_lock(registry.mutex);
      if (registry.domain.get() == this) {
        if (poisoned_.load(std::memory_order_acquire)) {
          if (!persistent_) {
            registry.domain.reset();
            registry.owner_generation = 0U;
          }
          registry.binding_generation = 0U;
          registry.phase = ManagedSignalRegistryPhase::kPoisoned;
        } else if (persistent_) {
          registry.binding_generation = 0U;
          registry.phase = forwarding_ ? ManagedSignalRegistryPhase::kForwarding
                                       : ManagedSignalRegistryPhase::kSpent;
        } else {
          registry.domain.reset();
          registry.owner_generation = 0U;
          registry.binding_generation = 0U;
          registry.phase = ManagedSignalRegistryPhase::kUnowned;
        }
      }
    }
    binding->mark_retired();
  }

  [[nodiscard]] bool initialize_broker_task() noexcept {
    const int injected_error =
        inject_failure(setup_test_control_, RuntimeManagedSignalTestPoint::kOpenBrokerTask);
    int broker_task_fd{-1};
    int broker_task_error{injected_error};
    if (broker_task_error == 0) {
      do {
        broker_task_fd = open("/proc/thread-self", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
      } while (broker_task_fd < 0 && errno == EINTR);
      broker_task_error = broker_task_fd < 0 ? errno : 0;
    }

    std::unique_lock lock(mutex_);
    broker_task_fd_ = broker_task_fd;
    broker_task_error_ = broker_task_error;
    broker_ready_ = true;
    domain_cv_.notify_all();
    if (broker_task_error_ != 0) {
      return false;
    }
    domain_cv_.wait(lock, [this] { return broker_enabled_ || close_requested_; });
    return !close_requested_;
  }

  void broker_main() noexcept {
    if (!initialize_broker_task()) {
      return;
    }
    while (true) {
      std::shared_ptr<ManagedSignalBinding> binding;
      std::shared_ptr<RuntimeManagedSignalTestControl> wait_test_control;
      bool close_requested{};
      bool drain_unbound{};
      {
        const std::lock_guard lock(mutex_);
        binding = binding_;
        wait_test_control = close_test_control_;
        close_requested = close_requested_;
        if (binding == nullptr && !close_requested) {
          unbound_draining_ = true;
          drain_unbound = true;
        }
      }
      if (binding != nullptr) {
        drive_binding(binding);
        continue;
      }
      if (const int control_error = drain_control(nullptr); control_error != 0) {
        record_broker_failure(control_error);
        {
          const std::lock_guard lock(mutex_);
          unbound_draining_ = false;
        }
        domain_cv_.notify_all();
        return;
      }
      // close_public() does not wait for the unbound forwarding boundary. If it publishes
      // close_requested_ after the snapshot above but before drain_control(), that drain can
      // consume the corresponding eventfd wake. Recheck the durable state before entering an
      // indefinite descriptor wait so the close transition cannot depend on a polling timeout.
      {
        const std::lock_guard lock(mutex_);
        close_requested = close_requested_;
      }
      if (close_requested) {
        {
          const std::lock_guard lock(mutex_);
          unbound_draining_ = false;
        }
        domain_cv_.notify_all();
        return;
      }
      if (drain_unbound && !forwarding_) {
        drain_signals(nullptr, RuntimeManagedSignalPhase::kRead);
      }
      if (poisoned_.load(std::memory_order_acquire)) {
        {
          const std::lock_guard lock(mutex_);
          unbound_draining_ = false;
        }
        domain_cv_.notify_all();
        return;
      }
      {
        const std::lock_guard lock(mutex_);
        unbound_draining_ = false;
      }
      domain_cv_.notify_all();
      if (const int error = wait_for_descriptor_activity(
              wait_test_control, RuntimeManagedSignalTestPoint::kUnboundWaitForActivity);
          error != 0) {
        record_broker_failure(error);
        return;
      }
    }
  }

  bool persistent_{};
  std::uint64_t generation_{};
  pthread_t creator_thread_{};
  std::thread::id creator_cpp_thread_;
  pid_t creator_pid_{};
  sigset_t prior_mask_{};
  sigset_t managed_mask_{};
  std::array<struct sigaction, 2> prior_actions_{};
  int signal_fd_{-1};
  int control_fd_{-1};
  int broker_task_fd_{-1};
  std::thread broker_;
  mutable std::mutex mutex_;
  mutable std::mutex descriptor_mutex_;
  std::condition_variable domain_cv_;
  std::shared_ptr<ManagedSignalBinding> binding_;
  bool binding_used_{};
  bool unbound_draining_{};
  bool close_requested_{};
  bool forwarding_{};
  bool mask_owned_{};
  bool closed_{};
  bool broker_ready_{};
  bool broker_enabled_{};
  int broker_task_error_{};
  std::atomic_bool poisoned_{};
  int broker_failure_error_{};
  const std::shared_ptr<RuntimeManagedSignalTestControl> setup_test_control_;
  std::shared_ptr<RuntimeManagedSignalTestControl> close_test_control_;
};

void ManagedSignalBinding::wake_controller() noexcept {
  const std::lock_guard lock(mutex_);
  wake_controller_locked();
}

void ManagedSignalBinding::wake_controller_locked() noexcept {
  if (const int test_error =
          inject_failure(test_control_, RuntimeManagedSignalTestPoint::kNotifyController);
      test_error != 0) {
    force_failed_ = true;
    if (notification_error_ == 0) {
      notification_error_ = test_error;
    }
  }
  if (const int wake_error = domain_->wake(); wake_error != 0) {
    force_failed_ = true;
    if (notification_error_ == 0) {
      notification_error_ = wake_error;
    }
  }
}

void ManagedSignalBinding::drive() noexcept {
  if (caller_drives_) {
    domain_->drive_binding(shared_from_this());
  }
}

[[nodiscard]] expected<std::shared_ptr<ManagedSignalDomainState>, ManagedSignalAdmissionFailure>
claim_and_establish_domain(const bool persistent,
                           std::shared_ptr<RuntimeManagedSignalTestControl> test_control = {}) {
  if (inherited_managed_signal_registry()) {
    return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                             RuntimeManagedSignalPhase::kAcquireDomain,
                                             ErrorCode::kNotSupported,
                                             0,
                                             RuntimeManagedSignalCensusReason::kForkedProcess));
  }
  auto& registry = managed_signal_registry();
  std::uint64_t generation{};
  {
    const std::lock_guard lock(registry.mutex);
    if (registry.domain != nullptr && registry.domain->is_forked_process()) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kAcquireDomain,
                                               ErrorCode::kNotSupported,
                                               0,
                                               RuntimeManagedSignalCensusReason::kForkedProcess,
                                               0U,
                                               registry.domain->generation()));
    }
    if (registry.phase != ManagedSignalRegistryPhase::kUnowned) {
      const std::uint64_t active_generation = registry.owner_generation;
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_OWNER_BUSY",
                                               RuntimeManagedSignalPhase::kAcquireDomain,
                                               ErrorCode::kResourceExhausted,
                                               0,
                                               RuntimeManagedSignalCensusReason::kNone,
                                               0U,
                                               active_generation));
    }
    if (const int test_error =
            inject_failure(test_control, RuntimeManagedSignalTestPoint::kReserveDomainGeneration);
        test_error != 0) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kAcquireDomain,
                                               ErrorCode::kResourceExhausted,
                                               test_error));
    }
    if (test_control != nullptr && test_control->next_generation_seed() != 0U) {
      registry.next_generation = test_control->next_generation_seed();
    }
    auto reserved = reserve_generation_locked(registry);
    if (!reserved) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kAcquireDomain,
                                               reserved.error().code));
    }
    generation = *reserved;
    registry.owner_generation = generation;
    registry.binding_generation = 0U;
    registry.phase = ManagedSignalRegistryPhase::kEstablishing;
  }

  auto established =
      ManagedSignalDomainState::establish(persistent, generation, std::move(test_control));
  if (!established) {
    auto failure = std::move(established).error();
    const std::lock_guard lock(registry.mutex);
    registry.owner_generation = 0U;
    registry.binding_generation = 0U;
    registry.domain.reset();
    registry.phase = failure.poison_process ? ManagedSignalRegistryPhase::kPoisoned
                                            : ManagedSignalRegistryPhase::kUnowned;
    return make_unexpected(std::move(failure));
  }
  {
    const std::lock_guard lock(registry.mutex);
    registry.domain = *established;
    registry.owner_generation = generation;
    registry.phase = persistent ? ManagedSignalRegistryPhase::kAvailable
                                : ManagedSignalRegistryPhase::kEstablishing;
  }
  (*established)->enable_broker();
  return established;
}

}  // namespace

[[nodiscard]] expected<void, Error> close_public_managed_signal_domain(
    const std::shared_ptr<void>& erased) noexcept {
  if (inherited_managed_signal_registry()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return std::static_pointer_cast<ManagedSignalDomainState>(erased)->close_public();
}

void best_effort_destroy_managed_signal_domain(const std::shared_ptr<void>& erased) noexcept {
  if (inherited_managed_signal_registry()) {
    return;
  }
  std::static_pointer_cast<ManagedSignalDomainState>(erased)->best_effort_destroy();
}

#endif  // defined(__linux__)

}  // namespace holoscan::detail

namespace holoscan {

class ManagedShutdownSignals::Impl {
 public:
#if defined(__linux__)
  explicit Impl(std::shared_ptr<void> domain) noexcept : domain_(std::move(domain)) {}

  [[nodiscard]] std::shared_ptr<void> domain() const noexcept { return domain_; }

  [[nodiscard]] expected<void, Error> close() noexcept {
    if (domain_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto result = detail::close_public_managed_signal_domain(domain_);
    if (result) {
      domain_.reset();
    }
    return result;
  }

  void best_effort_destroy() noexcept {
    if (domain_ != nullptr) {
      detail::best_effort_destroy_managed_signal_domain(domain_);
      domain_.reset();
    }
  }

 private:
  std::shared_ptr<void> domain_;
#else
  [[nodiscard]] expected<void, Error> close() noexcept {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  void best_effort_destroy() noexcept {}
#endif
};

}  // namespace holoscan

namespace holoscan::detail {

ManagedShutdownSignals::Impl* ManagedShutdownSignalsAccess::get(
    ManagedShutdownSignals& signals) noexcept {
  return signals.impl_.get();
}

#if defined(__linux__)

expected<ManagedShutdownSignals, ManagedSignalAdmissionFailure>
ManagedShutdownSignalsAccess::create_for_test(
    std::shared_ptr<RuntimeManagedSignalTestControl> test_control) {
  auto domain = claim_and_establish_domain(true, std::move(test_control));
  if (!domain) {
    return make_unexpected(std::move(domain).error());
  }
  return ManagedShutdownSignals(
      std::make_unique<ManagedShutdownSignals::Impl>(std::move(domain).value()));
}

expected<PreparedManagedSignalBinding, ManagedSignalAdmissionFailure>
prepare_default_managed_signal_binding(
    std::shared_ptr<RuntimeManagedSignalTestControl> test_control) {
  if (inherited_managed_signal_registry()) {
    return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                             RuntimeManagedSignalPhase::kAcquireDomain,
                                             ErrorCode::kNotSupported,
                                             0,
                                             RuntimeManagedSignalCensusReason::kForkedProcess));
  }
  std::shared_ptr<ManagedSignalDomainState> domain;
  auto& registry = managed_signal_registry();
  {
    const std::lock_guard lock(registry.mutex);
    if (registry.domain != nullptr && registry.domain->is_forked_process()) {
      return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                               RuntimeManagedSignalPhase::kAcquireDomain,
                                               ErrorCode::kNotSupported,
                                               0,
                                               RuntimeManagedSignalCensusReason::kForkedProcess,
                                               0U,
                                               registry.domain->generation()));
    }
    if (registry.phase == ManagedSignalRegistryPhase::kAvailable && registry.domain != nullptr &&
        registry.domain->persistent() && registry.domain->is_creator_thread()) {
      domain = registry.domain;
    } else if (registry.phase != ManagedSignalRegistryPhase::kUnowned) {
      return make_unexpected(
          admission_failure("RUNTIME_MANAGED_SIGNAL_OWNER_BUSY",
                            RuntimeManagedSignalPhase::kAcquireDomain,
                            ErrorCode::kResourceExhausted,
                            0,
                            RuntimeManagedSignalCensusReason::kIncompatibleDomain,
                            0U,
                            registry.owner_generation));
    }
  }
  bool caller_drives{};
  if (domain == nullptr) {
    auto established = claim_and_establish_domain(false, test_control);
    if (!established) {
      return make_unexpected(std::move(established).error());
    }
    domain = std::move(established).value();
    caller_drives = true;
  }
  auto binding = domain->reserve_binding(caller_drives, std::move(test_control));
  if (!binding) {
    auto failure = std::move(binding).error();
    if (caller_drives) {
      failure = domain->abandon_unbound_transient(std::move(failure));
    }
    return make_unexpected(std::move(failure));
  }
  return PreparedManagedSignalBinding{.controller = std::move(binding).value()};
}

expected<PreparedManagedSignalBinding, ManagedSignalAdmissionFailure>
prepare_explicit_managed_signal_binding(
    ManagedShutdownSignals& signals,
    std::shared_ptr<RuntimeManagedSignalTestControl> test_control) {
  if (inherited_managed_signal_registry()) {
    return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                             RuntimeManagedSignalPhase::kAcquireDomain,
                                             ErrorCode::kNotSupported,
                                             0,
                                             RuntimeManagedSignalCensusReason::kForkedProcess));
  }
  auto* const impl = ManagedShutdownSignalsAccess::get(signals);
  if (impl == nullptr) {
    return make_unexpected(admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                                             RuntimeManagedSignalPhase::kAcquireDomain,
                                             ErrorCode::kClosed));
  }
  auto domain = std::static_pointer_cast<ManagedSignalDomainState>(impl->domain());
  if (domain == nullptr || !domain->is_creator_thread()) {
    const auto census_reason = domain != nullptr && domain->is_forked_process()
                                   ? RuntimeManagedSignalCensusReason::kForkedProcess
                                   : RuntimeManagedSignalCensusReason::kWrongThread;
    return make_unexpected(
        admission_failure("RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
                          RuntimeManagedSignalPhase::kAcquireDomain,
                          census_reason == RuntimeManagedSignalCensusReason::kForkedProcess
                              ? ErrorCode::kNotSupported
                              : ErrorCode::kInvalidArgument,
                          0,
                          census_reason,
                          0U,
                          domain != nullptr ? domain->generation() : 0U));
  }
  auto binding = domain->reserve_binding(false, std::move(test_control));
  if (!binding) {
    return make_unexpected(std::move(binding).error());
  }
  return PreparedManagedSignalBinding{.controller = std::move(binding).value()};
}

#else

expected<ManagedShutdownSignals, ManagedSignalAdmissionFailure>
ManagedShutdownSignalsAccess::create_for_test(std::shared_ptr<RuntimeManagedSignalTestControl>) {
  return make_unexpected(ManagedSignalAdmissionFailure{
      .token = "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
      .detail =
          RuntimeManagedSignalDiagnostic{
              .present = true,
              .phase = RuntimeManagedSignalPhase::kAcquireDomain,
          },
      .error = Error{ErrorCode::kNotSupported},
  });
}

expected<PreparedManagedSignalBinding, ManagedSignalAdmissionFailure>
prepare_default_managed_signal_binding(std::shared_ptr<RuntimeManagedSignalTestControl>) {
  return make_unexpected(ManagedSignalAdmissionFailure{
      .token = "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
      .detail =
          RuntimeManagedSignalDiagnostic{
              .present = true,
              .phase = RuntimeManagedSignalPhase::kAcquireDomain,
          },
      .error = Error{ErrorCode::kNotSupported},
  });
}

expected<PreparedManagedSignalBinding, ManagedSignalAdmissionFailure>
prepare_explicit_managed_signal_binding(
    ManagedShutdownSignals&, std::shared_ptr<RuntimeManagedSignalTestControl> test_control) {
  return prepare_default_managed_signal_binding(std::move(test_control));
}

#endif

}  // namespace holoscan::detail

namespace holoscan {

ManagedShutdownSignals::ManagedShutdownSignals(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ManagedShutdownSignals::ManagedShutdownSignals(ManagedShutdownSignals&&) noexcept = default;

ManagedShutdownSignals::~ManagedShutdownSignals() noexcept {
  if (impl_ != nullptr) {
    impl_->best_effort_destroy();
  }
}

expected<ManagedShutdownSignals, Error> ManagedShutdownSignals::create() {
#if defined(__linux__)
  auto domain = detail::claim_and_establish_domain(true);
  if (!domain) {
    return make_unexpected(std::move(domain).error().error);
  }
  auto retained = std::move(domain).value();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return ManagedShutdownSignals{std::make_unique<Impl>(retained)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    retained->best_effort_destroy();
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
#else
  return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
}

expected<void, Error> ManagedShutdownSignals::close() noexcept {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto result = impl_->close();
  if (result) {
    impl_.reset();
  }
  return result;
}

}  // namespace holoscan
