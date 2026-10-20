/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "physical_readiness_binding.hpp"

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace holoscan::detail {
namespace {

enum class PhysicalReadinessTerminal : std::uint8_t {
  kActive = 0,
  kClosed = 1,
  kStale = 2,
  kFailed = 3,
  kRestarting = 4,
};

enum class PhysicalReadinessCellPhase : std::uint8_t {
  kFree = 0,
  kClaimed = 1,
  kReady = 2,
  kDrained = 3,
  kReleasing = 4,
  kCanceled = 5,
};

// Publisher admission and terminal transition form two-atomic handshakes.
// One sequentially consistent order prevents either side from missing the
// other's active-publisher or credit-owning cell-phase publication.
constexpr std::memory_order kAdmissionMemoryOrder = std::memory_order_seq_cst;

[[nodiscard]] bool is_physical_kind(RuntimeEventKind kind) noexcept {
  return kind == RuntimeEventKind::kReadable || kind == RuntimeEventKind::kNotification ||
         kind == RuntimeEventKind::kDoorbell || kind == RuntimeEventKind::kSpin;
}

[[nodiscard]] bool health_disposition_valid(EndpointClockHealthDisposition disposition) noexcept {
  return disposition == EndpointClockHealthDisposition::kSkip ||
         disposition == EndpointClockHealthDisposition::kDegraded ||
         disposition == EndpointClockHealthDisposition::kFailRun;
}

[[nodiscard]] Error terminal_error(PhysicalReadinessTerminal terminal, ErrorCode failure) noexcept {
  switch (terminal) {
    case PhysicalReadinessTerminal::kActive:
      return Error{ErrorCode::kSuccess};
    case PhysicalReadinessTerminal::kClosed:
      return Error{ErrorCode::kClosed};
    case PhysicalReadinessTerminal::kStale:
      return Error{ErrorCode::kStaleEndpoint};
    case PhysicalReadinessTerminal::kFailed:
      return Error{failure == ErrorCode::kSuccess ? ErrorCode::kFailure : failure};
    case PhysicalReadinessTerminal::kRestarting:
      return Error{ErrorCode::kNotReady};
  }
  return Error{ErrorCode::kFailure};
}

[[nodiscard]] bool request_equal(const RuntimePortBindingRequest& lhs,
                                 const RuntimePortBindingRequest& rhs) noexcept {
  return lhs.plan_generation == rhs.plan_generation && lhs.run_generation == rhs.run_generation &&
         lhs.kind == rhs.kind && lhs.source == rhs.source && lhs.port == rhs.port &&
         lhs.source_clock == rhs.source_clock && lhs.reference_clock == rhs.reference_clock &&
         lhs.source_to_reference_conversion_id == rhs.source_to_reference_conversion_id &&
         lhs.execution_lane == rhs.execution_lane && lhs.capacity == rhs.capacity &&
         lhs.source_generation == rhs.source_generation &&
         lhs.initial_source_sequence == rhs.initial_source_sequence;
}

[[nodiscard]] bool descriptor_valid(const PhysicalReadinessBindingDescriptor& descriptor) noexcept {
  const auto& request = descriptor.request;
  if (request.plan_generation == 0U || request.run_generation == 0U ||
      !is_physical_kind(request.kind) || !request.source || request.port ||
      !request.source_clock.valid() || !request.reference_clock.valid() ||
      request.source_clock != request.reference_clock ||
      request.source_to_reference_conversion_id != 0U || request.capacity == 0U ||
      request.source_generation == 0U ||
      request.initial_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.execution_lane != request.execution_lane ||
      descriptor.execution_lane == std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  if ((request.kind == RuntimeEventKind::kReadable || request.kind == RuntimeEventKind::kSpin) &&
      request.capacity != 1U) {
    return false;
  }
  const bool busy_poll_required =
      request.kind == RuntimeEventKind::kSpin ||
      descriptor.wake_requirement == ReadinessWakeRequirement::kBusyPollRequired;
  if (busy_poll_required && descriptor.wake_mode != WakeMode::kBusyPoll) {
    return false;
  }
  if (request.kind == RuntimeEventKind::kSpin &&
      (descriptor.poll_state == nullptr || descriptor.poll_asserted == nullptr)) {
    return false;
  }
  if (descriptor.poll_asserted != nullptr && descriptor.poll_state == nullptr) {
    return false;
  }
  if (descriptor.wake != nullptr && descriptor.wake_state == nullptr) {
    return false;
  }
  if (descriptor.test_publish_checkpoint != nullptr && descriptor.test_publish_state == nullptr) {
    return false;
  }
  if (descriptor.monitored_health_required) {
    return descriptor.monitored_health_bound &&
           health_disposition_valid(descriptor.monitored_health_disposition);
  }
  return !descriptor.monitored_health_bound &&
         descriptor.monitored_health_disposition == EndpointClockHealthDisposition::kNotApplicable;
}

}  // namespace

struct PhysicalReadinessState {
  struct Cell {
    std::atomic<std::size_t> ring_sequence{};
    std::atomic<PhysicalReadinessCellPhase> phase{PhysicalReadinessCellPhase::kFree};
    RuntimeEvent event;
  };

  // Cells contain non-movable atomics and require stable, plan-bounded indices.
  using CellStorage = std::unique_ptr<Cell[]>;  // NOLINT(cppcoreguidelines-avoid-c-arrays)

  explicit PhysicalReadinessState(PhysicalReadinessBindingDescriptor source)
      : descriptor(std::move(source)),
        cells(new Cell[descriptor.request.capacity]),
        source_generation(descriptor.request.source_generation) {
    for (std::size_t index = 0U; index < descriptor.request.capacity; ++index) {
      cells[index].ring_sequence.store(index, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] Error current_terminal_error() const noexcept {
    return terminal_error(terminal.load(std::memory_order_acquire),
                          fatal_error.load(std::memory_order_acquire));
  }

  void notify_wake() noexcept {
    std::uint64_t epoch = wake_epoch.load(std::memory_order_relaxed);
    while (epoch != std::numeric_limits<std::uint64_t>::max() &&
           !wake_epoch.compare_exchange_weak(
               epoch, epoch + 1U, std::memory_order_release, std::memory_order_relaxed)) {
    }
    if (descriptor.wake != nullptr) {
      descriptor.wake(descriptor.wake_state);
    }
  }

  void publication_checkpoint(PhysicalReadinessPublicationCheckpoint checkpoint) noexcept {
    if (descriptor.test_publish_checkpoint != nullptr) {
      descriptor.test_publish_checkpoint(descriptor.test_publish_state, checkpoint);
    }
  }

  [[nodiscard]] bool enter_publisher() noexcept {
    std::size_t count = active_publishers.load(std::memory_order_acquire);
    while (count != std::numeric_limits<std::size_t>::max()) {
      if (active_publishers.compare_exchange_weak(
              count, count + 1U, kAdmissionMemoryOrder, std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  void leave_publisher() noexcept { active_publishers.fetch_sub(1U, std::memory_order_release); }

  [[nodiscard]] bool release_outstanding_credit() noexcept {
    std::size_t count = outstanding.load(std::memory_order_acquire);
    while (count != 0U) {
      if (outstanding.compare_exchange_weak(
              count, count - 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool cancel_credit(Cell& cell, bool include_drained) noexcept {
    auto phase = cell.phase.load(kAdmissionMemoryOrder);
    while (phase == PhysicalReadinessCellPhase::kClaimed ||
           phase == PhysicalReadinessCellPhase::kReady ||
           (include_drained && phase == PhysicalReadinessCellPhase::kDrained)) {
      if (cell.phase.compare_exchange_weak(phase,
                                           PhysicalReadinessCellPhase::kCanceled,
                                           kAdmissionMemoryOrder,
                                           kAdmissionMemoryOrder)) {
        if (!release_outstanding_credit()) {
          fail(ErrorCode::kProtocolError);
        }
        return true;
      }
    }
    return false;
  }

  // Keep admission, bounded-ring publication, and rollback adjacent so each claimed credit has
  // one auditable terminal path.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<std::uint64_t, Error> publish(RuntimeEventKind kind,
                                                       TimePoint reference_time,
                                                       std::uint64_t expected_generation) noexcept {
    if (!reference_time.valid() || reference_time.clock_id != descriptor.request.reference_clock ||
        kind != descriptor.request.kind || expected_generation == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!enter_publisher()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    struct PublisherGuard {
      PhysicalReadinessState* state;
      ~PublisherGuard() { state->leave_publisher(); }
    };
    const PublisherGuard publisher_guard{this};
    if (source_generation.load(std::memory_order_acquire) != expected_generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (terminal.load(kAdmissionMemoryOrder) != PhysicalReadinessTerminal::kActive) {
      return make_unexpected(current_terminal_error());
    }
    publication_checkpoint(PhysicalReadinessPublicationCheckpoint::kPublisherEntered);

    std::size_t position = enqueue_position.load(std::memory_order_relaxed);
    Cell* cell = nullptr;
    for (;;) {
      if (position == std::numeric_limits<std::size_t>::max() ||
          position > std::numeric_limits<std::uint64_t>::max() -
                         descriptor.request.initial_source_sequence - 1U) {
        fail(ErrorCode::kResourceExhausted);
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      cell = &cells[position % descriptor.request.capacity];
      const std::size_t sequence = cell->ring_sequence.load(std::memory_order_acquire);
      if (sequence == position) {
        // A one-cell ring publishes the same sequence that the next producer
        // position expects. The phase therefore remains the authoritative
        // accepted-through-retirement credit check for capacity one.
        if (cell->phase.load(std::memory_order_acquire) != PhysicalReadinessCellPhase::kFree) {
          return make_unexpected(Error{ErrorCode::kBackpressured});
        }
        if (enqueue_position.compare_exchange_weak(
                position, position + 1U, std::memory_order_relaxed, std::memory_order_relaxed)) {
          break;
        }
        continue;
      }
      if (sequence < position) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      position = enqueue_position.load(std::memory_order_relaxed);
    }

    publication_checkpoint(PhysicalReadinessPublicationCheckpoint::kPositionClaimed);
    outstanding.fetch_add(1U, std::memory_order_acq_rel);
    auto free_phase = PhysicalReadinessCellPhase::kFree;
    if (!cell->phase.compare_exchange_strong(free_phase,
                                             PhysicalReadinessCellPhase::kClaimed,
                                             kAdmissionMemoryOrder,
                                             std::memory_order_acquire)) {
      static_cast<void>(release_outstanding_credit());
      cell->phase.store(PhysicalReadinessCellPhase::kCanceled, std::memory_order_relaxed);
      cell->ring_sequence.store(position + 1U, std::memory_order_release);
      fail(ErrorCode::kProtocolError);
      notify_wake();
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    publication_checkpoint(PhysicalReadinessPublicationCheckpoint::kCreditReserved);
    if (terminal.load(kAdmissionMemoryOrder) != PhysicalReadinessTerminal::kActive ||
        source_generation.load(std::memory_order_acquire) != expected_generation) {
      static_cast<void>(cancel_credit(*cell, false));
      cell->ring_sequence.store(position + 1U, std::memory_order_release);
      notify_wake();
      return make_unexpected(source_generation.load(std::memory_order_acquire) !=
                                     expected_generation
                                 ? Error{ErrorCode::kStaleEndpoint}
                                 : current_terminal_error());
    }

    const std::uint64_t source_sequence =
        descriptor.request.initial_source_sequence + static_cast<std::uint64_t>(position) + 1U;
    cell->event = RuntimeEvent{
        .kind = kind,
        .source = descriptor.request.source,
        .port = descriptor.request.port,
        .source_decision_time = reference_time,
        .reference_time = reference_time,
        .source_generation = expected_generation,
        .source_sequence = source_sequence,
    };
    auto claimed = PhysicalReadinessCellPhase::kClaimed;
    if (!cell->phase.compare_exchange_strong(claimed,
                                             PhysicalReadinessCellPhase::kReady,
                                             kAdmissionMemoryOrder,
                                             std::memory_order_acquire)) {
      cell->ring_sequence.store(position + 1U, std::memory_order_release);
      notify_wake();
      return make_unexpected(current_terminal_error());
    }
    publication_checkpoint(PhysicalReadinessPublicationCheckpoint::kReadyPublished);
    if (terminal.load(kAdmissionMemoryOrder) != PhysicalReadinessTerminal::kActive ||
        source_generation.load(std::memory_order_acquire) != expected_generation) {
      static_cast<void>(cancel_credit(*cell, false));
      cell->ring_sequence.store(position + 1U, std::memory_order_release);
      notify_wake();
      return make_unexpected(source_generation.load(std::memory_order_acquire) !=
                                     expected_generation
                                 ? Error{ErrorCode::kStaleEndpoint}
                                 : current_terminal_error());
    }
    cell->ring_sequence.store(position + 1U, std::memory_order_release);
    notify_wake();
    return source_sequence;
  }

  [[nodiscard]] expected<void, Error> observe_readable(bool readable, TimePoint observed_at,
                                                       std::uint64_t expected_generation) noexcept {
    if (source_generation.load(std::memory_order_acquire) != expected_generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (terminal.load(std::memory_order_acquire) != PhysicalReadinessTerminal::kActive) {
      return make_unexpected(current_terminal_error());
    }
    readable_level.store(readable, std::memory_order_release);
    if (!readable) {
      return {};
    }
    bool expected = false;
    if (!readable_outstanding.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return {};
    }
    auto published = publish(RuntimeEventKind::kReadable, observed_at, expected_generation);
    if (!published) {
      readable_outstanding.store(false, std::memory_order_release);
      return make_unexpected(std::move(published.error()));
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> observe_spin(bool asserted, TimePoint observed_at,
                                                   std::uint64_t expected_generation) noexcept {
    if (source_generation.load(std::memory_order_acquire) != expected_generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (terminal.load(std::memory_order_acquire) != PhysicalReadinessTerminal::kActive) {
      return make_unexpected(current_terminal_error());
    }
    if (!asserted) {
      spin_asserted.store(false, std::memory_order_release);
      spin_armed.store(true, std::memory_order_release);
      return {};
    }
    if (spin_asserted.exchange(true, std::memory_order_acq_rel)) {
      return {};
    }
    bool armed = true;
    if (!spin_armed.compare_exchange_strong(
            armed, false, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return {};
    }
    auto published = publish(RuntimeEventKind::kSpin, observed_at, expected_generation);
    if (!published) {
      // Preserve a retryable unaccepted edge under finite backpressure.
      spin_asserted.store(false, std::memory_order_release);
      spin_armed.store(true, std::memory_order_release);
      return make_unexpected(std::move(published.error()));
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> poll(TimePoint observed_at,
                                           std::uint64_t expected_generation) noexcept {
    if (!observed_at.valid() || observed_at.clock_id != descriptor.request.reference_clock) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (source_generation.load(std::memory_order_acquire) != expected_generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    const RuntimeEventKind kind = descriptor.request.kind;
    if (kind != RuntimeEventKind::kReadable && kind != RuntimeEventKind::kSpin) {
      return {};
    }
    if (descriptor.poll_asserted == nullptr) {
      // A readable source may be driven exclusively by its event callback.
      if (kind == RuntimeEventKind::kReadable) {
        return {};
      }
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    auto observed = descriptor.poll_asserted(descriptor.poll_state);
    if (!observed) {
      fail(observed.error().code);
      return make_unexpected(std::move(observed.error()));
    }
    return kind == RuntimeEventKind::kReadable
               ? observe_readable(*observed, observed_at, expected_generation)
               : observe_spin(*observed, observed_at, expected_generation);
  }

  [[nodiscard]] expected<std::size_t, Error> drain(std::span<RuntimeEvent> destination) noexcept {
    if (terminal.load(std::memory_order_acquire) == PhysicalReadinessTerminal::kFailed) {
      cancel_ready();
      return make_unexpected(current_terminal_error());
    }
    if (terminal.load(std::memory_order_acquire) != PhysicalReadinessTerminal::kActive) {
      cancel_ready();
      return std::size_t{0U};
    }

    std::size_t count = 0U;
    while (count < destination.size()) {
      const std::size_t position = dequeue_position.load(std::memory_order_relaxed);
      Cell& cell = cells[position % descriptor.request.capacity];
      if (cell.ring_sequence.load(std::memory_order_acquire) != position + 1U) {
        break;
      }
      auto expected = PhysicalReadinessCellPhase::kReady;
      if (cell.phase.compare_exchange_strong(expected,
                                             PhysicalReadinessCellPhase::kDrained,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        destination[count++] = cell.event;
      } else if (expected == PhysicalReadinessCellPhase::kCanceled) {
        auto canceled = PhysicalReadinessCellPhase::kCanceled;
        if (!cell.phase.compare_exchange_strong(canceled,
                                                PhysicalReadinessCellPhase::kFree,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        cell.ring_sequence.store(position + descriptor.request.capacity, std::memory_order_release);
      } else {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      dequeue_position.store(position + 1U, std::memory_order_relaxed);
    }
    return count;
  }

  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence,
                                             std::uint64_t expected_generation) noexcept {
    if (source_generation.load(std::memory_order_acquire) != expected_generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (source_sequence <= descriptor.request.initial_source_sequence) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    const std::uint64_t relative =
        source_sequence - descriptor.request.initial_source_sequence - 1U;
    if (relative > std::numeric_limits<std::size_t>::max()) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    const std::size_t position = static_cast<std::size_t>(relative);
    Cell& cell = cells[position % descriptor.request.capacity];
    if (cell.event.source_generation != expected_generation ||
        cell.event.source_sequence != source_sequence) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    auto expected = PhysicalReadinessCellPhase::kDrained;
    if (!cell.phase.compare_exchange_strong(expected,
                                            PhysicalReadinessCellPhase::kReleasing,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    cell.ring_sequence.store(position + descriptor.request.capacity, std::memory_order_release);
    if (!release_outstanding_credit()) {
      fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    // Publish the reusable phase only after the ring handoff and credit return. In a one-cell
    // ring, the old ready sequence is also the next producer sequence, so releasing kFree first
    // would let a new publication race and have its final sequence overwritten here.
    cell.phase.store(PhysicalReadinessCellPhase::kFree, std::memory_order_release);
    if (descriptor.request.kind == RuntimeEventKind::kReadable) {
      readable_outstanding.store(false, std::memory_order_release);
      if (readable_level.load(std::memory_order_acquire)) {
        notify_wake();
      }
    }
    return {};
  }

  void cancel_ready() noexcept {
    for (std::size_t index = 0U; index < descriptor.request.capacity; ++index) {
      static_cast<void>(cancel_credit(cells[index], false));
    }
  }

  void cancel_all_credits() noexcept {
    for (std::size_t index = 0U; index < descriptor.request.capacity; ++index) {
      static_cast<void>(cancel_credit(cells[index], true));
    }
    readable_outstanding.store(false, std::memory_order_release);
  }

  void transition(PhysicalReadinessTerminal next, ErrorCode error = ErrorCode::kSuccess) noexcept {
    if (next == PhysicalReadinessTerminal::kFailed) {
      fatal_error.store(error == ErrorCode::kSuccess ? ErrorCode::kFailure : error,
                        std::memory_order_release);
    }
    auto expected = PhysicalReadinessTerminal::kActive;
    if (terminal.compare_exchange_strong(
            expected, next, kAdmissionMemoryOrder, std::memory_order_acquire)) {
      cancel_ready();
      notify_wake();
    }
  }

  void fail(ErrorCode error) noexcept { transition(PhysicalReadinessTerminal::kFailed, error); }

  [[nodiscard]] expected<std::uint64_t, Error> advance_generation() noexcept {
    auto expected_terminal = PhysicalReadinessTerminal::kActive;
    if (!terminal.compare_exchange_strong(expected_terminal,
                                          PhysicalReadinessTerminal::kRestarting,
                                          kAdmissionMemoryOrder,
                                          std::memory_order_acquire)) {
      return make_unexpected(current_terminal_error());
    }
    const auto restore_active = [this]() noexcept {
      terminal.store(PhysicalReadinessTerminal::kActive, std::memory_order_release);
    };
    if (active_publishers.load(kAdmissionMemoryOrder) != 0U ||
        outstanding.load(std::memory_order_acquire) != 0U ||
        enqueue_position.load(std::memory_order_acquire) !=
            dequeue_position.load(std::memory_order_acquire)) {
      restore_active();
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    for (std::size_t index = 0U; index < descriptor.request.capacity; ++index) {
      if (cells[index].phase.load(std::memory_order_acquire) != PhysicalReadinessCellPhase::kFree) {
        restore_active();
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
    }
    if (doorbell_serial.test_and_set(std::memory_order_acquire)) {
      restore_active();
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    const std::uint64_t previous = source_generation.load(std::memory_order_relaxed);
    if (previous == std::numeric_limits<std::uint64_t>::max()) {
      doorbell_serial.clear(std::memory_order_release);
      fatal_error.store(ErrorCode::kResourceExhausted, std::memory_order_release);
      terminal.store(PhysicalReadinessTerminal::kFailed, std::memory_order_release);
      notify_wake();
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    enqueue_position.store(0U, std::memory_order_relaxed);
    dequeue_position.store(0U, std::memory_order_relaxed);
    for (std::size_t index = 0U; index < descriptor.request.capacity; ++index) {
      cells[index].event = RuntimeEvent{};
      cells[index].ring_sequence.store(index, std::memory_order_relaxed);
    }
    readable_level.store(false, std::memory_order_relaxed);
    readable_outstanding.store(false, std::memory_order_relaxed);
    spin_asserted.store(false, std::memory_order_relaxed);
    spin_armed.store(true, std::memory_order_relaxed);
    doorbell_watermark = 0U;
    source_generation.store(previous + 1U, std::memory_order_release);
    doorbell_serial.clear(std::memory_order_release);
    terminal.store(PhysicalReadinessTerminal::kActive, std::memory_order_release);
    notify_wake();
    return previous + 1U;
  }

  PhysicalReadinessBindingDescriptor descriptor;
  CellStorage cells;
  std::atomic<std::size_t> enqueue_position{};
  std::atomic<std::size_t> dequeue_position{};
  std::atomic<std::size_t> active_publishers{};
  std::atomic<std::size_t> outstanding{};
  std::atomic<std::uint64_t> wake_epoch{};
  std::atomic<PhysicalReadinessTerminal> terminal{PhysicalReadinessTerminal::kActive};
  std::atomic<ErrorCode> fatal_error{ErrorCode::kSuccess};
  std::atomic<std::uint64_t> source_generation{};
  std::atomic_bool bound{};

  std::atomic_bool readable_level{};
  std::atomic_bool readable_outstanding{};
  std::atomic_bool spin_asserted{};
  std::atomic_bool spin_armed{true};
  std::atomic_flag doorbell_serial = ATOMIC_FLAG_INIT;
  std::uint64_t doorbell_watermark{};
};

namespace {

class PhysicalReadinessPortBinding final : public PortBinding {
 public:
  explicit PhysicalReadinessPortBinding(std::shared_ptr<PhysicalReadinessState> state)
      : state_(std::move(state)) {}

  ~PhysicalReadinessPortBinding() override {
    if (state_ != nullptr) {
      state_->transition(PhysicalReadinessTerminal::kClosed);
    }
  }

  [[nodiscard]] expected<void, Error> poll(TimePoint observed_at) noexcept override {
    return state_->poll(observed_at, state_->source_generation.load(std::memory_order_acquire));
  }

  [[nodiscard]] std::optional<PortBindingPollTarget> poll_target() const noexcept override {
    const auto& descriptor = state_->descriptor;
    if ((descriptor.request.kind != RuntimeEventKind::kReadable &&
         descriptor.request.kind != RuntimeEventKind::kSpin) ||
        descriptor.poll_asserted == nullptr) {
      return std::nullopt;
    }
    return PortBindingPollTarget{
        .execution_lane = descriptor.execution_lane,
        .reference_clock = descriptor.request.reference_clock,
        .busy_poll = descriptor.wake_mode == WakeMode::kBusyPoll,
    };
  }

  [[nodiscard]] SourceId source() const noexcept override {
    return state_->descriptor.request.source;
  }

  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence) noexcept override {
    return state_->retire(source_sequence,
                          state_->source_generation.load(std::memory_order_acquire));
  }

  void cancel() noexcept override {
    state_->transition(PhysicalReadinessTerminal::kClosed);
    state_->cancel_all_credits();
  }

  [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override {
    return state_->advance_generation();
  }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept override {
    return state_->drain(destination);
  }

  [[nodiscard]] std::size_t capacity() const noexcept override {
    return state_->descriptor.request.capacity;
  }

  [[nodiscard]] bool closed() const noexcept override {
    return state_->terminal.load(std::memory_order_acquire) != PhysicalReadinessTerminal::kActive;
  }

  [[nodiscard]] expected<void, Error> terminal_status() const noexcept override {
    if (state_->terminal.load(std::memory_order_acquire) == PhysicalReadinessTerminal::kFailed) {
      return make_unexpected(state_->current_terminal_error());
    }
    return {};
  }

  // Every publication and terminal transition notifies the owner's wake; polled readable and spin
  // sources do not.
  [[nodiscard]] bool progress_announced() const noexcept override {
    return !poll_target().has_value();
  }

 private:
  std::shared_ptr<PhysicalReadinessState> state_;
};

class DoorbellSerialGuard {
 public:
  explicit DoorbellSerialGuard(std::atomic_flag& flag) noexcept : flag_(&flag) {}
  DoorbellSerialGuard(const DoorbellSerialGuard&) = delete;
  DoorbellSerialGuard& operator=(const DoorbellSerialGuard&) = delete;
  ~DoorbellSerialGuard() { flag_->clear(std::memory_order_release); }

 private:
  std::atomic_flag* flag_;
};

}  // namespace

PhysicalReadinessControl::PhysicalReadinessControl(std::shared_ptr<PhysicalReadinessState> state,
                                                   PlanGenerationId plan_generation,
                                                   RunGenerationId run_generation,
                                                   std::uint64_t source_generation) noexcept
    : state_(std::move(state)),
      plan_generation_(plan_generation),
      run_generation_(run_generation),
      source_generation_(source_generation) {}

expected<std::shared_ptr<PhysicalReadinessState>, Error> PhysicalReadinessControl::checked_state(
    RuntimeEventKind expected_kind) const noexcept {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (plan_generation_ != state_->descriptor.request.plan_generation ||
      run_generation_ != state_->descriptor.request.run_generation ||
      source_generation_ != state_->source_generation.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (state_->descriptor.request.kind != expected_kind) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return state_;
}

expected<void, Error> PhysicalReadinessControl::observe_readable(
    bool readable, TimePoint observed_at) const noexcept {
  auto checked = checked_state(RuntimeEventKind::kReadable);
  if (!checked) {
    return make_unexpected(std::move(checked.error()));
  }
  auto& state = **checked;
  return state.observe_readable(readable, observed_at, source_generation_);
}

expected<void, Error> PhysicalReadinessControl::post_notification(
    TimePoint accepted_at) const noexcept {
  auto checked = checked_state(RuntimeEventKind::kNotification);
  if (!checked) {
    return make_unexpected(std::move(checked.error()));
  }
  auto published =
      (*checked)->publish(RuntimeEventKind::kNotification, accepted_at, source_generation_);
  if (!published) {
    return make_unexpected(std::move(published.error()));
  }
  return {};
}

expected<PhysicalDoorbellAcceptance, Error> PhysicalReadinessControl::ring_doorbell(
    std::uint64_t token, TimePoint accepted_at) const noexcept {
  auto checked = checked_state(RuntimeEventKind::kDoorbell);
  if (!checked) {
    return make_unexpected(std::move(checked.error()));
  }
  if (token == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto& state = **checked;
  if (state.doorbell_serial.test_and_set(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const DoorbellSerialGuard guard{state.doorbell_serial};
  const std::uint64_t previous = state.doorbell_watermark;
  if (token < previous) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (token == previous) {
    return PhysicalDoorbellAcceptance{
        .coalesced = true,
        .previous_token = previous,
        .accepted_token = token,
    };
  }
  auto published = state.publish(RuntimeEventKind::kDoorbell, accepted_at, source_generation_);
  if (!published) {
    return make_unexpected(std::move(published.error()));
  }
  state.doorbell_watermark = token;
  return PhysicalDoorbellAcceptance{
      .witness_accepted = true,
      .previous_token = previous,
      .accepted_token = token,
      .missed_count = token - previous - 1U,
  };
}

expected<void, Error> PhysicalReadinessControl::observe_spin(bool asserted,
                                                             TimePoint observed_at) const noexcept {
  auto checked = checked_state(RuntimeEventKind::kSpin);
  if (!checked) {
    return make_unexpected(std::move(checked.error()));
  }
  auto& state = **checked;
  return state.observe_spin(asserted, observed_at, source_generation_);
}

expected<void, Error> PhysicalReadinessControl::poll_once(TimePoint observed_at) const noexcept {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const RuntimeEventKind kind = state_->descriptor.request.kind;
  if (kind != RuntimeEventKind::kReadable && kind != RuntimeEventKind::kSpin) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  if (plan_generation_ != state_->descriptor.request.plan_generation ||
      run_generation_ != state_->descriptor.request.run_generation ||
      source_generation_ != state_->source_generation.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  return state_->poll(observed_at, source_generation_);
}

expected<void, Error> PhysicalReadinessControl::retire(
    std::uint64_t source_sequence) const noexcept {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (plan_generation_ != state_->descriptor.request.plan_generation ||
      run_generation_ != state_->descriptor.request.run_generation ||
      source_generation_ != state_->source_generation.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  return state_->retire(source_sequence, source_generation_);
}

void PhysicalReadinessControl::close() const noexcept {
  if (state_ != nullptr && plan_generation_ == state_->descriptor.request.plan_generation &&
      run_generation_ == state_->descriptor.request.run_generation &&
      source_generation_ == state_->source_generation.load(std::memory_order_acquire)) {
    state_->transition(PhysicalReadinessTerminal::kClosed);
  }
}

void PhysicalReadinessControl::invalidate_stale() const noexcept {
  if (state_ != nullptr && plan_generation_ == state_->descriptor.request.plan_generation &&
      run_generation_ == state_->descriptor.request.run_generation &&
      source_generation_ == state_->source_generation.load(std::memory_order_acquire)) {
    state_->transition(PhysicalReadinessTerminal::kStale);
  }
}

void PhysicalReadinessControl::fail(ErrorCode error) const noexcept {
  if (state_ != nullptr && plan_generation_ == state_->descriptor.request.plan_generation &&
      run_generation_ == state_->descriptor.request.run_generation &&
      source_generation_ == state_->source_generation.load(std::memory_order_acquire) &&
      error != ErrorCode::kSuccess) {
    state_->fail(error);
  }
}

std::uint32_t PhysicalReadinessControl::execution_lane() const noexcept {
  return state_ == nullptr ? 0U : state_->descriptor.execution_lane;
}

WakeMode PhysicalReadinessControl::wake_mode() const noexcept {
  return state_ == nullptr ? WakeMode::kSleepingWaitset : state_->descriptor.wake_mode;
}

std::uint64_t PhysicalReadinessControl::wake_epoch() const noexcept {
  return state_ == nullptr ||
                 source_generation_ != state_->source_generation.load(std::memory_order_acquire)
             ? 0U
             : state_->wake_epoch.load(std::memory_order_acquire);
}

std::size_t PhysicalReadinessControl::outstanding() const noexcept {
  return state_ == nullptr ||
                 source_generation_ != state_->source_generation.load(std::memory_order_acquire)
             ? 0U
             : state_->outstanding.load(std::memory_order_acquire);
}

struct PhysicalReadinessBindingFactory::Slot {
  PhysicalReadinessBindingDescriptor descriptor;
  std::shared_ptr<PhysicalReadinessState> state;
};

PhysicalReadinessBindingFactory::PhysicalReadinessBindingFactory(SlotStorage slots,
                                                                 std::size_t slot_count) noexcept
    : slots_(std::move(slots)), slot_count_(slot_count) {}

PhysicalReadinessBindingFactory::~PhysicalReadinessBindingFactory() {
  close_all();
}

expected<std::shared_ptr<PhysicalReadinessBindingFactory>, Error>
PhysicalReadinessBindingFactory::create(
    std::span<const PhysicalReadinessBindingDescriptor> descriptors) {
  if (descriptors.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  // Validate the complete set before allocating any receiver state. This is
  // what makes missing required monitoring a startup/materialization failure
  // rather than a partially constructed receiver.
  for (std::size_t index = 0U; index < descriptors.size(); ++index) {
    if (!descriptor_valid(descriptors[index])) {
      const ErrorCode code =
          descriptors[index].monitored_health_required && !descriptors[index].monitored_health_bound
              ? ErrorCode::kNotSupported
              : ErrorCode::kInvalidArgument;
      return make_unexpected(Error{code});
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (descriptors[prior].request.source == descriptors[index].request.source) {
        return make_unexpected(Error{ErrorCode::kDuplicateName});
      }
    }
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    SlotStorage slots{new Slot[descriptors.size()]};
    for (std::size_t index = 0U; index < descriptors.size(); ++index) {
      slots[index].descriptor = descriptors[index];
      slots[index].state = std::make_shared<PhysicalReadinessState>(descriptors[index]);
    }
    return std::shared_ptr<PhysicalReadinessBindingFactory>(
        new PhysicalReadinessBindingFactory(std::move(slots), descriptors.size()));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<std::unique_ptr<PortBinding>, Error> PhysicalReadinessBindingFactory::bind(
    const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept {
  Slot* slot = nullptr;
  for (std::size_t index = 0U; index < slot_count_; ++index) {
    if (slots_[index].descriptor.request.source == request.source) {
      slot = &slots_[index];
      break;
    }
  }
  if (slot == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!request_equal(slot->descriptor.request, request) || !start_coordinate.valid() ||
      start_coordinate.clock_id != request.reference_clock) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  bool expected = false;
  if (!slot->state->bound.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kDuplicateName});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return std::unique_ptr<PortBinding>{new PhysicalReadinessPortBinding(slot->state)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    slot->state->bound.store(false, std::memory_order_release);
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<PhysicalReadinessControl, Error> PhysicalReadinessBindingFactory::control(
    SourceId source, PlanGenerationId plan_generation, RunGenerationId run_generation,
    std::uint64_t source_generation) const noexcept {
  for (std::size_t index = 0U; index < slot_count_; ++index) {
    const auto& slot = slots_[index];
    if (slot.descriptor.request.source != source) {
      continue;
    }
    if (plan_generation != slot.descriptor.request.plan_generation ||
        run_generation != slot.descriptor.request.run_generation ||
        source_generation != slot.state->source_generation.load(std::memory_order_acquire)) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (!slot.state->bound.load(std::memory_order_acquire)) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return PhysicalReadinessControl{slot.state, plan_generation, run_generation, source_generation};
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

void PhysicalReadinessBindingFactory::close_all() noexcept {
  for (std::size_t index = 0U; index < slot_count_; ++index) {
    if (slots_[index].state != nullptr) {
      slots_[index].state->transition(PhysicalReadinessTerminal::kClosed);
    }
  }
}

}  // namespace holoscan::detail
