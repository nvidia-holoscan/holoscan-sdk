/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::detail {

/// Identify one normalized bounded event accepted by the Dispatcher.
enum class RuntimeEventKind : std::uint8_t {
  kInputArrival,
  kInputEndOfStream,
  kClockTick,
  kWindowTimeout,
  kReadable,
  kNotification,
  kDoorbell,
  kSpin,
  kFiniteSourceComplete,
  kSourceError,
};

/// Fixed-size event passed from a physical or test binding to the Dispatcher.
///
/// The binding freezes source generation/sequence and the admitted reference
/// time before the event becomes visible. It does not assign an activation
/// \ref Tag or \ref TagTime: the Dispatcher does that later, when it forms a
/// candidate after draining all bounded ingress for the freeze cycle. Payload
/// retention and selection remain owned by the typed Channel binding
/// introduced by HSDK5-012; this record deliberately carries no payload pointer
/// or ownership claim.
struct RuntimeEvent {
  RuntimeEventKind kind{RuntimeEventKind::kInputArrival};
  SourceId source{};
  PortId port{};
  /// Exact source-domain coordinate that established deadline eligibility.
  ///
  /// An internally scheduled window-timeout witness stores its terminal acceptance coordinate in
  /// both time fields; the selector recovers the window anchor from its pending transaction. An
  /// input-end-of-stream event likewise stores its reference-clock observation in both fields,
  /// but remains control evidence and never establishes deadline eligibility.
  TimePoint source_decision_time{};
  /// Normalized trigger-reference coordinate used for candidate ordering.
  TimePoint reference_time{};
  std::uint64_t source_generation{};
  std::uint64_t source_sequence{};
  /// Receiver health action frozen after every preceding publisher operation was applied.
  EndpointClockHealthDisposition clock_health_disposition{
      EndpointClockHealthDisposition::kNotApplicable};
  /// Receiver health generation paired atomically with \ref clock_health_disposition.
  std::uint64_t clock_health_generation{};
  ErrorCode error{ErrorCode::kSuccess};
};

/// Describe the immutable lane/clock placement for one nonblocking predicate poll.
struct PortBindingPollTarget {
  std::uint32_t execution_lane{};
  ClockId reference_clock{};
  bool busy_poll{};
};

/// Private bounded event ingress implemented by runtime bindings.
///
/// Implementations must not allocate, block, or invoke user code from
/// `drain()`. A successful call removes exactly the returned prefix. An empty
/// result means no event is currently available, not end-of-run.
class PortBinding {
 public:
  virtual ~PortBinding() = default;

  /// Poll one nonblocking physical predicate at the supplied plan-domain coordinate.
  ///
  /// RunSession invokes this hook on the compiler-selected execution lane before
  /// draining ingress. Event-driven, notification, doorbell, clock, and test
  /// bindings retain the default no-op. An override must not block, allocate,
  /// acquire transport ownership, or invoke Operator code.
  [[nodiscard]] virtual expected<void, Error> poll(TimePoint observed_at) noexcept {
    static_cast<void>(observed_at);
    return {};
  }

  /// Poll while waiting out a contended cache lock rather than abandoning the pass.
  ///
  /// Only a caller that has already given something up on the strength of this poll completing
  /// should use this. Every other caller must use \ref poll, which never waits on a producer.
  [[nodiscard]] virtual expected<void, Error> poll_blocking(TimePoint observed_at) noexcept {
    return poll(observed_at);
  }

  /// Retry one declined \ref poll while waiting out the bounded contended cache lock.
  ///
  /// An ordinary poll that loses a cache lock leaves its published revision unobserved, but the
  /// wake token that announced it has already been consumed and the lock holder does not signal
  /// again on release. The owner that consumed that token may retry once here instead of parking
  /// for the bounded fallback wait. Every holder of an input cache lock is a short, non-blocking
  /// critical section, so this wait is bounded by that section rather than by a producer.
  [[nodiscard]] virtual expected<void, Error> poll_after_contention(
      TimePoint observed_at) noexcept {
    return poll(observed_at);
  }

  /// Return the admitted poll placement, or no target for event-driven bindings.
  [[nodiscard]] virtual std::optional<PortBindingPollTarget> poll_target() const noexcept {
    return std::nullopt;
  }

  /// Return the plan-scoped source identity when binding retirement is required.
  [[nodiscard]] virtual SourceId source() const noexcept { return {}; }

  /// Return whether the most recent \ref poll returned without draining everything available.
  ///
  /// A poll that loses a contended cache lock reports no error and creates no event, so a caller
  /// that published a readiness change and needs it converted in this same pass cannot otherwise
  /// tell that outcome from "nothing was ready".
  [[nodiscard]] virtual bool last_poll_incomplete() const noexcept { return false; }

  /// Return whether the most recent \ref poll was declined by a contended input cache lock.
  ///
  /// This is the exact condition \ref poll_after_contention exists for: a lost try_lock, not any
  /// other reason a poll can report incomplete.
  [[nodiscard]] virtual bool last_poll_lock_contended() const noexcept { return false; }

  /// Enqueue one internally scheduled window-timeout witness.
  ///
  /// The scheduler supplies the terminal acceptance coordinate that established timeout
  /// eligibility. A binding that supports this hook stores that coordinate in both Runtime event
  /// time fields and uses its existing bounded ingress and source-local sequence. It must not
  /// inspect payload storage, allocate, block, or invoke user code. The window anchor remains in
  /// the pending selector transaction and is not carried by this witness.
  ///
  /// @param port Input port whose pending window reached its admitted timeout.
  /// @param terminal_time Terminal acceptance coordinate in the admitted reference clock.
  /// @return Empty success after bounded enqueue, or a typed validation, pressure, closure, or
  /// unsupported-binding error without mutation.
  [[nodiscard]] virtual expected<void, Error> enqueue_window_timeout(
      PortId port, TimePoint terminal_time) noexcept {
    static_cast<void>(port);
    static_cast<void>(terminal_time);
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  /// Return one drained witness credit after exact-once Runtime retirement.
  [[nodiscard]] virtual expected<void, Error> retire(std::uint64_t source_sequence) noexcept {
    static_cast<void>(source_sequence);
    return {};
  }

  /// Release all remaining binding credits during terminal Runtime cancellation.
  virtual void cancel() noexcept {}

  /// Advance one quiescent physical source to its next lifecycle generation.
  ///
  /// The caller must first fence backend ingress and retire or cancel every
  /// accepted witness. Failure leaves the current generation active. Success
  /// stales all prior controls and resets bounded source-local sequencing; it
  /// does not recreate or take ownership of the transport resource.
  [[nodiscard]] virtual expected<std::uint64_t, Error> advance_source_generation() noexcept {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  [[nodiscard]] virtual expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept = 0;
  [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;
  [[nodiscard]] virtual bool closed() const noexcept = 0;

  /// Return the next clock-domain coordinate that can make this source READY.
  ///
  /// Physical bindings and finite direct bindings return no coordinate. A
  /// clock binding returns one plan-scoped coordinate so RunSession can park
  /// its controller without changing the schedule or manufacturing a tick.
  [[nodiscard]] virtual std::optional<TimePoint> next_wake() const noexcept { return std::nullopt; }
};

/// Describe one immutable-plan source that requires a concrete Runtime binding.
///
/// This cold request copies only plan-scoped identities and admitted finite
/// bounds. A binding factory must interpret them with the enclosing plan and
/// run generations; numeric identity equality alone is never provenance.
struct RuntimePortBindingRequest {
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  RuntimeEventKind kind{RuntimeEventKind::kInputArrival};
  SourceId source{};
  PortId port{};
  ClockId source_clock{};
  ClockId reference_clock{};
  std::uint64_t source_to_reference_conversion_id{};
  std::uint32_t execution_lane{};
  std::uint32_t capacity{};
  std::uint64_t source_generation{1U};
  std::uint64_t initial_source_sequence{};
};

/// Materialize one private readiness adapter at its admitted lifecycle boundary.
///
/// Production physical adapters and the non-installed direct test adapter use
/// this same seam. The factory is retained only in process-local executable
/// inventory; it is excluded from plan serialization, digest, and public API.
/// A lifecycle-owned notification adapter may bind before c kArm so that its
/// sender can be acquired there, but it shall not accept witnesses until the
/// complete global c kStart barrier succeeds.
class RuntimePortBindingFactory {
 public:
  virtual ~RuntimePortBindingFactory() = default;

  /// Create one bounded adapter whose first witness cannot predate successful start.
  [[nodiscard]] virtual expected<std::unique_ptr<PortBinding>, Error> bind(
      const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept = 0;
};

/// Own a preallocated in-process event ring behind the ordinary Runtime seam.
///
/// This private adapter is used by the source-tree direct backend and by
/// process-local bindings that already normalized their physical witness. It
/// is not an Operator testing queue and is never installed as public API.
class BoundedEventPortBinding final : public PortBinding {
 public:
  explicit BoundedEventPortBinding(std::size_t capacity, SourceId source = {});

  BoundedEventPortBinding(const BoundedEventPortBinding&) = delete;
  BoundedEventPortBinding& operator=(const BoundedEventPortBinding&) = delete;
  BoundedEventPortBinding(BoundedEventPortBinding&&) = delete;
  BoundedEventPortBinding& operator=(BoundedEventPortBinding&&) = delete;
  ~BoundedEventPortBinding() override = default;

  [[nodiscard]] expected<void, Error> push(RuntimeEvent event) noexcept;
  [[nodiscard]] SourceId source() const noexcept override { return source_; }
  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept override;
  [[nodiscard]] std::size_t capacity() const noexcept override { return storage_.size(); }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }
  void close() noexcept { closed_ = true; }

 private:
  SourceId source_{};
  std::vector<RuntimeEvent> storage_;
  std::size_t head_{};
  std::size_t tail_{};
  std::size_t size_{};
  bool closed_{};
};

}  // namespace holoscan::detail
