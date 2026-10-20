/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/readiness_handle.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/time/time_point.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan {
namespace channel_internal {

struct InlinePayloadTestAccess;
class SubscriberWaitContext;

/// @brief Attach one private Channel subscriber to a shared backend wait set.
///
/// Implementations retain their own backend attachment, which keeps the shared wait backend and
/// subscriber transport port alive independently of the wait-set facade. The facade therefore may
/// be destroyed before attached subscriber wrappers; later attachment destruction still detaches
/// safely. This transport-private interface lets Runtime multiplex inline, Tensor, and
/// Tensor-schema subscribers without exposing backend endpoint or wait-set types.
class HOLOSCAN_CORE_EXPORT WaitSetSubscriber {
 public:
  /// @brief Hold one exclusive wait-owner claim for a complete public wait operation.
  class WaitOwnerGuard {
   public:
    WaitOwnerGuard(const WaitOwnerGuard&) = delete;
    WaitOwnerGuard& operator=(const WaitOwnerGuard&) = delete;

    /// @brief Transfer a wait-owner claim from another guard.
    /// @param other Guard whose claim is transferred.
    WaitOwnerGuard(WaitOwnerGuard&& other) noexcept
        : waiting_(std::exchange(other.waiting_, nullptr)) {}

    /// @brief Replace this guard's claim with another guard's claim.
    /// @param other Guard whose claim is transferred.
    /// @return This guard.
    WaitOwnerGuard& operator=(WaitOwnerGuard&& other) noexcept {
      if (this != &other) {
        release();
        waiting_ = std::exchange(other.waiting_, nullptr);
      }
      return *this;
    }

    /// @brief Release the wait-owner claim, if any.
    ~WaitOwnerGuard() { release(); }

   private:
    friend class WaitSetSubscriber;
    explicit WaitOwnerGuard(std::atomic_flag& waiting) noexcept : waiting_(&waiting) {}
    void release() noexcept {
      if (waiting_ != nullptr) {
        waiting_->clear(std::memory_order_release);
        waiting_ = nullptr;
      }
    }
    std::atomic_flag* waiting_{};
  };

  /// @brief Construct a subscriber with no wait context; one is built on the first wait.
  /// @note Defined out of line because the cached wait context is an incomplete type here, and a
  /// defaulted constructor would need its deleter instantiated for construction-time unwinding.
  HOLOSCAN_CORE_EXPORT WaitSetSubscriber();
  WaitSetSubscriber(const WaitSetSubscriber&) = delete;
  WaitSetSubscriber& operator=(const WaitSetSubscriber&) = delete;
  WaitSetSubscriber(WaitSetSubscriber&&) = delete;
  WaitSetSubscriber& operator=(WaitSetSubscriber&&) = delete;
  HOLOSCAN_CORE_EXPORT virtual ~WaitSetSubscriber();

  /// @brief Attach this subscriber to an opaque backend wait set.
  /// @param wait_set Non-null backend-private wait-set address retained by the caller.
  /// @return Stable wait-set-local identity, or \ref ErrorCode::kInvalidArgument for a null or
  /// repeated attachment, or another checked backend setup error.
  [[nodiscard]] virtual expected<std::uint64_t, Error> attach_wait_set(void* wait_set) = 0;

  /// @brief Detach this subscriber from its current opaque backend wait set.
  /// @return Success after the attachment is retired, including when none was created.
  /// @note A readiness pump or waiter must be joined before this cold lifecycle operation.
  [[nodiscard]] virtual expected<void, Error> detach_wait_set() noexcept { return {}; }

  /// @brief Retire source-private local endpoint resources before owner destruction.
  /// @return Success, or the allocation-free code of the first detach failure.
  /// @note This operation is local consumer teardown. It publishes no producer operation or EOS.
  [[nodiscard]] virtual expected<void, Error> close_local_endpoint() noexcept {
    auto detached = detach_wait_set();
    if (!detached) {
      return make_unexpected(Error{detached.error().code});
    }
    return {};
  }

  /// @brief Query pending samples without consuming one.
  /// @return Readiness, or a checked backend query error.
  /// @note Readiness is a snapshot hint. A following receive may still report
  /// \ref ErrorCode::kNotReady when a lossy replacement, a private control operation, or a
  /// competing consumer changed the queue in between. Implementations that own more than one
  /// transport lane report the union, so one call answers for the whole subscriber.
  [[nodiscard]] virtual expected<bool, Error> has_samples() const = 0;

  /// @brief Return whether this subscriber's service admits kernel event wake.
  /// @return True when the service-static \ref ChannelOptions::enable_event_wake contract is
  /// enabled for this endpoint.
  /// @note This function does not throw. It is a frozen creation-time fact, not a runtime state.
  [[nodiscard]] virtual bool event_wake_enabled() const noexcept = 0;

  /// @brief Return whether a backend wait owner currently retains this subscriber's lanes.
  /// @return True for a Runtime pump, cached private wait context, or readiness handle.
  [[nodiscard]] virtual bool wait_set_attached() const noexcept { return false; }

  /// @brief Return whether a readiness handle specifically retains this subscriber's lanes.
  /// @return True for an external readiness handle, false for a Runtime or private wait set.
  [[nodiscard]] virtual bool readiness_handle_attached() const noexcept { return false; }

  /// @brief Claim this subscriber's readiness descriptor for an external event loop.
  /// @return A move-only readiness handle, or a checked capability or ownership error.
  [[nodiscard]] virtual expected<ChannelReadinessHandle, Error> readiness() {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  /// @brief Claim the subscriber's single wait-owner slot for one complete wait operation.
  /// @return An RAII claim, or \ref ErrorCode::kInvalidArgument when another owner is active.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<WaitOwnerGuard, Error> claim_wait_owner() noexcept;

  /// @brief Claim the owner slot and release a cached private wait context before handle creation.
  /// @return An RAII claim, or a checked ownership or detach error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<WaitOwnerGuard, Error>
  claim_readiness_owner() noexcept;

  /// @brief Block until this subscriber may have a sample, the deadline expires, or a stop is
  /// requested.
  ///
  /// The first call builds and caches a private wait set, attaches this subscriber's transport
  /// lanes and a cancellation trigger, and establishes the inbound connection. Later calls reuse
  /// that context and rebuild none of it. They are not unconditionally allocation-free: the wait
  /// backend records which lanes woke it in a vector that is cleared rather than shrunk and is
  /// still empty after setup, so the first later call that a lane actually satisfies grows it once.
  /// That growth is bounded by the lane count and does not recur, and the first wait after a
  /// cancellation allocates a replacement trigger. Nothing on this path allocates per call.
  ///
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param stop_token Optional borrowed cancellation observer; null disables cancellation.
  /// @return Empty success when a sample may be available, or a checked wait error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed with no readiness.
  /// @retval ErrorCode::kInterrupted \p stop_token requested a stop.
  /// @retval ErrorCode::kNotSupported This endpoint was created without event wake, so it has no
  /// kernel event lane. This operation deliberately does not fall back to polling.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, such as the Runtime readiness pump, or another wait on this subscriber is already
  /// in flight. Both are caller errors that leave the subscriber unchanged.
  /// @retval ErrorCode::kResourceExhausted Setting up the wait ran out of memory. Building the
  /// first wait context, re-arming a cancelled trigger, and the diagnostics attached to the errors
  /// above all allocate, and none of them can report failure any other way.
  /// @note The wait is event-driven with a bounded recovery poll, not purely event-driven. A
  /// publication wakes it from a kernel wait, but transport notification is best effort, so a
  /// blocked wait also re-checks readiness about once a second. An idle waiting subscriber
  /// therefore costs roughly one wakeup per second rather than nothing at all, and readiness that
  /// is reported by a lane which is not attached to the wait set -- currently the catalog
  /// clock-health lane -- is discovered on that fallback rather than by a wake. Treat the interval
  /// as an implementation bound rather than a contract.
  /// @note Cancellation is spelled \c std::stop_token here on purpose: a token is reusable and
  /// standard, where the transport's own wake primitive is a monotonic single-use latch. This
  /// layer owns one of those internally and fires it from a \c std::stop_callback, so a caller
  /// never has to see it or replace it after a cancellation. Naming the transport type here would
  /// breach this header's transport opacity, so the pointer runs the other way: the transport
  /// header documents the correspondence.
  /// @note This function does not throw: an allocation failure on the setup path is translated to
  /// \ref ErrorCode::kResourceExhausted rather than crossing the \c noexcept boundary. One
  /// subscriber supports one waiting thread; calls are not
  /// internally serialized. A second concurrent waiter is rejected with
  /// \ref ErrorCode::kInvalidArgument in every build configuration, rather than being allowed to
  /// share the first waiter's wait context. The rejected call does not disturb the wait already in
  /// flight.
  /// @note Public subscriber helpers retain one owner claim across their complete receive loop and
  /// call \ref wait_for_readiness_claimed. This unclaimed entry point remains exported for existing
  /// direct callers and acquires its own claim.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> wait_for_readiness(
      std::chrono::steady_clock::time_point deadline, const std::stop_token* stop_token) noexcept;

  /// @brief Wait while the caller retains a successful \ref claim_wait_owner result.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param stop_token Optional borrowed cancellation observer.
  /// @return Empty success when readiness may exist, or a checked wait error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> wait_for_readiness_claimed(
      std::chrono::steady_clock::time_point deadline, const std::stop_token* stop_token) noexcept;

  /// @brief Return the private transport runtime used to build this subscriber's wait context.
  /// @return Borrowed backend-private runtime address, or nullptr when this binding has none.
  /// @note The parent Channel runtime outlives every endpoint created from it, so the returned
  /// borrow stays valid for this subscriber's lifetime. This stays in \ref channel_internal rather
  /// than being protected so a composite binding can delegate to the inner binding that owns the
  /// transport lanes.
  [[nodiscard]] virtual void* wait_context_runtime() noexcept = 0;

 private:
  friend struct InlinePayloadTestAccess;

  /// Body of \ref wait_for_readiness. Separated so the public entry point is a thin translator that
  /// can catch an allocation failure and honour its \c noexcept, rather than asserting a promise
  /// the body cannot keep. Private and non-virtual, so it adds no layout or vtable surface.
  [[nodiscard]] expected<void, Error> wait_for_readiness_checked(
      std::chrono::steady_clock::time_point deadline, const std::stop_token* stop_token);

  /// @warning The readiness virtuals above deliberately make one pre-GA ABI transition from EA
  /// snapshots that did not expose readiness handles. A client compiled against those older
  /// headers must not be mixed with the current Core library; rebuild all in-process components
  /// from one SDK package. The current virtual order and these two members are ABI state for every
  /// endpoint family. Adding a third field, inserting or reordering another virtual, or changing a
  /// signature is another ABI break. New wait state belongs inside \ref SubscriberWaitContext --
  /// which is heap-held, private to the implementation, and already owns the rest of the wait
  /// state. The context pointer makes a warmed wait rebuild nothing, and the flag must be readable
  /// before the context exists.
  std::unique_ptr<SubscriberWaitContext> wait_context_;
  /// Claimed for the duration of one wait, so a second concurrent waiter can be rejected rather
  /// than allowed to share wait_context_. Both the claim and the rejection are unconditional, so
  /// this exported class has the same layout and the same behaviour in every configuration.
  std::atomic_flag waiting_;
};

/// @brief Consume and retire one source-private subscriber owner.
/// @tparam Subscriber Concrete endpoint type derived from @ref WaitSetSubscriber.
/// @param subscriber Owning endpoint pointer; null on return, including after a retirement error.
/// @return Success, @ref ErrorCode::kClosed when already consumed, or a code-only cleanup error.
/// @note Existing endpoint destruction remains the sole authority for staged and backend cleanup.
template <typename Subscriber>
[[nodiscard]] expected<void, Error> close_local_subscriber(
    std::unique_ptr<Subscriber>& subscriber) noexcept {
  if (!subscriber) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto retiring = std::move(subscriber);
  auto closed = retiring->close_local_endpoint();
  if (!closed) {
    return make_unexpected(Error{closed.error().code});
  }
  return {};
}

/// @brief Return a borrowed stop-token pointer for \ref WaitSetSubscriber::wait_for_readiness.
///
/// This is the one place the two spellings of a stop token meet. Everything a caller can see takes
/// \c const \c std::stop_token&, matching the rest of the SDK: a wait observes the token, it never
/// owns it, and passing by value would cost an atomic refcount pair per call even on
/// \ref ChannelSubscriber::wait_for_sample's fast path, which can return without waiting at all.
/// The virtual layer below takes a raw pointer instead, because that is what lets this function
/// answer "no usable token" with nullptr so the wait can skip constructing a
/// \c std::stop_callback. That asymmetry is deliberate and is confined to \ref channel_internal.
///
/// @param token Caller-owned token that must outlive every wait it is passed to.
/// @return Address of \p token when it can still observe a stop, or nullptr when it cannot.
/// @note This function does not throw. A default-constructed token has no stop state, so this
/// returns nullptr and the wait skips constructing a \c std::stop_callback entirely.
[[nodiscard]] inline const std::stop_token* borrowed_stop_token(
    const std::stop_token& token) noexcept {
  return token.stop_possible() ? &token : nullptr;
}

/// @brief Report readiness for \p subscriber, blocking until \p deadline only if it is not ready.
///
/// The shared tail of every family's \c wait_for_sample. Each facade still owns its own prologue --
/// the moved-from check and, where the family has one, the end-of-stream latch -- because that is
/// facade state rather than subscriber state. Everything after it is identical, and was
/// triplicated across three headers until the three drifted apart once already.
///
/// @param subscriber Endpoint whose readiness is queried and then awaited; must outlive this call.
/// @param deadline Steady-clock coordinate after which the wait reports a timeout.
/// @param stop_token Optional borrowed cancellation observer; null disables cancellation.
/// @return Empty success when a sample may be available, or a checked wait error.
/// @note An endpoint that is already ready returns success without consulting the event lane, so a
/// poll-only endpoint holding a sample is not refused with \ref ErrorCode::kNotSupported.
[[nodiscard]] inline expected<void, Error> wait_for_sample_readiness(
    WaitSetSubscriber& subscriber, const std::chrono::steady_clock::time_point deadline,
    const std::stop_token* const stop_token) {
  auto claim = subscriber.claim_wait_owner();
  if (!claim) {
    return make_unexpected(std::move(claim).error());
  }
  auto ready = subscriber.has_samples();
  if (!ready) {
    return make_unexpected(std::move(ready).error());
  }
  if (*ready) {
    return {};
  }
  return subscriber.wait_for_readiness_claimed(deadline, stop_token);
}

/// @brief Receive one sample from \p subscriber, waiting for readiness until \p deadline.
///
/// The first attempt runs before any wait context exists, so an already queued sample is returned
/// without building a wait set and without a kernel wait. \ref ErrorCode::kNotReady is absorbed
/// from both the attempt and the wait, because readiness is only a hint: an attempt that loses its
/// race against a lossy replacement, a private control operation, or a competing consumer waits
/// again against the same absolute deadline. Every other error is returned unchanged from the
/// first attempt that produces it, so a terminal \ref ErrorCode::kClosed or an active-loan
/// \ref ErrorCode::kFailure is reported before any wait is built.
///
/// @tparam Receive Callable taking no arguments and returning this endpoint family's expected
/// receive result.
/// @param subscriber Endpoint whose readiness is awaited; it must outlive this call.
/// @param deadline Steady-clock coordinate after which the wait reports a timeout.
/// @param stop_token Optional borrowed cancellation observer; null disables cancellation.
/// @param receive One repeatable receive attempt, invoked at least once.
/// @return The receive result, or the first error that is not \ref ErrorCode::kNotReady.
/// @retval ErrorCode::kTimeout \p deadline elapsed with no receivable sample.
/// @retval ErrorCode::kInterrupted \p stop_token requested a stop.
/// @note This function never returns \ref ErrorCode::kNotReady. A subscriber that keeps reporting
/// readiness while every attempt loses its race retries at full speed until \p deadline, which is
/// checked after every readiness query on both its outcomes; that absolute deadline is the only
/// bound, which is why no duration overload exists.
template <typename Receive>
[[nodiscard]] std::invoke_result_t<Receive&> wait_and_receive_next(
    WaitSetSubscriber& subscriber, const std::chrono::steady_clock::time_point deadline,
    const std::stop_token* const stop_token, Receive receive) {
  auto claim = subscriber.claim_wait_owner();
  if (!claim) {
    return make_unexpected(std::move(claim).error());
  }
  while (true) {
    auto received = receive();
    if (received || received.error().code != ErrorCode::kNotReady) {
      return received;
    }

    auto ready = subscriber.wait_for_readiness_claimed(deadline, stop_token);
    if (!ready && ready.error().code != ErrorCode::kNotReady) {
      return make_unexpected(std::move(ready).error());
    }
    // Both outcomes land here. Readiness is only a hint and wait_for_readiness returns success as
    // soon as has_samples() is true, without blocking and without consulting the deadline, so a
    // subscriber whose readiness never yields a receivable sample would otherwise retry at full
    // speed forever -- the caller's absolute deadline is the only bound this loop has, and it has
    // to be read on the hint path too, not just when the readiness query itself reports not-ready.
    if (deadline <= std::chrono::steady_clock::now()) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
  }
}

/// @brief Classify whether an inline publication has crossed the submit boundary.
enum class InlinePublicationPhase : std::uint8_t {
  kPreSubmitRejected,    ///< No backend submit occurred and producer authority is retained.
  kAccepted,             ///< Backend accepted the publication attempt.
  kPostSubmitUncertain,  ///< Submit began but its terminal delivery result is unknown.
};

/// @brief Preserve private publication phase and recipient evidence for Runtime.
struct InlinePublicationResult {
  /// Submit-boundary disposition of this publication attempt.
  InlinePublicationPhase phase{InlinePublicationPhase::kPreSubmitRejected};
  /// Backend recipient count after an accepted submit.
  std::uint32_t recipient_count{};
  /// Checked failure evidence; success when \ref phase is \ref InlinePublicationPhase::kAccepted.
  Error error{ErrorCode::kSuccess};
};

class InlinePayloadSubscriber;

/// @brief Describe one allocation-free encoder for a bounded transport payload loan.
struct InlinePayloadEncoder {
  /// Opaque caller-owned encoder state retained for the duration of the call.
  void* state{};
  /// Encode into the supplied transport-owned destination and return the used byte count.
  expected<std::size_t, Error> (*encode)(void*, std::span<std::byte>) noexcept {};
  /// Maximum destination size admitted by the caller's codec contract.
  std::size_t capacity{};
};

/// @brief Keep one received inline payload borrowed through verification and materialization.
class HOLOSCAN_CORE_EXPORT InlinePayloadReceiveGuard {
 public:
  /** @brief Construct a guard for one active subscriber payload loan.
   * @param subscriber Subscriber whose active loan will be released by this guard.
   * @param payload Exact used payload prefix borrowed from the transport.
   * @param metadata Metadata carried by the borrowed sample. */
  HOLOSCAN_CORE_EXPORT InlinePayloadReceiveGuard(InlinePayloadSubscriber& subscriber,
                                                 std::span<const std::byte> payload,
                                                 SampleMetadata metadata) noexcept;
  InlinePayloadReceiveGuard(const InlinePayloadReceiveGuard&) = delete;
  InlinePayloadReceiveGuard& operator=(const InlinePayloadReceiveGuard&) = delete;
  /** @brief Move ownership of an active subscriber payload loan.
   * @param other Guard whose loan ownership is transferred. */
  HOLOSCAN_CORE_EXPORT InlinePayloadReceiveGuard(InlinePayloadReceiveGuard&& other) noexcept;
  /** @brief Replace this guard's payload loan with another guard's loan.
   * @param other Guard whose loan ownership is transferred.
   * @return Reference to this guard after releasing its previous loan. */
  HOLOSCAN_CORE_EXPORT InlinePayloadReceiveGuard& operator=(
      InlinePayloadReceiveGuard&& other) noexcept;
  /// @brief Release the active subscriber payload loan, if any.
  HOLOSCAN_CORE_EXPORT ~InlinePayloadReceiveGuard() noexcept;

  /// @brief Return the exact used payload prefix while this guard remains live.
  /// @return Borrowed transport bytes, or an empty span after a move.
  [[nodiscard]] std::span<const std::byte> payload() const noexcept { return payload_; }
  /// @brief Return the metadata carried by the borrowed sample.
  /// @return Immutable sample metadata.
  [[nodiscard]] const SampleMetadata& metadata() const noexcept { return metadata_; }

 private:
  InlinePayloadSubscriber* subscriber_{};
  std::span<const std::byte> payload_{};
  SampleMetadata metadata_{};
};

/// @brief Multiplex private subscriber readiness without exposing backend handles.
///
/// The facade retains shared backend wait state and is neither copyable nor movable. Each
/// attachment independently retains that backend and its subscriber port, so attachments may
/// outlive the facade, and one subscriber binding may be attached only once. Setup and wait
/// operations are externally serialized while the facade is live; a successful warmed wait copies
/// results only into caller-provided bounded storage.
class HOLOSCAN_CORE_EXPORT InlinePayloadWaitSet {
 public:
  /// @brief Construct an empty private wait set before backend initialization.
  InlinePayloadWaitSet() = default;

  /// @brief Reject copying backend wait-set ownership.
  /// @param other Wait set that would otherwise be copied.
  InlinePayloadWaitSet(const InlinePayloadWaitSet&) = delete;

  /// @brief Reject copy assignment of backend wait-set ownership.
  /// @param other Wait set that would otherwise replace this one.
  InlinePayloadWaitSet& operator=(const InlinePayloadWaitSet&) = delete;

  /// @brief Reject moving a polymorphic backend wait set.
  /// @param other Wait set that would otherwise be moved.
  InlinePayloadWaitSet(InlinePayloadWaitSet&&) = delete;

  /// @brief Reject move assignment of a polymorphic backend wait set.
  /// @param other Wait set that would otherwise replace this one.
  InlinePayloadWaitSet& operator=(InlinePayloadWaitSet&&) = delete;

  /// @brief Release the facade's backend reference; subscriber attachments may retain it longer.
  HOLOSCAN_CORE_EXPORT virtual ~InlinePayloadWaitSet();

  /// @brief Attach one event-enabled subscriber and return its wait-set-local identity.
  /// @param subscriber Subscriber whose backend port is retained by the attachment.
  /// @return Stable wait-set-local identity, or a checked setup error.
  [[nodiscard]] virtual expected<std::uint64_t, Error> attach(WaitSetSubscriber& subscriber) = 0;
  /// @brief Perform one bounded kernel wait.
  /// @param timeout Maximum duration to block.
  /// @return True when the backend reported a possible readiness event, false on timeout, or a
  /// checked wait error.
  [[nodiscard]] virtual expected<bool, Error> wait_for(
      std::chrono::nanoseconds timeout) noexcept = 0;
  /// @brief Copy ready wait identities into preallocated destination storage.
  /// @param destination Storage for identities reported by the most recent successful wait.
  /// @return Number of copied identities, or \ref ErrorCode::kResourceExhausted when the
  /// destination is too small.
  [[nodiscard]] virtual expected<std::size_t, Error> ready_handles(
      std::span<std::uint64_t> destination) const noexcept = 0;
};

/** @brief Construct a channel Error.
 * @param code Error code.
 * @param message Diagnostic message.
 * @return Error containing code and message. */
// The by-value string is a sink: callers may transfer an owned diagnostic without another copy.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] inline Error make_channel_error(ErrorCode code, std::string message) noexcept {
  return Error{code, std::move(message)};
}

/// @brief Publish inline payload bytes and metadata through a private transport binding.
///
/// Implementations own one backend endpoint but borrow their parent Channel runtime. They are
/// neither copyable nor movable, and callers externally serialize mutation on one object.
class HOLOSCAN_CORE_EXPORT InlinePayloadPublisher {
 public:
  InlinePayloadPublisher() = default;
  InlinePayloadPublisher(const InlinePayloadPublisher&) = delete;
  InlinePayloadPublisher& operator=(const InlinePayloadPublisher&) = delete;
  InlinePayloadPublisher(InlinePayloadPublisher&&) = delete;
  InlinePayloadPublisher& operator=(InlinePayloadPublisher&&) = delete;
  HOLOSCAN_CORE_EXPORT virtual ~InlinePayloadPublisher();

  /** @brief Send inline payload bytes.
   * @param payload Bytes to send.
   * @param metadata Public sample metadata.
   * @return Recipient count after accepted publication, including zero when no subscriber was
   * connected; a pre-submit error; \ref ErrorCode::kPublicationPartial for a known strict subset;
   * or \ref ErrorCode::kPublicationIndeterminate after an uncertain submit failure. */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit(
      std::span<const std::byte> payload, const SampleMetadata& metadata);
  /// @brief Publish while preserving the private submit-boundary disposition.
  /// @param payload Bytes to send.
  /// @param metadata Public sample metadata.
  /// @return Submit phase, recipient count, and checked error evidence for Runtime accounting.
  [[nodiscard]] virtual InlinePublicationResult emit_phase(std::span<const std::byte> payload,
                                                           const SampleMetadata& metadata) = 0;
  /// @brief Encode directly into a bounded outbound transport loan and publish its used prefix.
  /// @param encoder Allocation-free callback and maximum codec destination size.
  /// @param metadata Public sample metadata; the publisher replaces byte_size with the encoded
  /// byte count.
  /// @return Submit phase, recipient count, and checked error evidence for Runtime accounting.
  [[nodiscard]] virtual InlinePublicationResult emit_encoded_phase(
      const InlinePayloadEncoder& encoder, const SampleMetadata& metadata);
  /// @brief Label one capture coordinate with this catalog binding's resolved clock domain.
  /// @param timestamp_ns Exact caller-supplied nanosecond coordinate.
  /// @return Bound time point, or \ref ErrorCode::kInvalidArgument when this publisher has no
  /// catalog capture-clock authority.
  [[nodiscard]] virtual expected<TimePoint, Error> make_capture_time(
      std::int64_t timestamp_ns) const noexcept;
  /// @brief Submit one publisher-bound clock-health control report.
  /// @param report Assessment and portable primary reason.
  /// @return Success after local bounded transport acceptance; this is not receiver
  /// acknowledgment and does not prove that Runtime accepted or applied the report.
  /// @retval ErrorCode::kInvalidArgument The endpoint has no admitted status-report capability or
  /// the assessment/reason combination is invalid.
  /// @retval ErrorCode::kNotReady The checked health lane has no connected subscriber.
  /// @retval ErrorCode::kResourceExhausted A bounded sequence or transport resource is exhausted.
  /// @retval ErrorCode::kPublicationIndeterminate Submission began but its terminal outcome is
  /// unknown; the committed operation identity is consumed and the caller must not retry the same
  /// logical report.
  /// @retval ErrorCode::kPublicationPartial Submission reached a known strict subset; the
  /// committed operation identity is consumed and the caller must not retry the same report.
  /// @note A failure that occurs before backend submission leaves operation/report sequence and
  /// receiver freshness state unchanged. Calls on one publisher are externally serialized with
  /// every data publication.
  [[nodiscard]] virtual expected<void, Error> report_clock_health(ClockHealthReport report);
  /// @brief Publish an explicit end-of-stream operation for this endpoint generation.
  /// @return Success after one receiver accepted the ordered close operation.
  /// @retval ErrorCode::kNotReady No receiver observed the close, so producer authority remains.
  /// @retval ErrorCode::kClosed This endpoint generation was already closed.
  /// @retval ErrorCode::kPublicationIndeterminate Submission began but its terminal outcome is
  /// unknown; producer authority is consumed and the close must not be retried.
  /// @retval ErrorCode::kPublicationPartial Submission reached a known strict subset; producer
  /// authority is consumed and the close must not be retried.
  /// @note Silence, endpoint destruction, disconnect, and process exit never call this operation
  /// and therefore never constitute end-of-stream evidence.
  [[nodiscard]] virtual expected<void, Error> close_stream();
  /// @brief Refresh the backend connection snapshot used by \ref has_subscribers.
  /// @return Success, or a checked closed or backend refresh error.
  [[nodiscard]] virtual expected<void, Error> update_connections() noexcept = 0;
  /// @brief Return the most recently refreshed subscriber-presence snapshot.
  /// @return True when at least one subscriber was observed; false otherwise.
  /// @note This advisory snapshot is not proof that a subsequent publication has a recipient.
  [[nodiscard]] virtual bool has_subscribers() const noexcept = 0;
};

/// @brief Report one received inline payload's metadata and the exact bytes this operation wrote.
///
/// \ref payload_size is transport evidence: it counts only the bytes the completed receive placed
/// in the caller's destination. \ref SampleMetadata::byte_size is a producer claim carried on the
/// wire and may disagree. A caller that reuses a destination buffer across receives must reject
/// any sample whose \ref payload_size does not cover every byte it later reads; otherwise the
/// unwritten tail still holds bytes retained from an earlier sample.
struct InlinePayloadReceiveResult {
  SampleMetadata metadata{};   ///< Metadata carried by the received sample.
  std::size_t payload_size{};  ///< Exact bytes written into the caller's destination.
};

/// @brief Receive inline payload bytes and metadata through a private transport binding.
///
/// Implementations own one backend endpoint but borrow their parent Channel runtime. They are
/// neither copyable nor movable, and callers externally serialize receive and wait-set attachment.
class HOLOSCAN_CORE_EXPORT InlinePayloadSubscriber : public WaitSetSubscriber {
 public:
  InlinePayloadSubscriber() = default;
  InlinePayloadSubscriber(const InlinePayloadSubscriber&) = delete;
  InlinePayloadSubscriber& operator=(const InlinePayloadSubscriber&) = delete;
  InlinePayloadSubscriber(InlinePayloadSubscriber&&) = delete;
  InlinePayloadSubscriber& operator=(InlinePayloadSubscriber&&) = delete;
  HOLOSCAN_CORE_EXPORT virtual ~InlinePayloadSubscriber();

  /// @brief Return whether this subscriber's service currently reports at least one publisher.
  /// @return True when a publisher is connected; false otherwise.
  /// @note This cold-path advisory observation is not receive readiness and does not prove that a
  /// sample will become available or that a publication will be delivered.
  [[nodiscard]] virtual bool has_publishers() const noexcept = 0;
  /** @brief Receive one payload.
   * @param payload Destination bytes.
   * @return Metadata and the exact written length, or ErrorCode::kNotReady/receive failure.
   * @note Implementations must report the bytes they actually wrote, never a length taken from
   * \ref SampleMetadata::byte_size. A peer may claim one width and submit another; only the
   * reported length proves which destination bytes belong to this sample. */
  [[nodiscard]] virtual expected<InlinePayloadReceiveResult, Error> receive(
      std::span<std::byte> payload) = 0;
  /// @brief Borrow one transport payload through verification and typed materialization.
  /// @return Guard exposing the exact used payload prefix and its metadata, or a receive error.
  [[nodiscard]] virtual expected<InlinePayloadReceiveGuard, Error> receive_borrowed();
  /// @brief Stage one bounded transport operation and borrow data only when Runtime can retain it.
  /// @param accept_data Whether a data operation may be returned to the caller.
  /// @return A borrowed data guard, kBackpressured with data retained, kClosed for terminal
  /// control, kNotReady for an empty transport, or a checked transport error.
  /// @note This source-private admission hook lets Runtime observe terminal control without first
  /// reserving typed payload storage. Implementations that do not support bounded staging retain
  /// the conservative default behavior.
  [[nodiscard]] virtual expected<InlinePayloadReceiveGuard, Error>
  receive_borrowed_if_data_accepted(bool accept_data);

  /// @brief Release the active borrowed payload loan, if any.
  /// @note This function does not throw.
  virtual void release_payload_loan() noexcept;

  /// @brief Detach this subscriber from its current opaque backend wait set.
  /// @return Success after the attachment is retired, including when already detached.
  /// @note The readiness pump must be joined before this cold lifecycle operation. Detaching
  /// fences wait-set observation only; callers must separately retire the endpoint before
  /// publishing a replacement attachment generation.
  [[nodiscard]] virtual expected<void, Error> detach_wait_set() noexcept = 0;
};

/// @brief Carry compiler-bounded storage facts for one private Tensor-schema staging pool.
struct TensorSchemaStagingOptions {
  /// Maximum logical byte span admitted for this field.
  std::uint64_t max_byte_span{};
  /// Number of finite staging slots, or zero when staging is forbidden.
  std::uint32_t slot_count{};
  /// Exact element type when constrained.
  std::optional<DLDataType> dtype;
  /// Exact logical rank when constrained.
  std::optional<std::uint32_t> rank;
  /// Per-axis extent and absolute byte-stride limits.
  std::vector<TensorAxisBounds> axes;
  /// Admitted logical stride family.
  TensorStridePolicy strides{TensorStridePolicy::kContiguousOnly};
  /// Exact physical storage class.
  MemoryKind memory_kind{MemoryKind::kUnknown};
  /// Exact CUDA device for device storage.
  DeviceId device{};
  /// Whether foreign backing must already be exportable or may be copied.
  InterfaceTensorEmitPolicy foreign_emit_policy{InterfaceTensorEmitPolicy::kRequireExportable};
};

/// @brief Retain one source-private schema-Tensor staging claim.
///
/// This move-only value invokes its no-throw release operation exactly once. The opaque owner
/// keeps cold staging state alive until the transport releases its submitted sample.
class TensorSchemaPayloadRetainedState {
 public:
  TensorSchemaPayloadRetainedState() = default;
  TensorSchemaPayloadRetainedState(std::shared_ptr<void> owner, void* state,
                                   void (*release)(void*) noexcept) noexcept
      : owner_(std::move(owner)), state_(state), release_(release) {}
  TensorSchemaPayloadRetainedState(const TensorSchemaPayloadRetainedState&) = delete;
  TensorSchemaPayloadRetainedState& operator=(const TensorSchemaPayloadRetainedState&) = delete;
  TensorSchemaPayloadRetainedState(TensorSchemaPayloadRetainedState&& other) noexcept
      : owner_(std::move(other.owner_)),
        state_(std::exchange(other.state_, nullptr)),
        release_(std::exchange(other.release_, nullptr)) {}
  TensorSchemaPayloadRetainedState& operator=(TensorSchemaPayloadRetainedState&& other) noexcept {
    if (this != &other) {
      reset();
      owner_ = std::move(other.owner_);
      state_ = std::exchange(other.state_, nullptr);
      release_ = std::exchange(other.release_, nullptr);
    }
    return *this;
  }
  ~TensorSchemaPayloadRetainedState() noexcept { reset(); }

  /// @brief Return whether this value owns one staging claim.
  /// @return True while a release operation remains active.
  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }

  /// @brief Release the retained staging claim, if any.
  void reset() noexcept {
    if (state_ != nullptr && release_ != nullptr) {
      release_(state_);
    }
    state_ = nullptr;
    release_ = nullptr;
    owner_.reset();
  }

 private:
  std::shared_ptr<void> owner_;
  void* state_{};
  void (*release_)(void*) noexcept {};
};

inline constexpr std::size_t kMaximumDescriptorTensorFields = 64U;

/** @brief Pair one schema payload descriptor with ownership retained through publication.
 * @details readiness_owner keeps an endpoint-owned readiness resource alive until the transport
 * sample retires. retained_tensor is populated for direct-export storage. retained_state is
 * populated when a bounded field-specific staging pool owns the exported backing. */
struct TensorSchemaPayloadFieldReservation {
  /// Payload reference encoded in the schema transport sample.
  PayloadDescriptor descriptor{};
  /// Optional readiness-resource owner; declared before Tensor so it is destroyed after Tensor.
  std::shared_ptr<void> readiness_owner{};
  /// Optional Tensor owner retained until the transport sample is released.
  std::optional<Tensor> retained_tensor{};
  /// Optional source-private staging claim retained until transport release.
  TensorSchemaPayloadRetainedState retained_state{};
};

/** @brief Retain every field reservation for one descriptor-backed schema publication. */
struct TensorSchemaPayloadReservation {
  /// Number of live field reservations in canonical Tensor-field order.
  std::uint32_t field_count{};
  /// Inline field-zero reservation preserving the singular endpoint's allocation behavior.
  TensorSchemaPayloadFieldReservation first_field{};
  /// Borrowed additional fields from endpoint-owned, cold-preallocated storage.
  TensorSchemaPayloadFieldReservation* additional_fields{};
  /// Claim returning additional fields to their bounded endpoint pool on retirement.
  TensorSchemaPayloadRetainedState field_storage{};

  /// @brief Return one mutable field reservation in canonical Tensor-field order.
  /// @param index Zero-based canonical Tensor-field index.
  /// @return Corresponding field reservation, or nullptr when \p index is out of range.
  [[nodiscard]] TensorSchemaPayloadFieldReservation* field_at(const std::size_t index) noexcept {
    if (index >= field_count) {
      return nullptr;
    }
    return index == 0U ? &first_field
                       : (additional_fields != nullptr ? &additional_fields[index - 1U] : nullptr);
  }

  /// @brief Return one field reservation in canonical Tensor-field order.
  /// @param index Zero-based canonical Tensor-field index.
  /// @return Corresponding field reservation, or nullptr when \p index is out of range.
  [[nodiscard]] const TensorSchemaPayloadFieldReservation* field_at(
      const std::size_t index) const noexcept {
    if (index >= field_count) {
      return nullptr;
    }
    return index == 0U ? &first_field
                       : (additional_fields != nullptr ? &additional_fields[index - 1U] : nullptr);
  }
};

/// @brief Describe one allocation-free descriptor-schema encoder for a transport payload loan.
struct TensorSchemaPayloadEncoder {
  /// Opaque caller-owned encoder state retained for the duration of the call.
  void* state{};
  /// Encode metadata and the reservation's canonical descriptor inventory into the destination.
  expected<std::size_t, Error> (*encode)(void*, std::span<std::byte>,
                                         const TensorSchemaPayloadReservation&) noexcept {};
  /// Maximum destination size admitted by the descriptor-schema codec contract.
  std::size_t capacity{};
};

/** @brief Transport interface for schema payloads and Tensor ownership. */
class HOLOSCAN_CORE_EXPORT TensorSchemaPayloadPublisher {
 public:
  TensorSchemaPayloadPublisher() = default;
  TensorSchemaPayloadPublisher(const TensorSchemaPayloadPublisher&) = delete;
  TensorSchemaPayloadPublisher& operator=(const TensorSchemaPayloadPublisher&) = delete;
  TensorSchemaPayloadPublisher(TensorSchemaPayloadPublisher&&) = delete;
  TensorSchemaPayloadPublisher& operator=(TensorSchemaPayloadPublisher&&) = delete;
  HOLOSCAN_CORE_EXPORT virtual ~TensorSchemaPayloadPublisher();

  /** @brief Reserve transport state and ownership for canonically indexed Tensor fields.
   * @param tensors Tensors retained through the configured local or interprocess transport path.
   * @return Aggregate reservation on success, or Error after rolling back every partial claim. */
  [[nodiscard]] expected<TensorSchemaPayloadReservation, Error> reserve_tensors(
      std::span<const Tensor> tensors) {
    return reserve_tensors(tensors, false);
  }
  /** @brief Reserve transport state and ownership for canonically indexed Tensor fields.
   * @param tensors Tensors retained through the configured local or interprocess transport path.
   * @param require_direct_export Reject interprocess Tensors that cannot be exported directly.
   * @return Aggregate reservation on success, or Error after rolling back every partial claim. */
  [[nodiscard]] virtual expected<TensorSchemaPayloadReservation, Error> reserve_tensors(
      std::span<const Tensor> tensors, bool require_direct_export) = 0;
  /** @brief Publish schema bytes with their reserved Tensor payload reference.
   * @param payload Serialized schema payload bytes.
   * @param reservation Descriptor and optional Tensor ownership consumed by the publication.
   * @param transport_metadata Catalog operation metadata, or an empty record for a manual channel.
   * @return Recipient count on success, or Error for invalid size, loan, or send failure. */
  // Defaults are part of this installed convenience API and are intentionally visible through
  // the abstract endpoint type used by callers.
  // NOLINTNEXTLINE(google-default-arguments)
  [[nodiscard]] virtual expected<std::uint32_t, Error> publish_payload(
      std::span<const std::byte> payload, TensorSchemaPayloadReservation reservation,
      SampleMetadata transport_metadata = {}) = 0;
  /** @brief Encode directly into an outbound transport loan and publish its used prefix.
   * @param encoder Allocation-free encoder callback and bounded codec capacity.
   * @param reservation Descriptor and Tensor ownership consumed by the publication.
   * @param transport_metadata Catalog operation metadata, or an empty record for a manual channel.
   * @return Recipient count on success, or a checked pre-submit or publication error. */
  // NOLINTNEXTLINE(google-default-arguments)
  [[nodiscard]] virtual expected<std::uint32_t, Error> publish_encoded_payload(
      const TensorSchemaPayloadEncoder& encoder, TensorSchemaPayloadReservation reservation,
      SampleMetadata transport_metadata = {});
  /** @brief Validate caller-authored semantic metadata for this endpoint.
   * @param metadata Metadata before sequence, byte size, and payload kind normalization.
   * @return Success when the metadata belongs to this endpoint's capture-clock contract. */
  [[nodiscard]] virtual expected<void, Error> validate_sample_metadata(
      const SampleMetadata& metadata) const noexcept;
  /** @brief Validate one decoded descriptor field against this endpoint's immutable contract.
   * @param field_index Canonical Tensor-field index.
   * @param metadata Schema-derived Tensor view metadata.
   * @param descriptor Transport descriptor paired with the schema field.
   * @warning The default implementation accepts every field. Third-party catalog transports must
   * override this function to enforce immutable descriptor-field contracts.
   * @return Success for an admitted field, or a checked pre-submit contract error. */
  [[nodiscard]] virtual expected<void, Error> validate_field_descriptor(
      std::uint32_t field_index, const TensorViewMetadata& metadata,
      const PayloadDescriptor& descriptor) const noexcept;
  /** @brief Label one capture coordinate with this endpoint's catalog clock.
   * @param timestamp_ns Exact caller-supplied nanosecond coordinate.
   * @return Bound time point, or \ref ErrorCode::kInvalidArgument when unavailable. */
  [[nodiscard]] virtual expected<TimePoint, Error> make_capture_time(
      std::int64_t timestamp_ns) const noexcept;
  /// @brief Publish one ordered catalog clock-health operation.
  /// @param report Valid payload-independent health assessment.
  /// @return Success after bounded acceptance, or a checked unsupported/publication error.
  [[nodiscard]] virtual expected<void, Error> report_clock_health(ClockHealthReport report);
  /// @brief Publish an ordered end-of-stream operation for this catalog generation.
  /// @param metadata Sequence and semantic payload identity for the terminal operation.
  /// @return Submit phase, recipient count, and checked error evidence for authority accounting.
  [[nodiscard]] virtual InlinePublicationResult close_stream(SampleMetadata metadata);
  /// @brief Return this catalog publisher's producer-incarnation identity.
  /// @return Nonzero identity for catalog publishers; zero for manual compatibility publishers.
  [[nodiscard]] virtual std::uint64_t producer_generation() const noexcept;
  /// @brief Return the next ordered catalog operation sequence without consuming it.
  /// @return Next nonzero sequence, or a checked unsupported/closed/exhaustion error.
  [[nodiscard]] virtual expected<std::uint64_t, Error> next_operation_sequence() const noexcept;
  /** @brief Release an unpublished Tensor reservation.
   * @param reservation Reservation whose local sidecar claim or retained owner is released.
   * @note This function does not throw. */
  virtual void release_tensor(TensorSchemaPayloadReservation reservation) noexcept = 0;
  /** @brief Refresh backend connection state.
   * @return Empty success or Error when the backend refresh fails. */
  [[nodiscard]] virtual expected<void, Error> update_connections() noexcept = 0;
  /** @brief Return whether at least one subscriber is connected.
   * @return True when a subscriber is connected; false otherwise. */
  [[nodiscard]] virtual bool has_subscribers() const noexcept = 0;
  /** @brief Return graph-derived Tensor allocation options.
   * @return MemoryBuffer options on success, or Error when unavailable. */
  [[nodiscard]] virtual expected<MemoryBuffer::Options, Error> tensor_memory_options() const = 0;
};

class TensorSchemaPayloadSubscriber;

/** @brief Keep a received schema transport sample borrowed through Tensor materialization.
 * @details This move-only guard releases the subscriber's active payload loan on destruction. The
 * referenced subscriber must outlive the guard. */
class HOLOSCAN_CORE_EXPORT TensorSchemaPayloadReceiveGuard {
 public:
  /** @brief Construct a guard for an active subscriber payload loan.
   * @param subscriber Subscriber whose active loan will be released by this guard. */
  HOLOSCAN_CORE_EXPORT explicit TensorSchemaPayloadReceiveGuard(
      TensorSchemaPayloadSubscriber& subscriber, std::span<const std::byte> payload = {}) noexcept;
  TensorSchemaPayloadReceiveGuard(const TensorSchemaPayloadReceiveGuard&) = delete;
  TensorSchemaPayloadReceiveGuard& operator=(const TensorSchemaPayloadReceiveGuard&) = delete;
  /** @brief Move ownership of an active payload loan.
   * @param other Guard whose loan ownership is transferred. */
  HOLOSCAN_CORE_EXPORT TensorSchemaPayloadReceiveGuard(
      TensorSchemaPayloadReceiveGuard&& other) noexcept;
  /** @brief Replace this guard's payload loan with another guard's loan.
   * @param other Guard whose loan ownership is transferred.
   * @return Reference to this guard after releasing its previous loan. */
  HOLOSCAN_CORE_EXPORT TensorSchemaPayloadReceiveGuard& operator=(
      TensorSchemaPayloadReceiveGuard&& other) noexcept;
  /** @brief Release the active subscriber payload loan, if any. */
  HOLOSCAN_CORE_EXPORT ~TensorSchemaPayloadReceiveGuard() noexcept;

  /// @brief Return the exact used descriptor-envelope prefix while this guard remains live.
  /// @return Borrowed transport bytes, or an empty span after a move.
  [[nodiscard]] std::span<const std::byte> payload() const noexcept { return payload_; }

 private:
  TensorSchemaPayloadSubscriber* subscriber_{};
  std::span<const std::byte> payload_{};
};

/** @brief Transport interface for receiving schema payloads and resolving Tensor storage. */
class HOLOSCAN_CORE_EXPORT TensorSchemaPayloadSubscriber : public WaitSetSubscriber {
 public:
  TensorSchemaPayloadSubscriber() = default;
  TensorSchemaPayloadSubscriber(const TensorSchemaPayloadSubscriber&) = delete;
  TensorSchemaPayloadSubscriber& operator=(const TensorSchemaPayloadSubscriber&) = delete;
  TensorSchemaPayloadSubscriber(TensorSchemaPayloadSubscriber&&) = delete;
  TensorSchemaPayloadSubscriber& operator=(TensorSchemaPayloadSubscriber&&) = delete;
  HOLOSCAN_CORE_EXPORT virtual ~TensorSchemaPayloadSubscriber();

  /// @brief Return whether this subscriber currently observes a publisher.
  /// @return True when a publisher is connected; false otherwise.
  [[nodiscard]] virtual bool has_publishers() const noexcept = 0;
  /** @brief Receive schema bytes while retaining the underlying transport loan.
   * @param payload Destination for the fixed-size schema payload bytes.
   * @return Receive guard on success, or Error for invalid size, an active loan, or receive
   * failure. */
  [[nodiscard]] virtual expected<TensorSchemaPayloadReceiveGuard, Error> receive_payload(
      std::span<std::byte> payload) = 0;
  /** @brief Borrow one exact descriptor-envelope prefix from the active transport sample.
   * @return Receive guard exposing borrowed bytes, or a checked receive error. */
  [[nodiscard]] virtual expected<TensorSchemaPayloadReceiveGuard, Error> borrow_payload();
  /** @brief Stage one bounded schema operation and borrow data only when Runtime can retain it.
   * @param accept_data Whether a data operation may be returned to the caller.
   * @return A borrowed data guard, kBackpressured with data retained, kClosed for terminal
   * control, kNotReady for an empty transport, or a checked transport error.
   * @note This source-private admission hook keeps terminal control independent of typed payload
   * capacity. Implementations without bounded staging retain the conservative default behavior. */
  [[nodiscard]] virtual expected<TensorSchemaPayloadReceiveGuard, Error>
  borrow_payload_if_data_accepted(bool accept_data);
  /** @brief Return the accepted ordinary-channel terminal sequence.
   * @return The terminal sequence after a validated manual EOS, otherwise no value.
   * @note Catalog-controlled imports keep EOS metadata in the Runtime-private receive path. */
  [[nodiscard]] virtual std::optional<std::uint64_t> end_of_stream_sequence() const noexcept;
  /** @brief Resolve a same-process payload descriptor through the Tensor sidecar.
   * @param descriptor Local payload reference to resolve.
   * @return Tensor on success, or Error when the descriptor is stale or otherwise invalid. */
  [[nodiscard]] virtual expected<Tensor, Error> acquire_tensor(
      const PayloadDescriptor& descriptor) = 0;
  /** @brief Import an opaque external-buffer reference and materialize its Tensor view.
   * @param descriptor Opaque external-buffer payload reference to import.
   * @param metadata Shape, dtype, stride, device, and byte-size metadata for the Tensor view.
   * @param field_index Canonical Tensor-field index selecting its bounded receive pool.
   * @return Read-only imported Tensor on success, or Error for invalid metadata, import failure, or
   * incompatible storage. */
  [[nodiscard]] virtual expected<Tensor, Error> import_buffer_handle_tensor(
      const PayloadDescriptor& descriptor, const TensorViewMetadata& metadata,
      std::uint32_t field_index) = 0;
  /** @brief Validate one decoded field before resolving or importing its backing.
   * @param field_index Canonical Tensor-field index.
   * @param metadata Schema-derived Tensor view metadata.
   * @param descriptor Transport descriptor paired with the schema field.
   * @warning The default implementation accepts every field. Third-party catalog transports must
   * override this function to enforce immutable descriptor-field contracts.
   * @return Success for an admitted field, or a checked pre-import contract error. */
  [[nodiscard]] virtual expected<void, Error> validate_field_descriptor(
      std::uint32_t field_index, const TensorViewMetadata& metadata,
      const PayloadDescriptor& descriptor) const noexcept;
  /** @brief Release a same-process recipient claim without materializing its Tensor.
   * @param descriptor Local payload reference whose claim is released.
   * @note This function does not throw. */
  virtual void release_recipient_claim(const PayloadDescriptor& descriptor) noexcept = 0;
  /** @brief Release the active schema transport loan, if any.
   * @note This function does not throw. */
  virtual void release_payload_loan() noexcept = 0;
};

}  // namespace channel_internal
}  // namespace holoscan
