/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/readiness_handle.hpp>
#include <holoscan/channel/tensor_loan.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/time_point.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan::detail {
class RuntimeChannelBindingAccess;
}  // namespace holoscan::detail

namespace holoscan {

namespace channel_internal {
struct TensorPublisherOps;
class TensorPublishLoanAccess;
class WaitSetSubscriber;
}  // namespace channel_internal

/// @brief Publish Tensors through one catalog-attached, endpoint-owned allocation domain.
///
/// The attached import endpoint owns a finite preallocated publication pool and supplies the
/// logical Tensor contract, capture-clock authority, and optional clock-health lane. The facade
/// never exposes allocator injection, a bare-Tensor emit, publisher-created
/// `Sample<Tensor>`, HoloMQ/BufferHandle state, or a process-local physical resource handle.
/// The caller must serialize every publication call on one live facade, including `emit*`,
/// `close()`, and `report_clock_health()`. Transport release callbacks are synchronized internally.
/// Filling distinct acquired loans may overlap only within bounds later admitted by the attached
/// endpoint.
template <>
class ChannelPublisher<Tensor> {
 public:
  /// @brief Transfer endpoint publication authority from another facade.
  /// @param other Publisher whose authority becomes moved from.
  /// @post `other` is closed and its checked operations return \ref ErrorCode::kClosed.
  HOLOSCAN_CORE_EXPORT ChannelPublisher(ChannelPublisher&& other) noexcept;

  /// @brief Reject move assignment so replacing a live endpoint cannot hide teardown.
  /// @param other Publisher that would otherwise replace this one.
  ChannelPublisher& operator=(ChannelPublisher&& other) = delete;

  /// @brief Reject copying endpoint and allocation authority.
  /// @param other Publisher that would otherwise be copied.
  ChannelPublisher(const ChannelPublisher& other) = delete;

  /// @brief Reject copy assignment of endpoint and allocation authority.
  /// @param other Publisher that would otherwise replace this one.
  ChannelPublisher& operator=(const ChannelPublisher& other) = delete;

  /// @brief Destroy the facade and release its source-private endpoint ownership.
  HOLOSCAN_CORE_EXPORT ~ChannelPublisher() noexcept;

  /// @brief Acquire one endpoint-owned external Tensor publication loan without blocking.
  /// @param request Logical shape, dtype, and optional byte strides copied during this call.
  /// @return Sole unpublished loan, \ref ErrorCode::kResourceExhausted when a finite admitted
  /// resource is unavailable, or another checked endpoint/layout error.
  /// @note Request shape and stride views are copied before this call returns. Pool exhaustion is
  /// reported immediately and never falls back to allocation or blocking.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorPublishLoan, Error> loan_tensor(
      TensorLoanRequest request);

  /// @brief Consume and publish one ready endpoint-owned Tensor loan.
  /// @param loan Sole unpublished authority consumed on success or failure.
  /// @param options Optional capture, frame, trace, and admitted flag metadata.
  /// @return Exact reached-recipient count, or a checked publication error.
  /// @note A post-submit outcome that cannot prove zero delivery returns
  /// \ref ErrorCode::kPublicationIndeterminate or \ref ErrorCode::kPublicationPartial and must not
  /// be retried as the same sample.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit(TensorPublishLoan loan,
                                                                         EmitOptions options = {});

  /// @brief Consume and publish one loan with a coordinate in the bound capture clock.
  /// @param loan Sole unpublished authority consumed on success or failure.
  /// @param capture_timestamp_ns Exact caller-supplied capture coordinate in nanoseconds.
  /// @param options Remaining metadata; an existing capture time is invalid.
  /// @return Exact reached-recipient count, or a checked capture/publication error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit_captured(
      TensorPublishLoan loan, std::int64_t capture_timestamp_ns, EmitOptions options = {});

  /// @brief Consume and publish one loan with explicit public sample metadata.
  /// @param loan Sole unpublished authority consumed on success or failure.
  /// @param metadata Caller semantic metadata. Runtime replaces sequence, publication time/clock,
  /// byte size, and payload kind with locally authoritative values.
  /// @return Exact reached-recipient count, or a checked metadata/publication error.
  /// @note This overload deliberately does not accept `Sample<Tensor>` because a copyable Tensor
  /// cannot represent sole unpublished loan authority.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit_sample(
      TensorPublishLoan loan, SampleMetadata metadata);

  /// @brief Copy an admitted foreign Tensor into endpoint-owned catalog storage and publish it.
  /// @param source Contiguous CPU, CUDA device, or CUDA managed source. Device source backing is
  /// retained or release-ordered internally until the asynchronous copy completes.
  /// @param producer_stream Borrowed producer stream. CPU sources require a null stream; CUDA
  /// device and managed sources require the non-null stream that orders producer completion.
  /// @param options Optional capture, frame, trace, and admitted flag metadata.
  /// @return Exact reached-recipient count, or a checked source/copy/publication error.
  /// @note This is the sole catalog-backed foreign-storage path. It never changes the source into
  /// exportable storage in place. A host destination performs one producer-stream-ordered
  /// device-to-host materialization and synchronizes before submission. A same-device CUDA
  /// destination enqueues one device-to-device copy and generation fence without blocking the
  /// host. Direct zero-copy export requires separately admitted physical resources.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit_copy(
      const Tensor& source, ::cuda::stream_ref producer_stream, EmitOptions options = {});

  /// @brief Label one coordinate with this catalog endpoint's capture clock.
  /// @param timestamp_ns Exact caller-supplied nanosecond coordinate.
  /// @return Bound time point, or a checked closed/authority error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TimePoint, Error> make_capture_time(
      std::int64_t timestamp_ns) const noexcept;

  /// @brief Publish one endpoint-bound clock-health report without borrowing a Tensor slot.
  /// @param report Valid payload-independent assessment and portable reason.
  /// @return Empty success after local bounded acceptance, or a checked contract/publication error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> report_clock_health(
      ClockHealthReport report);

  /// @brief Publish an explicit ordered end-of-stream operation and consume this generation.
  /// @return Success after every frozen recipient accepts EOS.
  /// @retval ErrorCode::kNotReady EOS was rejected before submission; this publisher remains
  /// usable.
  /// @retval ErrorCode::kClosed This facade is moved from or already closed.
  /// @retval ErrorCode::kPublicationIndeterminate Submission began but its outcome is unknown;
  /// this facade becomes closed and EOS must not be retried.
  /// @retval ErrorCode::kPublicationPartial Submission reached a known strict subset; this facade
  /// becomes closed and EOS must not be retried.
  /// @retval ErrorCode::kProtocolError The backend reported a malformed recipient outcome; this
  /// facade becomes closed and EOS must not be retried.
  /// @retval ErrorCode::kInvalidArgument The backend reported an impossible recipient count; this
  /// facade becomes closed and EOS must not be retried.
  /// @note Destruction, disconnect, silence, and process exit do not publish or imply EOS.
  /// @note With a lossy replacement policy, BufferHandle EOS uses a bounded protected terminal
  /// lane and a recipient drains its Tensor data lane before observing EOS. New data may replace
  /// stale data, but EOS cannot replace the final unread Tensor accepted by that recipient.
  /// QueuePolicy::kBlocking keeps data and EOS on its existing lossless FIFO lane.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> close();

  /// @brief Refresh the backend connection snapshot used by \ref has_subscribers.
  /// @return Empty success, or a checked closed/backend refresh error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> update_connections() noexcept;

  /// @brief Return the most recently refreshed subscriber-presence snapshot.
  /// @return True when a subscriber was observed; false when moved from, closed, or disconnected.
  /// @note This advisory value is not proof that a later emit reaches any recipient.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool has_subscribers() noexcept;

 private:
  ChannelPublisher(void* state, const channel_internal::TensorPublisherOps* ops) noexcept
      : state_(state), ops_(ops) {}

  void* state_{};
  const channel_internal::TensorPublisherOps* ops_{};

  friend class channel_internal::TensorPublishLoanAccess;
};

/// @brief Consume read-only Tensors from one manual or catalog-attached Channel endpoint.
///
/// Catalog attachment imports the endpoint's opaque BufferHandle transport into a bounded
/// Holoscan-owned read-only Tensor lifetime. The facade exposes no HoloMQ handle, pool slot, or
/// endpoint-generation control. Each successful receive consumes one queued sample; copied Tensor
/// wrappers retain the same read-only storage claim until the last wrapper or derived view retires.
template <>
class ChannelSubscriber<Tensor> {
 public:
  HOLOSCAN_CORE_EXPORT ChannelSubscriber(ChannelSubscriber&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ChannelSubscriber& operator=(ChannelSubscriber&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~ChannelSubscriber();

  ChannelSubscriber(const ChannelSubscriber&) = delete;
  ChannelSubscriber& operator=(const ChannelSubscriber&) = delete;

  /// @brief Retire this process's local Tensor subscriber endpoint.
  /// @return Success on the first live close, or a code-only cleanup error. Repeated close and
  /// close on a moved-from facade return @ref ErrorCode::kClosed.
  /// @note This externally serialized, consuming operation publishes no producer end-of-stream
  /// record and supplies no graph-completion evidence. Cancel and join any active waiter before
  /// closing. Already returned Tensors and derived views retain their ordinary owning lifetime;
  /// unread local data may be discarded. A cleanup failure still consumes the endpoint; later
  /// checked operations return @ref ErrorCode::kClosed.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> close() noexcept;

  /// Return whether the transport reports a pending Tensor descriptor without consuming it.
  ///
  /// @return The readiness state, or an error when the backend cannot query it.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<bool, Error> has_samples() const;

  /// @brief Claim an externally pollable readiness descriptor for this subscriber.
  /// @return A move-only handle, or a checked closed, capability, or ownership error.
  /// @retval ErrorCode::kClosed This facade is locally closed, moved from, or already observed
  /// producer end of stream.
  /// @retval ErrorCode::kNotSupported Native descriptor readiness is unavailable on this platform,
  /// or this endpoint was created without event wake.
  /// @retval ErrorCode::kInvalidArgument A wait is in flight, the Runtime readiness pump owns the
  /// subscriber, or another readiness handle is live.
  /// @note Readability is a hint. A consumer calls \ref ChannelReadinessHandle::acknowledge first,
  /// then drains \ref receive or \ref receive_sample until \ref ErrorCode::kNotReady. Producer end
  /// of stream and a changed producer incarnation also wake the descriptor so the next receive
  /// reports \ref ErrorCode::kClosed or \ref ErrorCode::kStaleEndpoint.
  /// @note Blocking wait operations return \ref ErrorCode::kInvalidArgument while the handle is
  /// live.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelReadinessHandle, Error> readiness();

  /// Receive the next Tensor handle.
  ///
  /// @return The next Tensor, or \ref ErrorCode::kNotReady when no descriptor is queued.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> receive();

  /// Receive the next Tensor handle together with public sample metadata.
  ///
  /// Sidecar token, generation, process id, and other transport internals are intentionally not
  /// exposed through this API.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Sample<Tensor>, Error> receive_sample();

  /// @brief Block until a Tensor sample is available, then receive and consume it.
  ///
  /// A receive is attempted before any wait is built, so an already queued sample costs no wait
  /// context and no kernel wait. \ref ErrorCode::kNotReady never escapes: readiness is a hint, so
  /// an attempt that loses its race against a dropped-oldest replacement waits again against the
  /// same absolute \p deadline. Queued data always drains before end of stream is observed, so a
  /// producer that emits N Tensors and then closes yields N samples and then
  /// \ref ErrorCode::kClosed.
  ///
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @return The sample on success, or the first checked error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed. Retryable; endpoint state is unchanged.
  /// @retval ErrorCode::kClosed The producer published end of stream, this facade was locally
  /// closed, or this facade was moved from. The end-of-stream state is latched, so every later
  /// call reports it.
  /// @retval ErrorCode::kStaleEndpoint The producer incarnation changed; rebuild the subscriber.
  /// @retval ErrorCode::kNotSupported This endpoint was created with
  /// \c ChannelOptions::enable_event_wake disabled, so it has no kernel event lane. Reported only
  /// when a wait is actually required, and before any wait context is built. This operation
  /// deliberately does not fall back to polling.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, such as the Runtime readiness pump, or another wait on this subscriber is already
  /// in flight.
  /// @note \ref ErrorCode::kProtocolError, \ref ErrorCode::kCodecError,
  /// \ref ErrorCode::kReceiveError, and \ref ErrorCode::kResourceExhausted pass through unchanged.
  /// @note There is no duration overload by design. An absolute deadline cannot silently restart
  /// its budget on a spurious wake.
  /// @note Readiness is a hint, so a subscriber that keeps reporting it while every receive loses
  /// its race retries at full speed until \p deadline. The absolute deadline is the only bound on
  /// that loop; there is no backoff. It is not reachable by an idle endpoint -- something has to be
  /// producing readiness that no receive can consume, such as a competing consumer on the same
  /// endpoint -- but a caller that shares a subscriber should size the deadline accordingly.
  /// @warning One subscriber supports one waiting thread and calls are not internally serialized.
  /// A second concurrent waiter is rejected with \ref ErrorCode::kInvalidArgument rather than
  /// sharing the first one's wait context. The facade must outlive an in-flight wait and must not
  /// be moved while one is in flight.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Sample<Tensor>, Error> wait_and_receive_sample(
      std::chrono::steady_clock::time_point deadline);

  /// @brief Block until a Tensor sample is available or a stop is requested, then consume it.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param token Cancellation observer for this call; a default-constructed token never cancels.
  /// @return The sample on success, or the first checked error.
  /// @retval ErrorCode::kInterrupted \p token requested a stop. Caller-local and non-terminal.
  /// @note Every other retval of the single-argument overload applies unchanged.
  /// @note Cancellation is scoped to one call. A stop that fires ends this wait and leaves the
  /// subscriber usable; the next wait arms a fresh cancellation trigger.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Sample<Tensor>, Error> wait_and_receive_sample(
      std::chrono::steady_clock::time_point deadline, const std::stop_token& token);

  /// @brief Block until this endpoint may have a sample, without consuming one.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param token Cancellation observer for this call; a default-constructed token never cancels.
  /// @return Empty success when a sample may be available, or a checked wait error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed with no readiness.
  /// @retval ErrorCode::kInterrupted \p token requested a stop.
  /// @retval ErrorCode::kClosed End of stream was already consumed, or this facade was locally
  /// closed or moved from.
  /// @retval ErrorCode::kNotSupported This endpoint was created with
  /// \c ChannelOptions::enable_event_wake disabled and has no queued sample.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, or another wait on this subscriber is already in flight.
  /// @note Success is a hint, exactly like \ref has_samples. A following \ref receive_sample may
  /// still report \ref ErrorCode::kNotReady.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> wait_for_sample(
      std::chrono::steady_clock::time_point deadline, const std::stop_token& token = {});

  /// @brief Refresh the backend connection snapshot used by has_publishers().
  /// @return Empty success, or a checked closed/backend refresh error.
  /// @note This cold-path operation does not consume a sample or report receive readiness.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> refresh_connections();

  /// @brief Return whether this endpoint currently observes a publisher.
  /// @return True when a publisher is connected; false when moved from, closed, or disconnected.
  /// @note This cold-path advisory snapshot is neither receive readiness nor proof that a later
  /// sample arrives. Required-peer catalog attachment applies its finite wait before returning.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool has_publishers() const noexcept;

 private:
  friend class ChannelRuntime;
  friend class detail::RuntimeChannelBindingAccess;

  class Impl;

  HOLOSCAN_CORE_EXPORT explicit ChannelSubscriber(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT channel_internal::WaitSetSubscriber*
  wait_set_subscriber() noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Sample<Tensor>, Error>
  receive_sample_if_data_accepted(bool accept_data);
  [[nodiscard]] HOLOSCAN_CORE_EXPORT std::uint64_t accepted_end_of_stream_sequence() const noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan
