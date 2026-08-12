/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/tensor_loan.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/time/time_point.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
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
/// Calls on one live facade are externally serialized; filling distinct acquired loans may overlap
/// only within bounds later admitted by the attached endpoint.
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
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> update_connections();

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

  /// Return whether the transport reports a pending Tensor descriptor without consuming it.
  ///
  /// @return The readiness state, or an error when the backend cannot query it.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<bool, Error> has_samples() const;

  /// Receive the next Tensor handle.
  ///
  /// @return The next Tensor, or \ref ErrorCode::kNotReady when no descriptor is queued.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> receive();

  /// Receive the next Tensor handle together with public sample metadata.
  ///
  /// Sidecar token, generation, process id, and other transport internals are intentionally not
  /// exposed through this API.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Sample<Tensor>, Error> receive_sample();

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
