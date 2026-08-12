/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {
namespace channel_internal {

/// @brief Attach one private Channel subscriber to a shared backend wait set.
///
/// Implementations retain their own backend attachment, which keeps the shared wait backend and
/// subscriber transport port alive independently of the wait-set facade. The facade therefore may
/// be destroyed before attached subscriber wrappers; later attachment destruction still detaches
/// safely. This transport-private interface lets Runtime multiplex inline, Tensor, and
/// Tensor-schema subscribers without exposing backend endpoint or wait-set types.
class HOLOSCAN_CORE_EXPORT WaitSetSubscriber {
 public:
  WaitSetSubscriber() = default;
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
};

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

  /** @brief Query pending samples.
   * @return Readiness or Error from the transport. */
  [[nodiscard]] virtual expected<bool, Error> has_samples() const = 0;
  /// @brief Return whether this subscriber's service currently reports at least one publisher.
  /// @return True when a publisher is connected; false otherwise.
  /// @note This cold-path advisory observation is not receive readiness and does not prove that a
  /// sample will become available or that a publication will be delivered.
  [[nodiscard]] virtual bool has_publishers() const noexcept = 0;
  /** @brief Receive one payload.
   * @param payload Destination bytes.
   * @return SampleMetadata or ErrorCode::kNotReady/receive failure. */
  [[nodiscard]] virtual expected<SampleMetadata, Error> receive(std::span<std::byte> payload) = 0;
  /// @brief Borrow one transport payload through verification and typed materialization.
  /// @return Guard exposing the exact used payload prefix and its metadata, or a receive error.
  [[nodiscard]] virtual expected<InlinePayloadReceiveGuard, Error> receive_borrowed();

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
  std::uint64_t max_byte_span{};
  std::uint32_t slot_count{};
  std::optional<DLDataType> dtype;
  std::optional<std::uint32_t> rank;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
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
   * @return Recipient count on success, or Error for invalid size, loan, or send failure. */
  [[nodiscard]] virtual expected<std::uint32_t, Error> publish_payload(
      std::span<const std::byte> payload, TensorSchemaPayloadReservation reservation) = 0;
  /** @brief Encode directly into an outbound transport loan and publish its used prefix.
   * @param encoder Allocation-free encoder callback and bounded codec capacity.
   * @param reservation Descriptor and Tensor ownership consumed by the publication.
   * @return Recipient count on success, or a checked pre-submit or publication error. */
  [[nodiscard]] virtual expected<std::uint32_t, Error> publish_encoded_payload(
      const TensorSchemaPayloadEncoder& encoder, TensorSchemaPayloadReservation reservation);
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

  /** @brief Query pending schema payloads without consuming one.
   * @return Readiness or Error when the backend query fails. */
  [[nodiscard]] virtual expected<bool, Error> has_samples() const = 0;
  /** @brief Receive schema bytes while retaining the underlying transport loan.
   * @param payload Destination for the fixed-size schema payload bytes.
   * @return Receive guard on success, or Error for invalid size, an active loan, or receive
   * failure. */
  [[nodiscard]] virtual expected<TensorSchemaPayloadReceiveGuard, Error> receive_payload(
      std::span<std::byte> payload) = 0;
  /** @brief Borrow one exact descriptor-envelope prefix from the active transport sample.
   * @return Receive guard exposing borrowed bytes, or a checked receive error. */
  [[nodiscard]] virtual expected<TensorSchemaPayloadReceiveGuard, Error> borrow_payload();
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
