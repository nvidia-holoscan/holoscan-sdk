/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_catalog.hpp>

namespace holoscan {

namespace detail {
class ParamSnapshotArchiveAccess;
}  // namespace detail

/// @brief Identify one accepted parameter-update request within an active run.
///
/// Zero means no accepted update. A nonzero value is monotonic and unique only within the run that
/// issued it; it is not a process-global or cross-run identity.
using ParamUpdateSequence = std::uint64_t;

/// @brief Request publication at the next scheduler-safe compute boundary.
///
/// The current callback, if any, retains its pinned immutable generation. A future callback may
/// capture the accepted generation after publication. This marker does not request immediate
/// mutation, a Tag boundary, a clock time, or a lifecycle barrier.
struct NextCompute {
  /// @brief Compare stateless timing markers.
  /// @return Always `true`.
  friend constexpr bool operator==(NextCompute, NextCompute) noexcept = default;
};

/// @brief Select one coherent control-plane parameter view.
enum class ParamReadView : std::uint8_t {
  kPublished = 0,       ///< Read the generation available to a future compute capture now.
  kLatestAccepted = 1,  ///< Read newest accepted intent, or published state when none is pending.
  kPending = 2,  ///< Read only newest accepted intent not yet published; absence is `kNotReady`.
};

/// @brief Identify the lifecycle state of one parameter-update request.
enum class ParamUpdateState : std::uint8_t {
  kSubmitted = 0,         ///< Bounded ingress owns the request but has not accepted it.
  kRejected = 1,          ///< Validation or admission rejected the request without acceptance.
  kAccepted = 2,          ///< Required generation and evidence resources are reserved.
  kQuiescing = 3,         ///< A lifecycle-mutable transaction is retiring affected work.
  kRestarting = 4,        ///< A lifecycle-mutable transaction is executing lifecycle stages.
  kAcceptedNoChange = 5,  ///< Equivalent accepted intent terminated without publication.
  kPublished = 6,         ///< Publication processing completed; `await()` treats this as terminal.
  kObserved = 7,          ///< A later activation pinned the terminally published generation.
  kSuperseded = 8,        ///< A later accepted intent superseded this request.
  kCanceled = 9,          ///< An accepted request terminated without publication.
  kRolledBack = 10,       ///< Lifecycle publication failed and the prior generation was restored.
  kFailedStopped = 11,    ///< Recovery failed and the affected run stopped.
};

/// @brief Classify immediate parameter-update admission and completion outcomes.
enum class ParamUpdateStatus : std::uint8_t {
  kAccepted = 0,           ///< The update was accepted and reserved, but need not be published yet.
  kAcceptedNoChange = 1,   ///< Equivalent intent was accepted without publication.
  kSubmitted = 2,          ///< Bounded ingress owns the request pending acceptance.
  kRejected = 3,           ///< The request was rejected without a narrower category.
  kDeferred = 4,           ///< Completion awaits its admitted publication boundary.
  kUpdatesDisabled = 5,    ///< The plan or capability disables parameter updates.
  kOutOfBounds = 6,        ///< A declared range or finite bound rejected the value.
  kValidationFailed = 7,   ///< Field or aggregate validation rejected the candidate.
  kTypeMismatch = 8,       ///< The value kind or width mismatches the declared field.
  kUnknownParameter = 9,   ///< The address does not identify a declared field.
  kStaleOperatorRef = 10,  ///< The target does not belong to this active run.
  kExpectedGenerationMismatch = 11,  ///< Optimistic expected-generation validation failed.
  kRequiresLifecycleBarrier = 12,    ///< The field cannot publish through `NextCompute`.
  kRequiresRecompile = 13,           ///< The field is immutable in the active plan.
  kUnsupportedAtomicScope = 14,      ///< A requested atomic batch spans multiple operators.
  kPastEffectiveTime = 15,           ///< A future timing form named a boundary already passed.
  kUnsupportedClock = 16,            ///< A future timing form named an unsupported clock.
  kQueueFull =
      17,  ///< Finite ingress, generation, evidence, or transaction capacity is unavailable.
  kInvalidCallingContext = 18,  ///< The operation is forbidden from this runtime context.
  kUnsupportedInProfile = 19,   ///< The effective profile rejects this operation.
  kClosed = 20,                 ///< The run-scoped update ingress is irreversibly closed.
};

/// @brief Own one supported fixed scalar parameter value.
///
/// `ParamValue` deliberately supports only Boolean, fixed-width signed/unsigned integer, enum
/// underlying, IEC 60559 `float`/`double`, and integral-representation duration values in baseline.
/// Durations are checked-converted to signed 64-bit canonical nanoseconds; a source overflow or
/// non-integral nanosecond value produces an empty value that submission rejects. The carrier
/// contains no heap storage, string, container, object address, or C++ type token. Enum identity
/// is not retained beyond its underlying scalar kind and width.
class ParamValue {
 public:
  /// @brief Construct an empty value.
  /// @note An empty value is not a valid patch payload and is rejected as a type mismatch.
  constexpr ParamValue() noexcept = default;

  /// @brief Own one supported scalar value.
  /// @tparam T Boolean, fixed-width signed/unsigned integer, enum with a supported underlying
  /// type, IEC 60559 `float`/`double`, or an integral-representation duration.
  /// @param value Value copied into the fixed scalar carrier.
  /// @return Allocation-free owning parameter value.
  /// @note A duration that cannot be represented as an exact signed 64-bit nanosecond count returns
  /// an empty value; a later submission rejects that empty payload as a type mismatch.
  template <typename T>
    requires detail::kFixedParameterScalar<std::remove_cv_t<T>>
  [[nodiscard]] static constexpr ParamValue from(T value) noexcept {
    return ParamValue{ParamScalarValue::from(value)};
  }

  /// @brief Decode the value into an exactly compatible scalar family and width.
  /// @tparam T Supported target scalar or integral-representation duration type.
  /// @return Decoded value, or `ErrorCode::kIncompatible` when kind or width differs.
  /// @note Enum decoding checks the underlying scalar family and width; this carrier does not
  /// retain or prove the original C++ enum type identity.
  template <typename T>
    requires std::same_as<T, std::remove_cv_t<T>> && detail::kFixedParameterScalar<T>
  [[nodiscard]] expected<T, Error> as() const noexcept {
    T value{};
    if (!scalar_.try_as(value)) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    return value;
  }

  /// @brief Return the low-level stable scalar carrier.
  /// @return Kind, bit width, and raw value bits owned by this value.
  [[nodiscard]] constexpr ParamScalarValue scalar() const noexcept { return scalar_; }

  /// @brief Return the stable scalar family.
  /// @return `kNone` for an empty value, otherwise the encoded scalar kind.
  [[nodiscard]] constexpr ParamScalarKind kind() const noexcept { return scalar_.kind; }

  /// @brief Return the semantic scalar width.
  /// @return One for Boolean, 8/16/32/64 for integers, 32/64 for floating point, 64 for a
  /// canonical duration nanosecond count, or zero when empty.
  [[nodiscard]] constexpr std::uint8_t bit_width() const noexcept { return scalar_.bit_width; }

  /// @brief Return the exact right-aligned scalar bits.
  /// @return Owned raw bits interpreted only together with the kind and bit-width accessors.
  [[nodiscard]] constexpr std::uint64_t raw_bits() const noexcept { return scalar_.raw_bits; }

  /// @brief Compare complete scalar representations.
  /// @param lhs Left value.
  /// @param rhs Right value.
  /// @return `true` when kind, width, and raw bits are equal.
  friend constexpr bool operator==(ParamValue lhs, ParamValue rhs) noexcept = default;

 private:
  explicit constexpr ParamValue(ParamScalarValue scalar) noexcept : scalar_(scalar) {}

  ParamScalarValue scalar_{};
  friend class detail::ParamSnapshotArchiveAccess;
};

static_assert(std::is_standard_layout_v<ParamValue>);
static_assert(std::is_trivially_copyable_v<ParamValue>);

/// @brief Own one cold-path field update request.
struct ParamPatch {
  ParamAddress target{};  ///< Declared target field.
  ParamValue value{};     ///< Complete fixed scalar value.
  std::optional<ParamGeneration>
      expected_generation;  ///< Optional precondition against latest accepted generation.
};

/// @brief Carry one fixed-size compute-originated field update request.
///
/// The record is trivially copyable and contains no dynamic ownership. `has_expected_generation`
/// controls whether `expected_generation` is authoritative.
struct ParamInlinePatch {
  ParamAddress target{};                  ///< Declared target field.
  ParamValue value{};                     ///< Complete fixed scalar value.
  ParamGeneration expected_generation{};  ///< Expected latest-accepted generation, if enabled.
  bool has_expected_generation{};         ///< Whether the optimistic precondition is present.
};

static_assert(std::is_standard_layout_v<ParamInlinePatch>);
static_assert(std::is_trivially_copyable_v<ParamInlinePatch>);

/// @brief Select the admitted parameter-audit detail request.
enum class ParamAudit : std::uint8_t {
  kProfileDefault = 0,  ///< Use the immutable plan-resolved profile disposition.
  kMinimal = 1,         ///< Emit fixed-size bounded records without dynamic text.
  kDetailed = 2,        ///< Request detailed evidence only when the profile permits it.
};

/// @brief Identify one registered bounded parameter-update reason.
using ParamReasonId = std::uint64_t;

/// @brief Carry a fixed registered reason identity without retaining caller text.
class ParamUpdateReason {
 public:
  /// @brief Return the absence of a registered reason.
  /// @return Reason identity zero.
  [[nodiscard]] static constexpr ParamUpdateReason none() noexcept {
    return ParamUpdateReason{ParamReasonId{0}};
  }

  /// @brief Return one registered reason identity.
  /// @param reason_id Stable application or SDK reason identity.
  /// @return Fixed-size reason value; zero is identical to \ref none and Core never interprets the
  /// identity as authority.
  [[nodiscard]] static constexpr ParamUpdateReason interned(ParamReasonId reason_id) noexcept {
    return ParamUpdateReason{reason_id};
  }

  /// @brief Return the registered reason identity.
  /// @return Zero for no reason, otherwise the supplied stable identity.
  [[nodiscard]] constexpr ParamReasonId id() const noexcept { return reason_id_; }

  /// @brief Compare reason identities.
  /// @return `true` when both identities match.
  friend constexpr bool operator==(ParamUpdateReason, ParamUpdateReason) noexcept = default;

 private:
  explicit constexpr ParamUpdateReason(ParamReasonId reason_id) noexcept : reason_id_(reason_id) {}

  ParamReasonId reason_id_{};
};

/// @brief Configure one owning parameter-update submission.
struct ParamUpdateOptions {
  /// Omitted timing derives `NextCompute` or the required lifecycle transaction from field policy.
  std::optional<NextCompute> effective;
  /// Requested admitted audit detail.
  ParamAudit audit{ParamAudit::kProfileDefault};
  /// Audit identity only; never authorization.
  std::uint64_t actor_id{};
  /// Fixed registered audit reason; zero means no registered reason.
  ParamUpdateReason reason{ParamUpdateReason::none()};
};

/// @brief Return one coherent parameter field value and its update lineage.
struct ParamReadResult {
  ParamValue value{};                     ///< Value from exactly one immutable record.
  ParamGeneration generation{};           ///< Generation associated with the returned value.
  ParamUpdateSequence update_sequence{};  ///< Accepted update, or zero for generation zero.
  ParamUpdateState state{ParamUpdateState::kRejected};  ///< State of the returned record.
};

/// @brief Summarize one accepted or rejected update through its current terminal lineage.
///
/// `kPublished` is terminal for update processing and for `await()`. A later
/// activation may monotonically advance the retained state to `kObserved`;
/// waiting neither requires nor causes that observation.
struct ParamUpdateResult {
  ParamUpdateStatus status{ParamUpdateStatus::kRejected};  ///< Typed result category.
  ParamUpdateState state{ParamUpdateState::kRejected};     ///< Current request state.
  ParamUpdateSequence update_sequence{};       ///< Nonzero only after acceptance assigns identity.
  ParamGeneration target_generation{};         ///< Intended complete generation, or zero.
  std::uint64_t lifecycle_transaction_id{};    ///< Coalesced lifecycle transaction, or zero.
  std::uint64_t coalesced_into_transaction{};  ///< Related transaction identity, or zero.
};

/// @brief Summarize one same-operator all-or-nothing patch batch.
struct ParamBatchUpdateResult {
  /// Sentinel used when no individual patch index caused rejection.
  static constexpr std::uint32_t kNoFailedPatch = std::numeric_limits<std::uint32_t>::max();

  ParamUpdateStatus status{ParamUpdateStatus::kRejected};  ///< Aggregate admission outcome.
  bool committed{};  ///< Whether all fields entered one accepted candidate, not publication.
  std::uint32_t patch_count{};                 ///< Number of patches evaluated atomically.
  std::uint32_t failed_patch{kNoFailedPatch};  ///< First canonical failing index, or sentinel.
  ParamUpdateSequence update_sequence{};       ///< Shared accepted sequence, or zero.
  ParamGeneration target_generation{};         ///< Complete candidate generation, or zero.
};

/// @brief Report immediate compute-originated bounded-ingress disposition.
struct ParamSubmitResult {
  ParamUpdateStatus status{ParamUpdateStatus::kRejected};  ///< Immediate nonblocking outcome.
  ParamUpdateState state{ParamUpdateState::kRejected};     ///< Submitted or rejected state.
  ParamUpdateSequence update_sequence{};  ///< Always zero in baseline; acceptance assigns identity
                                          ///< after ingress drain.
};

/// @brief Report the irreversible session-wide parameter-ingress stop fence.
struct ParamStopFenceResult {
  ParamUpdateSequence terminal_update_sequence{};  ///< Final sequence accepted before closure.
  bool already_closed{};                           ///< Whether an earlier call fixed this fence.
};

/// @brief Identify one retained Parameter observation record within an active run.
///
/// Zero means no event. Nonzero values are monotonic only within the run that
/// issued them and allow an independently positioned cursor to detect bounded
/// retention loss without treating the event stream as a coherent snapshot.
using ParamEventSequence = std::uint64_t;

/// @brief Classify one Parameter cursor record.
enum class ParamEventKind : std::uint8_t {
  kTransition = 0,  ///< Report one bounded Parameter state transition.
  kGap = 1,         ///< Report one opaque cursor-local retention discontinuity.
};

/// @brief Carry one value-free Parameter transition or explicit retention gap.
///
/// Transition records contain only scoped identities, generations, typed state,
/// and bounded audit identities. They never contain Parameter values, object
/// bytes, addresses, hashes derived from sensitive values, or formatted text.
/// A gap is deliberately non-counting: policy filtering and bounded overwrite
/// may hide any positive number of source records. The baseline sets
/// \ref first_missing_sequence and \ref last_missing_sequence to the same
/// cursor-local marker sequence. Callers must resnapshot and must not infer a
/// source-record count or global source position from those fields. For a gap,
/// all transition-specific fields are non-authoritative.
struct ParamEvent {
  ParamEventKind kind{ParamEventKind::kTransition};  ///< Transition or explicit gap marker.
  ParamEventSequence event_sequence{};          ///< Cursor-local policy-projected record sequence.
  ParamEventSequence first_missing_sequence{};  ///< Opaque gap marker; not a source-range start.
  ParamEventSequence last_missing_sequence{};   ///< Same opaque marker; not a lost-record count.
  OperatorRef operator_instance{};              ///< Exact run-bound transition owner.
  ParamFieldId field{};                         ///< Exact field, or zero for an aggregate event.
  ParamUpdateSequence update_sequence{};        ///< Accepted update identity, or zero.
  ParamGeneration prior_generation{};           ///< Previously effective generation.
  ParamGeneration target_generation{};          ///< Candidate/effective generation.
  ParamUpdateState state{ParamUpdateState::kRejected};     ///< Reported transition state.
  ParamUpdateStatus status{ParamUpdateStatus::kRejected};  ///< Typed transition disposition.
  std::uint64_t actor_id{};                                ///< Fixed audit actor identity.
  ParamReasonId reason_id{};                               ///< Fixed registered reason identity.
};

static_assert(std::is_standard_layout_v<ParamEvent>);
static_assert(std::is_trivially_copyable_v<ParamEvent>);

/// @brief Select the initial independent Parameter event-cursor position.
struct ParamEventCursorOptions {
  /// Reserved resume position. The baseline requires zero because event sequences are
  /// projected per restricted cursor and no finite per-policy resume index is
  /// admitted yet. Any nonzero value is rejected fail-closed.
  ParamEventSequence after_event_sequence{};
};

static_assert(std::is_standard_layout_v<ParamEventCursorOptions>);
static_assert(std::is_trivially_copyable_v<ParamEventCursorOptions>);

static_assert(std::is_standard_layout_v<ParamAddress>);
static_assert(std::is_trivially_copyable_v<ParamAddress>);
static_assert(std::is_standard_layout_v<ParamReadResult>);
static_assert(std::is_trivially_copyable_v<ParamReadResult>);
static_assert(std::is_standard_layout_v<ParamUpdateResult>);
static_assert(std::is_trivially_copyable_v<ParamUpdateResult>);
static_assert(std::is_standard_layout_v<ParamBatchUpdateResult>);
static_assert(std::is_trivially_copyable_v<ParamBatchUpdateResult>);
static_assert(std::is_standard_layout_v<ParamSubmitResult>);
static_assert(std::is_trivially_copyable_v<ParamSubmitResult>);
static_assert(std::is_standard_layout_v<ParamStopFenceResult>);
static_assert(std::is_trivially_copyable_v<ParamStopFenceResult>);

}  // namespace holoscan
