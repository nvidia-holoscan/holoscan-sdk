/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter_compute_control.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/core/parameter_update.hpp>

namespace holoscan {

namespace detail {
struct ParamControlState;
struct ParamEventCursorState;
class ParamControlAccess;

}  // namespace detail

/// @brief Identify the trusted origin represented by a parameter-control capability.
enum class ParamControlOrigin : std::uint8_t {
  kLocalApplication = 0,  ///< In-process application control code.
  kExternalAdapter = 1,   ///< Separately authenticated adapter at an external boundary.
};

/// @brief Represent independently removable parameter-control permissions.
enum class ParamPermission : std::uint8_t {
  kNone = 0,            ///< Permit no parameter-control operation.
  kRead = 1U << 0U,     ///< Permit policy-filtered discovery and coherent reads.
  kUpdate = 1U << 1U,   ///< Permit owning parameter-update submissions.
  kAwait = 1U << 2U,    ///< Permit waiting for an authorized update sequence.
  kCapture = 1U << 3U,  ///< Permit policy-filtered owning snapshots.
  kObserve = 1U << 4U,  ///< Permit independent bounded event cursors.
};

/// @brief Combine parameter-control permission bits.
/// @param lhs Left permission set.
/// @param rhs Right permission set.
/// @return Bitwise union of both sets.
[[nodiscard]] constexpr ParamPermission operator|(ParamPermission lhs,
                                                  ParamPermission rhs) noexcept {
  // ParamPermission is a bitmask; combined bits need not name a separate enumerator.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<ParamPermission>(static_cast<std::uint8_t>(lhs) |
                                      static_cast<std::uint8_t>(rhs));
}

/// @brief Intersect parameter-control permission bits.
/// @param lhs Left permission set.
/// @param rhs Right permission set.
/// @return Bits present in both sets.
[[nodiscard]] constexpr ParamPermission operator&(ParamPermission lhs,
                                                  ParamPermission rhs) noexcept {
  // ParamPermission is a bitmask; combined bits need not name a separate enumerator.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<ParamPermission>(static_cast<std::uint8_t>(lhs) &
                                      static_cast<std::uint8_t>(rhs));
}

/// @brief Return whether a permission set contains one requested bit.
/// @param permissions Combined permission set.
/// @param requested Single or combined permission bits to test.
/// @return `true` when every requested bit is present.
[[nodiscard]] constexpr bool has_param_permission(ParamPermission permissions,
                                                  ParamPermission requested) noexcept {
  return (permissions & requested) == requested;
}

/// @brief Restrict parameter authority to a run, one Operator, or one field.
///
/// The all-zero value names the complete active run. A nonzero run-bound
/// \ref operator_instance with a zero \ref field names that Operator. Supplying
/// both a run-bound Operator and nonzero field names exactly one declared field.
/// Core rejects every other combination when a capability is minted or
/// restricted.
struct ParamScopeFilter {
  OperatorRef operator_instance{};  ///< Run-bound Operator, or zero for the complete run.
  ParamFieldId field{};             ///< Exact field, or zero for the selected run/Operator.

  /// @brief Return a complete-run scope.
  /// @return All-zero scope resolved by the minting RunSession.
  [[nodiscard]] static constexpr ParamScopeFilter run() noexcept { return {}; }

  /// @brief Return an Operator-wide scope.
  /// @param owner Run-bound Operator identity from the minting RunSession.
  /// @return Scope containing every otherwise-authorized field of \p owner, or a fail-closed
  /// invalid scope when \p owner is not run-bound.
  [[nodiscard]] static constexpr ParamScopeFilter for_operator(OperatorRef owner) noexcept {
    // A default filter is the full-run scope. Preserve a nonzero field coordinate
    // on invalid input so a missing owner cannot silently widen authority.
    return owner.run_bound() ? ParamScopeFilter{.operator_instance = owner}
                             : ParamScopeFilter{.field = 1U};
  }

  /// @brief Return an exact-field scope.
  /// @param address Run-bound Operator and declared field identity.
  /// @return Scope containing only \p address, or a fail-closed invalid scope when its Operator is
  /// not run-bound or its field identity is zero.
  [[nodiscard]] static constexpr ParamScopeFilter for_field(ParamAddress address) noexcept {
    return address.operator_instance.run_bound() && address.field != 0U
               ? ParamScopeFilter{
                     .operator_instance = address.operator_instance,
                     .field = address.field,
                 }
               : ParamScopeFilter{.field = 1U};
  }

  /// @brief Compare complete scope records.
  /// @return `true` when both scope coordinates match.
  friend constexpr bool operator==(ParamScopeFilter, ParamScopeFilter) noexcept = default;
};

static_assert(std::is_standard_layout_v<ParamScopeFilter>);
static_assert(std::is_trivially_copyable_v<ParamScopeFilter>);

/// @brief Request a restricted run-scoped parameter-control capability.
///
/// The policy can only restrict compiler- and session-recorded authority. Internal and sensitive
/// inclusion are separate opt-ins; neither flag bypasses field visibility, external-access,
/// profile, or origin policy.
struct ParamAccessPolicy {
  ParamControlOrigin origin{ParamControlOrigin::kLocalApplication};  ///< Fixed caller origin.
  ParamPermission permissions{ParamPermission::kNone};               ///< Requested operations.
  ParamScopeFilter allowed_scope{};   ///< Maximum run/Operator/field scope.
  bool include_internal{};            ///< Request otherwise-authorized internal fields.
  bool include_sensitive{};           ///< Request otherwise-authorized sensitive values.
  std::uint32_t max_event_cursors{};  ///< Maximum simultaneously open cursors for this handle.
};

static_assert(std::is_standard_layout_v<ParamAccessPolicy>);
static_assert(std::is_trivially_copyable_v<ParamAccessPolicy>);

/// @brief Own one independent position in a run's bounded Parameter event stream.
///
/// A cursor is move-only and does not consume records for any other observer.
/// Retention overwrite is returned as an explicit \ref ParamEventKind::kGap
/// record. The marker is conservative and non-counting so filtered source
/// activity cannot become a cardinality oracle. Callers then resynchronize
/// through an atomic read or a policy-filtered live-per-Operator snapshot.
/// Retaining a cursor does not keep the RunSession executing. Each open cursor
/// consumes one slot from the compile-admitted finite cursor capacity and the
/// coordinator preallocates its policy/projection bookkeeping at run creation.
/// @note Distinct cursors may be drained concurrently. Moving, destroying, or draining one cursor
/// requires external exclusion from every other operation on that same cursor object.
class HOLOSCAN_CORE_EXPORT ParamEventCursor {
 public:
  ParamEventCursor() = delete;
  ParamEventCursor(const ParamEventCursor&) = delete;
  ParamEventCursor& operator=(const ParamEventCursor&) = delete;

  /// @brief Transfer one independent cursor position.
  ParamEventCursor(ParamEventCursor&& other) noexcept;

  /// @brief Replace this cursor by transferring another position.
  ParamEventCursor& operator=(ParamEventCursor&& other) noexcept;

  /// @brief Release the cursor's finite per-capability reservation.
  ~ParamEventCursor() noexcept;

  /// @brief Drain retained policy-visible events from this cursor position.
  /// @param output Caller-owned fixed output storage; an empty span is valid.
  /// @return Number of initialized records, or a typed closed error.
  /// @retval ErrorCode::kInvalidArgument The call occurs from a Runtime callback context.
  /// @retval ErrorCode::kClosed The cursor is moved-from or its session has retired.
  /// @retval ErrorCode::kResourceExhausted The cursor-local event sequence cannot advance without
  /// wrapping.
  /// @note This cold-path operation never invokes observer code on the
  /// coordinator or dispatcher path and never exposes Parameter values. One
  /// opaque gap can represent any positive number of lost policy-visible
  /// records; its sequence fields do not report a source-record count.
  [[nodiscard]] expected<std::size_t, Error> drain(std::span<ParamEvent> output) noexcept;

 private:
  explicit ParamEventCursor(detail::ParamEventCursorState* state) noexcept : state_(state) {}
  detail::ParamEventCursorState* state_{};
  friend class detail::ParamControlAccess;
};

static_assert(sizeof(ParamEventCursor) == sizeof(void*));

/// @brief Own a restricted run-scoped parameter-control capability.
///
/// Copies share the same already-restricted capability and cannot add authority. The single opaque
/// pointer refers to ref-counted or tombstone state owned by Core rather than exposing a standard-
/// library smart-pointer ABI. A retained handle does not keep a run executing; after session
/// retirement its operations fail with a typed closed or stale result and never dereference freed
/// runtime state. General operations are cold/control-path calls and are invalid from compute or
/// dispatcher contexts.
/// @warning EA1 provides this as a bounded Beta control substrate whose API may still evolve. It
/// supports only admitted \c NextCompute publication and shared lifecycle transactions; it does not
/// imply arbitrary effective times, graph-wide atomic updates, persistence, or a production
/// management service.
/// @note Calls through distinct wrappers that share this capability may be concurrent. Copying,
/// moving, assigning, or destroying one wrapper requires external exclusion from operations using
/// that same wrapper.
/// @note `RunSession::parameter_control()` may block during asynchronous startup until coordinator
/// attachment succeeds or startup retires the control link. The minting operation has no timeout;
/// an `await()` timeout starts only after a handle has been returned and cannot bound minting.
/// Once a handle is returned, its live operations serialize with coordinator retirement and may
/// briefly wait for an in-progress internal attachment/close handoff; they never keep the run
/// alive.
class HOLOSCAN_CORE_EXPORT ParamControlHandle {
 public:
  /// @brief Prohibit construction without a minted run-scoped capability.
  ParamControlHandle() = delete;

  /// @brief Share the same restricted capability.
  /// @param other Capability to retain.
  ParamControlHandle(const ParamControlHandle& other) noexcept;

  /// @brief Transfer one retained capability reference.
  /// @param other Capability to consume; it becomes inert.
  ParamControlHandle(ParamControlHandle&& other) noexcept;

  /// @brief Replace this reference with another shared restricted capability.
  /// @param other Capability to retain.
  /// @return This wrapper after replacement.
  ParamControlHandle& operator=(const ParamControlHandle& other) noexcept;

  /// @brief Replace this reference by transferring another capability.
  /// @param other Capability to consume; it becomes inert.
  /// @return This wrapper after replacement.
  ParamControlHandle& operator=(ParamControlHandle&& other) noexcept;

  /// @brief Release this wrapper's retained capability reference.
  ~ParamControlHandle() noexcept;

  /// @brief Return this capability's immutable policy-filtered discovery catalog.
  /// @return Catalog whose records own all names and remain valid until the last wrapper sharing
  /// this capability state is destroyed.
  /// @note A read-capable handle prepares this projection before its minting operation succeeds;
  /// this accessor never waits, allocates, or dereferences live runtime storage. The prepared
  /// catalog therefore remains available after session retirement. A moved-from capability, a
  /// capability minted without \ref ParamPermission::kRead, or a call from compute/dispatcher
  /// context returns an intentionally empty projection.
  /// @note Calls through distinct wrappers sharing the capability may read this immutable view
  /// concurrently. The reference remains valid until the last such wrapper is destroyed.
  [[nodiscard]] const ParameterCatalogView& catalog() const noexcept;

  /// @brief Read one field from a coherent immutable parameter record.
  /// @param address Run-bound declared field address authorized by this capability.
  /// @param view Published, latest-accepted, or pending-only view to sample. Latest-accepted falls
  /// back to the published record when no accepted candidate is pending.
  /// @return Owning fixed scalar plus generation, sequence, and state from the same record.
  /// @retval ErrorCode::kInvalidArgument The view is invalid or the call occurs from a Runtime
  /// callback context.
  /// @retval ErrorCode::kPermissionDenied The capability cannot read this field or view.
  /// @retval ErrorCode::kStaleOperatorRef The address is foreign to this active run.
  /// @retval ErrorCode::kNotFound The field is undeclared or has no scalar read projection.
  /// @retval ErrorCode::kNotReady The pending-only view has no unpublished accepted candidate.
  /// @retval ErrorCode::kIncompatible The admitted field cannot be represented by \ref ParamValue.
  /// @retval ErrorCode::kClosed The retained capability no longer has a live session.
  [[nodiscard]] expected<ParamReadResult, Error> read(
      ParamAddress address, ParamReadView view = ParamReadView::kPublished) const;

  /// @brief Derive a capability that can only remove authority.
  /// @param policy Requested origin, permission, scope, inclusion, and cursor ceilings.
  /// @return Independently retained restricted capability.
  /// @retval ErrorCode::kPermissionDenied The requested policy widens any parent authority.
  /// @retval ErrorCode::kInvalidArgument The policy contains an invalid enum or scope shape.
  /// @retval ErrorCode::kClosed The retained session capability has retired.
  /// @retval ErrorCode::kResourceExhausted The immutable child catalog could not be prepared.
  /// @note This cold operation may allocate the child capability and catalog projection. It is
  /// invalid from Runtime callback contexts.
  [[nodiscard]] expected<ParamControlHandle, Error> restrict(const ParamAccessPolicy& policy) const;

  /// @brief Submit one owning fixed-scalar patch.
  /// @param patch Complete patch copied before this call returns.
  /// @param options Field-derived or explicit `NextCompute` timing and bounded audit evidence.
  /// @return Immediate admission result for one same-Operator candidate.
  /// @note Invalid addresses, values, timing, audit text, access policy, closed ingress, or
  /// Runtime callback context reject synchronously without reserving a candidate generation.
  /// @note Acceptance and \ref ParamBatchUpdateResult::committed reserve a candidate generation;
  /// they do not mean the value is already published. This cold operation may block briefly on
  /// startup/retirement handoff and is invalid from Runtime callback contexts.
  [[nodiscard]] ParamBatchUpdateResult submit(const ParamPatch& patch,
                                              ParamUpdateOptions options = {});

  /// @brief Submit one same-operator all-or-nothing fixed-scalar patch batch.
  /// @param patches Complete patches copied before this call returns.
  /// @param options Field-derived or explicit `NextCompute` timing and bounded audit evidence.
  /// @return Aggregate admission result with no borrowed diagnostic or field-result storage.
  /// @note Invalid addresses, values, timing, audit text, access policy, closed ingress, or
  /// Runtime callback context reject synchronously without reserving a candidate generation.
  /// @note An empty or cross-operator batch rejects without accepting a sequence. Acceptance and
  /// \ref ParamBatchUpdateResult::committed reserve one complete candidate generation; they do not
  /// mean that candidate is published. The operation never retains the caller's span, may block
  /// briefly on startup/retirement handoff, and is invalid from Runtime callback contexts.
  [[nodiscard]] ParamBatchUpdateResult submit(std::span<const ParamPatch> patches,
                                              ParamUpdateOptions options = {});

  /// @brief Wait for one authorized accepted sequence to reach a terminal state.
  /// @param update_sequence Nonzero sequence issued by this run and visible to the capability.
  /// @param timeout Maximum relative wait; zero performs a nonblocking state query.
  /// @return Current terminal result, or a typed permission, stale, closed, or timeout error.
  /// @retval ErrorCode::kInvalidArgument The sequence is zero, timeout is negative, or the call
  /// occurs from a Runtime callback context.
  /// @retval ErrorCode::kPermissionDenied The sequence is absent or outside this capability's
  /// policy-visible scope.
  /// @retval ErrorCode::kClosed The retained capability no longer has a live session.
  /// @retval ErrorCode::kTimeout The sequence did not become terminal before the timeout.
  /// @note The relative timeout includes any wait for coordinator attachment and the terminal-state
  /// wait. `kPublished` is terminal even though a later activation may advance retained lineage to
  /// `kObserved`; this call neither waits for nor causes observation. A timeout never cancels the
  /// update. Waiting is invalid from Runtime callback contexts.
  [[nodiscard]] expected<ParamUpdateResult, Error> await(ParamUpdateSequence update_sequence,
                                                         std::chrono::nanoseconds timeout) const;

  /// @brief Capture owning policy-filtered Parameter state.
  /// @param options Published stopped or explicitly live-per-Operator capture options.
  /// @return Owning snapshot independent of live generation-slot lifetime.
  /// @retval ErrorCode::kNotReady Stopped capture was requested while callbacks may still run.
  /// @retval ErrorCode::kNotSupported A reserved consistency/view or unsupported strict value was
  /// requested.
  /// @retval ErrorCode::kInvalidArgument The call occurs from a Runtime callback context.
  /// @retval ErrorCode::kPermissionDenied The capability lacks capture permission.
  /// @retval ErrorCode::kClosed The retained capability no longer has a live session.
  /// @retval ErrorCode::kResourceExhausted Owning snapshot storage could not be allocated.
  /// @note The call may briefly wait for startup/retirement handoff. Distinct returned snapshots
  /// own independent storage and do not retain live generation slots.
  [[nodiscard]] expected<ParamSnapshot, Error> capture(ParamCaptureOptions options = {}) const;

  /// @brief Open one independently positioned bounded event cursor.
  /// @param options Reserved resume position; EA1 accepts only the default zero value.
  /// @return Move-only cursor, or a typed permission/capacity/closed result.
  /// @retval ErrorCode::kInvalidArgument A nonzero resume position was requested or the call
  /// occurs from a Runtime callback context.
  /// @retval ErrorCode::kPermissionDenied The capability lacks observation permission.
  /// @retval ErrorCode::kResourceExhausted The capability or run has no finite cursor slot.
  /// @retval ErrorCode::kClosed The retained capability no longer has a live session.
  /// @note Event sequences are cursor-local and policy-projected so filtered activity cannot be
  /// inferred from sequence holes. Resumable cursors require a future admitted per-policy index
  /// and capacity design.
  [[nodiscard]] expected<ParamEventCursor, Error> open_event_cursor(
      ParamEventCursorOptions options = {}) const;

 private:
  explicit ParamControlHandle(detail::ParamControlState* state) noexcept : state_(state) {}

  detail::ParamControlState* state_{};

  friend class detail::ParamControlAccess;
};

static_assert(sizeof(ParamControlHandle) == sizeof(void*));

}  // namespace holoscan
