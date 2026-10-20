/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <holoscan/holoscan_core_export.h>

namespace holoscan {
/// @brief Enumerate stable Core error categories.
///
/// Values use an explicit 32-bit unsigned representation. Existing numeric values are append-only
/// once landed; Core and backend error domains are intentionally numerically independent.
enum class ErrorCode : std::uint32_t {
  kSuccess = 0,             ///< No error.
  kFailure = 1,             ///< Generic Holoscan SDK error.
  kReceiveError = 2,        ///< Residual backend receive or materialization failure.
  kCodecError = 3,          ///< Serialization or deserialization failure.
  kInvalidArgument = 4,     ///< Invalid argument.
  kNotFound = 5,            ///< Requested object was not found.
  kDuplicateName = 6,       ///< A name conflicts with an existing object.
  kNotImplemented = 7,      ///< The requested SDK behavior has not been implemented.
  kBackpressured = 8,       ///< A channel or queue cannot currently accept more data.
  kNotReady = 9,            ///< An endpoint or runtime object is not ready.
  kStaleEndpoint = 10,      ///< An endpoint reference names a stale generation.
  kClosed = 11,             ///< An endpoint, channel, or session is closed.
  kResourceExhausted = 12,  ///< A bounded pool or owning claim has no available capacity.
  kTimeout = 13,            ///< An operation did not complete before its deadline.
  kPermissionDenied = 14,   ///< An operation lacks the required authority.
  kInterrupted = 15,        ///< An operation was interrupted before completion.
  kProtocolError = 16,      ///< A peer or transport violated the accepted protocol.
  kIncompatible = 17,       ///< Endpoint, schema, or resource configuration is incompatible.
  kInputEmpty = 18,         ///< The selected scalar input has no completed sample.
  kInputNotSelected = 19,   ///< The live input was not selected for this activation.
  kInputUndeclared = 20,    ///< The requested input is undeclared or unbound.
  kInputSelectionCardinalityMismatch = 21,  ///< Accessor and frozen selector shapes differ.
  /// Publication may have reached at least one recipient. The stamped sequence and publication
  /// authority are consumed, the same logical sample must not be retried, and unresolved release
  /// claims remain retained until the publication outcome is resolved.
  kPublicationIndeterminate = 22,
  kNotSupported = 23,  ///< A valid capability is unavailable on this platform or backend.
  /// The requested restart stage is not one of the forward stages from configure through start.
  kLifecycleInvalidRestartStage = 24,
  /// Active lifecycle transactions are absent from the accepted execution plan.
  kLifecycleTransactionsDisabled = 25,
  /// An operator reference names a plan generation other than the active plan generation.
  kStaleOperatorRef = 26,
  /// An operator or receipt reference belongs to a different run session.
  kCrossSessionOperatorRef = 27,
  /// The request origin lacks authority for the target, stage, or cross-operator edge.
  kLifecycleRestartUnauthorized = 28,
  /// A finite lifecycle request, transaction, receipt, fence, generation, event, audit, quiesce,
  /// or rollback reservation is unavailable. The request is rejected without a partial fence.
  kLifecycleRestartQueueFull = 29,
  /// The effective profile or region rejects active lifecycle transactions.
  kLifecycleRestartUnsupportedInProfile = 30,
  /// Lifecycle control was invoked from a context that cannot submit or wait for the operation.
  kLifecycleInvalidCallingContext = 31,
  /// The run session is not in a state that can accept the lifecycle operation.
  kLifecycleRunState = 32,
  /// A plan-recorded activation, CUDA, loan, external, or handle obligation did not quiesce.
  kLifecycleQuiesceFailed = 33,
  /// A lifecycle callback failed during the requested transition.
  kLifecycleStageFailed = 34,
  /// Rollback failed after a primary lifecycle failure; both causes remain structurally linked.
  kLifecycleRollbackFailed = 35,
  /// Publication reached a known strict subset of its expected recipients. The diagnostic message
  /// carries the observed and expected counts. Publication authority is consumed and the same
  /// logical sample must not be retried.
  kPublicationPartial = 36,
  /// A lifecycle-selected nominal rate is zero or has a zero denominator.
  kRateBindingInvalid = 37,
  /// A lifecycle-selected nominal rate exceeds the declaration's admitted finite maximum.
  kRateBindingAboveMaximum = 38,
  /// A lifecycle callback repeated one binding with a semantically different nominal rate.
  kRateBindingConflict = 39,
  /// Checked arithmetic could not represent its result in the required finite value domain.
  kArithmeticOverflow = 40,
  /// Prepared native storage was altered or a nested owner escaped its permitted lifetime.
  /// A local storage-preservation contract was violated, not a peer/wire protocol. Release
  /// escaped aliases and repair/recreate scratch; a terminal failed run is not auto-retried.
  kPreparedStorageInvalid = 41,
};

/**
 * @brief Return whether a publication error consumes publication authority.
 *
 * Both a known partial delivery and an outcome whose visibility is indeterminate are terminal for
 * the logical sample: neither may be retried, and any unresolved lifetime claims remain retained.
 *
 * @param code Error code to classify.
 * @return `true` for terminal post-submit publication outcomes; otherwise `false`.
 */
[[nodiscard]] constexpr bool publication_error_consumes_authority(const ErrorCode code) noexcept {
  return code == ErrorCode::kPublicationIndeterminate || code == ErrorCode::kPublicationPartial;
}

/**
 * @brief Get the name of an error code.
 *
 * Returns the enumerator's spelling (e.g. ErrorCode::kInvalidArgument -> "kInvalidArgument"), so
 * new ErrorCode values are named automatically with no separate table to maintain. The returned
 * string_view points at static storage.
 *
 * @param code The error code.
 * @return The enumerator name as a string_view ("unknown" if it is not a named enumerator).
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::string_view error_code_name(ErrorCode code) noexcept;

/**
 * @brief Lightweight error value for the expected<T, Error> return path.
 *
 * Error is the value type to return (via holoscan::expected / make_unexpected) when reporting a
 * recoverable failure, as opposed to throwing. It is intentionally cheap: it carries an ErrorCode
 * and an optional message, and allocates only when the message is non-empty. The common
 * "code only" case (e.g. returning kNotReady / kBackpressured from a hot polling path) performs no
 * allocation, in contrast to RuntimeError, which (as a std::runtime_error) always copies its
 * message into a heap buffer and carries exception RTTI.
 *
 * When an empty message is rendered (e.g. by the RuntimeError bridge), error_code_name(code) / the
 * code's error string is used in its place.
 *
 * Use RuntimeError when an exception must be thrown; a RuntimeError can be constructed from an
 * Error (see RuntimeError(const Error&)), so `throw RuntimeError(result.error());` stays a
 * one-liner.
 */
struct Error {
  ErrorCode code = ErrorCode::kFailure;  ///< The error code.
  /// Optional context. Empty (and allocation-free) for code-only errors.
  std::string message;
  /// Opaque identity of the originating failure, or zero when absent. Copy the complete Error to
  /// preserve this identity; constructing another Error from its code and message does not.
  std::uint64_t provenance = 0;
};

/**
 * @brief Construct the stable diagnostic for a known strict-subset publication.
 * @param delivered_recipient_count Exact number of recipients that accepted the publication.
 * @param expected_recipient_count Frozen expected recipient count.
 * @return Non-retryable partial-publication error with both counts when formatting succeeds.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT Error make_publication_partial_error(
    std::uint32_t delivered_recipient_count, std::uint32_t expected_recipient_count) noexcept;

/**
 * @brief Render an Error as a human-readable string for logs and diagnostics.
 *
 * Produces the code's name when the message is empty, otherwise `<code name>: <message>` — the same
 * rendering used by the RuntimeError(const Error&) bridge. Used by library code that logs a failed
 * expected<T, Error> (e.g. in HOLOSCAN_LOG_ERROR) and by tests for assertion messages, without
 * having to construct a RuntimeError just to format.
 *
 * @param error The error value to render.
 * @return The formatted message string.
 */
[[nodiscard]] inline std::string runtime_error_message(const Error& error) {
  const std::string_view code_name = error_code_name(error.code);
  if (error.message.empty()) {
    return std::string(code_name);
  }
  return std::string(code_name) + ": " + error.message;
}

/**
 * @brief Class for runtime errors related to Holoscan SDK.
 *
 *  This class is used to hold runtime error message in Holoscan SDK.
 *
 * New error codes are added by appending to the ErrorCode enum; their names are derived
 * automatically by error_code_name().
 */
class RuntimeError : public std::runtime_error {
 public:
  // Inherit constructors from std::runtime_error
  using std::runtime_error::runtime_error;

  HOLOSCAN_CORE_EXPORT explicit RuntimeError(holoscan::ErrorCode error_code);
  HOLOSCAN_CORE_EXPORT RuntimeError(holoscan::ErrorCode error_code, const std::string& what_arg);
  HOLOSCAN_CORE_EXPORT RuntimeError(holoscan::ErrorCode error_code, std::string_view what_arg);
  HOLOSCAN_CORE_EXPORT RuntimeError(holoscan::ErrorCode error_code, const char* what_arg);

  /**
   * @brief Construct a RuntimeError from an Error value (the bridge from the expected<T, Error>
   * return path to the throwing path).
   *
   * The error code is preserved (see code()) and the message mirrors the (ErrorCode, message)
   * constructor: the code's name when the Error message is empty, otherwise
   * `<code name>: <message>`.
   *
   * @param error The error value to convert.
   */
  HOLOSCAN_CORE_EXPORT explicit RuntimeError(const Error& error);

  /**
   * @brief Get the error code carried by this error.
   *
   * @return The ErrorCode (kFailure for errors constructed without an explicit code).
   */
  [[nodiscard]] holoscan::ErrorCode code() const noexcept { return code_; }

 private:
  // Helper function to construct the error message (`<code name>` or `<code name>: <what_arg>`).
  static std::string construct_error_message(holoscan::ErrorCode error_code, const char* what_arg);
  static std::string construct_error_message(holoscan::ErrorCode error_code,
                                             std::string_view what_arg);

  /// The error code carried by this error (kFailure when constructed without an explicit code).
  holoscan::ErrorCode code_ = holoscan::ErrorCode::kFailure;
};

/**
 * @brief Build an Error value from a RuntimeError (the reverse of RuntimeError(const Error&)).
 *
 * The code is taken from error.code(). The message is recovered from what() with the generated
 * `<code name>: ` prefix stripped, so round-tripping (Error -> RuntimeError -> Error) does not
 * duplicate the prefix; a code-only RuntimeError maps to an empty message. A message that does not
 * carry the generated prefix (e.g. from an inherited std::runtime_error constructor) is kept
 * verbatim.
 *
 * @param error The runtime error to convert.
 * @return The equivalent Error value.
 */
inline Error make_error(const RuntimeError& error) {
  const std::string_view what = error.what();
  const std::string_view code_string = error_code_name(error.code());
  if (what == code_string) {
    return Error{error.code()};  // code-only: no message
  }
  const std::string prefix = std::string(code_string) + ": ";
  if (what.size() >= prefix.size() && what.substr(0, prefix.size()) == prefix) {
    return Error{error.code(), std::string(what.substr(prefix.size()))};
  }
  return Error{error.code(), std::string(what)};
}
}  // namespace holoscan
