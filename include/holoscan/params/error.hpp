/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <holoscan/holoscan_params_export.h>
#include <holoscan/config/document.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::params {

/// @brief Classify failures produced by optional Parameter tooling.
enum class ParamToolErrorCode : std::uint8_t {
  kInvalidArgument = 0,    ///< Caller input is malformed or internally inconsistent.
  kInvalidSnapshot = 1,    ///< Snapshot structure or value projection is unusable.
  kTargetMismatch = 2,     ///< The target Graph or catalog does not match the request.
  kPermissionDenied = 3,   ///< The restricted Core capability does not grant the operation.
  kNotSupported = 4,       ///< The package, platform, profile, or value route is unsupported.
  kClosed = 5,             ///< The associated run, adapter, or tuner is closed.
  kTimeout = 6,            ///< A bounded control or shutdown wait expired.
  kPersistence = 7,        ///< Archive validation or filesystem persistence failed.
  kProtocol = 8,           ///< A bounded loopback protocol request was rejected.
  kResourceExhausted = 9,  ///< A finite tooling capacity could not accept more work.
};

/// @brief Own one typed optional-tooling failure without widening Core ErrorCode.
///
/// The value owns a bounded machine token and optional structured causes. Copying
/// the value preserves every cause. Construction performs no unbounded token
/// allocation and the resulting object is safe to read concurrently while no
/// thread mutates or destroys it.
class HOLOSCAN_PARAMS_EXPORT ParamToolError {
 public:
  /// Maximum number of bytes retained for one stable diagnostic token.
  static constexpr std::size_t kTokenCapacity = 96U;

  /// @brief Construct one bounded tooling failure.
  /// @param code Stable tooling category.
  /// @param token Stable diagnostic token beginning with an ASCII uppercase letter and containing
  /// only ASCII uppercase letters, digits, or underscores. An invalid or oversized token maps to
  /// the bounded PARAM_TOOL_TOKEN_INVALID sentinel.
  /// @param core Optional owning Core cause to preserve structurally.
  /// @param config Optional owning Config cause to preserve structurally.
  /// @note Construction does not throw and does not log.
  ParamToolError(ParamToolErrorCode code, std::string_view token,
                 std::optional<Error> core = std::nullopt,
                 std::optional<config::ConfigError> config = std::nullopt) noexcept;

  /// @brief Return the stable tooling category.
  [[nodiscard]] ParamToolErrorCode code() const noexcept { return code_; }

  /// @brief Return the stable diagnostic token.
  /// @note The view remains valid until this error is assigned or destroyed.
  [[nodiscard]] std::string_view token() const noexcept { return {token_.data(), token_size_}; }

  /// @brief Return the preserved Core cause when one exists.
  /// @return Reference valid until this error is assigned or destroyed.
  [[nodiscard]] const std::optional<Error>& core_cause() const noexcept { return core_; }

  /// @brief Return the preserved Config cause when one exists.
  /// @return Owning bounded Config diagnostic, or empty when the failure did not originate there.
  /// @note The returned reference remains valid until this error is assigned or destroyed.
  [[nodiscard]] const std::optional<config::ConfigError>& config_cause() const noexcept {
    return config_;
  }

 private:
  ParamToolErrorCode code_{ParamToolErrorCode::kInvalidArgument};
  std::array<char, kTokenCapacity> token_{};
  std::uint8_t token_size_{};
  std::optional<Error> core_;
  std::optional<config::ConfigError> config_;
};

/// @brief Carry one complete structured Parameter tooling failure through an exception boundary.
///
/// `what()` is the result of `format_error(error())`. Constructing the exception can allocate;
/// callers that need explicit failure handling use the corresponding result-returning API.
class HOLOSCAN_PARAMS_EXPORT ParamException : public RuntimeError {
 public:
  /// @brief Construct an exception while preserving the complete structured error.
  /// @param error Owning Parameter tooling failure to retain.
  explicit ParamException(ParamToolError error);

  /// @brief Destroy the exception through its exported shared-library implementation.
  ~ParamException() override;

  /// @brief Copy an exception without changing its structured failure.
  ParamException(const ParamException&) = default;
  /// @brief Replace an exception with a copy of another structured failure.
  ParamException& operator=(const ParamException&) = default;

  /// @brief Return the retained structured Parameter tooling failure.
  [[nodiscard]] const ParamToolError& error() const noexcept { return error_; }

 private:
  ParamToolError error_;
};

/// @brief Render a parameter tooling error as a single line for logs and diagnostics.
///
/// The result begins with the stable tooling token and includes any preserved Core or Config
/// cause. Config source excerpts are never included. The wording is intended for people and is
/// not a stable format; use ParamToolError accessors for programmatic decisions.
///
/// @param error Parameter tooling error to render.
/// @return Human-readable text without a trailing newline.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT std::string format_error(const ParamToolError& error);

/// @brief Return an optional-tooling failure translated from a Core error.
/// @param error Core failure to preserve structurally.
/// @param token Stable tooling diagnostic token.
/// @return Tooling failure retaining \p error as its cause.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamToolError from_core_error(Error error,
                                                                    std::string_view token);

/// @brief Return an optional-tooling failure translated from a Config error.
/// @param error Bounded Config failure to preserve structurally.
/// @param token Stable tooling diagnostic token.
/// @return Tooling failure retaining \p error as its cause.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamToolError from_config_error(config::ConfigError error,
                                                                      std::string_view token);

/// @brief Result type used by optional Parameter tooling.
/// @tparam T Successful value type.
template <typename T>
using ParamResult = expected<T, ParamToolError>;

}  // namespace holoscan::params
