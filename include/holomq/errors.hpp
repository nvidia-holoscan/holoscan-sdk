/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/expected.hpp>

#include <string>
#include <string_view>

namespace holomq {

/// Error codes for HoloMQ expected-return paths.
///
/// These values belong exclusively to HoloMQ. Integrations translate every named meaning into
/// their own error domain and must not cast, assert numeric parity, or interpret diagnostic text.
enum class ErrorCode {
  kSuccess = 0,
  kFailure = 1,
  kReceiveError = 2,
  kInvalidArgument = 4,
  kNotFound = 5,
  kNotImplemented = 7,
  kBackpressured = 8,
  kNotReady = 9,
  kStaleEndpoint = 10,
  kClosed = 11,
  kTimeout = 12,
  kResourceExhausted = 13,
  kPermissionDenied = 14,
  kInterrupted = 15,
  kProtocolError = 16,
  kIncompatible = 17,
  kIncompleteFanout = 18,
  kNotSupported = 19,
};

struct Error {
  ErrorCode code = ErrorCode::kFailure;
  std::string message;
};

/**
 * @brief Construct an unexpected result with a best-effort diagnostic message.
 *
 * If allocating the diagnostic message fails, the returned Error retains the requested code and
 * omits the message so this helper remains safe to call from noexcept functions.
 *
 * @tparam T Expected value type.
 * @param code Error code to return.
 * @param message Diagnostic message to retain when allocation succeeds.
 * @return An unexpected result containing the requested code and best-effort message.
 */
template <typename T>
[[nodiscard]] expected<T, Error> make_unexpected_nothrow(const ErrorCode code,
                                                         const std::string_view message) noexcept {
  try {
    return make_unexpected(Error{code, std::string(message)});
  } catch (...) {
    return make_unexpected(Error{code});
  }
}

[[nodiscard]] HOLOMQ_EXPORT std::string_view error_code_name(ErrorCode code) noexcept;

[[nodiscard]] HOLOMQ_EXPORT std::string error_message(const Error& error);

}  // namespace holomq
