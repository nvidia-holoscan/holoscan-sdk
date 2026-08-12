/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/params/error.hpp>

#include <algorithm>
#include <utility>

namespace holoscan::params {

ParamToolError::ParamToolError(ParamToolErrorCode code, std::string_view token,
                               std::optional<Error> core,
                               std::optional<config::ConfigError> config) noexcept
    : code_(code), core_(std::move(core)), config_(std::move(config)) {
  constexpr std::string_view kInvalidToken{"PARAM_TOOL_TOKEN_INVALID"};
  const std::string_view retained =
      token.empty() || token.size() > token_.size() ? kInvalidToken : token;
  std::ranges::copy(retained, token_.begin());
  token_size_ = static_cast<std::uint8_t>(retained.size());
}

ParamToolError from_core_error(Error error, std::string_view token) {
  ParamToolErrorCode code = ParamToolErrorCode::kInvalidArgument;
  switch (error.code) {
    case ErrorCode::kPermissionDenied:
      code = ParamToolErrorCode::kPermissionDenied;
      break;
    case ErrorCode::kNotSupported:
      code = ParamToolErrorCode::kNotSupported;
      break;
    case ErrorCode::kClosed:
      code = ParamToolErrorCode::kClosed;
      break;
    case ErrorCode::kTimeout:
      code = ParamToolErrorCode::kTimeout;
      break;
    case ErrorCode::kResourceExhausted:
      code = ParamToolErrorCode::kResourceExhausted;
      break;
    default:
      break;
  }
  return ParamToolError{code, token, std::move(error)};
}

ParamToolError from_config_error(config::ConfigError error, std::string_view token) {
  ParamToolErrorCode code = ParamToolErrorCode::kPersistence;
  switch (error.code()) {
    case config::ConfigErrorCode::kInvalidArgument:
    case config::ConfigErrorCode::kParseError:
    case config::ConfigErrorCode::kUnsupportedValue:
    case config::ConfigErrorCode::kDuplicateKey:
    case config::ConfigErrorCode::kTypeMismatch:
    case config::ConfigErrorCode::kOutOfRange:
      code = ParamToolErrorCode::kInvalidArgument;
      break;
    case config::ConfigErrorCode::kPermissionDenied:
      code = ParamToolErrorCode::kPermissionDenied;
      break;
    case config::ConfigErrorCode::kLimitExceeded:
    case config::ConfigErrorCode::kGuardBusy:
      code = ParamToolErrorCode::kResourceExhausted;
      break;
    case config::ConfigErrorCode::kUnsupportedFormat:
    case config::ConfigErrorCode::kLosslessPreservationUnavailable:
    case config::ConfigErrorCode::kAtomicReplaceUnsupported:
      code = ParamToolErrorCode::kNotSupported;
      break;
    default:
      break;
  }
  return ParamToolError{code, token, std::nullopt, std::move(error)};
}

}  // namespace holoscan::params
