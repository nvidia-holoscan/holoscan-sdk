/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/params/error.hpp>

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace holoscan::params {
namespace {

void append_single_line(std::string& output, std::string_view text) {
  constexpr unsigned char kAsciiDelete = 0x7FU;
  for (const char value : text) {
    const auto byte = static_cast<unsigned char>(value);
    const bool is_control = byte < static_cast<unsigned char>(' ') || byte == kAsciiDelete;
    output.push_back(is_control ? ' ' : value);
  }
}

[[nodiscard]] bool valid_diagnostic_token(std::string_view token) noexcept {
  if (token.empty() || token.front() < 'A' || token.front() > 'Z') {
    return false;
  }
  return std::ranges::all_of(token, [](const char value) {
    return (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') || value == '_';
  });
}

}  // namespace

ParamToolError::ParamToolError(ParamToolErrorCode code, std::string_view token,
                               std::optional<Error> core,
                               std::optional<config::ConfigError> config) noexcept
    : code_(code), core_(std::move(core)), config_(std::move(config)) {
  constexpr std::string_view kInvalidToken{"PARAM_TOOL_TOKEN_INVALID"};
  const std::string_view retained =
      token.size() <= token_.size() && valid_diagnostic_token(token) ? token : kInvalidToken;
  std::ranges::copy(retained, token_.begin());
  token_size_ = static_cast<std::uint8_t>(retained.size());
}

ParamException::ParamException(ParamToolError error)
    : RuntimeError(format_error(error)), error_(std::move(error)) {}

ParamException::~ParamException() = default;

std::string format_error(const ParamToolError& error) {
  std::string result{error.token()};
  if (error.core_cause().has_value()) {
    result.append("; caused by: ");
    append_single_line(result, runtime_error_message(*error.core_cause()));
  }
  if (error.config_cause().has_value()) {
    result.append("; caused by: ");
    result.append(config::format_error(*error.config_cause()));
  }
  return result;
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
