/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared error factories for the public handle method bodies in src/rhi/*.cpp.
// Every handle method checks impl_ (and any handle arguments) before
// dereferencing and returns an error instead of null-dereferencing: an empty
// *handle* is kInvalidState, an empty handle *argument* is kNullArg, and a gate
// from a superseded swapchain generation is kInvalidState.

#pragma once

#include <string>

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

namespace holoviz::rhi::detail {

/// Error for a method called on an empty (default-constructed or moved-from) handle.
[[nodiscard]] inline unexpected<Error> null_impl_error(const char* method) {
  return make_unexpected(Error{
      .code = ErrorCode::kInvalidState,
      .message = std::string("[holoviz] ") + method +
                 " called on an empty (default-constructed or moved-from) handle",
  });
}

/// Error for a method that received an empty handle as an argument.
[[nodiscard]] inline unexpected<Error> null_arg_error(const char* method, const char* argument) {
  return make_unexpected(Error{
      .code = ErrorCode::kNullArg,
      .message = std::string("[holoviz] ") + method + ": argument '" + argument +
                 "' is an empty (default-constructed or moved-from) handle",
  });
}

/// Error for a gate whose minting swapchain generation has been superseded by a
/// recreation; the gate's impl points into recycled storage and must not be used.
[[nodiscard]] inline unexpected<Error> stale_gate_error(const char* method) {
  return make_unexpected(Error{
      .code = ErrorCode::kInvalidState,
      .message = std::string("[holoviz] ") + method +
                 " called on a gate from a superseded swapchain generation (the swapchain was "
                 "recreated); drop it and re-acquire",
  });
}

}  // namespace holoviz::rhi::detail
