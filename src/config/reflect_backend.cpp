/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config/reflect_backend.hpp"

#include <rfl.hpp>
#include <rfl/json.hpp>

#include <string>
#include <string_view>

namespace holoscan::config_io::detail {

config::ConfigResult<void> verify_reflect_generic(std::string_view canonical_json) {
  // The Holoscan lexer has already frozen duplicate-key, numeric-kind, reserved-escape, and
  // resource-limit semantics. reflect-cpp is intentionally a secondary generic/typed backend:
  // parsing only Holoscan-owned canonical bytes prevents its inferred number model from becoming
  // public configuration authority while still exercising the MR !4537 integration path.
  auto parsed = rfl::json::read<rfl::Generic>(std::string{canonical_json});
  if (!parsed) {
    return tl::make_unexpected(config::ConfigError::make(config::ConfigErrorCode::kCodecFailure,
                                                         std::string_view{parsed.error().what()},
                                                         {},
                                                         {},
                                                         {},
                                                         config::ConfigOperationStage::kValidate));
  }
  return {};
}

}  // namespace holoscan::config_io::detail
