/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "../core/diagnostic_context.hpp"

namespace holoscan::detail {

/// Attach optional evidence only after retaining the structural rejection. An allocation failure
/// may lose text, but must not turn a captured callback error into a different failure.
inline void attach_diagnostic_capture(std::optional<DiagnosticDetail>& destination,
                                      const DiagnosticCapture& capture,
                                      DiagnosticOperation operation) noexcept {
  auto& detail = destination.emplace(DiagnosticDetail{
      .operation = operation,
      .error = Error{capture.count == 0U ? ErrorCode::kSuccess : capture.causes[0].code},
      .truncated = capture.truncated});
  try {
    if (capture.count != 0U) {
      const auto& first = capture.causes[0];
      detail.error.message.assign(capture.text.data() + first.offset, first.length);
    }
    for (std::size_t index = 1U; index < capture.count; ++index) {
      const auto& cause = capture.causes[index];
      detail.causes.push_back(
          Error{cause.code, std::string{capture.text.data() + cause.offset, cause.length}});
    }
  } catch (...) {
    detail.truncated = true;
  }
}

/// Preserve the first real cause and the first retained context across structural duplicates.
/// Supplemental allocation cannot erase a retained row, context, or primary error code.
inline void enrich_diagnostic_detail(std::optional<DiagnosticDetail>& retained,
                                     const DiagnosticDetail* incoming) noexcept {
  if (incoming == nullptr) {
    return;
  }
  if (!retained) {
    retained.emplace(
        DiagnosticDetail{.operation = incoming->operation, .error = Error{incoming->error.code}});
    try {
      *retained = *incoming;
    } catch (...) {
      retained->truncated = true;
    }
    return;
  }
  if (retained->error.code != ErrorCode::kSuccess || !retained->causes.empty() ||
      (incoming->error.code == ErrorCode::kSuccess && incoming->causes.empty())) {
    return;
  }
  retained->operation = incoming->operation;
  retained->error.code = incoming->error.code;
  retained->truncated |= incoming->truncated;
  try {
    retained->error.message = incoming->error.message;
    retained->causes = incoming->causes;
  } catch (...) {
    retained->truncated = true;
  }
}

/// Bound each producer-owned context field before retaining it, independently of rendering.
inline void copy_diagnostic_context(std::string& destination, std::string_view source,
                                    DiagnosticDetail& detail, std::size_t limit) {
  destination.assign(source.substr(0U, limit));
  detail.truncated |= source.size() > limit;
}

/// Retain only admitted schema/path identities, never Parameter values or projector messages.
/// This 1024-byte/four-field budget leaves room for the separately bounded owner and port names
/// within Core's 4096-byte context budget. Cause capture has its own existing fixed bound.
inline void append_derivation_context(DiagnosticDetail& detail, std::string_view field,
                                      std::string_view value) noexcept {
  constexpr std::size_t kContextCapacity = DiagnosticCapture::kTextCapacity / 4U;
  constexpr std::size_t kContextFields = 4U;
  std::size_t used{};
  for (const auto& item : detail.mismatches) {
    used += item.field.size() + item.provided.size() + item.required.size();
  }
  if (detail.mismatches.size() >= kContextFields || used + field.size() > kContextCapacity) {
    detail.truncated = true;
    return;
  }
  const auto length = std::min(value.size(), kContextCapacity - used - field.size());
  detail.truncated |= length != value.size();
  try {
    detail.mismatches.push_back(DiagnosticMismatch{
        .field = std::string{field}, .provided = std::string{value.substr(0U, length)}});
  } catch (...) {
    detail.truncated = true;
  }
}

}  // namespace holoscan::detail
