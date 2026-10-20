/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace holoscan {

using TensorShape = std::vector<int64_t>;
using TensorByteStrides = std::vector<int64_t>;
using TensorShapeView = std::span<const int64_t>;
using TensorByteStridesSpan = std::span<const int64_t>;

/// Borrowed, optional byte strides: absent strides mean a compact layout. The view does not
/// own the strides, which must outlive it.
class TensorByteStridesView {
 public:
  TensorByteStridesView(std::nullopt_t) {}                   // NOLINT(runtime/explicit)
  TensorByteStridesView(TensorByteStridesSpan byte_strides)  // NOLINT(runtime/explicit)
      : byte_strides_(byte_strides) {}
  TensorByteStridesView(const TensorByteStrides& byte_strides)  // NOLINT(runtime/explicit)
      : byte_strides_(byte_strides) {}
  TensorByteStridesView(  // NOLINT(runtime/explicit)
      const std::optional<TensorByteStrides>& byte_strides) {
    if (byte_strides) {
      byte_strides_ = TensorByteStridesSpan{*byte_strides};
    }
  }

  /// @brief Reject borrowing strides from a temporary owner.
  /// @param byte_strides Temporary storage that cannot outlive this view.
  // NOLINTNEXTLINE(runtime/explicit)
  TensorByteStridesView(TensorByteStrides&& byte_strides) = delete;

  /// @brief Reject borrowing strides from a const temporary owner.
  /// @param byte_strides Temporary storage that cannot outlive this view.
  // NOLINTNEXTLINE(runtime/explicit)
  TensorByteStridesView(const TensorByteStrides&& byte_strides) = delete;

  /// @brief Reject borrowing strides from a temporary optional owner.
  /// @param byte_strides Temporary storage that cannot outlive this view.
  // NOLINTNEXTLINE(runtime/explicit)
  TensorByteStridesView(std::optional<TensorByteStrides>&& byte_strides) = delete;

  /// @brief Reject borrowing strides from a const temporary optional owner.
  /// @param byte_strides Temporary storage that cannot outlive this view.
  // NOLINTNEXTLINE(runtime/explicit)
  TensorByteStridesView(const std::optional<TensorByteStrides>&& byte_strides) = delete;

  explicit operator bool() const { return byte_strides_.has_value(); }
  [[nodiscard]] bool has_value() const { return byte_strides_.has_value(); }
  // Deliberately mirrors std::optional::value(): callers that need a non-throwing check use
  // has_value() or operator bool() first.
  [[nodiscard]] TensorByteStridesSpan value() const {
    assert(byte_strides_.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): asserted immediately above.
    return byte_strides_.value();
  }
  [[nodiscard]] size_t size() const { return value().size(); }
  [[nodiscard]] const int64_t& operator[](size_t index) const { return value()[index]; }
  [[nodiscard]] TensorByteStridesSpan::iterator begin() const { return value().begin(); }
  [[nodiscard]] TensorByteStridesSpan::iterator end() const { return value().end(); }

 private:
  std::optional<TensorByteStridesSpan> byte_strides_;
};

}  // namespace holoscan
