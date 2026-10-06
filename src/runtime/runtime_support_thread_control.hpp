/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/threading/thread_placement.hpp>

namespace holoscan::detail {

/// @brief Apply source-tree measurement controls when one Runtime support thread starts.
///
/// Production plans do not install this interface. Benchmarks and conformance tests may retain one
/// through the private executable inventory to apply cold-start affinity and collect thread
/// evidence without adding work to the activation path.
class RuntimeSupportThreadControl {
 public:
  virtual ~RuntimeSupportThreadControl() = default;

  /// @brief Enter one Runtime-owned support thread before it performs session work.
  /// @param role Stable role of the calling Runtime thread.
  /// @return Success, or a typed setup error that prevents unqualified measurement.
  /// @throws Any exception propagated by a source-tree control implementation.
  [[nodiscard]] virtual expected<void, Error> enter(SupportThreadRole role) = 0;
};

}  // namespace holoscan::detail
