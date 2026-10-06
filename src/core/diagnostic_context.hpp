/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <holoscan/core/diagnostic.hpp>

namespace holoscan::detail {

/// First rejected Tensor output in one compute activation. Buffers own their bytes and are filled
/// only after validation fails; resetting the port is enough to start a new activation.
struct TensorContractMismatchCapture {
  PortId port{};
  std::array<char, 48U> field{};
  std::array<char, 96U> provided{};
  std::array<char, 96U> required{};
  bool truncated{};
};

struct DiagnosticCapture {
  static constexpr std::size_t kTextCapacity = 4096U;
  static constexpr std::size_t kCauseCapacity = 4U;

  struct Cause {
    ErrorCode code{ErrorCode::kFailure};
    std::size_t offset{};
    std::size_t length{};
  };

  std::array<char, kTextCapacity> text{};
  std::array<Cause, kCauseCapacity> causes{};
  std::size_t size{};
  std::size_t count{};
  bool truncated{};

  void clear() noexcept;
  void append(ErrorCode code, std::string_view message) noexcept;
  void capture_exception() noexcept;
  [[nodiscard]] DiagnosticDetail detail(DiagnosticOperation operation) const;
};

[[nodiscard]] std::string diagnostic_operator_path(const ExecutionGraphView& graph,
                                                   VertexId vertex);
[[nodiscard]] std::string diagnostic_operator_path(const GraphView& graph, VertexId vertex);
[[nodiscard]] std::string diagnostic_display_text(std::string_view text, std::size_t limit = 4096U);

}  // namespace holoscan::detail
