/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <chrono>
#include <optional>
#include <utility>

#include <holoscan/core/sample.hpp>

namespace holoscan::testing {

/// @brief Preserve the sample that could not be captured, without retaining its payload.
struct CaptureFailure {
  SampleMetadata metadata{};         ///< Original failing sample coordinates and flags.
  Error error{ErrorCode::kFailure};  ///< Native copy/capacity error; earlier captures remain valid.
};

/// @brief Own a staged value and an optional capture coordinate in the fixture clock domain.
/// @tparam T Exact input payload type.
template <PublishedPayload T>
struct TestSample {
  T data;  ///< Owning payload; Tensor storage is protected by a staged read lease.
  SampleMetadata metadata{};  ///< Canonical metadata retained at emission.
  std::optional<std::chrono::nanoseconds> capture_coordinate;  ///< Coordinate to bind at emission.
};

/// @brief Stage a capture coordinate without guessing a plan-local clock identity.
/// @tparam T Exact input payload type.
/// @param value Owning payload.
/// @param coordinate Capture timestamp in the fixture default clock domain; does not advance it.
/// @return Staged envelope whose emitter supplies the actual clock identity.
template <PublishedPayload T>
[[nodiscard]] TestSample<T> sample_at(T value, std::chrono::nanoseconds coordinate) {
  return TestSample<T>{.data = std::move(value), .capture_coordinate = coordinate};
}

}  // namespace holoscan::testing
