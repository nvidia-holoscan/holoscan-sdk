/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/time/clock_registry.hpp>

#include "port_binding.hpp"
#include "temporal_input_cache.hpp"

namespace holoscan::detail {

/// Materialize compiler-frozen OnClock readiness without manufacturing catch-up ticks.
class ClockTriggerPortBinding final : public PortBinding {
 public:
  [[nodiscard]] static expected<std::unique_ptr<ClockTriggerPortBinding>, Error> create(
      RuntimePortBindingRequest request, ClockRef clock, TimePoint phase,
      std::chrono::nanoseconds period, bool recapture_phase_on_restart);

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept override;
  [[nodiscard]] std::size_t capacity() const noexcept override { return 1U; }
  void cancel() noexcept override { closed_ = true; }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }
  [[nodiscard]] SourceId source() const noexcept override { return request_.source; }
  [[nodiscard]] std::optional<TimePoint> next_wake() const noexcept override { return next_wake_; }

  [[nodiscard]] expected<std::optional<TimePoint>, Error> next_reference() const noexcept;

  [[nodiscard]] expected<TimePoint, Error> restart_phase() const noexcept;
  [[nodiscard]] std::uint64_t publish_restart(TimePoint phase) noexcept;

 private:
  ClockTriggerPortBinding(RuntimePortBindingRequest request, ClockRef clock, TimePoint phase,
                          std::chrono::nanoseconds period,
                          bool recapture_phase_on_restart) noexcept;

  RuntimePortBindingRequest request_;
  ClockRef clock_;
  TimePoint explicit_phase_;
  std::chrono::nanoseconds period_{};
  CoalescingClockSchedule schedule_;
  std::optional<TimePoint> next_wake_;
  std::optional<Error> terminal_error_;
  std::uint64_t source_sequence_{};
  bool recapture_phase_on_restart_{};
  bool closed_{};
};

/// Construct the source-tree-only finite Spin witness used by Runtime test plans.
class TestDirectPortBindingFactory final : public RuntimePortBindingFactory {
 public:
  explicit TestDirectPortBindingFactory(std::uint32_t witness_count) noexcept
      : witness_count_(witness_count) {}

  [[nodiscard]] expected<std::unique_ptr<PortBinding>, Error> bind(
      const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept override;

 private:
  std::uint32_t witness_count_{};
};

}  // namespace holoscan::detail
