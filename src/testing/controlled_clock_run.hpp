/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "runtime/runtime_fixture_observation.hpp"

namespace holoscan::testing::detail {

struct ClockFixtureConnection {
  std::string operator_port;
  std::string helper;
  std::string helper_port;
};

struct ClockFixtureEdge {
  std::string name;
  PortId input{};
  PortId output{};
  VertexId source{};
  VertexId target{};
};

// One cold-qualified attempt owns every live record and its immutable plan/run identity through
// join. This controller observes native progress; it never selects inputs or schedules callbacks.
struct ControlledClockAttempt {
  std::shared_ptr<holoscan::detail::RuntimeFixtureObservation> observation;
  VertexId tested{};
  ClockId clock{};
  std::vector<ClockFixtureEdge> inputs;
  std::vector<ClockFixtureEdge> outputs;
  const holoscan::detail::FixtureObservationRecord* tick_record{};
  bool boundary_reached{};
  bool watchdog_expired{};
  std::string failure;

  void drive(RunSession& session, SyntheticClock& clock, ControlledClockRun options);
  [[nodiscard]] ClockTestOutcome outcome() const noexcept;
  [[nodiscard]] InputSelection selection(std::string_view name) const;
};

[[nodiscard]] std::shared_ptr<ControlledClockAttempt> prepare_controlled_clock(
    const ExecutionPlan& plan, std::string_view tested_name,
    std::span<const ClockFixtureConnection> inputs, std::span<const ClockFixtureConnection> outputs,
    ControlledClockRun options);

}  // namespace holoscan::testing::detail
