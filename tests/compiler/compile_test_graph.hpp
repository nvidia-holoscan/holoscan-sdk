/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/compile.hpp>
#include <holoscan/time/manual_clock.hpp>

namespace holoscan::test {

class CompileTestOperator : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2);
    spec.output(output, "output").max_emits_per_compute(1);
    spec.lifecycle().stage(LifecycleStage::kConfigure, &CompileTestOperator::on_configure);
  }

  expected<void, Error> compute(ExecutionContext&) override {
    ++compute_calls;
    return {};
  }

  LifecycleStatus on_configure(LifecycleContext&) noexcept {
    ++lifecycle_calls;
    return LifecycleStatus::kOk;
  }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  static inline std::uint32_t compute_calls{};
  static inline std::uint32_t lifecycle_calls{};
};

class CompileReadinessOperator : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification, "notification")
        .capacity(4)
        .sender_reference_capacity(3)
        .clock(ClockKey::named("notification_clock"));
    spec.doorbell_source(doorbell, "doorbell").capacity(5).clock(ClockKey::named("doorbell_clock"));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  NotificationSource notification;
  DoorbellSource doorbell;
};

inline ClockKey add_compile_test_default_clock(Graph& graph) {
  const ClockKey clock =
      graph.add_clock<ManualClock>("compile_test_default_clock", std::int64_t{0});
  graph.set_default_clock(clock);
  return clock;
}

inline void add_linear_graph_impl(Graph& graph, bool reverse_authoring) {
  if (reverse_authoring) {
    auto target = graph.op<CompileTestOperator>("target");
    auto source = graph.op<CompileTestOperator>("source");
    graph.add_flow(source->output, target->input);
    return;
  }
  auto source = graph.op<CompileTestOperator>("source");
  auto target = graph.op<CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
}

inline void add_linear_graph(Graph& graph, bool reverse_authoring = false) {
  add_compile_test_default_clock(graph);
  add_linear_graph_impl(graph, reverse_authoring);
}

inline void add_linear_graph_without_default_clock(Graph& graph, bool reverse_authoring = false) {
  add_linear_graph_impl(graph, reverse_authoring);
}

template <typename Bytes>
[[nodiscard]] bool has_nonzero_byte(const Bytes& bytes) {
  return std::any_of(
      bytes.begin(), bytes.end(), [](std::byte value) { return value != std::byte{}; });
}

[[nodiscard]] inline bool has_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  return std::any_of(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
    return item.token == token;
  });
}

}  // namespace holoscan::test
