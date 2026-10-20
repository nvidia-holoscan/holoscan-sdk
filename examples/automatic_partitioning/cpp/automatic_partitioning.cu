/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// automatic_partitioning demonstrates the supported partition-authoring choices:
// automatic full-DAG planning, a named Green Context anchor with an automatic
// remainder, and root include_all() for one explicit serial owner. The program
// stops after compilation; partition inspection is a cold control-plane API.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

constexpr int kNoGreenContextReturnCode = 77;
constexpr int kUsageReturnCode = 64;

class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output;
};

class Stage final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;
};

class Join final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(2U);
    spec.input(right, "right").queue_depth(2U);
    spec.output(output, "output");
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> left;
  holoscan::Input<std::uint32_t> right;
  holoscan::Output<std::uint32_t> output;
};

class EmbeddedStage final : public holoscan::Graph {
 protected:
  void compose() override { (void)op<Join>("embedded-stage"); }
};

[[nodiscard]] std::string_view mode_name(holoscan::PartitioningMode mode) noexcept {
  switch (mode) {
    case holoscan::PartitioningMode::kAutomatic:
      return "automatic";
    case holoscan::PartitioningMode::kConstrainedAutomatic:
      return "constrained-automatic";
    case holoscan::PartitioningMode::kManual:
      return "manual";
  }
  return "unknown";
}

[[nodiscard]] std::string_view origin_name(holoscan::ExecutionPartitionOrigin origin) noexcept {
  return origin == holoscan::ExecutionPartitionOrigin::kAuthored ? "authored" : "automatic";
}

void print_diagnostics(const holoscan::ExecutionPlan& plan) {
  for (const auto& diagnostic : plan.diagnostics()) {
    std::cerr << "  " << diagnostic.token;
    if (!diagnostic.provided.empty()) {
      std::cerr << ": " << diagnostic.provided;
    }
    std::cerr << '\n';
  }
}

void print_partitions(std::string_view label, const holoscan::ExecutionPlan& plan) {
  const auto partitions = plan.partitions();
  std::cout << '[' << label << "] mode=" << mode_name(partitions.mode())
            << " authored=" << partitions.authored_count()
            << " automatic=" << partitions.automatic_count() << " total=" << partitions.size()
            << '\n';
  for (const auto partition : partitions) {
    std::cout << "  id=" << partition.id().value << " origin=" << origin_name(partition.origin())
              << " lane=" << partition.execution_lane() << " members=" << partition.members().size()
              << " paths=";
    bool first_path = true;
    for (const auto& path : partition.operator_paths()) {
      if (!first_path) {
        std::cout << ',';
      }
      std::cout << path;
      first_path = false;
    }
    if (const auto name = partition.authored_name()) {
      std::cout << " name=" << *name;
    }
    std::cout << '\n';
  }
}

[[nodiscard]] bool inspect_automatic_plan(const holoscan::ExecutionPlan& plan,
                                          const holoscan::OperatorHandle<Source>& source) {
  if (!plan.ok()) {
    return false;
  }
  const auto partitions = plan.partitions();
  if (partitions.mode() != holoscan::PartitioningMode::kAutomatic || partitions.empty()) {
    return false;
  }

  const auto first = *partitions.begin();
  const auto by_id = partitions.lookup(first.id());
  const auto by_path = partitions.lookup_for_path("join");
  const auto by_handle = partitions.of(source);
  const auto generated_name = partitions.lookup_named("automatic#1");
  const auto missing_path = partitions.lookup_for_path("does-not-exist");
  if (!by_id || !by_path || by_handle.id() == holoscan::ExecutionPartitionId() || generated_name ||
      missing_path || generated_name.error().code != holoscan::ErrorCode::kNotFound ||
      missing_path.error().code != holoscan::ErrorCode::kNotFound) {
    return false;
  }

  std::cout << "[automatic] lookup id=" << by_id->id().value << " path=join->"
            << by_path->id().value << " of(source)->" << by_handle.id().value
            << " failed-name=" << static_cast<int>(generated_name.error().code)
            << " failed-path=" << static_cast<int>(missing_path.error().code) << '\n';
  return true;
}

[[nodiscard]] holoscan::ExecutionPlan compile_automatic(bool& inspection_ok) {
  holoscan::Graph graph{"automatic-diamond"};
  const auto source_handle = graph.op<Source>("source");
  const auto left = graph.op<Stage>("left");
  const auto right = graph.op<Stage>("right");
  const auto join = graph.op<Join>("join");
  const auto tail = graph.op<Stage>("tail");
  graph.add_flow(source_handle->output, left->input);
  graph.add_flow(source_handle->output, right->input);
  graph.add_flow(left->output, join->left);
  graph.add_flow(right->output, join->right);
  graph.add_flow(join->output, tail->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (plan.ok()) {
    print_partitions("automatic", plan);
    inspection_ok = inspect_automatic_plan(plan, source_handle);
    if (!inspection_ok) {
      std::cerr << "automatic_partitioning: automatic inspection failed\n";
    }
  }
  return plan;
}

[[nodiscard]] holoscan::ExecutionPlan compile_include_all(bool& inspection_ok) {
  holoscan::Graph graph{"include-all"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  (void)graph.op<Source>("root-source");
  (void)graph.embed<EmbeddedStage>("embedded");
  graph.partition("whole-application").include_all();
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (plan.ok()) {
    print_partitions("include_all", plan);
    const auto nested = plan.partitions().lookup_for_path("embedded/embedded-stage");
    const auto partitions = plan.partitions();
    const auto by_name = partitions.lookup_named("whole-application");
    inspection_ok = partitions.mode() == holoscan::PartitioningMode::kManual &&
                    partitions.authored_count() == 1U && partitions.automatic_count() == 0U &&
                    partitions.size() == 1U && nested && by_name && nested->id() == by_name->id() &&
                    nested->members().size() == 2U && nested->authored_name() &&
                    *nested->authored_name() == "whole-application";
    if (!inspection_ok) {
      std::cerr << "include_all: embedded lookup failed\n";
    } else {
      std::cout << "[include_all] name=whole-application->" << by_name->id().value
                << " embedded path -> " << nested->id().value << '\n';
    }
  }
  return plan;
}

[[nodiscard]] holoscan::ExecutionPlan compile_hybrid(
    holoscan::CudaGreenContextCapability capability, bool& inspection_ok) {
  holoscan::Graph graph{"hybrid-green-context"};
  const auto source = graph.op<Source>("source");
  const auto left = graph.op<Stage>("left");
  const auto right = graph.op<Stage>("right");
  const auto join = graph.op<Join>("join");
  const auto tail = graph.op<Stage>("tail");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);
  graph.add_flow(left->output, join->left);
  graph.add_flow(right->output, join->right);
  graph.add_flow(join->output, tail->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  graph.partition("green-anchor").add(left);
  graph.auto_partition_remaining();

  holoscan::CompileOptions options;
  options.deployment.bind_cuda_green_context("green-anchor",
                                             holoscan::CudaGreenContextPlacement{
                                                 .device = capability.device,
                                                 .sm_count = capability.structured_sm_quantum,
                                             });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (plan.ok()) {
    print_partitions("hybrid", plan);
    const auto partitions = plan.partitions();
    const auto named = partitions.lookup_named("green-anchor");
    inspection_ok = named &&
                    partitions.mode() == holoscan::PartitioningMode::kConstrainedAutomatic &&
                    partitions.authored_count() == 1U && partitions.automatic_count() > 0U &&
                    named->members().size() == 1U;
    if (!inspection_ok) {
      std::cerr << "hybrid: anchor or automatic remainder inspection failed\n";
    } else {
      std::cout << "[hybrid] named anchor -> " << named->id().value
                << "; automatic remainder is compiler-owned\n";
    }
  }
  return plan;
}

}  // namespace

int main(int argc, char** argv) {
  std::string_view requested_mode{"all"};
  if (argc == 3 && std::string_view(argv[1]) == "--mode") {
    requested_mode = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: automatic_partitioning [--mode cpu|hybrid]\n";
    return kUsageReturnCode;
  }
  if (requested_mode != "all" && requested_mode != "cpu" && requested_mode != "hybrid") {
    std::cerr << "usage: automatic_partitioning [--mode cpu|hybrid]\n";
    return kUsageReturnCode;
  }

  if (requested_mode == "all" || requested_mode == "cpu") {
    bool automatic_inspection_ok = false;
    const holoscan::ExecutionPlan automatic = compile_automatic(automatic_inspection_ok);
    if (!automatic.ok() || !automatic_inspection_ok) {
      std::cerr << "automatic_partitioning: automatic compile rejected\n";
      print_diagnostics(automatic);
      return 1;
    }

    bool include_all_inspection_ok = false;
    const holoscan::ExecutionPlan include_all = compile_include_all(include_all_inspection_ok);
    if (!include_all.ok() || !include_all_inspection_ok) {
      std::cerr << "automatic_partitioning: include_all compile rejected\n";
      print_diagnostics(include_all);
      return 2;
    }
    std::cout << "automatic_partitioning cpu example complete\n";
    if (requested_mode == "cpu") {
      return 0;
    }
  }

  const holoscan::DeviceId device{0};
  const auto capability = holoscan::query_cuda_green_context_capability(device);
  if (!capability) {
    std::cout << "automatic_partitioning: skipped: "
              << holoscan::runtime_error_message(capability.error()) << '\n';
    return kNoGreenContextReturnCode;
  }
  if (capability->structured_sm_quantum == 0U || capability->max_structured_context_count < 2U) {
    std::cout << "automatic_partitioning: skipped: target cannot provide an anchor plus residual\n";
    return kNoGreenContextReturnCode;
  }

  bool hybrid_inspection_ok = false;
  const holoscan::ExecutionPlan hybrid = compile_hybrid(*capability, hybrid_inspection_ok);
  if (!hybrid.ok() || !hybrid_inspection_ok) {
    std::cerr << "automatic_partitioning: hybrid compile rejected\n";
    print_diagnostics(hybrid);
    return 3;
  }

  std::cout << "automatic_partitioning hybrid example complete\n";
  if (requested_mode == "all") {
    std::cout << "automatic_partitioning example complete\n";
  }
  return 0;
}
