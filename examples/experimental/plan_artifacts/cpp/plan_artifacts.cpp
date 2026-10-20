/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// plan_artifacts: the plan-tooling workflow as a program. Nothing here runs a
// pipeline — every lesson is a cold operation on the immutable ExecutionPlan:
//
//   1. accessor table — survey the headline read-only record accessors
//      (execution_partitions(), scheduler_plan(), clock_catalog(),
//      temporal_triggers(), tensor_allocations(), ...);
//   2. plan.json()    — canonical schema-versioned JSON persisted to a file;
//   3. plan.digest()  — the 256-bit CI baseline: two independent builds of the
//      identical graph produce equal digests, an unused endpoint pin does not
//      move the digest, and a small tuning edit does;
//   4. emit_artifacts() — publish the deployment bundle twice into the same
//      root and prove idempotence (second ArtifactEmitResult.changed == false);
//   5. one DeploymentBinding::allow_generated_endpoint_name pin in
//      CompileOptions, echoed back from the deep-owned pin record.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fmt/format.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../../common/temporary_workspace.hpp"

namespace {

using namespace std::chrono_literals;

// Fixed-rate producer. This example never runs the plan; the operator exists so
// the compiler has real structure and timing to plan against.
class Telemetry final : public holoscan::Operator<> {
 public:
  explicit Telemetry(std::chrono::nanoseconds period) noexcept : period_(period) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return out.emit(++next_);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::chrono::nanoseconds period_;
  std::uint32_t next_{};
};

// Consumes every sample; the data-driven counterpart to the clock-driven producer.
class Recorder final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(const std::uint32_t value, in.receive_data());
    (void)value;
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

// One authoring function for every variant, so digest comparisons see exactly
// the differences we intend and nothing else.
void build_pipeline(holoscan::Graph& graph, std::chrono::nanoseconds tick_period,
                    std::uint32_t queue_depth) {
  const auto telemetry = graph.op<Telemetry>("telemetry", tick_period);
  const auto recorder = graph.op<Recorder>("recorder");
  graph.add_flow(
      telemetry->out, recorder->in, holoscan::ConnectionOptions{.queue_depth = queue_depth});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("app-clock"));
}

// The one pin this example demonstrates. The graph exports no external topic,
// so the pin stays *unused*: it is deployment authority, not plan semantics,
// and the digest section proves it does not move the plan identity.
[[nodiscard]] holoscan::CompileOptions pinned_options() {
  holoscan::CompileOptions options;
  options.deployment.allow_generated_endpoint_name(
      holoscan::GeneratedEndpointNamePin{.final_name = "telemetry.samples"});
  return options;
}

// Full 64-character lowercase hex of the 256-bit digest byte array.
[[nodiscard]] std::string digest_hex(const holoscan::PlanDigest& digest) {
  static constexpr std::string_view kHex = "0123456789abcdef";
  std::string text;
  text.reserve(digest.value.size() * 2U);
  for (const std::byte byte : digest.value) {
    const auto value = std::to_integer<std::uint8_t>(byte);
    text.push_back(kHex[value >> 4U]);
    text.push_back(kHex[value & 0x0FU]);
  }
  return text;
}

[[nodiscard]] constexpr std::string_view trigger_kind_name(
    holoscan::TemporalRecordTriggerKind kind) noexcept {
  switch (kind) {
    case holoscan::TemporalRecordTriggerKind::kAnyInput:
      return "kAnyInput";
    case holoscan::TemporalRecordTriggerKind::kEach:
      return "kEach";
    case holoscan::TemporalRecordTriggerKind::kClock:
      return "kClock";
    default:
      return "other";
  }
}

// Resolve canonical plan VertexIds to authoring names via plan.graph() — the
// plan's own frozen name table, never the pre-compile Graph object.
[[nodiscard]] std::string member_names(const holoscan::ExecutionPlan& plan,
                                       const std::vector<holoscan::VertexId>& members) {
  std::string text;
  for (const holoscan::VertexId member : members) {
    for (const holoscan::VertexRecord& vertex : plan.graph().vertices()) {
      if (vertex.id == member) {
        if (!text.empty()) {
          text += ',';
        }
        text += vertex.name;
      }
    }
  }
  return text;
}

void print_row(std::string_view accessor, const std::string& value) {
  std::cout << "[table] " << accessor;
  for (std::size_t pad = accessor.size(); pad < 34U; ++pad) {
    std::cout << ' ';
  }
  std::cout << "| " << value << '\n';
}

[[nodiscard]] bool print_and_validate_accessor_table(const holoscan::ExecutionPlan& plan) {
  const auto partitions = plan.execution_partitions();
  const holoscan::SchedulerPlanRecord scheduler = plan.scheduler_plan();
  const holoscan::ClockCatalogView clocks = plan.clock_catalog();
  const auto triggers = plan.temporal_triggers();
  const auto tensor_pools = plan.tensor_allocations();

  print_row("execution_partitions().size()", std::to_string(partitions.size()));
  for (const holoscan::ExecutionPartitionRecord& partition : partitions) {
    print_row(
        fmt::format("  partition {}", partition.id.value),
        fmt::format(
            "lane={} members={}", partition.execution_lane, member_names(plan, partition.members)));
  }
  print_row("scheduler_plan()",
            fmt::format("ready_policy={} v{} lanes={} max_concurrent={} workers={}",
                        scheduler.ready_policy.id,
                        scheduler.ready_policy.version,
                        scheduler.execution_lane_count,
                        scheduler.max_concurrent_activations,
                        scheduler.worker_count));
  {
    std::string clock_text =
        fmt::format("plan_generation={} clocks={}", clocks.plan_generation, clocks.clocks.size());
    if (!clocks.clocks.empty()) {
      clock_text += " default=" + std::string(clocks.clocks.front().key.name());
    }
    print_row("clock_catalog()", clock_text);
  }
  {
    std::string trigger_text = std::to_string(triggers.size()) + " (";
    for (std::size_t index = 0; index < triggers.size(); ++index) {
      trigger_text += index == 0 ? "" : ", ";
      trigger_text += trigger_kind_name(triggers[index].kind);
    }
    trigger_text += ')';
    print_row("temporal_triggers().size()", trigger_text);
  }
  print_row("tensor_allocations().size()",
            fmt::format("{} (no tensor_allocation outputs here)", tensor_pools.size()));
  print_row("json().size()", fmt::format("{} bytes", plan.json().size()));

  return !partitions.empty() && scheduler.execution_lane_count == plan.execution_lanes().size() &&
         clocks.plan_generation != 0U && clocks.clocks.size() == 1U &&
         clocks.clocks.front().is_default && clocks.clocks.front().key.name() == "app-clock" &&
         triggers.size() == 2U && tensor_pools.empty() && !plan.json().empty();
}

}  // namespace

// REQ: SYSTEM-015
int main() {
  // --- 1. Compile once, with the endpoint pin in the options. ---------------
  holoscan::Graph graph{"plan-artifacts"};
  build_pipeline(graph, 10ms, 4U);

  const holoscan::CompileOptions options = pinned_options();
  const auto pins = options.deployment.generated_endpoint_names();
  if (pins.size() != 1U || pins[0].final_name != "telemetry.samples" ||
      pins[0].algorithm != "holoscan.auto_endpoint_name" || pins[0].version != 1U) {
    std::cerr << "unexpected generated-endpoint pin record\n";
    return 1;
  }
  std::cout << "[pin] final_name=" << pins[0].final_name << " algorithm=" << pins[0].algorithm
            << " version=" << pins[0].version << '\n';

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, options);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 2;
  }
  std::cout << "[plan] ok: yes (diagnostics: " << plan.diagnostics().size() << ")\n";

  // --- 2. Headline accessor table: cheap, read-only, side-effect-free. ------
  if (!print_and_validate_accessor_table(plan)) {
    std::cerr << "accessor table failed a consistency check\n";
    return 3;
  }

  const std::filesystem::path workspace = holoscan_examples::create_temporary_workspace();
  if (workspace.empty()) {
    std::cerr << "failed to create a unique temporary workspace\n";
    return 4;
  }
  const holoscan_examples::TemporaryWorkspaceCleanup workspace_cleanup{workspace};

  // --- 3. plan.json(): canonical bytes, safe to archive and diff. -----------
  const std::filesystem::path json_path = workspace / "plan.json";
  {
    std::ofstream json_file(json_path, std::ios::binary | std::ios::trunc);
    json_file << plan.json();
    json_file.close();
    if (!json_file) {
      std::cerr << "failed to write " << json_path << '\n';
      return 4;
    }
  }
  std::cout << "[json] plan.json written: " << json_path.filename().string() << " ("
            << plan.json().size() << " bytes)\n";

  // --- 4. Digest baseline: the CI comparison, three ways. -------------------
  // (a) An independently authored, identical graph compiles to the same digest.
  holoscan::Graph second{"plan-artifacts"};
  build_pipeline(second, 10ms, 4U);
  const holoscan::ExecutionPlan second_plan = holoscan::compile(second, pinned_options());
  if (!second_plan.ok()) {
    std::cerr << second_plan.json() << '\n';
    return 5;
  }
  std::cout << "[digest] first  compile: " << digest_hex(plan.digest()) << '\n';
  std::cout << "[digest] second compile: " << digest_hex(second_plan.digest()) << '\n';
  const bool stable = plan.digest() == second_plan.digest();
  std::cout << "[digest] identical authoring digests equal: " << (stable ? "yes" : "no") << '\n';
  if (!stable) {
    return 5;
  }

  // (b) The unused pin is deployment authority, not plan semantics: compiling
  // without it yields the same digest (and the same canonical JSON).
  const holoscan::ExecutionPlan unpinned_plan = holoscan::compile(second);
  if (!unpinned_plan.ok()) {
    std::cerr << unpinned_plan.json() << '\n';
    return 5;
  }
  const bool pin_neutral =
      unpinned_plan.digest() == plan.digest() && unpinned_plan.json() == plan.json();
  std::cout << "[digest] unused pin changes digest: " << (pin_neutral ? "no" : "yes") << '\n';
  if (!pin_neutral) {
    return 5;
  }

  // (c) A small tuning edit — faster tick, deeper queue — must move the digest.
  holoscan::Graph edited{"plan-artifacts"};
  build_pipeline(edited, 5ms, 8U);
  const holoscan::ExecutionPlan edited_plan = holoscan::compile(edited, pinned_options());
  if (!edited_plan.ok()) {
    std::cerr << edited_plan.json() << '\n';
    return 6;
  }
  std::cout << "[digest] edited compile: " << digest_hex(edited_plan.digest()) << '\n';
  const bool moved = !(edited_plan.digest() == plan.digest());
  std::cout << "[digest] edited graph digest differs: " << (moved ? "yes" : "no") << '\n';
  if (!moved) {
    return 6;
  }

  // --- 5. emit_artifacts twice: atomic, manifest-last, idempotent. ----------
  const std::filesystem::path root = workspace / "bundle";

  const auto first_emit = holoscan::emit_artifacts(
      plan, holoscan::ArtifactEmitOptions{.root = root, .deployment_id = "plan-artifacts-demo"});
  if (!first_emit) {
    std::cerr << "first emit failed: " << holoscan::error_code_name(first_emit.error().code)
              << '\n';
    return 7;
  }
  const bool first_healthy = first_emit->changed &&
                             std::filesystem::exists(first_emit->catalog_path) &&
                             std::filesystem::exists(first_emit->manifest_path) &&
                             first_emit->catalog_digest.starts_with("sha256:");
  std::cout << "[artifacts] first emit:  changed=" << (first_emit->changed ? "yes" : "no")
            << " catalog=" << first_emit->catalog_path.filename().string()
            << " manifest=" << first_emit->manifest_path.filename().string() << '\n';
  std::cout << "[artifacts] catalog_digest=" << first_emit->catalog_digest << '\n';
  if (!first_healthy) {
    return 7;
  }

  const auto second_emit = holoscan::emit_artifacts(
      plan, holoscan::ArtifactEmitOptions{.root = root, .deployment_id = "plan-artifacts-demo"});
  if (!second_emit) {
    std::cerr << "second emit failed: " << holoscan::error_code_name(second_emit.error().code)
              << '\n';
    return 8;
  }
  const bool idempotent =
      !second_emit->changed && second_emit->catalog_digest == first_emit->catalog_digest;
  std::cout << "[artifacts] second emit: changed=" << (second_emit->changed ? "yes" : "no")
            << " digest identical: "
            << (second_emit->catalog_digest == first_emit->catalog_digest ? "yes" : "no") << '\n';
  if (!idempotent) {
    return 8;
  }

  std::cout << "plan_artifacts example complete: accessors, digest, and idempotent artifacts "
               "verified\n";
  return 0;
}
