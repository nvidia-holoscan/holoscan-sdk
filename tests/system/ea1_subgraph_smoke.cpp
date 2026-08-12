/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: a two-level Graph hierarchy (SensorPipeline embeds FusionGraph as
// "fusion-stage") is authored with the correct vertex/edge/binding/region counts
// and produces a matching canonical compiler handoff — proving embed<> wires
// subgraph interface bindings and region records end-to-end.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>

#include <holoscan/core/graph.hpp>

#include "core/graph_compiler_access.hpp"

namespace {

// Emits a uint32_t token on each compute call; models a single-channel sensor feed.
class SensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output;
};

// Merges two uint32_t sensor streams (camera + IMU) into a single fused output token.
class FusionOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera, "camera");
    spec.input(imu, "imu");
    spec.output(fused, "fused");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> camera;
  holoscan::Input<std::uint32_t> imu;
  holoscan::Output<std::uint32_t> fused;
};

// Terminal consumer of the fused stream; discards data — topology shape is what
// this smoke verifies, not data values.
class FusedSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

// Reusable subgraph wrapping FusionOperator; exposes camera/imu as graph-level
// inputs and fused as a graph-level output so a parent can wire them without
// knowing the internal operator name.  Also declares a bounded-memory region.
// REQ: GRAPH-021
class FusionGraph final : public holoscan::Graph {
 public:
  holoscan::Input<std::uint32_t> camera{"camera"};
  holoscan::Input<std::uint32_t> imu{"imu"};
  holoscan::Output<std::uint32_t> fused{"fused"};

 protected:
  void compose() override {
    auto fusion = op<FusionOperator>("fusion");
    export_input(camera, fusion->camera);
    export_input(imu, fusion->imu);
    export_output(fused, fusion->fused);
    region("fusion-critical")
        .add(fusion)
        .constrain(holoscan::RegionConstraint::bounded_memory())
        .profile(holoscan::Profile::kRobotics);
  }
};

// Root graph that embeds FusionGraph as "fusion-stage" and connects two
// SensorSource operators (camera, imu) to it, with a FusedSink at the output.
// REQ: GRAPH-021
class SensorPipeline final : public holoscan::Graph {
 public:
  SensorPipeline() : Graph("sensor-pipeline") {}

 protected:
  void compose() override {
    auto camera = op<SensorSource>("camera");
    auto imu = op<SensorSource>("imu");
    auto fusion = embed<FusionGraph>("fusion-stage");
    auto sink = op<FusedSink>("sink");

    if (!fusion.graph_valid() || !fusion) {
      throw holoscan::RuntimeError(
          holoscan::Error{holoscan::ErrorCode::kFailure, "SUBGRAPH_HANDLE_INVALID"});
    }
    const auto fusion_path = fusion.path();
    if (!fusion_path || fusion_path->segments().size() != 1U ||
        fusion_path->segments().front() != "fusion-stage") {
      throw holoscan::RuntimeError(
          holoscan::Error{holoscan::ErrorCode::kFailure, "SUBGRAPH_HANDLE_PATH_INVALID"});
    }
    (void)add_flow(camera->output, fusion->camera);
    (void)add_flow(imu->output, fusion->imu);
    (void)add_flow(fusion->fused, sink->input);
  }
};

void print_graph_path(const std::optional<holoscan::GraphInstancePath>& path) {
  if (!path) {
    std::cout << "_root";
    return;
  }
  bool first = true;
  for (const auto& segment : path->segments()) {
    if (!first)
      std::cout << '/';
    std::cout << segment;
    first = false;
  }
}

void print_hash(const holoscan::GraphAuthoringHash& hash) {
  const auto flags = std::cout.flags();
  const auto fill = std::cout.fill();
  std::cout << std::hex << std::setfill('0');
  for (const std::byte byte : hash.value) {
    std::cout << std::setw(2) << std::to_integer<unsigned int>(byte);
  }
  std::cout.flags(flags);
  std::cout.fill(fill);
}

// Emit stable pointer-free authored snapshot facts without the process-local
// GraphInstanceId so fresh-process output remains comparable.
// REQ: GRAPH-022
void print_stable_authored_snapshot(const holoscan::GraphView& view) {
  std::cout << "hash:";
  print_hash(view.identity().hash);
  std::cout << '\n';

  for (const auto& graph : view.graphs()) {
    std::cout << "graph:" << graph.id.value << ':' << (graph.parent ? graph.parent->value : 0U)
              << ':';
    print_graph_path(graph.path);
    std::cout << ':' << graph.name << '\n';
  }
  for (const auto& vertex : view.vertices()) {
    std::cout << "vertex:" << vertex.id.value << ':' << vertex.graph.value << ':'
              << static_cast<unsigned int>(vertex.kind) << ':' << vertex.name << '\n';
  }
  for (const auto& port : view.ports()) {
    std::cout << "port:" << port.id.value << ':' << port.owner.value << ':'
              << static_cast<unsigned int>(port.owner_kind) << ':'
              << static_cast<unsigned int>(port.direction) << ':' << port.name << '\n';
  }
  for (const auto& edge : view.edges()) {
    std::cout << "edge:" << edge.id.value << ':' << edge.source.value << ':' << edge.target.value
              << '\n';
  }
  for (const auto& binding : view.interface_bindings()) {
    std::cout << "interface:" << binding.public_port.value << ':' << binding.internal_port.value
              << ':' << static_cast<unsigned int>(binding.direction) << '\n';
  }
  for (const auto& region : view.regions()) {
    std::cout << "region:" << region.id.value << ':' << region.graph.value << ':' << region.name
              << ':' << (region.profile ? static_cast<unsigned int>(*region.profile) : 255U)
              << '\n';
  }
}

// Emit records obtained from the private canonical compiler handoff, not a
// reconstruction from the public authored view.
// REQ: GRAPH-022
void print_stable_canonical_handoff(holoscan::Graph& graph) {
  const auto snapshot = holoscan::detail::GraphCompilerAccess::canonical_snapshot(graph);
  std::cout << "canonical-hash:";
  print_hash(snapshot->identity.hash);
  std::cout << '\n';
  std::cout << "canonical-counts:" << snapshot->graphs.size() << ':' << snapshot->vertices.size()
            << ':' << snapshot->ports.size() << ':' << snapshot->edges.size() << ':'
            << snapshot->interface_bindings.size() << ':' << snapshot->external_topics.size() << ':'
            << snapshot->readiness_sources.size() << ':' << snapshot->regions.size() << ':'
            << snapshot->clocks.size() << ':' << snapshot->clock_conversions.size() << ':'
            << snapshot->lifecycle_dependencies.size() << ':'
            << snapshot->lifecycle_restart_permissions.size() << '\n';

  for (const auto& record : snapshot->graphs) {
    std::cout << "canonical-graph:" << record.id.value << ':'
              << (record.parent ? record.parent->value : 0U) << ':';
    print_graph_path(record.path);
    std::cout << ':' << record.name << '\n';
  }
  for (const auto& record : snapshot->vertices) {
    std::cout << "canonical-vertex:" << record.id.value << ':' << record.graph.value << ':'
              << static_cast<unsigned int>(record.kind) << ':' << record.name << '\n';
  }
  for (const auto& record : snapshot->ports) {
    std::cout << "canonical-port:" << record.id.value << ':' << record.owner.value << ':'
              << static_cast<unsigned int>(record.owner_kind) << ':'
              << static_cast<unsigned int>(record.direction) << ':' << record.name << '\n';
  }
  for (const auto& record : snapshot->edges) {
    std::cout << "canonical-edge:" << record.id.value << ':' << record.source.value << ':'
              << record.target.value << '\n';
  }
  for (const auto& record : snapshot->interface_bindings) {
    std::cout << "canonical-interface:" << record.public_port.value << ':'
              << record.internal_port.value << ':' << static_cast<unsigned int>(record.direction)
              << '\n';
  }
  for (const auto& record : snapshot->regions) {
    std::cout << "canonical-region:" << record.id.value << ':' << record.graph.value << ':'
              << record.name << ':'
              << (record.profile ? static_cast<unsigned int>(*record.profile) : 255U) << '\n';
    for (const auto member : record.members) {
      std::cout << "canonical-region-member:" << record.id.value << ':' << member.value << '\n';
    }
    for (const auto member : record.graph_members) {
      std::cout << "canonical-region-graph:" << record.id.value << ':' << member.value << '\n';
    }
    for (const auto& constraint : record.constraints) {
      std::cout << "canonical-region-constraint:" << record.id.value << ':'
                << static_cast<unsigned int>(constraint.kind()) << ':'
                << constraint.latency().count() << '\n';
    }
    for (const auto hint : record.hints) {
      std::cout << "canonical-region-hint:" << record.id.value << ':'
                << static_cast<unsigned int>(hint) << '\n';
    }
  }
}

}  // namespace

int main() {
  // camera(SensorSource) ─┐
  //                        ├─► [fusion-stage] fusion(FusionOperator) ─► sink(FusedSink)
  // imu(SensorSource) ────┘
  SensorPipeline pipeline;
  const holoscan::GraphView view = pipeline.view();
  if (view.graphs().size() != 2U || view.vertices().size() != 7U || view.edges().size() != 3U ||
      view.interface_bindings().size() != 3U || view.regions().size() != 1U) {
    return 1;  // exit 1: authored view must have 2 graphs, 7 vertices, 3 edges,
               //         3 interface bindings, and 1 region
  }

  const auto child =
      std::find_if(view.graphs().begin(), view.graphs().end(), [](const auto& graph) {
        return graph.path && graph.path->segments().size() == 1U &&
               graph.path->segments().front() == "fusion-stage";
      });
  const auto root = std::find_if(view.graphs().begin(), view.graphs().end(), [](const auto& graph) {
    return !graph.path && !graph.parent;
  });
  if (child == view.graphs().end() || root == view.graphs().end() ||
      child->parent.value_or(holoscan::GraphId{}) != root->id) {
    return 2;  // exit 2: "fusion-stage" child graph must exist and be parented to root
  }

  print_stable_authored_snapshot(view);
  print_stable_canonical_handoff(pipeline);
  return 0;
}
