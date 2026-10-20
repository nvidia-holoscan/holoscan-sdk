/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>

#include "../support/optional_assertions.hpp"
#include "artifact_test_support.hpp"
#include "compile_test_graph.hpp"
#include "compiler/closed_island_activation_plan.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/cuda_execution_plan_validation.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_generation.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/diagnostic_registry.hpp"
#include "core/graph_compiler_access.hpp"

namespace holoscan {

struct ClosedIslandPartitionPod {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<ClosedIslandPartitionPod> {
  static constexpr std::string_view name = "holoscan.test.ClosedIslandPartitionPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ClosedIslandPartitionPod> {
  static constexpr auto fields = pod_fields(pod_field("value", &ClosedIslandPartitionPod::value,
                                                      offsetof(ClosedIslandPartitionPod, value)));
};

namespace {

class PartitionJoinOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(2U);
    spec.input(second, "second").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
  Output<std::uint32_t> output;
};

class ClosedIslandSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext& context) override { return {}; }

  Output<ClosedIslandPartitionPod> output;
};

class ScalarClockSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class ClosedIslandSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<ClosedIslandPartitionPod> input;
};

class IncludeAllEmbeddedGraph final : public Graph {
 protected:
  void compose() override { (void)op<test::CompileTestOperator>("nested"); }
};

// Declares an isolated CUDA-device Tensor producer so compiler tests can require compute streams
// without running an operator. Used by
// CompilePartitionAuthoring.GreenContextRecordsStructuredResidualAndIrregularTail.
class PartitionTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U}},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Two explicitly placed CUDA-device Tensor outputs exercise the planner's exact, canonical
// multi-device obligation set without requiring Runtime execution.
class PartitionMultiDeviceTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec tensor{
        .representation = {.memory_kind = MemoryKind::kCudaDevice,
                           .dtype = DLDataType{kDLUInt, 8U, 1U}},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
    spec.output(first, "first").max_emits_per_compute(1U).produces_tensor(tensor);
    spec.output(second, "second").max_emits_per_compute(1U).produces_tensor(tensor);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext&) override { return {}; }

  Output<Tensor> first;
  Output<Tensor> second;
};

class AtomicMultiDeviceTensorGraph final : public Graph {
 protected:
  void compose() override {
    test::add_compile_test_default_clock(*this);
    (void)op<PartitionMultiDeviceTensorSourceOperator>("source");
  }
};

// REQ: COMPILE-075
TEST(CompilePartitionAuthoring, AtomicInstanceMultiOutputDeviceDomainsAgreeAcrossCoverModes) {
  for (const bool automatic : {false, true}) {
    for (const bool conflict : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "automatic=" << automatic << " conflict=" << conflict);
      Graph graph{"root"};
      test::add_compile_test_default_clock(graph);
      graph.partition("owner").add(graph.embed<AtomicMultiDeviceTensorGraph>("child"));
      if (automatic) {
        (void)graph.op<test::CompileTestOperator>("peer");
        graph.auto_partition_remaining();
      }
      CompileOptions options;
      options.deployment.bind_tensor_output_device({"child/source", "first", DeviceId{0}})
          .bind_tensor_output_device({"child/source", "second", DeviceId{conflict ? 1 : 0}});
      const auto plan = compile(graph, std::move(options));
      if (conflict) {
        EXPECT_FALSE(plan.ok());
        EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
        EXPECT_TRUE(plan.execution_partitions().empty());
        EXPECT_TRUE(plan.execution_lanes().empty());
      } else {
        EXPECT_TRUE(plan.ok()) << plan.json();
      }
    }
  }
}

// Keeps the only graph edge scalar while two disconnected Tensor outputs contribute an exact
// intrinsic multi-device conflict to the source vertex.
class PartitionMultiDeviceScalarSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec tensor{
        .representation = {.memory_kind = MemoryKind::kCudaDevice,
                           .dtype = DLDataType{kDLUInt, 8U, 1U}},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
    spec.output(first, "first").max_emits_per_compute(1U).produces_tensor(tensor);
    spec.output(second, "second").max_emits_per_compute(1U).produces_tensor(tensor);
    spec.output(scalar, "scalar").max_emits_per_compute(1U);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext&) override { return {}; }

  Output<Tensor> first;
  Output<Tensor> second;
  Output<std::uint32_t> scalar;
};

[[nodiscard]] ExecutionPlan compile_zero_binding_tensor_graph(
    std::optional<CudaComputeStreamPolicy> policy = std::nullopt) {
  Graph graph{"ordinary-tensor-stream"};
  test::add_compile_test_default_clock(graph);
  graph.op<PartitionTensorSourceOperator>("source");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput;
      });
  if (output == snapshot->ports.end()) {
    return {};
  }
  const std::array devices{detail::TensorDeviceBindingRecord{
      .port = PortId{output->id.value},
      .device = DeviceId{0},
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  CompileOptions options;
  if (policy) {
    options.cuda_execution.compute_stream_policy = *policy;
  }
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Declares a CUDA-device Tensor consumer used to exercise stream planning across a partition edge.
// Used by CompilePartitionAuthoring.GreenContextSeparatesResidualPartitionAndServiceStreams.
class PartitionTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U}},
        });
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

// Forwards the received Tensor without declaring a new pool. Downstream instances therefore
// become CUDA-active only when explicit Green Context placement supplies their final bindings.
class PartitionPassThroughTensorOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation representation{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    };
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = representation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = representation,
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Two individually representable CUDA pools whose same-device aggregate exceeds uint32_t.
// Compilation must retain the diagnosed rejected plan without granting runtime authority.
class PartitionAggregateOverflowTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec tensor{
        .representation = {.memory_kind = MemoryKind::kCudaDevice,
                           .dtype = DLDataType{kDLUInt, 8U, 1U}},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 1U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
    constexpr std::uint32_t kMaximumEmits = 1U << 30U;
    spec.output(first, "first").max_emits_per_compute(kMaximumEmits).produces_tensor(tensor);
    spec.output(second, "second").max_emits_per_compute(kMaximumEmits).produces_tensor(tensor);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext&) override { return {}; }

  Output<Tensor> first;
  Output<Tensor> second;
};

class PartitionTensorSinkGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<Tensor> input{"input"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto sink = op<PartitionTensorSinkOperator>("sink");
    export_input(input, sink->input);
  }

 private:
  ClockKey clock_import_;
};

class PartitionTensorTransformOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U}},
        });
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U}},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

[[nodiscard]] detail::ProcessPlacementBinding process_placement(std::string_view source_group,
                                                                std::string_view target_group) {
  constexpr std::array source{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array target{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("source",
                         detail::ProcessPlacement{.members = source, .launch_group = source_group})
      .bind_process_role("target",
                         detail::ProcessPlacement{.members = target, .launch_group = target_group});
  return placement;
}

[[nodiscard]] detail::ProcessPlacementBinding external_process_placement(
    std::string_view endpoint) {
  const std::array external{
      detail::GraphObjectSelector{.external_topic = endpoint},
  };
  detail::ProcessPlacementBinding placement;
  placement.bind_process_role("external-endpoint",
                              detail::ProcessPlacement{
                                  .isolation = detail::ProcessIsolation::kExternalProcess,
                                  .members = external,
                                  .launch_group = "external-endpoint",
                              });
  return placement;
}

[[nodiscard]] ExecutionPlan compile_manual_process_partition(std::string_view source_group,
                                                             std::string_view target_group) {
  Graph graph{"manual-process-partition"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
  graph.partition("pipeline").add(source).add(target);
  return detail::compile_with_process_placement(
      graph, CompileOptions{}, process_placement(source_group, target_group));
}

struct CanonicalPartitionEvidence {
  std::vector<std::string> partitions;
  std::vector<std::string> schedule_edges;
  std::vector<std::string> placements;
  std::vector<std::string> probes;
  std::vector<std::string> lanes;

  friend bool operator==(const CanonicalPartitionEvidence&,
                         const CanonicalPartitionEvidence&) = default;
};

[[nodiscard]] CanonicalPartitionEvidence partition_evidence(const ExecutionPlan& plan) {
  CanonicalPartitionEvidence result;
  for (const auto& partition : plan.execution_partitions()) {
    std::string record = std::to_string(partition.id.value) + ":" +
                         partition.authored_name.value_or("<implicit>") + ":" +
                         std::to_string(partition.execution_lane);
    for (const VertexId member : partition.members) {
      record += ":" + std::to_string(member.value);
    }
    result.partitions.push_back(std::move(record));
  }
  for (const auto& edge : plan.schedule_edges()) {
    result.schedule_edges.push_back(std::to_string(edge.edge.value) + ":" +
                                    std::to_string(edge.source.value) + ":" +
                                    std::to_string(edge.destination.value));
  }
  for (const auto& placement : plan.partition_placements()) {
    result.placements.push_back(std::to_string(placement.partition.value) + ":" +
                                placement.authored_name + ":" +
                                std::to_string(placement.execution_lane) + ":" +
                                std::to_string(static_cast<std::uint32_t>(placement.disposition)));
  }
  for (const auto& probe : plan.placement_probes()) {
    result.probes.push_back(std::to_string(probe.partition.value) + ":" + probe.probe_id + ":" +
                            std::to_string(static_cast<std::uint32_t>(probe.status)) + ":" +
                            probe.evidence_id + ":" + std::to_string(probe.reason.value));
  }
  for (const auto& lane : plan.execution_lanes()) {
    result.lanes.push_back(std::to_string(lane.execution_lane) + ":" +
                           std::to_string(lane.partition.value));
  }
  return result;
}

[[nodiscard]] ExecutionPlan compile_canonical_manual_partitions(bool reverse) {
  Graph graph{"canonical-manual-partitions"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  const auto third = graph.op<test::CompileTestOperator>("third");
  const auto fourth = graph.op<test::CompileTestOperator>("fourth");
  graph.add_flow(first->output, second->input);
  graph.add_flow(second->output, third->input);
  graph.add_flow(third->output, fourth->input);

  CompileOptions options;
  if (reverse) {
    graph.partition("beta").add(fourth).add(third);
    graph.partition("alpha").add(second).add(first);
    options.deployment.bind_thread_placement("beta", ThreadPlacement{});
    options.deployment.bind_thread_placement("alpha", ThreadPlacement{});
  } else {
    graph.partition("alpha").add(first).add(second);
    graph.partition("beta").add(third).add(fourth);
    options.deployment.bind_thread_placement("alpha", ThreadPlacement{});
    options.deployment.bind_thread_placement("beta", ThreadPlacement{});
  }
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan compile_manual_fusion_partitions(bool reverse) {
  Graph graph{"manual-fusion-groups"};
  test::add_compile_test_default_clock(graph);
  const auto camera = graph.op<test::CompileTestOperator>("camera");
  const auto camera_pre = graph.op<test::CompileTestOperator>("camera-pre");
  const auto imu = graph.op<test::CompileTestOperator>("imu");
  const auto imu_pre = graph.op<test::CompileTestOperator>("imu-pre");
  const auto fusion = graph.op<PartitionJoinOperator>("fusion");
  const auto display = graph.op<test::CompileTestOperator>("display");
  const auto visualization = graph.op<test::CompileTestOperator>("visualization");
  const auto inference = graph.op<test::CompileTestOperator>("inference");
  const auto storage = graph.op<test::CompileTestOperator>("storage");
  graph.add_flow(camera->output, camera_pre->input);
  graph.add_flow(camera_pre->output, fusion->first);
  graph.add_flow(imu->output, imu_pre->input);
  graph.add_flow(imu_pre->output, fusion->second);
  graph.add_flow(fusion->output, display->input);
  graph.add_flow(display->output, visualization->input);
  graph.add_flow(fusion->output, inference->input);
  graph.add_flow(inference->output, storage->input);

  if (reverse) {
    graph.partition("inference-storage").add(storage).add(inference);
    graph.partition("fusion-display").add(visualization).add(display).add(fusion);
    graph.partition("imu").add(imu_pre).add(imu);
    graph.partition("camera").add(camera_pre).add(camera);
  } else {
    graph.partition("camera").add(camera).add(camera_pre);
    graph.partition("imu").add(imu).add(imu_pre);
    graph.partition("fusion-display").add(fusion).add(display).add(visualization);
    graph.partition("inference-storage").add(inference).add(storage);
  }
  return compile(graph);
}

// Graph topology (each node is in the named execution partition):
//   source(alpha) --> target(beta)
[[nodiscard]] ExecutionPlan compile_green_context_partitions(
    bool reverse_bindings, std::uint32_t alpha_sms = 16U, std::uint32_t beta_sms = 24U,
    PlacementProbeStatus probe_status = PlacementProbeStatus::kPassed,
    Profile profile = Profile::kStandard, std::uint32_t device_sms = 80U) {
  Graph graph{"manual-green-context-partitions"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
  graph.partition("alpha").add(source);
  graph.partition("beta").add(target);

  CompileOptions options;
  options.profile = profile;
  if (reverse_bindings) {
    options.deployment
        .bind_cuda_green_context("beta", {.device = DeviceId{0}, .sm_count = beta_sms})
        .bind_cuda_green_context("alpha", {.device = DeviceId{0}, .sm_count = alpha_sms});
  } else {
    options.deployment
        .bind_cuda_green_context("alpha", {.device = DeviceId{0}, .sm_count = alpha_sms})
        .bind_cuda_green_context("beta", {.device = DeviceId{0}, .sm_count = beta_sms});
  }
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.0",
      .total_sm_count = device_sms,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = probe_status,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.cuda_green_context_capabilities = capabilities;
  std::optional<detail::RuntimePackageCapabilityManifest> profile_package;
  if (profile == Profile::kRealTime) {
    profile_package.emplace(detail::runtime_package_capability_manifest());
    profile_package->offers.push_back(detail::RuntimePackageCapabilityOffer{
        .id = "holoscan.profile.real_time.bounded_hot_path",
        .version = 1U,
        .evidence_id = "holoscan.profile.real_time.bounded_hot_path/v1",
    });
    const auto sealed = detail::seal_runtime_package_capability_manifest(*profile_package);
    EXPECT_TRUE(sealed) << (sealed ? std::string{} : runtime_error_message(sealed.error()));
    const auto package_view =
        detail::validate_runtime_package_capability_manifest(*profile_package);
    EXPECT_TRUE(package_view) << (package_view ? std::string{}
                                               : runtime_error_message(package_view.error()));
    if (package_view) {
      evidence.package_capabilities = *package_view;
    }
  }
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (each isolated partition is assigned to a distinct physical device):
//   device-zero   device-one
[[nodiscard]] ExecutionPlan compile_green_context_partitions_on_two_devices() {
  Graph graph{"manual-green-context-two-devices"};
  test::add_compile_test_default_clock(graph);
  const auto device_zero = graph.op<test::CompileTestOperator>("device-zero");
  const auto device_one = graph.op<test::CompileTestOperator>("device-one");
  graph.partition("device-zero").add(device_zero);
  graph.partition("device-one").add(device_one);

  CompileOptions options;
  options.deployment
      .bind_cuda_green_context("device-zero", {.device = DeviceId{0}, .sm_count = 80U})
      .bind_cuda_green_context("device-one", {.device = DeviceId{1}, .sm_count = 80U});
  const std::array capabilities{
      detail::CudaGreenContextCapabilityEvidence{
          .device = DeviceId{0},
          .deployment_device_id = "gpu.test.uuid.0",
          .total_sm_count = 80U,
          .min_sm_partition_size = 8U,
          .sm_coscheduled_alignment = 8U,
          .max_structured_context_count = 10U,
          .evidence_id = "test.cuda.green_context.capability.device-0/v1",
          .probe_status = PlacementProbeStatus::kPassed,
      },
      detail::CudaGreenContextCapabilityEvidence{
          .device = DeviceId{1},
          .deployment_device_id = "gpu.test.uuid.1",
          .total_sm_count = 80U,
          .min_sm_partition_size = 8U,
          .sm_coscheduled_alignment = 8U,
          .max_structured_context_count = 10U,
          .evidence_id = "test.cuda.green_context.capability.device-1/v1",
          .probe_status = PlacementProbeStatus::kPassed,
      },
  };
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.cuda_green_context_capabilities = capabilities;
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (all operators are isolated CUDA Tensor sources):
//   alpha   beta   [residual-a]   [residual-b]
[[nodiscard]] ExecutionPlan compile_green_context_residual_partitions(
    std::uint32_t residual_partition_count) {
  Graph graph{"green-context-residual-partitions"};
  test::add_compile_test_default_clock(graph);
  const auto alpha = graph.op<PartitionTensorSourceOperator>("alpha");
  const auto beta = graph.op<PartitionTensorSourceOperator>("beta");
  graph.partition("alpha").add(alpha);
  graph.partition("beta").add(beta);
  if (residual_partition_count >= 1U) {
    const auto residual_a = graph.op<PartitionTensorSourceOperator>("residual-a");
    graph.partition("residual-a").add(residual_a);
  }
  if (residual_partition_count >= 2U) {
    const auto residual_b = graph.op<PartitionTensorSourceOperator>("residual-b");
    graph.partition("residual-b").add(residual_b);
  }

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical test snapshot unavailable"}};
  }
  std::vector<detail::TensorDeviceBindingRecord> devices;
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput) {
      devices.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value},
          .device = DeviceId{0},
      });
    }
  }
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.irregular",
      .total_sm_count = 82U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_cuda_green_context("alpha", {.device = DeviceId{0}, .sm_count = 16U})
      .bind_cuda_green_context("beta", {.device = DeviceId{0}, .sm_count = 24U});
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (all operators are isolated):
//   dedicated   [residual CUDA Tensor source]
[[nodiscard]] ExecutionPlan compile_single_green_context(
    std::uint32_t exact_sm_count, std::uint32_t total_sm_count, bool add_unbound_cuda_partition,
    DeviceId device = DeviceId{0}, std::string_view bound_partition = "dedicated",
    bool duplicate_binding = false, std::optional<std::uint32_t> max_context_count = std::nullopt,
    std::optional<Profile> residual_profile = std::nullopt) {
  Graph graph{"single-green-context"};
  test::add_compile_test_default_clock(graph);
  const auto dedicated = graph.op<test::CompileTestOperator>("dedicated");
  graph.partition("dedicated").add(dedicated);
  if (add_unbound_cuda_partition) {
    const auto residual = graph.op<PartitionTensorSourceOperator>("residual");
    graph.partition("residual").add(residual);
    if (residual_profile) {
      graph.region("residual-profile").add(residual).profile(*residual_profile);
    }
  }
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical test snapshot unavailable"}};
  }
  std::vector<detail::TensorDeviceBindingRecord> devices;
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput && port.setup.tensor_layout) {
      devices.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value},
          .device = device,
      });
    }
  }
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = device,
      .deployment_device_id = "gpu.test.uuid.single",
      .total_sm_count = total_sm_count,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = max_context_count.value_or(total_sm_count / 8U),
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_cuda_green_context(bound_partition,
                                             {.device = device, .sm_count = exact_sm_count});
  if (duplicate_binding) {
    options.deployment.bind_cuda_green_context(bound_partition,
                                               {.device = device, .sm_count = exact_sm_count});
  }
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (both operators are isolated):
//   dedicated (exact Green Context anchor)
//   residual  (automatic CUDA Tensor remainder)
[[nodiscard]] ExecutionPlan compile_hybrid_green_context_remainder() {
  Graph graph{"hybrid-green-context-remainder"};
  test::add_compile_test_default_clock(graph);
  const auto dedicated = graph.op<test::CompileTestOperator>("dedicated");
  (void)graph.op<PartitionTensorSourceOperator>("residual");
  graph.partition("dedicated").add(dedicated);
  graph.auto_partition_remaining();
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical test snapshot unavailable"}};
  }
  std::vector<detail::TensorDeviceBindingRecord> devices;
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput && port.setup.tensor_layout) {
      devices.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value},
          .device = DeviceId{0},
      });
    }
  }
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.hybrid",
      .total_sm_count = 80U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_thread_placement("dedicated",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.deployment.bind_cuda_green_context("dedicated", {.device = DeviceId{0}, .sm_count = 16U});
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (source and sink are in separate execution partitions):
//   source --> sink
[[nodiscard]] ExecutionPlan compile_green_context_tensor_service(
    CudaComputeStreamPolicy policy = CudaComputeStreamPolicy::kPerExecutionPartition) {
  Graph graph{"green-context-tensor-service"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  const auto sink = graph.op<PartitionTensorSinkOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical test snapshot unavailable"}};
  }
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput;
      });
  if (output == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical Tensor output unavailable"}};
  }
  const std::array devices{detail::TensorDeviceBindingRecord{
      .port = PortId{output->id.value},
      .device = DeviceId{0},
  }};
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.service",
      .total_sm_count = 82U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.cuda_execution.compute_stream_policy = policy;
  options.deployment.bind_cuda_green_context("source", {.device = DeviceId{0}, .sm_count = 16U});
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (one Operator per explicit Green Context partition):
//   source --> relay-a --> relay-b --> relay-c --> sink
// Only the source owns a Tensor pool; the relays forward its original backing.
[[nodiscard]] ExecutionPlan compile_green_context_forwarding_chain(
    CudaComputeStreamPolicy policy, std::optional<DeviceId> relay_device = std::nullopt) {
  Graph graph{"green-context-forwarding-chain"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  const auto relay_a = graph.op<PartitionPassThroughTensorOperator>("relay-a");
  const auto relay_b = graph.op<PartitionPassThroughTensorOperator>("relay-b");
  const auto relay_c = graph.op<PartitionPassThroughTensorOperator>("relay-c");
  const auto sink = graph.op<PartitionTensorSinkOperator>("sink");
  graph.add_flow(source->output, relay_a->input);
  graph.add_flow(relay_a->output, relay_b->input);
  graph.add_flow(relay_b->output, relay_c->input);
  graph.add_flow(relay_c->output, sink->input);
  graph.partition("source").add(source);
  graph.partition("relay-a").add(relay_a);
  graph.partition("relay-b").add(relay_b);
  graph.partition("relay-c").add(relay_c);
  graph.partition("sink").add(sink);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical Tensor chain unavailable"}};
  }
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
        return source_vertex != snapshot->vertices.end() && port.owner == source_vertex->id &&
               port.direction == PortDirection::kOutput;
      });
  if (output == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "source Tensor output unavailable"}};
  }
  const std::array devices{detail::TensorDeviceBindingRecord{
      .port = PortId{output->id.value},
      .device = DeviceId{0},
  }};
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.forwarding",
      .total_sm_count = 40U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 5U,
      .evidence_id = "test.cuda.green_context.capability.forwarding/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.cuda_execution.compute_stream_policy = policy;
  if (relay_device) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "relay-a", .output_port = "output", .device = *relay_device});
  }
  for (const std::string_view partition : {"source", "relay-a", "relay-b", "relay-c", "sink"}) {
    options.deployment.bind_cuda_green_context(
        partition, CudaGreenContextPlacement{.device = DeviceId{0}, .sm_count = 8U});
  }
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

// Graph topology (both operators are fused into one execution partition):
//   source --> sink
[[nodiscard]] ExecutionPlan compile_green_context_fused_tensor_partition() {
  Graph graph{"green-context-fused-tensor-partition"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  const auto sink = graph.op<PartitionTensorSinkOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.partition("fused").add(source).add(sink);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical test snapshot unavailable"}};
  }
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput && port.setup.tensor_layout;
      });
  if (output == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "canonical Tensor output unavailable"}};
  }
  const std::array devices{detail::TensorDeviceBindingRecord{
      .port = PortId{output->id.value},
      .device = DeviceId{0},
  }};
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.fused",
      .total_sm_count = 80U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_cuda_green_context("fused", {.device = DeviceId{0}, .sm_count = 16U});
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

[[nodiscard]] std::vector<std::string> profile_capacity_evidence(const ExecutionPlan& plan) {
  std::vector<std::string> result;
  for (const auto& capacity : plan.profile_manifest().derived_capacities) {
    result.push_back(capacity.object_id + ":" + capacity.capacity_id + ":" +
                     std::to_string(capacity.limit) + ":" + capacity.unit);
  }
  return result;
}

// REQ: COMPILE-058
// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextBindingsAdmitCanonicalCpuOnlyPartitions) {
  // Key setup: compile two CPU-only partitions with 16-SM and 24-SM Green Context bindings.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: lowering emits two dedicated assignments, one structured residual, and two
  // partition streams while preserving the capability evidence.
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);
  ASSERT_EQ(plan.cuda_green_context_assignments().size(), 2U);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 3U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 2U);
  EXPECT_EQ(plan.cuda_green_context_requests()[0].authored_name, "alpha");
  EXPECT_EQ(plan.cuda_green_context_requests()[0].requested.sm_count, 16U);
  const std::optional<CudaGreenContextPlacement> accepted_alpha{
      {.device = DeviceId{0}, .sm_count = 16U}};
  EXPECT_EQ(plan.cuda_green_context_requests()[0].accepted, accepted_alpha);
  EXPECT_EQ(plan.cuda_green_context_requests()[0].disposition, PlanDisposition::kAccepted);
  EXPECT_EQ(plan.cuda_green_context_requests()[1].authored_name, "beta");
  EXPECT_EQ(plan.cuda_green_context_assignments()[0].partition, ExecutionPartitionId{1U});
  EXPECT_EQ(plan.cuda_green_context_assignments()[1].partition, ExecutionPartitionId{2U});
  const auto& pool = plan.cuda_green_context_pools().front();
  EXPECT_EQ(pool.deployment_device_id, "gpu.test.uuid.0");
  EXPECT_EQ(pool.device_sm_count, 80U);
  EXPECT_EQ(pool.min_sm_partition_size, 8U);
  EXPECT_EQ(pool.sm_coscheduled_alignment, 8U);
  EXPECT_EQ(pool.structured_sm_quantum, 8U);
  EXPECT_EQ(pool.max_structured_context_count, 10U);
  EXPECT_EQ(pool.unprovisioned_sm_count, 0U);
  EXPECT_TRUE(pool.fallback_rejected);
  EXPECT_EQ(plan.cuda_green_context_resources()[0].accepted_sm_count, 16U);
  EXPECT_EQ(plan.cuda_green_context_resources()[0].role,
            CudaGreenContextResourceRole::kDedicatedPartition);
  EXPECT_EQ(plan.cuda_green_context_resources()[1].accepted_sm_count, 24U);
  EXPECT_EQ(plan.cuda_green_context_resources()[1].role,
            CudaGreenContextResourceRole::kDedicatedPartition);
  EXPECT_EQ(plan.cuda_green_context_resources()[2].accepted_sm_count, 40U);
  EXPECT_EQ(plan.cuda_green_context_resources()[2].role,
            CudaGreenContextResourceRole::kReservedResidual);
  EXPECT_EQ(plan.cuda_compute_streams()[0].partition, ExecutionPartitionId{1U});
  EXPECT_EQ(plan.cuda_compute_streams()[1].partition, ExecutionPartitionId{2U});
  EXPECT_TRUE(plan.tensor_device_resources().empty());
  EXPECT_TRUE(std::all_of(
      plan.placement_probes().begin(), plan.placement_probes().end(), [](const auto& probe) {
        return probe.probe_id != "cuda.green_context.split" ||
               (probe.status == PlacementProbeStatus::kPassed &&
                probe.evidence_id == "test.cuda.green_context.capability/v1");
      }));
  EXPECT_EQ(std::count_if(plan.diagnostics().begin(),
                          plan.diagnostics().end(),
                          [](const auto& diagnostic) {
                            return diagnostic.id ==
                                   detail::diagnostic_id("CUDA_GREEN_CONTEXT_INACTIVE_RESERVATION");
                          }),
            2);
  const auto materializations = plan.profile_manifest().materialization_obligations;
  EXPECT_EQ(std::count_if(materializations.begin(),
                          materializations.end(),
                          [](const auto& obligation) {
                            return obligation.strategy_id ==
                                   "holoscan.runtime.cuda_green_context_pool/v1";
                          }),
            plan.cuda_green_context_pools().size());
  EXPECT_EQ(std::count_if(materializations.begin(),
                          materializations.end(),
                          [](const auto& obligation) {
                            return obligation.strategy_id ==
                                   "holoscan.runtime.cuda_green_context/v1";
                          }),
            plan.cuda_green_context_resources().size());
}

// REQ: COMPILE-063
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, HybridGreenContextAnchorKeepsCompleteFinalOwnerJoins) {
  const ExecutionPlan plan = compile_hybrid_green_context_remainder();
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 1U);
  const auto anchor =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return partition.authored_name == "dedicated"; });
  ASSERT_NE(anchor, plan.execution_partitions().end());
  const auto automatic =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return !partition.authored_name.has_value(); });
  ASSERT_NE(automatic, plan.execution_partitions().end());
  ASSERT_EQ(plan.partition_placements().size(), 1U);
  const auto& thread_placement = plan.partition_placements().front();
  EXPECT_EQ(thread_placement.partition, anchor->id);
  EXPECT_EQ(thread_placement.authored_name, "dedicated");
  EXPECT_EQ(thread_placement.execution_lane, anchor->execution_lane);
  EXPECT_EQ(thread_placement.requested.wake_mode, WakeMode::kBusyPoll);
  ASSERT_TRUE(thread_placement.accepted.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(thread_placement.accepted->wake_mode, WakeMode::kBusyPoll);
  EXPECT_EQ(thread_placement.disposition, PlanDisposition::kAccepted);
  const auto anchor_lane = std::ranges::find(
      plan.execution_lanes(), anchor->execution_lane, &ExecutionLanePlanRecord::execution_lane);
  const auto automatic_lane = std::ranges::find(
      plan.execution_lanes(), automatic->execution_lane, &ExecutionLanePlanRecord::execution_lane);
  ASSERT_NE(anchor_lane, plan.execution_lanes().end());
  ASSERT_NE(automatic_lane, plan.execution_lanes().end());
  EXPECT_EQ(anchor_lane->partition, anchor->id);
  EXPECT_EQ(anchor_lane->accepted_placement.wake_mode, WakeMode::kBusyPoll);
  EXPECT_EQ(automatic_lane->partition, automatic->id);
  EXPECT_EQ(automatic_lane->accepted_placement.wake_mode, WakeMode::kSleepingWaitset);
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_assignments().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 2U);
  const auto& request = plan.cuda_green_context_requests().front();
  const auto& assignment = plan.cuda_green_context_assignments().front();
  const auto& pool = plan.cuda_green_context_pools().front();
  EXPECT_EQ(request.partition, anchor->id);
  EXPECT_EQ(request.execution_lane, anchor->execution_lane);
  EXPECT_EQ(assignment.partition, anchor->id);
  EXPECT_EQ(assignment.execution_lane, anchor->execution_lane);
  EXPECT_EQ(assignment.pool_id, pool.pool_id);
  const auto dedicated =
      std::find_if(plan.cuda_green_context_resources().begin(),
                   plan.cuda_green_context_resources().end(),
                   [&](const auto& resource) {
                     return resource.green_context_id == assignment.green_context_id;
                   });
  ASSERT_NE(dedicated, plan.cuda_green_context_resources().end());
  EXPECT_EQ(dedicated->pool_id, pool.pool_id);
  EXPECT_EQ(dedicated->process_group, assignment.process_group);
  EXPECT_EQ(dedicated->device, assignment.device);
  EXPECT_EQ(dedicated->role, CudaGreenContextResourceRole::kDedicatedPartition);
  const auto residual = std::ranges::find(plan.cuda_green_context_resources(),
                                          CudaGreenContextResourceRole::kSharedServiceResidual,
                                          &CudaGreenContextResourcePlanRecord::role);
  ASSERT_NE(residual, plan.cuda_green_context_resources().end());
  EXPECT_EQ(std::count_if(plan.cuda_green_context_resources().begin(),
                          plan.cuda_green_context_resources().end(),
                          [](const auto& resource) {
                            return resource.role ==
                                   CudaGreenContextResourceRole::kSharedServiceResidual;
                          }),
            1);
  EXPECT_EQ(residual->pool_id, pool.pool_id);
  EXPECT_EQ(residual->process_group, pool.process_group);
  EXPECT_EQ(residual->device, pool.device);
  EXPECT_NE(residual->green_context_id, assignment.green_context_id);
  const auto anchor_stream =
      std::find_if(plan.cuda_compute_streams().begin(),
                   plan.cuda_compute_streams().end(),
                   [&](const auto& stream) { return stream.partition == anchor->id; });
  ASSERT_NE(anchor_stream, plan.cuda_compute_streams().end());
  EXPECT_EQ(anchor_stream->process_group, assignment.process_group);
  EXPECT_EQ(anchor_stream->device, assignment.device);
  EXPECT_EQ(anchor_stream->green_context_id, assignment.green_context_id);
  const auto automatic_stream =
      std::find_if(plan.cuda_compute_streams().begin(),
                   plan.cuda_compute_streams().end(),
                   [&](const auto& stream) { return stream.partition == automatic->id; });
  ASSERT_NE(automatic_stream, plan.cuda_compute_streams().end());
  EXPECT_EQ(std::count_if(plan.cuda_compute_streams().begin(),
                          plan.cuda_compute_streams().end(),
                          [&](const auto& stream) { return stream.partition == automatic->id; }),
            1);
  EXPECT_EQ(automatic_stream->process_group, residual->process_group);
  EXPECT_EQ(automatic_stream->device, residual->device);
  ASSERT_TRUE(automatic_stream->green_context_id.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(*automatic_stream->green_context_id, residual->green_context_id);
  EXPECT_NE(automatic_stream->green_context_id, anchor_stream->green_context_id);
  EXPECT_TRUE(std::ranges::none_of(plan.cuda_green_context_assignments(), [&](const auto& row) {
    return row.partition == automatic->id;
  }));
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);
  EXPECT_EQ(plan.tensor_device_resources().front().process_group, pool.process_group);
  EXPECT_EQ(plan.tensor_device_resources().front().device, pool.device);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().allocator_root_id,
            plan.tensor_device_resources().front().allocator_root_id);
  EXPECT_EQ(plan.tensor_allocations().front().allocator_process_group,
            automatic_stream->process_group);
  EXPECT_EQ(plan.tensor_allocations().front().device, automatic_stream->device);
  const auto output = std::ranges::find(
      plan.ports(), plan.tensor_allocations().front().output, &PortPlanRecord::id);
  ASSERT_NE(output, plan.ports().end());
  EXPECT_NE(std::ranges::find(automatic->members, output->owner), automatic->members.end());
  EXPECT_EQ(std::ranges::find(anchor->members, output->owner), anchor->members.end());
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextBindingsAreOrderIndependent) {
  // Key setup: compile the same two bindings in forward and reverse authoring order.
  const ExecutionPlan forward = compile_green_context_partitions(false);
  const ExecutionPlan reverse = compile_green_context_partitions(true);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();
  // Key assertion: canonical records, digest, and serialized plan are byte-for-byte identical.
  EXPECT_TRUE(std::ranges::equal(forward.cuda_green_context_requests(),
                                 reverse.cuda_green_context_requests()));
  EXPECT_TRUE(std::ranges::equal(forward.cuda_green_context_assignments(),
                                 reverse.cuda_green_context_assignments()));
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextSeparatesTwoPhysicalDevicePools) {
  // Key setup: bind two isolated CPU-only partitions to exact full-device groups on devices 0
  // and 1 with distinct stable physical identities.
  const ExecutionPlan plan = compile_green_context_partitions_on_two_devices();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);
  ASSERT_EQ(plan.cuda_green_context_assignments().size(), 2U);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 2U);
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 2U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 2U);

  // Key assertion: each physical device owns an independent pool, context, assignment, and
  // partition stream; no cross-device residual or shared owner is fabricated.
  EXPECT_EQ(plan.cuda_green_context_pools()[0].device, DeviceId{0});
  EXPECT_EQ(plan.cuda_green_context_pools()[0].deployment_device_id, "gpu.test.uuid.0");
  EXPECT_EQ(plan.cuda_green_context_pools()[1].device, DeviceId{1});
  EXPECT_EQ(plan.cuda_green_context_pools()[1].deployment_device_id, "gpu.test.uuid.1");
  EXPECT_NE(plan.cuda_green_context_pools()[0].pool_id, plan.cuda_green_context_pools()[1].pool_id);
  EXPECT_NE(plan.cuda_green_context_resources()[0].green_context_id,
            plan.cuda_green_context_resources()[1].green_context_id);
  EXPECT_NE(plan.cuda_compute_streams()[0].stream_id, plan.cuda_compute_streams()[1].stream_id);
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextArtifactExposesCompleteStableLineage) {
  // Key setup: serialize a two-request plan with dedicated contexts and a structured residual.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const std::string_view json = plan.json();
  constexpr std::array<std::string_view, 18U> kRequiredEvidence{
      "\"requested\"",
      "\"accepted\"",
      "\"source\":\"developer_binding\"",
      "\"disposition\"",
      "\"reason\"",
      "\"deployment_device_id\":\"gpu.test.uuid.0\"",
      "\"pool_id\"",
      "\"green_context_id\"",
      "\"stream_id\"",
      "\"owner_id\"",
      "\"accepted_sm_count\"",
      "\"role\":\"reserved_residual\"",
      "\"unprovisioned_sm_count\"",
      "\"structured_sm_quantum\":8",
      "\"sm_coscheduled_alignment\":8",
      "\"sm_set_guarantee\":\"driver_documented\"",
      "\"workqueue_scope\":\"device_context\"",
      "\"materialize_before\":4",
  };
  // Key assertion: the replayable artifact explains request, split, owner, guarantee, and
  // lifecycle lineage using only stable plan/deployment identities and no native CUDA handles.
  for (const std::string_view evidence : kRequiredEvidence) {
    EXPECT_NE(json.find(evidence), std::string_view::npos) << evidence;
  }
  EXPECT_EQ(json.find("CUgreenCtx"), std::string_view::npos);
  EXPECT_EQ(json.find("CUcontext"), std::string_view::npos);
  EXPECT_EQ(json.find("CUstream"), std::string_view::npos);
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextOvercommitRejectsEveryCompetingRequest) {
  // Key setup: request 48 SMs and 40 SMs from one 80-SM device.
  const ExecutionPlan plan = compile_green_context_partitions(false, 48U, 40U);
  ASSERT_FALSE(plan.ok());
  // Key assertion: both competing requests are rejected without partial resource assignment.
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  EXPECT_TRUE(plan.cuda_green_context_resources().empty());
  const DiagnosticId expected = detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED");
  EXPECT_EQ(plan.cuda_green_context_requests()[0].reason, expected);
  EXPECT_EQ(plan.cuda_green_context_requests()[1].reason, expected);
  EXPECT_EQ(std::count_if(plan.diagnostics().begin(),
                          plan.diagnostics().end(),
                          [&](const auto& diagnostic) { return diagnostic.id == expected; }),
            2);
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextPartialRejectionPublishesOneCompleteAcceptedGroup) {
  // Graph topology (each node is in the named execution partition):
  //   source(alpha) --> target(beta)
  // Key setup: combine one valid 16-SM request with one below-granularity four-SM request.
  const ExecutionPlan plan = compile_green_context_partitions(false, 16U, 4U);
  ASSERT_FALSE(plan.ok());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);

  // Key assertion: the rejected beta request publishes no partial descendants while the accepted
  // alpha transaction publishes one complete assignment, pool, dedicated context, and residual.
  EXPECT_EQ(plan.cuda_green_context_requests()[0].disposition, PlanDisposition::kAccepted);
  EXPECT_EQ(plan.cuda_green_context_requests()[1].disposition, PlanDisposition::kRejected);
  EXPECT_EQ(plan.cuda_green_context_requests()[1].reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY"));
  ASSERT_EQ(plan.cuda_green_context_assignments().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_assignments().front().partition, ExecutionPartitionId{1U});
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 2U);
  EXPECT_EQ(plan.cuda_green_context_resources()[0].role,
            CudaGreenContextResourceRole::kDedicatedPartition);
  EXPECT_EQ(plan.cuda_green_context_resources()[1].role,
            CudaGreenContextResourceRole::kReservedResidual);
}

// REQ: MEM-GREENCTX-014
// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsInvalidRequestVocabulary) {
  struct InvalidRequest {
    DeviceId device;
    std::uint32_t sm_count;
    DiagnosticId reason;
  };
  const std::array invalid_requests{
      InvalidRequest{
          DeviceId{}, 16U, detail::diagnostic_id("CUDA_GREEN_CONTEXT_INVALID_PLACEMENT")},
      InvalidRequest{
          DeviceId{-2}, 16U, detail::diagnostic_id("CUDA_GREEN_CONTEXT_INVALID_PLACEMENT")},
      InvalidRequest{
          DeviceId{0}, 0U, detail::diagnostic_id("CUDA_GREEN_CONTEXT_INVALID_PLACEMENT")},
      InvalidRequest{
          DeviceId{0}, 4U, detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY")},
      InvalidRequest{
          DeviceId{0}, 12U, detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY")},
      InvalidRequest{
          DeviceId{0}, 88U, detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED")},
  };

  // Key setup: compile unresolved, negative, zero, below-minimum, misaligned, and over-device
  // exact requests against the same qualified 80-SM capability.
  for (const auto& invalid : invalid_requests) {
    const ExecutionPlan plan =
        compile_single_green_context(invalid.sm_count, 80U, false, invalid.device);
    ASSERT_FALSE(plan.ok());
    ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
    // Key assertion: each invalid value fails closed at its owning admission boundary without
    // producing an assignment, pool, or native-materialization obligation.
    EXPECT_EQ(plan.cuda_green_context_requests().front().reason, invalid.reason);
    EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
    EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  }
}

// REQ: MEM-GREENCTX-014
// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsUnknownEmptyAndDuplicatePartitions) {
  // Key setup: compile bindings for an empty name, an unknown name, and the known partition twice.
  const std::array plans{
      compile_single_green_context(16U, 80U, false, DeviceId{0}, ""),
      compile_single_green_context(16U, 80U, false, DeviceId{0}, "unknown"),
      compile_single_green_context(16U, 80U, false, DeviceId{0}, "dedicated", true),
  };

  // Key assertion: every authored row receives the stable unknown-or-duplicate reason and no
  // request can partially materialize a resource topology.
  const DiagnosticId expected = detail::diagnostic_id("CUDA_GREEN_CONTEXT_UNKNOWN_OR_DUPLICATE");
  for (const auto& plan : plans) {
    ASSERT_FALSE(plan.ok());
    ASSERT_FALSE(plan.cuda_green_context_requests().empty());
    EXPECT_TRUE(std::ranges::all_of(plan.cuda_green_context_requests(), [&](const auto& request) {
      return request.reason == expected && request.disposition == PlanDisposition::kRejected;
    }));
    EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
    EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  }
}

// REQ: MEM-GREENCTX-014
// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsTensorPlacementDeviceConflict) {
  // Graph topology:
  //   source   (isolated CUDA Tensor producer in dedicated partition)
  // Key setup: place the Tensor output on device 0 but request the partition's Green Context on
  // device 1.
  Graph graph{"green-context-tensor-device-conflict"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  graph.partition("dedicated").add(source);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput;
      });
  ASSERT_NE(output, snapshot->ports.end());
  const std::array devices{detail::TensorDeviceBindingRecord{
      .port = PortId{output->id.value},
      .device = DeviceId{0},
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  CompileOptions options;
  options.deployment.bind_cuda_green_context(
      "dedicated", CudaGreenContextPlacement{.device = DeviceId{1}, .sm_count = 16U});
  const ExecutionPlan plan =
      detail::compile_with_runtime_evidence(graph, std::move(options), evidence);

  // Key assertion: the explicit Tensor compute authority wins and admission emits no assignment
  // or native resource obligation for the conflicting Green Context request.
  ASSERT_FALSE(plan.ok());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_DEVICE_CONFLICT"));
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
}

// REQ: COMPILE-063
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
TEST(CompilePartitionAuthoring,
     GreenContextRejectsConflictingSourceDeviceObligationsBeforeBinding) {
  for (const CudaComputeStreamPolicy policy : {CudaComputeStreamPolicy::kSharedProcessDevice,
                                               CudaComputeStreamPolicy::kPerExecutionPartition}) {
    for (const DeviceId requested_device : {DeviceId{0}, DeviceId{1}}) {
      SCOPED_TRACE(::testing::Message() << "policy=" << static_cast<std::uint32_t>(policy)
                                        << " requested_device=" << requested_device.value);
      Graph graph{"green-context-conflicting-source-devices"};
      test::add_compile_test_default_clock(graph);
      const auto source = graph.op<PartitionMultiDeviceTensorSourceOperator>("source");
      graph.partition("dedicated").add(source);
      const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
      ASSERT_TRUE(snapshot);
      std::vector<detail::TensorDeviceBindingRecord> devices;
      for (const auto& port : snapshot->ports) {
        if (port.direction != PortDirection::kOutput || !port.setup.tensor_layout) {
          continue;
        }
        devices.push_back(detail::TensorDeviceBindingRecord{
            .port = PortId{port.id.value},
            .device = port.name == "first" ? DeviceId{0} : DeviceId{1},
        });
      }
      ASSERT_EQ(devices.size(), 2U);
      const std::array capabilities{
          detail::CudaGreenContextCapabilityEvidence{
              .device = DeviceId{0},
              .deployment_device_id = "gpu.test.uuid.conflict.0",
              .total_sm_count = 80U,
              .min_sm_partition_size = 8U,
              .sm_coscheduled_alignment = 8U,
              .max_structured_context_count = 10U,
              .evidence_id = "test.cuda.green_context.capability.conflict.0/v1",
              .probe_status = PlacementProbeStatus::kPassed,
          },
          detail::CudaGreenContextCapabilityEvidence{
              .device = DeviceId{1},
              .deployment_device_id = "gpu.test.uuid.conflict.1",
              .total_sm_count = 80U,
              .min_sm_partition_size = 8U,
              .sm_coscheduled_alignment = 8U,
              .max_structured_context_count = 10U,
              .evidence_id = "test.cuda.green_context.capability.conflict.1/v1",
              .probe_status = PlacementProbeStatus::kPassed,
          },
      };
      auto evidence = detail::runtime_compile_evidence_view();
      evidence.tensor_device_bindings = devices;
      evidence.cuda_green_context_capabilities = capabilities;
      CompileOptions options;
      options.cuda_execution.compute_stream_policy = policy;
      options.deployment.bind_cuda_green_context(
          "dedicated", CudaGreenContextPlacement{.device = requested_device, .sm_count = 16U});

      const ExecutionPlan plan =
          detail::compile_with_runtime_evidence(graph, std::move(options), evidence);

      ASSERT_FALSE(plan.ok());
      EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"));
      ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
      EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
                detail::diagnostic_id("CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE"));
      EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
      EXPECT_TRUE(plan.cuda_green_context_pools().empty());
      EXPECT_TRUE(plan.operator_cuda_stream_bindings().empty());
      EXPECT_FALSE(plan.json().empty());
    }
  }
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextExactGroupMayConsumeFullDeviceWithoutResidualWork) {
  // Key setup: assign one exact 80-SM group on an 80-SM device with no other CUDA work.
  const ExecutionPlan plan = compile_single_green_context(80U, 80U, false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: the single dedicated group owns all SMs and no residual or tail is recorded.
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_resources().front().accepted_sm_count, 80U);
  EXPECT_EQ(plan.cuda_green_context_resources().front().role,
            CudaGreenContextResourceRole::kDedicatedPartition);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_pools().front().unprovisioned_sm_count, 0U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  EXPECT_EQ(plan.cuda_compute_streams().front().partition, ExecutionPartitionId{1U});
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextWholeDeviceMayBypassStructuredQuantum) {
  // Key setup: request one whole-device Green Context on an 82-SM device with an 8-SM quantum.
  const ExecutionPlan plan = compile_single_green_context(82U, 82U, false);
  ASSERT_TRUE(plan.ok()) << plan.json();

  // Key assertion: compiler admission matches the native pool's one-group whole-device exception.
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_resources().front().accepted_sm_count, 82U);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_pools().front().structured_sm_quantum, 8U);
  EXPECT_EQ(plan.cuda_green_context_pools().front().unprovisioned_sm_count, 0U);
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextWholeDeviceWaiverRequiresOneCompleteGroup) {
  // Key setup: combine an irregular 82-SM request with a second structured request on an 82-SM
  // device, so the first request is not the native single-partition whole-device case.
  const ExecutionPlan plan = compile_green_context_partitions(
      false, 82U, 8U, PlacementProbeStatus::kPassed, Profile::kStandard, 82U);
  ASSERT_FALSE(plan.ok());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);

  // Key assertion: the irregular request fails its own structured-granularity contract rather
  // than borrowing the whole-device waiver and being diagnosed only as aggregate overcommit.
  EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY"));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRecordsStructuredResidualAndIrregularTail) {
  // Key setup: lower two dedicated requests plus two unbound CUDA partitions on an 82-SM device.
  const ExecutionPlan plan = compile_green_context_residual_partitions(2U);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: a 40-SM structured residual serves both partitions while the 2-SM tail remains
  // unprovisioned.
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 3U);
  EXPECT_EQ(plan.cuda_green_context_resources()[0].accepted_sm_count, 16U);
  EXPECT_EQ(plan.cuda_green_context_resources()[1].accepted_sm_count, 24U);
  EXPECT_EQ(plan.cuda_green_context_resources()[2].accepted_sm_count, 40U);
  EXPECT_EQ(plan.cuda_green_context_resources()[2].role,
            CudaGreenContextResourceRole::kSharedServiceResidual);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_pools().front().unprovisioned_sm_count, 2U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 4U);
  EXPECT_EQ(
      std::count_if(plan.cuda_compute_streams().begin(),
                    plan.cuda_compute_streams().end(),
                    [](const auto& stream) { return stream.green_context_id && stream.partition; }),
      4);
  EXPECT_TRUE(std::all_of(plan.cuda_compute_streams().begin(),
                          plan.cuda_compute_streams().end(),
                          [](const auto& stream) { return stream.fused_linear_edges.empty(); }));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextHonorsDriverQualifiedStructuredCapacity) {
  // Key setup: model Thor's 132 SMs, 8-SM quantum, and driver-qualified limit of fifteen units.
  const ExecutionPlan plan =
      compile_single_green_context(120U, 132U, false, DeviceId{0}, "dedicated", false, 15U);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: the exact 120-SM group is admitted and all 12 unavailable SMs remain outside
  // HSDK work instead of fabricating a sixteenth structured unit.
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_resources().front().accepted_sm_count, 120U);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_pools().front().unprovisioned_sm_count, 12U);

  const ExecutionPlan overcommitted =
      compile_single_green_context(128U, 132U, false, DeviceId{0}, "dedicated", false, 15U);
  ASSERT_FALSE(overcommitted.ok());
  ASSERT_EQ(overcommitted.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(overcommitted.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED"));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextAllowsUnstructuredTailOnlyWhenItOwnsNoCudaWork) {
  // Key setup: compare an unused 2-SM tail with the same tail required by an unbound CUDA source.
  const ExecutionPlan unused_tail = compile_single_green_context(80U, 82U, false);
  // Key assertion: the unused tail is admitted, but assigning CUDA work to it rejects the plan.
  ASSERT_TRUE(unused_tail.ok()) << unused_tail.json();
  ASSERT_EQ(unused_tail.cuda_green_context_resources().size(), 1U);
  EXPECT_EQ(unused_tail.cuda_green_context_pools().front().unprovisioned_sm_count, 2U);

  const ExecutionPlan required_residual = compile_single_green_context(80U, 82U, true);
  ASSERT_FALSE(required_residual.ok());
  ASSERT_EQ(required_residual.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(required_residual.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED"));
  EXPECT_TRUE(required_residual.cuda_green_context_resources().empty());
  EXPECT_TRUE(std::none_of(required_residual.cuda_compute_streams().begin(),
                           required_residual.cuda_compute_streams().end(),
                           [](const auto& stream) { return stream.green_context_id.has_value(); }));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsUnqualifiedResidualParticipant) {
  // Graph topology (isolated partitions):
  //   dedicated(Standard)   residual(Robotics, CUDA Tensor source)
  // Key setup: opt only the Standard partition into Green Context execution while an unbound
  // Robotics partition requires CUDA work on the same device.
  const ExecutionPlan plan = compile_single_green_context(
      16U, 80U, true, DeviceId{0}, "dedicated", false, std::nullopt, Profile::kRobotics);

  // Key assertion: admission validates the complete residual closure and rejects the authored
  // request instead of silently assigning the Robotics partition to the residual context.
  ASSERT_FALSE(plan.ok());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED"));
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsMultiDeviceResidualParticipant) {
  // Graph topology (isolated partitions):
  //   dedicated   residual-a(device 0)  residual-b(device 1)
  //               \____________________/
  //                   residual partition
  // Key setup: opt the inactive dedicated partition into device zero while one unbound partition
  // owns CUDA Tensor obligations on devices zero and one.
  Graph graph{"green-context-multi-device-residual"};
  test::add_compile_test_default_clock(graph);
  const auto dedicated = graph.op<test::CompileTestOperator>("dedicated");
  const auto residual_a = graph.op<PartitionTensorSourceOperator>("residual-a");
  const auto residual_b = graph.op<PartitionTensorSourceOperator>("residual-b");
  graph.partition("dedicated").add(dedicated);
  graph.partition("residual").add(residual_a).add(residual_b);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  std::vector<detail::TensorDeviceBindingRecord> devices;
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput && port.setup.tensor_layout) {
      devices.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value},
          .device = DeviceId{static_cast<std::int32_t>(devices.size())},
      });
    }
  }
  ASSERT_EQ(devices.size(), 2U);
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.multi-device-residual",
      .total_sm_count = 80U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_cuda_green_context(
      "dedicated", CudaGreenContextPlacement{.device = DeviceId{0}, .sm_count = 16U});
  const ExecutionPlan plan =
      detail::compile_with_runtime_evidence(graph, std::move(options), evidence);

  // Key assertion: the residual participant is rejected before streams or pools are published.
  ASSERT_FALSE(plan.ok());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE"));
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
  // The diagnostic names the residual partition and a port requiring each of its devices.
  const auto diagnostic = std::ranges::find_if(plan.diagnostics(), [](const auto& item) {
    return item.token == "CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE";
  });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  EXPECT_EQ(diagnostic->provided,
            "residual_partition=residual,tensor_devices=0:residual-a.output|1:residual-b.output");
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextCountsReservedResidualAgainstContextBudget) {
  // Key setup: request 72 of 80 SMs when capability evidence permits only one structured context.
  const ExecutionPlan plan =
      compile_single_green_context(72U, 80U, false, DeviceId{0}, "dedicated", false, 1U);
  ASSERT_FALSE(plan.ok());
  // Key assertion: the eight-SM reserved residual counts as a second context and admission fails
  // before publishing a pool that exceeds the qualified context budget.
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED"));
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  EXPECT_TRUE(plan.cuda_green_context_resources().empty());
}

// REQ: COMPILE-063
// REQ: CHAN-134
TEST(CompilePartitionAuthoring, GreenContextOmitsProcessLocalNoCopyServiceStream) {
  // Key setup: bind only the Tensor source, leaving the sink on the residual while the
  // process-local Tensor service retains backing without compiler-owned staging or copy work.
  const ExecutionPlan plan = compile_green_context_tensor_service();
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: source and sink receive distinct partition streams, but no unused service
  // stream is planned for the process-local no-copy transport.
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 2U);
  EXPECT_EQ(plan.cuda_green_context_resources()[1].role,
            CudaGreenContextResourceRole::kSharedServiceResidual);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 2U);
  const auto source =
      std::find_if(plan.cuda_compute_streams().begin(),
                   plan.cuda_compute_streams().end(),
                   [](const auto& stream) { return stream.partition == ExecutionPartitionId{2U}; });
  const auto sink =
      std::find_if(plan.cuda_compute_streams().begin(),
                   plan.cuda_compute_streams().end(),
                   [](const auto& stream) { return stream.partition == ExecutionPartitionId{1U}; });
  ASSERT_NE(source, plan.cuda_compute_streams().end());
  ASSERT_NE(sink, plan.cuda_compute_streams().end());
  EXPECT_NE(source->stream_id, sink->stream_id);
  EXPECT_TRUE(std::ranges::none_of(plan.cuda_compute_streams(),
                                   [](const auto& stream) { return !stream.partition; }));
  EXPECT_TRUE(source->fused_linear_edges.empty());
  EXPECT_TRUE(sink->fused_linear_edges.empty());
}

// REQ: CHAN-134
// REQ: COMPILE-063
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
// REQ: TENSOR-STORAGE-014
TEST(CompilePartitionAuthoring, GreenContextResolvesExactTensorBindingsUnderBothOrdinaryPolicies) {
  std::optional<std::vector<TensorDeviceResourcePlanRecord>> expected_device_resources;
  const auto resource_geometry = [](const TensorDeviceResourcePlanRecord& resource) {
    return std::tie(resource.run_generation,
                    resource.process_group,
                    resource.device,
                    resource.pool_count,
                    resource.slot_capacity,
                    resource.loan_credit_capacity,
                    resource.control_capacity,
                    resource.readiness_capacity,
                    resource.release_capacity,
                    resource.retirement_capacity,
                    resource.quarantine_capacity,
                    resource.materialize_before,
                    resource.gpu_resident_backend_selected,
                    resource.fallback_rejected);
  };
  for (const CudaComputeStreamPolicy policy : {CudaComputeStreamPolicy::kSharedProcessDevice,
                                               CudaComputeStreamPolicy::kPerExecutionPartition}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    const ExecutionPlan plan = compile_green_context_tensor_service(policy);
    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_EQ(plan.cuda_compute_stream_policy(), policy);
    ASSERT_EQ(plan.cuda_compute_streams().size(), 2U);
    ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 2U);
    EXPECT_TRUE(std::ranges::all_of(plan.cuda_compute_streams(), [](const auto& stream) {
      return stream.partition.has_value() && stream.green_context_id.has_value() &&
             stream.fused_linear_edges.empty();
    }));
    EXPECT_NE(plan.operator_cuda_stream_bindings()[0].stream_id,
              plan.operator_cuda_stream_bindings()[1].stream_id);
    if (expected_device_resources) {
      EXPECT_TRUE(std::ranges::equal(plan.tensor_device_resources(),
                                     *expected_device_resources,
                                     {},
                                     resource_geometry,
                                     resource_geometry));
    } else {
      expected_device_resources = std::vector<TensorDeviceResourcePlanRecord>{
          plan.tensor_device_resources().begin(), plan.tensor_device_resources().end()};
    }
  }
}

// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
// REQ: TENSOR-STORAGE-015
TEST(CompilePartitionAuthoring, GreenOnlyForwardingDomainSetsFinalReleaseCapacity) {
  for (const CudaComputeStreamPolicy policy : {CudaComputeStreamPolicy::kSharedProcessDevice,
                                               CudaComputeStreamPolicy::kPerExecutionPartition}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    const ExecutionPlan plan = compile_green_context_forwarding_chain(policy);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.tensor_allocations().size(), 1U);
    ASSERT_EQ(plan.cuda_green_context_assignments().size(), 5U);
    ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 5U);
    ASSERT_EQ(plan.cuda_compute_streams().size(), 5U);
    EXPECT_EQ(plan.tensor_allocations().front().release_stream_capacity, 9U);
    EXPECT_TRUE(std::ranges::all_of(plan.cuda_compute_streams(), [](const auto& stream) {
      return stream.partition.has_value() && stream.green_context_id.has_value();
    }));
    std::vector<std::string_view> stream_ids;
    for (const auto& binding : plan.operator_cuda_stream_bindings()) {
      stream_ids.push_back(binding.stream_id);
    }
    std::ranges::sort(stream_ids);
    EXPECT_EQ(std::ranges::unique(stream_ids).begin(), stream_ids.end());
  }
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenOnlyForwardingRejectsConflictingOutputPlacement) {
  // relay-a receives device 0 from the source but binds its output to device 1. relay-b then
  // receives device 1 while its output inherits device 0 from its Green Context partition.
  const auto plan = compile_green_context_forwarding_chain(
      CudaComputeStreamPolicy::kPerExecutionPartition, DeviceId{1});
  ASSERT_FALSE(plan.ok());
  const auto provided_for = [&](std::string_view token) {
    std::vector<std::string> provided;
    for (const auto& diagnostic : plan.diagnostics()) {
      if (diagnostic.token == token) {
        provided.push_back(diagnostic.provided);
      }
    }
    std::ranges::sort(provided);
    return provided;
  };

  // Key assertion: each conflict names both ports and where each device came from.
  EXPECT_EQ(provided_for("PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"),
            (std::vector<std::string>{
                "first_port=relay-a.output,first_device=1,first_device_source=binding,"
                "conflicting_port=relay-a.input,conflicting_device=0,"
                "conflicting_device_source=producer(source.output),"
                "reason=operator_device_conflict",
                "first_port=relay-b.output,first_device=0,"
                "first_device_source=inherited(green_context_partition:relay-b),"
                "conflicting_port=relay-b.input,conflicting_device=1,"
                "conflicting_device_source=producer(relay-a.output),"
                "reason=operator_device_conflict",
            }));
  EXPECT_EQ(provided_for("CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE"),
            (std::vector<std::string>{
                "partition=relay-a,requested=0,tensor_devices=0:relay-a.input|1:relay-a.output",
                "partition=relay-b,requested=0,tensor_devices=0:relay-b.output|1:relay-b.input",
            }));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextReplacementStreamRetainsFusedEdgeOwnership) {
  // Graph topology (both operators are fused into one execution partition):
  //   source --> sink
  // Key setup: lower a process/device ordinary stream that owns one fused Tensor edge, then bind
  // that partition to a dedicated Green Context.
  const ExecutionPlan plan = compile_green_context_fused_tensor_partition();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto partition_stream =
      std::find_if(plan.cuda_compute_streams().begin(),
                   plan.cuda_compute_streams().end(),
                   [](const auto& stream) { return stream.partition.has_value(); });
  ASSERT_NE(partition_stream, plan.cuda_compute_streams().end());

  // Key assertion: stream replacement preserves the exact fused edge on the dedicated stream
  // instead of dropping it with the ordinary stream or transferring it to a residual owner.
  ASSERT_TRUE(partition_stream->green_context_id);
  ASSERT_EQ(partition_stream->fused_linear_edges.size(), 1U);
  EXPECT_EQ(partition_stream->fused_linear_edges.front(), EdgeId{1U});
  EXPECT_TRUE(std::none_of(plan.cuda_compute_streams().begin(),
                           plan.cuda_compute_streams().end(),
                           [&](const auto& stream) {
                             return &stream != &*partition_stream &&
                                    !stream.fused_linear_edges.empty();
                           }));
}

// REQ: COMPILE-063
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
TEST(CompilePartitionAuthoring, ZeroBindingTensorGraphKeepsOneOrdinaryStream) {
  // Graph topology:
  //   source   (isolated CUDA Tensor producer)
  // Key setup: compile one device-backed Tensor source without a Green Context binding.
  const ExecutionPlan plan = compile_zero_binding_tensor_graph();
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: the default emits one partition-owned ordinary stream and no Green Context
  // records for the graph's single active execution partition.
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  EXPECT_TRUE(plan.cuda_green_context_resources().empty());
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().release_stream_capacity, 1U);
  EXPECT_FALSE(plan.cuda_compute_streams().front().green_context_id);
  const auto& partition = plan.cuda_compute_streams().front().partition;
  HOLOSCAN_ASSERT_HAS_VALUE(partition);
  EXPECT_EQ(plan.cuda_compute_streams().front().owner_id,
            "partition." + std::to_string(partition->value));
}

// REQ: COMPILE-063
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
TEST(CompilePartitionAuthoring, PublicCudaPolicyRetainsExactPartitionBinding) {
  // Key setup: select per-partition CUDA streams for one isolated Tensor producer through the
  // public compile option.
  const ExecutionPlan plan =
      compile_zero_binding_tensor_graph(CudaComputeStreamPolicy::kPerExecutionPartition);
  const ExecutionPlan shared =
      compile_zero_binding_tensor_graph(CudaComputeStreamPolicy::kSharedProcessDevice);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(shared.ok()) << shared.json();

  // Key assertion: the accepted policy and exact Operator join remain inspectable without
  // reverse-engineering partition or stream identifiers.
  EXPECT_EQ(plan.cuda_compute_stream_policy(), CudaComputeStreamPolicy::kPerExecutionPartition);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
  const auto& partition = plan.cuda_compute_streams().front().partition;
  HOLOSCAN_ASSERT_HAS_VALUE(partition);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().partition, *partition);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().stream_id,
            plan.cuda_compute_streams().front().stream_id);
  EXPECT_NE(plan.digest(), shared.digest());
}

// REQ: COMPILE-063
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
TEST(CompilePartitionAuthoring, RejectsUnknownCudaComputeStreamPolicy) {
  // Key setup: pass a value outside the closed public CUDA stream-policy enum.
  const ExecutionPlan plan =
      // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
      compile_zero_binding_tensor_graph(static_cast<CudaComputeStreamPolicy>(255U));

  // Key assertion: compilation rejects the value with a stable public diagnostic while leaving
  // the rejected plan safe to inspect.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "CUDA_STREAM_POLICY_VIOLATION"));
  // The rejected plan must retain the unknown input value.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(plan.cuda_compute_stream_policy(), static_cast<CudaComputeStreamPolicy>(255U));
  const auto diagnostic = std::ranges::find_if(plan.diagnostics(), [](const auto& row) {
    return row.token == "CUDA_STREAM_POLICY_VIOLATION";
  });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  const auto& detail = diagnostic->detail;
  HOLOSCAN_ASSERT_HAS_VALUE(detail);
  EXPECT_EQ(detail->operation, DiagnosticOperation::kCompile);
  EXPECT_EQ(detail->error.code, ErrorCode::kInvalidArgument);
  ASSERT_EQ(detail->mismatches.size(), 1U);
  EXPECT_EQ(detail->mismatches.front().field, "compute_stream_policy");
  EXPECT_EQ(detail->mismatches.front().provided, "255");
  EXPECT_EQ(detail->mismatches.front().required, "kPerExecutionPartition or kSharedProcessDevice");
  try {
    plan.require_no_errors();
    FAIL() << "rejected plan did not throw";
  } catch (const CompileError& error) {
    EXPECT_NE(std::string_view{error.what()}.find("compute_stream_policy"), std::string_view::npos);
    EXPECT_NE(std::string_view{error.what()}.find("255"), std::string_view::npos);
  }
}

// REQ: COMPILE-001
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
TEST(CompilePartitionAuthoring, AggregateCudaPoolOverflowRetainsInspectableRejectedPlan) {
  for (const CudaComputeStreamPolicy policy : {CudaComputeStreamPolicy::kSharedProcessDevice,
                                               CudaComputeStreamPolicy::kPerExecutionPartition}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    Graph graph{"aggregate-cuda-pool-overflow"};
    test::add_compile_test_default_clock(graph);
    const auto overflow = graph.op<PartitionAggregateOverflowTensorSourceOperator>("a_overflow");
    const auto other = graph.op<PartitionTensorSourceOperator>("z_other");
    graph.partition("overflow").add(overflow);
    graph.partition("other").add(other);

    CompileOptions options;
    options.cuda_execution.compute_stream_policy = policy;
    options.deployment.bind_tensor_output_device({"a_overflow", "first", DeviceId{0}})
        .bind_tensor_output_device({"a_overflow", "second", DeviceId{0}})
        .bind_tensor_output_device({"z_other", "output", DeviceId{1}});

    std::optional<ExecutionPlan> plan;
    EXPECT_NO_THROW(plan.emplace(compile(graph, std::move(options))));
    HOLOSCAN_ASSERT_HAS_VALUE(plan);
    EXPECT_FALSE(plan->ok());
    EXPECT_FALSE(plan->json().empty());
    EXPECT_TRUE(test::has_diagnostic(*plan, "PHYSICAL_TENSOR_CAPACITY_OVERFLOW"));
    ASSERT_EQ(plan->tensor_device_resources().size(), 2U);
    EXPECT_TRUE(std::ranges::equal(plan->tensor_device_resources(),
                                   std::array{DeviceId{0}, DeviceId{1}},
                                   {},
                                   &TensorDeviceResourcePlanRecord::device));
    ASSERT_EQ(plan->operator_cuda_stream_bindings().size(), 2U);
    EXPECT_EQ(std::ranges::count(plan->operator_cuda_stream_bindings(),
                                 DeviceId{0},
                                 &OperatorCudaStreamBindingPlanRecord::device),
              1U);
    EXPECT_EQ(std::ranges::count(plan->operator_cuda_stream_bindings(),
                                 DeviceId{1},
                                 &OperatorCudaStreamBindingPlanRecord::device),
              1U);
    ASSERT_EQ(plan->cuda_compute_streams().size(), 2U);
    for (const auto& binding : plan->operator_cuda_stream_bindings()) {
      const auto stream = std::ranges::find(
          plan->cuda_compute_streams(), binding.stream_id, &CudaComputeStreamPlanRecord::stream_id);
      ASSERT_NE(stream, plan->cuda_compute_streams().end());
      EXPECT_EQ(stream->process_group, binding.process_group);
      EXPECT_EQ(stream->device, binding.device);
    }
  }
}

TEST(CompilePartitionAuthoring, CudaStreamDefaultsAndOrderingUsePartitionSemantics) {
  EXPECT_EQ(ExecutionPlan{}.cuda_compute_stream_policy(),
            CudaComputeStreamPolicy::kPerExecutionPartition);

  std::vector<CudaComputeStreamPlanRecord> streams{
      {.process_group = ProcessGroupId{1U},
       .device = DeviceId{0},
       .owner_id = "partition.10",
       .partition = ExecutionPartitionId{10U}},
      {.process_group = ProcessGroupId{1U},
       .device = DeviceId{0},
       .owner_id = "partition.2",
       .partition = ExecutionPartitionId{2U}},
  };
  std::sort(streams.begin(), streams.end(), detail::canonical_cuda_compute_stream_order);

  ASSERT_EQ(streams.size(), 2U);
  const auto& first_partition = streams[0].partition;
  const auto& second_partition = streams[1].partition;
  if (!first_partition.has_value() || !second_partition.has_value()) {
    FAIL() << "Expected both stream partitions";
  }
  EXPECT_EQ(*first_partition, ExecutionPartitionId{2U});
  EXPECT_EQ(*second_partition, ExecutionPartitionId{10U});
}

// REQ: COMPILE-063
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
TEST(CompilePartitionAuthoring, ZeroBindingArtifactMatchesNormalizedBaseline) {
  // Graph topology:
  //   source   (isolated CUDA Tensor producer)
  // Key setup: compile the no-binding graph, replace its environment-derived identities with
  // fixed values, and regenerate the complete canonical artifact before comparing it.
  const ExecutionPlan plan = compile_zero_binding_tensor_graph();
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.cuda_compute_stream_policy(), CudaComputeStreamPolicy::kPerExecutionPartition);
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  auto normalized = std::make_shared<detail::ExecutionPlanState>(*accepted);
  detail::rebind_plan_generation(*normalized, PlanGenerationId{1U});
  normalized->profile_manifest.target_triple = "normalized-target-triple";
  const auto finalized = detail::finalize_plan_artifact(*normalized);
  ASSERT_TRUE(finalized) << runtime_error_message(finalized.error());
  const ExecutionPlan normalized_plan = detail::ExecutionPlanAccess::make(std::move(normalized));

  const std::string golden_path = std::string(HOLOSCAN_COMPILER_TEST_DATA_DIR) +
                                  "/zero_binding_tensor_graph.v9.normalized.json";
  std::ifstream golden_stream{golden_path, std::ios::binary};
  ASSERT_TRUE(golden_stream.is_open()) << golden_path;
  const std::string golden{std::istreambuf_iterator<char>{golden_stream},
                           std::istreambuf_iterator<char>{}};
  ASSERT_TRUE(golden_stream.good() || golden_stream.eof()) << golden_path;

  // Key assertion: every retained canonical artifact byte, not only selected stream-shape fields,
  // matches the checked-in, versioned no-binding baseline after the intentional normalizations.
  // Inspect the complete diff before updating this golden. A schema bump requires a new versioned
  // filename; never replace this readable evidence by pasting a new opaque digest.
  EXPECT_EQ(detail::artifact_test::normalized_artifact_json(normalized_plan.json()), golden)
      << "review the artifact diff before updating " << golden_path;
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextArtifactMatchesFeatureRichNormalizedBaseline) {
  // Graph topology (each node is in the named execution partition):
  //   source(alpha) --> target(beta)
  // Key setup: compile accepted dedicated and residual Green Context records, replace its
  // environment-derived identities with fixed values, and regenerate the complete schema-v7
  // artifact.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  auto normalized = std::make_shared<detail::ExecutionPlanState>(*accepted);
  detail::rebind_plan_generation(*normalized, PlanGenerationId{1U});
  normalized->profile_manifest.target_triple = "normalized-target-triple";
  const auto finalized = detail::finalize_plan_artifact(*normalized);
  ASSERT_TRUE(finalized) << runtime_error_message(finalized.error());
  const ExecutionPlan normalized_plan = detail::ExecutionPlanAccess::make(std::move(normalized));

  const std::string golden_path =
      std::string(HOLOSCAN_COMPILER_TEST_DATA_DIR) + "/green_context_partitions.v9.normalized.json";
  std::ifstream golden_stream{golden_path, std::ios::binary};
  ASSERT_TRUE(golden_stream.is_open()) << golden_path;
  const std::string golden{std::istreambuf_iterator<char>{golden_stream},
                           std::istreambuf_iterator<char>{}};
  ASSERT_TRUE(golden_stream.good() || golden_stream.eof()) << golden_path;

  // Key assertion: nested placement optionals and cross-family pool/resource/request/assignment
  // joins retain readable byte-level coverage. Inspect the complete diff before updating this
  // golden. A schema bump requires a new versioned filename; never paste a new opaque digest.
  EXPECT_EQ(detail::artifact_test::normalized_artifact_json(normalized_plan.json()), golden)
      << "review the artifact diff before updating " << golden_path;
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextPlacementChangesSemanticIdentityCoherently) {
  // Key setup: compile two valid versions of the same partition cover with one accepted nested
  // placement changed from 16 to 8 SMs. Lowering updates the request, resource, and residual joins.
  const ExecutionPlan baseline = compile_green_context_partitions(false, 16U, 24U);
  const ExecutionPlan changed = compile_green_context_partitions(false, 8U, 24U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_TRUE(changed.ok()) << changed.json();
  ASSERT_TRUE(baseline.cuda_green_context_requests().front().accepted);
  ASSERT_TRUE(changed.cuda_green_context_requests().front().accepted);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(baseline.cuda_green_context_requests().front().accepted->sm_count, 16U);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(changed.cuda_green_context_requests().front().accepted->sm_count, 8U);
  ASSERT_EQ(baseline.cuda_green_context_resources().size(), 3U);
  ASSERT_EQ(changed.cuda_green_context_resources().size(), 3U);
  EXPECT_FALSE(std::ranges::equal(baseline.cuda_green_context_resources(),
                                  changed.cuda_green_context_resources()));

  // Key assertion: the coherent nested optional and cross-family resource changes perturb both
  // canonical bytes and semantic identity.
  EXPECT_NE(baseline.json(), changed.json());
  EXPECT_NE(baseline.digest(), changed.digest());
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextTargetEvidenceRequiresRuntimeRecheck) {
  // Key setup: compile valid bindings whose capability probe remains runtime-required.
  const ExecutionPlan plan =
      compile_green_context_partitions(false, 16U, 24U, PlacementProbeStatus::kRuntimeRequired);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: both split probes preserve the runtime-recheck status and evidence identity.
  EXPECT_EQ(std::count_if(plan.placement_probes().begin(),
                          plan.placement_probes().end(),
                          [](const auto& probe) {
                            return probe.probe_id == "cuda.green_context.split" &&
                                   probe.status == PlacementProbeStatus::kRuntimeRequired &&
                                   probe.evidence_id == "test.cuda.green_context.capability/v1";
                          }),
            2);
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsUnqualifiedProfiles) {
  // Key setup: compile the same bindings under profiles without qualified Green Context policy.
  // Key assertion: Robotics and Regulated both reject every authored request.
  for (const Profile profile : {Profile::kRobotics, Profile::kRegulated}) {
    const ExecutionPlan plan =
        compile_green_context_partitions(false, 16U, 24U, PlacementProbeStatus::kPassed, profile);
    ASSERT_FALSE(plan.ok());
    ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);
    EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
    EXPECT_TRUE(std::all_of(
        plan.cuda_green_context_requests().begin(),
        plan.cuda_green_context_requests().end(),
        [](const auto& request) {
          return request.disposition == PlanDisposition::kRejected && !request.accepted &&
                 request.reason ==
                     detail::diagnostic_id("CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED");
        }));
  }
}

// REQ: COMPILE-063
// REQ: LIFECYCLE-010
TEST(CompilePartitionAuthoring, GreenContextAdmitsStandardResearchAndRealTimeHotPathPolicy) {
  // Key setup: compile identical exact bindings under each profile with qualified hot-path policy.
  for (const Profile profile : {Profile::kStandard, Profile::kResearch, Profile::kRealTime}) {
    const ExecutionPlan plan =
        compile_green_context_partitions(false, 16U, 24U, PlacementProbeStatus::kPassed, profile);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
    // Key assertion: every admitted profile retains the CUDA driver-documented guarantee,
    // device-context workqueues, zero concurrency override, and exact structured quantum.
    const auto& pool = plan.cuda_green_context_pools().front();
    EXPECT_EQ(pool.sm_set_guarantee, CudaGreenContextSmSetGuarantee::kDriverDocumented);
    EXPECT_EQ(pool.workqueue_scope, CudaGreenContextWorkqueueScope::kDeviceContext);
    EXPECT_EQ(pool.workqueue_concurrency_limit, 0U);
    EXPECT_EQ(pool.structured_sm_quantum, 8U);
    const auto& features = plan.profile_manifest().feature_dispositions;
    const auto green_context_feature =
        std::find_if(features.begin(), features.end(), [](const auto& feature) {
          return feature.feature_id == "cuda.green_context";
        });
    ASSERT_NE(green_context_feature, features.end());
    EXPECT_EQ(green_context_feature->rule_id, "holoscan.profile.cuda_green_context.hot_path/v1");
  }
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsUnavailableTargetEvidence) {
  // Key setup: supply a target capability row whose split probe is explicitly rejected.
  const ExecutionPlan plan =
      compile_green_context_partitions(false, 16U, 24U, PlacementProbeStatus::kRejected);
  // Key assertion: target evidence that is neither passed nor runtime-required rejects every
  // request without emitting a pool, assignment, or runtime materialization obligation.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(plan.cuda_green_context_pools().empty());
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(plan.cuda_green_context_requests(), [](const auto& request) {
    return request.disposition == PlanDisposition::kRejected && !request.accepted &&
           request.reason == detail::diagnostic_id("CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED");
  }));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextReservedPoliciesRemainUnselectable) {
  // Key setup: mutate an accepted phase-1 pool independently to each parseable future policy.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto baseline = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(detail::validate_cuda_execution_plan_records(*baseline));
  using Mutation = void (*)(CudaGreenContextPoolPlanRecord&);
  const std::array<Mutation, 4U> mutations{
      [](auto& pool) { pool.co_scheduling = CudaGreenContextCoScheduling::kRelaxed; },
      [](auto& pool) {
        pool.workqueue_scope = CudaGreenContextWorkqueueScope::kGreenContextBalanced;
      },
      [](auto& pool) { pool.workqueue_concurrency_limit = 1U; },
      [](auto& pool) {
        pool.sm_set_guarantee = CudaGreenContextSmSetGuarantee::kStrictNoKnownSuperset;
      },
  };

  // Key assertion: reserved enum values remain source-compatible but fail phase-1 plan validation
  // instead of silently selecting unqualified co-scheduling, workqueue, or isolation behavior.
  for (const auto mutate : mutations) {
    auto candidate = *baseline;
    mutate(candidate.cuda_green_context_pools.front());
    EXPECT_FALSE(detail::validate_cuda_execution_plan_records(candidate));
  }
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextValidationRejectsResourceCountAboveBudget) {
  // Key setup: lower a valid two-dedicated-plus-residual pool, then corrupt its retained context
  // budget below the published resource count.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto baseline = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(baseline);
  auto candidate = *baseline;
  ASSERT_EQ(candidate.cuda_green_context_resources.size(), 3U);
  candidate.cuda_green_context_pools.front().max_structured_context_count = 2U;
  // Key assertion: plan validation rejects a full pool that exceeds qualified device evidence.
  EXPECT_FALSE(detail::validate_cuda_execution_plan_records(candidate));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextValidationRejectsOrdinaryStreamOnOwnedDevice) {
  // Key setup: remove Green Context ownership from one otherwise valid partition stream while its
  // process/device pair remains owned by the published pool.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto baseline = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(baseline);
  auto candidate = *baseline;
  ASSERT_FALSE(candidate.cuda_compute_streams.empty());
  candidate.cuda_compute_streams.front().green_context_id.reset();

  // Key assertion: plan validation fails closed instead of materializing a primary-context stream
  // beside the structured Green Context family.
  EXPECT_FALSE(detail::validate_cuda_execution_plan_records(candidate));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextBindingCannotNameAnotherPartitionsStream) {
  // Key setup: retain a valid two-partition Green Context plan, then redirect one Operator binding
  // to the other partition's materialized stream on the same process and device.
  const ExecutionPlan plan = compile_green_context_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto baseline = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(baseline);
  auto candidate = *baseline;
  ASSERT_EQ(candidate.cuda_compute_streams.size(), 2U);
  ASSERT_EQ(candidate.operator_cuda_stream_bindings.size(), 2U);
  candidate.operator_cuda_stream_bindings.front().stream_id =
      candidate.cuda_compute_streams.back().stream_id;

  // Key assertion: matching process/device facts cannot authorize a cross-partition binding.
  EXPECT_FALSE(detail::validate_cuda_execution_plan_records(candidate));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextValidationRejectsConflictingOperatorDeviceEvidence) {
  const ExecutionPlan plan = compile_green_context_residual_partitions(2U);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto baseline = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(baseline);
  auto candidate = *baseline;
  ASSERT_FALSE(candidate.tensor_allocations.empty());
  const PortId output = candidate.tensor_allocations.front().output;
  const auto port = std::ranges::find(candidate.ports, output, &PortPlanRecord::id);
  ASSERT_NE(port, candidate.ports.end());
  const VertexId owner = port->owner;
  ASSERT_TRUE(
      std::ranges::any_of(candidate.cuda_green_context_assignments, [&](const auto& assignment) {
        const auto partition = std::ranges::find(
            candidate.execution_partitions, assignment.partition, &ExecutionPartitionRecord::id);
        return partition != candidate.execution_partitions.end() &&
               std::ranges::find(partition->members, owner) != partition->members.end();
      }));
  candidate.tensor_allocations.front().device = DeviceId{1};
  std::erase_if(candidate.operator_cuda_stream_bindings,
                [&](const auto& binding) { return binding.operator_vertex == owner; });

  // Key assertion: removing the conflicting Operator from both bookkeeping sets cannot make an
  // unbound, ambiguously placed CUDA Operator pass exact plan validation.
  EXPECT_FALSE(detail::validate_cuda_execution_plan_records(candidate));
}

// REQ: COMPILE-063
TEST(CompilePartitionAuthoring, GreenContextRejectsOrdinaryCudaWorkInAnotherProcess) {
  // Graph topology (isolated CUDA Tensor producers in separate process groups):
  //   source(green-worker)   target(ordinary-worker)
  // Key setup: bind source to a Green Context while target uses an ordinary context on the same
  // physical GPU from another process group.
  Graph graph{"green-context-cross-process-owner"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  const auto target = graph.op<PartitionTensorSourceOperator>("target");
  graph.partition("green-source").add(source);
  graph.partition("ordinary-target").add(target);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  std::vector<detail::TensorDeviceBindingRecord> devices;
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput) {
      devices.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value},
          .device = DeviceId{0},
      });
    }
  }
  ASSERT_EQ(devices.size(), 2U);
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.0",
      .total_sm_count = 80U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_cuda_green_context(
      "green-source", CudaGreenContextPlacement{.device = DeviceId{0}, .sm_count = 16U});

  const ExecutionPlan plan = detail::compile_with_runtime_evidence_and_process_placement(
      graph, std::move(options), evidence, process_placement("green-worker", "ordinary-worker"));
  // Key assertion: admission rejects the unproven process-wide device ownership arrangement.
  ASSERT_FALSE(plan.ok());
  ASSERT_EQ(plan.cuda_green_context_requests().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_requests().front().reason,
            detail::diagnostic_id("CUDA_GREEN_CONTEXT_DEVICE_OWNERSHIP_UNPROVEN"));
  EXPECT_TRUE(plan.cuda_green_context_assignments().empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& row) {
        return row.id == detail::diagnostic_id("CUDA_GREEN_CONTEXT_DEVICE_OWNERSHIP_UNPROVEN");
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  EXPECT_NE(diagnostic->required.find("gpu.test.uuid.0"), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("process_group="), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("local_device=0"), std::string::npos);
}

TEST(CompilePartitionAuthoring, RejectsManualPartitionAcrossPhysicalProcessGroups) {
  const ExecutionPlan split = compile_manual_process_partition("source-process", "target-process");
  EXPECT_FALSE(split.ok());
  EXPECT_TRUE(test::has_diagnostic(split, "PARTITION_COVER_INVALID"));

  const ExecutionPlan colocated = compile_manual_process_partition("workers", "workers");
  ASSERT_TRUE(colocated.ok()) << colocated.json();
  ASSERT_EQ(colocated.execution_partitions().size(), 1U);
  EXPECT_EQ(colocated.execution_partitions().front().authored_name,
            std::optional<std::string>{"pipeline"});
  const auto process_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(colocated);
  ASSERT_EQ(process_placements.size(), 2U);
  EXPECT_EQ(process_placements[0].process_group, process_placements[1].process_group);
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, AutomaticSourceChainIsTheProductionDefault) {
  Graph graph{"automatic-source-chain-default"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<ClosedIslandSourceOperator>("first");
  (void)graph.op<ClosedIslandSourceOperator>("second");

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_EQ(plan.partitioning_plan().maturity, AlgorithmMaturity::kProduction);
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  EXPECT_EQ(plan.execution_partitions().size(), 2U);
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, AutomaticAndHybridCoversQualifyAllFiveProfilesWithoutFallback) {
  struct Expectation {
    Profile profile;
    bool accepted;
    std::string_view diagnostic;
  };
  constexpr std::array expectations{
      Expectation{Profile::kStandard, true, {}},
      Expectation{Profile::kRobotics, true, {}},
      Expectation{Profile::kRealTime, false, "PROFILE_CAPABILITY_UNSATISFIED"},
      Expectation{Profile::kRegulated, false, "PROFILE_EVIDENCE_UNSATISFIED"},
      Expectation{Profile::kResearch, true, {}},
  };

  for (const auto& expectation : expectations) {
    for (const bool hybrid : {false, true}) {
      SCOPED_TRACE(static_cast<std::uint32_t>(expectation.profile));
      SCOPED_TRACE(hybrid);
      Graph graph{hybrid ? "five-profile-hybrid" : "five-profile-automatic"};
      test::add_compile_test_default_clock(graph);
      const auto first = graph.op<ClosedIslandSourceOperator>("first");
      (void)graph.op<ClosedIslandSourceOperator>("second");
      if (hybrid) {
        graph.partition("anchor").add(first);
        graph.auto_partition_remaining();
      }
      CompileOptions options;
      options.profile = expectation.profile;
      const ExecutionPlan plan = compile(graph, std::move(options));

      EXPECT_EQ(plan.ok(), expectation.accepted);
      ASSERT_TRUE(plan.partitioning_plan().requested_algorithm.has_value());
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      EXPECT_EQ(plan.partitioning_plan().requested_algorithm->id,
                "holoscan.partition.source_chain");
      ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
      EXPECT_EQ(plan.partitioning_plan().mode,
                hybrid ? PartitioningMode::kConstrainedAutomatic : PartitioningMode::kAutomatic);
      EXPECT_EQ(plan.execution_partitions().size(), 2U);
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      EXPECT_NE(plan.partitioning_plan().accepted_algorithm->id,
                "holoscan.partition.serial_baseline");
      if (!expectation.diagnostic.empty()) {
        EXPECT_TRUE(test::has_diagnostic(plan, expectation.diagnostic));
      }
    }
  }
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, AutomaticSourceChainSplitsDifferentTensorDevices) {
  Graph graph{"automatic-device-transfer"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  const auto transform = graph.op<PartitionTensorTransformOperator>("transform");
  graph.add_flow(source->output, transform->input);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  std::vector<const PortRecord*> outputs;
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput) {
      outputs.push_back(&port);
    }
  }
  ASSERT_EQ(outputs.size(), 2U);
  const std::array devices{
      detail::TensorDeviceBindingRecord{.port = PortId{outputs[0]->id.value},
                                        .device = DeviceId{0}},
      detail::TensorDeviceBindingRecord{.port = PortId{outputs[1]->id.value},
                                        .device = DeviceId{1}},
  };
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;

  const ExecutionPlan plan = detail::compile_with_runtime_evidence(graph, {}, evidence);

  // The physical phase separately rejects the transform's conflicting input/output device
  // obligations. Cover planning still consumes the frozen per-output facts and must preserve the
  // transfer boundary instead of collapsing it before that later diagnostic is published.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"));
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  EXPECT_TRUE(std::any_of(plan.partition_decisions().begin(),
                          plan.partition_decisions().end(),
                          [](const auto& decision) {
                            return std::find(
                                       decision.reasons.begin(),
                                       decision.reasons.end(),
                                       PartitionDecisionReason::kDeviceTransferIncompatibility) !=
                                   decision.reasons.end();
                          }));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, TensorOutputDevicePropagatesToUnboundConsumerWithoutSpuriousSplit) {
  Graph graph{"automatic-same-device-propagation"};
  const ClockKey root_clock = test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionTensorSourceOperator>("source");
  const auto consumer = graph.embed<PartitionTensorSinkGraph>("consumer");
  graph.bind_clock(consumer, consumer->clock_import(), root_clock);
  graph.add_flow(source->output, consumer->input);

  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "output", .device = DeviceId{1}});
  const ExecutionPlan plan = compile(graph, std::move(options));

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  ASSERT_EQ(plan.partition_decisions().size(), 2U);
  const auto sink_decision = std::find_if(
      plan.partition_decisions().begin(), plan.partition_decisions().end(), [](const auto& row) {
        return row.decision == PartitionDecisionKind::kJoin;
      });
  ASSERT_NE(sink_decision, plan.partition_decisions().end());
  EXPECT_EQ(sink_decision->decision, PartitionDecisionKind::kJoin);
  EXPECT_EQ(sink_decision->reasons, std::vector{PartitionDecisionReason::kLinearChainJoin});
  EXPECT_TRUE(std::none_of(
      plan.partition_decisions().begin(),
      plan.partition_decisions().end(),
      [](const auto& decision) {
        return std::find(decision.reasons.begin(),
                         decision.reasons.end(),
                         PartitionDecisionReason::kDeviceTransferIncompatibility) !=
                   decision.reasons.end() ||
               std::find(decision.reasons.begin(),
                         decision.reasons.end(),
                         PartitionDecisionReason::kCollapseRejected) != decision.reasons.end();
      }));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, MultiOutputDeviceFactsSplitCanonicallyWithConcreteDeviceReasons) {
  Graph graph{"automatic-multi-output-device-facts"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionMultiDeviceTensorSourceOperator>("source");
  const auto first_sink = graph.op<PartitionTensorSinkOperator>("first-sink");
  const auto second_sink = graph.op<PartitionTensorSinkOperator>("second-sink");
  graph.add_flow(source->first, first_sink->input);
  graph.add_flow(source->second, second_sink->input);

  const auto compile_order = [&](bool reverse) {
    CompileOptions options;
    if (reverse) {
      options.deployment
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "source", .output_port = "second", .device = DeviceId{1}})
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "source", .output_port = "first", .device = DeviceId{0}});
    } else {
      options.deployment
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "source", .output_port = "first", .device = DeviceId{0}})
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "source", .output_port = "second", .device = DeviceId{1}});
    }
    return compile(graph, std::move(options));
  };

  const ExecutionPlan forward = compile_order(false);
  const ExecutionPlan reverse = compile_order(true);

  EXPECT_FALSE(forward.ok());
  EXPECT_FALSE(reverse.ok());
  EXPECT_TRUE(test::has_diagnostic(forward, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"));
  EXPECT_TRUE(test::has_diagnostic(reverse, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"));
  ASSERT_EQ(forward.execution_partitions().size(), 3U);
  ASSERT_EQ(reverse.execution_partitions().size(), 3U);
  EXPECT_EQ(partition_evidence(forward), partition_evidence(reverse));
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
  EXPECT_EQ(std::count_if(forward.partition_decisions().begin(),
                          forward.partition_decisions().end(),
                          [](const auto& decision) {
                            return std::find(
                                       decision.reasons.begin(),
                                       decision.reasons.end(),
                                       PartitionDecisionReason::kDeviceTransferIncompatibility) !=
                                   decision.reasons.end();
                          }),
            2);
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, IntrinsicMultiDeviceConflictSplitsUnconstrainedScalarEdge) {
  Graph graph{"automatic-intrinsic-multi-device-scalar-edge"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionMultiDeviceScalarSourceOperator>("source");
  const auto sink = graph.op<test::CompileTestOperator>("sink");
  graph.add_flow(source->scalar, sink->input);

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "first", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "second", .device = DeviceId{1}});
  const ExecutionPlan plan = compile(graph, std::move(options));

  // The later physical phase rejects the source's conflicting Tensor output obligations. The
  // automatic cover must nevertheless retain the intrinsic conflict at its scalar-only edge to
  // the otherwise unconstrained sink instead of collapsing both operators into one partition.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"));
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.schedule_edges().size(), 1U);
  const auto sink_decision = std::ranges::find(plan.partition_decisions(),
                                               plan.schedule_edges().front().destination,
                                               &PartitionDecisionRecord::partition);
  ASSERT_NE(sink_decision, plan.partition_decisions().end());
  EXPECT_EQ(sink_decision->decision, PartitionDecisionKind::kBoundarySplit);
  EXPECT_NE(std::ranges::find(sink_decision->reasons,
                              PartitionDecisionReason::kDeviceTransferIncompatibility),
            sink_decision->reasons.end());
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ExactAnchorRejectsConflictingMultiOutputDeviceFacts) {
  Graph graph{"automatic-remainder-multi-output-device-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PartitionMultiDeviceTensorSourceOperator>("source");
  const auto first_sink = graph.op<PartitionTensorSinkOperator>("first-sink");
  const auto second_sink = graph.op<PartitionTensorSinkOperator>("second-sink");
  graph.add_flow(source->first, first_sink->input);
  graph.add_flow(source->second, second_sink->input);
  graph.partition("anchor").add(source);
  graph.auto_partition_remaining();

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "first", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "second", .device = DeviceId{1}});
  const ExecutionPlan plan = compile(graph, std::move(options));

  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
  EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_TRUE(plan.execution_partitions().empty());
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ExactAnchorRejectsTensorAndGreenContextDeviceMismatchCanonically) {
  Graph graph{"automatic-remainder-tensor-green-context-device-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto conflict = graph.op<PartitionTensorSourceOperator>("conflict");
  const auto stable = graph.op<PartitionTensorSourceOperator>("stable");
  (void)graph.op<test::CompileTestOperator>("remainder");
  graph.partition("conflict").add(conflict);
  graph.partition("stable").add(stable);
  graph.auto_partition_remaining();

  const std::array capabilities{
      detail::CudaGreenContextCapabilityEvidence{
          .device = DeviceId{0},
          .deployment_device_id = "gpu.test.uuid.anchor.0",
          .total_sm_count = 80U,
          .min_sm_partition_size = 8U,
          .sm_coscheduled_alignment = 8U,
          .max_structured_context_count = 10U,
          .evidence_id = "test.cuda.green_context.capability.anchor.0/v1",
          .probe_status = PlacementProbeStatus::kPassed,
      },
      detail::CudaGreenContextCapabilityEvidence{
          .device = DeviceId{1},
          .deployment_device_id = "gpu.test.uuid.anchor.1",
          .total_sm_count = 80U,
          .min_sm_partition_size = 8U,
          .sm_coscheduled_alignment = 8U,
          .max_structured_context_count = 10U,
          .evidence_id = "test.cuda.green_context.capability.anchor.1/v1",
          .probe_status = PlacementProbeStatus::kPassed,
      },
  };
  const auto compile_order = [&](bool reverse) {
    CompileOptions options;
    if (reverse) {
      options.deployment
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "stable", .output_port = "output", .device = DeviceId{1}})
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "conflict", .output_port = "output", .device = DeviceId{0}})
          .bind_cuda_green_context("stable", {.device = DeviceId{1}, .sm_count = 16U})
          .bind_cuda_green_context("conflict", {.device = DeviceId{1}, .sm_count = 16U});
    } else {
      options.deployment
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "conflict", .output_port = "output", .device = DeviceId{0}})
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "stable", .output_port = "output", .device = DeviceId{1}})
          .bind_cuda_green_context("conflict", {.device = DeviceId{1}, .sm_count = 16U})
          .bind_cuda_green_context("stable", {.device = DeviceId{1}, .sm_count = 16U});
    }
    auto evidence = detail::runtime_compile_evidence_view();
    evidence.cuda_green_context_capabilities = capabilities;
    return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
  };

  const ExecutionPlan forward = compile_order(false);
  const ExecutionPlan reverse = compile_order(true);

  EXPECT_FALSE(forward.ok());
  EXPECT_FALSE(reverse.ok());
  EXPECT_TRUE(test::has_diagnostic(forward, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
  EXPECT_TRUE(test::has_diagnostic(reverse, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
  EXPECT_EQ(forward.partitioning_plan().reason,
            detail::diagnostic_id("PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
  EXPECT_EQ(forward.partitioning_plan().reason, reverse.partitioning_plan().reason);
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ExactNamedAnchorKeepsOnlyItsAuthoredMembers) {
  Graph graph{"automatic-remainder-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  const auto third = graph.op<test::CompileTestOperator>("third");
  const auto fourth = graph.op<test::CompileTestOperator>("fourth");
  graph.add_flow(first->output, second->input);
  graph.add_flow(second->output, third->input);
  graph.add_flow(third->output, fourth->input);
  graph.partition("anchor").add(second, third);
  graph.auto_partition_remaining();

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  const auto canonical_id = [&](std::string_view name) {
    const auto found =
        std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [&](const auto& vertex) {
          return vertex.kind == VertexKind::kOperator && vertex.name == name;
        });
    return found == snapshot->vertices.end() ? VertexId{} : found->id;
  };

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
  const auto anchor =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(anchor, plan.execution_partitions().end());
  EXPECT_EQ(anchor->members,
            std::vector<VertexId>({canonical_id("second"), canonical_id("third")}));
  EXPECT_EQ(plan.execution_partitions().size(), 3U);
  const auto partitions = plan.partitions();
  EXPECT_EQ(partitions.mode(), PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(partitions.authored_count(), 1U);
  EXPECT_EQ(partitions.automatic_count(), 2U);
  const auto named = partitions.lookup_named("anchor");
  ASSERT_TRUE(named);
  EXPECT_EQ(named->origin(), ExecutionPartitionOrigin::kAuthored);
  EXPECT_TRUE(std::ranges::equal(named->operator_paths(), anchor->operator_paths));
  const auto automatic = partitions.lookup_for_path("first");
  ASSERT_TRUE(automatic);
  EXPECT_EQ(automatic->origin(), ExecutionPartitionOrigin::kAutomatic);
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, AutomaticRemainderCannotBeBoundByInventedNames) {
  const auto compile_with_binding = [](bool green_context) {
    Graph graph{"automatic-remainder-invented-placement"};
    test::add_compile_test_default_clock(graph);
    const auto first = graph.op<test::CompileTestOperator>("first");
    (void)graph.op<ClosedIslandSourceOperator>("second");
    graph.partition("anchor").add(first);
    graph.auto_partition_remaining();
    CompileOptions options;
    if (green_context) {
      options.deployment.bind_cuda_green_context("automatic#1",
                                                 {.device = DeviceId{0}, .sm_count = 16U});
    } else {
      options.deployment.bind_thread_placement("automatic#1", ThreadPlacement{});
    }
    return compile(graph, std::move(options));
  };

  const ExecutionPlan thread = compile_with_binding(false);
  EXPECT_FALSE(thread.ok());
  EXPECT_TRUE(test::has_diagnostic(thread, "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE"));
  const ExecutionPlan green_context = compile_with_binding(true);
  EXPECT_FALSE(green_context.ok());
  EXPECT_TRUE(test::has_diagnostic(green_context, "CUDA_GREEN_CONTEXT_UNKNOWN_OR_DUPLICATE"));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ConstrainedAnchorOwnsOnlyUnambiguousSyntheticEndpoint) {
  Graph graph{"automatic-remainder-synthetic-anchor"};
  test::add_compile_test_default_clock(graph);
  auto imported = graph.import_topic<std::uint32_t>("holomq://automatic-anchor-input");
  const auto sink = graph.op<test::CompileTestOperator>("sink");
  (void)graph.op<ClosedIslandSourceOperator>("automatic-root");
  graph.add_flow(imported.output(), sink->input);
  graph.partition("anchor").add(sink);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  const auto anchor =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(anchor, plan.execution_partitions().end());
  EXPECT_EQ(anchor->members.size(), 2U);
  EXPECT_EQ(std::count_if(anchor->members.begin(),
                          anchor->members.end(),
                          [&](VertexId member) {
                            const auto found = std::find_if(
                                plan.graph().vertices().begin(),
                                plan.graph().vertices().end(),
                                [&](const auto& vertex) { return vertex.id == member; });
                            return found != plan.graph().vertices().end() &&
                                   found->kind == VertexKind::kImportedTopic;
                          }),
            1);
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ConstrainedAnchorOwnsUnambiguousExportedEndpoint) {
  Graph graph{"automatic-remainder-synthetic-export-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<ScalarClockSourceOperator>("source");
  auto exported = graph.export_topic<std::uint32_t>("holomq://automatic-anchor-output");
  (void)graph.op<ClosedIslandSourceOperator>("automatic-root");
  graph.add_flow(source->output, exported.input());
  graph.partition("anchor").add(source);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  const auto anchor =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(anchor, plan.execution_partitions().end());
  EXPECT_EQ(anchor->members.size(), 2U);
  EXPECT_EQ(std::count_if(anchor->members.begin(),
                          anchor->members.end(),
                          [&](VertexId member) {
                            const auto found = std::find_if(
                                plan.graph().vertices().begin(),
                                plan.graph().vertices().end(),
                                [&](const auto& vertex) { return vertex.id == member; });
                            return found != plan.graph().vertices().end() &&
                                   found->kind == VertexKind::kExportedTopic;
                          }),
            1);
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, CrossProcessSyntheticEndpointsRemainSeparateFromExactAnchors) {
  const auto verify = [](const ExecutionPlan& plan, VertexKind endpoint_kind) {
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.execution_partitions().size(), 3U);
    ASSERT_EQ(plan.schedule_edges().size(), 1U);
    const auto endpoint_vertex = std::find_if(
        plan.graph().vertices().begin(), plan.graph().vertices().end(), [&](const auto& vertex) {
          return vertex.kind == endpoint_kind;
        });
    ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
    const auto endpoint_partition =
        std::find_if(plan.execution_partitions().begin(),
                     plan.execution_partitions().end(),
                     [&](const auto& partition) {
                       return partition.members == std::vector<VertexId>{endpoint_vertex->id};
                     });
    ASSERT_NE(endpoint_partition, plan.execution_partitions().end());
    EXPECT_FALSE(endpoint_partition->authored_name.has_value());
    EXPECT_TRUE(endpoint_partition->operator_paths.empty());
    const auto anchor = std::find_if(plan.execution_partitions().begin(),
                                     plan.execution_partitions().end(),
                                     [](const auto& row) { return row.authored_name == "anchor"; });
    ASSERT_NE(anchor, plan.execution_partitions().end());
    EXPECT_NE(anchor->id, endpoint_partition->id);
    EXPECT_TRUE((plan.schedule_edges().front().source == endpoint_partition->id &&
                 plan.schedule_edges().front().destination == anchor->id) ||
                (plan.schedule_edges().front().source == anchor->id &&
                 plan.schedule_edges().front().destination == endpoint_partition->id));
    const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
    const auto endpoint_placement =
        std::find_if(placements.begin(), placements.end(), [&](const auto& row) {
          return row.vertex == endpoint_vertex->id;
        });
    const auto anchor_placement =
        std::find_if(placements.begin(), placements.end(), [&](const auto& row) {
          return std::find(anchor->members.begin(), anchor->members.end(), row.vertex) !=
                 anchor->members.end();
        });
    ASSERT_NE(endpoint_placement, placements.end());
    ASSERT_NE(anchor_placement, placements.end());
    EXPECT_NE(endpoint_placement->process_group, anchor_placement->process_group);
    const auto endpoint_decision = std::ranges::find(
        plan.partition_decisions(), endpoint_vertex->id, &PartitionDecisionRecord::vertex);
    ASSERT_NE(endpoint_decision, plan.partition_decisions().end());
    EXPECT_EQ(endpoint_decision->decision, PartitionDecisionKind::kBoundarySplit);
    EXPECT_NE(
        std::ranges::find(endpoint_decision->reasons, PartitionDecisionReason::kProcessBoundary),
        endpoint_decision->reasons.end());
    EXPECT_NE(std::ranges::find(endpoint_decision->reasons,
                                PartitionDecisionReason::kExternalEndpointBoundary),
              endpoint_decision->reasons.end());
    EXPECT_FALSE(test::has_diagnostic(plan, "PARTITION_RESOURCE_REALIZATION_FAILED"));
  };

  {
    constexpr std::string_view kEndpoint{"holomq://automatic-cross-process-import"};
    Graph graph{"automatic-remainder-cross-process-import"};
    test::add_compile_test_default_clock(graph);
    auto imported = graph.import_topic<std::uint32_t>(std::string{kEndpoint});
    const auto sink = graph.op<test::CompileTestOperator>("sink");
    (void)graph.op<ClosedIslandSourceOperator>("automatic-root");
    graph.add_flow(imported.output(), sink->input);
    graph.partition("anchor").add(sink);
    graph.auto_partition_remaining();
    verify(detail::compile_with_process_placement(
               graph, CompileOptions{}, external_process_placement(kEndpoint)),
           VertexKind::kImportedTopic);
  }
  {
    constexpr std::string_view kEndpoint{"holomq://automatic-cross-process-export"};
    Graph graph{"automatic-remainder-cross-process-export"};
    test::add_compile_test_default_clock(graph);
    const auto source = graph.op<ScalarClockSourceOperator>("source");
    auto exported = graph.export_topic<std::uint32_t>(std::string{kEndpoint});
    (void)graph.op<ClosedIslandSourceOperator>("automatic-root");
    graph.add_flow(source->output, exported.input());
    graph.partition("anchor").add(source);
    graph.auto_partition_remaining();
    verify(detail::compile_with_process_placement(
               graph, CompileOptions{}, external_process_placement(kEndpoint)),
           VertexKind::kExportedTopic);
  }
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring,
     ConstrainedSyntheticEndpointWithMixedAdjacencyKeepsExternalBoundary) {
  Graph graph{"automatic-remainder-ambiguous-synthetic-anchor"};
  test::add_compile_test_default_clock(graph);
  auto imported = graph.import_topic<std::uint32_t>("holomq://automatic-mixed-input");
  const auto anchored = graph.op<test::CompileTestOperator>("anchored");
  const auto automatic = graph.op<test::CompileTestOperator>("automatic");
  graph.add_flow(imported.output(), anchored->input);
  graph.add_flow(imported.output(), automatic->input);
  graph.partition("anchor").add(anchored);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  // The inherited external-channel contract admits one boundary-to-Operator relation only, so
  // imported-topic fan-out remains non-runnable. Partition planning must nevertheless retain the
  // ambiguous endpoint boundary and its complete crossing-edge evidence in the rejected plan.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "EXTERNAL_CHANNEL_PLAN_INVALID"));
  EXPECT_EQ(plan.partitioning_plan().disposition, PlanDisposition::kAccepted);
  ASSERT_EQ(plan.execution_partitions().size(), 3U);
  const auto anchor =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(anchor, plan.execution_partitions().end());
  EXPECT_EQ(anchor->operator_paths, std::vector<std::string>{"anchored"});
  EXPECT_EQ(anchor->members.size(), 1U);
  const auto endpoint_partition = std::find_if(plan.execution_partitions().begin(),
                                               plan.execution_partitions().end(),
                                               [](const auto& partition) {
                                                 return !partition.authored_name.has_value() &&
                                                        partition.operator_paths.empty();
                                               });
  ASSERT_NE(endpoint_partition, plan.execution_partitions().end());
  ASSERT_EQ(endpoint_partition->members.size(), 1U);
  const auto endpoint_vertex = std::find_if(
      plan.graph().vertices().begin(), plan.graph().vertices().end(), [&](const auto& vertex) {
        return vertex.id == endpoint_partition->members.front();
      });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  EXPECT_EQ(endpoint_vertex->kind, VertexKind::kImportedTopic);
  ASSERT_EQ(plan.schedule_edges().size(), 2U);
  EXPECT_EQ(plan.schedule_edges()[0].edge, plan.graph().edges()[0].id);
  EXPECT_EQ(plan.schedule_edges()[1].edge, plan.graph().edges()[1].id);
  try {
    (void)plan.partitions();
    FAIL() << "rejected plan unexpectedly published a partition view";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
  }
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, RootIncludeAllExpandsAcrossEmbeddedDescendants) {
  Graph graph{"include-all-composed-root"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<test::CompileTestOperator>("root");
  (void)graph.embed<IncludeAllEmbeddedGraph>("child");
  graph.partition("application").include_all();

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 2U);
  EXPECT_TRUE(plan.schedule_edges().empty());
}

// REQ: COMPILE-068
// REQ: GRAPH-032
TEST(CompilePartitionAuthoring, IncludeAllOwnsDisconnectedExecutableEndpoints) {
  const auto verify_endpoint_only = [](bool imported) {
    Graph graph{imported ? "include-all-standalone-import" : "include-all-standalone-export"};
    if (imported) {
      (void)graph.import_topic<std::uint32_t>("holomq://include-all-standalone-import");
    } else {
      (void)graph.export_topic<std::uint32_t>("holomq://include-all-standalone-export");
    }
    graph.partition("application").include_all();
    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.execution_partitions().size(), 1U);
    EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kManual);
    EXPECT_EQ(plan.execution_partitions().front().authored_name,
              std::optional<std::string>{"application"});
    ASSERT_EQ(plan.execution_partitions().front().members.size(), 1U);
    EXPECT_TRUE(plan.execution_partitions().front().operator_paths.empty());
    EXPECT_TRUE(plan.schedule_edges().empty());
  };
  verify_endpoint_only(true);
  verify_endpoint_only(false);

  Graph graph{"include-all-disconnected-mixed"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<ClosedIslandSourceOperator>("operator");
  (void)graph.import_topic<std::uint32_t>("holomq://include-all-disconnected-mixed");
  graph.partition("application").include_all();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 2U);
  EXPECT_EQ(plan.execution_partitions().front().operator_paths,
            std::vector<std::string>{"operator"});
}

// REQ: COMPILE-068
// REQ: GRAPH-032
TEST(CompilePartitionAuthoring, IncludeAllRejectsCrossProcessExecutableEndpoints) {
  constexpr std::string_view kEndpoint{"holomq://include-all-cross-process"};
  Graph graph{"include-all-cross-process"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<ClosedIslandSourceOperator>("operator");
  (void)graph.import_topic<std::uint32_t>(std::string{kEndpoint});
  graph.partition("application").include_all();

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, external_process_placement(kEndpoint));

  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  EXPECT_FALSE(test::has_diagnostic(plan, "PARTITION_RESOURCE_REALIZATION_FAILED"));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, AutomaticRemainderRequiresOneNonemptyExactAnchor) {
  Graph graph{"automatic-remainder-empty"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<test::CompileTestOperator>("operator");
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "GRAPH_PARTITION_AUTOMATIC_REMAINDER_EMPTY"));
  EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
}

// A request with exact authored intent remains constrained automatic even when that anchor covers
// every logical Operator. The zero automatic count is intentional evidence, not a missing cover.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, CompleteExactAnchorPreservesConstrainedAutomaticIntent) {
  Graph graph{"automatic-remainder-complete-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto sink = graph.op<test::CompileTestOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.partition("anchor").add(source, sink);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 0U);
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"anchor"});
}

// Exact anchors retain the landed one-independent-root legality rule even when an automatic
// remainder exists. Automatic planning may not make an illegal authored serial owner acceptable.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ConstrainedAutomaticRejectsMultiRootExactAnchor) {
  Graph graph{"automatic-remainder-multi-root-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  (void)graph.op<ClosedIslandSourceOperator>("automatic-root");
  graph.partition("invalid-anchor").add(first, second);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, IncompleteManualCoverStillRequiresExplicitAutomaticRemainder) {
  Graph graph{"incomplete-manual-cover"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<test::CompileTestOperator>("first");
  (void)graph.op<test::CompileTestOperator>("second");
  graph.partition("only-first").add(first);

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, RejectsNonConvexExactAnchor) {
  Graph graph{"non-convex-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto middle = graph.op<test::CompileTestOperator>("middle");
  const auto last = graph.op<test::CompileTestOperator>("last");
  graph.add_flow(first->output, middle->input);
  graph.add_flow(middle->output, last->input);
  graph.partition("leave-and-reenter").add(first, last);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ANCHOR_NON_CONVEX"));
}

// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, RejectsExactAnchorAcrossProcessDomains) {
  Graph graph{"cross-process-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
  graph.partition("cross-process").add(source, target);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, process_placement("source-process", "target-process"));
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
}

// REQ: COMPILE-058
// REQ: COMPILE-059
// REQ: COMPILE-101
TEST(CompilePartitionAuthoring, CanonicalizesManualDeclarationMemberAndPlacementOrder) {
  const ExecutionPlan forward = compile_canonical_manual_partitions(false);
  const ExecutionPlan reverse = compile_canonical_manual_partitions(true);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();

  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
  EXPECT_EQ(partition_evidence(forward), partition_evidence(reverse));
  ASSERT_EQ(forward.execution_partitions().size(), 2U);
  EXPECT_EQ(forward.execution_partitions()[0].id, ExecutionPartitionId{1U});
  EXPECT_EQ(forward.execution_partitions()[0].authored_name, std::optional<std::string>{"alpha"});
  EXPECT_EQ(forward.execution_partitions()[1].id, ExecutionPartitionId{2U});
  EXPECT_EQ(forward.execution_partitions()[1].authored_name, std::optional<std::string>{"beta"});
  ASSERT_EQ(forward.schedule_edges().size(), 1U);
  EXPECT_EQ(forward.schedule_edges().front().source, ExecutionPartitionId{1U});
  EXPECT_EQ(forward.schedule_edges().front().destination, ExecutionPartitionId{2U});
}

// Mirrors Jay's two-source fusion/fanout execution shape:
//
//   camera -> camera-pre --\
//                           fusion -> display -> visualization
//   imu    -> imu-pre    --/      \-> inference -> storage
//
// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, LowersFourManualFusionExecutionGroups) {
  const ExecutionPlan plan = compile_manual_fusion_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 4U);
  EXPECT_EQ(plan.execution_partitions()[0].authored_name, std::optional<std::string>{"camera"});
  EXPECT_EQ(plan.execution_partitions()[1].authored_name,
            std::optional<std::string>{"fusion-display"});
  EXPECT_EQ(plan.execution_partitions()[2].authored_name, std::optional<std::string>{"imu"});
  EXPECT_EQ(plan.execution_partitions()[3].authored_name,
            std::optional<std::string>{"inference-storage"});
  EXPECT_EQ(plan.schedule_edges().size(), 3U);
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  // The assertion above establishes that the optional is engaged.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 4U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 4U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 4U);
  EXPECT_EQ(plan.scheduler_plan().intra_path_parallelism, 1U);

  const auto& capacities = plan.profile_manifest().derived_capacities;
  const auto scheduler_capacity =
      std::find_if(capacities.begin(), capacities.end(), [](const auto& capacity) {
        return capacity.object_id == "scheduler" &&
               capacity.capacity_id == "max_concurrent_activations";
      });
  ASSERT_NE(scheduler_capacity, capacities.end());
  EXPECT_EQ(scheduler_capacity->limit, 4U);

  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->partition_runners.size(), 4U);
  for (std::size_t index = 0U; index < state->partition_runners.size(); ++index) {
    const auto& runner = state->partition_runners[index];
    EXPECT_EQ(runner.ordinal, static_cast<std::uint32_t>(index));
    EXPECT_EQ(runner.partition, plan.execution_partitions()[index].id);
    EXPECT_EQ(runner.execution_lane, plan.execution_partitions()[index].execution_lane);
    ASSERT_FALSE(runner.operators.empty());
    EXPECT_TRUE(runner.operators.front().topology_root);
    EXPECT_EQ(std::count_if(runner.operators.begin(),
                            runner.operators.end(),
                            [](const auto& member) { return member.topology_root; }),
              1);
  }
  EXPECT_EQ(state->partition_runners[0].operators.size(), 2U);
  EXPECT_EQ(state->partition_runners[0].internal_edges.size(), 1U);
  EXPECT_TRUE(state->partition_runners[0].incoming_boundary_edges.empty());
  EXPECT_EQ(state->partition_runners[0].outgoing_boundary_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[1].operators.size(), 3U);
  EXPECT_EQ(state->partition_runners[1].internal_edges.size(), 2U);
  EXPECT_EQ(state->partition_runners[1].incoming_boundary_edges.size(), 2U);
  EXPECT_EQ(state->partition_runners[1].outgoing_boundary_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[2].operators.size(), 2U);
  EXPECT_EQ(state->partition_runners[2].internal_edges.size(), 1U);
  EXPECT_TRUE(state->partition_runners[2].incoming_boundary_edges.empty());
  EXPECT_EQ(state->partition_runners[2].outgoing_boundary_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[3].operators.size(), 2U);
  EXPECT_EQ(state->partition_runners[3].internal_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[3].incoming_boundary_edges.size(), 1U);
  EXPECT_TRUE(state->partition_runners[3].outgoing_boundary_edges.empty());
}

// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, CanonicalizesFourManualPartitionPublicProjections) {
  const ExecutionPlan forward = compile_manual_fusion_partitions(false);
  const ExecutionPlan reverse = compile_manual_fusion_partitions(true);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();

  // Key assertion: private partition-runner reconstruction authority does not leak into the
  // canonical public artifact. Authoring-order differences preserve public JSON, digest, and
  // profile capacity evidence for the same four-partition cover.
  EXPECT_EQ(forward.json(), reverse.json());
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(partition_evidence(forward), partition_evidence(reverse));
  EXPECT_EQ(profile_capacity_evidence(forward), profile_capacity_evidence(reverse));
  EXPECT_EQ(forward.json().find("partition_runners"), std::string::npos);
  EXPECT_EQ(reverse.json().find("partition_runners"), std::string::npos);

  const auto scheduler_capacity =
      std::find_if(forward.profile_manifest().derived_capacities.begin(),
                   forward.profile_manifest().derived_capacities.end(),
                   [](const auto& capacity) {
                     return capacity.object_id == "scheduler" &&
                            capacity.capacity_id == "max_concurrent_activations";
                   });
  ASSERT_NE(scheduler_capacity, forward.profile_manifest().derived_capacities.end());
  EXPECT_EQ(scheduler_capacity->limit, 4U);
}

// A closed-island composite is owned by one partition runner. Even if two
// partitions share a physical lane, the compiler must leave their boundary
// edge on the ordinary controller path rather than mint cross-owner authority.
TEST(CompilePartitionAuthoring, ClosedIslandCompositeDoesNotCrossPartitionIdentity) {
  Graph graph{"closed-island-partition-boundary"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<ClosedIslandSourceOperator>("source");
  const auto sink = graph.op<ClosedIslandSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("pipeline").add(source).add(sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* const compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  const auto baseline = detail::derive_closed_island_activation_plan(*compiled);
  ASSERT_EQ(baseline.size(), 1U);
  ASSERT_EQ(baseline.front().source_partition, baseline.front().successor_partition);

  detail::ExecutionPlanState split = *compiled;
  split.execution_partitions = {
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .authored_name = std::string{"source"},
          .members = {baseline.front().source},
          .execution_lane = baseline.front().execution_lane,
      },
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U},
          .authored_name = std::string{"sink"},
          .members = {baseline.front().successor},
          .execution_lane = baseline.front().execution_lane,
      },
  };

  EXPECT_TRUE(detail::derive_closed_island_activation_plan(split).empty());
}

}  // namespace
}  // namespace holoscan
