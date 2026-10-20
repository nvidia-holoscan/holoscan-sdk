/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/time/manual_clock.hpp>

#include "../support/optional_assertions.hpp"
#include "compiler/closed_island_activation_plan.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/cuda_execution_plan_validation.hpp"
#include "compiler/cuda_graph_execution_lowering.hpp"
#include "compiler/execution_plan_identity.hpp"
#include "compiler/execution_plan_lookup.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/inline_partition_execution_plan.hpp"
#include "compiler/partition_runner_plan.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_generation.hpp"
#include "compiler/plan_lowering_diagnostic.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"

namespace holoscan {

struct CUDAGraphFastPathPod {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<CUDAGraphFastPathPod> {
  static constexpr std::string_view name = "holoscan.test.CUDAGraphFastPathPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CUDAGraphFastPathPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &CUDAGraphFastPathPod::value, offsetof(CUDAGraphFastPathPod, value)));
};

namespace {

using namespace std::chrono_literals;

constexpr std::array<std::int64_t, 1U> kFusedRegionTensorShape{1024};
constexpr std::array<std::uint64_t, 1U> kFusedRegionTensorStrides{2U};

class FusedRegionTensorOperator final : public Operator<> {
 public:
  explicit FusedRegionTensorOperator(
      bool independent_source = false, MemoryKind input_memory_kind = MemoryKind::kCudaDevice,
      MemoryKind output_memory_kind = MemoryKind::kCudaDevice,
      TensorOutputStorage output_storage = TensorOutputStorage::kRuntimePool)
      : independent_source_(independent_source),
        input_memory_kind_(input_memory_kind),
        output_memory_kind_(output_memory_kind),
        output_storage_(output_storage) {}

  void setup(OperatorSpec& spec) override {
    const TensorRepresentation representation{
        .memory_kind = input_memory_kind_,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
        .strides = TensorStridePolicy::kContiguousOnly,
    };
    spec.input(input, "input").expects_tensor(TensorInputSpec{.representation = representation});
    auto output_representation = representation;
    output_representation.memory_kind = output_memory_kind_;
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = output_representation,
            .bounds = tensor_bounds(kFusedRegionTensorShape),
            .storage = output_storage_,
        });
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    if (independent_source_) {
      result.trigger(OnClock{.period = 1ms});
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;

 private:
  bool independent_source_{};
  MemoryKind input_memory_kind_{MemoryKind::kCudaDevice};
  MemoryKind output_memory_kind_{MemoryKind::kCudaDevice};
  TensorOutputStorage output_storage_{TensorOutputStorage::kRuntimePool};
};

class FusedRegionTensorSourceOperator final : public Operator<> {
 public:
  explicit FusedRegionTensorSourceOperator(
      MemoryKind output_memory_kind = MemoryKind::kCudaDevice,
      TensorOutputStorage output_storage = TensorOutputStorage::kRuntimePool)
      : output_memory_kind_(output_memory_kind), output_storage_(output_storage) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = output_memory_kind_,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                    .strides = TensorStridePolicy::kContiguousOnly,
                },
            .bounds = tensor_bounds(kFusedRegionTensorShape),
            .storage = output_storage_,
        });
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  MemoryKind output_memory_kind_{MemoryKind::kCudaDevice};
  TensorOutputStorage output_storage_{TensorOutputStorage::kRuntimePool};
};

class FusedRegionRelaxedTensorOperator final : public Operator<> {
 public:
  explicit FusedRegionRelaxedTensorOperator(bool constrain_input_memory)
      : constrain_input_memory_(constrain_input_memory) {}

  void setup(OperatorSpec& spec) override {
    auto& input_spec = spec.input(input, "input");
    if (constrain_input_memory_) {
      input_spec.expects_tensor(TensorInputSpec{
          .representation = TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice},
      });
    }
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaDevice,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                    .strides = TensorStridePolicy::kContiguousOnly,
                },
            .bounds = tensor_bounds(kFusedRegionTensorShape),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;

 private:
  bool constrain_input_memory_{};
};

class FusedRegionTensorSinkOperator final : public Operator<> {
 public:
  explicit FusedRegionTensorSinkOperator(MemoryKind input_memory_kind = MemoryKind::kCudaDevice)
      : input_memory_kind_(input_memory_kind) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = input_memory_kind_,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                    .strides = TensorStridePolicy::kContiguousOnly,
                },
        });
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;

 private:
  MemoryKind input_memory_kind_{MemoryKind::kCudaDevice};
};

template <TensorStridePolicy Strides, std::uint32_t Rank>
class FusedRegionGeometrySourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    TensorOutputSpec output_spec{
        .representation =
            TensorRepresentation{
                .memory_kind = MemoryKind::kCudaDevice,
                .dtype = DLDataType{kDLUInt, 8U, 1U},
                .rank = Rank,
                .strides = Strides,
            },
        .storage = TensorOutputStorage::kRuntimePool,
    };
    if constexpr (Rank == 0U) {
      output_spec.bounds = tensor_bounds(1U);
    } else if constexpr (Strides == TensorStridePolicy::kBoundedStrided) {
      output_spec.bounds =
          tensor_bounds(kFusedRegionTensorShape,
                        TensorBoundsOptions{.max_abs_byte_strides = kFusedRegionTensorStrides});
    } else {
      output_spec.bounds = tensor_bounds(kFusedRegionTensorShape);
    }
    spec.output(output, "output").produces_tensor(output_spec);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

template <TensorStridePolicy Strides, std::uint32_t Rank>
class FusedRegionGeometrySinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    // Bounded-strided instantiations add axis bounds below.
    // NOLINTNEXTLINE(misc-const-correctness)
    TensorInputSpec input_spec{
        .representation =
            TensorRepresentation{
                .memory_kind = MemoryKind::kCudaDevice,
                .dtype = DLDataType{kDLUInt, 8U, 1U},
                .rank = Rank,
                .strides = Strides,
            },
    };
    if constexpr (Strides == TensorStridePolicy::kBoundedStrided) {
      input_spec.bounds =
          tensor_bounds(kFusedRegionTensorShape,
                        TensorBoundsOptions{.max_abs_byte_strides = kFusedRegionTensorStrides});
    }
    spec.input(input, "input").queue_depth(1U).expects_tensor(input_spec);
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

class FusedRegionDirectBoundsSourceOperator final : public Operator<> {
 public:
  FusedRegionDirectBoundsSourceOperator(std::vector<std::uint64_t> fixed_shape,
                                        std::uint8_t dtype_bits, std::uint64_t max_byte_span,
                                        std::vector<std::uint64_t> max_abs_byte_strides)
      : dtype_bits_(dtype_bits), max_byte_span_(max_byte_span) {
    axes_.reserve(fixed_shape.size());
    for (std::size_t axis = 0U; axis < fixed_shape.size(); ++axis) {
      axes_.push_back(TensorAxisBounds{
          .max_extent = fixed_shape[axis],
          .max_abs_stride_bytes = max_abs_byte_strides.empty() ? 0U : max_abs_byte_strides.at(axis),
      });
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaDevice,
                    .dtype = DLDataType{kDLUInt, dtype_bits_, 1U},
                    .rank = static_cast<std::uint32_t>(axes_.size()),
                    .strides = TensorStridePolicy::kContiguousOnly,
                },
            .bounds =
                TensorBounds{
                    .max_byte_span = max_byte_span_,
                    .axes = std::span<const TensorAxisBounds>{axes_},
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  std::vector<TensorAxisBounds> axes_;
  std::uint8_t dtype_bits_{};
  std::uint64_t max_byte_span_{};
};

class FusedRegionDirectBoundsSinkOperator final : public Operator<> {
 public:
  FusedRegionDirectBoundsSinkOperator(std::vector<std::uint64_t> fixed_shape,
                                      std::uint8_t dtype_bits, std::uint64_t max_byte_span,
                                      std::vector<std::uint64_t> max_abs_byte_strides)
      : dtype_bits_(dtype_bits), max_byte_span_(max_byte_span) {
    axes_.reserve(fixed_shape.size());
    for (std::size_t axis = 0U; axis < fixed_shape.size(); ++axis) {
      axes_.push_back(TensorAxisBounds{
          .max_extent = fixed_shape[axis],
          .max_abs_stride_bytes = max_abs_byte_strides.at(axis),
      });
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaDevice,
                    .dtype = DLDataType{kDLUInt, dtype_bits_, 1U},
                    .rank = static_cast<std::uint32_t>(axes_.size()),
                    .strides = TensorStridePolicy::kContiguousOnly,
                },
            .bounds =
                TensorBounds{
                    .max_byte_span = max_byte_span_,
                    .axes = std::span<const TensorAxisBounds>{axes_},
                },
        });
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;

 private:
  std::vector<TensorAxisBounds> axes_;
  std::uint8_t dtype_bits_{};
  std::uint64_t max_byte_span_{};
};

class FusedRegionUnconstrainedTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

class FusedRegionEdgelessOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.cuda_graph_capture_eligible(); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class FastPathSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<CUDAGraphFastPathPod> output;
};

class FastPathRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<CUDAGraphFastPathPod> input;
  Output<CUDAGraphFastPathPod> output;
};

class FastPathSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<CUDAGraphFastPathPod> input;
};

[[nodiscard]] std::vector<detail::TensorDeviceBindingRecord> cuda_device_bindings(
    const detail::GraphSnapshotData& snapshot) {
  std::vector<detail::TensorDeviceBindingRecord> result;
  for (const auto& port : snapshot.ports) {
    const auto memory_kind =
        port.setup.tensor_layout ? port.setup.tensor_layout->memory_kind : std::nullopt;
    if (port.owner_kind == PortOwnerKind::kOperator && port.direction == PortDirection::kOutput &&
        (memory_kind == MemoryKind::kCudaDevice || memory_kind == MemoryKind::kCudaManaged)) {
      result.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value},
          .device = DeviceId{0},
      });
    }
  }
  return result;
}

[[nodiscard]] ExecutionPlan compile_fused_region(
    MemoryKind memory_kind = MemoryKind::kCudaDevice, bool split_partition = false,
    bool add_cuda_graph_region = true,
    TensorOutputStorage source_storage = TensorOutputStorage::kRuntimePool,
    TensorOutputStorage middle_storage = TensorOutputStorage::kRuntimePool,
    std::string_view region_name = "capture", bool reverse_region_members = false) {
  Graph graph{"fused-cuda-graph-region"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionTensorSourceOperator>("source", memory_kind, source_storage);
  auto middle = graph.op<FusedRegionTensorOperator>(
      "middle", false, memory_kind, memory_kind, middle_storage);
  auto sink = graph.op<FusedRegionTensorSinkOperator>("sink", memory_kind);
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  if (add_cuda_graph_region) {
    auto region = graph.region(region_name);
    if (reverse_region_members) {
      region.add(sink).add(middle).add(source);
    } else {
      region.add(source).add(middle).add(sink);
    }
    region.constrain(RegionConstraint::cuda_graph());
  }
  if (split_partition) {
    graph.partition("source-partition").add(source);
    graph.partition("middle-partition").add(middle);
    graph.partition("sink-partition").add(sink);
  }

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

[[nodiscard]] ExecutionPlan compile_fused_region_across_two_cuda_devices() {
  Graph graph{"two-device-fused-cuda-graph-region"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionTensorSourceOperator>("source");
  auto middle = graph.op<FusedRegionTensorOperator>("middle");
  auto sink = graph.op<FusedRegionTensorSinkOperator>("sink");
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());
  graph.partition("captured").add(source).add(middle).add(sink);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  auto devices = cuda_device_bindings(*snapshot);
  if (devices.size() != 2U) {
    return {};
  }
  devices.front().device = DeviceId{0};
  devices.back().device = DeviceId{1};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

[[nodiscard]] ExecutionPlan compile_fused_region_with_boundaries(
    MemoryKind boundary_memory,
    TensorOutputStorage ingress_storage = TensorOutputStorage::kRuntimePool,
    bool ingress_output_has_other_consumer = false,
    TensorOutputStorage egress_storage = TensorOutputStorage::kRuntimePool,
    std::uint32_t ingress_queue_depth = 1U, std::uint32_t internal_queue_depth = 1U,
    std::uint32_t egress_queue_depth = 1U) {
  Graph graph{"fused-cuda-graph-region-boundaries"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto reader = graph.op<FusedRegionTensorOperator>(
      "reader", true, boundary_memory, boundary_memory, ingress_storage);
  auto source = graph.op<FusedRegionTensorOperator>(
      "source", false, boundary_memory, MemoryKind::kCudaDevice);
  auto middle = graph.op<FusedRegionTensorOperator>(
      "middle", false, MemoryKind::kCudaDevice, MemoryKind::kCudaDevice);
  auto sink = graph.op<FusedRegionTensorOperator>(
      "sink", false, MemoryKind::kCudaDevice, boundary_memory, egress_storage);
  auto writer =
      graph.op<FusedRegionTensorOperator>("writer", false, boundary_memory, boundary_memory);
  graph.add_flow(
      reader->output, source->input, ConnectionOptions{.queue_depth = ingress_queue_depth});
  graph.add_flow(
      source->output, middle->input, ConnectionOptions{.queue_depth = internal_queue_depth});
  graph.add_flow(
      middle->output, sink->input, ConnectionOptions{.queue_depth = internal_queue_depth});
  graph.add_flow(sink->output, writer->input, ConnectionOptions{.queue_depth = egress_queue_depth});
  if (ingress_output_has_other_consumer) {
    auto other_consumer =
        graph.op<FusedRegionTensorSinkOperator>("other-consumer", boundary_memory);
    graph.add_flow(reader->output, other_consumer->input);
  }
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

[[nodiscard]] ExecutionPlan compile_fused_region_with_cross_process_ingress() {
  Graph graph{"fused-cuda-graph-region-cross-process-ingress"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto reader = graph.op<FusedRegionTensorOperator>("reader", true);
  auto source = graph.op<FusedRegionTensorOperator>("source");
  auto sink = graph.op<FusedRegionTensorOperator>("sink");
  graph.add_flow(reader->output, source->input);
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;

  constexpr std::array reader_member{detail::GraphObjectSelector{.graph_path = "reader"}};
  constexpr std::array region_members{detail::GraphObjectSelector{.graph_path = "source"},
                                      detail::GraphObjectSelector{.graph_path = "sink"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("reader",
                         detail::ProcessPlacement{
                             .members = reader_member,
                             .launch_group = "reader-process",
                         })
      .bind_process_role("region",
                         detail::ProcessPlacement{
                             .members = region_members,
                             .launch_group = "region-process",
                         });
  return detail::compile_with_runtime_evidence_and_process_placement(
      graph, {}, evidence, std::move(placement));
}

[[nodiscard]] ExecutionPlan compile_fused_region_with_cross_process_egress() {
  Graph graph{"fused-cuda-graph-region-cross-process-egress"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionTensorSourceOperator>("source");
  auto sink = graph.op<FusedRegionTensorOperator>("sink");
  auto writer = graph.op<FusedRegionTensorSinkOperator>("writer");
  graph.add_flow(source->output, sink->input);
  graph.add_flow(sink->output, writer->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;

  constexpr std::array region_members{detail::GraphObjectSelector{.graph_path = "source"},
                                      detail::GraphObjectSelector{.graph_path = "sink"}};
  constexpr std::array writer_member{detail::GraphObjectSelector{.graph_path = "writer"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("region",
                         detail::ProcessPlacement{
                             .members = region_members,
                             .launch_group = "region-process",
                         })
      .bind_process_role("writer",
                         detail::ProcessPlacement{
                             .members = writer_member,
                             .launch_group = "writer-process",
                         });
  return detail::compile_with_runtime_evidence_and_process_placement(
      graph, {}, evidence, std::move(placement));
}

[[nodiscard]] ExecutionPlan compile_fused_region_with_relaxed_input(bool constrain_input_memory) {
  Graph graph{"fused-cuda-graph-region-relaxed-input"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionTensorSourceOperator>("source");
  auto middle = graph.op<FusedRegionRelaxedTensorOperator>("middle", constrain_input_memory);
  auto sink = graph.op<FusedRegionTensorSinkOperator>("sink");
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

template <TensorStridePolicy Strides, std::uint32_t Rank>
[[nodiscard]] ExecutionPlan compile_fused_region_with_ingress_geometry() {
  Graph graph{"fused-cuda-graph-region-ingress-geometry"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionGeometrySourceOperator<Strides, Rank>>("source");
  auto sink = graph.op<FusedRegionGeometrySinkOperator<Strides, Rank>>("sink");
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(sink).constrain(RegionConstraint::cuda_graph());

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

[[nodiscard]] ExecutionPlan compile_fused_region_with_direct_ingress_geometry(
    std::vector<std::uint64_t> fixed_shape, std::uint8_t dtype_bits, std::uint64_t max_byte_span,
    std::vector<std::uint64_t> producer_max_abs_byte_strides = {},
    std::vector<std::uint64_t> consumer_max_abs_byte_strides = {}) {
  Graph graph{"fused-cuda-graph-region-direct-ingress-geometry"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionDirectBoundsSourceOperator>(
      "source", fixed_shape, dtype_bits, max_byte_span, std::move(producer_max_abs_byte_strides));
  if (consumer_max_abs_byte_strides.empty()) {
    auto sink = graph.op<FusedRegionUnconstrainedTensorSinkOperator>("sink");
    graph.add_flow(source->output, sink->input);
    graph.region("capture").add(sink).constrain(RegionConstraint::cuda_graph());
  } else {
    auto sink =
        graph.op<FusedRegionDirectBoundsSinkOperator>("sink",
                                                      std::move(fixed_shape),
                                                      dtype_bits,
                                                      max_byte_span,
                                                      std::move(consumer_max_abs_byte_strides));
    graph.add_flow(source->output, sink->input);
    graph.region("capture").add(sink).constrain(RegionConstraint::cuda_graph());
  }

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

[[nodiscard]] ExecutionPlan compile_single_operator_green_context_region() {
  Graph graph{"single-operator-green-context-cuda-graph-region"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FusedRegionTensorSourceOperator>("source");
  auto captured = graph.op<FusedRegionTensorOperator>("captured");
  auto sink = graph.op<FusedRegionTensorSinkOperator>("sink");
  graph.add_flow(source->output, captured->input);
  graph.add_flow(captured->output, sink->input);
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());
  graph.partition("source").add(source);
  graph.partition("captured").add(captured);
  graph.partition("sink").add(sink);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.single-operator-capture",
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
  options.deployment.bind_cuda_green_context("captured", {.device = DeviceId{0}, .sm_count = 16U});
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

[[nodiscard]] ExecutionPlan compile_edgeless_single_operator_region(bool bind_green_context) {
  Graph graph{"edgeless-single-operator-cuda-graph-region"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto captured = graph.op<FusedRegionEdgelessOperator>("captured");
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());
  graph.partition("captured").add(captured);

  if (!bind_green_context) {
    return compile(graph);
  }

  const std::array capabilities{detail::CudaGreenContextCapabilityEvidence{
      .device = DeviceId{0},
      .deployment_device_id = "gpu.test.uuid.edgeless-single-operator-capture",
      .total_sm_count = 80U,
      .min_sm_partition_size = 8U,
      .sm_coscheduled_alignment = 8U,
      .max_structured_context_count = 10U,
      .evidence_id = "test.cuda.green_context.capability/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  }};
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.cuda_green_context_capabilities = capabilities;
  CompileOptions options;
  options.deployment.bind_cuda_green_context("captured", {.device = DeviceId{0}, .sm_count = 16U});
  return detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
}

[[nodiscard]] ExecutionPlan compile_two_fused_regions_in_one_partition() {
  Graph graph{"two-fused-cuda-graph-regions"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source_a = graph.op<FusedRegionTensorOperator>("source-a", true);
  auto sink_a = graph.op<FusedRegionTensorOperator>("sink-a");
  auto source_b = graph.op<FusedRegionTensorOperator>("source-b");
  auto sink_b = graph.op<FusedRegionTensorOperator>("sink-b");
  graph.add_flow(source_a->output, sink_a->input);
  graph.add_flow(sink_a->output, source_b->input);
  graph.add_flow(source_b->output, sink_b->input);
  graph.region("capture-a").add(source_a).add(sink_a).constrain(RegionConstraint::cuda_graph());
  graph.region("capture-b").add(source_b).add(sink_b).constrain(RegionConstraint::cuda_graph());

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  const auto devices = cuda_device_bindings(*snapshot);
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  return detail::compile_with_runtime_evidence(graph, {}, evidence);
}

[[nodiscard]] ExecutionPlan compile_closed_island_candidate() {
  Graph graph{"closed-island-candidate"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FastPathSourceOperator>("source");
  auto sink = graph.op<FastPathSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  return compile(graph);
}

[[nodiscard]] ExecutionPlan compile_inline_partition_candidate() {
  Graph graph{"inline-partition-candidate"};
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
  auto source = graph.op<FastPathSourceOperator>("source");
  auto relay = graph.op<FastPathRelayOperator>("relay");
  auto sink = graph.op<FastPathSinkOperator>("sink");
  graph.add_flow(source->output, relay->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  return compile(graph);
}

// Add the private records used by fast-path derivation to recognize a fused region. Physical CUDA
// Graph admission is tested separately; these records isolate each fast-path exclusion rule.
[[nodiscard]] bool add_fused_region_records(detail::ExecutionPlanState& state) {
  if (state.execution_partitions.size() != 1U ||
      state.execution_partitions.front().members.empty()) {
    return false;
  }
  constexpr GraphId kGraph{1U};
  constexpr RegionId kRegion{1U};
  detail::CUDAGraphCapturePlanRecord capture{
      .graph = kGraph,
      .region = kRegion,
      .region_name = "capture",
  };
  for (const VertexId member : state.execution_partitions.front().members) {
    capture.members.push_back(detail::CUDAGraphCaptureMemberRecord{.vertex = member});
  }
  for (const auto& edge : state.payload_edges) {
    capture.internal_edges.push_back(detail::CUDAGraphCaptureEdgeRecord{.edge = edge.edge});
  }
  state.cuda_graph_capture_plans.push_back(std::move(capture));
  state.cuda_graph_execution_regions.push_back(detail::CUDAGraphExecutionRegionPlanRecord{
      .graph = kGraph,
      .region = kRegion,
      .fused_operator = state.execution_partitions.front().members.front(),
  });
  return true;
}

[[nodiscard]] bool has_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  return std::any_of(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
    return item.token == token;
  });
}

[[nodiscard]] const detail::PlanLoweringDiagnostic* find_execution_rule(
    const detail::CUDAGraphExecutionLoweringResult& result, std::string_view rule) {
  const auto diagnostic =
      std::find_if(result.errors.begin(), result.errors.end(), [&](const auto& candidate) {
        return candidate.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               candidate.required == rule;
      });
  return diagnostic == result.errors.end() ? nullptr : &*diagnostic;
}

[[nodiscard]] bool has_execution_rule(const detail::CUDAGraphExecutionLoweringResult& result,
                                      std::string_view rule) {
  return find_execution_rule(result, rule) != nullptr;
}

// Re-run fused-region admission after changing one physical plan fact. This isolates each rule
// from earlier compiler passes that originally produced the accepted baseline.
template <typename Modify>
[[nodiscard]] std::optional<detail::CUDAGraphExecutionLoweringResult> lower_after_plan_change(
    const ExecutionPlan& plan, Modify&& modify) {
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  if (compiled == nullptr) {
    return std::nullopt;
  }
  // Other Modify instantiations mutate this plan before lowering.
  // NOLINTNEXTLINE(misc-const-correctness)
  detail::ExecutionPlanState changed = *compiled;
  if (!std::forward<Modify>(modify)(changed)) {
    return std::nullopt;
  }
  return detail::finalize_cuda_graph_execution_regions(changed.graph_identity.version, changed);
}

// Verify one physical admission rule by changing an accepted plan fact and re-running CUDA Graph
// execution lowering.
template <typename Modify>
void expect_lowering_rejection_after_plan_change(
    const ExecutionPlan& plan, std::string_view rule, std::string_view required_evidence,
    Modify&& modify, std::initializer_list<std::string_view> additional_evidence = {},
    std::initializer_list<std::string_view> excluded_evidence = {}, VertexId expected_vertex = {},
    PortId expected_port = {}) {
  SCOPED_TRACE(required_evidence);
  const auto lowered = lower_after_plan_change(plan, std::forward<Modify>(modify));
  HOLOSCAN_ASSERT_HAS_VALUE(lowered) << "plan change could not be applied";
  EXPECT_TRUE(lowered->regions.empty());
  const auto* diagnostic = find_execution_rule(*lowered, rule);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_NE(diagnostic->provided.find(required_evidence), std::string::npos);
  for (const std::string_view evidence : additional_evidence) {
    EXPECT_NE(diagnostic->provided.find(evidence), std::string::npos);
  }
  for (const std::string_view evidence : excluded_evidence) {
    EXPECT_EQ(diagnostic->provided.find(evidence), std::string::npos);
  }
  if (expected_vertex) {
    EXPECT_EQ(diagnostic->source.vertex, expected_vertex);
  }
  if (expected_port) {
    EXPECT_EQ(diagnostic->source.port, expected_port);
  }
}

void expect_fixed_slot_compilation_rejection(const ExecutionPlan& plan) {
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kSingleFixedTensorSlot;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=fixed_address_requires_single_live_sample"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("queue_depth=2"), std::string::npos);
}

[[nodiscard]] bool split_internal_edges_across_cuda_streams(detail::ExecutionPlanState& state) {
  if (state.cuda_graph_capture_plans.empty() ||
      state.cuda_graph_capture_plans.front().internal_edges.size() < 2U) {
    return false;
  }
  const auto& internal_edges = state.cuda_graph_capture_plans.front().internal_edges;
  const EdgeId first_edge = internal_edges[0].edge;
  const EdgeId second_edge = internal_edges[1].edge;
  auto stream = std::find_if(
      state.cuda_compute_streams.begin(), state.cuda_compute_streams.end(), [&](const auto& item) {
        return std::find(item.fused_linear_edges.begin(),
                         item.fused_linear_edges.end(),
                         first_edge) != item.fused_linear_edges.end() &&
               std::find(item.fused_linear_edges.begin(),
                         item.fused_linear_edges.end(),
                         second_edge) != item.fused_linear_edges.end();
      });
  if (stream == state.cuda_compute_streams.end()) {
    return false;
  }
  CudaComputeStreamPlanRecord second_stream = *stream;
  second_stream.stream_id += ".split";
  second_stream.fused_linear_edges = {second_edge};
  std::erase(stream->fused_linear_edges, second_edge);
  state.cuda_compute_streams.push_back(std::move(second_stream));
  return true;
}

template <typename Modify>
void expect_region_record_mutation_rejected(const detail::ExecutionPlanState& baseline,
                                            std::string_view field, Modify&& modify) {
  SCOPED_TRACE(field);
  detail::ExecutionPlanState malformed = baseline;
  std::forward<Modify>(modify)(malformed.cuda_graph_execution_regions.front());
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
}

template <typename Modify>
void expect_buffer_record_mutation_rejected(const detail::ExecutionPlanState& baseline,
                                            std::string_view field, Modify&& modify) {
  SCOPED_TRACE(field);
  detail::ExecutionPlanState malformed = baseline;
  std::forward<Modify>(modify)(malformed.cuda_graph_execution_regions.front().buffers.front());
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
}

// REQ: COMPILE-081
// REQ: COMPILE-084
// REQ: COMPILE-089
TEST(CUDAGraphExecutionLowering, PublishesFusedOperatorStreamAndInternalDeviceBuffers) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_capture_plans.size(), 1U) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();

  const auto& capture = state->cuda_graph_capture_plans.front();
  const auto& region = state->cuda_graph_execution_regions.front();
  EXPECT_EQ(region.graph, capture.graph);
  EXPECT_EQ(region.region, capture.region);
  EXPECT_EQ(region.fused_operator, capture.members.front().vertex);
  EXPECT_EQ(region.device, DeviceId{0});
  EXPECT_FALSE(region.stream_id.empty());
  EXPECT_EQ(region.buffers.size(), capture.internal_edges.size());
  EXPECT_TRUE(std::all_of(region.buffers.begin(), region.buffers.end(), [](const auto& buffer) {
    return buffer.kind == detail::CUDAGraphCaptureBufferKind::kInternal &&
           buffer.memory_kind == MemoryKind::kCudaDevice && buffer.device == DeviceId(0) &&
           buffer.bounds.max_byte_span == 1024U &&
           buffer.fixed_shape == std::vector<std::uint64_t>{1024U} &&
           buffer.active_slot_count == 1U && buffer.aggregate_byte_capacity == 1024U &&
           !buffer.fixed_allocation_id.empty() && !buffer.pool_id.empty() &&
           !buffer.allocator_root_id.empty();
  }));
  EXPECT_NE(plan.json().find("\"fixed_allocation_id\""), std::string::npos);
  EXPECT_NE(plan.json().find("\"fixed_shape\":[1024]"), std::string::npos);
  EXPECT_NE(plan.json().find("\"active_slot_count\":1"), std::string::npos);
  const auto fixed_slot_capacity =
      std::ranges::find_if(plan.profile_manifest().derived_capacities, [&](const auto& capacity) {
        return capacity.object_id == region.buffers.front().fixed_allocation_id &&
               capacity.capacity_id == "active_slots";
      });
  ASSERT_NE(fixed_slot_capacity, plan.profile_manifest().derived_capacities.end()) << plan.json();
  EXPECT_EQ(fixed_slot_capacity->limit, 1U);
  const auto stream =
      std::find_if(state->cuda_compute_streams.begin(),
                   state->cuda_compute_streams.end(),
                   [&](const auto& candidate) { return candidate.stream_id == region.stream_id; });
  ASSERT_NE(stream, state->cuda_compute_streams.end());
  for (const auto& edge : capture.internal_edges) {
    EXPECT_NE(
        std::find(stream->fused_linear_edges.begin(), stream->fused_linear_edges.end(), edge.edge),
        stream->fused_linear_edges.end());
    EXPECT_EQ(std::count_if(state->process_local_channel_subscribers.begin(),
                            state->process_local_channel_subscribers.end(),
                            [&](const auto& subscriber) { return subscriber.edge == edge.edge; }),
              0);
  }
  EXPECT_TRUE(state->process_local_channel_services.empty());
  EXPECT_TRUE(state->process_local_channel_publishers.empty());
  EXPECT_TRUE(detail::validate_cuda_execution_plan_records(*state));
  detail::ExecutionPlanState missing_member_binding = *state;
  const VertexId removed_member = capture.members.back().vertex;
  std::erase_if(missing_member_binding.operator_cuda_stream_bindings,
                [&](const auto& binding) { return binding.operator_vertex == removed_member; });
  EXPECT_FALSE(detail::validate_cuda_execution_plan_records(missing_member_binding));
  EXPECT_FALSE(has_diagnostic(plan, "CUDA_GRAPH_RUNTIME_UNAVAILABLE"));
  EXPECT_FALSE(has_diagnostic(plan, "CUDA_GRAPH_EXECUTION_REGION_INVALID"));
  EXPECT_NE(plan.json().find("\"cuda_graph_execution_regions\""), std::string::npos);
  EXPECT_TRUE(detail::valid_cuda_graph_execution_regions(*state));

  ASSERT_EQ(state->partition_runners.size(), 1U);
  ASSERT_EQ(state->partition_runners.front().operators.size(), 1U);
  const auto& fused_region =
      state->partition_runners.front().operators.front().fused_cuda_graph_region;
  HOLOSCAN_ASSERT_HAS_VALUE(fused_region);
  EXPECT_EQ(fused_region->graph, region.graph);
  EXPECT_EQ(fused_region->region, region.region);
  EXPECT_TRUE(state->partition_runners.front().internal_edges.empty());
  EXPECT_TRUE(state->closed_island_activations.empty());
  EXPECT_FALSE(state->inline_partition_execution.has_value());
}

// REQ: COMPILE-002
// REQ: COMPILE-081
TEST(CUDAGraphExecutionLowering, ExecutionRecordsAndDigestDoNotDependOnRegionMemberInsertionOrder) {
  const ExecutionPlan forward = compile_fused_region();
  const ExecutionPlan reverse = compile_fused_region(MemoryKind::kCudaDevice,
                                                     false,
                                                     true,
                                                     TensorOutputStorage::kRuntimePool,
                                                     TensorOutputStorage::kRuntimePool,
                                                     "capture",
                                                     true);
  const auto* forward_state = detail::ExecutionPlanAccess::state(forward);
  const auto* reverse_state = detail::ExecutionPlanAccess::state(reverse);
  ASSERT_NE(forward_state, nullptr);
  ASSERT_NE(reverse_state, nullptr);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();
  ASSERT_EQ(forward_state->cuda_graph_execution_regions.size(), 1U);
  ASSERT_EQ(reverse_state->cuda_graph_execution_regions.size(), 1U);

  EXPECT_EQ(forward_state->cuda_graph_execution_regions,
            reverse_state->cuda_graph_execution_regions);
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

// REQ: COMPILE-081
TEST(CUDAGraphExecutionLowering, GraphInspectionPreservesFusedRegionMembers) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto partitions = plan.partitions();
  ASSERT_EQ(partitions.size(), 1U);
  const auto partition = *partitions.begin();
  ASSERT_EQ(partition.operator_paths().size(), 3U);
  for (const std::string_view expected : {"source", "middle", "sink"}) {
    EXPECT_NE(
        std::find(partition.operator_paths().begin(), partition.operator_paths().end(), expected),
        partition.operator_paths().end());
  }
  EXPECT_EQ(plan.graph().vertices().size(), 3U);
  EXPECT_TRUE(std::any_of(plan.graph().vertices().begin(),
                          plan.graph().vertices().end(),
                          [](const auto& vertex) { return vertex.name == "middle"; }));
  EXPECT_TRUE(std::any_of(plan.graph().vertices().begin(),
                          plan.graph().vertices().end(),
                          [](const auto& vertex) { return vertex.name == "sink"; }));
  EXPECT_NE(plan.json().find("\"middle\""), std::string::npos);
  EXPECT_NE(plan.json().find("\"sink\""), std::string::npos);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AcceptsUnconstrainedTensorInputInRegion) {
  const ExecutionPlan plan = compile_fused_region_with_relaxed_input(false);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AcceptsPartiallyConstrainedTensorInputInRegion) {
  const ExecutionPlan plan = compile_fused_region_with_relaxed_input(true);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsBoundedStridedCaptureBufferDuringCompilation) {
  const ExecutionPlan plan =
      compile_fused_region_with_ingress_geometry<TensorStridePolicy::kBoundedStrided, 1U>();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_ingress = state->cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_ingress);
  const auto& ingress = *optional_ingress;

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kFixedTensorBuffer;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=capture_requires_contiguous_tensor_layout"),
            std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, ingress.source);
  EXPECT_EQ(diagnostic->source.port, ingress.source_port);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsScalarCaptureIngressDuringCompilation) {
  const ExecutionPlan plan =
      compile_fused_region_with_ingress_geometry<TensorStridePolicy::kContiguousOnly, 0U>();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_ingress = state->cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_ingress);
  const auto& ingress = *optional_ingress;

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kExactTensorShape;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=scalar_capture_ingress_unsupported"),
            std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, ingress.source);
  EXPECT_EQ(diagnostic->source.port, ingress.source_port);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsFixedShapeExceedingDeclaredByteCapacity) {
  const ExecutionPlan plan = compile_fused_region_with_direct_ingress_geometry({4U}, 8U, 1U);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_ingress = state->cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_ingress);
  const auto& ingress = *optional_ingress;

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kExactTensorShape;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=fixed_shape_exceeds_byte_capacity"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("required_byte_span=4,max_byte_span=1"), std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, ingress.source);
  EXPECT_EQ(diagnostic->source.port, ingress.source_port);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AcceptsFixedShapeMatchingDeclaredByteCapacity) {
  const ExecutionPlan plan = compile_fused_region_with_direct_ingress_geometry({4U}, 8U, 4U);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.front().buffers.size(), 1U);
  EXPECT_EQ(state->cuda_graph_execution_regions.front().buffers.front().fixed_shape,
            std::vector<std::uint64_t>{4U});
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsFixedShapeExceedingProducerStrideBound) {
  const ExecutionPlan plan =
      compile_fused_region_with_direct_ingress_geometry({4U, 3U}, 8U, 12U, {1U, 1U});
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_ingress = state->cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_ingress);
  const auto& ingress = *optional_ingress;

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kExactTensorShape;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=fixed_shape_exceeds_stride_bound,axis=0"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("required_byte_stride=3,max_abs_stride_bytes=1"),
            std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, ingress.source);
  EXPECT_EQ(diagnostic->source.port, ingress.source_port);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AcceptsFixedShapeMatchingCanonicalStrideBounds) {
  const ExecutionPlan plan =
      compile_fused_region_with_direct_ingress_geometry({4U, 3U}, 8U, 12U, {3U, 1U});
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.front().buffers.size(), 1U);
  EXPECT_EQ(state->cuda_graph_execution_regions.front().buffers.front().fixed_shape,
            (std::vector<std::uint64_t>{4U, 3U}));
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AcceptsFixedShapeWithUnconstrainedStrideBounds) {
  const ExecutionPlan plan = compile_fused_region_with_direct_ingress_geometry({4U, 3U}, 8U, 12U);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.front().buffers.size(), 1U);
  EXPECT_EQ(state->cuda_graph_execution_regions.front().buffers.front().fixed_shape,
            (std::vector<std::uint64_t>{4U, 3U}));
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsFixedShapeExceedingConsumerStrideBound) {
  const ExecutionPlan plan =
      compile_fused_region_with_direct_ingress_geometry({4U, 3U}, 8U, 12U, {3U, 1U}, {3U, 1U});
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_ingress = state->cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_ingress);
  const auto ingress = *optional_ingress;

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kExactTensorShape,
      "reason=fixed_shape_exceeds_stride_bound,axis=0",
      [&](auto& changed) {
        auto consumer =
            std::find_if(changed.ports.begin(), changed.ports.end(), [&](const auto& port) {
              return port.id == ingress.target_port;
            });
        if (consumer == changed.ports.end() || !consumer->tensor_contract ||
            consumer->tensor_contract->axes.size() != 2U) {
          return false;
        }
        consumer->tensor_contract->axes[0].max_abs_stride_bytes = 1U;
        return true;
      },
      {"required_byte_stride=3,max_abs_stride_bytes=1", "buffer_kind=ingress"},
      {},
      ingress.target,
      ingress.target_port);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsOverflowingFixedShapeByteSpan) {
  const ExecutionPlan plan =
      compile_fused_region_with_direct_ingress_geometry({4U, 3U}, 8U, 12U, {3U, 1U});
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kExactTensorShape,
      "reason=fixed_shape_byte_span_overflow_or_invalid_dtype",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            !state.cuda_graph_capture_plans.front().ingress) {
          return false;
        }
        const PortId output = state.cuda_graph_capture_plans.front().ingress->source_port;
        auto producer = std::find_if(state.ports.begin(), state.ports.end(), [&](const auto& port) {
          return port.id == output;
        });
        auto allocation =
            std::find_if(state.tensor_allocations.begin(),
                         state.tensor_allocations.end(),
                         [&](const auto& candidate) { return candidate.output == output; });
        if (producer == state.ports.end() || !producer->tensor_contract ||
            producer->tensor_contract->axes.size() != 2U ||
            allocation == state.tensor_allocations.end() || allocation->axes.size() != 2U) {
          return false;
        }
        constexpr std::uint64_t kLargestExtent =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        producer->tensor_contract->axes[0].max_extent = kLargestExtent;
        producer->tensor_contract->axes[1].max_extent = 3U;
        allocation->axes[0].max_extent = kLargestExtent;
        allocation->axes[1].max_extent = 3U;
        allocation->bounds.max_byte_span = std::numeric_limits<std::uint64_t>::max();
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AcceptsCanonicalRoundedSubByteElementSize) {
  const ExecutionPlan plan = compile_fused_region_with_direct_ingress_geometry({4U}, 1U, 4U);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.front().buffers.size(), 1U);
  const auto& dtype = state->cuda_graph_execution_regions.front().buffers.front().layout.dtype;
  HOLOSCAN_ASSERT_HAS_VALUE(dtype);
  EXPECT_EQ(dtype->bits, 1U);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsAllocationThatDoesNotSatisfyConsumerTensorLayout) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.empty());
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.front().internal_edges.empty());
  const auto edge = compiled->cuda_graph_capture_plans.front().internal_edges.front();

  const auto lowered = lower_after_plan_change(plan, [&](auto& state) {
    const PortId input = edge.target_port;
    auto consumer = std::find_if(
        state.ports.begin(), state.ports.end(), [&](const auto& port) { return port.id == input; });
    if (consumer == state.ports.end() || !consumer->tensor_layout) {
      return false;
    }
    consumer->tensor_layout->dtype = DLDataType{kDLUInt, 16U, 1U};
    return true;
  });
  HOLOSCAN_ASSERT_HAS_VALUE(lowered) << "plan change could not be applied";
  EXPECT_TRUE(lowered->regions.empty());
  const auto* diagnostic =
      find_execution_rule(*lowered, detail::cuda_graph_execution_rule::kFixedTensorBuffer);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_NE(diagnostic->provided.find("reason=allocation_does_not_satisfy_input"),
            std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, edge.target);
  EXPECT_EQ(diagnostic->source.port, edge.target_port);
}

// REQ: COMPILE-082
// REQ: COMPILE-089
TEST(CUDAGraphExecutionLowering, UsesOperatorStreamForSingleOperatorGreenContextRegion) {
  const ExecutionPlan plan = compile_single_operator_green_context_region();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_capture_plans.size(), 1U) << plan.json();
  ASSERT_TRUE(state->cuda_graph_capture_plans.front().internal_edges.empty());
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();

  const auto& region = state->cuda_graph_execution_regions.front();
  const auto* operator_stream = detail::find_cuda_compute_stream_by_partition(
      *state, region.process_group, region.device, region.partition);
  ASSERT_NE(operator_stream, nullptr);
  ASSERT_TRUE(operator_stream->green_context_id);
  EXPECT_EQ(region.stream_id, operator_stream->stream_id);
  EXPECT_EQ(region.green_context_id, operator_stream->green_context_id);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsServiceStreamForSingleOperatorGreenContextRegion) {
  const ExecutionPlan plan = compile_single_operator_green_context_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=operator_stream_missing",
      [](auto& state) {
        if (state.cuda_graph_execution_regions.size() != 1U) {
          return false;
        }
        const auto& region = state.cuda_graph_execution_regions.front();
        const auto operator_stream =
            std::find_if(state.cuda_compute_streams.begin(),
                         state.cuda_compute_streams.end(),
                         [&](const auto& stream) { return stream.partition == region.partition; });
        if (operator_stream == state.cuda_compute_streams.end() ||
            !operator_stream->green_context_id) {
          return false;
        }

        // Leave a same-device Green Context service stream as the only partitionless candidate. It
        // must not be substituted for the removed operator stream.
        CudaComputeStreamPlanRecord service_stream = *operator_stream;
        service_stream.owner_id = "service.test.single-operator-capture";
        service_stream.partition.reset();
        service_stream.stream_id += ".service";
        service_stream.fused_linear_edges.clear();
        state.cuda_compute_streams.erase(operator_stream);
        state.cuda_compute_streams.push_back(std::move(service_stream));
        return true;
      },
      {"green_context_service_streams=1"});
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsAmbiguousOrdinaryStreamsForSingleOperatorRegion) {
  const ExecutionPlan plan = compile_single_operator_green_context_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=multiple_ordinary_operator_streams",
      [](auto& state) {
        if (state.cuda_graph_execution_regions.size() != 1U) {
          return false;
        }
        const auto& region = state.cuda_graph_execution_regions.front();
        const auto operator_stream =
            std::find_if(state.cuda_compute_streams.begin(),
                         state.cuda_compute_streams.end(),
                         [&](const auto& stream) { return stream.partition == region.partition; });
        if (operator_stream == state.cuda_compute_streams.end()) {
          return false;
        }

        CudaComputeStreamPlanRecord first = *operator_stream;
        first.owner_id = "operator.test.first";
        first.partition.reset();
        first.green_context_id.reset();
        first.stream_id = "operator.stream.first";
        CudaComputeStreamPlanRecord second = first;
        second.owner_id = "operator.test.second";
        second.stream_id = "operator.stream.second";
        state.cuda_compute_streams.erase(operator_stream);
        state.cuda_compute_streams.push_back(std::move(first));
        state.cuda_compute_streams.push_back(std::move(second));
        return true;
      },
      {"stream_count=2"},
      {"operator.stream.first", "operator.stream.second"});
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, AcceptsEdgelessSingleOperatorWithGreenContextStream) {
  const ExecutionPlan plan = compile_edgeless_single_operator_region(true);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_capture_plans.size(), 1U) << plan.json();
  EXPECT_TRUE(state->cuda_graph_capture_plans.front().internal_edges.empty());
  EXPECT_FALSE(state->cuda_graph_capture_plans.front().ingress);
  EXPECT_FALSE(state->cuda_graph_capture_plans.front().egress);
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();

  const auto& region = state->cuda_graph_execution_regions.front();
  const auto* operator_stream = detail::find_cuda_compute_stream_by_partition(
      *state, region.process_group, region.device, region.partition);
  ASSERT_NE(operator_stream, nullptr);
  EXPECT_TRUE(region.buffers.empty());
  EXPECT_EQ(region.device, DeviceId{0});
  EXPECT_EQ(region.stream_id, operator_stream->stream_id);
  EXPECT_EQ(region.green_context_id, operator_stream->green_context_id);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsEdgelessSingleOperatorWithoutCUDADeviceEvidence) {
  const ExecutionPlan plan = compile_edgeless_single_operator_region(false);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kSingleCUDADevice;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=no_cuda_device_evidence"), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("buffer_devices=none"), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("partition_stream_devices=none"), std::string::npos);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsRegionSplitAcrossExecutionPartitions) {
  const ExecutionPlan plan = compile_fused_region(MemoryKind::kCudaDevice, true);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kSingleExecutionPartition;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("member_partitions="), std::string::npos);
  EXPECT_EQ(diagnostic->provided.find(":none"), std::string::npos);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsMembersSplitAcrossProcessGroups) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.empty());
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.front().members.empty());
  const VertexId source = compiled->cuda_graph_capture_plans.front().members.front().vertex;

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleProcessGroup,
      "reason=members_span_process_groups",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.size() < 2U) {
          return false;
        }
        const VertexId member = state.cuda_graph_capture_plans.front().members[1].vertex;
        auto placement =
            std::find_if(state.vertex_process_placements.begin(),
                         state.vertex_process_placements.end(),
                         [&](const auto& candidate) { return candidate.vertex == member; });
        if (placement == state.vertex_process_placements.end()) {
          return false;
        }
        placement->process_group = ProcessGroupId{placement->process_group.value + 1U};
        return true;
      },
      {"member_process_groups="},
      {":none"},
      source);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsIngressAllocationOwnedByAnotherProcess) {
  const ExecutionPlan plan = compile_fused_region_with_cross_process_ingress();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_ingress = state->cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_ingress);
  const auto& ingress = *optional_ingress;

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kSingleProcessGroup;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=allocation_process_group_mismatch"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("allocator_process_group="), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("region_process_group="), std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, ingress.source);
  EXPECT_EQ(diagnostic->source.port, ingress.source_port);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsEgressConsumerOwnedByAnotherProcess) {
  const ExecutionPlan plan = compile_fused_region_with_cross_process_egress();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(state->cuda_graph_capture_plans.empty());
  const auto& optional_egress = state->cuda_graph_capture_plans.front().egress;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_egress);
  const auto& egress = *optional_egress;

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kSingleProcessGroup;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=egress_consumer_process_group_mismatch"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("consumer_process_group="), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("region_process_group="), std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, egress.target);
  EXPECT_EQ(diagnostic->source.port, egress.target_port);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsInternalEdgesAssignedToDifferentCUDAStreams) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.empty());
  const auto& internal_edges = compiled->cuda_graph_capture_plans.front().internal_edges;
  ASSERT_EQ(internal_edges.size(), 2U);
  const std::string expected_edges = "edges=" + std::to_string(internal_edges[0].edge.value) + "|" +
                                     std::to_string(internal_edges[1].edge.value);

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=internal_edge_streams_do_not_match",
      split_internal_edges_across_cuda_streams,
      {expected_edges, "requirement=all_internal_edges_use_region_source_stream"},
      {".split"});
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, StreamRejectionEvidenceSurvivesPlanGenerationRebinding) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  detail::ExecutionPlanState changed = *compiled;
  ASSERT_TRUE(split_internal_edges_across_cuda_streams(changed));

  const auto before =
      detail::finalize_cuda_graph_execution_regions(changed.graph_identity.version, changed);
  ASSERT_EQ(before.errors.size(), 1U);

  detail::rebind_plan_generation(changed, changed.plan_generation + 1U);
  const auto after =
      detail::finalize_cuda_graph_execution_regions(changed.graph_identity.version, changed);
  ASSERT_EQ(after.errors.size(), 1U);
  EXPECT_TRUE(
      detail::plan_lowering_diagnostic_exactly_equal(before.errors.front(), after.errors.front()));
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsInternalEdgeWithoutAcceptedTensorAllocation) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=allocation_not_accepted",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto allocation = std::find_if(state.tensor_allocations.begin(),
                                       state.tensor_allocations.end(),
                                       [&](const auto& item) { return item.output == output; });
        if (allocation == state.tensor_allocations.end()) {
          return false;
        }
        allocation->disposition = PlanDisposition::kRejected;
        return true;
      },
      {"disposition=kRejected"});
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsCaptureBufferWithMissingSourcePort) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=source_port_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        state.cuda_graph_capture_plans.front().internal_edges.front().source_port = PortId{};
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsCaptureBufferWithMissingTargetPort) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=target_port_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        state.cuda_graph_capture_plans.front().internal_edges.front().target_port = PortId{};
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsCaptureBufferWithoutTensorAllocation) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=allocation_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        return std::erase_if(state.tensor_allocations,
                             [&](const auto& item) { return item.output == output; }) == 1U;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsCaptureBufferWithoutFixedAllocationBounds) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=allocation_bounds_not_fixed",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto allocation = std::find_if(state.tensor_allocations.begin(),
                                       state.tensor_allocations.end(),
                                       [&](const auto& item) { return item.output == output; });
        if (allocation == state.tensor_allocations.end()) {
          return false;
        }
        allocation->bounds.max_byte_span = 0U;
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsAllocationWhoseMemoryKindDisagreesWithItsLayout) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=allocation_memory_mismatch",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto allocation = std::find_if(state.tensor_allocations.begin(),
                                       state.tensor_allocations.end(),
                                       [&](const auto& item) { return item.output == output; });
        if (allocation == state.tensor_allocations.end() || !allocation->layout) {
          return false;
        }
        allocation->layout->memory_kind = MemoryKind::kPinnedHost;
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsAllocationWithoutPoolLineage) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto clear_allocation_lineage = [](auto& state, bool clear_pool) {
    if (state.cuda_graph_capture_plans.empty() ||
        state.cuda_graph_capture_plans.front().internal_edges.empty()) {
      return false;
    }
    const PortId output = state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
    auto allocation = std::find_if(state.tensor_allocations.begin(),
                                   state.tensor_allocations.end(),
                                   [&](const auto& item) { return item.output == output; });
    if (allocation == state.tensor_allocations.end()) {
      return false;
    }
    if (clear_pool) {
      allocation->pool_id.clear();
    } else {
      allocation->allocator_root_id.clear();
    }
    return true;
  };

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=pool_id_missing",
      [&](auto& state) { return clear_allocation_lineage(state, true); });
  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=allocator_root_id_missing",
      [&](auto& state) { return clear_allocation_lineage(state, false); });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsAllocationThatDisagreesWithProducerTensorLayout) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.empty());
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.front().internal_edges.empty());
  const auto edge = compiled->cuda_graph_capture_plans.front().internal_edges.front();

  const auto lowered = lower_after_plan_change(plan, [&](auto& state) {
    const PortId output = edge.source_port;
    auto producer = std::find_if(state.ports.begin(), state.ports.end(), [&](const auto& port) {
      return port.id == output;
    });
    if (producer == state.ports.end() || !producer->tensor_layout) {
      return false;
    }
    producer->tensor_layout->dtype = DLDataType{kDLUInt, 16U, 1U};
    return true;
  });
  HOLOSCAN_ASSERT_HAS_VALUE(lowered) << "plan change could not be applied";
  EXPECT_TRUE(lowered->regions.empty());
  const auto* diagnostic =
      find_execution_rule(*lowered, detail::cuda_graph_execution_rule::kFixedTensorBuffer);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_NE(diagnostic->provided.find("reason=allocation_source_layout_mismatch"),
            std::string::npos);
  EXPECT_EQ(diagnostic->source.vertex, edge.source);
  EXPECT_EQ(diagnostic->source.port, edge.source_port);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsEdgeWhoseEndpointPayloadIdentitiesDisagree) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=payload_identity_mismatch",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId input =
            state.cuda_graph_capture_plans.front().internal_edges.front().target_port;
        auto consumer = std::find_if(state.ports.begin(), state.ports.end(), [&](const auto& port) {
          return port.id == input;
        });
        if (consumer == state.ports.end()) {
          return false;
        }
        consumer->payload_identity.kind = PayloadIdentityKind::kScalar;
        return true;
      });
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsInternalEdgeWithoutCUDAStream) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=internal_edge_stream_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().internal_edges.front().edge;
        std::size_t removed{};
        for (auto& stream : state.cuda_compute_streams) {
          removed += std::erase(stream.fused_linear_edges, edge);
        }
        return removed == 1U;
      });
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsInternalEdgeStreamWithoutIdentity) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=internal_edge_stream_id_empty",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().internal_edges.front().edge;
        auto stream = std::find_if(state.cuda_compute_streams.begin(),
                                   state.cuda_compute_streams.end(),
                                   [&](const auto& item) {
                                     return std::find(item.fused_linear_edges.begin(),
                                                      item.fused_linear_edges.end(),
                                                      edge) != item.fused_linear_edges.end();
                                   });
        if (stream == state.cuda_compute_streams.end()) {
          return false;
        }
        stream->stream_id.clear();
        return true;
      });
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsInternalEdgeStreamFromAnotherPartition) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=edge_stream_partition_mismatch",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const auto& capture_plan = state.cuda_graph_capture_plans.front();
        if (capture_plan.members.empty()) {
          return false;
        }
        const auto* region_partition = detail::find_unique_execution_partition_by_member(
            state, capture_plan.members.front().vertex);
        if (region_partition == nullptr) {
          return false;
        }
        const EdgeId edge = capture_plan.internal_edges.front().edge;
        auto stream = std::find_if(state.cuda_compute_streams.begin(),
                                   state.cuda_compute_streams.end(),
                                   [&](const auto& item) {
                                     return std::find(item.fused_linear_edges.begin(),
                                                      item.fused_linear_edges.end(),
                                                      edge) != item.fused_linear_edges.end();
                                   });
        if (stream == state.cuda_compute_streams.end()) {
          return false;
        }
        stream->partition = ExecutionPartitionId{region_partition->id.value + 1U};
        return true;
      });
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsSingleOperatorPartitionStreamWithoutIdentity) {
  const ExecutionPlan plan = compile_single_operator_green_context_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSourceCUDAStream,
      "reason=partition_operator_stream_id_empty",
      [](auto& state) {
        if (state.cuda_graph_execution_regions.size() != 1U) {
          return false;
        }
        const auto& region = state.cuda_graph_execution_regions.front();
        auto stream = std::find_if(state.cuda_compute_streams.begin(),
                                   state.cuda_compute_streams.end(),
                                   [&](const auto& item) {
                                     return item.process_group == region.process_group &&
                                            item.device == region.device &&
                                            item.partition == region.partition;
                                   });
        if (stream == state.cuda_compute_streams.end()) {
          return false;
        }
        stream->stream_id.clear();
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsIncompleteTensorAllocationLayout) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=allocation_layout_incomplete",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto allocation = std::find_if(state.tensor_allocations.begin(),
                                       state.tensor_allocations.end(),
                                       [&](const auto& item) { return item.output == output; });
        if (allocation == state.tensor_allocations.end() || !allocation->layout) {
          return false;
        }
        allocation->layout->memory_kind.reset();
        return true;
      });
}

// REQ: COMPILE-081
TEST(CUDAGraphExecutionLowering, RejectsCapturePlanWithoutMembersUsingSpecificRule) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan, detail::cuda_graph_execution_rule::kNonemptyCapturePlan, "members=0", [](auto& state) {
        if (state.cuda_graph_capture_plans.empty()) {
          return false;
        }
        state.cuda_graph_capture_plans.front().members.clear();
        return true;
      });
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsInternalBuffersSplitAcrossCUDADevices) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleCUDADevice,
      "reason=conflicting_cuda_devices",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.size() < 2U) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.back().source_port;
        auto allocation = std::find_if(state.tensor_allocations.begin(),
                                       state.tensor_allocations.end(),
                                       [&](const auto& item) { return item.output == output; });
        if (allocation == state.tensor_allocations.end()) {
          return false;
        }
        allocation->device = DeviceId{1};
        return true;
      },
      {"buffer_devices=0|1"});
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsRegionAcrossTwoAvailableCUDADevices) {
  const ExecutionPlan plan = compile_fused_region_across_two_cuda_devices();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kSingleCUDADevice;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=conflicting_cuda_devices"), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("buffer_devices=0|1"), std::string::npos);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsManagedMemoryOnInternalRegionEdges) {
  const ExecutionPlan plan = compile_fused_region(MemoryKind::kCudaManaged,
                                                  false,
                                                  true,
                                                  TensorOutputStorage::kRuntimePool,
                                                  TensorOutputStorage::kRuntimePool,
                                                  "stage,storage=kPassThrough");
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kInternalCUDADeviceBuffer;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("memory_kind=kCudaManaged"), std::string::npos);
  EXPECT_NE(diagnostic->provided.find("region_name=stage%2Cstorage%3DkPassThrough"),
            std::string::npos);
  EXPECT_EQ(diagnostic->provided.find("region_name=stage,storage=kPassThrough"), std::string::npos);
  const auto source_port =
      std::find_if(state->ports.begin(), state->ports.end(), [&](const auto& port) {
        return port.id == diagnostic->source.port;
      });
  ASSERT_NE(source_port, state->ports.end());
  EXPECT_EQ(diagnostic->source.vertex, source_port->owner);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, ReportsHostInternalEdgeAsInvalidBufferMemory) {
  const ExecutionPlan plan = compile_fused_region(MemoryKind::kHost);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());

  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kInternalCUDADeviceBuffer;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("memory_kind=kHost"), std::string::npos);
  EXPECT_EQ(diagnostic->provided.find("reason=no_cuda_device_evidence"), std::string::npos);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsMissingOrMismatchedPhysicalRegionEdges) {
  const ExecutionPlan internal_plan = compile_fused_region();
  expect_lowering_rejection_after_plan_change(
      internal_plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=payload_edge_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().internal_edges.front().edge;
        std::erase_if(state.payload_edges, [&](const auto& record) { return record.edge == edge; });
        return true;
      });

  expect_lowering_rejection_after_plan_change(
      internal_plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=payload_edge_ports_mismatch",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const auto& edge = state.cuda_graph_capture_plans.front().internal_edges.front();
        auto physical = std::find_if(state.payload_edges.begin(),
                                     state.payload_edges.end(),
                                     [&](const auto& record) { return record.edge == edge.edge; });
        if (physical == state.payload_edges.end()) {
          return false;
        }
        physical->target = PortId{physical->target.value() + 1U};
        return true;
      });

  const ExecutionPlan boundary_plan = compile_fused_region_with_boundaries(MemoryKind::kCudaDevice);
  expect_lowering_rejection_after_plan_change(
      boundary_plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=payload_edge_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            !state.cuda_graph_capture_plans.front().egress) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().egress->edge;
        std::erase_if(state.payload_edges, [&](const auto& record) { return record.edge == edge; });
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsPhysicalPayloadEdgeWithoutAcceptedDisposition) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kFixedTensorBuffer,
      "reason=payload_edge_not_accepted",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().internal_edges.front().edge;
        auto physical = std::find_if(state.payload_edges.begin(),
                                     state.payload_edges.end(),
                                     [&](const auto& record) { return record.edge == edge; });
        if (physical == state.payload_edges.end()) {
          return false;
        }
        physical->disposition = PlanDisposition::kRejected;
        return true;
      });
}

// REQ: COMPILE-081
TEST(CUDAGraphExecutionLowering, PartitionEvidenceUsesTheAdmissionMembershipRule) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.empty());
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.front().members.empty());
  const VertexId source = compiled->cuda_graph_capture_plans.front().members.front().vertex;

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleExecutionPartition,
      std::to_string(source.value) + ":none",
      [](auto& state) {
        if (state.execution_partitions.empty() ||
            state.execution_partitions.front().members.size() < 2U ||
            state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.empty()) {
          return false;
        }
        std::reverse(state.execution_partitions.front().members.begin(),
                     state.execution_partitions.front().members.end());
        return true;
      });
}

// REQ: COMPILE-083
// REQ: COMPILE-084
TEST(CUDAGraphExecutionLowering, AcceptsPinnedHostBoundaryBuffers) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(MemoryKind::kPinnedHost);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U) << plan.json();
  const auto& buffers = state->cuda_graph_execution_regions.front().buffers;
  ASSERT_EQ(buffers.size(), 4U);
  EXPECT_EQ(buffers.front().kind, detail::CUDAGraphCaptureBufferKind::kIngress);
  EXPECT_EQ(buffers.front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_FALSE(buffers.front().device.valid());
  EXPECT_EQ(buffers.back().kind, detail::CUDAGraphCaptureBufferKind::kEgress);
  EXPECT_EQ(buffers.back().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_FALSE(buffers.back().device.valid());
  EXPECT_EQ(std::count_if(buffers.begin(),
                          buffers.end(),
                          [](const auto& buffer) {
                            return buffer.kind == detail::CUDAGraphCaptureBufferKind::kInternal &&
                                   buffer.memory_kind == MemoryKind::kCudaDevice;
                          }),
            2);

  ASSERT_EQ(state->partition_runners.size(), 1U);
  const auto& runner = state->partition_runners.front();
  const auto& capture = state->cuda_graph_capture_plans.front();
  const auto& region = state->cuda_graph_execution_regions.front();
  EXPECT_EQ(std::count_if(runner.operators.begin(),
                          runner.operators.end(),
                          [&](const auto& member) {
                            return std::any_of(capture.members.begin(),
                                               capture.members.end(),
                                               [&](const auto& captured) {
                                                 return member.vertex == captured.vertex;
                                               });
                          }),
            1);
  EXPECT_NE(std::find_if(runner.operators.begin(),
                         runner.operators.end(),
                         [&](const auto& member) {
                           return member.vertex == region.fused_operator &&
                                  member.fused_cuda_graph_region;
                         }),
            runner.operators.end());
  const auto& ingress = region.ingress;
  const auto& egress = region.egress;
  HOLOSCAN_ASSERT_HAS_VALUE(ingress);
  HOLOSCAN_ASSERT_HAS_VALUE(egress);
  EXPECT_NE(std::find(runner.internal_edges.begin(), runner.internal_edges.end(), ingress->edge),
            runner.internal_edges.end());
  EXPECT_NE(std::find(runner.internal_edges.begin(), runner.internal_edges.end(), egress->edge),
            runner.internal_edges.end());
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsProducerWithoutExactCaptureShape) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kExactTensorShape,
      "reason=producer_exact_shape_missing,buffer_kind=internal",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto producer = std::find_if(state.ports.begin(), state.ports.end(), [&](const auto& port) {
          return port.id == output;
        });
        if (producer == state.ports.end() || !producer->tensor_contract) {
          return false;
        }
        producer->tensor_contract->axes.clear();
        return true;
      },
      {"rank=1", "axis_bounds=0"});
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsCaptureMemberRuntimeParameterUpdates) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kImmutableCaptureState,
      "reason=parameter_updates_enabled",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.empty()) {
          return false;
        }
        const OperatorRef owner =
            state.cuda_graph_capture_plans.front().members.front().operator_ref;
        auto parameters = std::ranges::find(
            state.operator_parameters, owner, &OperatorParameterPlanRecord::owner);
        if (parameters == state.operator_parameters.end()) {
          return false;
        }
        parameters->runtime_updates_enabled = true;
        return true;
      });
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsCaptureMemberLifecycleParameterUpdates) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kImmutableCaptureState,
      "reason=parameter_updates_enabled",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.empty()) {
          return false;
        }
        const OperatorRef owner =
            state.cuda_graph_capture_plans.front().members.front().operator_ref;
        auto parameters = std::ranges::find(
            state.operator_parameters, owner, &OperatorParameterPlanRecord::owner);
        if (parameters == state.operator_parameters.end()) {
          return false;
        }
        parameters->lifecycle_updates_enabled = true;
        return true;
      });
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsCaptureMemberWithoutParameterPlan) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kImmutableCaptureState,
      "reason=parameter_plan_missing",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.empty()) {
          return false;
        }
        const OperatorRef owner =
            state.cuda_graph_capture_plans.front().members.front().operator_ref;
        const auto previous_size = state.operator_parameters.size();
        std::erase_if(state.operator_parameters,
                      [&](const auto& parameters) { return parameters.owner == owner; });
        return state.operator_parameters.size() != previous_size;
      });
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsCaptureMemberLifecycleRestart) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kImmutableCaptureState,
      "reason=lifecycle_restart_enabled",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.empty()) {
          return false;
        }
        const OperatorRef target =
            state.cuda_graph_capture_plans.front().members.front().operator_ref;
        auto restart = std::ranges::find(
            state.lifecycle_restart_targets, target, &LifecycleRestartTargetPlanRecord::target);
        if (restart == state.lifecycle_restart_targets.end()) {
          state.lifecycle_restart_targets.push_back(LifecycleRestartTargetPlanRecord{
              .target = target,
              .disposition = LifecycleRestartTargetDisposition::kAdmitted,
          });
          return true;
        }
        restart->disposition = LifecycleRestartTargetDisposition::kAdmitted;
        return true;
      });
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsCaptureMemberInLifecycleRestartClosure) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kImmutableCaptureState,
      "reason=lifecycle_restart_enabled",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().members.empty()) {
          return false;
        }
        const OperatorRef member =
            state.cuda_graph_capture_plans.front().members.front().operator_ref;
        OperatorRef target = member;
        target.id += 1000U;
        state.lifecycle_restart_targets.push_back(LifecycleRestartTargetPlanRecord{
            .target = target,
            .disposition = LifecycleRestartTargetDisposition::kAdmitted,
        });
        state.lifecycle_restart_closures.push_back(LifecycleRestartClosurePlanRecord{
            .target = target,
            .closure = {target, member},
        });
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsAllocationSmallerThanExactCaptureShape) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kExactTensorShape,
      "reason=allocation_does_not_cover_exact_shape,buffer_kind=internal",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto allocation =
            std::find_if(state.tensor_allocations.begin(),
                         state.tensor_allocations.end(),
                         [&](const auto& candidate) { return candidate.output == output; });
        if (allocation == state.tensor_allocations.end() || allocation->axes.empty()) {
          return false;
        }
        allocation->axes.front().max_extent = 512U;
        return true;
      },
      {"fixed_rank=1", "axis_bounds=1"});
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsConsumerShapeBoundSmallerThanExactCaptureShape) {
  const ExecutionPlan plan = compile_fused_region();
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kExactTensorShape,
      "reason=consumer_does_not_accept_exact_shape,buffer_kind=internal",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId input =
            state.cuda_graph_capture_plans.front().internal_edges.front().target_port;
        auto consumer = std::find_if(state.ports.begin(), state.ports.end(), [&](const auto& port) {
          return port.id == input;
        });
        if (consumer == state.ports.end() || !consumer->tensor_contract) {
          return false;
        }
        consumer->tensor_contract->max_rank = 1U;
        consumer->tensor_contract->max_byte_span = 512U;
        consumer->tensor_contract->axes = {
            TensorAxisBounds{.max_extent = 512U, .max_abs_stride_bytes = 1U}};
        return true;
      },
      {"fixed_rank=1", "axis_bounds=1"});
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsPassThroughRegionIngress) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(
      MemoryKind::kCudaDevice, TensorOutputStorage::kPassThrough);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kRuntimePoolTensorOutput;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=producer_output_not_runtime_pool"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("storage=kPassThrough"), std::string::npos);
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsRegionIngressRequiringMultipleTensorSlots) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(MemoryKind::kCudaDevice,
                                                                  TensorOutputStorage::kRuntimePool,
                                                                  false,
                                                                  TensorOutputStorage::kRuntimePool,
                                                                  2U,
                                                                  1U,
                                                                  1U);
  expect_fixed_slot_compilation_rejection(plan);
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsInternalRegionEdgeRequiringMultipleTensorSlots) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(MemoryKind::kCudaDevice,
                                                                  TensorOutputStorage::kRuntimePool,
                                                                  false,
                                                                  TensorOutputStorage::kRuntimePool,
                                                                  1U,
                                                                  2U,
                                                                  1U);
  expect_fixed_slot_compilation_rejection(plan);
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsRegionEgressRequiringMultipleTensorSlots) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(MemoryKind::kCudaDevice,
                                                                  TensorOutputStorage::kRuntimePool,
                                                                  false,
                                                                  TensorOutputStorage::kRuntimePool,
                                                                  1U,
                                                                  1U,
                                                                  2U);
  expect_fixed_slot_compilation_rejection(plan);
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsFixedBufferWithMultipleInflightSamples) {
  const ExecutionPlan plan = compile_fused_region();
  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleFixedTensorSlot,
      "max_inflight_samples=2",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().internal_edges.front().edge;
        auto physical = std::ranges::find(state.payload_edges, edge, &PayloadEdgePlanRecord::edge);
        if (physical == state.payload_edges.end()) {
          return false;
        }
        physical->max_inflight_samples = 2U;
        return true;
      });
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsFixedBufferWithTemporalCache) {
  const ExecutionPlan plan = compile_fused_region();
  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleFixedTensorSlot,
      "cache_capacity=1",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const EdgeId edge = state.cuda_graph_capture_plans.front().internal_edges.front().edge;
        auto physical = std::ranges::find(state.payload_edges, edge, &PayloadEdgePlanRecord::edge);
        if (physical == state.payload_edges.end()) {
          return false;
        }
        physical->cache_capacity = 1U;
        return true;
      });
}

// REQ: COMPILE-088
TEST(CUDAGraphExecutionLowering, RejectsFixedBufferWithOutputHistory) {
  const ExecutionPlan plan = compile_fused_region();
  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleFixedTensorSlot,
      "history_depth=1",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto port = std::ranges::find(state.ports, output, &PortPlanRecord::id);
        if (port == state.ports.end()) {
          return false;
        }
        port->late_joiner_history_depth = 1U;
        return true;
      });
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsPassThroughInternalRegionOutput) {
  const ExecutionPlan plan =
      compile_fused_region(MemoryKind::kCudaDevice, false, true, TensorOutputStorage::kPassThrough);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kRuntimePoolTensorOutput;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("storage=kPassThrough"), std::string::npos);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsPassThroughRegionEgress) {
  const ExecutionPlan plan =
      compile_fused_region_with_boundaries(MemoryKind::kCudaDevice,
                                           TensorOutputStorage::kRuntimePool,
                                           false,
                                           TensorOutputStorage::kPassThrough);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kRuntimePoolTensorOutput;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("storage=kPassThrough"), std::string::npos);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsRegionIngressOutputWithAnotherConsumer) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(
      MemoryKind::kCudaDevice, TensorOutputStorage::kRuntimePool, true);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kExclusiveIngressOutput;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=ingress_output_has_other_consumers"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("fanout=2"), std::string::npos);
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsRegionOutputThatCanEmitMoreThanOneTensor) {
  const ExecutionPlan plan = compile_fused_region();
  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleTensorEmit,
      "reason=producer_emit_bound_not_one",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            state.cuda_graph_capture_plans.front().internal_edges.empty()) {
          return false;
        }
        const PortId output =
            state.cuda_graph_capture_plans.front().internal_edges.front().source_port;
        auto port = std::find_if(state.ports.begin(), state.ports.end(), [&](const auto& item) {
          return item.id == output;
        });
        if (port == state.ports.end()) {
          return false;
        }
        port->max_emits_per_compute = 2U;
        return true;
      },
      {"max_emits_per_compute=2"});
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, RejectsPageableHostBoundaryBuffers) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(MemoryKind::kHost);
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(state->cuda_graph_execution_regions.empty());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "CUDA_GRAPH_EXECUTION_REGION_INVALID" &&
               item.required == detail::cuda_graph_execution_rule::kCaptureCompatibleBoundaryBuffer;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find("memory_kind=kHost"), std::string::npos);
  const auto source_port =
      std::find_if(state->ports.begin(), state->ports.end(), [&](const auto& port) {
        return port.id == diagnostic->source.port;
      });
  ASSERT_NE(source_port, state->ports.end());
  EXPECT_EQ(diagnostic->source.vertex, source_port->owner);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RejectsBoundaryBufferOnAnotherCUDADevice) {
  const ExecutionPlan plan = compile_fused_region_with_boundaries(MemoryKind::kCudaDevice);
  ASSERT_TRUE(plan.ok()) << plan.json();

  expect_lowering_rejection_after_plan_change(
      plan,
      detail::cuda_graph_execution_rule::kSingleCUDADevice,
      "reason=conflicting_cuda_devices",
      [](auto& state) {
        if (state.cuda_graph_capture_plans.empty() ||
            !state.cuda_graph_capture_plans.front().ingress) {
          return false;
        }
        const PortId output = state.cuda_graph_capture_plans.front().ingress->source_port;
        auto allocation = std::find_if(state.tensor_allocations.begin(),
                                       state.tensor_allocations.end(),
                                       [&](const auto& item) { return item.output == output; });
        if (allocation == state.tensor_allocations.end()) {
          return false;
        }
        allocation->device = DeviceId{1};
        return true;
      },
      {"buffer_devices=0|1"});
}

// REQ: COMPILE-084
TEST(CUDAGraphExecutionLowering, ExcludesFusedRegionFromClosedIslandFastPath) {
  const ExecutionPlan ordinary = compile_closed_island_candidate();
  const auto* ordinary_state = detail::ExecutionPlanAccess::state(ordinary);
  ASSERT_NE(ordinary_state, nullptr);
  ASSERT_TRUE(ordinary.ok()) << ordinary.json();
  ASSERT_EQ(detail::derive_closed_island_activation_plan(*ordinary_state).size(), 1U);

  detail::ExecutionPlanState fused_source = *ordinary_state;
  ASSERT_TRUE(add_fused_region_records(fused_source));
  fused_source.cuda_graph_capture_plans.front().members.resize(1U);
  fused_source.cuda_graph_capture_plans.front().internal_edges.clear();
  EXPECT_TRUE(detail::derive_closed_island_activation_plan(fused_source).empty());

  detail::ExecutionPlanState fused_target = *ordinary_state;
  ASSERT_TRUE(add_fused_region_records(fused_target));
  auto& target_capture = fused_target.cuda_graph_capture_plans.front();
  target_capture.members.erase(target_capture.members.begin());
  target_capture.internal_edges.clear();
  fused_target.cuda_graph_execution_regions.front().fused_operator =
      target_capture.members.front().vertex;
  EXPECT_TRUE(detail::derive_closed_island_activation_plan(fused_target).empty());
}

// REQ: COMPILE-084
TEST(CUDAGraphExecutionLowering, ExcludesFusedRegionFromInlinePartitionFastPath) {
  const ExecutionPlan ordinary = compile_inline_partition_candidate();
  const auto* ordinary_state = detail::ExecutionPlanAccess::state(ordinary);
  ASSERT_NE(ordinary_state, nullptr);
  ASSERT_TRUE(ordinary.ok()) << ordinary.json();
  ASSERT_TRUE(detail::derive_inline_partition_execution_plan(*ordinary_state).has_value());

  detail::ExecutionPlanState fused = *ordinary_state;
  ASSERT_TRUE(add_fused_region_records(fused));
  EXPECT_FALSE(detail::derive_inline_partition_execution_plan(fused).has_value());
}

// REQ: COMPILE-081
TEST(CUDAGraphExecutionLowering, AllowsTwoFusedRegionsInOneExecutionPartition) {
  const ExecutionPlan plan = compile_two_fused_regions_in_one_partition();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 2U) << plan.json();
  EXPECT_EQ(state->cuda_graph_execution_regions[0].partition,
            state->cuda_graph_execution_regions[1].partition);

  ASSERT_EQ(state->partition_runners.size(), 1U);
  ASSERT_EQ(state->partition_runners.front().operators.size(), 2U);
  EXPECT_TRUE(
      std::all_of(state->partition_runners.front().operators.begin(),
                  state->partition_runners.front().operators.end(),
                  [](const auto& member) { return member.fused_cuda_graph_region.has_value(); }));
  // The edge between the two regions remains an ordinary Runtime edge.
  EXPECT_EQ(state->partition_runners.front().internal_edges.size(), 1U);

  const auto& first_region = state->cuda_graph_execution_regions[0];
  const auto& second_region = state->cuda_graph_execution_regions[1];
  const auto& first_egress = first_region.egress;
  const auto& second_ingress = second_region.ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(first_egress);
  HOLOSCAN_ASSERT_HAS_VALUE(second_ingress);
  ASSERT_EQ(first_egress->edge, second_ingress->edge);

  // The boundary between two regions has one physical allocation even though both region plans
  // retain a buffer record for their side of the edge.
  const auto shared_edge = first_egress->edge;
  const auto first_buffer = std::find_if(
      first_region.buffers.begin(), first_region.buffers.end(), [&](const auto& buffer) {
        return buffer.edge == shared_edge &&
               buffer.kind == detail::CUDAGraphCaptureBufferKind::kEgress;
      });
  const auto second_buffer = std::find_if(
      second_region.buffers.begin(), second_region.buffers.end(), [&](const auto& buffer) {
        return buffer.edge == shared_edge &&
               buffer.kind == detail::CUDAGraphCaptureBufferKind::kIngress;
      });
  ASSERT_NE(first_buffer, first_region.buffers.end());
  ASSERT_NE(second_buffer, second_region.buffers.end());
  EXPECT_EQ(first_buffer->fixed_allocation_id, second_buffer->fixed_allocation_id);

  // Each fused operator still uses the stream selected from its own region source. Sharing the
  // boundary allocation does not combine the regions or change either launch stream.
  for (const auto& region : state->cuda_graph_execution_regions) {
    const auto* capture_plan =
        detail::find_unique_cuda_graph_capture_plan_by_region(*state, region.graph, region.region);
    ASSERT_NE(capture_plan, nullptr);
    ASSERT_FALSE(capture_plan->internal_edges.empty());
    const auto stream = std::find_if(
        state->cuda_compute_streams.begin(),
        state->cuda_compute_streams.end(),
        [&](const auto& candidate) { return candidate.stream_id == region.stream_id; });
    ASSERT_NE(stream, state->cuda_compute_streams.end());
    EXPECT_NE(std::find(stream->fused_linear_edges.begin(),
                        stream->fused_linear_edges.end(),
                        capture_plan->internal_edges.front().edge),
              stream->fused_linear_edges.end());
  }
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, AssignsOneDistinctFixedAllocationIdentityPerTensorEdge) {
  const ExecutionPlan plan = compile_two_fused_regions_in_one_partition();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();

  std::vector<std::uint64_t> source_ports;
  std::vector<std::string> allocation_ids;
  for (const auto& region : state->cuda_graph_execution_regions) {
    for (const auto& buffer : region.buffers) {
      source_ports.push_back(buffer.source_port.value());
      allocation_ids.push_back(buffer.fixed_allocation_id);
    }
  }
  std::sort(source_ports.begin(), source_ports.end());
  source_ports.erase(std::unique(source_ports.begin(), source_ports.end()), source_ports.end());
  std::sort(allocation_ids.begin(), allocation_ids.end());
  allocation_ids.erase(std::unique(allocation_ids.begin(), allocation_ids.end()),
                       allocation_ids.end());

  ASSERT_EQ(source_ports.size(), 3U);  // This fixture contains three physical Tensor edges.
  EXPECT_EQ(allocation_ids.size(), source_ports.size());
}

// REQ: COMPILE-083
TEST(CUDAGraphExecutionLowering, UsesCanonicalFixedAllocationIdentityFormat) {
  EXPECT_EQ(detail::cuda_graph_fixed_allocation_id(17U, GraphId{23U}, EdgeId{41U}),
            "cuda-graph-fixed-allocation.17.graph.23.edge.41");
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, RebindsSharedAllocationIdentityWithPlanGeneration) {
  const ExecutionPlan plan = compile_two_fused_regions_in_one_partition();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 2U);

  detail::ExecutionPlanState rebound = *compiled;
  const PlanGenerationId next_generation{compiled->plan_generation + 1U};
  detail::rebind_plan_generation(rebound, next_generation);

  for (const auto& region : rebound.cuda_graph_execution_regions) {
    for (const auto& buffer : region.buffers) {
      EXPECT_EQ(buffer.fixed_allocation_id,
                detail::cuda_graph_fixed_allocation_id(next_generation, region.graph, buffer.edge));
    }
  }

  const auto& first_region = rebound.cuda_graph_execution_regions[0];
  const auto& second_region = rebound.cuda_graph_execution_regions[1];
  const auto& first_egress = first_region.egress;
  const auto& second_ingress = second_region.ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(first_egress);
  HOLOSCAN_ASSERT_HAS_VALUE(second_ingress);
  ASSERT_EQ(first_egress->edge, second_ingress->edge);
  const EdgeId shared_edge = first_egress->edge;
  const auto first_buffer = std::find_if(
      first_region.buffers.begin(), first_region.buffers.end(), [&](const auto& buffer) {
        return buffer.edge == shared_edge;
      });
  const auto second_buffer = std::find_if(
      second_region.buffers.begin(), second_region.buffers.end(), [&](const auto& buffer) {
        return buffer.edge == shared_edge;
      });
  ASSERT_NE(first_buffer, first_region.buffers.end());
  ASSERT_NE(second_buffer, second_region.buffers.end());
  EXPECT_EQ(first_buffer->fixed_allocation_id, second_buffer->fixed_allocation_id);
}

// REQ: COMPILE-082
TEST(CUDAGraphExecutionLowering, RebindsRegionGreenContextFromSelectedStream) {
  const ExecutionPlan plan = compile_single_operator_green_context_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 1U);

  detail::ExecutionPlanState rebound = *compiled;
  const auto& green_context_id = rebound.cuda_graph_execution_regions.front().green_context_id;
  HOLOSCAN_ASSERT_HAS_VALUE(green_context_id);
  const std::string prior_green_context_id = *green_context_id;
  // Exercise rebinding without a context-resource lookup entry. The selected stream still owns
  // the complete Green Context identity used by the fused region.
  rebound.cuda_green_context_resources.clear();
  detail::rebind_plan_generation(rebound, PlanGenerationId{compiled->plan_generation + 1U});

  const auto& region = rebound.cuda_graph_execution_regions.front();
  const auto stream =
      std::find_if(rebound.cuda_compute_streams.begin(),
                   rebound.cuda_compute_streams.end(),
                   [&](const auto& candidate) { return candidate.stream_id == region.stream_id; });
  ASSERT_NE(stream, rebound.cuda_compute_streams.end());
  ASSERT_TRUE(stream->green_context_id);
  EXPECT_EQ(region.green_context_id, stream->green_context_id);
  EXPECT_NE(region.green_context_id, prior_green_context_id);
  EXPECT_TRUE(detail::valid_cuda_graph_execution_regions(rebound));
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, RestoresOrdinaryExecutionAfterArtifactValidationFailure) {
  // Artifact consistency failures concern private execution-plan records. This test constructs
  // that internal failure and verifies the production recovery sequence directly. Exercising the
  // compile() branch requires controlled finalizer failure injection.
  const ExecutionPlan fused_plan = compile_fused_region();
  const ExecutionPlan ordinary_plan = compile_fused_region(MemoryKind::kCudaDevice, false, false);
  const auto* fused = detail::ExecutionPlanAccess::state(fused_plan);
  const auto* ordinary = detail::ExecutionPlanAccess::state(ordinary_plan);
  ASSERT_NE(fused, nullptr);
  ASSERT_NE(ordinary, nullptr);
  ASSERT_TRUE(fused_plan.ok()) << fused_plan.json();
  ASSERT_TRUE(ordinary_plan.ok()) << ordinary_plan.json();

  detail::ExecutionPlanState recovery = *ordinary;
  recovery.cuda_graph_capture_plans = fused->cuda_graph_capture_plans;
  recovery.cuda_graph_execution_regions = fused->cuda_graph_execution_regions;
  const std::size_t service_count = recovery.process_local_channel_services.size();
  const std::size_t publisher_count = recovery.process_local_channel_publishers.size();
  const std::size_t subscriber_count = recovery.process_local_channel_subscribers.size();

  auto channel_backup = detail::remove_cuda_graph_internal_runtime_channels(recovery);
  ASSERT_TRUE(channel_backup.has_value());
  EXPECT_LT(recovery.process_local_channel_subscribers.size(), subscriber_count);

  // Reproduce the state at compile()'s first artifact-finalization failure: fusion records and
  // removed internal channels are still present.
  EXPECT_FALSE(detail::finalize_plan_artifact(recovery));

  // Apply the same recovery sequence as compile(): discard CUDA Graph state first, then rebuild
  // the general execution records for an ordinary rejected plan.
  detail::discard_cuda_graph_capture_and_execution_plan(recovery, std::move(channel_backup));
  recovery.admitted = false;
  recovery.partition_runners.clear();
  recovery.inline_partition_execution.reset();
  recovery.closed_island_activations = detail::derive_closed_island_activation_plan(recovery);

  EXPECT_TRUE(recovery.partition_runners.empty());
  EXPECT_FALSE(recovery.inline_partition_execution);
  EXPECT_TRUE(recovery.cuda_graph_capture_plans.empty());
  EXPECT_TRUE(recovery.cuda_graph_execution_regions.empty());
  EXPECT_EQ(recovery.process_local_channel_services.size(), service_count);
  EXPECT_EQ(recovery.process_local_channel_publishers.size(), publisher_count);
  EXPECT_EQ(recovery.process_local_channel_subscribers.size(), subscriber_count);
  EXPECT_TRUE(detail::valid_process_local_channel_plan(recovery));
  EXPECT_TRUE(detail::valid_closed_island_activation_plan(recovery));
  EXPECT_TRUE(detail::finalize_plan_artifact(recovery));
}

// REQ: COMPILE-084
TEST(CUDAGraphExecutionLowering, LeavesSharedInternalChannelServiceIntact) {
  const ExecutionPlan fused_plan = compile_fused_region();
  const ExecutionPlan ordinary_plan = compile_fused_region(MemoryKind::kCudaDevice, false, false);
  const auto* fused = detail::ExecutionPlanAccess::state(fused_plan);
  const auto* ordinary = detail::ExecutionPlanAccess::state(ordinary_plan);
  ASSERT_NE(fused, nullptr);
  ASSERT_NE(ordinary, nullptr);

  detail::ExecutionPlanState malformed = *ordinary;
  malformed.cuda_graph_capture_plans = fused->cuda_graph_capture_plans;
  malformed.cuda_graph_execution_regions = fused->cuda_graph_execution_regions;
  ASSERT_FALSE(malformed.process_local_channel_subscribers.empty());
  auto outside_subscriber = malformed.process_local_channel_subscribers.front();
  outside_subscriber.edge = EdgeId{malformed.payload_edges.size() + 1U};
  malformed.process_local_channel_subscribers.push_back(std::move(outside_subscriber));
  const std::size_t service_count = malformed.process_local_channel_services.size();
  const std::size_t publisher_count = malformed.process_local_channel_publishers.size();
  const std::size_t subscriber_count = malformed.process_local_channel_subscribers.size();
  const std::string service_id = malformed.process_local_channel_services.front().service_id;
  const EdgeId internal_edge = malformed.process_local_channel_subscribers.front().edge;
  const EdgeId outside_edge = malformed.process_local_channel_subscribers.back().edge;

  EXPECT_FALSE(detail::remove_cuda_graph_internal_runtime_channels(malformed));
  ASSERT_EQ(malformed.process_local_channel_services.size(), service_count);
  ASSERT_EQ(malformed.process_local_channel_publishers.size(), publisher_count);
  ASSERT_EQ(malformed.process_local_channel_subscribers.size(), subscriber_count);
  EXPECT_EQ(malformed.process_local_channel_services.front().service_id, service_id);
  EXPECT_EQ(malformed.process_local_channel_publishers.front().service_id, service_id);
  EXPECT_EQ(malformed.process_local_channel_subscribers.front().edge, internal_edge);
  EXPECT_EQ(malformed.process_local_channel_subscribers.back().edge, outside_edge);
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationRejectsChangedFusedRegionStream) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 1U) << plan.json();
  detail::ExecutionPlanState malformed = *compiled;
  malformed.cuda_graph_execution_regions.front().stream_id += ".changed";
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
  const auto finalized = detail::finalize_plan_artifact(malformed);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().message, "CUDA_GRAPH_EXECUTION_ARTIFACT_INVALID");
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationRejectsStaleExecutionRegionDiagnostic) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 1U);

  detail::ExecutionPlanState malformed = *compiled;
  const auto& region = malformed.cuda_graph_execution_regions.front();
  malformed.diagnostics.push_back(Diagnostic{
      .severity = DiagnosticSeverity::kError,
      .token = "CUDA_GRAPH_EXECUTION_REGION_INVALID",
      .source = DiagnosticSource{.graph_generation = malformed.graph_identity.version,
                                 .graph = region.graph,
                                 .vertex = region.fused_operator},
      .json_pointer = "/graphs/0/regions/0/constraints/cuda_graph",
      .required = "stale_rule",
      .provided = "stale_rejection",
  });

  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationRequiresOneFusedRegionPerAcceptedCapturePlan) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_capture_plans.size(), 1U);
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 1U);

  detail::ExecutionPlanState missing_region = *compiled;
  missing_region.cuda_graph_execution_regions.clear();
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(missing_region));

  detail::ExecutionPlanState missing_capture_plan = *compiled;
  missing_capture_plan.cuda_graph_capture_plans.clear();
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(missing_capture_plan));

  detail::ExecutionPlanState duplicate_region = *compiled;
  duplicate_region.cuda_graph_execution_regions.push_back(
      duplicate_region.cuda_graph_execution_regions.front());
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(duplicate_region));

  detail::ExecutionPlanState duplicate_capture_plan = *compiled;
  duplicate_capture_plan.cuda_graph_capture_plans.push_back(
      duplicate_capture_plan.cuda_graph_capture_plans.front());
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(duplicate_capture_plan));
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationRejectsUnreportedCapturePlanFailure) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);

  detail::ExecutionPlanState malformed = *compiled;
  malformed.cuda_graph_capture_plans.front().members.clear();
  malformed.cuda_graph_execution_regions.clear();
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationDoesNotRequireGraphSnapshot) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();

  detail::ExecutionPlanState detached = *compiled;
  detached.graph = GraphView{};
  EXPECT_TRUE(detail::valid_cuda_graph_execution_regions(detached));
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationRejectsChangedFixedAllocationIdentity) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_FALSE(compiled->cuda_graph_execution_regions.empty());
  ASSERT_FALSE(compiled->cuda_graph_execution_regions.front().buffers.empty());
  detail::ExecutionPlanState malformed = *compiled;
  malformed.cuda_graph_execution_regions.front().buffers.front().fixed_allocation_id += ".changed";
  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
}

// REQ: COMPILE-083
// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationRejectsProducerStrideDriftFromAllocation) {
  const ExecutionPlan plan =
      compile_fused_region_with_direct_ingress_geometry({4U, 3U}, 8U, 12U, {3U, 1U});
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_FALSE(compiled->cuda_graph_capture_plans.empty());

  detail::ExecutionPlanState malformed = *compiled;
  const auto& ingress = malformed.cuda_graph_capture_plans.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(ingress);
  const PortId source_port = ingress->source_port;
  auto producer = std::find_if(malformed.ports.begin(),
                               malformed.ports.end(),
                               [&](const auto& port) { return port.id == source_port; });
  ASSERT_NE(producer, malformed.ports.end());
  auto& tensor_contract = producer->tensor_contract;
  HOLOSCAN_ASSERT_HAS_VALUE(tensor_contract);
  ASSERT_EQ(tensor_contract->axes.size(), 2U);
  tensor_contract->axes[0].max_abs_stride_bytes = 4U;

  EXPECT_FALSE(detail::valid_cuda_graph_execution_regions(malformed));
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationChecksEveryFusedRegionField) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 1U);
  ASSERT_FALSE(compiled->cuda_graph_execution_regions.front().buffers.empty());

  expect_region_record_mutation_rejected(
      *compiled, "graph", [](auto& record) { record.graph = GraphId{record.graph.value + 1U}; });
  expect_region_record_mutation_rejected(*compiled, "region", [](auto& record) {
    record.region = RegionId{record.region.value + 1U};
  });
  expect_region_record_mutation_rejected(*compiled, "fused_operator", [](auto& record) {
    record.fused_operator = VertexId{record.fused_operator.value + 1U};
  });
  expect_region_record_mutation_rejected(*compiled, "partition", [](auto& record) {
    record.partition = ExecutionPartitionId{record.partition.value + 1U};
  });
  expect_region_record_mutation_rejected(
      *compiled, "execution_lane", [](auto& record) { ++record.execution_lane; });
  expect_region_record_mutation_rejected(*compiled, "process_group", [](auto& record) {
    record.process_group = ProcessGroupId{record.process_group.value + 1U};
  });
  expect_region_record_mutation_rejected(
      *compiled, "device", [](auto& record) { record.device = DeviceId{record.device.value + 1}; });
  expect_region_record_mutation_rejected(
      *compiled, "stream_id", [](auto& record) { record.stream_id += ".changed"; });
  expect_region_record_mutation_rejected(*compiled, "green_context_id", [](auto& record) {
    record.green_context_id = record.green_context_id ? *record.green_context_id + ".changed"
                                                      : "unexpected-green-context";
  });
  expect_region_record_mutation_rejected(*compiled, "ingress", [](auto& record) {
    record.ingress = detail::CUDAGraphCaptureEdgeRecord{.edge = record.buffers.front().edge};
  });
  expect_region_record_mutation_rejected(*compiled, "egress", [](auto& record) {
    record.egress = detail::CUDAGraphCaptureEdgeRecord{.edge = record.buffers.back().edge};
  });
  expect_region_record_mutation_rejected(
      *compiled, "buffers", [](auto& record) { record.buffers.clear(); });
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, ArtifactValidationChecksEveryFixedBufferField) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_execution_regions.size(), 1U);
  ASSERT_FALSE(compiled->cuda_graph_execution_regions.front().buffers.empty());

  expect_buffer_record_mutation_rejected(*compiled, "kind", [](auto& record) {
    record.kind = detail::CUDAGraphCaptureBufferKind::kEgress;
  });
  expect_buffer_record_mutation_rejected(
      *compiled, "edge", [](auto& record) { record.edge = EdgeId{record.edge.value + 1U}; });
  expect_buffer_record_mutation_rejected(*compiled, "source_port", [](auto& record) {
    record.source_port = PortId{record.source_port.value() + 1U};
  });
  expect_buffer_record_mutation_rejected(*compiled, "target_port", [](auto& record) {
    record.target_port = PortId{record.target_port.value() + 1U};
  });
  expect_buffer_record_mutation_rejected(*compiled, "payload_identity", [](auto& record) {
    record.payload_identity.kind = PayloadIdentityKind::kScalar;
  });
  expect_buffer_record_mutation_rejected(*compiled, "layout", [](auto& record) {
    record.layout.rank = record.layout.rank.value_or(0U) + 1U;
  });
  expect_buffer_record_mutation_rejected(
      *compiled, "bounds.max_rank", [](auto& record) { ++record.bounds.max_rank; });
  expect_buffer_record_mutation_rejected(
      *compiled, "bounds.max_byte_span", [](auto& record) { ++record.bounds.max_byte_span; });
  expect_buffer_record_mutation_rejected(*compiled, "axes", [](auto& record) {
    if (record.axes.empty()) {
      record.axes.push_back(TensorAxisBounds{.max_extent = 1U, .max_abs_stride_bytes = 1U});
    } else {
      ++record.axes.front().max_extent;
    }
  });
  expect_buffer_record_mutation_rejected(*compiled, "fixed_shape", [](auto& record) {
    if (record.fixed_shape.empty()) {
      record.fixed_shape.push_back(1U);
    } else {
      record.fixed_shape.front() /= 2U;
    }
  });
  expect_buffer_record_mutation_rejected(
      *compiled, "active_slot_count", [](auto& record) { ++record.active_slot_count; });
  expect_buffer_record_mutation_rejected(
      *compiled, "aggregate_byte_capacity", [](auto& record) { ++record.aggregate_byte_capacity; });
  expect_buffer_record_mutation_rejected(*compiled, "memory_kind", [](auto& record) {
    record.memory_kind = MemoryKind::kCudaManaged;
  });
  expect_buffer_record_mutation_rejected(
      *compiled, "device", [](auto& record) { record.device = DeviceId{record.device.value + 1}; });
  expect_buffer_record_mutation_rejected(*compiled, "fixed_allocation_id", [](auto& record) {
    record.fixed_allocation_id += ".changed";
  });
  expect_buffer_record_mutation_rejected(
      *compiled, "pool_id", [](auto& record) { record.pool_id += ".changed"; });
  expect_buffer_record_mutation_rejected(
      *compiled, "allocator_root_id", [](auto& record) { record.allocator_root_id += ".changed"; });
}

// REQ: COMPILE-085
TEST(CUDAGraphExecutionLowering, RunnerValidationRejectsChangedFusedRegionGraph) {
  const ExecutionPlan plan = compile_fused_region();
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  detail::ExecutionPlanState malformed = *compiled;
  ASSERT_FALSE(malformed.partition_runners.empty());
  ASSERT_FALSE(malformed.partition_runners.front().operators.empty());
  auto& fused_region =
      malformed.partition_runners.front().operators.front().fused_cuda_graph_region;
  HOLOSCAN_ASSERT_HAS_VALUE(fused_region);
  fused_region->graph = GraphId{fused_region->graph.value + 1U};
  EXPECT_FALSE(detail::valid_partition_runner_plan(malformed));
}

}  // namespace
}  // namespace holoscan
