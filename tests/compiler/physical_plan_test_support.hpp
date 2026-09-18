/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "channel/catalog_inline_payload_contract.hpp"
#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/compile_platform.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/physical_plan_lowering.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_generation.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "compiler/profile_admission.hpp"
#include "compiler/profile_manifest_lowering.hpp"
#include "compiler/rate_plan_lowering.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/source_id_internal.hpp"
#include "core/tensor_allocation_record_validation.hpp"

namespace holoscan {

struct CompileLoanPod {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<CompileLoanPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CompileLoanPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CompileLoanPod> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &CompileLoanPod::value, offsetof(CompileLoanPod, value)));
};

struct CompileConstructorOnlyPod {
  explicit CompileConstructorOnlyPod(std::uint32_t initial) : value(initial) {}
  CompileConstructorOnlyPod() = delete;

  std::uint32_t value;
};

struct CompileMaximumCatalogInlinePod {
  std::array<std::uint8_t, channel_internal::kMaximumCatalogDataFrameBytes -
                               channel_internal::kCatalogDataAttributionBytes>
      bytes{};
};

struct CompileOversizedCatalogInlinePod {
  std::array<std::uint8_t, channel_internal::kMaximumCatalogDataFrameBytes -
                               channel_internal::kCatalogDataAttributionBytes + 1U>
      bytes{};
};

template <>
struct ChannelTypeIdentity<CompileConstructorOnlyPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CompileConstructorOnlyPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CompileConstructorOnlyPod> {
  static constexpr auto fields = pod_fields(pod_field("value", &CompileConstructorOnlyPod::value,
                                                      offsetof(CompileConstructorOnlyPod, value)));
};

template <>
struct ChannelTypeIdentity<CompileMaximumCatalogInlinePod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CompileMaximumCatalogInlinePod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CompileMaximumCatalogInlinePod> {
  static constexpr auto fields =
      pod_fields(pod_field("bytes", &CompileMaximumCatalogInlinePod::bytes,
                           offsetof(CompileMaximumCatalogInlinePod, bytes)));
};

template <>
struct ChannelTypeIdentity<CompileOversizedCatalogInlinePod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CompileOversizedCatalogInlinePod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CompileOversizedCatalogInlinePod> {
  static constexpr auto fields =
      pod_fields(pod_field("bytes", &CompileOversizedCatalogInlinePod::bytes,
                           offsetof(CompileOversizedCatalogInlinePod, bytes)));
};

}  // namespace holoscan

namespace holoscan::detail::physical_plan_test {

[[nodiscard]] PayloadIdentityRecord scalar_payload();
[[nodiscard]] PayloadIdentityRecord tensor_payload();
[[nodiscard]] PayloadIdentityRecord schema_payload();
[[nodiscard]] PayloadIdentityRecord image_payload();
[[nodiscard]] PayloadIdentityRecord tensor_packet_payload();
[[nodiscard]] VertexRecord vertex(std::uint64_t id, std::string_view name,
                                  VertexKind kind = VertexKind::kOperator);
[[nodiscard]] PortRecord output_port(std::uint64_t id, std::uint64_t owner, std::string_view name,
                                     PayloadIdentityRecord payload, std::uint32_t burst = 1U,
                                     PortOwnerKind owner_kind = PortOwnerKind::kOperator);
[[nodiscard]] PortRecord input_port(std::uint64_t id, std::uint64_t owner, std::string_view name,
                                    PayloadIdentityRecord payload, std::uint32_t depth = 1U,
                                    PortOwnerKind owner_kind = PortOwnerKind::kOperator);
[[nodiscard]] EdgeRecord edge(std::uint64_t id, std::uint64_t source, std::uint64_t target,
                              std::uint32_t depth = 1U,
                              QueuePolicy policy = QueuePolicy::kBlocking);
[[nodiscard]] TensorFieldConfigurationRecord tensor_field(
    std::uint16_t field_id, std::uint16_t tensor_index, MemoryKind memory_kind, std::uint32_t rank,
    std::uint64_t max_byte_span,
    InterfaceTensorEmitPolicy emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded);
[[nodiscard]] std::vector<std::optional<Profile>> standard_profiles(std::size_t count);
[[nodiscard]] std::vector<VertexProcessPlacementPlanRecord> default_process_placements(
    const GraphSnapshotData& snapshot);
[[nodiscard]] std::vector<VertexProcessPlacementPlanRecord> split_process_placements(
    const GraphSnapshotData& snapshot);
[[nodiscard]] std::vector<ExecutionPartitionRecord> default_execution_partitions(
    const GraphSnapshotData& snapshot);
[[nodiscard]] std::vector<ChannelEndpointRecord> catalog_endpoints(
    const GraphSnapshotData& snapshot);

struct LowerForTestOptions {
  bool force_holomq_process_local_inline{};
  bool force_holomq_process_local_event_wake{};
  std::span<const ResolvedQueueSizing> resolved_queue_sizings{};
  std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes{};
};
inline constexpr LowerForTestOptions kForceHoloMQProcessLocalInline{
    .force_holomq_process_local_inline = true,
};
inline constexpr LowerForTestOptions kForceHoloMQProcessLocalInlineAndEventWake{
    .force_holomq_process_local_inline = true,
    .force_holomq_process_local_event_wake = true,
};

[[nodiscard]] PhysicalPlanLoweringResult lower_for_test(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> profiles,
    std::span<const InputSelectionPlanRecord> selections = {},
    std::span<const BracketedWindowPlanRecord> windows = {},
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets = {},
    const RuntimePackageCapabilityView* package_override = nullptr,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings = {},
    std::span<const ExecutionPartitionRecord> execution_partitions = {},
    LowerForTestOptions options = {});
[[nodiscard]] ExecutionPlanState published_physical_plan_state(
    PhysicalPlanLoweringResult result, PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> placements = {},
    const RatePlanLoweringResult* exact_rate_plan = nullptr);
[[nodiscard]] bool has_error(const PhysicalPlanLoweringResult& result, std::string_view token);
[[nodiscard]] const PortPlanRecord* find_port(const PhysicalPlanLoweringResult& result, PortId id);
[[nodiscard]] const TensorAllocationPlanRecord* find_allocation(
    const PhysicalPlanLoweringResult& result, PortId output);
[[nodiscard]] const PoolCapacityContributorRecord* find_contributor(
    const PhysicalPlanLoweringResult& result, std::string_view pool_id, std::string_view suffix);

struct MultiTensorSnapshot {
  GraphSnapshotData graph;
  std::vector<std::optional<Profile>> profiles;
  std::vector<VertexProcessPlacementPlanRecord> placements;
  std::vector<ExecutionPartitionRecord> partitions;
};

[[nodiscard]] MultiTensorSnapshot make_multi_tensor_snapshot(
    std::uint64_t serialized_payload_capacity_bytes);
[[nodiscard]] ExecutionPlanState physical_artifact_state();

class ManifestTensorOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(tensor, "tensor")
        .max_emits_per_compute(2U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLFloat, 32U, 1U},
                               .rank = 2U},
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(scalar, "scalar").max_emits_per_compute(1U);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> tensor;
  Output<std::uint32_t> scalar;
};

// Minimal Tensor input sink with queue_depth(2); paired with ManifestTensorOperator.
class ManifestTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

// Emits a CUDA-device Tensor (rank 1, 4096 B bound) and a scalar on separate outputs.
// Used in device-binding and placement tests where an explicit device ID must be supplied
// (e.g. PublicTensorOutputPlacementResolvesStableNamesAndOwnsStrings,
// PinnedCatalogPoolUsesPublicOutputDevicePlacement).
class DeviceTensorOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(tensor, "tensor")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLFloat, 32U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(scalar, "scalar");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> tensor;
  Output<std::uint32_t> scalar;
};

// Passes a device Tensor through one input to one output; used to verify that an
// Operator holding both consumer and producer obligations cannot carry two distinct
// device bindings (PublicTensorOutputPlacementRejectsTwoDevicesForOneOperator).
class DeviceTensorRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLFloat, 32U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Minimal uint32 input sink; used in external-import roundtrip and closed-relation
// corruption tests (e.g. ExternalImportPreservesGraphQueueGeometryAcrossCatalogRoundTrip,
// RejectsExternalChannelClosedRelationCorruption).
class ExternalScalarSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

// Minimal uint32 source; adds a decoy process-local edge in closed-relation corruption
// tests (RejectsExternalChannelClosedRelationCorruption).
class ExternalScalarSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

// Consumes schema::StringT; used in schema-capacity validation and corruption tests
// (RejectsExternalSchemaCapacityCorruption).
class ExternalSchemaSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::StringT> input;
};

template <typename T>
class ExternalPodSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<T> input;
};

// Covers every generic-loan eligibility outcome across three output types:
//   pod             → eligible (nothrow-default-constructible POD layout)
//   schema          → eligible (nothrow-default-constructible generated schema)
//   constructor-only → ineligible (no default constructor → no allocate() capability)
// Key test: FreezesTypedGenericLoanEvidenceWithoutFunctionPointers.
class GenericLoanEvidenceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(pod, "pod").max_emits_per_compute(2U);
    spec.output(schema, "schema").max_emits_per_compute(3U);
    spec.output(constructor_only, "constructor-only").max_emits_per_compute(4U);
  }

  expected<void, Error> compute(ExecutionContext&) override {
    return constructor_only.emit(CompileConstructorOnlyPod{17U});
  }

  Output<CompileLoanPod> pod;
  Output<schema::StringT> schema;
  Output<CompileConstructorOnlyPod> constructor_only;
};

}  // namespace holoscan::detail::physical_plan_test
