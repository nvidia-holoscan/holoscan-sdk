/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "compiler/cuda_service_stream_plan.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "physical_plan_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan::detail {
namespace {

using namespace physical_plan_test;

class CanonicalOperatorPathTensorGraph final : public Graph {
 public:
  explicit CanonicalOperatorPathTensorGraph(std::string operator_name)
      : operator_name_(std::move(operator_name)) {}

  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto source = op<DeviceTensorOperator>(operator_name_);
    const auto sink = op<ManifestTensorSinkOperator>("sink");
    add_flow(source->tensor, sink->input);
  }

 private:
  std::string operator_name_;
  ClockKey clock_import_{};
};

// One compatible descriptor-schema service fans out to both consumers. The right consumer also
// owns an unrelated Tensor output on another device:
//
//   image-source [device 0] --+--> left
//                             `--> right -- tensor-out [device 1]
//
// REQ: COMPILE-052
TEST(CompilePhysicalPlan, CrossProcessTensorFanoutValidatesEverySubscriberDevice) {
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "image-source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "image", payload),
      input_port(2U, 2U, "image", payload),
      input_port(3U, 3U, "image", payload),
      output_port(4U, 3U, "tensor-out", tensor_payload()),
  };
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  for (std::size_t index = 0U; index < 3U; ++index) {
    snapshot.ports[index].serialized_payload_capacity_bytes = type_ops->wire_capacity;
    snapshot.ports[index].setup.tensor_layout = layout;
  }
  snapshot.ports[3].setup.tensor_layout = layout;
  snapshot.ports[3].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  for (auto& item : snapshot.edges) {
    item.effective_tensor_payload = PayloadOptionsRecord{
        .max_tensor_byte_span = 1024U,
        .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
    };
  }
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{4U}, .device = DeviceId{1}},
  };

  // Key setup: the second subscriber is not the service anchor and conflicts only with its
  // descriptor-schema input obligation.
  const auto result = lower_physical_plan(snapshot,
                                          42U,
                                          profiles,
                                          runtime_package_capability_view(),
                                          {},
                                          {},
                                          {},
                                          placements,
                                          {},
                                          partitions,
                                          bindings);

  // Key assertion: every subscriber contributes its device, so the right Operator is rejected.
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID");
  EXPECT_EQ(result.errors.front().source.vertex, VertexId{3U});
  EXPECT_NE(result.errors.front().provided.find("reason=operator_device_conflict"),
            std::string::npos);
}

// A generated schema uses the canonical generated payload directly across process groups. The
// private schema envelope belongs only to process-local retention and must not inflate the HoloMQ
// service capacity reconstructed independently by each worker.
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: CHAN-085
TEST(CompilePhysicalPlan, CrossProcessGeneratedSchemaUsesCanonicalWireCapacity) {
  const auto payload = schema_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::StringT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", payload),
      input_port(2U, 2U, "in", payload),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[1].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto endpoints = catalog_endpoints(snapshot);
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  const RuntimePackageCapabilityView package = runtime_package_capability_view();

  auto result = lower_physical_plan(
      snapshot, 42U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});

  ASSERT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.process_local_channel_services.empty());
  ASSERT_EQ(result.cross_process_channel_services.size(), 1U);
  EXPECT_EQ(result.cross_process_channel_services.front().payload_size_bytes,
            type_ops->wire_capacity);

  auto state = published_physical_plan_state(std::move(result), 42U, placements);
  ASSERT_EQ(state.payload_edges.size(), 1U);
  ASSERT_EQ(state.channel_locality.size(), 1U);
  state.payload_edges.front().locality = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requested = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().accepted = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requires_runtime_attach = true;
  EXPECT_TRUE(valid_cross_process_channel_plan(state));
}

// A Tensor-bearing generated schema cannot send an ordinary Tensor descriptor across a process
// boundary. A positive authored byte bound plus kCopyIfNeeded gives the compiler exact authority
// to provision one fixed same-kind HoloMQ staging pool instead.
//
//   Image source [process A] --[bounded same-kind staging]--> Image sink [process B]
//
// REQ: CHAN-085
// REQ: CHAN-099
// REQ: CHAN-105
// REQ: CHAN-106
// REQ: CHAN-135
// REQ: TENSOR-SCHEMA-005
TEST(CompilePhysicalPlan, CrossProcessImageRequiresAndCarriesBoundedMemoryPreservingStaging) {
  constexpr std::uint64_t kMaximumTensorBytes = 1024U;
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "image-source"), vertex(2U, "image-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "image", payload),
      input_port(2U, 2U, "image", payload),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[1].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports[0].setup.tensor_layout = layout;
  snapshot.ports[1].setup.tensor_layout = layout;
  snapshot.edges = {edge(1U, 1U, 2U)};
  snapshot.edges.front().effective_tensor_payload = PayloadOptionsRecord{
      .max_tensor_byte_span = kMaximumTensorBytes,
      .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto endpoints = catalog_endpoints(snapshot);
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  const RuntimePackageCapabilityView package = runtime_package_capability_view();

  // Key setup: removing the finite staging bound leaves no realizable cross-process provider.
  auto unbounded = snapshot;
  unbounded.edges.front().effective_tensor_payload->max_tensor_byte_span.reset();
  const auto rejected = lower_physical_plan(
      unbounded, 42U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});
  EXPECT_TRUE(has_error(rejected, "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED"));
  EXPECT_TRUE(rejected.cross_process_channel_services.empty());

  auto result = lower_physical_plan(
      snapshot, 42U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_EQ(result.cross_process_channel_services.size(), 1U);
  const auto& service = result.cross_process_channel_services.front();
  // Key assertion: the service carries exact fixed-pool facts without fabricating a direct output
  // allocation owned by the application port.
  EXPECT_TRUE(result.tensor_allocations.empty());
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_size_bytes, type_ops->wire_capacity);
  EXPECT_EQ(service.tensor_max_byte_span, kMaximumTensorBytes);
  EXPECT_FALSE(service.tensor_pool_id.empty());
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kHost);
  EXPECT_FALSE(service.tensor_device.valid());
  EXPECT_EQ(service.tensor_emit_policy, InterfaceTensorEmitPolicy::kCopyIfNeeded);

  // Key setup: pinned-host and device staging retain their exact memory kind and require the
  // producing schema output's explicit CUDA placement.
  for (const MemoryKind memory_kind : {MemoryKind::kPinnedHost, MemoryKind::kCudaDevice}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(memory_kind));
    auto placed = snapshot;
    placed.ports[0].setup.tensor_layout->memory_kind = memory_kind;
    placed.ports[1].setup.tensor_layout->memory_kind = memory_kind;
    constexpr DeviceId kDevice{0};
    const std::array bindings{TensorDeviceBindingRecord{.port = PortId{1U}, .device = kDevice}};
    auto placed_result = lower_physical_plan(
        placed, 43U, profiles, package, {}, {}, endpoints, placements, {}, partitions, bindings);
    ASSERT_TRUE(placed_result.errors.empty())
        << (placed_result.errors.empty() ? std::string{} : placed_result.errors.front().token);
    ASSERT_EQ(placed_result.cross_process_channel_services.size(), 1U);
    const auto& placed_service = placed_result.cross_process_channel_services.front();
    // Key assertion: no pageable-host fallback is admitted; both placements retain the exact
    // device identity needed for their memory-preserving BufferHandle contract.
    EXPECT_EQ(placed_service.tensor_memory_kind, memory_kind);
    EXPECT_EQ(placed_service.tensor_device, kDevice);
    ExecutionPlanState service_state;
    service_state.cross_process_channel_services = placed_result.cross_process_channel_services;
    const auto requirements = lower_cuda_service_stream_requirements(service_state);
    if (memory_kind == MemoryKind::kCudaDevice) {
      ASSERT_EQ(placed_result.tensor_device_resources.size(), 2U);
      EXPECT_TRUE(
          std::ranges::all_of(placed_result.tensor_device_resources,
                              [&](const auto& resource) { return resource.device == kDevice; }));
      ASSERT_EQ(requirements.size(), 2U);
      EXPECT_EQ(requirements[0].process_group, ProcessGroupId{1U});
      EXPECT_EQ(requirements[1].process_group, ProcessGroupId{2U});
      EXPECT_EQ(requirements[0].device, kDevice);
      EXPECT_EQ(requirements[1].device, kDevice);
      EXPECT_NE(requirements[0].owner_id, requirements[1].owner_id);
      EXPECT_TRUE(requirements[0].owner_id.starts_with("service.channel.cross.source.1."));
      EXPECT_TRUE(requirements[1].owner_id.starts_with("service.channel.cross.source.1."));
    } else {
      // Key assertion: pinned-host schema staging is host-to-host and emits neither an unused
      // CUDA service stream nor a zero-pool device resource for either process.
      EXPECT_TRUE(requirements.empty());
      EXPECT_TRUE(placed_result.cuda_compute_streams.empty());
      EXPECT_TRUE(placed_result.tensor_device_resources.empty());
    }
  }

  auto managed = snapshot;
  managed.ports[0].setup.tensor_layout->memory_kind = MemoryKind::kCudaManaged;
  managed.ports[1].setup.tensor_layout->memory_kind = MemoryKind::kCudaManaged;
  constexpr DeviceId kManagedDevice{0};
  const std::array managed_binding{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = kManagedDevice}};

  // Key setup: request managed storage for the otherwise valid descriptor-schema boundary.
  const auto managed_result = lower_physical_plan(managed,
                                                  44U,
                                                  profiles,
                                                  package,
                                                  {},
                                                  {},
                                                  endpoints,
                                                  placements,
                                                  {},
                                                  partitions,
                                                  managed_binding);

  // Key assertion: HoloMQ has no cross-process managed provider, so no managed-only service can
  // reach Runtime stream admission without a compiler-planned device resource.
  EXPECT_TRUE(has_error(managed_result, "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED"));
  EXPECT_TRUE(managed_result.cross_process_channel_services.empty());
  EXPECT_TRUE(managed_result.tensor_device_resources.empty());

  auto state = published_physical_plan_state(std::move(result), 42U, placements);
  ASSERT_EQ(state.payload_edges.size(), 1U);
  ASSERT_EQ(state.channel_locality.size(), 1U);
  state.payload_edges.front().locality = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requested = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().accepted = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requires_runtime_attach = true;
  EXPECT_TRUE(valid_cross_process_channel_plan(state));
}

// A multi-Tensor schema freezes one allocation and one staging contract per canonical field.
// The field-free placement binding supplies one ordinal to every stream-ordered field and is
// intentionally ignored by the pageable-host field.
//
//   multi-source --[frames: host, device, pinned-host]--> multi-sink
//
// REQ: CHAN-123
// REQ: CHAN-126
// REQ: OPERATOR-036
TEST(CompilePhysicalPlan,
     MultiTensorSchemaUsesCanonicalPerFieldPoolsCapacityAndFieldFreePlacement) {
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  auto fixture = make_multi_tensor_snapshot(type_ops->wire_capacity);
  auto& snapshot = fixture.graph;
  snapshot.ports[0].setup.tensor_fields[0].tensor_layout->memory_kind = MemoryKind::kPinnedHost;
  snapshot.ports[0].setup.tensor_fields[2].tensor_layout->memory_kind = MemoryKind::kCudaDevice;
  snapshot.ports[1].setup.tensor_fields = snapshot.ports[0].setup.tensor_fields;
  const std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kExact,
      .max_selected_samples = 1U,
      .cache_capacity = 1U,
  }};
  constexpr DeviceId kDevice{2};
  const std::array bindings{TensorDeviceBindingRecord{.port = PortId{1U}, .device = kDevice}};

  auto result = lower_physical_plan(snapshot,
                                    42U,
                                    fixture.profiles,
                                    runtime_package_capability_view(),
                                    selections,
                                    {},
                                    {},
                                    fixture.placements,
                                    {},
                                    fixture.partitions,
                                    bindings);

  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_NE(find_port(result, PortId{1U}), nullptr);
  ASSERT_NE(find_port(result, PortId{2U}), nullptr);
  EXPECT_TRUE(result.tensor_allocations.empty());
  ASSERT_EQ(result.tensor_field_allocations.size(), 3U);
  constexpr std::array<std::uint16_t, 3U> kFieldIds{2U, 7U, 11U};
  std::uint32_t total_slots = 0U;
  for (std::size_t index = 0U; index < result.tensor_field_allocations.size(); ++index) {
    const auto& allocation = result.tensor_field_allocations[index];
    EXPECT_EQ(allocation.field_id, kFieldIds[index]);
    EXPECT_EQ(allocation.tensor_index, index);
    EXPECT_TRUE(validate_tensor_field_allocation_record(allocation, 42U));
    EXPECT_TRUE(
        validate_tensor_field_capacity_contributors(allocation, result.pool_capacity_contributors));
    total_slots += allocation.slot_count;
    const auto contributor_count = std::count_if(
        result.pool_capacity_contributors.begin(),
        result.pool_capacity_contributors.end(),
        [&](const auto& contributor) { return contributor.pool_id == allocation.pool_id; });
    EXPECT_EQ(contributor_count, 5U);
  }
  EXPECT_LE(total_slots, kMaximumDescriptorTensorFieldPoolSlots);
  EXPECT_FALSE(result.tensor_field_allocations[0].device.valid());
  EXPECT_EQ(result.tensor_field_allocations[1].device, kDevice);
  EXPECT_EQ(result.tensor_field_allocations[2].device, kDevice);
  EXPECT_TRUE(validate_tensor_field_allocation_order(result.tensor_field_allocations));

  // Key setup: corrupt the canonical output/field order that enables allocation-free spans.
  auto noncanonical_allocations = result.tensor_field_allocations;
  std::swap(noncanonical_allocations.front(), noncanonical_allocations.back());
  // Key assertion: publication validation rejects an order that would truncate a binary lookup.
  EXPECT_FALSE(validate_tensor_field_allocation_order(noncanonical_allocations));
  auto noncanonical_result = result;
  noncanonical_result.tensor_field_allocations = std::move(noncanonical_allocations);
  auto noncanonical_state =
      published_physical_plan_state(std::move(noncanonical_result), 42U, fixture.placements);
  const auto noncanonical_finalized = finalize_plan_artifact(noncanonical_state);
  ASSERT_FALSE(noncanonical_finalized);
  EXPECT_EQ(noncanonical_finalized.error().message,
            "PHYSICAL_TENSOR_FIELD_ALLOCATION_ORDER_INVALID");

  ASSERT_EQ(result.cross_process_channel_services.size(), 1U);
  const auto& service = result.cross_process_channel_services.front();
  ASSERT_EQ(service.tensor_fields.size(), 3U);
  for (std::size_t index = 0U; index < service.tensor_fields.size(); ++index) {
    EXPECT_EQ(service.tensor_fields[index].field_id, kFieldIds[index]);
    EXPECT_EQ(service.tensor_fields[index].tensor_index, index);
    ASSERT_TRUE(service.tensor_fields[index].dtype);
    EXPECT_EQ(service.tensor_fields[index].dtype->code, kDLUInt);
    EXPECT_EQ(service.tensor_fields[index].dtype->bits, 8U);
    EXPECT_EQ(service.tensor_fields[index].dtype->lanes, 1U);
    ASSERT_TRUE(service.tensor_fields[index].rank);
    EXPECT_EQ(*service.tensor_fields[index].rank, index == 1U ? 2U : 1U);
  }
  EXPECT_FALSE(service.tensor_fields[0].device.valid());
  EXPECT_EQ(service.tensor_fields[1].device, kDevice);
  EXPECT_EQ(service.tensor_fields[2].device, kDevice);
  ASSERT_EQ(result.tensor_device_resources.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(result.tensor_device_resources, [&](const auto& resource) {
    return resource.device == kDevice;
  }));
  EXPECT_TRUE(cross_process_tensor_service_requires_stream_in_group(service, ProcessGroupId{1U}));
  EXPECT_TRUE(cross_process_tensor_service_requires_stream_in_group(service, ProcessGroupId{2U}));

  auto reversed = result.pool_capacity_contributors;
  std::reverse(reversed.begin(), reversed.end());
  for (const auto& allocation : result.tensor_field_allocations) {
    EXPECT_TRUE(validate_tensor_field_capacity_contributors(allocation, reversed));
  }

  auto state = published_physical_plan_state(std::move(result), 42U, fixture.placements);
  ASSERT_EQ(state.payload_edges.size(), 1U);
  ASSERT_EQ(state.channel_locality.size(), 1U);
  state.payload_edges.front().locality = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requested = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().accepted = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requires_runtime_attach = true;
  EXPECT_TRUE(valid_cross_process_channel_plan(state));

  // Key assertion: a self-consistent but source-divergent service field is rejected against the
  // authoritative allocation, and descriptor inventories above the 64-field mask ceiling fail.
  auto& service_fields = state.cross_process_channel_services.front().tensor_fields;
  const auto canonical_fields = service_fields;
  ++service_fields.front().max_byte_span;
  EXPECT_FALSE(valid_cross_process_channel_plan(state));
  service_fields = canonical_fields;
  while (service_fields.size() <= 64U) {
    auto field = service_fields.back();
    field.field_id = static_cast<std::uint16_t>(field.field_id + 1U);
    field.tensor_index = static_cast<std::uint16_t>(service_fields.size());
    field.pool_id += "." + std::to_string(service_fields.size());
    service_fields.push_back(std::move(field));
  }
  EXPECT_FALSE(valid_cross_process_channel_plan(state));
}

// A descriptor envelope larger than the HoloMQ fixed-payload family is rejected before Runtime.
//
//   multi-source --[oversized descriptor envelope]--X--> multi-sink
//
// REQ: CHAN-125
TEST(CompilePhysicalPlan, MultiTensorSchemaRejectsOversizedDescriptorEnvelopeAtCompileTime) {
  // Key setup: author a cross-process descriptor schema one byte beyond the endpoint maximum.
  auto fixture = make_multi_tensor_snapshot(channel_internal::kMaximumInlinePayloadBytes + 1U);

  const auto result = lower_physical_plan(fixture.graph,
                                          42U,
                                          fixture.profiles,
                                          runtime_package_capability_view(),
                                          {},
                                          {},
                                          {},
                                          fixture.placements,
                                          {},
                                          fixture.partitions,
                                          {});

  // Key assertion: compilation reports a stable capacity diagnostic and creates no endpoint.
  EXPECT_TRUE(has_error(result, "CHANNEL_SCHEMA_DESCRIPTOR_ENVELOPE_CAPACITY_EXCEEDED"));
  EXPECT_TRUE(result.cross_process_channel_services.empty());
}

//   multi-source[frames: three Tensor fields]   (isolated; no edges)
//
// REQ: CHAN-123
TEST(CompilePhysicalPlan, MultiTensorSchemaRejectsUnboundedUnsupportedAndAggregateFields) {
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "multi-source")};
  snapshot.ports = {output_port(1U, 1U, "frames", payload)};
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[0].descriptor_tensor_field_count = 3U;
  snapshot.ports[0].setup.tensor_fields = {
      tensor_field(2U, 0U, MemoryKind::kHost, 1U, 128U),
      tensor_field(7U, 1U, MemoryKind::kHost, 2U, 512U),
      tensor_field(11U, 2U, MemoryKind::kHost, 1U, 256U),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  auto unbounded = snapshot;
  unbounded.ports[0].setup.tensor_fields[1].tensor_allocation.reset();
  auto result = lower_for_test(unbounded, 42U, profiles);
  EXPECT_TRUE(has_error(result, "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED"));
  EXPECT_TRUE(result.tensor_field_allocations.empty());

  auto unsupported = snapshot;
  unsupported.ports[0].setup.tensor_fields[1].tensor_layout->memory_kind = MemoryKind::kUnknown;
  result = lower_for_test(unsupported, 42U, profiles);
  EXPECT_TRUE(has_error(result, "CHANNEL_SCHEMA_TENSOR_FIELD_UNSUPPORTED_MEMORY"));
  EXPECT_TRUE(result.tensor_field_allocations.empty());

  auto mixed_carriage = snapshot;
  mixed_carriage.ports[0].setup.tensor_fields[0].payload_options = TensorFieldTransferRecord{
      .max_inline_bytes = 64U,
      .max_tensor_byte_span = 128U,
      .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
  };
  result = lower_for_test(mixed_carriage, 42U, profiles);
  EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED"));
  EXPECT_TRUE(result.tensor_field_allocations.empty());

  auto excessive = snapshot;
  excessive.ports[0].setup.max_emits_per_compute = 1025U;
  result = lower_for_test(excessive, 42U, profiles);
  const auto diagnostic =
      std::find_if(result.errors.begin(), result.errors.end(), [](const auto& error) {
        return error.token == "CHANNEL_SCHEMA_TENSOR_BUFFER_COUNT_EXCEEDED";
      });
  ASSERT_NE(diagnostic, result.errors.end());
  EXPECT_EQ(diagnostic->required, "total_buffer_count<=4096");
  EXPECT_EQ(diagnostic->provided, "total_buffer_count=6150");
  EXPECT_TRUE(result.tensor_field_allocations.empty());
  EXPECT_TRUE(result.pool_capacity_contributors.empty());
}

// Descriptor-backed schema staging geometry is authored per edge. Compatible queue policies do
// not permit coalescing arms whose finite Tensor byte spans differ.
//
//   Image source [process A] --[1024 B]--> left  [process B]
//                            --[4096 B]--> right [process C]
//
// REQ: CHAN-085
// REQ: CHAN-099
TEST(CompilePhysicalPlan, CrossProcessImageFanoutSplitsPerArmStagingIncompatibility) {
  constexpr std::array<std::uint64_t, 2U> kMaximumTensorBytes{1024U, 4096U};
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "image-source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "image", payload),
      input_port(2U, 2U, "image", payload),
      input_port(3U, 3U, "image", payload),
  };
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  for (auto& port : snapshot.ports) {
    port.serialized_payload_capacity_bytes = type_ops->wire_capacity;
    port.setup.tensor_layout = layout;
  }
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  for (std::size_t index = 0U; index < snapshot.edges.size(); ++index) {
    snapshot.edges[index].effective_tensor_payload = PayloadOptionsRecord{
        .max_tensor_byte_span = kMaximumTensorBytes[index],
        .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
    };
  }
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);

  const auto result = lower_physical_plan(snapshot,
                                          42U,
                                          profiles,
                                          runtime_package_capability_view(),
                                          {},
                                          {},
                                          {},
                                          placements,
                                          {},
                                          partitions,
                                          {});

  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_EQ(result.cross_process_channel_services.size(), 2U);
  ASSERT_EQ(result.cross_process_channel_publishers.size(), 2U);
  ASSERT_EQ(result.cross_process_channel_subscribers.size(), 2U);
  std::array<std::uint64_t, 2U> admitted_spans{};
  for (std::size_t index = 0U; index < result.cross_process_channel_services.size(); ++index) {
    const auto& service = result.cross_process_channel_services[index];
    EXPECT_EQ(service.max_subscribers, 1U);
    EXPECT_EQ(service.tensor_emit_policy, InterfaceTensorEmitPolicy::kCopyIfNeeded);
    EXPECT_EQ(service.edge, result.cross_process_channel_publishers[index].edge);
    admitted_spans[index] = service.tensor_max_byte_span;
  }
  std::sort(admitted_spans.begin(), admitted_spans.end());
  EXPECT_EQ(admitted_spans, kMaximumTensorBytes);
  EXPECT_NE(result.cross_process_channel_services[0].service_id,
            result.cross_process_channel_services[1].service_id);
}

// A descriptor-backed schema output can allocate directly from its final HoloMQ-exportable pool.
// The existing staging contract remains independently available for ordinary foreign Image values.
//
//   Image source allocation [process A] --[BufferHandle]--> Image sink [process B]
//
// REQ: OPERATOR-031
// REQ: CHAN-107
// REQ: CHAN-122
// REQ: TENSOR-SCHEMA-005
TEST(CompilePhysicalPlan, CrossProcessImageOutputAllocationAdmitsFinalProviderBacking) {
  constexpr std::uint64_t kMaximumTensorBytes = 1024U;
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "image-source"), vertex(2U, "image-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "image", payload),
      input_port(2U, 2U, "image", payload),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[1].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports[0].setup.tensor_layout = layout;
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = kMaximumTensorBytes};
  snapshot.ports[1].setup.tensor_layout = layout;
  snapshot.edges = {edge(1U, 1U, 2U)};
  snapshot.edges.front().effective_tensor_payload = PayloadOptionsRecord{
      .max_tensor_byte_span = kMaximumTensorBytes,
      .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);

  const auto result = lower_physical_plan(snapshot,
                                          42U,
                                          profiles,
                                          runtime_package_capability_view(),
                                          {},
                                          {},
                                          {},
                                          placements,
                                          {},
                                          partitions,
                                          {});

  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  // Key assertion: the source allocation survives as one finite pool while the schema service
  // retains its separately identified fallback staging contract.
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  EXPECT_EQ(result.tensor_allocations.front().output, PortId{1U});
  EXPECT_EQ(result.tensor_allocations.front().memory_kind, MemoryKind::kHost);
  EXPECT_EQ(result.tensor_allocations.front().bounds.max_rank, 1U);
  EXPECT_EQ(result.tensor_allocations.front().bounds.max_byte_span, kMaximumTensorBytes);
  ASSERT_EQ(result.cross_process_channel_services.size(), 1U);
  const auto& service = result.cross_process_channel_services.front();
  EXPECT_EQ(service.payload_identity.payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kHost);
  EXPECT_EQ(service.tensor_max_byte_span, kMaximumTensorBytes);
  EXPECT_FALSE(service.tensor_pool_id.empty());
  EXPECT_NE(service.tensor_pool_id, result.tensor_allocations.front().pool_id);
}
// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, DirectTensorAllocationIsOptInAndOverrideMayOnlyTighten) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "tensor")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload(), 2U)};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  // Key setup phase 1: no tensor_allocation declared → no pool emitted.
  auto result = lower_for_test(snapshot, 9U, profiles);
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.tensor_allocations.empty());

  // Key setup phase 2: operator declares max_rank=4/4096B, graph override tightens to 2/1024B.
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 4U, .max_byte_span = 4096U};
  snapshot.ports.front().graph_override.tensor_allocation =
      TensorPoolBounds{.max_rank = 2U, .max_byte_span = 1024U};
  result = lower_for_test(snapshot, 9U, profiles);
  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  // Key assertion: the tighter graph override wins (max_rank=2, max_byte_span=1024).
  EXPECT_EQ(result.tensor_allocations.front().bounds.max_rank, 2U);
  EXPECT_EQ(result.tensor_allocations.front().bounds.max_byte_span, 1024U);
  EXPECT_EQ(result.tensor_allocations.front().plan_generation, 9U);
  EXPECT_EQ(result.tensor_allocations.front().run_generation, 0U);
  EXPECT_FALSE(result.tensor_allocations.front().device.valid());
  EXPECT_GE(result.tensor_allocations.front().readiness_capacity,
            result.tensor_allocations.front().max_outstanding_loans);
  EXPECT_EQ(result.publication_outcomes.front().pre_submit_authority,
            PreSubmitAuthorityDisposition::kConsumedToRetirement);
  EXPECT_EQ(result.publication_outcomes.front().post_submit_error,
            ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(result.publication_outcomes.front().post_submit_partial_error,
            ErrorCode::kPublicationPartial);

  snapshot.ports.front().graph_override.tensor_allocation =
      TensorPoolBounds{.max_rank = 5U, .max_byte_span = 1024U};
  result = lower_for_test(snapshot, 9U, profiles);
  EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_ALLOCATION_WIDENING"));
  EXPECT_TRUE(result.tensor_allocations.empty());
}

// Compiler and Runtime use this same immutable-record predicate before accepting a direct pool.
// Mutating either capacity form proves the shared check cannot drift between those two owners.
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, DirectTensorAllocationUsesSharedFrozenRecordValidation) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "tensor")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload(), 2U)};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
  };
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 2U, .max_byte_span = 1024U};

  const auto result = lower_for_test(snapshot, 9U, standard_profiles(snapshot.vertices.size()));
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  EXPECT_TRUE(validate_tensor_allocation_record(result.tensor_allocations.front(), 9U));

  TensorAllocationPlanRecord capacity_mismatch = result.tensor_allocations.front();
  ++capacity_mismatch.readiness_capacity;
  EXPECT_FALSE(validate_tensor_allocation_record(capacity_mismatch, 9U));

  TensorAllocationPlanRecord aggregate_mismatch = result.tensor_allocations.front();
  ++aggregate_mismatch.aggregate_byte_capacity;
  EXPECT_FALSE(validate_tensor_allocation_record(aggregate_mismatch, 9U));
}

// A missing effective profile is never an unreported physical-lowering state in the compile
// pipeline: profile admission rejects first. The physical pass must leave the authored edge in the
// Graph view without claiming an accepted edge, direct pool, or pass-through service.
// REQ: COMPILE-003
// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectedTensorProfileRetainsPriorDiagnosticWithoutFallbackService) {
  Graph graph{"rejected-tensor-profile"};
  const auto source = graph.op<ManifestTensorOperator>("source");
  const auto sink = graph.op<ManifestTensorSinkOperator>("sink");
  graph.add_flow(source->tensor, sink->input);
  graph.region("loosening-source").add(source).profile(Profile::kResearch);

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(
      std::any_of(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "PROFILE_REGION_LOOSENS_GRAPH";
      }));
  EXPECT_TRUE(plan.tensor_allocations().empty());
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->payload_edges.empty());
  EXPECT_TRUE(state->process_local_channel_services.empty());
}

// REQ: COMPILE-051
TEST(CompilePhysicalPlan, SubtokenizesTensorContractFailuresWithBoundedEvidence) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "tensor")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload())};
  snapshot.ports.front().setup.tensor_layout =
      TensorRepresentation{.memory_kind = MemoryKind::kHost};
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 0U, .max_byte_span = 64U};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto expect_evidence = [](const PhysicalPlanLoweringResult& lowered,
                                  std::string_view token,
                                  std::string_view json_pointer,
                                  std::string_view provided) {
    const auto found =
        std::find_if(lowered.errors.begin(), lowered.errors.end(), [&](const auto& e) {
          return e.token == token && e.json_pointer == json_pointer && e.provided == provided;
        });
    EXPECT_NE(found, lowered.errors.end()) << token << " " << provided;
  };

  auto result = lower_for_test(snapshot, 9U, profiles);
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_ALLOCATION_BOUNDS_INVALID");
  EXPECT_EQ(result.errors.front().json_pointer, "/ports/1/tensor_allocation");
  EXPECT_EQ(result.errors.front().required, "max_rank>0,max_byte_span>0");
  EXPECT_EQ(result.errors.front().provided, "max_rank=0,max_byte_span=64");

  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{};
  result = lower_for_test(snapshot, 9U, profiles);
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_MEMORY_KIND_UNRESOLVED");
  EXPECT_EQ(result.errors.front().json_pointer, "/ports/1/tensor_layout/memory_kind");
  EXPECT_EQ(result.errors.front().provided, "kUnknown");

  snapshot.ports.front().setup.tensor_layout =
      TensorRepresentation{.memory_kind = MemoryKind::kHost};
  const std::array binding{TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{3}}};
  result = lower_for_test(snapshot, 9U, profiles, {}, {}, {}, nullptr, binding);
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_DEVICE_BINDING_UNEXPECTED");
  EXPECT_EQ(result.errors.front().json_pointer, "/ports/1/tensor_device_binding");
  EXPECT_EQ(result.errors.front().required, "absent for non-device storage");
  EXPECT_EQ(result.errors.front().provided, "device=3");

  const std::array invalid_binding{
      TensorDeviceBindingRecord{.port = PortId{99U}, .device = DeviceId{0}}};
  result = lower_for_test(snapshot, 9U, profiles, {}, {}, {}, nullptr, invalid_binding);
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID");
  EXPECT_EQ(result.errors.front().json_pointer, "/tensor_device_bindings/0");
  EXPECT_EQ(result.errors.front().provided, "port=99,reason=unknown_port");

  const std::array zero_binding{TensorDeviceBindingRecord{.port = PortId{}, .device = DeviceId{0}}};
  result = lower_for_test(snapshot, 9U, profiles, {}, {}, {}, nullptr, zero_binding);
  expect_evidence(result,
                  "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
                  "/tensor_device_bindings/0",
                  "port=0,reason=invalid_port");

  auto non_tensor_binding_snapshot = snapshot;
  non_tensor_binding_snapshot.ports.push_back(output_port(2U, 1U, "scalar", scalar_payload()));
  const std::array non_tensor_binding{
      TensorDeviceBindingRecord{.port = PortId{2U}, .device = DeviceId{0}}};
  result = lower_for_test(
      non_tensor_binding_snapshot, 9U, profiles, {}, {}, {}, nullptr, non_tensor_binding);
  expect_evidence(result,
                  "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
                  "/tensor_device_bindings/0",
                  "port=2,reason=non_tensor_port");

  const std::array invalid_device_binding{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{-1}}};
  result = lower_for_test(snapshot, 9U, profiles, {}, {}, {}, nullptr, invalid_device_binding);
  expect_evidence(result,
                  "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
                  "/tensor_device_bindings/0",
                  "port=1,device=-1,reason=invalid_device");

  auto duplicate_binding_snapshot = snapshot;
  duplicate_binding_snapshot.ports.front().setup.tensor_layout =
      TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice};
  const std::array duplicate_bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{1}},
  };
  result = lower_for_test(
      duplicate_binding_snapshot, 9U, profiles, {}, {}, {}, nullptr, duplicate_bindings);
  expect_evidence(result,
                  "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
                  "/tensor_device_bindings/1",
                  "port=1,device=1,reason=duplicate_port");

  const auto partitions = default_execution_partitions(snapshot);
  result = lower_physical_plan(snapshot,
                               9U,
                               profiles,
                               runtime_package_capability_view(),
                               {},
                               {},
                               {},
                               {},
                               {},
                               partitions,
                               {});
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_PROCESS_GROUP_UNRESOLVED");
  EXPECT_EQ(result.errors.front().json_pointer, "/ports/1/owner_process_group");
  EXPECT_EQ(result.errors.front().provided, "owner=1");

  GraphSnapshotData boundary;
  boundary.generation = GraphGeneration{1U};
  boundary.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic)};
  boundary.ports = {
      output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic)};
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://detached-tensor",
      .payload_identity = tensor_payload(),
  };
  topic.tensor = TensorBoundaryContractRecord{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
      .max_rank = 1U,
      .max_byte_span = 64U,
      .axes = {{.max_extent = 64U}},
      .strides = TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  boundary.external_topics.push_back(std::move(topic));
  const auto boundary_profiles = standard_profiles(boundary.vertices.size());
  result = lower_for_test(boundary, 9U, boundary_profiles);
  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_INTERFACE_TOPOLOGY_INVALID");
  EXPECT_EQ(result.errors.front().json_pointer, "/external_topics/1/edge");
  EXPECT_EQ(result.errors.front().provided, "edge=absent");

  for (const auto& [label, mutate, provided] : std::array{
           std::tuple{"missing memory kind",
                      +[](TensorBoundaryContractRecord& logical) { logical.memory_kind.reset(); },
                      "memory_kind=absent"},
           std::tuple{"malformed logical requirements",
                      +[](TensorBoundaryContractRecord& logical) { logical.max_rank = 0U; },
                      "malformed_logical_requirements"},
           std::tuple{"logical span exceeds bound",
                      +[](TensorBoundaryContractRecord& logical) { logical.max_byte_span = 1U; },
                      "logical_span_exceeds_bound_or_overflows"},
       }) {
    SCOPED_TRACE(label);
    auto invalid_logical = boundary;
    ASSERT_TRUE(invalid_logical.external_topics.front().tensor.has_value());
    mutate(*invalid_logical.external_topics.front().tensor);
    result = lower_for_test(invalid_logical, 9U, boundary_profiles);
    expect_evidence(result,
                    "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID",
                    "/external_topics/1/tensor",
                    provided);
  }

  auto non_operator_adjacent = boundary;
  non_operator_adjacent.vertices.push_back(vertex(2U, "interface", VertexKind::kGraphInterface));
  non_operator_adjacent.ports.push_back(
      input_port(2U, 2U, "interface", tensor_payload(), 1U, PortOwnerKind::kGraphInterface));
  non_operator_adjacent.edges.push_back(edge(1U, 1U, 2U));
  const auto non_operator_profiles = standard_profiles(non_operator_adjacent.vertices.size());
  result = lower_for_test(non_operator_adjacent, 9U, non_operator_profiles);
  expect_evidence(result,
                  "PHYSICAL_TENSOR_INTERFACE_TOPOLOGY_INVALID",
                  "/external_topics/1/adjacent_port",
                  "adjacent_port=non_operator");
}

// Device Tensor fanout with per-edge temporal selectors and a copy policy. Each slot-claim
// reason must enrich an existing contributor without introducing a new independent slot.
//
//   tensor --[DrainAll + window, copy-policy]--> left
//          --[ReuseLast]---------------------> right
//
// Slot budget = 1 producer + 5 delivery + 5 imported-sample retention +
//               5 temporal retention + 1 retirement = 17.
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RecordsOnlyTensorClaimReasonsDerivedFromAcceptedFacts) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "tensor"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload()),
      input_port(3U, 3U, "in", tensor_payload()),
  };
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.edges = {edge(1U, 1U, 2U, 3U), edge(2U, 1U, 3U, 2U)};
  // Key setup: copy policy on edge 1; DrainAll+window on left input; ReuseLast on right input.
  snapshot.edges.front().effective_tensor_payload = PayloadOptionsRecord{
      .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  const std::array selections{
      InputSelectionPlanRecord{
          .input = PortId{2U},
          .selector = InputSelectorKind::kDrainAll,
          .cardinality = InputSelectionCardinality::kSampleBatch,
          .max_selected_samples = 3U,
          .cache_capacity = 3U,
      },
      InputSelectionPlanRecord{
          .input = PortId{3U},
          .selector = InputSelectorKind::kExact,
          .max_selected_samples = 1U,
          .cache_capacity = 2U,
          .missing = TemporalRecordMissingKind::kReuseLast,
      },
  };
  const std::array windows{BracketedWindowPlanRecord{
      .input = PortId{2U},
      .cache_capacity = 3U,
      .retained_overlap_capacity = 2U,
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array device_bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
  };

  const auto result =
      lower_for_test(snapshot, 10U, profiles, selections, windows, {}, nullptr, device_bindings);

  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  // Key assertion: transport queue occupancy and imported-sample retention are separate live
  // sets. CUDA consumer work and DLPack ownership enrich the imported-sample contributor, while
  // window, ReuseLast, and DrainAll enrich temporal retention without adding more slots.
  EXPECT_EQ(result.tensor_allocations.front().slot_count, 17U);
  std::uint32_t contributed_slots = 0U;
  std::array<bool, 17> reasons{};
  bool transport_retention_seen = false;
  for (const auto& contributor : result.pool_capacity_contributors) {
    if (contributor.pool_id == result.tensor_allocations.front().pool_id) {
      contributed_slots += contributor.maximum_distinct_slots;
    }
    if (contributor.contributor_id ==
        result.tensor_allocations.front().pool_id + ".transport-retention") {
      transport_retention_seen = true;
      EXPECT_EQ(contributor.maximum_distinct_slots, 5U);
      ASSERT_FALSE(contributor.reasons.empty());
      EXPECT_NE(std::find(contributor.reasons.begin(),
                          contributor.reasons.end(),
                          TensorSlotClaimReason::kSubscriberBorrowOrImport),
                contributor.reasons.end());
    }
    for (const auto reason : contributor.reasons) {
      reasons[static_cast<std::size_t>(reason)] = true;
    }
  }
  EXPECT_TRUE(transport_retention_seen);
  EXPECT_EQ(contributed_slots, result.tensor_allocations.front().slot_count);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kTemporalCache)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kPendingOrSelectedWindow)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kReuseLast)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kDrainAllActivation)]);
  EXPECT_TRUE(
      reasons[static_cast<std::size_t>(TensorSlotClaimReason::kCopyOrRecorderMaterialization)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kConsumerCudaRelease)]);
  EXPECT_FALSE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kRelayOrReexport)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kDlpackOrFrameworkBorrow)]);
}

//   source[Tensor output, late-joiner history = 3]   (isolated; no edges)
//
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, LateJoinerHistoryAlonePreservesTensorRetentionContributor) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload())};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  // Key setup: history is the only temporal/cache source contributing retained Tensor slots.
  snapshot.ports.front().setup.late_joiner_history_depth = 3U;

  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto result = lower_for_test(snapshot, 11U, profiles);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  const auto& allocation = result.tensor_allocations.front();
  const auto retention =
      std::ranges::find_if(result.pool_capacity_contributors, [&](const auto& contributor) {
        return contributor.contributor_id == allocation.pool_id + ".retention";
      });
  // Key assertion: history-only retention remains represented rather than being dropped because
  // its contributor had no reason.
  ASSERT_NE(retention, result.pool_capacity_contributors.end());
  EXPECT_EQ(retention->maximum_distinct_slots, 3U);
  EXPECT_EQ(retention->reasons,
            (std::vector<TensorSlotClaimReason>{TensorSlotClaimReason::kTemporalCache}));
  EXPECT_TRUE(validate_tensor_allocation_record(allocation, 11U));
}

// A direct-retained Tensor edge owns one admitted sample through the local handoff. The HoloMQ
// fallback instead permits one sample to leave the queue while its imported backing remains
// retained, so the two queue-depth-sized live sets require distinct capacity contributors.
//
//   source --[blocking, depth=2]--> sink
//
// Direct slot budget = 1 producer + 2 delivery + 1 retirement = 4.
// HoloMQ slot budget = 1 producer + 2 delivery + 2 imported retention + 1 retirement = 6.
// REQ: COMPILE-040
// REQ: TENSOR-STORAGE-011
TEST(CompilePhysicalPlan, SeparatesHoloMQTensorQueueAndImportedRetentionCapacity) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload(), 2U),
  };
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.edges = {edge(1U, 1U, 2U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  // Key setup: lower the same allocation once through direct retention and once through the
  // explicit process-local HoloMQ fallback.
  const auto direct = lower_for_test(snapshot, 10U, profiles);
  const auto holomq = lower_for_test(
      snapshot, 10U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);

  ASSERT_TRUE(direct.errors.empty());
  ASSERT_TRUE(holomq.errors.empty());
  ASSERT_EQ(direct.tensor_allocations.size(), 1U);
  ASSERT_EQ(holomq.tensor_allocations.size(), 1U);
  ASSERT_EQ(direct.process_local_channel_services.size(), 1U);
  ASSERT_EQ(holomq.process_local_channel_services.size(), 1U);
  EXPECT_EQ(direct.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_EQ(holomq.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);

  const auto find_transport_retention = [](const PhysicalPlanLoweringResult& result) {
    return std::find_if(result.pool_capacity_contributors.begin(),
                        result.pool_capacity_contributors.end(),
                        [](const auto& contributor) {
                          return contributor.contributor_id.ends_with(".transport-retention");
                        });
  };

  // Key assertion: only HoloMQ adds one queue-depth-sized imported-retention live set.
  EXPECT_EQ(direct.tensor_allocations.front().slot_count, 4U);
  EXPECT_EQ(find_transport_retention(direct), direct.pool_capacity_contributors.end());
  EXPECT_EQ(holomq.tensor_allocations.front().slot_count, 6U);
  const auto imported_retention = find_transport_retention(holomq);
  ASSERT_NE(imported_retention, holomq.pool_capacity_contributors.end());
  EXPECT_EQ(imported_retention->maximum_distinct_slots, 2U);
  ASSERT_EQ(imported_retention->reasons.size(), 2U);
  EXPECT_NE(std::find(imported_retention->reasons.begin(),
                      imported_retention->reasons.end(),
                      TensorSlotClaimReason::kSubscriberBorrowOrImport),
            imported_retention->reasons.end());
  EXPECT_NE(std::find(imported_retention->reasons.begin(),
                      imported_retention->reasons.end(),
                      TensorSlotClaimReason::kDlpackOrFrameworkBorrow),
            imported_retention->reasons.end());
}

// kDropOldest retains the authored newest-N unread backlog on both paths. Direct retention still
// needs a separate typed-retention live set while selected or borrowed samples remain protected.
// REQ: CHAN-115
// REQ: CHAN-118
// REQ: CHAN-128
TEST(CompilePhysicalPlan, DropOldestTensorUsesSplitIngressAndTypedCapacity) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload(), 5U),
  };
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.edges = {edge(1U, 1U, 2U, 5U, QueuePolicy::kDropOldest)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto direct = lower_for_test(snapshot, 10U, profiles);
  const auto holomq = lower_for_test(
      snapshot, 10U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);

  ASSERT_TRUE(direct.errors.empty());
  ASSERT_TRUE(holomq.errors.empty());
  ASSERT_EQ(direct.tensor_allocations.size(), 1U);
  ASSERT_EQ(holomq.tensor_allocations.size(), 1U);
  EXPECT_EQ(direct.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_EQ(holomq.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_EQ(direct.process_local_channel_services.front().queue_policy, QueuePolicy::kDropOldest);
  EXPECT_TRUE(direct.process_local_channel_publishers.front().replacement_attribution_required);

  const auto* direct_allocation = find_allocation(direct, PortId{1U});
  const auto* holomq_allocation = find_allocation(holomq, PortId{1U});
  ASSERT_NE(direct_allocation, nullptr);
  ASSERT_NE(holomq_allocation, nullptr);
  const auto* direct_delivery = find_contributor(direct, direct_allocation->pool_id, ".delivery");
  const auto* direct_transport_retention =
      find_contributor(direct, direct_allocation->pool_id, ".transport-retention");
  const auto* holomq_delivery = find_contributor(holomq, holomq_allocation->pool_id, ".delivery");
  const auto* holomq_transport_retention =
      find_contributor(holomq, holomq_allocation->pool_id, ".transport-retention");

  ASSERT_NE(direct_delivery, nullptr);
  EXPECT_EQ(direct_delivery->maximum_distinct_slots, 5U);
  ASSERT_NE(direct_transport_retention, nullptr);
  EXPECT_EQ(direct_transport_retention->maximum_distinct_slots, 5U);
  constexpr std::array expected_direct_reasons{
      TensorSlotClaimReason::kAcceptedPublication,
      TensorSlotClaimReason::kHolomqTransportOrFanout,
  };
  EXPECT_EQ(direct_delivery->reasons.size(), expected_direct_reasons.size());
  for (const auto reason : expected_direct_reasons) {
    EXPECT_NE(std::find(direct_delivery->reasons.begin(), direct_delivery->reasons.end(), reason),
              direct_delivery->reasons.end());
  }
  constexpr std::array expected_direct_retention_reasons{
      TensorSlotClaimReason::kSubscriberBorrowOrImport,
      TensorSlotClaimReason::kDlpackOrFrameworkBorrow,
  };
  EXPECT_EQ(direct_transport_retention->reasons.size(), expected_direct_retention_reasons.size());
  for (const auto reason : expected_direct_retention_reasons) {
    EXPECT_NE(std::find(direct_transport_retention->reasons.begin(),
                        direct_transport_retention->reasons.end(),
                        reason),
              direct_transport_retention->reasons.end());
  }
  EXPECT_EQ(direct_allocation->slot_count, 12U);

  ASSERT_NE(holomq_delivery, nullptr);
  ASSERT_NE(holomq_transport_retention, nullptr);
  EXPECT_EQ(holomq_delivery->maximum_distinct_slots, 5U);
  EXPECT_EQ(holomq_transport_retention->maximum_distinct_slots, 5U);
  ASSERT_FALSE(holomq_transport_retention->reasons.empty());
  constexpr std::array expected_transport_reasons{
      TensorSlotClaimReason::kSubscriberBorrowOrImport,
      TensorSlotClaimReason::kDlpackOrFrameworkBorrow,
  };
  EXPECT_EQ(holomq_transport_retention->reasons.size(), expected_transport_reasons.size());
  for (const auto reason : expected_transport_reasons) {
    EXPECT_NE(std::find(holomq_transport_retention->reasons.begin(),
                        holomq_transport_retention->reasons.end(),
                        reason),
              holomq_transport_retention->reasons.end());
  }
  EXPECT_EQ(holomq_allocation->slot_count, 12U);
}

// REQ: COMPILE-002
TEST(CompilePhysicalPlan, CanonicalizesUnconsumedTensorDeviceBindingDiagnostics) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "first"), vertex(2U, "second")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      output_port(2U, 2U, "out", tensor_payload()),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{2U}, .device = DeviceId{0}},
  };

  const auto forward = lower_for_test(snapshot, 10U, profiles, {}, {}, {}, nullptr, bindings);
  std::reverse(bindings.begin(), bindings.end());
  const auto reverse = lower_for_test(snapshot, 10U, profiles, {}, {}, {}, nullptr, bindings);

  ASSERT_EQ(forward.errors.size(), 2U);
  ASSERT_EQ(reverse.errors.size(), forward.errors.size());
  for (std::size_t index = 0; index < forward.errors.size(); ++index) {
    EXPECT_EQ(forward.errors[index].token, "PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED");
    EXPECT_EQ(forward.errors[index].token, reverse.errors[index].token);
    EXPECT_EQ(forward.errors[index].source.graph_generation,
              reverse.errors[index].source.graph_generation);
    EXPECT_EQ(forward.errors[index].source.vertex, reverse.errors[index].source.vertex);
    EXPECT_EQ(forward.errors[index].source.port, reverse.errors[index].source.port);
    EXPECT_EQ(forward.errors[index].json_pointer, reverse.errors[index].json_pointer);
    EXPECT_EQ(forward.errors[index].required, "binding consumed by one admitted Tensor resource");
    EXPECT_EQ(forward.errors[index].provided, "device=0");
  }
  EXPECT_EQ(forward.errors[0].source.port, PortId{1U});
  EXPECT_EQ(forward.errors[1].source.port, PortId{2U});
}

// An imported scalar topic flows through a boundary vertex into a local sink. The catalog
// endpoint attributes (loaned samples, queue depth, history) must be symmetric across
// the compiled service, publisher, and subscriber records.
//
//   [holomq://frames] --> boundary --[drop-oldest, depth=5]--> sink
//
// REQ: COMPILE-009
// REQ: COMPILE-040
// REQ: CHAN-069
TEST(CompilePhysicalPlan, ExternalServiceAndEndpointFactsAreSymmetricAndPublisherHistoryIsZero) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "topic", scalar_payload(), 1U, PortOwnerKind::kImportedTopic),
      input_port(2U, 2U, "in", scalar_payload(), 5U),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 5U, QueuePolicy::kDropOldest)};
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://frames",
      .payload_identity = scalar_payload(),
  };
  topic.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 3U};
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 15U, profiles);

  EXPECT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  // Key assertion: publisher and subscriber share one service_id; publisher_history_depth==0.
  ASSERT_EQ(result.channel_services.size(), 1U);
  ASSERT_EQ(result.channel_publishers.size(), 1U);
  ASSERT_EQ(result.channel_subscribers.size(), 1U);
  EXPECT_EQ(result.channel_services.front().max_subscriber_queue_depth, 5U);
  EXPECT_EQ(result.channel_services.front().max_subscriber_borrowed_samples, 5U);
  EXPECT_EQ(result.channel_services.front().publisher_history_depth, 0U);
  EXPECT_EQ(result.channel_publishers.front().max_loaned_samples, 3U);
  EXPECT_EQ(result.channel_publishers.front().service_id,
            result.channel_subscribers.front().service_id);
  EXPECT_EQ(result.channel_services.front().service_id, "catalog-service-1");
  EXPECT_TRUE(result.channel_publishers.front().replacement_attribution_required);
  ASSERT_EQ(result.channel_locality.size(), 1U);
  EXPECT_EQ(result.channel_locality.front().accepted, ChannelLocalityKind::kExternalImport);
  ASSERT_EQ(result.imported_generation_capacities.size(), 1U);
  const auto& generations = result.imported_generation_capacities.front();
  EXPECT_EQ(generations.topic, ExternalTopicId{1U});
  EXPECT_EQ(generations.endpoint_or_service_id, "catalog-service-1");
  EXPECT_EQ(generations.endpoint_max_generations, 1U);
  EXPECT_EQ(generations.service_max_generations, 1U);
  EXPECT_EQ(generations.runtime_max_generations, 1U);
  EXPECT_EQ(generations.candidate_reservation_capacity, 0U);
  EXPECT_FALSE(generations.restart_capable_minimum_two);
  EXPECT_TRUE(generations.no_wrap_aba_generation);
  EXPECT_TRUE(generations.fallback_rejected);
}

// All service families project one shared immutable contract. This locks the compiler's external,
// process-local, and cross-process records to the Runtime adaptation used to construct endpoints.
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, TensorServiceRecordFamiliesShareOneContractProjection) {
  const auto contract = make_tensor_service_contract(tensor_payload(),
                                                     QueuePolicy::kDropOldest,
                                                     7U,
                                                     5U,
                                                     true,
                                                     TensorServiceStorageContract{
                                                         .max_byte_span = 4096U,
                                                         .pool_id = "pool",
                                                         .memory_kind = MemoryKind::kCudaDevice,
                                                         .device = DeviceId{2},
                                                     },
                                                     InterfaceTensorEmitPolicy::kCopyIfNeeded);
  const auto external = make_channel_service_record(ExternalTopicId{1U}, "external", contract);
  const auto local = make_process_local_service_record(PortId{2U}, "local", contract, 3U, false);
  const auto cross = make_cross_process_service_record(EdgeId{4U},
                                                       PortId{2U},
                                                       PortId{3U},
                                                       "cross",
                                                       contract,
                                                       1U,
                                                       ProcessGroupId{5U},
                                                       ProcessGroupId{6U});
  const auto adapted = make_process_local_service_record(cross.source,
                                                         cross.service_id,
                                                         make_tensor_service_contract(cross),
                                                         cross.max_subscribers,
                                                         true);

  EXPECT_EQ(external.payload_identity.kind, PayloadIdentityKind::kTensor);
  EXPECT_EQ(external.queue_policy, contract.queue_policy);
  EXPECT_EQ(external.max_subscriber_queue_depth, contract.max_subscriber_queue_depth);
  EXPECT_EQ(external.max_subscriber_borrowed_samples, contract.max_subscriber_borrowed_samples);
  EXPECT_EQ(external.tensor_emit_policy, InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_TRUE(external.tensor_payload);

  EXPECT_EQ(local.payload_identity.kind, cross.payload_identity.kind);
  EXPECT_EQ(local.queue_policy, cross.queue_policy);
  EXPECT_EQ(local.tensor_max_byte_span, cross.tensor_max_byte_span);
  EXPECT_EQ(local.tensor_pool_id, cross.tensor_pool_id);
  EXPECT_EQ(local.tensor_memory_kind, cross.tensor_memory_kind);
  EXPECT_EQ(local.tensor_device, cross.tensor_device);
  EXPECT_EQ(local.max_subscriber_queue_depth, cross.max_subscriber_queue_depth);
  EXPECT_EQ(local.max_subscriber_borrowed_samples, cross.max_subscriber_borrowed_samples);
  EXPECT_EQ(local.publisher_history_depth, cross.publisher_history_depth);
  EXPECT_EQ(local.event_wake, cross.event_wake);

  EXPECT_TRUE(adapted.cross_process_transport);
  EXPECT_EQ(adapted.payload_identity.kind, cross.payload_identity.kind);
  EXPECT_EQ(adapted.tensor_max_byte_span, cross.tensor_max_byte_span);
  EXPECT_EQ(adapted.tensor_pool_id, cross.tensor_pool_id);
  EXPECT_EQ(adapted.tensor_memory_kind, cross.tensor_memory_kind);
  EXPECT_EQ(adapted.tensor_device, cross.tensor_device);
  EXPECT_EQ(adapted.queue_policy, cross.queue_policy);
  EXPECT_EQ(adapted.max_subscribers, cross.max_subscribers);
  EXPECT_EQ(adapted.max_subscriber_queue_depth, cross.max_subscriber_queue_depth);
  EXPECT_EQ(adapted.max_subscriber_borrowed_samples, cross.max_subscriber_borrowed_samples);
  EXPECT_EQ(adapted.publisher_history_depth, cross.publisher_history_depth);
  EXPECT_EQ(adapted.event_wake, cross.event_wake);
}
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, DerivesRestartGenerationAndInlineRealizationFromFrozenFacts) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
      input_port(2U, 2U, "in", tensor_payload(), 2U),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 2U)};
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://inline",
      .payload_identity = tensor_payload(),
  };
  topic.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  topic.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = std::chrono::milliseconds{25},
  };
  topic.tensor = TensorBoundaryContractRecord{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
      .max_rank = 1U,
      .max_byte_span = 4096U,
      .axes = {{.max_extent = 4096U}},
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
      .max_inline_bytes = 256U,
      .max_staging_byte_span = 1024U,
  };
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array lifecycle_targets{LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 2U, .plan_generation = 80U, .run_generation = 0U},
      .effective_profile = Profile::kStandard,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  }};

  const auto result = lower_for_test(snapshot, 80U, profiles, {}, {}, lifecycle_targets);

  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.imported_generation_capacities.size(), 1U);
  const auto& generations = result.imported_generation_capacities.front();
  EXPECT_EQ(generations.endpoint_max_generations, 2U);
  EXPECT_EQ(generations.service_max_generations, 2U);
  EXPECT_EQ(generations.runtime_max_generations, 2U);
  EXPECT_EQ(generations.candidate_reservation_capacity, 1U);
  EXPECT_EQ(generations.release_record_capacity, 2U);
  EXPECT_EQ(generations.quarantine_record_capacity, 2U);
  EXPECT_EQ(generations.effective_profile, Profile::kStandard);
  EXPECT_EQ(generations.peer_requirement, PeerRequirement::kRequired);
  EXPECT_EQ(generations.attach_max_wait_ns, 25'000'000U);
  EXPECT_TRUE(generations.restart_capable_minimum_two);
  const auto inline_host =
      std::find_if(result.tensor_transfer_plans.begin(),
                   result.tensor_transfer_plans.end(),
                   [](const auto& transfer) {
                     return transfer.source_class == TensorSourceClass::kOrdinaryHostContiguous;
                   });
  ASSERT_NE(inline_host, result.tensor_transfer_plans.end());
  ASSERT_TRUE(inline_host->realization.has_value());
  EXPECT_EQ(*inline_host->realization, TensorTransferRealization::kInline);
  EXPECT_EQ(inline_host->min_logical_bytes, 0U);
  EXPECT_EQ(inline_host->max_logical_bytes, 256U);
  EXPECT_TRUE(inline_host->source_must_be_host_ready);
  EXPECT_TRUE(inline_host->requires_inline_record);
  EXPECT_FALSE(inline_host->requires_exportable_copy_pool);
  const auto oversized_host =
      std::find_if(result.tensor_transfer_plans.begin(),
                   result.tensor_transfer_plans.end(),
                   [](const auto& cell) {
                     return cell.source_class == TensorSourceClass::kOrdinaryHostContiguous &&
                            cell.min_logical_bytes == 257U;
                   });
  ASSERT_NE(oversized_host, result.tensor_transfer_plans.end());
  ASSERT_TRUE(oversized_host->realization.has_value());
  EXPECT_EQ(*oversized_host->realization, TensorTransferRealization::kCopiedToExportablePool);
  EXPECT_EQ(oversized_host->max_logical_bytes, 1024U);
  const auto staging_exceeded =
      std::find_if(result.tensor_transfer_plans.begin(),
                   result.tensor_transfer_plans.end(),
                   [](const auto& cell) {
                     return cell.source_class == TensorSourceClass::kOrdinaryHostContiguous &&
                            cell.min_logical_bytes == 1025U;
                   });
  ASSERT_NE(staging_exceeded, result.tensor_transfer_plans.end());
  EXPECT_EQ(staging_exceeded->disposition, PlanDisposition::kRejected);
  EXPECT_FALSE(staging_exceeded->realization);
  EXPECT_EQ(staging_exceeded->max_logical_bytes, 4096U);
  const auto exportable = std::find_if(
      result.tensor_transfer_plans.begin(),
      result.tensor_transfer_plans.end(),
      [](const auto& cell) { return cell.source_class == TensorSourceClass::kAlreadyExportable; });
  ASSERT_NE(exportable, result.tensor_transfer_plans.end());
  ASSERT_TRUE(exportable->realization);
  EXPECT_EQ(*exportable->realization, TensorTransferRealization::kZeroCopyExportable);
  EXPECT_EQ(exportable->max_logical_bytes, 4096U);
  ASSERT_EQ(result.tensor_inline_records.size(), 1U);
  const auto& inline_record = result.tensor_inline_records.front();
  EXPECT_EQ(inline_record.service_id, "catalog-service-1");
  EXPECT_EQ(inline_record.capability_id, kTensorInlineRecordCapabilityId);
  EXPECT_EQ(inline_record.capability_version, kTensorInlineRecordCapabilityVersion);
  EXPECT_EQ(inline_record.record_format_id, kTensorRecordFormatId);
  EXPECT_EQ(inline_record.record_format_version, kTensorRecordFormatVersion);
  EXPECT_EQ(inline_record.inline_discriminator, kTensorInlineDiscriminator);
  EXPECT_EQ(inline_record.opaque_descriptor_discriminator, kTensorOpaqueDescriptorDiscriminator);
  EXPECT_EQ(inline_record.fixed_metadata_overhead_bytes, 512U);
  EXPECT_EQ(inline_record.provider_max_inline_logical_bytes, 65'536U);
  EXPECT_EQ(inline_record.profile_max_inline_logical_bytes, 65'536U);
  EXPECT_EQ(inline_record.effective_max_inline_logical_bytes, 65'536U);
  EXPECT_EQ(inline_record.accepted_max_inline_logical_bytes, 256U);
  EXPECT_EQ(inline_record.fixed_record_bucket_bytes, 768U);
  EXPECT_EQ(inline_record.inline_arm_schema_id, kTensorInlineArmSchemaId);
  EXPECT_EQ(inline_record.opaque_descriptor_arm_schema_id, kTensorOpaqueDescriptorArmSchemaId);
  EXPECT_TRUE(inline_record.inline_arm_materialized);
  EXPECT_TRUE(inline_record.descriptor_arm_materialized);
  EXPECT_TRUE(inline_record.fallback_rejected);
  EXPECT_EQ(inline_record.disposition, PlanDisposition::kAccepted);
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsInlineProviderCapCapabilityAndRecordSchemaMismatch) {
  const auto make_snapshot = [](std::uint32_t max_inline_bytes) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic), vertex(2U, "sink")};
    snapshot.ports = {
        output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
        input_port(2U, 2U, "in", tensor_payload(), 1U),
    };
    snapshot.edges = {edge(1U, 1U, 2U)};
    ExternalTopicRecord topic{
        .id = ExternalTopicId{1U},
        .vertex = VertexId{1U},
        .graph_port = GraphPortId{1U},
        .imported = true,
        .uri = "holomq://inline-provider",
        .payload_identity = tensor_payload(),
    };
    topic.tensor = TensorBoundaryContractRecord{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
        .max_rank = 1U,
        .max_byte_span = max_inline_bytes,
        .axes = {{.max_extent = max_inline_bytes}},
        .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
        .max_inline_bytes = max_inline_bytes,
    };
    snapshot.external_topics.push_back(std::move(topic));
    return snapshot;
  };
  const auto expect_provider_rejection = [&](const GraphSnapshotData& snapshot,
                                             RuntimePackageCapabilityManifest package) {
    ASSERT_TRUE(seal_runtime_package_capability_manifest(package));
    const auto view = validate_runtime_package_capability_manifest(package);
    ASSERT_TRUE(view);
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto result = lower_for_test(snapshot, 81U, profiles, {}, {}, {}, &view.value());
    EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED"));
    EXPECT_TRUE(result.tensor_inline_records.empty());
    EXPECT_TRUE(result.tensor_transfer_plans.empty());
  };

  expect_provider_rejection(make_snapshot(65'537U), runtime_package_capability_manifest());

  auto missing = runtime_package_capability_manifest();
  missing.tensor_inline_record.reset();
  expect_provider_rejection(make_snapshot(256U), std::move(missing));

  auto wrong_version = runtime_package_capability_manifest();
  ++wrong_version.tensor_inline_record->record_format_version;
  expect_provider_rejection(make_snapshot(256U), std::move(wrong_version));

  auto wrong_discriminator = runtime_package_capability_manifest();
  wrong_discriminator.tensor_inline_record->inline_discriminator = "unknown_inline_arm";
  expect_provider_rejection(make_snapshot(256U), std::move(wrong_discriminator));

  auto wrong_capability = runtime_package_capability_manifest();
  const std::string prior_id = wrong_capability.tensor_inline_record->capability_id;
  wrong_capability.tensor_inline_record->capability_id = "holoscan.runtime.other_inline_record";
  const auto offer = std::find_if(wrong_capability.offers.begin(),
                                  wrong_capability.offers.end(),
                                  [&](const auto& item) { return item.id == prior_id; });
  ASSERT_NE(offer, wrong_capability.offers.end());
  offer->id = wrong_capability.tensor_inline_record->capability_id;
  expect_provider_rejection(make_snapshot(256U), std::move(wrong_capability));
}

// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsStridedGeometryOverflowAndDistinctSlotOverflow) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic),
                       vertex(2U, "sink"),
                       vertex(3U, "producer")};
  snapshot.ports = {
      output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
      input_port(2U, 2U, "in", tensor_payload(), 1U),
      output_port(3U, 3U, "out", tensor_payload(), std::numeric_limits<std::uint32_t>::max()),
  };
  snapshot.edges = {edge(1U, 1U, 2U)};
  snapshot.ports[2].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
  snapshot.ports[2].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://overflow",
      .payload_identity = tensor_payload(),
  };
  topic.tensor = TensorBoundaryContractRecord{
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .max_rank = 1U,
      .max_byte_span = 64U,
      .axes = {{.max_extent = std::numeric_limits<std::uint64_t>::max(),
                .max_abs_stride_bytes = std::numeric_limits<std::uint64_t>::max()}},
      .strides = TensorStridePolicy::kBoundedStrided,
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 25U, profiles);

  EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID"));
  EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_CAPACITY_OVERFLOW"));
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsCheckedAggregateTensorByteCapacityOverflow) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "tensor")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload(), 2U)};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports.front().setup.tensor_allocation = TensorPoolBounds{
      .max_rank = 1U,
      .max_byte_span = std::numeric_limits<std::size_t>::max(),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 26U, profiles);

  EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_CAPACITY_OVERFLOW"));
  EXPECT_TRUE(result.tensor_allocations.empty());
}

// REQ: COMPILE-005
// REQ: COMPILE-009
TEST(CompilePhysicalPlan, PreservesUnresolvedProfileWithoutFabricatingStandardCapabilities) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "invalid-profile")};
  snapshot.ports = {output_port(1U, 1U, "out", scalar_payload())};
  const std::array<std::optional<Profile>, 1> unresolved_profiles{std::nullopt};

  auto result = lower_for_test(snapshot, 30U, unresolved_profiles);

  ASSERT_EQ(result.ports.size(), 1U);
  EXPECT_FALSE(result.ports.front().effective_profile.has_value());
  EXPECT_TRUE(result.generic_output_loans.empty());
  EXPECT_TRUE(result.publication_outcomes.empty());

  auto state = published_physical_plan_state(std::move(result), 30U);
  const auto finalized = finalize_plan_artifact(state);
  ASSERT_TRUE(finalized) << finalized.error().message;
  EXPECT_NE(state.json.find("\"effective_profile\":null"), std::string::npos);
  EXPECT_EQ(state.json.find("\"effective_profile\":\"standard\""), std::string::npos);
}

// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsDeviceVisiblePoolsUntilCompileHasDeviceSelectionEvidence) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "tensor")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload())};
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 2U, .max_byte_span = 4096U};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  for (const MemoryKind memory : {MemoryKind::kCudaDevice, MemoryKind::kCudaManaged}) {
    snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
        .memory_kind = memory,
        .dtype = DLDataType{kDLFloat, 32U, 1U},
        .rank = 2U,
    };
    const auto result = lower_for_test(snapshot, 31U, profiles);

    EXPECT_TRUE(has_error(result, "IMPLICIT_DEVICE_ZERO_REJECTED"));
    EXPECT_TRUE(result.tensor_allocations.empty());
    EXPECT_TRUE(result.pool_capacity_contributors.empty());
  }
}

// One device binding (DeviceId=0) covers both a direct allocation pool and a boundary
// interface pool; both must report DeviceId=0 and share a single TensorDeviceResource.
//
//   direct (port 1, device=0)              --> direct-sink
//   [holomq://device-boundary] (port 2, device=0) --> boundary-sink
//
// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(CompilePhysicalPlan,
     ExplicitDeviceBindingsRealizeDirectAndInterfacePoolsWithOneRunDeviceStream) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {
      vertex(1U, "direct"),
      vertex(2U, "boundary", VertexKind::kImportedTopic),
      vertex(3U, "direct-sink"),
      vertex(4U, "boundary-sink"),
  };
  snapshot.ports = {
      output_port(1U, 1U, "direct", tensor_payload(), 2U),
      output_port(2U, 2U, "boundary", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
      input_port(3U, 3U, "direct-sink", tensor_payload(), 2U),
      input_port(4U, 4U, "boundary-sink", tensor_payload(), 2U),
  };
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 2U, .max_byte_span = 4096U};
  snapshot.ports[0].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
  };
  snapshot.edges = {edge(1U, 1U, 3U, 2U), edge(2U, 2U, 4U, 2U)};
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{2U},
      .graph_port = GraphPortId{2U},
      .imported = true,
      .uri = "holomq://device-boundary",
      .payload_identity = tensor_payload(),
  };
  topic.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  topic.tensor = TensorBoundaryContractRecord{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
      .max_rank = 2U,
      .max_byte_span = 4096U,
      .axes = {{.max_extent = 16U}, {.max_extent = 16U}},
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{2U}, .device = DeviceId{0}},
  };

  auto result = lower_for_test(snapshot, 32U, profiles, {}, {}, {}, nullptr, bindings);

  ASSERT_TRUE(result.errors.empty());
  // Key assertion: one direct pool and one interface pool, both on DeviceId=0, one service.
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  ASSERT_EQ(result.tensor_interface_pools.size(), 1U);
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_size_bytes, 0U);
  EXPECT_EQ(service.tensor_max_byte_span, 4096U);
  EXPECT_EQ(service.tensor_pool_id, result.tensor_allocations.front().pool_id);
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(service.tensor_device, DeviceId{0});
  EXPECT_EQ(result.tensor_allocations.front().device, DeviceId{0});
  EXPECT_EQ(result.tensor_interface_pools.front().device, DeviceId{0});
  const auto direct_delivery =
      std::find_if(result.pool_capacity_contributors.begin(),
                   result.pool_capacity_contributors.end(),
                   [&](const auto& contributor) {
                     return contributor.pool_id == result.tensor_allocations.front().pool_id &&
                            contributor.contributor_id ==
                                result.tensor_allocations.front().pool_id + ".delivery";
                   });
  ASSERT_NE(direct_delivery, result.pool_capacity_contributors.end());
  EXPECT_NE(std::find(direct_delivery->reasons.begin(),
                      direct_delivery->reasons.end(),
                      TensorSlotClaimReason::kConsumerCudaRelease),
            direct_delivery->reasons.end());
  EXPECT_NE(std::find(direct_delivery->reasons.begin(),
                      direct_delivery->reasons.end(),
                      TensorSlotClaimReason::kDlpackOrFrameworkBorrow),
            direct_delivery->reasons.end());
  std::uint32_t direct_capacity = 0U;
  for (const auto& contributor : result.pool_capacity_contributors) {
    if (contributor.pool_id == result.tensor_allocations.front().pool_id) {
      direct_capacity += contributor.maximum_distinct_slots;
    }
  }
  EXPECT_EQ(direct_capacity, result.tensor_allocations.front().slot_count);
  ASSERT_EQ(result.tensor_device_resources.size(), 1U);
  const auto& resource = result.tensor_device_resources.front();
  EXPECT_EQ(resource.plan_generation, 32U);
  EXPECT_EQ(resource.run_generation, 0U);
  EXPECT_EQ(resource.process_group, ProcessGroupId{1U});
  EXPECT_EQ(resource.device, DeviceId{0});
  EXPECT_EQ(resource.pool_count, 2U);
  EXPECT_EQ(resource.slot_capacity,
            result.tensor_allocations.front().slot_count +
                result.tensor_interface_pools.front().slot_count);
  EXPECT_EQ(resource.loan_credit_capacity,
            result.tensor_allocations.front().max_outstanding_loans +
                result.tensor_interface_pools.front().publisher_loan_capacity);
  EXPECT_EQ(resource.readiness_capacity,
            result.tensor_allocations.front().readiness_capacity +
                result.tensor_interface_pools.front().readiness_capacity);
  EXPECT_EQ(resource.release_capacity,
            result.tensor_allocations.front().release_capacity +
                result.tensor_interface_pools.front().release_capacity);
  EXPECT_EQ(resource.materialize_before, LifecycleStage::kArm);
  EXPECT_FALSE(resource.gpu_resident_backend_selected);
  EXPECT_TRUE(resource.fallback_rejected);
  ASSERT_EQ(result.cuda_compute_streams.size(), 1U);
  const auto& stream = result.cuda_compute_streams.front();
  EXPECT_EQ(stream.plan_generation, resource.plan_generation);
  EXPECT_EQ(stream.process_group, resource.process_group);
  EXPECT_EQ(stream.device, resource.device);
  EXPECT_EQ(stream.owner_id, resource.allocator_root_id);
  EXPECT_FALSE(stream.partition);
  EXPECT_EQ(stream.stream_id, "tensor-stream.32.process.1.device.0");
  EXPECT_FALSE(stream.green_context_id);
  ASSERT_EQ(stream.fused_linear_edges.size(), 1U);
  EXPECT_EQ(stream.fused_linear_edges.front(), EdgeId{1U});
  EXPECT_EQ(stream.materialize_before, LifecycleStage::kArm);

  auto state = published_physical_plan_state(std::move(result), 32U);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
}

// Expected fusion assignment (all three linear Operators share S0):
//
//   linear-source --S0--> linear-middle --S0--> linear-sink
//
//   fanout-source -----> fanout-left
//                 `----> fanout-right
//
// The fanout Operators currently resolve to the same process/device stream resource, but neither
// fanout edge receives linear-chain fusion evidence.
// REQ: COMPILE-049
// REQ: COMPILE-065
TEST(CompilePhysicalPlan, AssignsOnlyStrictLinearTensorChainsToTheSharedDeviceStream) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {
      vertex(1U, "linear-source"),
      vertex(2U, "linear-middle"),
      vertex(3U, "linear-sink"),
      vertex(4U, "fanout-source"),
      vertex(5U, "fanout-left"),
      vertex(6U, "fanout-right"),
  };
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload()),
      output_port(3U, 2U, "out", tensor_payload()),
      input_port(4U, 3U, "in", tensor_payload()),
      output_port(5U, 4U, "out", tensor_payload()),
      input_port(6U, 5U, "in", tensor_payload()),
      input_port(7U, 6U, "in", tensor_payload()),
  };
  for (const std::size_t port_index : {0U, 2U, 4U}) {
    snapshot.ports[port_index].setup.tensor_allocation =
        TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
    snapshot.ports[port_index].setup.tensor_layout = TensorRepresentation{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
  }
  snapshot.edges = {
      edge(1U, 1U, 2U),
      edge(2U, 3U, 4U),
      edge(3U, 5U, 6U),
      edge(4U, 5U, 7U),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{3U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{5U}, .device = DeviceId{0}},
  };

  const auto result = lower_for_test(snapshot, 36U, profiles, {}, {}, {}, nullptr, bindings);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.tensor_device_resources.size(), 1U);
  ASSERT_EQ(result.cuda_compute_streams.size(), 1U);
  const std::vector expected{EdgeId{1U}, EdgeId{2U}};
  EXPECT_EQ(result.cuda_compute_streams.front().fused_linear_edges, expected);
}

// The Operator has one process-local Tensor successor, but its catalog export makes total fanout
// two. The local edge must not receive fusion authority that Runtime would reject at
// materialization. Its pinned direct pool also needs explicit synchronization-device evidence
// even though the external catalog destination is host-pageable.
//
//   source -----> local-sink
//          `----> catalog-export
// REQ: COMPILE-049
// REQ: COMPILE-065
TEST(CompilePhysicalPlan, CatalogFanoutDoesNotReceiveLinearStreamAuthority) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {
      vertex(1U, "source"),
      vertex(2U, "local-sink"),
      vertex(3U, "catalog-export", VertexKind::kExportedTopic),
  };
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload()),
      input_port(3U, 3U, "topic", tensor_payload(), 1U, PortOwnerKind::kExportedTopic),
  };
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.ports[0].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kPinnedHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{3U},
      .graph_port = GraphPortId{3U},
      .uri = "holomq://mixed-device-export",
      .payload_identity = tensor_payload(),
  };
  topic.bounds.subscriber = SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  topic.tensor = TensorBoundaryContractRecord{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
      .max_rank = 1U,
      .max_byte_span = 1024U,
      .axes = {{.max_extent = 1024U, .max_abs_stride_bytes = 1U}},
      .strides = TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
  };

  const auto unbound = lower_for_test(snapshot, 37U, profiles);
  EXPECT_TRUE(std::any_of(unbound.errors.begin(), unbound.errors.end(), [](const auto& error) {
    return error.token == "IMPLICIT_DEVICE_ZERO_REJECTED";
  }));
  EXPECT_TRUE(unbound.tensor_allocations.empty());

  const auto result = lower_for_test(snapshot, 37U, profiles, {}, {}, {}, nullptr, bindings);

  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_EQ(result.payload_edges.size(), 2U);
  EXPECT_EQ(result.payload_edges[0].fanout, 2U);
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  EXPECT_EQ(result.tensor_allocations.front().device, DeviceId{0});
  ASSERT_EQ(result.tensor_device_resources.size(), 1U);
  ASSERT_EQ(result.cuda_compute_streams.size(), 1U);
  EXPECT_TRUE(result.cuda_compute_streams.front().fused_linear_edges.empty());
}

// The middle Operator would need both S0/device 0 and S1/device 1, so neither edge acquires
// same-stream fusion authority:
//
//   source --S0/device0--> middle --S1/device1--> sink
// REQ: COMPILE-049
// REQ: COMPILE-065
// REQ: COMPILE-052
TEST(CompilePhysicalPlan, RejectsOperatorsWithAmbiguousDeviceObligations) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "middle"), vertex(3U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload()),
      output_port(3U, 2U, "out", tensor_payload()),
      input_port(4U, 3U, "in", tensor_payload()),
  };
  for (const std::size_t port_index : {0U, 2U}) {
    snapshot.ports[port_index].setup.tensor_allocation =
        TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
    snapshot.ports[port_index].setup.tensor_layout = TensorRepresentation{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
  }
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{3U}, .device = DeviceId{1}},
  };

  const auto result = lower_for_test(snapshot, 37U, profiles, {}, {}, {}, nullptr, bindings);

  ASSERT_EQ(result.errors.size(), 1U);
  EXPECT_EQ(result.errors.front().token, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID");
  EXPECT_NE(result.errors.front().provided.find("reason=operator_device_conflict"),
            std::string::npos);
  ASSERT_EQ(result.tensor_device_resources.size(), 2U);
  ASSERT_EQ(result.cuda_compute_streams.size(), 2U);
  EXPECT_TRUE(result.cuda_compute_streams[0].fused_linear_edges.empty());
  EXPECT_TRUE(result.cuda_compute_streams[1].fused_linear_edges.empty());
}

// REQ: COMPILE-052
TEST(CompilePhysicalPlan, PublicTensorOutputPlacementResolvesStableNamesAndOwnsStrings) {
  Graph graph{"public-tensor-placement"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DeviceTensorOperator>("source");
  const auto sink = graph.op<ManifestTensorSinkOperator>("sink");
  graph.add_flow(source->tensor, sink->input);

  std::string operator_path{"source"};
  std::string output_port{"tensor"};
  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = operator_path,
      .output_port = output_port,
      .device = DeviceId{1},
  });
  operator_path = "changed";
  output_port = "changed";

  const auto plan = compile(graph, std::move(options));

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().device, DeviceId{1});
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);
  EXPECT_EQ(plan.tensor_device_resources().front().device, DeviceId{1});
}

// REQ: COMPILE-019
// REQ: COMPILE-052
// REQ: COMPILE-068
TEST(CompilePhysicalPlan, CanonicalOperatorPathJoinsParameterPartitionAndTensorPlacement) {
  Graph graph{"canonical-operator-path-surfaces"};
  const auto first = graph.embed<CanonicalOperatorPathTensorGraph>("a", "b/c");
  const auto second = graph.embed<CanonicalOperatorPathTensorGraph>("a/b", "c");
  const ClockKey root_clock = test::add_compile_test_default_clock(graph);
  graph.bind_clock(first, first->clock_import(), root_clock);
  graph.bind_clock(second, second->clock_import(), root_clock);

  constexpr std::string_view kFirst{"a/b%2Fc"};
  constexpr std::string_view kSecond{"a%2Fb/c"};
  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = kFirst, .output_port = "tensor", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = kSecond, .output_port = "tensor", .device = DeviceId{1}});

  const ExecutionPlan plan = compile(graph, std::move(options));

  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto first_partition = plan.partitions().lookup_for_path(kFirst);
  const auto second_partition = plan.partitions().lookup_for_path(kSecond);
  ASSERT_TRUE(first_partition);
  ASSERT_TRUE(second_partition);
  EXPECT_NE(first_partition->id(), second_partition->id());
  EXPECT_EQ(plan.partitions().lookup_for_path("a/b/c").error().code, ErrorCode::kNotFound);
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);
  const std::array expected_placements{
      std::pair{kFirst, DeviceId{0}},
      std::pair{kSecond, DeviceId{1}},
  };
  for (const auto& [path, device] : expected_placements) {
    const auto parameter = std::ranges::find(
        plan.operator_parameters(), path, &OperatorParameterPlanRecord::operator_path);
    ASSERT_NE(parameter, plan.operator_parameters().end());
    const auto output = std::ranges::find_if(plan.ports(), [&](const PortPlanRecord& port) {
      return port.owner_kind == PortOwnerKind::kOperator &&
             port.owner.value == parameter->owner.id && port.direction == PortDirection::kOutput &&
             port.local_name == "tensor";
    });
    ASSERT_NE(output, plan.ports().end());
    const auto allocation = std::ranges::find(
        plan.tensor_allocations(), output->id, &TensorAllocationPlanRecord::output);
    ASSERT_NE(allocation, plan.tensor_allocations().end());
    EXPECT_EQ(allocation->device, device);
  }
}

// A CUDA-device output requires explicit deployment placement. The rejected plan must identify
// the authored output and the missing binding rather than reporting only a broad plan failure.
// REQ: COMPILE-052
TEST(CompilePhysicalPlan, MissingPublicTensorOutputDeviceBindingNamesAffectedOutput) {
  Graph graph{"missing-public-tensor-placement"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DeviceTensorOperator>("source");
  const auto sink = graph.op<ManifestTensorSinkOperator>("sink");
  graph.add_flow(source->tensor, sink->input);

  // Key setup: the CUDA-device Tensor output has no deployment device binding.
  const auto plan = compile(graph);

  // Key assertion: the specific diagnostic names the output and tells the author what is missing.
  ASSERT_FALSE(plan.ok());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "IMPLICIT_DEVICE_ZERO_REJECTED";
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  const auto output = std::find_if(plan.ports().begin(), plan.ports().end(), [](const auto& port) {
    return port.direction == PortDirection::kOutput && port.local_name == "tensor";
  });
  ASSERT_NE(output, plan.ports().end());
  EXPECT_EQ(diagnostic->json_pointer,
            "/ports/" + std::to_string(output->id.value()) + "/tensor_device_binding");
  EXPECT_NE(diagnostic->required.find("bind_tensor_output_device"), std::string::npos);
  EXPECT_EQ(diagnostic->provided, "operator_path=source,output_port=tensor,device_binding=missing");
  EXPECT_TRUE(
      std::none_of(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "PROCESS_LOCAL_CHANNEL_PLAN_INVALID";
      }));
}

// A pinned catalog pool uses the adjacent output's public placement as private synchronization
// evidence. Without that explicit device, compilation must not infer ordinal zero.
// REQ: CHAN-095
// REQ: TENSOR-STORAGE-011
TEST(CompilePhysicalPlan, PinnedCatalogPoolUsesPublicOutputDevicePlacement) {
  Graph graph{"public-pinned-catalog-placement"};
  const auto source = graph.op<DeviceTensorOperator>("source");
  static constexpr std::array axes{
      TensorAxisBounds{.max_extent = 1024U, .max_abs_stride_bytes = 1U}};
  TopicOptions topic_options;
  topic_options.endpoint_name = "public-pinned-catalog-placement";
  topic_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kPinnedHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 1024U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported =
      graph.export_topic<Tensor>("holomq://public-pinned-catalog-placement", topic_options);
  graph.add_flow(source->tensor, exported.input());

  const auto implicit = compile(graph);
  EXPECT_FALSE(implicit.ok());
  EXPECT_NE(implicit.json().find("IMPLICIT_DEVICE_ZERO_REJECTED"), std::string_view::npos);
  EXPECT_TRUE(implicit.tensor_allocations().empty());

  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "tensor",
      .device = DeviceId{1},
  });
  const auto explicit_plan = compile(graph, std::move(options));

  ASSERT_TRUE(explicit_plan.ok()) << explicit_plan.json();
  ASSERT_EQ(explicit_plan.tensor_interface_pools().size(), 1U);
  EXPECT_EQ(explicit_plan.tensor_interface_pools().front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(explicit_plan.tensor_interface_pools().front().device, DeviceId{1});
  ASSERT_EQ(explicit_plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(explicit_plan.tensor_allocations().front().device, DeviceId{1});
  const auto copied_transfer =
      std::find_if(explicit_plan.tensor_transfer_plans().begin(),
                   explicit_plan.tensor_transfer_plans().end(),
                   [](const auto& transfer) {
                     return transfer.disposition == PlanDisposition::kAccepted &&
                            transfer.requires_exportable_copy_pool;
                   });
  ASSERT_NE(copied_transfer, explicit_plan.tensor_transfer_plans().end());

  // Key assertion: raw Tensor transfer evidence carries the same placement as its staging pool.
  EXPECT_EQ(copied_transfer->interface_process_group,
            explicit_plan.tensor_interface_pools().front().allocator_process_group);
  EXPECT_EQ(copied_transfer->interface_device,
            explicit_plan.tensor_interface_pools().front().device);
  EXPECT_TRUE(copied_transfer->service_owner_id.starts_with("service.tensor.transfer.topic."));
  const auto state = ExecutionPlanAccess::retain(explicit_plan);
  ASSERT_NE(state, nullptr);
  const auto transport_retention =
      std::find_if(state->pool_capacity_dependencies.begin(),
                   state->pool_capacity_dependencies.end(),
                   [&](const auto& dependency) {
                     return dependency.pool_id == state->tensor_allocations.front().pool_id &&
                            dependency.origin == PoolClaimOrigin::kTransportRetention;
                   });
  ASSERT_NE(transport_retention, state->pool_capacity_dependencies.end());
  // Key assertion: an exported direct output keeps relay/re-export ownership on the separate
  // imported-retention live set rather than folding it into queue delivery.
  EXPECT_NE(std::find(transport_retention->reasons.begin(),
                      transport_retention->reasons.end(),
                      TensorSlotClaimReason::kRelayOrReexport),
            transport_retention->reasons.end());
}

// Independent Operators may use distinct devices because each callback still
// owns one coherent compute-stream obligation. Authored placement order does
// not affect the canonical plan.
// REQ: COMPILE-052
TEST(CompilePhysicalPlan, PublicTensorOutputPlacementAdmitsIndependentDevicesCanonically) {
  Graph graph{"public-multi-device-placement"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<DeviceTensorOperator>("first");
  const auto second = graph.op<DeviceTensorOperator>("second");
  const auto first_sink = graph.op<ManifestTensorSinkOperator>("first-sink");
  const auto second_sink = graph.op<ManifestTensorSinkOperator>("second-sink");
  graph.add_flow(first->tensor, first_sink->input);
  graph.add_flow(second->tensor, second_sink->input);
  graph.partition("first-device").add(first).add(first_sink);
  graph.partition("second-device").add(second).add(second_sink);

  const auto compile_order = [&](bool reverse) {
    CompileOptions options;
    if (reverse) {
      options.deployment
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "second", .output_port = "tensor", .device = DeviceId{1}})
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "first", .output_port = "tensor", .device = DeviceId{0}});
    } else {
      options.deployment
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "first", .output_port = "tensor", .device = DeviceId{0}})
          .bind_tensor_output_device(TensorOutputDevicePlacement{
              .operator_path = "second", .output_port = "tensor", .device = DeviceId{1}});
    }
    return compile(graph, std::move(options));
  };

  const auto forward = compile_order(false);
  const auto reverse = compile_order(true);

  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();
  ASSERT_EQ(forward.tensor_device_resources().size(), 2U);
  EXPECT_EQ(forward.tensor_device_resources()[0].device, DeviceId{0});
  EXPECT_EQ(forward.tensor_device_resources()[1].device, DeviceId{1});
  EXPECT_EQ(forward.digest(), reverse.digest());
}

// REQ: COMPILE-052
TEST(CompilePhysicalPlan, PublicTensorOutputPlacementRejectsInvalidSelectorsAndDuplicates) {
  Graph graph{"invalid-public-tensor-placement"};
  const auto source = graph.op<DeviceTensorOperator>("source");
  const auto sink = graph.op<ManifestTensorSinkOperator>("sink");
  graph.add_flow(source->tensor, sink->input);

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "tensor", .device = DeviceId{-1}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "tensor", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "tensor", .device = DeviceId{1}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "scalar", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "missing", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "sink", .output_port = "input", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "missing", .output_port = "tensor", .device = DeviceId{0}});

  const auto plan = compile(graph, std::move(options));

  EXPECT_FALSE(plan.ok());
  const auto has_reason = [&](std::string_view reason) {
    return std::any_of(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
      return item.token == "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID" &&
             item.provided.find(reason) != std::string::npos;
    });
  };
  EXPECT_TRUE(has_reason("reason=duplicate_output"));
  EXPECT_TRUE(has_reason("reason=invalid_device"));
  EXPECT_TRUE(has_reason("reason=non_tensor_output"));
  EXPECT_TRUE(has_reason("reason=unknown_output"));
  EXPECT_TRUE(has_reason("reason=not_operator_output"));
  EXPECT_TRUE(has_reason("reason=unknown_or_ambiguous_operator"));

  const auto pointer_for_reason = [&](std::string_view reason) -> std::string_view {
    const auto found =
        std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
          return item.token == "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID" &&
                 item.provided.find(reason) != std::string::npos;
        });
    return found == plan.diagnostics().end() ? std::string_view{} : found->json_pointer;
  };
  EXPECT_EQ(pointer_for_reason("reason=invalid_device"), "/deployment/tensor_output_devices/0");
  EXPECT_EQ(pointer_for_reason("reason=duplicate_output"), "/deployment/tensor_output_devices/2");
  EXPECT_EQ(pointer_for_reason("reason=non_tensor_output"), "/deployment/tensor_output_devices/3");
  EXPECT_EQ(pointer_for_reason("reason=unknown_output"), "/deployment/tensor_output_devices/4");
  EXPECT_EQ(pointer_for_reason("reason=not_operator_output"),
            "/deployment/tensor_output_devices/5");
  EXPECT_EQ(pointer_for_reason("reason=unknown_or_ambiguous_operator"),
            "/deployment/tensor_output_devices/6");
}

// REQ: COMPILE-052
TEST(CompilePhysicalPlan, PublicTensorOutputPlacementRejectsTwoDevicesForOneOperator) {
  Graph graph{"conflicting-public-tensor-placement"};
  const auto source = graph.op<DeviceTensorOperator>("source");
  const auto relay = graph.op<DeviceTensorRelayOperator>("relay");
  const auto sink = graph.op<ManifestTensorSinkOperator>("sink");
  graph.add_flow(source->tensor, relay->input);
  graph.add_flow(relay->output, sink->input);

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "tensor", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "relay", .output_port = "output", .device = DeviceId{1}});

  const auto plan = compile(graph, std::move(options));

  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(
      std::any_of(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID" &&
               item.provided.find("reason=operator_device_conflict") != std::string::npos;
      }));
}

// REQ: CHAN-098
// REQ: CHAN-118
// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(CompilePhysicalPlan, PassThroughTensorUsesTypedServiceWithoutFabricatingPoolEvidence) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload(), 2U),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 2U, QueuePolicy::kDropOldest)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  auto result = lower_for_test(snapshot, 35U, profiles);

  ASSERT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.tensor_allocations.empty());
  EXPECT_TRUE(result.tensor_interface_pools.empty());
  EXPECT_TRUE(result.tensor_device_resources.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_size_bytes, 0U);
  EXPECT_EQ(service.tensor_max_byte_span, 0U);
  EXPECT_TRUE(service.tensor_pool_id.empty());
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kUnknown);
  EXPECT_FALSE(service.tensor_device.valid());
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_FALSE(service.event_wake);
  EXPECT_TRUE(result.process_local_channel_publishers.front().replacement_attribution_required);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 1U);
  EXPECT_FALSE(result.process_local_channel_subscribers.front().event_wake);

  auto state = published_physical_plan_state(std::move(result), 35U);
  EXPECT_TRUE(valid_process_local_channel_plan(state));

  auto fallback = lower_for_test(
      snapshot, 35U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);
  ASSERT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(fallback.process_local_channel_services.front().event_wake);
  ASSERT_EQ(fallback.process_local_channel_subscribers.size(), 1U);
  EXPECT_FALSE(fallback.process_local_channel_subscribers.front().event_wake);
  ExecutionPlanState fallback_state;
  fallback_state.plan_generation = 35U;
  fallback_state.execution_partitions = default_execution_partitions(snapshot);
  detail::publish_physical_plan_records(std::move(fallback), fallback_state);
  EXPECT_TRUE(valid_process_local_channel_plan(fallback_state));

  // Key setup: put the otherwise exact forced-HoloMQ Tensor edge on two execution partitions.
  const std::array split_partitions{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U}, .members = {VertexId{1U}}, .execution_lane = 0U},
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{2U}}, .execution_lane = 1U},
  };
  const auto split = lower_for_test(snapshot,
                                    35U,
                                    profiles,
                                    {},
                                    {},
                                    {},
                                    nullptr,
                                    {},
                                    split_partitions,
                                    kForceHoloMQProcessLocalInline);
  ASSERT_TRUE(split.errors.empty());
  // Key assertion: cross-partition Tensor delivery retains its transport event wake.
  EXPECT_TRUE(split.process_local_channel_services.front().event_wake);
  EXPECT_TRUE(split.process_local_channel_subscribers.front().event_wake);
}

// A complete raw-Tensor fanout uses the serialized activation cut while retaining one independent
// descriptor/sidecar claim per subscriber; descriptor-backed schemas remain a later payload gate.
// REQ: CHAN-098
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, SuppressesRawTensorFanoutEventWakeServiceWide) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload()),
      input_port(3U, 3U, "in", tensor_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  // Key setup: lower one complete two-target raw-Tensor service on a shared execution partition.
  auto result = lower_for_test(snapshot, 35U, profiles, {}, {}, {}, nullptr, {}, partitions);

  // Key assertion: HoloMQ wake suppression is uniform across both Tensor subscribers and replay
  // rejects a changed subscriber wake fact.
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_identity.kind, PayloadIdentityKind::kTensor);
  EXPECT_EQ(service.payload_identity.payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_FALSE(service.event_wake);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 35U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_subscribers.back().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// A complete single-Tensor descriptor schema fanout uses the same serialized activation cut while
// preserving one verified owning schema envelope and sidecar claim per subscriber.
// REQ: CHAN-099
// REQ: CHAN-101
// REQ: CHAN-119
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, SuppressesSingleDescriptorTensorSchemaFanoutEventWakeServiceWide) {
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", payload),
      input_port(2U, 2U, "in", payload),
      input_port(3U, 3U, "in", payload),
  };
  for (auto& port : snapshot.ports) {
    port.serialized_payload_capacity_bytes = type_ops->wire_capacity;
  }
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  // Key setup: lower one complete two-target Image service on a shared execution partition.
  auto result = lower_for_test(snapshot, 35U, profiles, {}, {}, {}, nullptr, {}, partitions);

  // Key assertion: both descriptor-sidecar subscribers are uniformly eventless, while replay
  // rejects a changed subscriber wake fact.
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_identity.kind, PayloadIdentityKind::kSchema);
  EXPECT_EQ(service.payload_identity.payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_LE(service.tensor_fields.size(), 1U);
  EXPECT_FALSE(service.event_wake);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 35U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_subscribers.back().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// A multi-Tensor descriptor schema shares one publication/readiness sequence while preserving the
// complete frozen field-storage inventory for every subscriber.
// REQ: CHAN-099
// REQ: CHAN-101
// REQ: CHAN-120
TEST(CompilePhysicalPlan, SuppressesMultiDescriptorTensorSchemaFanoutEventWakeServiceWide) {
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);
  auto fixture = make_multi_tensor_snapshot(type_ops->wire_capacity);
  fixture.graph.vertices.push_back(vertex(3U, "second-sink"));
  fixture.graph.ports.push_back(input_port(3U, 3U, "frames", image_payload(), 2U));
  auto& second_input = fixture.graph.ports.back();
  second_input.serialized_payload_capacity_bytes = type_ops->wire_capacity;
  second_input.descriptor_tensor_field_count = 3U;
  second_input.setup.tensor_fields = fixture.graph.ports.front().setup.tensor_fields;
  fixture.graph.edges.push_back(edge(2U, 1U, 3U, 2U));
  fixture.profiles = standard_profiles(fixture.graph.vertices.size());
  fixture.partitions = default_execution_partitions(fixture.graph);

  // Key setup: lower one complete two-target, three-field descriptor schema service on a shared
  // execution partition.
  auto result = lower_for_test(
      fixture.graph, 35U, fixture.profiles, {}, {}, {}, nullptr, {}, fixture.partitions);

  // Key assertion: the complete three-field service is uniformly eventless with its per-field
  // contracts intact, while replay rejects a changed subscriber wake fact.
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kHoloMQ);
  ASSERT_EQ(service.tensor_fields.size(), 3U);
  EXPECT_FALSE(service.event_wake);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 35U;
  state.execution_partitions = fixture.partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_subscribers.back().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// A cross-process direct Tensor edge is represented by one BufferHandle-capable service
// carrying the exact source allocation contract. No process-local service is emitted.
//
//   source [process A] --[cross-process, drop-oldest, depth=2]--> sink [process B]
//
// REQ: COMPILE-048
// REQ: COMPILE-064
// REQ: CHAN-085
// REQ: CHAN-088
TEST(CompilePhysicalPlan, CrossProcessTensorCarriesPoolFactsIntoDedicatedService) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload(), 2U),
      input_port(2U, 2U, "in", tensor_payload(), 2U),
  };
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.ports[0].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.edges = {edge(1U, 1U, 2U, 2U, QueuePolicy::kDropOldest)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto endpoints = catalog_endpoints(snapshot);
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  const RuntimePackageCapabilityView package = runtime_package_capability_view();

  auto result = lower_physical_plan(
      snapshot, 37U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});

  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  // Key assertion: one cross-process service carries pool facts; no process-local service emitted.
  ASSERT_EQ(result.tensor_allocations.size(), 1U);
  EXPECT_TRUE(result.process_local_channel_services.empty());
  ASSERT_EQ(result.cross_process_channel_services.size(), 1U);
  ASSERT_EQ(result.cross_process_channel_publishers.size(), 1U);
  ASSERT_EQ(result.cross_process_channel_subscribers.size(), 1U);
  const auto& service = result.cross_process_channel_services.front();
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_size_bytes, 0U);
  EXPECT_EQ(service.tensor_max_byte_span, 1024U);
  EXPECT_EQ(service.tensor_pool_id, result.tensor_allocations.front().pool_id);
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kHost);
  EXPECT_FALSE(service.tensor_device.valid());
  EXPECT_NE(service.source_process_group, service.target_process_group);

  auto state = published_physical_plan_state(std::move(result), 37U, placements);
  // The top-level compile reconciliation stamps the partition-derived locality after physical
  // lowering. This source-private fixture invokes the physical pass directly, so mirror that one
  // final-plan transformation before exercising the cross-process validator.
  ASSERT_EQ(state.payload_edges.size(), 1U);
  ASSERT_EQ(state.channel_locality.size(), 1U);
  state.payload_edges.front().locality = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requested = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().accepted = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requires_runtime_attach = true;
  EXPECT_TRUE(valid_cross_process_channel_plan(state));

  const PayloadEdgePlanRecord original_edge = state.payload_edges.front();
  const PortPlanRecord original_source = state.ports.front();
  const PortPlanRecord original_target = state.ports.back();
  const auto expect_rejected = [&](std::string_view mutation) {
    SCOPED_TRACE(mutation);
    EXPECT_FALSE(valid_cross_process_channel_plan(state));
    state.payload_edges.front() = original_edge;
    state.ports.front() = original_source;
    state.ports.back() = original_target;
  };

  state.payload_edges.front().edge = {};
  expect_rejected("invalid edge identity");
  state.payload_edges.front().source = PortId{99U};
  expect_rejected("missing source port");
  state.payload_edges.front().target = PortId{99U};
  expect_rejected("missing target port");
  state.ports.front().owner_kind = PortOwnerKind::kGraphInterface;
  expect_rejected("non-Operator source owner");
  state.ports.back().owner_kind = PortOwnerKind::kGraphInterface;
  expect_rejected("non-Operator target owner");
  state.ports.front().direction = PortDirection::kInput;
  expect_rejected("source is not an output");
  state.ports.back().direction = PortDirection::kOutput;
  expect_rejected("target is not an input");
  state.payload_edges.front().queue_depth = 0U;
  expect_rejected("zero queue depth");
  state.payload_edges.front().producer_max_emits_per_compute = 0U;
  expect_rejected("zero producer burst");
  ++state.ports.front().max_emits_per_compute;
  expect_rejected("source and edge burst mismatch");
  state.payload_edges.front().disposition = PlanDisposition::kRejected;
  expect_rejected("rejected edge disposition");

  // Key assertion: malformed coalesced relations cannot hide duplicate service/publisher
  // identities or silently map an unplaced target to ProcessGroupId{}.
  const auto original_placements = state.vertex_process_placements;
  state.vertex_process_placements.erase(
      std::remove_if(
          state.vertex_process_placements.begin(),
          state.vertex_process_placements.end(),
          [&](const auto& placement) { return placement.vertex == original_target.owner; }),
      state.vertex_process_placements.end());
  EXPECT_FALSE(valid_cross_process_channel_plan(state));
  state.vertex_process_placements = original_placements;

  state.cross_process_channel_services.push_back(state.cross_process_channel_services.front());
  state.cross_process_channel_publishers.push_back(state.cross_process_channel_publishers.front());
  EXPECT_FALSE(valid_cross_process_channel_plan(state));
  state.cross_process_channel_services.pop_back();
  state.cross_process_channel_publishers.pop_back();

  auto unmatched_service = state.cross_process_channel_services.front();
  unmatched_service.service_id += "-unmatched";
  state.cross_process_channel_services.push_back(unmatched_service);
  state.cross_process_channel_publishers.push_back(state.cross_process_channel_publishers.front());
  EXPECT_FALSE(valid_cross_process_channel_plan(state));
  state.cross_process_channel_services.pop_back();
  state.cross_process_channel_publishers.pop_back();

  // Admitted artifact finalization must reject a broken cross-process join instead of deferring
  // the same closed-relation failure to Runtime session construction.
  state.admitted = true;
  ++state.cross_process_channel_services.front().max_subscriber_queue_depth;
  const auto finalized = finalize_plan_artifact(state);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().message, "CROSS_PROCESS_CHANNEL_PLAN_INVALID");
}

// A distributed direct-Tensor edge has no implicit conversion provider. The source allocation
// must therefore satisfy every explicit target memory, dtype, and rank constraint before a
// BufferHandle service is admitted.
// REQ: CHAN-088
// REQ: CHAN-093
TEST(CompilePhysicalPlan, CrossProcessTensorRejectsUnrealizableDirectLayouts) {
  const auto lower_with_target = [](TensorRepresentation target_layout,
                                    MemoryKind source_memory = MemoryKind::kHost,
                                    std::uint32_t maximum_rank = 1U,
                                    bool provide_device_binding = true) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
    snapshot.ports = {
        output_port(1U, 1U, "out", tensor_payload(), 1U),
        input_port(2U, 2U, "in", tensor_payload(), 1U),
    };
    snapshot.ports[0].setup.tensor_allocation =
        TensorPoolBounds{.max_rank = maximum_rank, .max_byte_span = 1024U};
    snapshot.ports[0].setup.tensor_layout = TensorRepresentation{
        .memory_kind = source_memory,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    snapshot.ports[1].setup.tensor_layout = target_layout;
    snapshot.edges = {edge(1U, 1U, 2U, 1U)};
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto endpoints = catalog_endpoints(snapshot);
    const auto placements = split_process_placements(snapshot);
    const auto partitions = default_execution_partitions(snapshot);
    const std::array device_binding{
        TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}},
    };
    const std::span<const TensorDeviceBindingRecord> device_bindings =
        provide_device_binding && (source_memory == MemoryKind::kPinnedHost ||
                                   source_memory == MemoryKind::kCudaDevice ||
                                   source_memory == MemoryKind::kCudaManaged)
            ? std::span<const TensorDeviceBindingRecord>{device_binding}
            : std::span<const TensorDeviceBindingRecord>{};
    return lower_physical_plan(snapshot,
                               41U,
                               profiles,
                               runtime_package_capability_view(),
                               {},
                               {},
                               endpoints,
                               placements,
                               {},
                               partitions,
                               device_bindings);
  };

  const std::array incompatible{
      TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice},
      TensorRepresentation{.dtype = DLDataType{kDLFloat, 8U, 1U}},
      TensorRepresentation{.rank = 2U},
      TensorRepresentation{.strides = TensorStridePolicy::kBoundedStrided},
  };
  for (const auto& target : incompatible) {
    const auto result = lower_with_target(target);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(), [](const auto& error) {
      return error.token == "PHYSICAL_TENSOR_LAYOUT_MISMATCH";
    }));
    EXPECT_TRUE(result.cross_process_channel_services.empty());
  }

  const auto managed = lower_with_target({}, MemoryKind::kCudaManaged);
  EXPECT_TRUE(std::any_of(managed.errors.begin(), managed.errors.end(), [](const auto& error) {
    return error.token == "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED";
  }));
  EXPECT_TRUE(managed.cross_process_channel_services.empty());

  const auto unbound_pinned =
      lower_with_target({}, MemoryKind::kPinnedHost, 1U, /*provide_device_binding=*/false);
  EXPECT_TRUE(std::any_of(
      unbound_pinned.errors.begin(), unbound_pinned.errors.end(), [](const auto& error) {
        return error.token == "IMPLICIT_DEVICE_ZERO_REJECTED";
      }));
  const auto excessive_rank = lower_with_target({}, MemoryKind::kHost, 9U);
  EXPECT_TRUE(std::any_of(
      excessive_rank.errors.begin(), excessive_rank.errors.end(), [](const auto& error) {
        return error.token == "PHYSICAL_TENSOR_RANK_UNSUPPORTED";
      }));
  EXPECT_TRUE(excessive_rank.cross_process_channel_services.empty());

  const auto compatible = lower_with_target(TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  });
  EXPECT_TRUE(compatible.errors.empty());
  EXPECT_EQ(compatible.cross_process_channel_services.size(), 1U);

  const auto pinned = lower_with_target(
      TensorRepresentation{
          .memory_kind = MemoryKind::kPinnedHost,
          .dtype = DLDataType{kDLUInt, 8U, 1U},
          .rank = 1U,
      },
      MemoryKind::kPinnedHost);
  EXPECT_TRUE(pinned.errors.empty());
  ASSERT_EQ(pinned.cross_process_channel_services.size(), 1U);
  EXPECT_EQ(pinned.cross_process_channel_services.front().tensor_memory_kind,
            MemoryKind::kPinnedHost);
  EXPECT_EQ(pinned.cross_process_channel_services.front().tensor_device, DeviceId{0});
  EXPECT_EQ(std::count_if(pinned.tensor_device_resources.begin(),
                          pinned.tensor_device_resources.end(),
                          [](const auto& resource) { return resource.device == DeviceId{0}; }),
            2U);
}

// Graph authoring rejects conflicts between explicit port declarations. Physical lowering also
// joins the resolved direct-pool contract so an implicit host default cannot contradict a target.
// REQ: GRAPH-025
TEST(CompilePhysicalPlan, SameProcessDirectTensorPoolSatisfiesEveryTargetLayout) {
  const auto lower_with_layouts = [](std::optional<TensorRepresentation> source_layout,
                                     TensorRepresentation target_layout) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
    snapshot.ports = {
        output_port(1U, 1U, "out", tensor_payload(), 1U),
        input_port(2U, 2U, "in", tensor_payload(), 1U),
    };
    snapshot.ports[0].setup.tensor_allocation =
        TensorPoolBounds{.max_rank = 2U, .max_byte_span = 1024U};
    snapshot.ports[0].setup.tensor_layout = source_layout;
    snapshot.ports[1].setup.tensor_layout = target_layout;
    snapshot.edges = {edge(1U, 1U, 2U, 1U)};
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const std::array target_device_binding{
        TensorDeviceBindingRecord{.port = PortId{2U}, .device = DeviceId{0}}};
    const std::span<const TensorDeviceBindingRecord> device_bindings =
        target_layout.memory_kind == MemoryKind::kCudaDevice
            ? std::span<const TensorDeviceBindingRecord>{target_device_binding}
            : std::span<const TensorDeviceBindingRecord>{};
    return lower_for_test(snapshot, 42U, profiles, {}, {}, {}, nullptr, device_bindings);
  };

  const auto expect_rejected = [&](std::string_view label,
                                   std::optional<TensorRepresentation>
                                       source_layout,
                                   TensorRepresentation target_layout,
                                   std::string_view expected_token) {
    SCOPED_TRACE(label);
    const auto result = lower_with_layouts(std::move(source_layout), std::move(target_layout));
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(), [&](const auto& error) {
      return error.token == expected_token;
    }));
    EXPECT_TRUE(result.process_local_channel_services.empty());
  };

  const TensorRepresentation compatible{.memory_kind = MemoryKind::kHost,
                                        .dtype = DLDataType{kDLUInt, 8U, 1U},
                                        .rank = 1U,
                                        .strides = TensorStridePolicy::kContiguousOnly};
  expect_rejected("implicit host",
                  std::nullopt,
                  TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice},
                  "PHYSICAL_TENSOR_MEMORY_KIND_UNRESOLVED");
  expect_rejected("memory kind",
                  compatible,
                  TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice},
                  "PHYSICAL_TENSOR_LAYOUT_MISMATCH");
  expect_rejected("dtype",
                  compatible,
                  TensorRepresentation{.dtype = DLDataType{kDLFloat, 8U, 1U}},
                  "PHYSICAL_TENSOR_LAYOUT_MISMATCH");
  expect_rejected(
      "rank", compatible, TensorRepresentation{.rank = 2U}, "PHYSICAL_TENSOR_LAYOUT_MISMATCH");
  expect_rejected("strides",
                  compatible,
                  TensorRepresentation{.strides = TensorStridePolicy::kBoundedStrided},
                  "PHYSICAL_TENSOR_LAYOUT_MISMATCH");

  const auto accepted = lower_with_layouts(compatible, compatible);
  EXPECT_TRUE(accepted.errors.empty());
  EXPECT_EQ(accepted.process_local_channel_services.size(), 1U);
}

// A direct allocation can leave some axis bounds unconstrained, but it cannot satisfy an input
// that requires bounds for more axes than the allocation records.
//
// Graph topology:
//   source.out -- Tensor --> sink.in
// REQ: GRAPH-025
TEST(CompilePhysicalPlan, SameProcessDirectTensorPoolRejectsIncompleteAxisInventory) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload(), 1U),
      input_port(2U, 2U, "in", tensor_payload(), 1U),
  };
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 2U,
      .strides = TensorStridePolicy::kContiguousOnly,
  };
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 2U, .max_byte_span = 64U};
  snapshot.ports[0].setup.tensor_layout = layout;
  snapshot.ports[0].setup.tensor_contract = TensorContractRecord{
      .role = TensorContractRole::kOutput,
      .representation = layout,
      .max_rank = 2U,
      .max_byte_span = 64U,
      .axes = {{.max_extent = 8U}},
  };
  snapshot.ports[1].setup.tensor_layout = layout;
  snapshot.ports[1].setup.tensor_contract = TensorContractRecord{
      .role = TensorContractRole::kInput,
      .representation = layout,
      .max_rank = 2U,
      .max_byte_span = 64U,
      .axes = {{.max_extent = 8U}, {.max_extent = 8U}},
  };
  snapshot.edges = {edge(1U, 1U, 2U, 1U)};

  // Key setup: the direct provider has one axis row while the input requires two.
  const auto result = lower_for_test(snapshot, 42U, standard_profiles(snapshot.vertices.size()));

  // Key assertion: direct admission delegates to the complete provider-compatibility contract.
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const auto& error) {
    return error.token == "PHYSICAL_TENSOR_LAYOUT_MISMATCH";
  }));
  EXPECT_TRUE(result.process_local_channel_services.empty());
}

// Every policy layer is retained for future converter insertion. Until that lowerer exists, the
// effective non-default request must reject rather than silently behaving as a direct edge.
// REQ: COMPILE-050
TEST(CompilePhysicalPlan, SameProcessTensorRejectsUnimplementedMemoryConversionPolicies) {
  const auto lower_with_policies = [](std::optional<MemoryConversionPolicy> input_setup,
                                      std::optional<MemoryConversionPolicy>
                                          input_override,
                                      std::optional<MemoryConversionPolicy>
                                          edge_option,
                                      std::optional<MemoryConversionPolicy>
                                          edge_override) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
    snapshot.ports = {
        output_port(1U, 1U, "out", tensor_payload(), 1U),
        input_port(2U, 2U, "in", tensor_payload(), 1U),
    };
    const TensorRepresentation layout{
        .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U};
    snapshot.ports[0].setup.tensor_allocation =
        TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
    snapshot.ports[0].setup.tensor_layout = layout;
    snapshot.ports[1].setup.tensor_layout = layout;
    snapshot.ports[1].setup.memory_conversion = input_setup;
    snapshot.ports[1].graph_override.memory_conversion = input_override;
    snapshot.edges = {edge(1U, 1U, 2U, 1U)};
    snapshot.edges[0].add_flow_options.memory_conversion = edge_option;
    snapshot.edges[0].graph_override.memory_conversion = edge_override;
    const auto profiles = standard_profiles(snapshot.vertices.size());
    return lower_for_test(snapshot, 42U, profiles);
  };

  const auto expect_rejected = [&](std::string_view label,
                                   std::optional<MemoryConversionPolicy>
                                       input_setup,
                                   std::optional<MemoryConversionPolicy>
                                       input_override,
                                   std::optional<MemoryConversionPolicy>
                                       edge_option,
                                   std::optional<MemoryConversionPolicy>
                                       edge_override) {
    SCOPED_TRACE(label);
    const auto result =
        lower_with_policies(input_setup, input_override, edge_option, edge_override);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(), [](const auto& error) {
      return error.token == "PHYSICAL_TENSOR_CONVERSION_UNSUPPORTED";
    }));
    EXPECT_TRUE(result.payload_edges.empty());
    EXPECT_TRUE(result.process_local_channel_services.empty());
  };

  expect_rejected("input setup", MemoryConversionPolicy::kRequire, {}, {}, {});
  expect_rejected("input override", {}, MemoryConversionPolicy::kAllowIfSafe, {}, {});
  expect_rejected("edge option", {}, {}, MemoryConversionPolicy::kDisable, {});
  expect_rejected("edge override", {}, {}, {}, MemoryConversionPolicy::kRequire);

  const auto default_policy = lower_with_policies({}, {}, {}, {});
  EXPECT_TRUE(default_policy.errors.empty());
  EXPECT_EQ(default_policy.process_local_channel_services.size(), 1U);

  // Edge-specific provenance follows the same precedence as queue options: override, add-flow,
  // target Graph override, then target setup.
  const auto reset_by_edge_override = lower_with_policies(MemoryConversionPolicy::kRequire,
                                                          MemoryConversionPolicy::kRequire,
                                                          MemoryConversionPolicy::kRequire,
                                                          MemoryConversionPolicy::kDefault);
  EXPECT_TRUE(reset_by_edge_override.errors.empty());
  EXPECT_EQ(reset_by_edge_override.process_local_channel_services.size(), 1U);
}

// Descriptor-schema bindings are discovered before raw Tensor bindings, but the final private
// Channel families must be canonical regardless of that lowering-phase insertion order.
//
//   raw-source   --[edge 1]--> raw-sink
//   image-source --[edge 2]--> image-sink
//
// REQ: COMPILE-009
// REQ: CHAN-096
// REQ: CHAN-119
TEST(CompilePhysicalPlan, CanonicalizesDescriptorAndRawTensorBindingsAfterCompleteLowering) {
  const auto image = image_payload();
  const RuntimePortTypeOps* const image_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(image_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "raw-source"),
                       vertex(2U, "raw-sink"),
                       vertex(3U, "image-source"),
                       vertex(4U, "image-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "raw", tensor_payload()),
      input_port(2U, 2U, "raw", tensor_payload()),
      output_port(3U, 3U, "image", image),
      input_port(4U, 4U, "image", image),
  };
  snapshot.ports[2].serialized_payload_capacity_bytes = image_ops->wire_capacity;
  snapshot.ports[3].serialized_payload_capacity_bytes = image_ops->wire_capacity;
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  // Key setup: descriptor lowering appends source 3 before raw Tensor lowering appends source 1,
  // making insertion order the reverse of the canonical source key.
  const auto result = lower_for_test(snapshot, 35U, profiles);
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 2U);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 2U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);

  // Key assertion: one final canonicalization orders every private Channel family by its complete
  // key after both binding paths have contributed records.
  EXPECT_EQ(result.process_local_channel_services[0].source, PortId{1U});
  EXPECT_EQ(result.process_local_channel_services[0].payload_identity.kind,
            PayloadIdentityKind::kTensor);
  EXPECT_EQ(result.process_local_channel_services[1].source, PortId{3U});
  EXPECT_EQ(result.process_local_channel_services[1].payload_identity.kind,
            PayloadIdentityKind::kSchema);
  EXPECT_EQ(result.process_local_channel_publishers[0].source, PortId{1U});
  EXPECT_EQ(result.process_local_channel_publishers[1].source, PortId{3U});
  EXPECT_EQ(result.process_local_channel_subscribers[0].edge, EdgeId{1U});
  EXPECT_EQ(result.process_local_channel_subscribers[1].edge, EdgeId{2U});
}

// Descriptor-backed generated Image values carry bounded metadata beside one retained Tensor.
// Process-local lowering must therefore select the Tensor-aware service while leaving storage
// authority in the published Image sample rather than fabricating a direct-output allocation pool.
// REQ: COMPILE-048
// REQ: COMPILE-064
// REQ: CHAN-084
// REQ: CHAN-099
// REQ: CHAN-119
// REQ: TENSOR-SCHEMA-005
// REQ: OPERATOR-035
// REQ: CHAN-128
TEST(CompilePhysicalPlan, DescriptorTensorImageUsesTypedServiceWithoutFabricatingPoolEvidence) {
  const auto payload = image_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);
  ASSERT_EQ(type_ops->codec_family, RuntimePayloadCodecFamily::kDescriptorTensorSchema);
  ASSERT_GT(type_ops->wire_capacity, 0U);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "image-source"), vertex(2U, "image-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "image", payload, 2U),
      input_port(2U, 2U, "image", payload, 2U),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[1].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.edges = {edge(1U, 1U, 2U, 2U, QueuePolicy::kDropOldest)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  auto result = lower_for_test(snapshot, 36U, profiles);

  ASSERT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.tensor_allocations.empty());
  EXPECT_TRUE(result.tensor_interface_pools.empty());
  EXPECT_TRUE(result.tensor_device_resources.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 1U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.payload_identity.kind, PayloadIdentityKind::kSchema);
  EXPECT_EQ(service.payload_identity.payload_kind, PayloadKind::kDescriptorBacked);
  ASSERT_TRUE(service.payload_identity.schema);
  ASSERT_TRUE(payload.schema);
  EXPECT_EQ(service.payload_identity.schema->canonical_name, payload.schema->canonical_name);
  EXPECT_EQ(service.payload_identity.schema->version, payload.schema->version);
  EXPECT_EQ(service.payload_identity.schema->schema_sha256, payload.schema->schema_sha256);
  EXPECT_EQ(service.payload_identity.schema->id, payload.schema->id);
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_size_bytes, type_ops->wire_capacity);
  EXPECT_EQ(service.tensor_max_byte_span, 0U);
  EXPECT_TRUE(service.tensor_pool_id.empty());
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kUnknown);
  EXPECT_FALSE(service.tensor_device.valid());
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_FALSE(service.event_wake);
  EXPECT_TRUE(result.process_local_channel_publishers.front().replacement_attribution_required);
  EXPECT_FALSE(result.process_local_channel_subscribers.front().event_wake);

  auto state = published_physical_plan_state(std::move(result), 36U);
  EXPECT_TRUE(valid_process_local_channel_plan(state));

  auto fallback = lower_for_test(
      snapshot, 36U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);
  ASSERT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(fallback.process_local_channel_services.front().event_wake);
  EXPECT_FALSE(fallback.process_local_channel_subscribers.front().event_wake);
  ExecutionPlanState fallback_state;
  fallback_state.plan_generation = 36U;
  fallback_state.execution_partitions = default_execution_partitions(snapshot);
  detail::publish_physical_plan_records(std::move(fallback), fallback_state);
  EXPECT_TRUE(valid_process_local_channel_plan(fallback_state));
}

// A second Tensor-bearing root must select the same bounded descriptor service as Image. Its
// different scalar/string metadata cannot create a schema-specific transport or staging path.
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(CompilePhysicalPlan, DescriptorTensorPacketUsesGenericBoundedServiceAndStagingRules) {
  constexpr std::uint64_t kMaximumTensorBytes = 2048U;
  const auto payload = tensor_packet_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::TensorPacketT>();
  ASSERT_NE(type_ops, nullptr);
  ASSERT_EQ(type_ops->codec_family, RuntimePayloadCodecFamily::kDescriptorTensorSchema);
  ASSERT_GT(type_ops->wire_capacity, 0U);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "packet-source"), vertex(2U, "packet-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "packet", payload, 2U),
      input_port(2U, 2U, "packet", payload, 2U),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[1].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.edges = {edge(1U, 1U, 2U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto local = lower_for_test(snapshot, 37U, profiles);
  ASSERT_TRUE(local.errors.empty());
  ASSERT_EQ(local.process_local_channel_services.size(), 1U);
  EXPECT_TRUE(local.process_local_channel_services.front().tensor_payload);
  EXPECT_EQ(local.process_local_channel_services.front().payload_size_bytes,
            type_ops->wire_capacity);
  EXPECT_TRUE(local.tensor_allocations.empty());
  EXPECT_TRUE(local.tensor_interface_pools.empty());

  // Key setup: force the exact 1:1 descriptor schema through HoloMQ instead of direct retention.
  const auto fallback = lower_for_test(
      snapshot, 37U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);
  ASSERT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  // Key assertion: Schema+Tensor(s) uses the same eventless activation-cut contract as Image.
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(fallback.process_local_channel_services.front().event_wake);
  EXPECT_FALSE(fallback.process_local_channel_subscribers.front().event_wake);

  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports[0].setup.tensor_layout = layout;
  snapshot.ports[1].setup.tensor_layout = layout;
  const auto endpoints = catalog_endpoints(snapshot);
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  const RuntimePackageCapabilityView package = runtime_package_capability_view();

  const auto rejected = lower_physical_plan(
      snapshot, 38U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});
  const auto unsupported = std::ranges::find(
      rejected.errors, "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED", &PhysicalPlanDiagnostic::token);
  ASSERT_NE(unsupported, rejected.errors.end());
  EXPECT_EQ(unsupported->provided,
            "cross-process descriptor schema has no admitted staging contract");
  EXPECT_TRUE(rejected.cross_process_channel_services.empty());

  snapshot.edges.front().effective_tensor_payload = PayloadOptionsRecord{
      .max_tensor_byte_span = kMaximumTensorBytes,
      .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  const auto admitted = lower_physical_plan(
      snapshot, 38U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});
  ASSERT_TRUE(admitted.errors.empty())
      << (admitted.errors.empty() ? std::string{} : admitted.errors.front().token);
  ASSERT_EQ(admitted.cross_process_channel_services.size(), 1U);
  const auto& service = admitted.cross_process_channel_services.front();
  EXPECT_TRUE(service.tensor_payload);
  EXPECT_EQ(service.payload_size_bytes, type_ops->wire_capacity);
  EXPECT_EQ(service.tensor_max_byte_span, kMaximumTensorBytes);
  EXPECT_FALSE(service.tensor_pool_id.empty());
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kHost);
  EXPECT_EQ(service.tensor_emit_policy, InterfaceTensorEmitPolicy::kCopyIfNeeded);
}

// A Tensor backing owns a finite number of release-stream slots shared across all fanout
// services. The compiler must reject when fanout exceeds MemoryBuffer::kMaximumFanoutReleaseStreams
// without emitting any services. Tested for both raw Tensor and descriptor-backed Image payloads.
//
//   source --> sink-0 --> ... --> sink-N   (N == kMaximumFanoutReleaseStreams)
//
// Admitted at N; rejected at N+1 with PHYSICAL_TENSOR_FANOUT_CAPACITY_INSUFFICIENT.
// REQ: CHAN-023
// REQ: CHAN-083
// REQ: CHAN-084
// REQ: CHAN-098
// REQ: CHAN-099
TEST(CompilePhysicalPlan, RejectsAggregateDescriptorTensorFanoutAboveMemoryBufferCapacity) {
  const auto make_snapshot = [](std::size_t recipients,
                                const PayloadIdentityRecord& payload,
                                std::uint64_t serialized_capacity) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices.reserve(recipients + 1U);
    snapshot.ports.reserve(recipients + 1U);
    snapshot.edges.reserve(recipients);
    snapshot.vertices.push_back(vertex(1U, "source"));
    snapshot.ports.push_back(output_port(1U, 1U, "out", payload));
    snapshot.ports.front().serialized_payload_capacity_bytes = serialized_capacity;
    for (std::size_t index = 0U; index < recipients; ++index) {
      const std::uint64_t id = index + 2U;
      snapshot.vertices.push_back(vertex(id, "sink-" + std::to_string(index)));
      snapshot.ports.push_back(input_port(id, id, "in", payload));
      snapshot.ports.back().serialized_payload_capacity_bytes = serialized_capacity;
      snapshot.edges.push_back(
          edge(index + 1U,
               1U,
               id,
               1U,
               index % 2U == 0U ? QueuePolicy::kBlocking : QueuePolicy::kDropOldest));
    }
    return snapshot;
  };

  const std::size_t maximum = MemoryBuffer::kMaximumFanoutReleaseStreams;
  // Key setup: exact capacity → admitted; one over → rejected for both Tensor and Image.
  auto admitted = make_snapshot(maximum, tensor_payload(), 0U);
  auto admitted_result = lower_for_test(admitted, 37U, standard_profiles(admitted.vertices.size()));
  EXPECT_TRUE(admitted_result.errors.empty());
  ASSERT_EQ(admitted_result.process_local_channel_services.size(), 2U);
  EXPECT_EQ(admitted_result.process_local_channel_subscribers.size(), maximum);
  // Key assertion: raw Tensor fanout remains on HoloMQ, while the complete same-partition sibling
  // family suppresses event wakes for both queue-policy services.
  EXPECT_TRUE(
      std::ranges::all_of(admitted_result.process_local_channel_services, [](const auto& service) {
        return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ &&
               !service.event_wake &&
               service.event_wake_reason == ProcessLocalHoloMQEventWakeReason::kSuppressed;
      }));
  EXPECT_TRUE(std::ranges::none_of(admitted_result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));

  const RuntimePortTypeOps* const image_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(image_ops, nullptr);
  auto admitted_image = make_snapshot(maximum, image_payload(), image_ops->wire_capacity);
  auto admitted_image_result =
      lower_for_test(admitted_image, 37U, standard_profiles(admitted_image.vertices.size()));
  ASSERT_TRUE(admitted_image_result.errors.empty());
  ASSERT_EQ(admitted_image_result.process_local_channel_services.size(), 2U);
  // Key assertion: descriptor-Tensor schema fanout retains HoloMQ and suppresses both sibling
  // service wakes under the same complete-family proof.
  EXPECT_TRUE(std::ranges::all_of(
      admitted_image_result.process_local_channel_services, [](const auto& service) {
        return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ &&
               !service.event_wake &&
               service.event_wake_reason == ProcessLocalHoloMQEventWakeReason::kSuppressed;
      }));
  EXPECT_TRUE(std::ranges::none_of(admitted_image_result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));

  auto tensor = make_snapshot(maximum + 1U, tensor_payload(), 0U);
  auto tensor_result = lower_for_test(tensor, 38U, standard_profiles(tensor.vertices.size()));
  ASSERT_EQ(tensor_result.errors.size(), 1U);
  EXPECT_TRUE(has_error(tensor_result, "PHYSICAL_TENSOR_FANOUT_CAPACITY_INSUFFICIENT"));
  EXPECT_TRUE(tensor_result.process_local_channel_services.empty());
  EXPECT_TRUE(tensor_result.process_local_channel_publishers.empty());
  EXPECT_TRUE(tensor_result.process_local_channel_subscribers.empty());

  auto image = make_snapshot(maximum + 1U, image_payload(), image_ops->wire_capacity);
  auto image_result = lower_for_test(image, 39U, standard_profiles(image.vertices.size()));
  ASSERT_EQ(image_result.errors.size(), 1U);
  EXPECT_TRUE(has_error(image_result, "PHYSICAL_TENSOR_FANOUT_CAPACITY_INSUFFICIENT"));
  EXPECT_TRUE(image_result.process_local_channel_services.empty());
}

// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(CompilePhysicalPlan, DeviceResourceLoweringRequiresTheFinalAcceptedCover) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "other")};
  snapshot.ports = {output_port(1U, 1U, "out", tensor_payload())};
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{1}},
  };
  const std::array incomplete_cover{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = {VertexId{2U}},
          .execution_lane = 0U,
      },
  };

  const auto result =
      lower_for_test(snapshot, 33U, profiles, {}, {}, {}, nullptr, bindings, incomplete_cover);

  EXPECT_TRUE(has_error(result, "PARTITION_RESOURCE_REALIZATION_FAILED"));
  EXPECT_TRUE(result.tensor_allocations.empty());
  EXPECT_TRUE(result.tensor_device_resources.empty());
}

// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(PartitionResourceSeparationTest, NoImplicitStreamPoolOrGreenContext) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", tensor_payload()),
      input_port(2U, 2U, "in", tensor_payload(), 2U),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 2U)};
  snapshot.ports.front().setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array bindings{
      TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{2}},
  };
  const std::array baseline_cover{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = {VertexId{1U}, VertexId{2U}},
          .execution_lane = 0U,
      },
  };
  const std::array source_chain_cover{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = {VertexId{1U}},
          .execution_lane = 0U,
      },
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U},
          .members = {VertexId{2U}},
          .execution_lane = 0U,
      },
  };

  const auto baseline =
      lower_for_test(snapshot, 34U, profiles, {}, {}, {}, nullptr, bindings, baseline_cover);
  const auto source_chain =
      lower_for_test(snapshot, 34U, profiles, {}, {}, {}, nullptr, bindings, source_chain_cover);

  ASSERT_TRUE(baseline.errors.empty());
  ASSERT_TRUE(source_chain.errors.empty());
  ASSERT_EQ(baseline.tensor_allocations.size(), 1U);
  ASSERT_EQ(source_chain.tensor_allocations.size(), 1U);
  EXPECT_EQ(baseline.tensor_allocations.front().pool_id,
            source_chain.tensor_allocations.front().pool_id);
  EXPECT_EQ(baseline.tensor_allocations.front().slot_count,
            source_chain.tensor_allocations.front().slot_count);
  ASSERT_EQ(baseline.tensor_device_resources.size(), 1U);
  ASSERT_EQ(source_chain.tensor_device_resources.size(), 1U);
  EXPECT_EQ(baseline.tensor_device_resources.front(), source_chain.tensor_device_resources.front());
  ASSERT_EQ(baseline.cuda_compute_streams.size(), 1U);
  ASSERT_EQ(source_chain.cuda_compute_streams.size(), 1U);
  EXPECT_EQ(baseline.cuda_compute_streams.front(), source_chain.cuda_compute_streams.front());
  EXPECT_FALSE(baseline.tensor_device_resources.front().gpu_resident_backend_selected);
}

}  // namespace
}  // namespace holoscan::detail
// NOLINTEND(bugprone-unchecked-optional-access)
