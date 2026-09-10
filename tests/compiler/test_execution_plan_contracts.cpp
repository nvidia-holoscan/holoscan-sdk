/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "compiler/channel_payload_contracts.hpp"
#include "compiler/execution_plan_contracts.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] PayloadIdentityRecord tensor_identity() {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kTensor,
      .payload_kind = PayloadKind::kDescriptorBacked,
  };
}

[[nodiscard]] PayloadIdentityRecord descriptor_schema_identity() {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kSchema,
      .payload_kind = PayloadKind::kDescriptorBacked,
  };
}

[[nodiscard]] ProcessLocalChannelServicePlanRecord local_service(bool descriptor_schema) {
  return ProcessLocalChannelServicePlanRecord{
      .payload_identity = descriptor_schema ? descriptor_schema_identity() : tensor_identity(),
      .tensor_payload = true,
  };
}

[[nodiscard]] CrossProcessChannelServicePlanRecord cross_process_service(bool descriptor_schema) {
  return CrossProcessChannelServicePlanRecord{
      .payload_identity = descriptor_schema ? descriptor_schema_identity() : tensor_identity(),
      .tensor_payload = true,
      .source_process_group = ProcessGroupId{1U},
      .target_process_group = ProcessGroupId{2U},
      .target_process_groups = {ProcessGroupId{2U}},
  };
}

TEST(ExecutionPlanContracts, ProfileSemanticRankCoversClosedPolicyOrder) {
  constexpr std::array kProfiles{
      Profile::kResearch,
      Profile::kStandard,
      Profile::kRobotics,
      Profile::kRealTime,
      Profile::kRegulated,
  };

  // Key setup: evaluate every profile in policy dominance order plus one invalid enum value.
  for (std::size_t index = 0U; index < kProfiles.size(); ++index) {
    const auto rank = profile_semantic_rank(kProfiles[index]);
    ASSERT_TRUE(rank);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
    EXPECT_EQ(*rank, index);
  }

  // Key assertion: the shared mapping is independent of enum storage and fails closed outside
  // the active profile policy.
  // The out-of-range value exercises fail-closed validation.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_FALSE(profile_semantic_rank(static_cast<Profile>(255U)));
}

TEST(ExecutionPlanContracts, ExternalTopicBoundsRespectEndpointDirection) {
  const ExternalTopicRecord imported{
      .imported = true,
      .bounds =
          EndpointBounds{
              .publisher = PublisherEndpointBounds{.max_loaned_samples = 7U},
              .subscriber =
                  SubscriberEndpointBounds{
                      .queue_depth = 97U,
                      .max_borrowed_samples = 98U,
                  },
          },
  };
  const ExternalTopicRecord exported{
      .imported = false,
      .bounds =
          EndpointBounds{
              .publisher = PublisherEndpointBounds{.max_loaned_samples = 99U},
              .subscriber =
                  SubscriberEndpointBounds{
                      .queue_depth = 11U,
                      .max_borrowed_samples = std::nullopt,
                  },
          },
  };

  // Key setup: resolve imported and exported topics with authored values on both endpoint sides.
  const auto imported_bounds = resolve_external_topic_bounds(imported, 3U, 5U);
  const auto exported_bounds = resolve_external_topic_bounds(exported, 3U, 5U);

  // Key assertion: only the direction-owned authored bounds override Graph fallbacks, and an
  // omitted export borrow bound follows the resolved subscriber queue.
  EXPECT_EQ(imported_bounds.publisher_max_loaned_samples, 7U);
  EXPECT_EQ(imported_bounds.subscriber_queue_depth, 3U);
  EXPECT_EQ(imported_bounds.subscriber_max_borrowed_samples, 3U);
  EXPECT_EQ(exported_bounds.publisher_max_loaned_samples, 5U);
  EXPECT_EQ(exported_bounds.subscriber_queue_depth, 11U);
  EXPECT_EQ(exported_bounds.subscriber_max_borrowed_samples, 11U);
}

TEST(ExecutionPlanContracts, ProviderSizingDependenciesRejectMalformedRanges) {
  std::array dependencies{
      PoolCapacityDependencyPlanRecord{
          .id = PoolCapacityDependencyId{1U},
          .pool_id = "pool-a",
      },
      PoolCapacityDependencyPlanRecord{
          .id = PoolCapacityDependencyId{2U},
          .pool_id = "pool-a",
      },
  };
  ProviderAllocationPlanRecord provider{
      .pool_id = "pool-a",
      .first_sizing_dependency = PoolCapacityDependencyId{1U},
      .sizing_dependency_count = 2U,
  };

  // Key setup: resolve one valid provider range, then mutate each structural range authority.
  const auto valid = provider_sizing_dependencies(dependencies, provider);
  ASSERT_TRUE(valid);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(valid->data(), dependencies.data());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(valid->size(), dependencies.size());

  // Key assertion: absent, empty, out-of-range, noncontiguous, and cross-pool ranges all fail
  // closed rather than returning a partial dependency projection.
  provider.first_sizing_dependency = {};
  EXPECT_FALSE(provider_sizing_dependencies(dependencies, provider));
  provider.first_sizing_dependency = PoolCapacityDependencyId{1U};
  provider.sizing_dependency_count = 0U;
  EXPECT_FALSE(provider_sizing_dependencies(dependencies, provider));
  provider.sizing_dependency_count = 3U;
  EXPECT_FALSE(provider_sizing_dependencies(dependencies, provider));
  provider.sizing_dependency_count = 2U;
  dependencies[1].id = PoolCapacityDependencyId{3U};
  EXPECT_FALSE(provider_sizing_dependencies(dependencies, provider));
  dependencies[1].id = PoolCapacityDependencyId{2U};
  dependencies[1].pool_id = "pool-b";
  EXPECT_FALSE(provider_sizing_dependencies(dependencies, provider));
}

TEST(ExecutionPlanContracts, DerivesEveryBoundedChannelPayloadSizeFromOneContract) {
  struct ScalarExpectation {
    ScalarIdentityKind kind;
    std::optional<std::uint64_t> size;
  };
  const std::array scalar_expectations{
      ScalarExpectation{ScalarIdentityKind::kBoolean, sizeof(bool)},
      ScalarExpectation{ScalarIdentityKind::kInt8, 1U},
      ScalarExpectation{ScalarIdentityKind::kUInt8, 1U},
      ScalarExpectation{ScalarIdentityKind::kInt16, 2U},
      ScalarExpectation{ScalarIdentityKind::kUInt16, 2U},
      ScalarExpectation{ScalarIdentityKind::kInt32, 4U},
      ScalarExpectation{ScalarIdentityKind::kUInt32, 4U},
      ScalarExpectation{ScalarIdentityKind::kFloat32, 4U},
      ScalarExpectation{ScalarIdentityKind::kInt64, 8U},
      ScalarExpectation{ScalarIdentityKind::kUInt64, 8U},
      ScalarExpectation{ScalarIdentityKind::kFloat64, 8U},
      ScalarExpectation{ScalarIdentityKind::kLocalEnum, std::nullopt},
  };

  // Key setup: evaluate every scalar identity and the POD, serialized-schema,
  // descriptor-schema, invalid-schema, and overflow forms used by Channel lowering.
  for (const auto& expectation : scalar_expectations) {
    const PayloadIdentityRecord scalar{
        .kind = PayloadIdentityKind::kScalar,
        .payload_kind = PayloadKind::kInlineTyped,
        .scalar = ScalarIdentityRecord{.kind = expectation.kind},
    };
    EXPECT_EQ(channel_payload_size_bytes(scalar, 0U, true), expectation.size);
  }
  const PayloadIdentityRecord pod{
      .kind = PayloadIdentityKind::kPod,
      .payload_kind = PayloadKind::kInlineTyped,
      .pod = PodIdentityRecord{.object_size = 23U},
  };
  const PayloadIdentityRecord serialized_schema{
      .kind = PayloadIdentityKind::kSchema,
      .payload_kind = PayloadKind::kSerializedBytes,
      .schema = SchemaIdentityRecord{},
  };
  PayloadIdentityRecord descriptor_schema = serialized_schema;
  descriptor_schema.payload_kind = PayloadKind::kDescriptorBacked;
  PayloadIdentityRecord invalid_schema = serialized_schema;
  invalid_schema.payload_kind = PayloadKind::kInlineTyped;

  // Key assertion: native scalar/POD widths, schema-envelope overhead, and
  // overflow/local-enum rejection have one exact producer/validator result.
  EXPECT_EQ(channel_payload_size_bytes(pod, 0U, true), 23U);
  EXPECT_EQ(channel_payload_size_bytes(serialized_schema, 41U, false), 41U);
  EXPECT_EQ(channel_payload_size_bytes(serialized_schema, 41U, true),
            41U + kSchemaEnvelopeHeaderSize);
  EXPECT_EQ(channel_payload_size_bytes(descriptor_schema, 41U, true), 41U);
  EXPECT_FALSE(channel_payload_size_bytes(invalid_schema, 41U, true));
  EXPECT_FALSE(channel_payload_size_bytes(serialized_schema, 0U, true));
  EXPECT_FALSE(channel_payload_size_bytes(
      serialized_schema, std::numeric_limits<std::uint64_t>::max(), true));
}

TEST(ExecutionPlanContracts, ServiceViewProjectsCommonFactsWithoutOwningStorage) {
  constexpr DeviceId kFirstDevice{3};
  constexpr DeviceId kSecondDevice{5};
  std::vector<TensorFieldServiceStorageContract> fields{
      TensorFieldServiceStorageContract{
          .field_id = 7U,
          .tensor_index = 0U,
          .max_byte_span = 4096U,
          .pool_id = "field-0",
          .memory_kind = MemoryKind::kPinnedHost,
          .device = kFirstDevice,
      },
      TensorFieldServiceStorageContract{
          .field_id = 8U,
          .tensor_index = 1U,
          .max_byte_span = 8192U,
          .pool_id = "field-1",
          .memory_kind = MemoryKind::kCudaDevice,
          .device = kSecondDevice,
      },
  };
  auto contract = make_tensor_service_contract(tensor_identity(),
                                               QueuePolicy::kDropOldest,
                                               4U,
                                               2U,
                                               true,
                                               TensorServiceStorageContract{
                                                   .payload_size_bytes = 128U,
                                                   .tensor_fields = std::move(fields),
                                               });
  contract.publisher_history_depth = 3U;
  const auto local = make_process_local_service_record(
      PortId{11U}, "local", contract, 2U, true, ProcessLocalChannelDeliveryKind::kHoloMQ);
  auto cross = make_cross_process_service_record(EdgeId{13U},
                                                 PortId{11U},
                                                 PortId{12U},
                                                 "cross",
                                                 contract,
                                                 2U,
                                                 ProcessGroupId{1U},
                                                 ProcessGroupId{2U});

  // Key setup: project the owning, process-local, and cross-process record families and visit the
  // effective per-field storage through their one shared non-owning vocabulary.
  const auto contract_view = tensor_service_contract_view(contract);
  const auto local_view = tensor_service_contract_view(local);
  const auto cross_view = tensor_service_contract_view(cross);
  std::vector<std::pair<MemoryKind, DeviceId>> visited;
  for_each_tensor_service_storage(local_view, [&](TensorServiceStorageView storage) {
    visited.emplace_back(storage.memory_kind, storage.device);
  });

  // Key assertion: projections retain common payload, queue, and borrowed storage facts while the
  // exact storage predicate detects a nested field mutation.
  EXPECT_EQ(local_view.payload_identity.kind, contract_view.payload_identity.kind);
  EXPECT_EQ(local_view.payload_size_bytes, cross_view.payload_size_bytes);
  EXPECT_EQ(local_view.queue_policy, cross_view.queue_policy);
  EXPECT_EQ(local_view.max_subscribers, cross_view.max_subscribers);
  EXPECT_EQ(local_view.max_subscriber_queue_depth, cross_view.max_subscriber_queue_depth);
  EXPECT_EQ(local_view.max_subscriber_borrowed_samples, cross_view.max_subscriber_borrowed_samples);
  EXPECT_EQ(local_view.publisher_history_depth, cross_view.publisher_history_depth);
  EXPECT_EQ(local_view.event_wake, cross_view.event_wake);
  EXPECT_EQ(local_view.tensor_fields.data(), local.tensor_fields.data());
  ASSERT_EQ(visited.size(), 2U);
  EXPECT_EQ(visited[0], std::make_pair(MemoryKind::kPinnedHost, kFirstDevice));
  EXPECT_EQ(visited[1], std::make_pair(MemoryKind::kCudaDevice, kSecondDevice));
  EXPECT_TRUE(tensor_service_payload_storage_exactly_equal(local_view, cross_view));
  cross.tensor_fields.back().device = DeviceId{6};
  EXPECT_FALSE(tensor_service_payload_storage_exactly_equal(local_view,
                                                            tensor_service_contract_view(cross)));
}

// Service transfer ordering and Operator CUDA access are separate obligations. This test keeps
// their complete memory-kind truth table at the shared contract boundary.
//
// REQ: CHAN-093
// REQ: CHAN-094
// REQ: CHAN-135
TEST(ExecutionPlanContracts, StorageStreamRequirementsCoverEveryMemoryKindAndServiceShape) {
  struct Expectation {
    MemoryKind memory_kind;
    bool ordinary_service_stream;
    bool ordinary_operator_stream;
    bool descriptor_service_stream;
    bool descriptor_operator_stream;
  };
  constexpr std::array kExpectations{
      Expectation{MemoryKind::kUnknown, false, false, false, false},
      Expectation{MemoryKind::kHost, false, false, false, false},
      Expectation{MemoryKind::kPinnedHost, true, true, false, false},
      Expectation{MemoryKind::kCudaDevice, true, true, true, true},
      Expectation{MemoryKind::kCudaManaged, false, true, false, true},
  };
  const auto local_ordinary = local_service(false);
  const auto local_descriptor = local_service(true);
  const auto cross_ordinary = cross_process_service(false);
  const auto cross_descriptor = cross_process_service(true);

  // Key setup: evaluate both ownership rules for every MemoryKind, both service record families,
  // and both relevant payload shapes from one table.
  for (const auto& expected : kExpectations) {
    SCOPED_TRACE(static_cast<std::uint32_t>(expected.memory_kind));
    EXPECT_EQ(tensor_service_storage_requires_stream(local_ordinary, expected.memory_kind),
              expected.ordinary_service_stream);
    EXPECT_EQ(
        tensor_service_storage_requires_operator_cuda_stream(local_ordinary, expected.memory_kind),
        expected.ordinary_operator_stream);
    EXPECT_EQ(tensor_service_storage_requires_stream(local_descriptor, expected.memory_kind),
              expected.descriptor_service_stream);
    EXPECT_EQ(tensor_service_storage_requires_operator_cuda_stream(local_descriptor,
                                                                   expected.memory_kind),
              expected.descriptor_operator_stream);

    // Key assertion: cross-process records obey the same storage rules without deriving either
    // ownership predicate from the other.
    EXPECT_EQ(tensor_service_storage_requires_stream(cross_ordinary, expected.memory_kind),
              expected.ordinary_service_stream);
    EXPECT_EQ(
        tensor_service_storage_requires_operator_cuda_stream(cross_ordinary, expected.memory_kind),
        expected.ordinary_operator_stream);
    EXPECT_EQ(tensor_service_storage_requires_stream(cross_descriptor, expected.memory_kind),
              expected.descriptor_service_stream);
    EXPECT_EQ(tensor_service_storage_requires_operator_cuda_stream(cross_descriptor,
                                                                   expected.memory_kind),
              expected.descriptor_operator_stream);
  }
}

// Process-local stream planning accepts only stream-ordered storage on the exact device. Field
// contracts replace, rather than augment, the scalar service placement.
TEST(ExecutionPlanContracts, ProcessLocalStreamRequirementUsesExactScalarOrFieldDevice) {
  constexpr DeviceId kFirstDevice{1};
  constexpr DeviceId kSecondDevice{2};
  constexpr DeviceId kManagedDevice{3};
  constexpr DeviceId kUnusedDevice{9};
  constexpr DeviceId kInvalidDevice{};
  auto ordinary = local_service(false);
  ordinary.tensor_memory_kind = MemoryKind::kPinnedHost;
  ordinary.tensor_device = kFirstDevice;

  // Key setup: begin with scalar pinned-host storage, then replace it with independent pinned,
  // device, managed, and host field placements.
  EXPECT_TRUE(process_local_tensor_service_requires_stream_for_device(ordinary, kFirstDevice));
  EXPECT_FALSE(process_local_tensor_service_requires_stream_for_device(ordinary, kSecondDevice));
  EXPECT_FALSE(process_local_tensor_service_requires_stream_for_device(ordinary, kInvalidDevice));
  ordinary.tensor_fields = {
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kPinnedHost,
                                        .device = kFirstDevice},
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kCudaDevice,
                                        .device = kSecondDevice},
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kCudaManaged,
                                        .device = kManagedDevice},
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kHost},
  };

  // Key assertion: only transfer-stream storage on an exact valid field device is selected;
  // managed storage still requires an Operator stream but not a service transfer stream.
  EXPECT_TRUE(process_local_tensor_service_requires_stream_for_device(ordinary, kFirstDevice));
  EXPECT_TRUE(process_local_tensor_service_requires_stream_for_device(ordinary, kSecondDevice));
  EXPECT_FALSE(process_local_tensor_service_requires_stream_for_device(ordinary, kManagedDevice));
  EXPECT_FALSE(process_local_tensor_service_requires_stream_for_device(ordinary, kUnusedDevice));

  auto descriptor = ordinary;
  descriptor.payload_identity = descriptor_schema_identity();
  EXPECT_FALSE(process_local_tensor_service_requires_stream_for_device(descriptor, kFirstDevice));
  EXPECT_TRUE(process_local_tensor_service_requires_stream_for_device(descriptor, kSecondDevice));
  descriptor.tensor_payload = false;
  EXPECT_FALSE(process_local_tensor_service_requires_stream_for_device(descriptor, kSecondDevice));
}

// Cross-process stream planning is scoped by both the participating group and exact device.
// Descriptor schemas additionally make their source group a participant.
TEST(ExecutionPlanContracts, CrossProcessStreamRequirementUsesGroupAndExactDevice) {
  constexpr ProcessGroupId kSourceGroup{1U};
  constexpr ProcessGroupId kTargetGroup{2U};
  constexpr ProcessGroupId kOtherGroup{3U};
  constexpr DeviceId kPinnedDevice{4};
  constexpr DeviceId kCUDADevice{5};
  constexpr DeviceId kManagedDevice{6};
  constexpr DeviceId kInvalidDevice{};
  auto ordinary = cross_process_service(false);
  ordinary.tensor_fields = {
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kPinnedHost,
                                        .device = kPinnedDevice},
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kCudaDevice,
                                        .device = kCUDADevice},
      TensorFieldServiceStorageContract{.memory_kind = MemoryKind::kCudaManaged,
                                        .device = kManagedDevice},
  };

  // Key setup: the ordinary service has three field memory kinds, one target group, and a source
  // group that is not a transfer participant for ordinary Tensor payloads.
  EXPECT_FALSE(cross_process_tensor_service_requires_stream_in_group(ordinary, kSourceGroup));
  EXPECT_TRUE(cross_process_tensor_service_requires_stream_in_group(ordinary, kTargetGroup));
  EXPECT_FALSE(cross_process_tensor_service_requires_stream_in_group(ordinary, kOtherGroup));
  EXPECT_TRUE(cross_process_tensor_service_requires_stream_for_device_in_group(
      ordinary, kTargetGroup, kPinnedDevice));
  EXPECT_TRUE(cross_process_tensor_service_requires_stream_for_device_in_group(
      ordinary, kTargetGroup, kCUDADevice));
  EXPECT_FALSE(cross_process_tensor_service_requires_stream_for_device_in_group(
      ordinary, kTargetGroup, kManagedDevice));
  EXPECT_FALSE(cross_process_tensor_service_requires_stream_for_device_in_group(
      ordinary, kTargetGroup, kInvalidDevice));

  auto descriptor = ordinary;
  descriptor.payload_identity = descriptor_schema_identity();

  // Key assertion: descriptor-schema source and target groups require only the CUDA-device field;
  // pinned-host remains host-ordered and a nonparticipant never acquires a stream.
  for (const auto group : {kSourceGroup, kTargetGroup}) {
    EXPECT_TRUE(cross_process_tensor_service_requires_stream_in_group(descriptor, group));
    EXPECT_FALSE(cross_process_tensor_service_requires_stream_for_device_in_group(
        descriptor, group, kPinnedDevice));
    EXPECT_TRUE(cross_process_tensor_service_requires_stream_for_device_in_group(
        descriptor, group, kCUDADevice));
  }
  EXPECT_FALSE(cross_process_tensor_service_requires_stream_in_group(descriptor, kOtherGroup));
  EXPECT_FALSE(cross_process_tensor_service_requires_stream_for_device_in_group(
      descriptor, kOtherGroup, kCUDADevice));
}

}  // namespace
}  // namespace holoscan::detail
