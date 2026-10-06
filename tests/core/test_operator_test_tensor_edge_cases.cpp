/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "../common/cuda_test_utils.hpp"
#include "core/domain/tensor_access.hpp"

namespace {
namespace test_detail = holoscan::testing::detail;

// Own metadata while leaving any non-null backing with this test's caller. Invalid-layout cases
// must be rejected before accessing that backing; they never allocate their declared huge spans.
holoscan::Tensor tensor_view(void* data, std::vector<std::int64_t> shape,
                             DLDataType dtype = {kDLUInt, 8, 1}, DLDevice device = {kDLCPU, 0},
                             std::optional<std::vector<std::int64_t>> strides = std::nullopt,
                             std::uint64_t byte_offset = 0U) {
  struct Owner {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::optional<std::vector<std::int64_t>> strides;
  };
  auto owner = std::make_unique<Owner>();
  owner->shape = std::move(shape);
  owner->strides = std::move(strides);
  auto& owned_strides = owner->strides;
  owner->managed.dl_tensor = {.data = data,
                              .device = device,
                              .ndim = static_cast<std::int32_t>(owner->shape.size()),
                              .dtype = dtype,
                              .shape = owner->shape.data(),
                              .strides = owned_strides ? owned_strides->data() : nullptr,
                              .byte_offset = byte_offset};
  owner->managed.manager_ctx = owner.get();
  owner->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Owner*>(value->manager_ctx);
  };
  return holoscan::Tensor{&owner.release()->managed};
}

class EmptyTensorPass final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").expects_tensor(
        holoscan::TensorInputSpec{.representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                                     .dtype = DLDataType{kDLUInt, 8, 1},
                                                     .rank = 3U}});
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 3U},
            .bounds = holoscan::tensor_bounds(64U)});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return out.emit(std::move(*sample));
  }
  holoscan::Input<holoscan::Tensor> in;
  holoscan::Output<holoscan::Tensor> out;
};

void expect_invalid_tensor_rejected(const holoscan::Tensor& tensor) {
  const std::array<const holoscan::Tensor*, 1> values{&tensor};
  EXPECT_THROW(static_cast<void>(test_detail::derive_tensor_source_spec(values)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(test_detail::acquire_staged_tensor_leases(values)),
               std::invalid_argument);
  // A zero byte bound makes this probe safe even if a malformed huge span escapes validation.
  const auto snapshot = test_detail::capture_tensor_snapshot(tensor, {}, 0U, nullptr);
  ASSERT_FALSE(snapshot);
  EXPECT_EQ(snapshot.error().code, holoscan::ErrorCode::kNotSupported);
}

class OperatorTestTensorCudaEdges : public ::testing::Test {
 protected:
  void SetUp() override {
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
      GTEST_SKIP() << "CUDA device unavailable";
    ASSERT_EQ(cudaGetDevice(&previous_device_), cudaSuccess);
    device_ = count > 1 ? 1 : 0;
    ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
  }
  void TearDown() override {
    if (previous_device_ >= 0)
      EXPECT_EQ(cudaSetDevice(previous_device_), cudaSuccess);
  }
  int device_{};

 private:
  int previous_device_{-1};
};

class OperatorTestTensorCudaCopies
    : public OperatorTestTensorCudaEdges,
      public ::testing::WithParamInterface<holoscan::MemoryStorageType> {};
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, NullBackedEmptySnapshotOwnsCompleteMetadataAtZeroByteBound) {
  auto tensor = tensor_view(nullptr,
                            {0},
                            DLDataType{kDLUInt, 16, 1},
                            DLDevice{kDLCPU, 0},
                            std::vector<std::int64_t>{1},
                            7U);
  const holoscan::SampleMetadata metadata{.sequence = 11U,
                                          .capture_timestamp_ns = 123,
                                          .source_clock_id = holoscan::ClockId{2},
                                          .publish_timestamp_ns = 456,
                                          .publish_clock_id = holoscan::ClockId{3},
                                          .trace_id = 17U,
                                          .frame_id = 19U,
                                          .byte_size = 0U,
                                          .payload_kind = holoscan::PayloadKind::kDescriptorBacked,
                                          .flags = holoscan::SampleFlags::kDegraded};
  const auto result = test_detail::capture_tensor_snapshot(tensor, metadata, 0U, nullptr);
  ASSERT_TRUE(result) << result.error().message;
  tensor = {};
  EXPECT_TRUE(result->bytes().empty());
  EXPECT_EQ(result->shape(), (std::vector<std::int64_t>{0}));
  EXPECT_EQ(result->source_byte_strides(), (std::vector<std::int64_t>{2}));
  EXPECT_EQ(result->source_byte_offset(), 7U);
  EXPECT_EQ(result->source_device().device_type, kDLCPU);
  EXPECT_EQ(result->source_device().device_id, 0);
  EXPECT_EQ(result->dtype().code, kDLUInt);
  EXPECT_EQ(result->dtype().bits, 16U);
  EXPECT_EQ(result->dtype().lanes, 1U);
  const auto& saved = result->sample_metadata();
  EXPECT_EQ(saved.version.major, metadata.version.major);
  EXPECT_EQ(saved.version.minor, metadata.version.minor);
  EXPECT_EQ(saved.sequence, metadata.sequence);
  EXPECT_EQ(saved.capture_timestamp_ns, metadata.capture_timestamp_ns);
  EXPECT_EQ(saved.source_clock_id, metadata.source_clock_id);
  EXPECT_EQ(saved.publish_timestamp_ns, metadata.publish_timestamp_ns);
  EXPECT_EQ(saved.publish_clock_id, metadata.publish_clock_id);
  EXPECT_EQ(saved.trace_id, metadata.trace_id);
  EXPECT_EQ(saved.frame_id, metadata.frame_id);
  EXPECT_EQ(saved.byte_size, metadata.byte_size);
  EXPECT_EQ(saved.payload_kind, metadata.payload_kind);
  EXPECT_EQ(saved.flags, metadata.flags);
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, EmptyShapesKeepEachZeroAxisAndRepresentableByteStride) {
  const std::array<std::vector<std::int64_t>, 3> shapes{{{0, 2, 3}, {2, 0, 3}, {2, 3, 0}}};
  const std::array<std::vector<std::int64_t>, 3> strides{{{24, 12, 4}, {0, 12, 4}, {0, 0, 4}}};
  for (std::size_t index = 0U; index < shapes.size(); ++index) {
    SCOPED_TRACE(index);
    const auto tensor = tensor_view(nullptr, shapes[index], DLDataType{kDLInt, 32, 1});
    const std::array<const holoscan::Tensor*, 1> values{&tensor};
    const auto source = test_detail::derive_tensor_source_spec(values);
    EXPECT_EQ(source.maximum_shape, shapes[index]);
    if (const auto& bounds = source.value.bounds; bounds.has_value()) {
      EXPECT_GT(bounds->max_byte_span, 0U);
    } else {
      FAIL() << "derived Tensor source has no bounds";
    }
    EXPECT_EQ(source.value.representation.strides, holoscan::TensorStridePolicy::kContiguousOnly);
    EXPECT_NO_THROW(static_cast<void>(test_detail::acquire_staged_tensor_leases(values)));
    const auto result = test_detail::capture_tensor_snapshot(tensor, {}, 0U, nullptr);
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_TRUE(result->bytes().empty());
    EXPECT_EQ(result->shape(), shapes[index]);
    EXPECT_EQ(result->source_byte_strides(), strides[index]);
  }
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, MultidimensionalEmptyStagingPublishesEveryRecord) {
  using namespace std::chrono_literals;
  const std::array<std::vector<std::int64_t>, 3> shapes{{{0, 2, 3}, {2, 0, 3}, {2, 3, 0}}};
  for (const auto& shape : shapes) {
    SCOPED_TRACE(::testing::PrintToString(shape));
    auto harness = holoscan::testing::create_operator_test<EmptyTensorPass>();
    harness.push_samples("in",
                         std::vector<holoscan::testing::TestSample<holoscan::Tensor>>{
                             holoscan::testing::sample_at(tensor_view(nullptr, shape), 10ms),
                             holoscan::testing::sample_at(tensor_view(nullptr, shape), 20ms)});
    harness.capture_snapshot<holoscan::Tensor>("out");
    const auto result = harness.run();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto records = result.snapshots<holoscan::Tensor>("out");
    ASSERT_EQ(records.size(), 2U);
    for (const auto& record : records) {
      EXPECT_TRUE(record.bytes().empty());
      EXPECT_EQ(record.shape(), shape);
      EXPECT_EQ(record.source_device().device_type, kDLCPU);
      EXPECT_EQ(record.dtype().code, kDLUInt);
      EXPECT_EQ(record.dtype().bits, 8U);
    }
    EXPECT_EQ(records[0].sample_metadata().capture_timestamp_ns, 10000000);
    EXPECT_EQ(records[1].sample_metadata().capture_timestamp_ns, 20000000);
  }
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, EmptyTensorWithBackingStillHoldsAStagingReadLease) {
  const std::array<std::int64_t, 1> shape{0};
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{kDLUInt, 8, 1},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kSystem,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  ASSERT_TRUE(tensor) << tensor.error().message;
  ASSERT_NE(tensor->data(), nullptr);
  ASSERT_TRUE(tensor->mutable_data());
  const std::array<const holoscan::Tensor*, 1> values{&*tensor};
  auto leases = test_detail::acquire_staged_tensor_leases(values);
  const auto blocked = tensor->mutable_data();
  ASSERT_FALSE(blocked);
  EXPECT_EQ(blocked.error().code, holoscan::ErrorCode::kPermissionDenied);
  leases.reset();
  EXPECT_TRUE(tensor->mutable_data());
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, EmptyDeviceMetadataNeedsNoCudaWorkButStillChecksPlacement) {
  for (const auto type : {kDLCUDA, kDLCUDAManaged}) {
    SCOPED_TRACE(type);
    const auto tensor =
        tensor_view(nullptr, {2, 0, 3}, DLDataType{kDLUInt, 8, 1}, DLDevice{type, 0});
    const auto snapshot = test_detail::capture_tensor_snapshot(tensor, {}, 0U, nullptr);
    ASSERT_TRUE(snapshot) << snapshot.error().message;
    EXPECT_TRUE(snapshot->bytes().empty());
    EXPECT_EQ(snapshot->source_device().device_type, type);
    const auto copied = test_detail::copy_staged_tensor_to_device(
        nullptr, 0U, holoscan::DeviceId{0}, tensor, ::cuda::stream_ref{cudaStream_t{nullptr}});
    ASSERT_TRUE(copied) << copied.error().message;
    const auto wrong_device = test_detail::copy_staged_tensor_to_device(
        nullptr, 0U, holoscan::DeviceId{1}, tensor, ::cuda::stream_ref{cudaStream_t{nullptr}});
    ASSERT_FALSE(wrong_device);
    EXPECT_EQ(wrong_device.error().code, holoscan::ErrorCode::kIncompatible);
  }
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, DefaultNullAndMalformedLayoutsDoNotBecomeEmptyRecords) {
  std::array<std::byte, 8> backing{};
  expect_invalid_tensor_rejected(holoscan::Tensor{});
  expect_invalid_tensor_rejected(tensor_view(nullptr, {1}));
  expect_invalid_tensor_rejected(tensor_view(nullptr, {0}, DLDataType{kDLUInt, 0, 1}));
  expect_invalid_tensor_rejected(tensor_view(backing.data(), {-1}));
  expect_invalid_tensor_rejected(tensor_view(backing.data(),
                                             {2, 2},
                                             DLDataType{kDLUInt, 8, 1},
                                             DLDevice{kDLCPU, 0},
                                             std::vector<std::int64_t>{1, 1}));
  expect_invalid_tensor_rejected(tensor_view(nullptr,
                                             {0, 2},
                                             DLDataType{kDLUInt, 8, 1},
                                             DLDevice{kDLCPU, 0},
                                             std::vector<std::int64_t>{2, -1}));
  const std::array<const holoscan::Tensor*, 1> null_value{nullptr};
  EXPECT_THROW(static_cast<void>(test_detail::derive_tensor_source_spec(null_value)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(test_detail::acquire_staged_tensor_leases(null_value)),
               std::invalid_argument);
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, ArithmeticAndLogicalPointerOverflowRemainColdRejections) {
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  std::array<std::byte, 8> backing{};
  expect_invalid_tensor_rejected(tensor_view(backing.data(), {maximum, 3}));
  const auto public_size_overflow =
      tensor_view(backing.data(), {maximum}, DLDataType{kDLUInt, 16, 1});
  ASSERT_TRUE(holoscan::detail::TensorAccess::snapshot(public_size_overflow).valid);
  ASSERT_EQ(public_size_overflow.nbytes(), 0);
  expect_invalid_tensor_rejected(public_size_overflow);
  const auto logical_pointer_overflow =
      tensor_view(reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1U),
                  {1},
                  DLDataType{kDLUInt, 8, 1},
                  DLDevice{kDLCPU, 0},
                  std::nullopt,
                  4U);
  ASSERT_TRUE(holoscan::detail::TensorAccess::snapshot(logical_pointer_overflow).valid);
  ASSERT_EQ(logical_pointer_overflow.data(), nullptr);
  expect_invalid_tensor_rejected(logical_pointer_overflow);
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, EmptySnapshotRejectsByteStridesThatCannotBeRepresented) {
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  const std::array tensors{tensor_view(nullptr, {0, maximum, 2}),
                           tensor_view(nullptr,
                                       {0},
                                       DLDataType{kDLUInt, 16, 1},
                                       DLDevice{kDLCPU, 0},
                                       std::vector<std::int64_t>{maximum})};
  for (const auto& tensor : tensors) {
    ASSERT_TRUE(holoscan::detail::TensorAccess::snapshot(tensor).valid);
    ASSERT_EQ(tensor.nbytes(), 0);
    const auto result = test_detail::capture_tensor_snapshot(tensor, {}, 0U, nullptr);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, holoscan::ErrorCode::kNotSupported);
  }
  // A boundary value that fits int64_t is retained exactly, without multiplying an unused volume.
  const auto representable = tensor_view(nullptr, {0, maximum});
  const auto result = test_detail::capture_tensor_snapshot(representable, {}, 0U, nullptr);
  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(result->source_byte_strides(), (std::vector<std::int64_t>{maximum, 1}));
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, EmptySourceRejectsOverflowingPhysicalCapacity) {
  const auto tensor = tensor_view(nullptr, {0, std::numeric_limits<std::int64_t>::max(), 3});
  ASSERT_TRUE(holoscan::detail::TensorAccess::snapshot(tensor).valid);
  const std::array<const holoscan::Tensor*, 1> values{&tensor};
  EXPECT_THROW(static_cast<void>(test_detail::derive_tensor_source_spec(values)),
               std::invalid_argument);
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, NonemptyStridedStagingAndSnapshotsRemainUnsupported) {
  std::array<std::byte, 8> backing{};
  const auto tensor = tensor_view(backing.data(),
                                  {2, 2},
                                  DLDataType{kDLUInt, 8, 1},
                                  DLDevice{kDLCPU, 0},
                                  std::vector<std::int64_t>{3, 1});
  ASSERT_TRUE(holoscan::detail::TensorAccess::snapshot(tensor).valid);
  ASSERT_FALSE(tensor.is_contiguous());
  const std::array<const holoscan::Tensor*, 1> values{&tensor};
  EXPECT_THROW(static_cast<void>(test_detail::derive_tensor_source_spec(values)),
               std::invalid_argument);
  const auto result = test_detail::capture_tensor_snapshot(tensor, {}, backing.size(), nullptr);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, holoscan::ErrorCode::kNotSupported);
}

// REQ: OPERATOR-073
TEST(OperatorTestTensorEdges, DeviceCopyRejectsCpuStorageBeforeReadinessOrCopy) {
  std::array<std::byte, 4> source{};
  std::array<std::byte, 4> destination{};
  const auto tensor = tensor_view(source.data(), {4});
  // Both ordinal fields are zero: storage type, rather than ordinal equality, must reject this.
  const auto copied =
      test_detail::copy_staged_tensor_to_device(destination.data(),
                                                destination.size(),
                                                holoscan::DeviceId{0},
                                                tensor,
                                                ::cuda::stream_ref{cudaStream_t{nullptr}});
  ASSERT_FALSE(copied);
  EXPECT_EQ(copied.error().code, holoscan::ErrorCode::kIncompatible);
}

// REQ: OPERATOR-073
TEST_P(OperatorTestTensorCudaCopies, SameDeviceCopiesAndSnapshotsPreserveBytesAndPlacement) {
  if (GetParam() == holoscan::MemoryStorageType::kCudaManaged) {
    // Core assigns DLPack managed tensors canonical ordinal zero. Qualify that existing profile.
    device_ = 0;
    ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
    int supported{};
    ASSERT_EQ(cudaDeviceGetAttribute(&supported, cudaDevAttrManagedMemory, device_), cudaSuccess);
    if (supported == 0)
      GTEST_SKIP() << "CUDA managed memory unavailable";
  }
  holoscan::test::UniqueCudaStream producer;
  holoscan::test::UniqueCudaStream consumer;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(producer), cudaSuccess);
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(consumer), cudaSuccess);
  const std::array<std::int64_t, 1> shape{4};
  const std::array<std::byte, 4> bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  const auto allocator = std::make_shared<holoscan::UnboundedAllocator>();
  auto source = holoscan::make_tensor(
      shape, DLDataType{kDLUInt, 8, 1}, std::nullopt, GetParam(), allocator, producer.get());
  ASSERT_TRUE(source) << source.error().message;
  auto destination = holoscan::make_tensor(shape,
                                           DLDataType{kDLUInt, 8, 1},
                                           std::nullopt,
                                           holoscan::MemoryStorageType::kDevice,
                                           allocator,
                                           consumer.get());
  ASSERT_TRUE(destination) << destination.error().message;
  auto source_pointer = source->mutable_data();
  auto destination_pointer = destination->mutable_data();
  ASSERT_TRUE(source_pointer) << source_pointer.error().message;
  ASSERT_TRUE(destination_pointer) << destination_pointer.error().message;
  ASSERT_EQ(
      cudaMemcpyAsync(
          *source_pointer, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, producer.get()),
      cudaSuccess);
  ASSERT_TRUE(source->set_readiness_stream(producer.get()));
  const auto copied = test_detail::copy_staged_tensor_to_device(*destination_pointer,
                                                                bytes.size(),
                                                                holoscan::DeviceId{device_},
                                                                *source,
                                                                ::cuda::stream_ref{consumer.get()});
  ASSERT_TRUE(copied) << copied.error().message;
  std::array<std::byte, 4> actual{};
  ASSERT_EQ(cudaMemcpy(actual.data(), *destination_pointer, actual.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, bytes);
  const auto snapshot =
      test_detail::capture_tensor_snapshot(*source, {}, bytes.size(), consumer.get());
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  EXPECT_EQ(snapshot->bytes(), (std::vector<std::byte>{bytes.begin(), bytes.end()}));
  EXPECT_EQ(snapshot->source_device().device_id, device_);
  EXPECT_EQ(snapshot->source_device().device_type,
            GetParam() == holoscan::MemoryStorageType::kCudaManaged ? kDLCUDAManaged : kDLCUDA);
  const auto wrong_device =
      test_detail::copy_staged_tensor_to_device(*destination_pointer,
                                                bytes.size(),
                                                holoscan::DeviceId{device_ + 1},
                                                *source,
                                                ::cuda::stream_ref{consumer.get()});
  ASSERT_FALSE(wrong_device);
  EXPECT_EQ(wrong_device.error().code, holoscan::ErrorCode::kIncompatible);
  const auto small_destination =
      test_detail::copy_staged_tensor_to_device(*destination_pointer,
                                                bytes.size() - 1U,
                                                holoscan::DeviceId{device_},
                                                *source,
                                                ::cuda::stream_ref{consumer.get()});
  ASSERT_FALSE(small_destination);
  EXPECT_EQ(small_destination.error().code, holoscan::ErrorCode::kIncompatible);
}

INSTANTIATE_TEST_SUITE_P(StorageKinds, OperatorTestTensorCudaCopies,
                         ::testing::Values(holoscan::MemoryStorageType::kDevice,
                                           holoscan::MemoryStorageType::kCudaManaged),
                         [](const ::testing::TestParamInfo<holoscan::MemoryStorageType>& value) {
                           return value.param == holoscan::MemoryStorageType::kCudaManaged
                                      ? "CudaManaged"
                                      : "CudaDevice";
                         });

// REQ: OPERATOR-073
// REQ: OPERATOR-066
// REQ: OPERATOR-067
TEST(OperatorTestTensorEdges, ManagedEmptyTensorRemainsOneCaptureWithMetadata) {
  using namespace std::chrono_literals;
  const std::vector<std::int64_t> shape{2, 0, 3};
  auto h = holoscan::testing::create_operator_test<EmptyTensorPass>();
  auto staged = holoscan::testing::sample_at(tensor_view(nullptr, shape), 17ms);
  staged.metadata.frame_id = 23U;
  staged.metadata.trace_id = 41U;
  staged.metadata.flags = holoscan::SampleFlags::kDegraded;
  h.push_samples("in",
                 std::vector<holoscan::testing::TestSample<holoscan::Tensor>>{std::move(staged)});
  // The public capture API requires a positive byte bound even for an empty payload.
  h.capture_snapshot<holoscan::Tensor>("out", 1U, 1U);
  auto running = h.run_async({.timeout = 5s});
  ASSERT_TRUE(running.wait_for_output("out", 1U, 5s));
  EXPECT_EQ(running.output_count("out"), 1U);
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_FALSE(running.is_running());
  EXPECT_EQ(running.output_count("out"), 1U);
  const auto records = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(records.size(), 1U);
  const auto& record = records.front();
  EXPECT_TRUE(record.bytes().empty());
  EXPECT_EQ(record.shape(), shape);
  EXPECT_EQ(record.source_byte_strides(), (std::vector<std::int64_t>{0, 3, 1}));
  EXPECT_EQ(record.source_byte_offset(), 0U);
  EXPECT_EQ(record.source_device().device_type, kDLCPU);
  EXPECT_EQ(record.source_device().device_id, 0);
  EXPECT_EQ(record.dtype().code, kDLUInt);
  EXPECT_EQ(record.dtype().bits, 8U);
  EXPECT_EQ(record.dtype().lanes, 1U);
  const auto& metadata = record.sample_metadata();
  EXPECT_EQ(metadata.capture_timestamp_ns, 17000000);
  EXPECT_NE(metadata.source_clock_id.value, 0U);
  EXPECT_NE(metadata.publish_clock_id.value, 0U);
  EXPECT_EQ(metadata.frame_id, 23U);
  EXPECT_EQ(metadata.trace_id, 41U);
  EXPECT_EQ(metadata.flags, holoscan::SampleFlags::kDegraded);
  // Repeated collection must neither duplicate nor drop the zero-byte record.
  EXPECT_EQ(running.finish().snapshots<holoscan::Tensor>("out").size(), 1U);
}
