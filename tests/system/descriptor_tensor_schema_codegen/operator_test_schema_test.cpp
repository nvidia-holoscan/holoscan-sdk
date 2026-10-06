/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../../testing/managed_tensor_source.hpp"
#include "descriptor_frame_multi_order_traits.hpp"

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, GeneratedMetadataCloneOwnsNestedFieldsAndClearsEveryTensor) {
  using Value = app3p::MultiDescriptorFrameT;
  Value original;
  original.sequence = 17U;
  original.exposure = 2.5F;
  original.label = "original";
  original.header = std::make_shared<app3p::FrameHeaderT>();
  original.header->source = "sensor";
  original.first = std::make_shared<holoscan::Tensor>();
  original.second = std::make_shared<holoscan::Tensor>();
  original.third = std::make_shared<holoscan::Tensor>();
  auto copied = holoscan::DescriptorTensorSchemaNativeTraits<Value>::clone_metadata(original);
  ASSERT_TRUE(copied) << copied.error().message;
  original.header->source = "changed";
  original.label = "changed";
  EXPECT_EQ(copied->header->source, "sensor");
  EXPECT_EQ(copied->label, "original");
  EXPECT_EQ(copied->sequence, 17U);
  EXPECT_EQ(copied->exposure, 2.5F);
  EXPECT_FALSE(copied->first);
  EXPECT_FALSE(copied->second);
  EXPECT_FALSE(copied->third);
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, CustomCodecImageMetadataClonePreservesOrdinaryFields) {
  holoscan::schema::ImageT original;
  original.width = 4U;
  original.height = 2U;
  original.data = std::make_shared<holoscan::Tensor>();
  original.header = std::make_shared<holoscan::schema::HeaderT>();
  original.header->frame_id = "camera";
  auto copied =
      holoscan::DescriptorTensorSchemaNativeTraits<holoscan::schema::ImageT>::clone_metadata(
          original);
  ASSERT_TRUE(copied) << copied.error().message;
  original.header->frame_id = "changed";
  EXPECT_EQ(copied->header->frame_id, "camera");
  EXPECT_EQ(copied->width, 4U);
  EXPECT_EQ(copied->height, 2U);
  EXPECT_FALSE(copied->data);
}

namespace {
template <holoscan::PublishedPayload T>
class SchemaEcho final : public holoscan::Operator<> {
 public:
  explicit SchemaEcho(holoscan::MemoryKind first_memory = holoscan::MemoryKind::kHost)
      : first_memory_(first_memory) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    auto& output = spec.output(out, "out");
    using Traits = holoscan::DescriptorTensorSchemaNativeTraits<T>;
    for (std::size_t i = 0; i < Traits::tensor_count; ++i) {
      output
          .tensor_field(holoscan::TensorFieldSelector<T>{Traits::tensor_field_ids[i],
                                                         static_cast<std::uint16_t>(i)})
          .produces_tensor(holoscan::TensorOutputSpec{
              .representation = {.memory_kind =
                                     i == 0U ? first_memory_ : holoscan::MemoryKind::kHost,
                                 .dtype = DLDataType{kDLUInt, 8, 1},
                                 .rank = 1U},
              .bounds = holoscan::tensor_bounds(4U),
              .storage = holoscan::TensorOutputStorage::kRuntimePool});
    }
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    return out.emit(std::move(*value));
  }
  holoscan::Input<T> in;
  holoscan::Output<T> out;

 private:
  holoscan::MemoryKind first_memory_;
};

std::shared_ptr<holoscan::Tensor> schema_tensor(std::uint8_t first, std::int64_t length = 4,
                                                DLDevice device = {kDLCPU, 0}) {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 1> shape{4};
    std::array<std::uint8_t, 4> bytes{};
  };
  auto storage = std::make_unique<Storage>();
  storage->shape[0] = length;
  storage->bytes = {
      first, std::uint8_t(first + 1), std::uint8_t(first + 2), std::uint8_t(first + 3)};
  storage->managed.dl_tensor = {.data = length == 0 ? nullptr : storage->bytes.data(),
                                .device = device,
                                .ndim = 1,
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  auto* managed = &storage.release()->managed;
  return std::make_shared<holoscan::Tensor>(managed);
}
}  // namespace

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, MovedSchemaStagingDetachesCallerTensorWrapperAliases) {
  using Value = app3p::MultiDescriptorFrameT;
  using Fields = app3p::MultiDescriptorFrameTensorFields;
  Value value;
  value.label = "staged";
  value.header = std::make_shared<app3p::FrameHeaderT>();
  value.header->source = "original";
  value.first = schema_tensor(1);
  value.second = schema_tensor(10);
  value.third = schema_tensor(20);
  auto caller_tensor = value.first;
  auto caller_header = value.header;
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push("in", std::move(value));
  // A moved schema can still have aliases retained by its caller. Replacing a Tensor wrapper
  // must not replace the staged descriptor or redirect the source to unleased backing.
  *caller_tensor = *schema_tensor(90);
  caller_header->source = "changed";
  h.capture_snapshot<Value>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshot = result.snapshot<Value>("out");
  EXPECT_EQ(snapshot.tensor(Fields::first).bytes()[0], std::byte{1});
  EXPECT_EQ(snapshot.metadata().header->source, "original");
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, MultiTensorSnapshotOwnsAllFieldsAndIndependentMetadata) {
  using Value = app3p::MultiDescriptorFrameT;
  using Fields = app3p::MultiDescriptorFrameTensorFields;
  Value original;
  original.label = "original";
  original.header = std::make_shared<app3p::FrameHeaderT>();
  original.header->source = "sensor";
  original.first = schema_tensor(1);
  original.second = schema_tensor(10);
  original.third = schema_tensor(20);
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push("in", original);
  h.capture_snapshot<Value>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshot = result.snapshot<Value>("out");
  EXPECT_FALSE(snapshot.metadata().first);
  EXPECT_FALSE(snapshot.metadata().second);
  EXPECT_FALSE(snapshot.metadata().third);
  EXPECT_EQ(snapshot.metadata().header->source, "sensor");
  EXPECT_EQ(snapshot.tensor(Fields::first).bytes()[0], std::byte{1});
  EXPECT_EQ(snapshot.tensor(Fields::second).bytes()[0], std::byte{10});
  EXPECT_EQ(snapshot.tensor(Fields::third).bytes()[0], std::byte{20});
  auto changed = snapshot.metadata();
  changed.header->source = "changed";
  EXPECT_EQ(snapshot.metadata().header->source, "sensor");
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, ImageSnapshotPreservesSchemaIdentityAndMetadata) {
  using Value = holoscan::schema::ImageT;
  Value frame;
  frame.height = 1U;
  frame.width = 4U;
  frame.encoding = holoscan::schema::ImageEncoding_CUSTOM;
  frame.header = std::make_shared<holoscan::schema::HeaderT>();
  frame.header->device_sequence = 42U;
  frame.header->stamp = std::make_shared<holoscan::schema::TimeT>();
  frame.header->stamp->time_ns = 1234;
  frame.data = schema_tensor(5);
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push("in", frame);
  frame.header->stamp->time_ns = 5678;
  h.capture_snapshot<Value>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshot = result.snapshot<Value>("out");
  EXPECT_EQ(snapshot.metadata().width, 4U);
  EXPECT_EQ(snapshot.metadata().height, 1U);
  EXPECT_EQ(snapshot.metadata().header->device_sequence, 42U);
  auto metadata = snapshot.metadata();
  EXPECT_NE(metadata.header.get(), frame.header.get());
  EXPECT_NE(metadata.header->stamp.get(), frame.header->stamp.get());
  EXPECT_EQ(metadata.header->stamp->time_ns, 1234);
  metadata.header->stamp->time_ns = 9999;
  EXPECT_EQ(snapshot.metadata().header->stamp->time_ns, 1234);
  EXPECT_FALSE(snapshot.metadata().data);
  EXPECT_EQ(snapshot.tensor(holoscan::schema::ImageTensorFields::data).bytes()[0], std::byte{5});
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, FinalTensorFieldCaptureFailureDoesNotAppendPartialRecord) {
  using Value = app3p::MultiDescriptorFrameT;
  Value frame;
  frame.first = schema_tensor(1);
  frame.second = schema_tensor(10);
  frame.third = schema_tensor(20);
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push("in", frame);
  h.capture_snapshot<Value>("out", 8U, 10U);
  const auto result = h.run();
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<Value>("out").empty());
  EXPECT_NE(result.describe_failure().find("tensor_field_id=5"), std::string::npos);
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, MixedDeviceAndHostFieldsUseExplicitPlacementAndCopyReadiness) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA unavailable";
  int prior_device{};
  ASSERT_EQ(cudaGetDevice(&prior_device), cudaSuccess);
  struct RestoreDevice {
    int value;
    ~RestoreDevice() { static_cast<void>(cudaSetDevice(value)); }
  };
  const RestoreDevice restore{prior_device};
  const int device = count > 1 ? 1 : 0;
  ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
  cudaStream_t stream{};
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  struct DestroyStream {
    cudaStream_t value;
    ~DestroyStream() { static_cast<void>(cudaStreamDestroy(value)); }
  };
  const DestroyStream destroy{stream};
  const std::array<std::int64_t, 1> shape{4};
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{kDLUInt, 8, 1},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kDevice,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  ASSERT_TRUE(tensor) << tensor.error().message;
  auto writable = tensor->mutable_data();
  ASSERT_TRUE(writable);
  ASSERT_EQ(cudaMemsetAsync(*writable, 9, 4U, stream), cudaSuccess);
  ASSERT_TRUE(tensor->set_readiness_stream(stream));

  using Value = app3p::MultiDescriptorFrameT;
  using Fields = app3p::MultiDescriptorFrameTensorFields;
  Value frame;
  frame.first = std::make_shared<holoscan::Tensor>(std::move(*tensor));
  frame.second = schema_tensor(10);
  frame.third = schema_tensor(20);
  auto h =
      holoscan::testing::create_operator_test<SchemaEcho<Value>>(holoscan::MemoryKind::kCudaDevice);
  h.push("in", frame);
  h.input_source_device("in", holoscan::DeviceId{device});
  h.output_device("out", holoscan::DeviceId{device});
  h.capture_snapshot<Value>("out");
  auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshot = result.snapshot<Value>("out");
  EXPECT_EQ(snapshot.tensor(Fields::first).source_device().device_id, device);
  EXPECT_EQ(snapshot.tensor(Fields::first).source_device().device_type, kDLCUDA);
  EXPECT_EQ(snapshot.tensor(Fields::first).bytes(), std::vector<std::byte>(4U, std::byte{9}));
  EXPECT_EQ(snapshot.tensor(Fields::second).source_device().device_type, kDLCPU);
  EXPECT_EQ(snapshot.tensor(Fields::second).bytes()[0], std::byte{10});
  EXPECT_EQ(snapshot.tensor(Fields::third).bytes()[0], std::byte{20});
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, InvalidLastFieldRejectsWholeBatchWithoutChangingPendingInput) {
  using Value = app3p::MultiDescriptorFrameT;
  Value valid;
  valid.first = schema_tensor(1);
  valid.second = schema_tensor(10);
  valid.third = schema_tensor(20);
  Value invalid = valid;
  invalid.third.reset();
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push("in", valid);
  EXPECT_THROW(h.push_many<Value>("in", {valid, invalid}), std::invalid_argument);
  h.capture_snapshot<Value>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.snapshots<Value>("out").size(), 1U);
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, FailedFinalFieldPreservesEarlierRecordAndFailedSampleMetadata) {
  using Value = app3p::MultiDescriptorFrameT;
  Value small;
  small.first = schema_tensor(1, 2);
  small.second = schema_tensor(10, 2);
  small.third = schema_tensor(20, 2);
  Value large;
  large.first = schema_tensor(30);
  large.second = schema_tensor(40);
  large.third = schema_tensor(50);
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push_samples<Value>("in",
                        {{.data = small,
                          .metadata = {.frame_id = 101U},
                          .capture_coordinate = std::chrono::milliseconds{10}},
                         {.data = large,
                          .metadata = {.frame_id = 102U},
                          .capture_coordinate = std::chrono::milliseconds{20}}});
  h.capture_snapshot<Value>("out", 8U, 10U);
  const auto result = h.run();
  EXPECT_FALSE(result.ok());
  const auto snapshots = result.snapshots<Value>("out");
  ASSERT_EQ(snapshots.size(), 1U);
  EXPECT_EQ(snapshots[0].sample_metadata().frame_id, 101U);
  const auto failure = result.capture_failure("out");
  if (failure) {
    EXPECT_EQ(failure->metadata.frame_id, 102U);
    EXPECT_EQ(failure->metadata.capture_timestamp_ns, 20'000'000);
    EXPECT_NE(failure->error.message.find("tensor_field_id=5"), std::string::npos);
  } else {
    FAIL() << "Expected failed-sample evidence";
  }
}

namespace {
class ShapeSchemaSink final : public holoscan::Operator<> {
 public:
  explicit ShapeSchemaSink(std::shared_ptr<std::vector<std::vector<std::int64_t>>> seen)
      : seen_(std::move(seen)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    seen_->push_back(value->data.first->shape());
    return {};
  }
  holoscan::Input<app3p::MultiDescriptorFrameT> in;

 private:
  std::shared_ptr<std::vector<std::vector<std::int64_t>>> seen_;
};
std::shared_ptr<holoscan::Tensor> writable_schema_tensor(std::vector<std::int64_t> shape) {
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{kDLUInt, 8, 1},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kSystem,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor)
    throw holoscan::RuntimeError{std::move(tensor).error()};
  auto bytes = tensor->mutable_data();
  if (!bytes)
    throw holoscan::RuntimeError{std::move(bytes).error()};
  std::memset(*bytes, 7, static_cast<std::size_t>(tensor->nbytes()));
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}
class StringEcho final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    return out.emit(std::move(*value));
  }
  holoscan::Input<holoscan::schema::StringT> in;
  holoscan::Output<holoscan::schema::StringT> out;
};
}  // namespace

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, CrossedFieldShapesUseAConsistentSourceAllocationEnvelope) {
  using Value = app3p::MultiDescriptorFrameT;
  Value first;
  first.first = writable_schema_tensor({1, 4});
  first.second = schema_tensor(10);
  first.third = schema_tensor(20);
  Value second = first;
  second.first = writable_schema_tensor({4, 1});
  auto seen = std::make_shared<std::vector<std::vector<std::int64_t>>>();
  auto h = holoscan::testing::create_operator_test<ShapeSchemaSink>(seen);
  h.push_many("in", std::vector<Value>{first, second});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(*seen, (std::vector<std::vector<std::int64_t>>{{1, 4}, {4, 1}}));
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, FinalFieldLeaseRejectionRollsBackClaimsAndPreservesPendingFrame) {
  using Value = app3p::MultiDescriptorFrameT;
  Value valid;
  valid.first = schema_tensor(1);
  valid.second = schema_tensor(10);
  valid.third = schema_tensor(20);
  auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
  h.push("in", valid);
  Value blocked;
  blocked.first = writable_schema_tensor({4});
  blocked.second = writable_schema_tensor({4});
  blocked.third = writable_schema_tensor({4});
  auto writer = blocked.third->to_dlpack();
  ASSERT_TRUE(writer);
  EXPECT_THROW(h.push("in", blocked), holoscan::RuntimeError);
  EXPECT_TRUE(blocked.first->mutable_data());
  EXPECT_TRUE(blocked.second->mutable_data());
  (*writer)->deleter(*writer);
  EXPECT_TRUE(blocked.third->mutable_data());
  h.capture_snapshot<Value>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(result.snapshots<Value>("out").size(), 1U);
}

// REQ: OPERATOR-064
TEST(OperatorTestingSchema, OrdinaryStringSchemaUsesOwningValueCapture) {
  holoscan::schema::StringT value;
  value.data = "independent generated value";
  auto h = holoscan::testing::create_operator_test<StringEcho>();
  h.push(&StringEcho::in, value);
  value.data = "changed";
  h.capture<holoscan::schema::StringT>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<holoscan::schema::StringT>("out").data, "independent generated value");
}

namespace {
template <holoscan::PublishedPayload T>
class RetainSchemaSource final : public holoscan::Operator<> {
 public:
  explicit RetainSchemaSource(T metadata, bool stage_payload = false)
      : metadata_(std::move(metadata)), stage_payload_(stage_payload) {}
  void setup(holoscan::OperatorSpec& spec) override {
    using Traits = holoscan::DescriptorTensorSchemaNativeTraits<T>;
    auto& output = spec.output(out, "out");
    const holoscan::TensorOutputSpec layout{
        .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8, 1},
                           .rank = 1U},
        .bounds = holoscan::tensor_bounds(4U),
        .storage = holoscan::TensorOutputStorage::kRuntimePool};
    if constexpr (Traits::tensor_count == 1U) {
      output.produces_tensor(layout);
    } else {
      for (std::size_t index = 0; index < Traits::tensor_count; ++index)
        output
            .tensor_field(holoscan::TensorFieldSelector<T>{Traits::tensor_field_ids[index],
                                                           static_cast<std::uint16_t>(index)})
            .produces_tensor(layout);
    }
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 2U});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    using Traits = holoscan::DescriptorTensorSchemaNativeTraits<T>;
    if constexpr (Traits::tensor_count > 1U) {
      if (stage_payload_)
        return holoscan::testing::detail::emit_staged_schema_bundle(out, metadata_, {}, context);
    }
    const std::array<std::int64_t, 1> shape{4};
    if constexpr (Traits::tensor_count == 1U) {
      auto loan = out.allocate_tensor(shape);
      if (!loan)
        return holoscan::make_unexpected(loan.error());
      auto filled = loan->write_host_and_commit(
          [&](const auto& write) { std::memset(write.data(), 10 + next_, write.byte_size()); });
      if (!filled)
        return filled;
      ++next_;
      return out.emit_tensor(std::move(*loan), metadata_);
    } else {
      std::array<holoscan::TensorFieldLoanRequest<T>, Traits::tensor_count> requests;
      for (std::size_t index = 0; index < requests.size(); ++index)
        requests[index] = {
            .field = {Traits::tensor_field_ids[index], static_cast<std::uint16_t>(index)},
            .tensor = {.shape = shape, .dtype = DLDataType{kDLUInt, 8, 1}}};
      auto bundle = out.allocate_tensors(requests);
      if (!bundle)
        return holoscan::make_unexpected(bundle.error());
      for (std::size_t index = 0; index < requests.size(); ++index) {
        auto filled = bundle->write_host_and_commit(requests[index].field, [&](const auto& write) {
          std::memset(write.data(), static_cast<int>(10 * (index + 1U) + next_), write.byte_size());
        });
        if (!filled)
          return filled;
      }
      ++next_;
      return out.emit_tensors(std::move(*bundle), metadata_);
    }
  }
  holoscan::Output<T> out;

 private:
  T metadata_;
  bool stage_payload_{};
  int next_{};
};
}  // namespace

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, RetainImageKeepsNativeTensorAndOwningMetadata) {
  using Value = holoscan::schema::ImageT;
  Value metadata;
  metadata.height = 1U;
  metadata.width = 4U;
  metadata.encoding = holoscan::schema::ImageEncoding_CUSTOM;
  metadata.header = std::make_shared<holoscan::schema::HeaderT>();
  metadata.header->frame_id = "image";
  auto h = holoscan::testing::create_operator_test<RetainSchemaSource<Value>>(metadata);
  h.capture<Value>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto values = result.values<Value>("out");
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values[0].data->template data_as<std::uint8_t>()[0], 10U);
  EXPECT_EQ(values[1].data->template data_as<std::uint8_t>()[0], 11U);
  values[0].header->frame_id = "modified copy";
  EXPECT_EQ(result.values<Value>("out")[0].header->frame_id, "image");
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, RetainMultiTensorPreservesEveryNativeFieldAndMetadata) {
  using Value = app3p::MultiDescriptorFrameT;
  Value metadata;
  metadata.label = "retained";
  metadata.header = std::make_shared<app3p::FrameHeaderT>();
  metadata.header->source = "source";
  auto h = holoscan::testing::create_operator_test<RetainSchemaSource<Value>>(metadata);
  h.capture<Value>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto values = result.values<Value>("out");
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values[0].first->template data_as<std::uint8_t>()[0], 10U);
  EXPECT_EQ(values[0].second->template data_as<std::uint8_t>()[0], 20U);
  EXPECT_EQ(values[0].third->template data_as<std::uint8_t>()[0], 30U);
  EXPECT_EQ(values[1].third->template data_as<std::uint8_t>()[0], 31U);
  EXPECT_EQ(values[1].header->source, "source");
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, MultiTensorNullBackedEmptyHostFieldsStageAndSnapshot) {
  using Value = app3p::MultiDescriptorFrameT;
  using Fields = app3p::MultiDescriptorFrameTensorFields;
  const std::array selectors{Fields::first, Fields::second, Fields::third};
  const std::array first_bytes{std::byte{1}, std::byte{10}, std::byte{20}};
  for (std::size_t empty_index = 0; empty_index < selectors.size(); ++empty_index) {
    SCOPED_TRACE(empty_index);
    Value frame;
    frame.label = "empty-field";
    frame.first = schema_tensor(1);
    frame.second = schema_tensor(10);
    frame.third = schema_tensor(20);
    const std::array members{&frame.first, &frame.second, &frame.third};
    *members[empty_index] = schema_tensor(0, 0);
    ASSERT_EQ((*members[empty_index])->data(), nullptr);
    auto h = holoscan::testing::create_operator_test<SchemaEcho<Value>>();
    h.push("in", frame);
    h.capture_snapshot<Value>("out");
    const auto result = h.run();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto snapshot = result.snapshot<Value>("out");
    EXPECT_EQ(snapshot.metadata().label, "empty-field");
    for (std::size_t index = 0; index < selectors.size(); ++index) {
      ASSERT_TRUE(snapshot.has_tensor(selectors[index]));
      const auto& tensor = snapshot.tensor(selectors[index]);
      EXPECT_EQ(tensor.source_device().device_type, kDLCPU);
      if (index == empty_index) {
        EXPECT_TRUE(tensor.bytes().empty());
        EXPECT_EQ(tensor.shape(), (std::vector<std::int64_t>{0}));
      } else {
        ASSERT_EQ(tensor.bytes().size(), 4U);
        EXPECT_EQ(tensor.bytes().front(), first_bytes[index]);
      }
    }
  }
}

// REQ: OPERATOR-057
TEST(OperatorTestingSchema, EmptyDeviceFieldStillRejectsImplicitHostStagingWithFieldContext) {
  using Value = app3p::MultiDescriptorFrameT;
  Value frame;
  // Empty device metadata performs no CUDA work, but cannot silently become a host source.
  frame.first = schema_tensor(0, 0, DLDevice{kDLCUDA, 0});
  frame.second = schema_tensor(10);
  frame.third = schema_tensor(20);
  // Reuse the generated schema's finite source with real host field allocations. The optional
  // fixture branch exercises the production staging helper against those pools directly.
  auto h = holoscan::testing::create_operator_test<RetainSchemaSource<Value>>(frame, true);
  h.capture_snapshot<Value>("out");
  const auto result = h.run();
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<Value>("out").empty());
  EXPECT_NE(result.describe_failure().find("host staging placement mismatch"), std::string::npos);
  EXPECT_NE(result.describe_failure().find("tensor_field_id=1"), std::string::npos);
}

namespace {
// Keep restart, capture, and joined-lifetime assertions in one ordered scenario.
// GoogleTest assertion expansion dominates this helper's reported complexity.
template <holoscan::PublishedPayload T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void verify_managed_schema(T metadata) {
  using namespace holoscan;
  using namespace holoscan::testing;
  using namespace holoscan::testing::test;
  using namespace std::chrono_literals;
  auto probe = std::make_shared<ManagedTensorProbe>();
  auto h = create_operator_test<ManagedTensorSource<T>>(probe, metadata, false, false, 3U);
  h.template capture<T>("out", Retain{.max_records = 8U});
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = tensor_restart_capacity();
  });
  auto run = h.run_async();
  ASSERT_TRUE(post_tensor(probe->wait_sender(1U)));
  ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
  const auto target = run.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  constexpr std::array stages{LifecycleStage::kAllocate};
  auto control = run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  for (std::size_t generation = 2U; generation <= 3U; ++generation) {
    const auto request =
        control->request_restart(*target, {.from_stage = LifecycleStage::kAllocate});
    ASSERT_TRUE(request);
    const auto completed = control->await(*request, 5s);
    ASSERT_TRUE(completed);
    ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);
    ASSERT_TRUE(post_tensor(probe->wait_sender(generation)));
    ASSERT_TRUE(run.wait_for_output("out", generation, 5s));
  }
  const auto result = run.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto values = result.template values<T>("out");
  ASSERT_EQ(values.size(), 3U);
  using Traits = DescriptorTensorSchemaNativeTraits<T>;
  for (std::size_t row = 0; row < values.size(); ++row) {
    for (std::size_t field = 0; field < Traits::tensor_count; ++field) {
      const auto* tensor = Traits::tensor_at(values[row], field);
      ASSERT_NE(tensor, nullptr);
      EXPECT_EQ(tensor->template data_as<std::uint8_t>()[0], 11U * (row + field + 1U));
    }
    if constexpr (std::same_as<T, schema::ImageT>) {
      EXPECT_EQ(values[row].header->frame_id, "managed image");
      EXPECT_EQ(values[row].width, 4U);
    } else {
      EXPECT_EQ(values[row].header->source, "managed source");
      EXPECT_EQ(values[row].label, "managed fields");
    }
  }
  EXPECT_EQ(probe->restart_allocations, 2U);
}
}  // namespace

// REQ: OPERATOR-057
// REQ: OPERATOR-070
TEST(OperatorTestingSchema, ManagedRestartRetainsImageAcrossThreeGenerations) {
  holoscan::schema::ImageT metadata;
  metadata.height = 1U;
  metadata.width = 4U;
  metadata.encoding = holoscan::schema::ImageEncoding_CUSTOM;
  metadata.header = std::make_shared<holoscan::schema::HeaderT>();
  metadata.header->frame_id = "managed image";
  verify_managed_schema(std::move(metadata));
}

// REQ: OPERATOR-057
// REQ: OPERATOR-070
TEST(OperatorTestingSchema, ManagedRestartRetainsEverySchemaFieldAcrossThreeGenerations) {
  app3p::MultiDescriptorFrameT metadata;
  metadata.label = "managed fields";
  metadata.header = std::make_shared<app3p::FrameHeaderT>();
  metadata.header->source = "managed source";
  verify_managed_schema(std::move(metadata));
}
