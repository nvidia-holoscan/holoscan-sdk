/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/memory/block_memory_pool.hpp>

#include <holomq/delivery_attribution.hpp>

#include "../../src/channel/channel_common.hpp"
#include "../../src/channel/channel_tensor_sidecar_store.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/holomq_tensor.hpp"
#include "../../src/runtime/channel_runtime_binding_access.hpp"
#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"
#include "channel_holomq_test_support.hpp"
#include "channel_test_support.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

using namespace holoscan::channel_test;  // NOLINT(build/namespaces)

namespace {

constexpr std::size_t kHoloMQTensorBufferBytes = 64U;
constexpr std::uint32_t kHoloMQTensorBufferCount = 1U;
constexpr std::uint32_t kHoloMQTensorPatternSeed = 0xC5C50000U;

[[nodiscard]] holoscan::expected<holoscan::ManualTensorPublisher, holoscan::Error>
create_tensor_publisher(holoscan::ChannelRuntime& runtime, const std::string_view channel_name,
                        const holoscan::ChannelOptions& options = {}) {
  return holoscan::channel_internal::TensorPublisherTestAccess::create(
      runtime, channel_name, options);
}

#ifndef HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE
#define HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE ""
#endif

// Tracks how many times its backing storage has been destroyed. The atomic counter
// makes Tensor storage lifetime observable without a real buffer or allocator.
// Used by make_counting_tensor(); see e.g. BlockingPublishesAndReceivesTensorHandle.
class CountingTensorStorage : public holoscan::TensorMemoryReference {
 public:
  CountingTensorStorage(std::shared_ptr<std::atomic<int>> destroyed_counter,
                        const std::uint8_t fill)
      : destroyed_counter_(std::move(destroyed_counter)) {
    data.fill(fill);
  }

  ~CountingTensorStorage() override { destroyed_counter_->fetch_add(1, std::memory_order_relaxed); }

  std::array<std::uint8_t, 6> data{};

 private:
  std::shared_ptr<std::atomic<int>> destroyed_counter_;
};

// Build a 2x3 uint8 host Tensor backed by CountingTensorStorage. Returns the raw data
// pointer via data_out for pointer-identity checks on receive; fill sets the first byte.
[[nodiscard]] holoscan::Tensor make_counting_tensor(
    const std::shared_ptr<std::atomic<int>>& destroyed_counter, void** data_out,
    const std::uint8_t fill) {
  auto storage = std::make_shared<CountingTensorStorage>(destroyed_counter, fill);
  *data_out = storage->data.data();

  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->memory_reference = std::move(storage);
  context->dl_shape = {2, 3};
  context->dl_strides = {3, 1};
  context->tensor.dl_tensor.data = *data_out;
  context->tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8, 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0U;
  return holoscan::Tensor(std::move(context));
}

[[nodiscard]] holomq::expected<holomq::BufferPool, holomq::Error> make_holomq_pinned_host_pool(
    holomq::Context& context) {
  return holoscan::channel_test::make_holomq_pinned_host_pool(
      context, kHoloMQTensorBufferBytes, kHoloMQTensorBufferCount);
}

[[nodiscard]] std::unique_ptr<HoloMQPoolFixture> make_tensor_pool_fixture(
    const HoloMQTestMemoryKind memory_kind) {
  return make_holomq_pool_fixture({.buffer_bytes = kHoloMQTensorBufferBytes,
                                   .buffer_count = kHoloMQTensorBufferCount,
                                   .socket_prefix = "hsdk_channel_holomq_tensor",
                                   .memory_kind = memory_kind,
                                   .fd_export = memory_kind == HoloMQTestMemoryKind::kCudaDevice});
}

[[nodiscard]] std::unique_ptr<HoloMQPoolFixture> make_holomq_device_pool_fixture() {
  return make_tensor_pool_fixture(HoloMQTestMemoryKind::kCudaDevice);
}

[[nodiscard]] std::unique_ptr<HoloMQPoolFixture> make_holomq_pinned_host_pool_fixture() {
  return make_tensor_pool_fixture(HoloMQTestMemoryKind::kPinnedHost);
}

// Wait until the publisher can deliver to the full expected recipient count, not just the first
// subscriber that connects. wait_for_subscribers() only proves >= 1 recipient is visible, which
// races fanout discovery and can make a 2-recipient emit intermittently report 1. This probes
// actual delivery with a throwaway emit and drains it from every subscriber so the probe samples
// do not leak into the test's first real receive.
[[nodiscard]] bool wait_for_tensor_recipients(
    holoscan::ManualTensorPublisher& publisher,
    const std::initializer_list<holoscan::ChannelSubscriber<holoscan::Tensor>*> subscribers,
    const std::uint32_t expected_count, const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  while (std::chrono::steady_clock::now() < deadline) {
    static_cast<void>(publisher.update_connections());
    void* data = nullptr;
    auto emitted = publisher.emit(make_counting_tensor(destroyed, &data, 0U));
    const bool reached = emitted.has_value() && emitted.value() >= expected_count;
    for (auto* subscriber : subscribers) {
      while (subscriber->receive().has_value()) {
      }
    }
    if (reached) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return false;
}

}  // namespace

TEST(ChannelTensorTest, SidecarCapacityIncludesMaterializationAndPublicationStagingRecords) {
  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.expected_recipient_count = 1U;
  options.max_subscriber_queue_depth = 1U;

  auto capacity = holoscan::channel_internal::tensor_sidecar_record_capacity(options);

  ASSERT_TRUE(capacity) << holoscan::runtime_error_message(capacity.error());
  EXPECT_EQ(*capacity, 3U);

  // Key setup: account independently for every fanout recipient's queue and materialization slot.
  options.expected_recipient_count = 2U;
  options.max_subscriber_queue_depth = 3U;
  capacity = holoscan::channel_internal::tensor_sidecar_record_capacity(options);
  ASSERT_TRUE(capacity) << holoscan::runtime_error_message(capacity.error());
  EXPECT_EQ(*capacity, 9U);

  // Leave exactly two records below the endpoint limit for one receiver and one staged publish.
  options.expected_recipient_count = 1U;
  options.max_subscriber_queue_depth = std::numeric_limits<std::uint32_t>::max() - 2U;
  capacity = holoscan::channel_internal::tensor_sidecar_record_capacity(options);
  ASSERT_TRUE(capacity) << holoscan::runtime_error_message(capacity.error());
  EXPECT_EQ(*capacity, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));

  // Key assertion: a queue leaving only one record cannot represent both transient states.
  options.max_subscriber_queue_depth = std::numeric_limits<std::uint32_t>::max() - 1U;
  capacity = holoscan::channel_internal::tensor_sidecar_record_capacity(options);
  ASSERT_FALSE(capacity);
  EXPECT_EQ(capacity.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies a resolved interprocess Tensor pool plan is accepted before output-owned allocation is
// enabled in a later slice.
TEST(ChannelTensorTest, AcceptsCompatibleInterfaceTensorPoolPlan) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  holoscan::ChannelOptions options{};
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  options.expected_recipient_count = 2U;
  options.max_subscriber_borrowed_samples = 3U;
  options.tensor_pool_plan = holoscan::InterfaceTensorPoolPlan{
      .storage_type = holoscan::MemoryStorageType::kDevice,
      .cuda_device_ordinal = 0,
      .max_buffer_bytes = 4096U,
      .slot_count = 8U,
      .fanout_capacity = 2U,
      .max_borrowed_samples_per_recipient = 3U,
  };

  auto publisher = create_tensor_publisher(runtime.value(), make_unique_channel_name(), options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
}

// Verifies malformed and endpoint-incompatible pool plans fail during endpoint creation.
TEST(ChannelTensorTest, RejectsInvalidInterfaceTensorPoolPlans) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  holoscan::ChannelOptions options{};
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  options.expected_recipient_count = 2U;
  options.max_subscriber_borrowed_samples = 3U;
  const holoscan::InterfaceTensorPoolPlan valid_plan{
      .storage_type = holoscan::MemoryStorageType::kDevice,
      .cuda_device_ordinal = 0,
      .max_buffer_bytes = 4096U,
      .slot_count = 8U,
      .fanout_capacity = 2U,
      .max_borrowed_samples_per_recipient = 3U,
  };

  const auto expect_rejected = [&runtime, &options](const holoscan::InterfaceTensorPoolPlan& plan) {
    options.tensor_pool_plan = plan;
    auto publisher = create_tensor_publisher(runtime.value(), make_unique_channel_name(), options);
    ASSERT_FALSE(publisher);
    EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
  };

  auto invalid_plan = valid_plan;
  invalid_plan.max_buffer_bytes = 0U;
  expect_rejected(invalid_plan);
  invalid_plan = valid_plan;
  invalid_plan.slot_count = 0U;
  expect_rejected(invalid_plan);
  invalid_plan = valid_plan;
  invalid_plan.fanout_capacity = 1U;
  expect_rejected(invalid_plan);
  invalid_plan = valid_plan;
  invalid_plan.max_borrowed_samples_per_recipient = 2U;
  expect_rejected(invalid_plan);
  invalid_plan = valid_plan;
  invalid_plan.cuda_device_ordinal = -1;
  expect_rejected(invalid_plan);
  invalid_plan = valid_plan;
  invalid_plan.storage_type = holoscan::MemoryStorageType::kHost;
  expect_rejected(invalid_plan);
  invalid_plan = valid_plan;
  invalid_plan.storage_type = holoscan::MemoryStorageType::kCudaManaged;
  expect_rejected(invalid_plan);

  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kSameProcessSidecar;
  expect_rejected(valid_plan);
}

// Verifies a mutable HoloMQ pool lease can be presented as a Tensor and remains checked out until
// the last Tensor handle releases the hidden lease owner.
TEST(ChannelTensorTest, HoloMQBufferTensorOwnsMutableLease) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kHoloMQTensorPatternSeed));
  void* const original_data_pointer = buffer.ptr();

  const std::array<int64_t, 1> shape{16};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  EXPECT_EQ(tensor->data(), original_data_pointer);
  EXPECT_EQ(tensor->device().device_type, kDLCUDAHost);
  EXPECT_FALSE(tensor->is_read_only());
  EXPECT_TRUE(tensor->mutable_data());
  EXPECT_TRUE(verify_host_pattern(
      tensor->data(), static_cast<std::size_t>(tensor->nbytes()), kHoloMQTensorPatternSeed));

  EXPECT_FALSE(fixture->pool.acquire().has_value());
  tensor = holoscan::Tensor{};
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies an imported HoloMQ view becomes a read-only Tensor and keeps the imported lease checked
// out until the Tensor is released.
TEST(ChannelTensorTest, HoloMQConstBufferTensorOwnsReadOnlyImportLease) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kHoloMQTensorPatternSeed));
  const auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  buffer = holomq::Buffer{};

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto imported_or = fixture->context.import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::ConstBuffer imported = std::move(*imported_or);
  EXPECT_FALSE(imported.has_producer_readiness_fence());
  const void* const original_data_pointer = imported.ptr();

  const std::array<int64_t, 1> shape{16};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(imported));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  EXPECT_EQ(tensor->data(), original_data_pointer);
  EXPECT_EQ(tensor->device().device_type, kDLCUDAHost);
  EXPECT_TRUE(tensor->is_read_only());
  EXPECT_FALSE(tensor->mutable_data());
  EXPECT_TRUE(verify_host_pattern(
      tensor->data(), static_cast<std::size_t>(tensor->nbytes()), kHoloMQTensorPatternSeed));

  EXPECT_FALSE(fixture->pool.acquire().has_value());
  tensor = holoscan::Tensor{};
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies Tensor construction rejects shapes whose logical extent would exceed the fixed HoloMQ
// pool slot size.
TEST(ChannelTensorTest, HoloMQBufferTensorRejectsShapeExceedingLeaseSize) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);

  const std::array<int64_t, 1> shape{17};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(buffer));
  ASSERT_FALSE(tensor);
  EXPECT_EQ(tensor.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies explicit strides are checked against the highest addressed byte, not only the logical
// element count, for both producer and imported HoloMQ leases.
TEST(ChannelTensorTest, HoloMQBufferTensorRejectsStridedExtentExceedingLeaseSize) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto imported_or = fixture->context.import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::ConstBuffer imported = std::move(*imported_or);

  const std::array<int64_t, 2> shape{2, 2};
  const std::array<int64_t, 2> byte_strides{64, 4};
  auto mutable_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape,
      DLDataType{kDLUInt, 32, 1},
      holoscan::TensorByteStridesSpan{byte_strides.data(), byte_strides.size()},
      std::move(buffer));
  ASSERT_FALSE(mutable_tensor);
  EXPECT_EQ(mutable_tensor.error().code, holoscan::ErrorCode::kInvalidArgument);

  auto imported_tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
      shape,
      DLDataType{kDLUInt, 32, 1},
      holoscan::TensorByteStridesSpan{byte_strides.data(), byte_strides.size()},
      std::move(imported));
  ASSERT_FALSE(imported_tensor);
  EXPECT_EQ(imported_tensor.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies host synchronization on an unsynchronized HoloMQ import consumes the ready generation
// carried by the BufferHandle before exposing producer data to CPU access.
TEST(ChannelTensorTest, HoloMQImportedHostTensorWaitsForProducerReadiness) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  holoscan::test::UniqueCudaStream producer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  holoscan::test::BlockingHostState producer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                producer_stream.get(), holoscan::test::blocking_host_callback, &producer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard producer_guard(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(producer_state));

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  const std::array<int64_t, 1> shape{16};
  auto source_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(buffer));
  ASSERT_TRUE(source_tensor) << holoscan::runtime_error_message(source_tensor.error());
  auto source_data = source_tensor->mutable_data();
  ASSERT_TRUE(source_data) << holoscan::runtime_error_message(source_data.error());
  constexpr std::uint8_t kPattern = 0x7bU;
  std::memset(*source_data, kPattern, kHoloMQTensorBufferBytes);
  ASSERT_TRUE(source_tensor->set_readiness_stream(producer_stream.get()));
  auto descriptor = holoscan::channel_internal::export_holomq_tensor_descriptor(*source_tensor);
  ASSERT_TRUE(descriptor) << holoscan::runtime_error_message(descriptor.error());

  auto imported_or = fixture->context.import_buffer_unsynchronized(descriptor->buffer_handle);
  ASSERT_TRUE(imported_or.has_value());
  EXPECT_TRUE(imported_or->has_producer_readiness_fence());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto imported_tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
      descriptor.value(), std::move(*imported_or));
  ASSERT_TRUE(imported_tensor) << holoscan::runtime_error_message(imported_tensor.error());

  // Key setup: imported Tensor carries the unsynchronized producer handle; synchronize_for_host
  // must block until S0 (producer_stream) is released.
  auto host_wait = std::async(
      std::launch::async, [&imported_tensor]() { return imported_tensor->synchronize_for_host(); });
  // Key assertion: host_wait is still pending while S0 is blocked.
  EXPECT_EQ(host_wait.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  auto synchronized = host_wait.get();
  ASSERT_TRUE(synchronized) << holoscan::runtime_error_message(synchronized.error());
  const auto* bytes = static_cast<const std::uint8_t*>(imported_tensor->data());
  for (std::size_t index = 0U; index < kHoloMQTensorBufferBytes; ++index) {
    EXPECT_EQ(bytes[index], kPattern);
  }
}

// Verifies the interprocess Tensor descriptor captures the Tensor view metadata and embedded
// HoloMQ buffer handle without exposing HoloMQ details to the public Tensor API.
TEST(ChannelTensorTest, HoloMQTensorDescriptorCapturesTensorMetadata) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  const std::array<int64_t, 2> shape{4, 4};
  const std::array<int64_t, 2> byte_strides{16, 4};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape,
      DLDataType{kDLUInt, 32, 1},
      holoscan::TensorByteStridesSpan{byte_strides.data(), byte_strides.size()},
      std::move(buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto descriptor = holoscan::channel_internal::make_holomq_tensor_descriptor(*tensor, *handle);
  ASSERT_TRUE(descriptor) << holoscan::runtime_error_message(descriptor.error());
  EXPECT_TRUE(descriptor->has_supported_abi());
  EXPECT_EQ(descriptor->rank, 2U);
  EXPECT_EQ(descriptor->dtype.code, kDLUInt);
  EXPECT_EQ(descriptor->dtype.bits, 32U);
  EXPECT_EQ(descriptor->dtype.lanes, 1U);
  EXPECT_EQ(descriptor->memory_kind,
            holoscan::channel_internal::HoloMQTensorDescriptorMemoryKind::kHostPinned);
  EXPECT_EQ(descriptor->cuda_device_ordinal, -1);
  EXPECT_EQ(descriptor->byte_size, 64U);
  EXPECT_EQ(descriptor->shape[0], 4);
  EXPECT_EQ(descriptor->shape[1], 4);
  EXPECT_EQ(descriptor->byte_strides[0], 16);
  EXPECT_EQ(descriptor->byte_strides[1], 4);

  const holoscan::PayloadDescriptor payload_descriptor =
      holoscan::channel_internal::payload_descriptor_from_holomq_tensor_descriptor(
          descriptor.value());
  EXPECT_EQ(payload_descriptor.payload_kind, holoscan::ChannelPayloadKind::kBufferHandle);
  EXPECT_EQ(payload_descriptor.flags, descriptor->flags);
  EXPECT_EQ(payload_descriptor.byte_size, descriptor->byte_size);
  EXPECT_TRUE(payload_descriptor.is_buffer_handle());
  EXPECT_FALSE(payload_descriptor.is_local_buffer());
  EXPECT_EQ(payload_descriptor.buffer.abi_version(),
            holoscan::BufferHandlePayloadReference::kAbiVersion);
  EXPECT_EQ(payload_descriptor.buffer.payload_size(),
            sizeof(holoscan::channel_internal::HoloMQPayloadReferenceCarrier));
  const auto decoded_handle =
      holoscan::channel_internal::holomq_buffer_handle_from_payload_reference(
          payload_descriptor.buffer);
  ASSERT_TRUE(decoded_handle) << holoscan::runtime_error_message(decoded_handle.error());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(decoded_handle->memory_type(), handle->memory_type());
}

// Verifies schema-extracted Tensor metadata can be paired with a HoloMQ BufferHandle to build the
// same descriptor shape used by the cross-process import path, without requiring a live Tensor
// object at descriptor construction time.
TEST(ChannelTensorTest, HoloMQTensorDescriptorCanBeBuiltFromTensorMetadata) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const holomq::BufferHandle exported_handle = *handle;
  auto imported = fixture->context.import_buffer_unsynchronized(exported_handle);
  ASSERT_TRUE(imported.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::ConstBuffer imported_buffer = std::move(*imported);

  const std::array<int64_t, 2> shape{4, 4};
  const std::array<int64_t, 2> byte_strides{16, 4};
  auto descriptor = holoscan::channel_internal::make_holomq_tensor_descriptor(
      holoscan::TensorShapeView{shape.data(), shape.size()},
      DLDataType{kDLUInt, 32, 1},
      holoscan::TensorByteStridesSpan{byte_strides.data(), byte_strides.size()},
      kHoloMQTensorBufferBytes,
      exported_handle,
      /*read_only=*/true);
  ASSERT_TRUE(descriptor) << holoscan::runtime_error_message(descriptor.error());
  EXPECT_TRUE(descriptor->has_supported_abi());
  EXPECT_EQ(descriptor->rank, 2U);
  EXPECT_EQ(descriptor->memory_kind,
            holoscan::channel_internal::HoloMQTensorDescriptorMemoryKind::kHostPinned);
  EXPECT_EQ(descriptor->cuda_device_ordinal, -1);
  EXPECT_EQ(descriptor->byte_size, kHoloMQTensorBufferBytes);
  EXPECT_EQ(descriptor->byte_strides[0], 16);
  EXPECT_EQ(descriptor->byte_strides[1], 4);

  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
      descriptor.value(), std::move(imported_buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  EXPECT_TRUE(tensor->is_read_only());
  EXPECT_EQ(tensor->nbytes(), static_cast<int64_t>(kHoloMQTensorBufferBytes));
  EXPECT_EQ(tensor->shape_span()[0], 4);
  EXPECT_EQ(tensor->shape_span()[1], 4);
}

// Verifies descriptor endpoints carry the production PayloadDescriptor shape directly, including a
// real HoloMQ BufferHandle for the cross-process Tensor path.
//
//   descriptor_pub --[BufferHandle payload]--> descriptor_sub
TEST(ChannelTensorTest, DescriptorEndpointPublishesBufferHandlePayloadDescriptor) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_descriptor_publisher(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_descriptor_subscriber(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const holoscan::PayloadDescriptor payload{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .sequence = 42U,
      .byte_size = static_cast<std::uint64_t>(buffer.size_bytes()),
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      .buffer = holoscan::channel_internal::make_holomq_payload_reference(*handle)};

  auto published = publisher->publish(payload);
  ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());
  EXPECT_EQ(published.value(), 1U);

  auto received = subscriber->receive_payload_descriptor();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_TRUE(received->is_buffer_handle());
  EXPECT_FALSE(received->is_local_buffer());
  EXPECT_EQ(received->sequence, payload.sequence);
  EXPECT_EQ(received->byte_size, payload.byte_size);
  const auto decoded_handle =
      holoscan::channel_internal::holomq_buffer_handle_from_payload_reference(received->buffer);
  ASSERT_TRUE(decoded_handle) << holoscan::runtime_error_message(decoded_handle.error());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(decoded_handle->memory_type(), handle->memory_type());
}

// Verifies a descriptor plus imported HoloMQ ConstBuffer can reconstruct a read-only Tensor view
// with the expected shape, dtype, strides, and backing storage lifetime.
TEST(ChannelTensorTest, HoloMQConstBufferTensorCanBeWrappedFromDescriptor) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kHoloMQTensorPatternSeed));
  auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  const std::array<int64_t, 2> shape{4, 4};
  const std::array<int64_t, 2> byte_strides{16, 4};
  auto source_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape,
      DLDataType{kDLUInt, 32, 1},
      holoscan::TensorByteStridesSpan{byte_strides.data(), byte_strides.size()},
      std::move(buffer));
  ASSERT_TRUE(source_tensor) << holoscan::runtime_error_message(source_tensor.error());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const holomq::BufferHandle descriptor_handle = *handle;
  auto descriptor =
      holoscan::channel_internal::make_holomq_tensor_descriptor(*source_tensor, descriptor_handle);
  ASSERT_TRUE(descriptor) << holoscan::runtime_error_message(descriptor.error());
  source_tensor = holoscan::Tensor{};

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto imported_or = fixture->context.import_buffer_unsynchronized(descriptor_handle);
  ASSERT_TRUE(imported_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::ConstBuffer imported = std::move(*imported_or);
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
      descriptor.value(), std::move(imported));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  EXPECT_EQ(tensor->device().device_type, kDLCUDAHost);
  EXPECT_TRUE(tensor->is_read_only());
  EXPECT_EQ(tensor->shape_span()[0], 4);
  EXPECT_EQ(tensor->shape_span()[1], 4);
  EXPECT_EQ(tensor->strides()[0], 16);
  EXPECT_EQ(tensor->strides()[1], 4);
  EXPECT_TRUE(verify_host_pattern(
      tensor->data(), static_cast<std::size_t>(tensor->nbytes()), kHoloMQTensorPatternSeed));
  EXPECT_FALSE(fixture->pool.acquire().has_value());
  tensor = holoscan::Tensor{};
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies descriptor packing fails closed rather than truncating Tensor metadata when the rank is
// larger than the fixed descriptor capacity.
TEST(ChannelTensorTest, HoloMQTensorDescriptorRejectsUnsupportedRank) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  std::array<int64_t, holoscan::channel_internal::kHoloMQTensorDescriptorMaxRank + 1U> shape{};
  shape.fill(1);
  std::array<int64_t, holoscan::channel_internal::kHoloMQTensorDescriptorMaxRank + 1U>
      byte_strides{};
  byte_strides.fill(sizeof(std::uint32_t));
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape,
      DLDataType{kDLUInt, 32, 1},
      holoscan::TensorByteStridesSpan{byte_strides.data(), byte_strides.size()},
      std::move(buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto descriptor = holoscan::channel_internal::make_holomq_tensor_descriptor(*tensor, *handle);
  ASSERT_FALSE(descriptor);
  EXPECT_EQ(descriptor.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies a HoloMQ-pool-backed Tensor can pass through the Tensor channel without copying data,
// and that the HoloMQ pool block stays checked out until the received Tensor handle is released.
//
//   publisher --[sidecar, pinned-host pool]--> subscriber
//   Received data pointer must equal the original pool buffer pointer (zero-copy).
TEST(ChannelTensorTest, HoloMQBufferTensorChannelEmitIsZeroCopy) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = create_tensor_publisher(runtime.value(), channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kHoloMQTensorPatternSeed));
  void* const original_data_pointer = buffer.ptr();

  const std::array<int64_t, 1> shape{16};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());

  auto emitted = publisher->emit(std::move(tensor).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto received_tensor = std::move(received).value();
  EXPECT_EQ(received_tensor.data(), original_data_pointer);
  EXPECT_TRUE(verify_host_pattern(received_tensor.data(),
                                  static_cast<std::size_t>(received_tensor.nbytes()),
                                  kHoloMQTensorPatternSeed));

  EXPECT_FALSE(fixture->pool.acquire().has_value());
  received_tensor = holoscan::Tensor{};
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies BufferHandle publication retains the producer's pool lease while the transport sample
// is queued. The subscriber intentionally delays import so stream-associated ref-count deferral
// cannot mask a missing transport-sample lease.
//
//   publisher --[kBufferHandle]--> subscriber
//   An auxiliary trigger_tensor forces async pool-release observation after receive.
TEST(ChannelTensorTest, HoloMQBufferTensorEmitRetainsOwnerUntilSubscriberReleasesSample) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  auto trigger_pool = make_holomq_pinned_host_pool(fixture->context);
  ASSERT_TRUE(trigger_pool.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto trigger_buffer = trigger_pool->acquire();
  ASSERT_TRUE(trigger_buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer trigger_source_buffer = std::move(*trigger_buffer);
  const std::array<int64_t, 1> shape{16};
  auto trigger_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(trigger_source_buffer));
  ASSERT_TRUE(trigger_tensor) << holoscan::runtime_error_message(trigger_tensor.error());

  holoscan::ChannelOptions options{};
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto buffer = fixture->pool.acquire();
  ASSERT_TRUE(buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer source_buffer = std::move(*buffer);
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(source_buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());

  auto emitted = publisher->emit(std::move(tensor).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  EXPECT_FALSE(fixture->pool.acquire().has_value());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_FALSE(fixture->pool.acquire().has_value());
  received = holoscan::Tensor{};
  auto trigger_emitted = publisher->emit(std::move(trigger_tensor).value());
  ASSERT_TRUE(trigger_emitted) << holoscan::runtime_error_message(trigger_emitted.error());

  std::optional<holomq::Buffer> reused;
  EXPECT_TRUE(
      wait_until([&fixture, &reused]() { return try_acquire_buffer(fixture->pool, reused); },
                 std::chrono::seconds(2)));
}

// Verifies BufferHandle receive defers the producer fence to the Tensor stream contract: receive
// remains nonblocking while S0 is blocked, S1 sees producer data only after the explicit wait, and
// the exportable pool slot remains unavailable until S1 finishes its consumer work.
//
//   publisher --[kBufferHandle, device pool, S0 blocked]--> subscriber (S1)
//   S1 captures S0 fence via synchronize_for_consumer_stream; slot held until S1 drains.
TEST(ChannelTensorTest, HoloMQBufferTensorChannelOrdersProducerAndConsumerStreams) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  auto trigger_pool = fixture->context.create_device_buffer_pool(kHoloMQTensorBufferBytes,
                                                                 kHoloMQTensorBufferCount);
  ASSERT_TRUE(trigger_pool.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto trigger_buffer = trigger_pool->acquire();
  ASSERT_TRUE(trigger_buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer trigger_source_buffer = std::move(*trigger_buffer);
  const std::array<int64_t, 1> shape{16};
  auto trigger_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(trigger_source_buffer));
  ASSERT_TRUE(trigger_tensor) << holoscan::runtime_error_message(trigger_tensor.error());

  holoscan::test::UniqueCudaStream producer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  holoscan::test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);

  holoscan::test::BlockingHostState producer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                producer_stream.get(), holoscan::test::blocking_host_callback, &producer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard producer_guard(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(producer_state));

  auto buffer = fixture->pool.acquire();
  ASSERT_TRUE(buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer source_buffer = std::move(*buffer);
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(source_buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  auto tensor_data = tensor->mutable_data();
  ASSERT_TRUE(tensor_data) << holoscan::runtime_error_message(tensor_data.error());
  constexpr std::uint8_t kPattern = 0x5aU;
  ASSERT_EQ(
      cudaMemsetAsync(*tensor_data, kPattern, kHoloMQTensorBufferBytes, producer_stream.get()),
      cudaSuccess);
  ASSERT_TRUE(tensor->set_readiness_stream(producer_stream.get()));

  holoscan::ChannelOptions options{};
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // Key setup: tensor carries an S0 fence; receive must return immediately while S0 blocks.
  auto emitted = publisher->emit(std::move(tensor).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto received_tensor = std::move(received).value();
  EXPECT_TRUE(received_tensor.has_readiness_stream_contract());
  auto synchronized = received_tensor.synchronize_for_consumer_stream(consumer_stream.get());
  ASSERT_TRUE(synchronized) << holoscan::runtime_error_message(synchronized.error());

  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, kHoloMQTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> host(
      static_cast<std::uint8_t*>(raw_host));
  ASSERT_EQ(cudaMemcpyAsync(host.get(),
                            received_tensor.data(),
                            kHoloMQTensorBufferBytes,
                            cudaMemcpyDeviceToHost,
                            consumer_stream.get()),
            cudaSuccess);

  holoscan::test::BlockingHostState consumer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                consumer_stream.get(), holoscan::test::blocking_host_callback, &consumer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard consumer_guard(consumer_state);
  ASSERT_TRUE(received_tensor.set_deallocation_stream(consumer_stream.get()));
  received_tensor = holoscan::Tensor{};
  auto trigger_emitted = publisher->emit(std::move(trigger_tensor).value());
  ASSERT_TRUE(trigger_emitted) << holoscan::runtime_error_message(trigger_emitted.error());
  // Key assertion: pool slot unavailable even after received_tensor is dropped, while
  // consumer_stream is still blocked — deallocation stream must complete first.
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(consumer_state));
  for (std::size_t i = 0; i < kHoloMQTensorBufferBytes; ++i) {
    EXPECT_EQ(host.get()[i], kPattern);
  }
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  consumer_guard.dismiss();
  holoscan::test::release_blocking_host(consumer_state);
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  std::optional<holomq::Buffer> reused;
  EXPECT_TRUE(
      wait_until([&fixture, &reused]() { return try_acquire_buffer(fixture->pool, reused); },
                 std::chrono::seconds(2)));
}

// Verifies a read-only imported Tensor can be relayed without releasing its pool claim before the
// downstream subscriber imports the queued BufferHandle.
//
//   upstream_pub --[kBufferHandle]--> upstream_sub --[relay]--> downstream_pub
//                                                           \--> downstream_sub
//   fixture->pool slot must stay checked out until downstream_sub releases the Tensor.
TEST(ChannelTensorTest, HoloMQImportedTensorRelayRetainsLeaseUntilDownstreamReceive) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  auto upstream_trigger_pool = make_holomq_pinned_host_pool(fixture->context);
  ASSERT_TRUE(upstream_trigger_pool.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto upstream_trigger_buffer = upstream_trigger_pool->acquire();
  ASSERT_TRUE(upstream_trigger_buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer upstream_trigger_source_buffer = std::move(*upstream_trigger_buffer);
  auto downstream_trigger_pool = make_holomq_pinned_host_pool(fixture->context);
  ASSERT_TRUE(downstream_trigger_pool.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto downstream_trigger_buffer = downstream_trigger_pool->acquire();
  ASSERT_TRUE(downstream_trigger_buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer downstream_trigger_source_buffer = std::move(*downstream_trigger_buffer);
  const std::array<int64_t, 1> shape{16};
  auto upstream_trigger_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(upstream_trigger_source_buffer));
  ASSERT_TRUE(upstream_trigger_tensor)
      << holoscan::runtime_error_message(upstream_trigger_tensor.error());
  auto downstream_trigger_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(downstream_trigger_source_buffer));
  ASSERT_TRUE(downstream_trigger_tensor)
      << holoscan::runtime_error_message(downstream_trigger_tensor.error());

  holoscan::ChannelOptions options{};
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string upstream_channel_name = make_unique_channel_name();
  auto upstream_publisher =
      create_tensor_publisher(runtime.value(), upstream_channel_name, options);
  ASSERT_TRUE(upstream_publisher) << holoscan::runtime_error_message(upstream_publisher.error());
  auto upstream_subscriber =
      runtime->create_subscriber<holoscan::Tensor>(upstream_channel_name, options);
  ASSERT_TRUE(upstream_subscriber) << holoscan::runtime_error_message(upstream_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(upstream_publisher.value(), std::chrono::seconds(2)));

  const std::string downstream_channel_name = make_unique_channel_name();
  auto downstream_publisher =
      create_tensor_publisher(runtime.value(), downstream_channel_name, options);
  ASSERT_TRUE(downstream_publisher)
      << holoscan::runtime_error_message(downstream_publisher.error());
  auto downstream_subscriber =
      runtime->create_subscriber<holoscan::Tensor>(downstream_channel_name, options);
  ASSERT_TRUE(downstream_subscriber)
      << holoscan::runtime_error_message(downstream_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(downstream_publisher.value(), std::chrono::seconds(2)));

  auto buffer = fixture->pool.acquire();
  ASSERT_TRUE(buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer source_buffer = std::move(*buffer);
  auto source = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(source_buffer));
  ASSERT_TRUE(source) << holoscan::runtime_error_message(source.error());
  auto emitted = upstream_publisher->emit(std::move(source).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());

  auto imported = upstream_subscriber->receive();
  ASSERT_TRUE(imported) << holoscan::runtime_error_message(imported.error());
  EXPECT_TRUE(imported->is_read_only());
  auto upstream_trigger_emitted =
      upstream_publisher->emit(std::move(upstream_trigger_tensor).value());
  ASSERT_TRUE(upstream_trigger_emitted)
      << holoscan::runtime_error_message(upstream_trigger_emitted.error());
  // Key setup: imported (read-only) Tensor is emitted downstream without dropping the
  // pool lease; the relay holds the lease across two channel hops.
  auto relayed = downstream_publisher->emit(std::move(imported).value());
  ASSERT_TRUE(relayed) << holoscan::runtime_error_message(relayed.error());

  // Key assertion: pool slot unavailable while downstream sample is in flight.
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  auto downstream = downstream_subscriber->receive();
  ASSERT_TRUE(downstream) << holoscan::runtime_error_message(downstream.error());
  downstream = holoscan::Tensor{};
  auto downstream_trigger_emitted =
      downstream_publisher->emit(std::move(downstream_trigger_tensor).value());
  ASSERT_TRUE(downstream_trigger_emitted)
      << holoscan::runtime_error_message(downstream_trigger_emitted.error());

  std::optional<holomq::Buffer> reused;
  EXPECT_TRUE(
      wait_until([&fixture, &reused]() { return try_acquire_buffer(fixture->pool, reused); },
                 std::chrono::seconds(2)));
}

// Verifies re-publishing an imported read-only Tensor preserves the original producer fence. The
// downstream Tensor must wait for S0 on S1 and keep the source pool slot unavailable until S1's
// consumer work completes.
//
//   upstream_pub --[kBufferHandle, S0 blocked]--> upstream_sub --[relay]-->
//   downstream_pub --> downstream_sub (S1 must capture S0 fence; slot held until S1 drains)
TEST(ChannelTensorTest, HoloMQImportedTensorRelayPreservesStreamOrdering) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto upstream_trigger_pool = fixture->context.create_device_buffer_pool(kHoloMQTensorBufferBytes,
                                                                          kHoloMQTensorBufferCount);
  ASSERT_TRUE(upstream_trigger_pool.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto upstream_trigger_buffer = upstream_trigger_pool->acquire();
  ASSERT_TRUE(upstream_trigger_buffer.has_value());
  auto downstream_trigger_pool = fixture->context.create_device_buffer_pool(
      kHoloMQTensorBufferBytes, kHoloMQTensorBufferCount);
  ASSERT_TRUE(downstream_trigger_pool.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  auto downstream_trigger_buffer = downstream_trigger_pool->acquire();
  ASSERT_TRUE(downstream_trigger_buffer.has_value());

  const std::array<int64_t, 1> shape{16};
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer upstream_trigger_source = std::move(*upstream_trigger_buffer);
  auto upstream_trigger = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(upstream_trigger_source));
  ASSERT_TRUE(upstream_trigger) << holoscan::runtime_error_message(upstream_trigger.error());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer downstream_trigger_source = std::move(*downstream_trigger_buffer);
  auto downstream_trigger = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(downstream_trigger_source));
  ASSERT_TRUE(downstream_trigger) << holoscan::runtime_error_message(downstream_trigger.error());

  holoscan::test::UniqueCudaStream producer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  holoscan::test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);
  holoscan::test::BlockingHostState producer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                producer_stream.get(), holoscan::test::blocking_host_callback, &producer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard producer_guard(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(producer_state));

  auto buffer = fixture->pool.acquire();
  ASSERT_TRUE(buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer source_buffer = std::move(*buffer);
  auto source = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(source_buffer));
  ASSERT_TRUE(source) << holoscan::runtime_error_message(source.error());
  auto source_data = source->mutable_data();
  ASSERT_TRUE(source_data) << holoscan::runtime_error_message(source_data.error());
  constexpr std::uint8_t kPattern = 0x3cU;
  ASSERT_EQ(
      cudaMemsetAsync(*source_data, kPattern, kHoloMQTensorBufferBytes, producer_stream.get()),
      cudaSuccess);
  ASSERT_TRUE(source->set_readiness_stream(producer_stream.get()));

  holoscan::ChannelOptions options{};
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string upstream_name = make_unique_channel_name();
  auto upstream_publisher = create_tensor_publisher(runtime.value(), upstream_name, options);
  ASSERT_TRUE(upstream_publisher) << holoscan::runtime_error_message(upstream_publisher.error());
  auto upstream_subscriber = runtime->create_subscriber<holoscan::Tensor>(upstream_name, options);
  ASSERT_TRUE(upstream_subscriber) << holoscan::runtime_error_message(upstream_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(upstream_publisher.value(), std::chrono::seconds(2)));

  const std::string downstream_name = make_unique_channel_name();
  auto downstream_publisher = create_tensor_publisher(runtime.value(), downstream_name, options);
  ASSERT_TRUE(downstream_publisher)
      << holoscan::runtime_error_message(downstream_publisher.error());
  auto downstream_subscriber =
      runtime->create_subscriber<holoscan::Tensor>(downstream_name, options);
  ASSERT_TRUE(downstream_subscriber)
      << holoscan::runtime_error_message(downstream_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(downstream_publisher.value(), std::chrono::seconds(2)));

  // Key setup: source Tensor carries a readiness fence on S0 (blocked); relaying the
  // imported Tensor must propagate the S0 fence to the downstream consumer on S1.
  auto emitted = upstream_publisher->emit(std::move(source).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto imported = upstream_subscriber->receive();
  ASSERT_TRUE(imported) << holoscan::runtime_error_message(imported.error());
  auto relayed = downstream_publisher->emit(std::move(imported).value());
  ASSERT_TRUE(relayed) << holoscan::runtime_error_message(relayed.error());
  auto upstream_trigger_emitted = upstream_publisher->emit(std::move(upstream_trigger).value());
  ASSERT_TRUE(upstream_trigger_emitted)
      << holoscan::runtime_error_message(upstream_trigger_emitted.error());

  auto downstream = downstream_subscriber->receive();
  ASSERT_TRUE(downstream) << holoscan::runtime_error_message(downstream.error());
  auto downstream_tensor = std::move(downstream).value();
  auto synchronized = downstream_tensor.synchronize_for_consumer_stream(consumer_stream.get());
  ASSERT_TRUE(synchronized) << holoscan::runtime_error_message(synchronized.error());
  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, kHoloMQTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> host(
      static_cast<std::uint8_t*>(raw_host));
  ASSERT_EQ(cudaMemcpyAsync(host.get(),
                            downstream_tensor.data(),
                            kHoloMQTensorBufferBytes,
                            cudaMemcpyDeviceToHost,
                            consumer_stream.get()),
            cudaSuccess);

  holoscan::test::BlockingHostState consumer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                consumer_stream.get(), holoscan::test::blocking_host_callback, &consumer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard consumer_guard(consumer_state);
  ASSERT_TRUE(downstream_tensor.set_deallocation_stream(consumer_stream.get()));
  downstream_tensor = holoscan::Tensor{};
  auto downstream_trigger_emitted =
      downstream_publisher->emit(std::move(downstream_trigger).value());
  ASSERT_TRUE(downstream_trigger_emitted)
      << holoscan::runtime_error_message(downstream_trigger_emitted.error());
  // Key assertion: pool slot still unavailable while S0 and S1 are both blocked.
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(consumer_state));
  for (std::size_t i = 0; i < kHoloMQTensorBufferBytes; ++i) {
    EXPECT_EQ(host.get()[i], kPattern);
  }
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  consumer_guard.dismiss();
  holoscan::test::release_blocking_host(consumer_state);
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  std::optional<holomq::Buffer> reused;
  EXPECT_TRUE(
      wait_until([&fixture, &reused]() { return try_acquire_buffer(fixture->pool, reused); },
                 std::chrono::seconds(2)));
}

// Verifies direct Tensor channels can use the BufferHandle transfer mode to send a HoloMQ-backed
// Tensor across a process boundary without embedding the Tensor metadata in a FlatBuffer schema.
//
//   parent publisher --[kBufferHandle, IPC]--> child subscriber
//   Handshake via pipes: child signals ready (code=1), parent publishes, child validates.
TEST(ChannelTensorTest, HoloMQBufferTensorChannelMaterializesImportedTensorInChild) {
  const char* const child_exe = HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE not configured";

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  const std::string channel_name = make_unique_channel_name();

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    const std::string expected_bytes = std::to_string(kHoloMQTensorBufferBytes);
    const std::string expected_seed = std::to_string(kHoloMQTensorPatternSeed);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_NAME", channel_name.c_str(), 1));
    static_cast<void>(
        ::setenv("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_BYTES", expected_bytes.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_SEED", expected_seed.c_str(), 1));
    ::execl(child_exe, "channel_tensor_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  if (ready_code != 1U) {
    int status = 0;
    ASSERT_EQ(child_cleanup.wait(&status), child);
    ASSERT_TRUE(WIFEXITED(status));
    FAIL() << "child failed before subscriber-ready handshake; exit=" << WEXITSTATUS(status)
           << " validation code " << static_cast<unsigned>(ready_code);
  }

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kHoloMQTensorPatternSeed));

  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQTensorBufferBytes / sizeof(std::uint32_t))};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  const holoscan::Tensor source_tensor = std::move(tensor).value();
  auto exported_descriptor =
      holoscan::channel_internal::export_holomq_tensor_descriptor(source_tensor);
  ASSERT_TRUE(exported_descriptor) << holoscan::runtime_error_message(exported_descriptor.error());

  const holoscan::Sample<holoscan::Tensor> sample(
      source_tensor, {.sequence = 8080U, .capture_timestamp_ns = 123456789});

  std::string last_emit_error;
  bool child_exited_early = false;
  int child_status = 0;
  std::uint8_t early_child_code = 0xffU;
  const bool delivered = wait_until(
      [&publisher,
       &sample,
       &last_emit_error,
       child,
       &child_exited_early,
       &child_status,
       &early_child_code,
       &child_cleanup] {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit_sample(sample);
        if (published && published.value() >= 1U) {
          return true;
        }
        if (!published && published.error().code != holoscan::ErrorCode::kNotReady) {
          last_emit_error = holoscan::runtime_error_message(published.error());
        }
        const pid_t waited = ::waitpid(child, &child_status, WNOHANG);
        if (waited == child) {
          child_exited_early = true;
          static_cast<void>(
              read_exact(child_cleanup.result_fd(), &early_child_code, sizeof(early_child_code)));
          return true;
        }
        return false;
      },
      std::chrono::seconds(5));
  if (child_exited_early) {
    child_cleanup.close_result();
    ASSERT_TRUE(WIFEXITED(child_status));
    FAIL() << "child exited before parent published; exit=" << WEXITSTATUS(child_status)
           << " validation code " << static_cast<unsigned>(early_child_code);
  }
  ASSERT_TRUE(delivered) << "failed to publish Tensor BufferHandle payload to child: "
                         << last_emit_error;

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// Verifies the producer readiness generation crosses a real process boundary. The child enqueues
// its S1 wait and reports that fact before the parent releases deliberately blocked S0 work.
//
//   parent publisher --[kBufferHandle, IPC, device pool, S0 blocked]--> child subscriber
//   Child reports stream-wait enqueued (code=2) before parent releases S0; validates data.
TEST(ChannelTensorTest, HoloMQBufferTensorChannelOrdersStreamsAcrossProcesses) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  const char* const child_exe = HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE not configured";

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  const std::string channel_name = make_unique_channel_name();

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    const std::string expected_bytes = std::to_string(kHoloMQTensorBufferBytes);
    const std::string expected_seed = std::to_string(kHoloMQTensorPatternSeed);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_NAME", channel_name.c_str(), 1));
    static_cast<void>(
        ::setenv("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_BYTES", expected_bytes.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_SEED", expected_seed.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_DEVICE_ASYNC", "1", 1));
    ::execl(child_exe, "channel_tensor_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  holoscan::test::UniqueCudaStream producer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  holoscan::test::BlockingHostState producer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                producer_stream.get(), holoscan::test::blocking_host_callback, &producer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard producer_guard(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(producer_state));

  void* raw_pattern = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_pattern, kHoloMQTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> pattern(
      static_cast<std::uint8_t*>(raw_pattern));
  ASSERT_TRUE(
      write_host_pattern(pattern.get(), kHoloMQTensorBufferBytes, kHoloMQTensorPatternSeed));
  auto buffer = fixture->pool.acquire();
  ASSERT_TRUE(buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer source_buffer = std::move(*buffer);
  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQTensorBufferBytes / sizeof(std::uint32_t))};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(source_buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  auto tensor_data = tensor->mutable_data();
  ASSERT_TRUE(tensor_data) << holoscan::runtime_error_message(tensor_data.error());
  ASSERT_EQ(cudaMemcpyAsync(*tensor_data,
                            pattern.get(),
                            kHoloMQTensorBufferBytes,
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(tensor->set_readiness_stream(producer_stream.get()));

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();
  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  ASSERT_EQ(child_code, 1U) << "child failed before subscriber-ready handshake";

  const holoscan::Sample<holoscan::Tensor> sample(
      std::move(tensor).value(), {.sequence = 8081U, .capture_timestamp_ns = 123456790});
  std::string last_emit_error;
  const bool delivered = wait_until(
      [&publisher, &sample, &last_emit_error]() {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit_sample(sample);
        if (published && published.value() >= 1U) {
          return true;
        }
        if (!published && published.error().code != holoscan::ErrorCode::kNotReady) {
          last_emit_error = holoscan::runtime_error_message(published.error());
        }
        return false;
      },
      std::chrono::seconds(5));
  ASSERT_TRUE(delivered) << "failed to publish async Tensor BufferHandle: " << last_emit_error;

  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  ASSERT_EQ(child_code, 2U) << "child did not enqueue its consumer-stream wait";
  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();
  ASSERT_EQ(cudaStreamSynchronize(producer_stream.get()), cudaSuccess);

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// Verifies BufferHandle mode is explicit: direct Tensor channels reject ordinary same-process
// Tensor storage instead of silently falling back to local sidecar transfer.
TEST(ChannelTensorTest, HoloMQBufferTensorChannelRejectsNonHoloMQTensorStorage) {
  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &data, 0x5aU);

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_FALSE(emitted);
  EXPECT_EQ(emitted.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies the HoloMQ buffer-handle import path protects pool reuse across an imported Tensor:
// dropping the producer Tensor and imported Tensor does not make the pool slot available until the
// consumer release stream reaches the deferred ref-count release.
TEST(ChannelTensorTest, HoloMQImportedTensorReleaseWaitsForConsumerStream) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  holoscan::test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kHoloMQTensorPatternSeed));
  void* const original_data_pointer = buffer.ptr();

  // Keep the producer lease synchronous so this test isolates the imported ConstBuffer release
  // stream. HoloMQ has lower-level tests for export-stream deferred release.
  const auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const holomq::BufferHandle exported_handle = handle.value();

  const std::array<int64_t, 1> shape{16};
  auto source_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(buffer));
  ASSERT_TRUE(source_tensor) << holoscan::runtime_error_message(source_tensor.error());

  auto descriptor =
      holoscan::channel_internal::make_holomq_tensor_descriptor(*source_tensor, exported_handle);
  ASSERT_TRUE(descriptor) << holoscan::runtime_error_message(descriptor.error());

  auto imported_or = fixture->context.import_buffer(exported_handle, consumer_stream.get());
  ASSERT_TRUE(imported_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::ConstBuffer imported = std::move(*imported_or);
  auto imported_tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
      descriptor.value(), std::move(imported));
  ASSERT_TRUE(imported_tensor) << holoscan::runtime_error_message(imported_tensor.error());
  EXPECT_TRUE(imported_tensor->is_read_only());
  EXPECT_TRUE(verify_host_pattern(imported_tensor->data(),
                                  static_cast<std::size_t>(imported_tensor->nbytes()),
                                  kHoloMQTensorPatternSeed));

  holoscan::test::BlockingHostState consumer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                consumer_stream.get(), holoscan::test::blocking_host_callback, &consumer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard release_guard(consumer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(consumer_state));

  // Key setup: consumer_stream is blocked; both source and imported handles are dropped.
  EXPECT_FALSE(fixture->pool.acquire().has_value());
  source_tensor = holoscan::Tensor{};
  imported_tensor = holoscan::Tensor{};
  // Key assertion: pool slot still held even after both C++ handles are gone, because the
  // ConstBuffer's stream-deferred release cannot complete while consumer_stream is blocked.
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  release_guard.dismiss();
  holoscan::test::release_blocking_host(consumer_state);
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);

  auto reused = fixture->pool.acquire();
  ASSERT_TRUE(reused.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(reused->ptr(), original_data_pointer);
}

// Verifies blocking Tensor channels transfer the Tensor handle while preserving storage lifetime.
//
//   publisher --[blocking, sidecar]--> subscriber
TEST(ChannelTensorTest, BlockingPublishesAndReceivesTensorHandle) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = create_tensor_publisher(runtime.value(), channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 7U);

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  holoscan::Tensor received_tensor = std::move(received).value();
  EXPECT_EQ(received_tensor.data(), expected_data);
  EXPECT_EQ(received_tensor.nbytes(), 6);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  received_tensor = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies Tensor receive_sample returns both the Tensor handle and public sample metadata.
TEST(ChannelTensorTest, ReceiveSampleReturnsTensorAndPublicMetadata) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = create_tensor_publisher(runtime.value(), channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 5U);

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data.data(), expected_data);
  EXPECT_EQ(received->data.nbytes(), 6);
  EXPECT_EQ(received->metadata.payload_kind, holoscan::PayloadKind::kDescriptorBacked);
  EXPECT_TRUE(received->metadata.valid());
  EXPECT_EQ(received->metadata.sequence, 1U);
  EXPECT_EQ(received->metadata.byte_size, 6U);

  received->data = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies Tensor emit_sample preserves caller-provided public metadata while retaining the Tensor
// through the same sidecar path as value-only emit.
TEST(ChannelTensorTest, EmitSamplePreservesTensorMetadata) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = create_tensor_publisher(runtime.value(), channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 6U);

  holoscan::Sample<holoscan::Tensor> sample(std::move(tensor),
                                            {.sequence = 77U,
                                             .capture_timestamp_ns = 1234,
                                             .publish_timestamp_ns = 5678,
                                             .trace_id = 10U,
                                             .frame_id = 11U});
  auto emitted = publisher->emit_sample(std::move(sample));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data.data(), expected_data);
  EXPECT_EQ(received->metadata.sequence, 77U);
  EXPECT_EQ(received->metadata.capture_timestamp_ns, 1234);
  EXPECT_EQ(received->metadata.publish_timestamp_ns, 5678);
  EXPECT_EQ(received->metadata.source_clock_id, holoscan::ClockId{});
  EXPECT_EQ(received->metadata.trace_id, 10U);
  EXPECT_EQ(received->metadata.frame_id, 11U);
  EXPECT_EQ(received->metadata.byte_size, 6U);
  EXPECT_EQ(received->metadata.payload_kind, holoscan::PayloadKind::kDescriptorBacked);
  EXPECT_TRUE(received->metadata.valid());

  received->data = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies the manual compatibility facade publishes to the retained Tensor subscriber surface.
TEST(ChannelTensorTest, ManualPublisherEmitsToTensorSubscriber) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 8U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 9U);

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  EXPECT_TRUE(subscriber->has_samples().value());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  holoscan::Tensor received_tensor = std::move(received).value();
  EXPECT_EQ(received_tensor.data(), expected_data);
  EXPECT_EQ(received_tensor.nbytes(), 6);
  EXPECT_FALSE(subscriber->has_samples().value());

  received_tensor = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies the retained Tensor subscriber receives metadata from the manual compatibility facade.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherPreservesPublicMetadata) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 8U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 13U);

  auto emitted = publisher->emit(std::move(tensor),
                                 {.frame_id = 17U,
                                  .trace = holoscan::TraceContext::root(19U),
                                  .flags = holoscan::SampleFlags::kDegraded});
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data.data(), expected_data);
  EXPECT_EQ(static_cast<const std::uint8_t*>(received->data.data())[0], 13U);
  EXPECT_EQ(received->metadata.sequence, 1U);
  EXPECT_EQ(received->metadata.trace_id, 19U);
  EXPECT_EQ(received->metadata.frame_id, 17U);
  EXPECT_EQ(received->metadata.byte_size, 6U);
  EXPECT_EQ(received->metadata.payload_kind, holoscan::PayloadKind::kDescriptorBacked);
  EXPECT_TRUE(received->metadata.degraded());

  received->data = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies invalid manual authority and publisher-local bounds fail before endpoint publication.
// REQ: CHANNEL-API-003
// REQ: CHAN-071
TEST(ChannelTensorTest, ManualPublisherRejectsInvalidBindingAndCapacity) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto empty_name = runtime->bind_manual_channel(
      {.service_name = {}, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_FALSE(empty_name);
  EXPECT_EQ(empty_name.error().code, holoscan::ErrorCode::kInvalidArgument);

  for (const std::uint32_t capacity : {0U, std::numeric_limits<std::uint32_t>::max()}) {
    auto binding = runtime->bind_manual_channel({.service_name = make_unique_channel_name(),
                                                 .role = holoscan::ManualChannelRole::kPublisher});
    ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
    auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                            {.max_inflight_samples = capacity});
    ASSERT_FALSE(publisher);
    EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
  }
}

// Verifies passing manual authority by value consumes it on both successful and failed factories.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherConsumesBindingOnEveryFactoryOutcome) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto successful_binding =
      runtime->bind_manual_channel({.service_name = make_unique_channel_name(),
                                    .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(successful_binding) << holoscan::runtime_error_message(successful_binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(successful_binding).value(),
                                                          {.max_inflight_samples = 1U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  // The first factory consumed the expected's binding value; the second access deliberately
  // exercises the moved-from authority object retained inside the expected.
  auto reused_after_success = holoscan::make_manual_tensor_publisher(
      std::move(successful_binding).value(),  // NOLINT(bugprone-use-after-move)
      {.max_inflight_samples = 1U});
  ASSERT_FALSE(reused_after_success);
  EXPECT_EQ(reused_after_success.error().code, holoscan::ErrorCode::kInvalidArgument);

  auto rejected_binding =
      runtime->bind_manual_channel({.service_name = make_unique_channel_name(),
                                    .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(rejected_binding) << holoscan::runtime_error_message(rejected_binding.error());
  auto rejected = holoscan::make_manual_tensor_publisher(std::move(rejected_binding).value(),
                                                         {.max_inflight_samples = 0U});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument);
  // A rejected factory still consumes its by-value binding before validating publisher options.
  auto reused_after_failure = holoscan::make_manual_tensor_publisher(
      std::move(rejected_binding).value(),  // NOLINT(bugprone-use-after-move)
      {.max_inflight_samples = 1U});
  ASSERT_FALSE(reused_after_failure);
  EXPECT_EQ(reused_after_failure.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies cold allocation and allocator-length failures are checked and publish no service state.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherMapsColdResourceFailuresAndAllowsFreshRetry) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  using FaultInjector = void (*)(holoscan::ChannelRuntime&) noexcept;
  constexpr std::array<FaultInjector, 2> injectors{
      &holoscan::channel_internal::TensorPublisherTestAccess::
          fail_next_manual_sidecar_with_bad_alloc,
      &holoscan::channel_internal::TensorPublisherTestAccess::
          fail_next_manual_sidecar_with_length_error,
  };

  for (const FaultInjector inject : injectors) {
    const std::string channel_name = make_unique_channel_name();
    auto binding = runtime->bind_manual_channel(
        {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
    ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
    inject(runtime.value());

    auto failed = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                         {.max_inflight_samples = 2U});
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, holoscan::ErrorCode::kResourceExhausted);

    auto retry_binding = runtime->bind_manual_channel(
        {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
    ASSERT_TRUE(retry_binding) << holoscan::runtime_error_message(retry_binding.error());
    auto retry = holoscan::make_manual_tensor_publisher(std::move(retry_binding).value(),
                                                        {.max_inflight_samples = 2U});
    ASSERT_TRUE(retry) << holoscan::runtime_error_message(retry.error());
  }
}

// Verifies an arbitrary backend setup exception after provisional registration is checked and
// rolls back both the store and service identity before another endpoint can observe them.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherMapsBackendExceptionAndAllowsFreshRetry) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  holoscan::channel_internal::TensorPublisherTestAccess::fail_next_manual_backend_with_exception(
      runtime.value());

  auto failed = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                       {.max_inflight_samples = 2U});
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, holoscan::ErrorCode::kFailure);

  auto retry_binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(retry_binding) << holoscan::runtime_error_message(retry_binding.error());
  auto retry = holoscan::make_manual_tensor_publisher(std::move(retry_binding).value(),
                                                      {.max_inflight_samples = 2U});
  ASSERT_TRUE(retry) << holoscan::runtime_error_message(retry.error());
}

// Verifies an exception before HoloMQ retains the release callback rolls back the inserted Tensor
// record and makes the finite slot immediately reusable.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherRollsBackReleaseRegistrationException) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 1U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto failed_counter = std::make_shared<std::atomic<int>>(0);
  void* failed_data = nullptr;
  auto failed_tensor = make_counting_tensor(failed_counter, &failed_data, 31U);
  holoscan::expected<std::uint32_t, holoscan::Error> failed;
  {
    // Key setup: allocation failure injected at the HoloMQ release-callback registry step.
    // The sidecar insertion and transport loan are allocation-free after setup; the first
    // allocation is HoloMQ preparing the release-callback registry entry.
    const holoscan::test::ScopedAllocationFailure fail_next_allocation{0U};
    failed = publisher->emit(std::move(failed_tensor));
  }
  // Key assertion: emit fails with kResourceExhausted and the Tensor is immediately destroyed.
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_EQ(failed_counter->load(std::memory_order_relaxed), 1);

  auto retry_counter = std::make_shared<std::atomic<int>>(0);
  void* retry_data = nullptr;
  auto retry = publisher->emit(make_counting_tensor(retry_counter, &retry_data, 33U));
  ASSERT_TRUE(retry) << holoscan::runtime_error_message(retry.error());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  holoscan::Tensor received_tensor = std::move(received).value();
  EXPECT_EQ(received_tensor.data(), retry_data);
  received_tensor = holoscan::Tensor{};
  EXPECT_EQ(retry_counter->load(std::memory_order_relaxed), 1);
}

// A known no-peer state rejects before transport submission and sidecar insertion, releases the
// moved Tensor, and leaves the finite slot available to a later new logical sample.
// REQ: CHAN-067
TEST(ChannelTensorTest, ManualPublisherNoPeerRejectionReleasesFiniteSidecarCapacity) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 1U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  auto rejected_counter = std::make_shared<std::atomic<int>>(0);
  void* rejected_data = nullptr;
  auto rejected = publisher->emit(make_counting_tensor(rejected_counter, &rejected_data, 41U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kNotReady);
  EXPECT_EQ(rejected_counter->load(std::memory_order_relaxed), 1);

  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto retry_counter = std::make_shared<std::atomic<int>>(0);
  void* retry_data = nullptr;
  auto retried = publisher->emit(make_counting_tensor(retry_counter, &retry_data, 42U));
  ASSERT_TRUE(retried) << holoscan::runtime_error_message(retried.error());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data(), retry_data);
  *received = holoscan::Tensor{};
  EXPECT_EQ(retry_counter->load(std::memory_order_relaxed), 1);
}

// Verifies the public manual facade enforces its isolated logical record bound without growth.
// REQ: CHANNEL-API-003
// REQ: CHAN-071
TEST(ChannelTensorTest, ManualPublisherReportsCapacityExhaustionAndReusesReleasedRecord) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 1U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto first_counter = std::make_shared<std::atomic<int>>(0);
  void* first_data = nullptr;
  auto first = publisher->emit(make_counting_tensor(first_counter, &first_data, 21U));
  ASSERT_TRUE(first) << holoscan::runtime_error_message(first.error());

  auto second_counter = std::make_shared<std::atomic<int>>(0);
  void* second_data = nullptr;
  auto exhausted = publisher->emit(make_counting_tensor(second_counter, &second_data, 22U));
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_EQ(second_counter->load(std::memory_order_relaxed), 1);

  auto received_first = subscriber->receive();
  ASSERT_TRUE(received_first) << holoscan::runtime_error_message(received_first.error());
  EXPECT_EQ(received_first->data(), first_data);
  received_first = holoscan::Tensor{};
  EXPECT_EQ(first_counter->load(std::memory_order_relaxed), 1);

  auto third_counter = std::make_shared<std::atomic<int>>(0);
  void* third_data = nullptr;
  auto third = publisher->emit(make_counting_tensor(third_counter, &third_data, 23U));
  ASSERT_TRUE(third) << holoscan::runtime_error_message(third.error());
  auto received_third = subscriber->receive();
  ASSERT_TRUE(received_third) << holoscan::runtime_error_message(received_third.error());
  EXPECT_EQ(received_third->data(), third_data);
  received_third = holoscan::Tensor{};
  EXPECT_EQ(third_counter->load(std::memory_order_relaxed), 1);
}

// Verifies separately minted authority cannot replace or alias an active manual publisher.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherRejectsDuplicateActiveService) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto first_binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  auto second_binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(first_binding) << holoscan::runtime_error_message(first_binding.error());
  ASSERT_TRUE(second_binding) << holoscan::runtime_error_message(second_binding.error());

  auto first = holoscan::make_manual_tensor_publisher(std::move(first_binding).value(),
                                                      {.max_inflight_samples = 2U});
  ASSERT_TRUE(first) << holoscan::runtime_error_message(first.error());
  auto duplicate = holoscan::make_manual_tensor_publisher(std::move(second_binding).value(),
                                                          {.max_inflight_samples = 2U});
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, holoscan::ErrorCode::kDuplicateName);
}

// Verifies the provisional manual sidecar domain does not guess an unbounded fanout topology.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherAdmitsExactlyOneSubscriber) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 2U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  auto first = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(first) << holoscan::runtime_error_message(first.error());
  auto second = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_FALSE(second);
  EXPECT_EQ(second.error().code, holoscan::ErrorCode::kResourceExhausted);
}

// Verifies a retired manual service cannot be rebound while an older subscriber still retains its
// sidecar generation. Baseline deliberately reserves the name until ChannelRuntime destruction.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherServiceNameIsOneShotWithinRuntime) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  std::optional<holoscan::ChannelSubscriber<holoscan::Tensor>> retained_subscriber;

  {
    auto binding = runtime->bind_manual_channel(
        {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
    ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
    auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                            {.max_inflight_samples = 2U});
    ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
    auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
    ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
    retained_subscriber.emplace(std::move(subscriber).value());
    ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));
  }

  auto replacement_binding = runtime->bind_manual_channel(
      {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(replacement_binding) << holoscan::runtime_error_message(replacement_binding.error());
  auto replacement = holoscan::make_manual_tensor_publisher(std::move(replacement_binding).value(),
                                                            {.max_inflight_samples = 2U});
  ASSERT_FALSE(replacement);
  EXPECT_EQ(replacement.error().code, holoscan::ErrorCode::kDuplicateName);
  EXPECT_TRUE(retained_subscriber.has_value());
}

// Verifies retirement removes the capacity-sized store while the service tombstone rejects every
// later Tensor endpoint creation path, even when no subscriber retained the old store.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherRetirementRejectsLateTensorEndpointCreation) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  {
    auto binding = runtime->bind_manual_channel(
        {.service_name = channel_name, .role = holoscan::ManualChannelRole::kPublisher});
    ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
    auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                            {.max_inflight_samples = 64U});
    ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  }

  auto late_subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_FALSE(late_subscriber);
  EXPECT_EQ(late_subscriber.error().code, holoscan::ErrorCode::kDuplicateName);

  auto late_private_publisher = create_tensor_publisher(runtime.value(), channel_name);
  ASSERT_FALSE(late_private_publisher);
  EXPECT_EQ(late_private_publisher.error().code, holoscan::ErrorCode::kDuplicateName);
}

// Verifies manual publication rejects capture-clock claims that require catalog authority.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherRejectsCaptureTime) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto binding = runtime->bind_manual_channel({.service_name = make_unique_channel_name(),
                                               .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto publisher = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 2U});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  auto value_result =
      publisher->emit(holoscan::Tensor{},
                      {.capture_time = holoscan::TimePoint{
                           .timestamp_ns = 7, .clock_id = holoscan::ClockId{.value = 1U}}});
  ASSERT_FALSE(value_result);
  EXPECT_EQ(value_result.error().code, holoscan::ErrorCode::kInvalidArgument);

  holoscan::SampleMetadata metadata{};
  metadata.capture_timestamp_ns = 8;
  metadata.source_clock_id = holoscan::ClockId{.value = 1U};
  auto sample_result =
      publisher->emit_sample(holoscan::Sample<holoscan::Tensor>(holoscan::Tensor{}, metadata));
  ASSERT_FALSE(sample_result);
  EXPECT_EQ(sample_result.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies moved-from publisher state takes precedence over unsupported capture-time metadata for
// both concise and sample-aware publication paths.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherReportsClosedBeforeCaptureTimeValidation) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto binding = runtime->bind_manual_channel({.service_name = make_unique_channel_name(),
                                               .role = holoscan::ManualChannelRole::kPublisher});
  ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
  auto created = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                        {.max_inflight_samples = 2U});
  ASSERT_TRUE(created) << holoscan::runtime_error_message(created.error());

  holoscan::ManualTensorPublisher source = std::move(created).value();
  const holoscan::ManualTensorPublisher retained(std::move(source));
  const holoscan::TimePoint capture_time{.timestamp_ns = 7,
                                         .clock_id = holoscan::ClockId{.value = 1U}};

  // The moved-from source is deliberately reused to prove closed-state precedence.
  auto value_result = source.emit(  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
      holoscan::Tensor{},
      {.capture_time = capture_time});
  ASSERT_FALSE(value_result);
  EXPECT_EQ(value_result.error().code, holoscan::ErrorCode::kClosed);

  holoscan::SampleMetadata metadata{};
  metadata.capture_timestamp_ns = capture_time.timestamp_ns;
  metadata.source_clock_id = capture_time.clock_id;
  auto sample_result =
      source.emit_sample(  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
          holoscan::Sample<holoscan::Tensor>(holoscan::Tensor{}, metadata));
  ASSERT_FALSE(sample_result);
  EXPECT_EQ(sample_result.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_FALSE(retained.has_subscribers());
}

// Verifies the consumed manual binding retains runtime state after the runtime facade is destroyed.
// REQ: CHANNEL-API-003
TEST(ChannelTensorTest, ManualPublisherRetainsRuntimeAuthority) {
  std::optional<holoscan::ManualTensorPublisher> publisher;
  {
    auto runtime = holoscan::ChannelRuntime::create();
    ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
    auto binding = runtime->bind_manual_channel({.service_name = make_unique_channel_name(),
                                                 .role = holoscan::ManualChannelRole::kPublisher});
    ASSERT_TRUE(binding) << holoscan::runtime_error_message(binding.error());
    auto created = holoscan::make_manual_tensor_publisher(std::move(binding).value(),
                                                          {.max_inflight_samples = 2U});
    ASSERT_TRUE(created) << holoscan::runtime_error_message(created.error());
    publisher.emplace(std::move(created).value());
  }

  auto refreshed = publisher->update_connections();
  EXPECT_TRUE(refreshed) << holoscan::runtime_error_message(refreshed.error());
}

// Verifies warmed-up 1:1 blocking Tensor emit/receive cycles do not allocate on the host heap.
TEST(ChannelTensorTest, BlockingOneToOneHotPathDoesNotAllocateAfterWarmup) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = create_tensor_publisher(runtime.value(), channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  std::array<std::shared_ptr<std::atomic<int>>, 4> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 4> data{};
  std::array<holoscan::Tensor, 4> tensors{
      make_counting_tensor(destroyed_counters[0], &data[0], 1U),
      make_counting_tensor(destroyed_counters[1], &data[1], 2U),
      make_counting_tensor(destroyed_counters[2], &data[2], 3U),
      make_counting_tensor(destroyed_counters[3], &data[3], 4U),
  };

  ASSERT_TRUE(publisher->emit(holoscan::Tensor(tensors[0])));
  auto warmup = subscriber->receive();
  ASSERT_TRUE(warmup) << holoscan::runtime_error_message(warmup.error());
  holoscan::Tensor warmup_tensor = std::move(warmup).value();
  ASSERT_EQ(warmup_tensor.data(), data[0]);
  warmup_tensor = holoscan::Tensor{};

  std::size_t allocation_count = 0U;
  bool all_cycles_succeeded = true;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t index = 1U; index < tensors.size(); ++index) {
      auto emitted = publisher->emit(holoscan::Tensor(tensors[index]));
      all_cycles_succeeded = all_cycles_succeeded && emitted.has_value() && emitted.value() == 1U;

      auto received = subscriber->receive();
      all_cycles_succeeded = all_cycles_succeeded && received.has_value();
      if (received) {
        const holoscan::Tensor received_tensor = std::move(received).value();
        all_cycles_succeeded = all_cycles_succeeded && received_tensor.data() == data[index];
      }
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  EXPECT_EQ(allocation_count, 0U);
}

// The data and protected-terminal services are observed independently. A final data publication
// that becomes visible after the receiver's first data probe must still drain before EOS.
// REQ: CHAN-136
TEST(ChannelTensorTest, ProtectedTerminalRechecksDataPublishedAfterInitialProbe) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.expected_recipient_count = 1U;
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 1U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  holoscan::channel_internal::TensorPublisherTestAccess::ReceiveProbeBarrier barrier;
  holoscan::channel_internal::TensorPublisherTestAccess::
      attach_receive_probe_barrier_to_next_subscriber_for_test(barrier);
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto receive_future =
      std::async(std::launch::async, [&subscriber] { return subscriber->receive_sample(); });
  const bool first_data_probe_completed = barrier.reached.try_acquire_for(std::chrono::seconds(2));

  std::optional<holoscan::expected<std::uint32_t, holoscan::Error>> emitted;
  std::optional<holoscan::expected<std::uint32_t, holoscan::Error>> terminal_emitted;
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  if (first_data_probe_completed) {
    holoscan::SampleMetadata data_metadata{};
    data_metadata.sequence = 1U;
    emitted.emplace(publisher->emit_sample(holoscan::Sample<holoscan::Tensor>(
        make_counting_tensor(destroyed, &expected_data, 0x39U), data_metadata)));

    holoscan::SampleMetadata terminal_metadata{};
    terminal_metadata.sequence = 2U;
    terminal_metadata.payload_kind = holoscan::PayloadKind::kDescriptorBacked;
    terminal_emitted.emplace(
        holoscan::channel_internal::TensorPublisherTestAccess::emit_end_of_stream(
            publisher.value(), terminal_metadata));
  }
  barrier.resume.release();

  auto received = receive_future.get();
  ASSERT_TRUE(first_data_probe_completed);
  ASSERT_TRUE(emitted.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& emitted_result = *emitted;
  ASSERT_TRUE(emitted_result) << holoscan::runtime_error_message(emitted_result.error());
  EXPECT_EQ(*emitted_result, 1U);
  ASSERT_TRUE(terminal_emitted.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& terminal_emitted_result = *terminal_emitted;
  ASSERT_TRUE(terminal_emitted_result)
      << holoscan::runtime_error_message(terminal_emitted_result.error());
  EXPECT_EQ(*terminal_emitted_result, 1U);

  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->metadata.sequence, 1U);
  EXPECT_EQ(received->data.data(), expected_data);
  EXPECT_EQ(holoscan::detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(
                *subscriber),
            0U);

  auto end_of_stream = subscriber->receive_sample();
  ASSERT_FALSE(end_of_stream);
  EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_EQ(holoscan::detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(
                *subscriber),
            2U);

  auto repeated = subscriber->receive_sample();
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_EQ(holoscan::detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(
                *subscriber),
            2U);
}

// Verifies blocking Tensor fanout keeps storage alive until every expected subscriber releases its
// received Tensor handle.
//
//   publisher --[blocking, fanout=2]--> subscriber_a
//                                  \--> subscriber_b
//   Storage must survive both handles; destroyed only after subscriber_b releases.
TEST(ChannelTensorTest, BlockingFanoutKeepsTensorAliveUntilAllSubscribersReceive) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 2U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&subscriber_a.value(), &subscriber_b.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 11U);

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 2U);

  auto received_a = subscriber_a->receive();
  ASSERT_TRUE(received_a) << holoscan::runtime_error_message(received_a.error());
  holoscan::Tensor tensor_a = std::move(received_a).value();
  EXPECT_EQ(tensor_a.data(), expected_data);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  tensor_a = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  auto received_b = subscriber_b->receive();
  ASSERT_TRUE(received_b) << holoscan::runtime_error_message(received_b.error());
  holoscan::Tensor tensor_b = std::move(received_b).value();
  EXPECT_EQ(tensor_b.data(), expected_data);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  tensor_b = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies static fanout setup repeatedly rejects Tensor emit until all expected recipients are
// registered without leaking the capacity-one transport loan or bounded sidecar records.
//
//   publisher --[blocking, fanout=2 declared]--> subscriber  (only 1 of 2 subscribed)
//   Every emit returns kNotReady and destroys the moved-in Tensor; later complete fanout succeeds.
TEST(ChannelTensorTest, RejectsTensorEmitBeforeExpectedFanoutIsRegistered) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 1U;
  options.max_loaned_samples = 1U;
  options.expected_recipient_count = 2U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  constexpr std::uint32_t kAttempts = 16U;
  for (std::uint32_t attempt = 0U; attempt < kAttempts; ++attempt) {
    auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
    void* expected_data = nullptr;
    auto emitted =
        publisher->emit(make_counting_tensor(destroyed_counter, &expected_data, 17U + attempt));

    ASSERT_FALSE(emitted) << "unexpected success at attempt " << attempt;
    EXPECT_EQ(emitted.error().code, holoscan::ErrorCode::kNotReady);
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
    EXPECT_FALSE(subscriber->has_samples().value());
  }

  auto second_subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(second_subscriber) << holoscan::runtime_error_message(second_subscriber.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&subscriber.value(), &second_subscriber.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  auto accepted_counter = std::make_shared<std::atomic<int>>(0);
  void* accepted_data = nullptr;
  auto accepted = publisher->emit(make_counting_tensor(accepted_counter, &accepted_data, 99U));
  ASSERT_TRUE(accepted) << holoscan::runtime_error_message(accepted.error());
  EXPECT_EQ(*accepted, 2U);
  auto first_received = subscriber->receive();
  auto second_received = second_subscriber->receive();
  ASSERT_TRUE(first_received) << holoscan::runtime_error_message(first_received.error());
  ASSERT_TRUE(second_received) << holoscan::runtime_error_message(second_received.error());
  holoscan::Tensor first_tensor = std::move(*first_received);
  holoscan::Tensor second_tensor = std::move(*second_received);
  EXPECT_EQ(first_tensor.data(), accepted_data);
  EXPECT_EQ(second_tensor.data(), accepted_data);
  first_tensor = holoscan::Tensor{};
  second_tensor = holoscan::Tensor{};
  EXPECT_EQ(accepted_counter->load(std::memory_order_relaxed), 1);
}

// Verifies warmed-up blocking Tensor fanout cycles remain host-allocation-free.
//
//   publisher --[blocking, fanout=2]--> subscriber_a
//                                  \--> subscriber_b
//   Hot-path emit+receive cycles (after warmup) must not touch the host heap.
TEST(ChannelTensorTest, BlockingFanoutHotPathDoesNotAllocateAfterWarmup) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 2U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&subscriber_a.value(), &subscriber_b.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  std::array<std::shared_ptr<std::atomic<int>>, 4> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 4> data{};
  std::array<holoscan::Tensor, 4> tensors{
      make_counting_tensor(destroyed_counters[0], &data[0], 1U),
      make_counting_tensor(destroyed_counters[1], &data[1], 2U),
      make_counting_tensor(destroyed_counters[2], &data[2], 3U),
      make_counting_tensor(destroyed_counters[3], &data[3], 4U),
  };

  auto warmup_emit = publisher->emit(holoscan::Tensor(tensors[0]));
  ASSERT_TRUE(warmup_emit) << holoscan::runtime_error_message(warmup_emit.error());
  ASSERT_EQ(warmup_emit.value(), 2U);
  auto warmup_a = subscriber_a->receive();
  ASSERT_TRUE(warmup_a) << holoscan::runtime_error_message(warmup_a.error());
  auto warmup_b = subscriber_b->receive();
  ASSERT_TRUE(warmup_b) << holoscan::runtime_error_message(warmup_b.error());
  holoscan::Tensor warmup_tensor_a = std::move(warmup_a).value();
  holoscan::Tensor warmup_tensor_b = std::move(warmup_b).value();
  ASSERT_EQ(warmup_tensor_a.data(), data[0]);
  ASSERT_EQ(warmup_tensor_b.data(), data[0]);
  warmup_tensor_a = holoscan::Tensor{};
  warmup_tensor_b = holoscan::Tensor{};

  std::size_t allocation_count = 0U;
  bool all_cycles_succeeded = true;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t index = 1U; index < tensors.size(); ++index) {
      auto emitted = publisher->emit(holoscan::Tensor(tensors[index]));
      all_cycles_succeeded = all_cycles_succeeded && emitted.has_value() && emitted.value() == 2U;

      auto received_a = subscriber_a->receive();
      auto received_b = subscriber_b->receive();
      all_cycles_succeeded =
          all_cycles_succeeded && received_a.has_value() && received_b.has_value();
      if (received_a && received_b) {
        const holoscan::Tensor tensor_a = std::move(received_a).value();
        const holoscan::Tensor tensor_b = std::move(received_b).value();
        all_cycles_succeeded = all_cycles_succeeded && tensor_a.data() == data[index] &&
                               tensor_b.data() == data[index];
      }
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  EXPECT_EQ(allocation_count, 0U);
}

// Verifies depth-N drop-oldest delivery releases overwritten sidecar records while preserving the
// newest queued Tensor handles in FIFO order.
//
//   publisher --[drop-oldest, depth=3]--> subscriber
//   Five emits; first two are evicted; subscriber drains the remaining three in order.
TEST(ChannelTensorTest, DropOldestReleasesSupersededTensorHandles) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 3U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  std::array<std::shared_ptr<std::atomic<int>>, 5> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 5> data{};

  for (std::size_t index = 0U; index < destroyed_counters.size(); ++index) {
    auto emitted = publisher->emit(make_counting_tensor(
        destroyed_counters[index], &data[index], static_cast<std::uint8_t>(index + 1U)));
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  }

  EXPECT_TRUE(wait_until(
      [&destroyed_counters]() {
        return destroyed_counters[0]->load(std::memory_order_relaxed) == 1 &&
               destroyed_counters[1]->load(std::memory_order_relaxed) == 1;
      },
      std::chrono::seconds(2)));
  for (std::size_t index = 2U; index < destroyed_counters.size(); ++index) {
    EXPECT_EQ(destroyed_counters[index]->load(std::memory_order_relaxed), 0);
  }

  for (std::size_t index = 2U; index < destroyed_counters.size(); ++index) {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    holoscan::Tensor tensor = std::move(received).value();
    EXPECT_EQ(tensor.data(), data[index]);
    EXPECT_EQ(static_cast<const std::uint8_t*>(tensor.data())[0], index + 1U);
    tensor = holoscan::Tensor{};
    EXPECT_EQ(destroyed_counters[index]->load(std::memory_order_relaxed), 1);
  }
  EXPECT_FALSE(subscriber->has_samples().value());
}

// Verifies drop-oldest fanout releases only the slow recipient's overwritten claim after the fast
// recipient has already received and released the same Tensor.
//
//   publisher --[drop-oldest, depth=2, fanout=2]--> fast_subscriber (drains eagerly)
//                                              \--> slow_subscriber (falls behind)
//   Overflow on slow side evicts oldest; fast already holds a handle for that slot.
TEST(ChannelTensorTest, DropOldestFanoutReleasesSlowRecipientClaim) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 2U;
  options.expected_recipient_count = 2U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto fast_subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(fast_subscriber) << holoscan::runtime_error_message(fast_subscriber.error());
  auto slow_subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(slow_subscriber) << holoscan::runtime_error_message(slow_subscriber.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&fast_subscriber.value(), &slow_subscriber.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  std::array<std::shared_ptr<std::atomic<int>>, 3> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 3> data{};

  for (std::size_t index = 0U; index < 2U; ++index) {
    auto emitted = publisher->emit(make_counting_tensor(
        destroyed_counters[index], &data[index], static_cast<std::uint8_t>(index + 1U)));
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
    ASSERT_EQ(emitted.value(), 2U);
    auto received_fast = fast_subscriber->receive();
    ASSERT_TRUE(received_fast) << holoscan::runtime_error_message(received_fast.error());
    holoscan::Tensor fast_tensor = std::move(received_fast).value();
    EXPECT_EQ(fast_tensor.data(), data[index]);
    fast_tensor = holoscan::Tensor{};
    EXPECT_EQ(destroyed_counters[index]->load(std::memory_order_relaxed), 0);
  }

  auto emitted = publisher->emit(
      make_counting_tensor(destroyed_counters[2], &data[2], static_cast<std::uint8_t>(3U)));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 2U);
  EXPECT_TRUE(wait_until(
      [&destroyed_counters]() {
        return destroyed_counters[0]->load(std::memory_order_relaxed) == 1;
      },
      std::chrono::seconds(2)));
  EXPECT_EQ(destroyed_counters[1]->load(std::memory_order_relaxed), 0);
  EXPECT_EQ(destroyed_counters[2]->load(std::memory_order_relaxed), 0);

  for (std::size_t index = 1U; index < data.size(); ++index) {
    auto received_slow = slow_subscriber->receive();
    ASSERT_TRUE(received_slow) << holoscan::runtime_error_message(received_slow.error());
    holoscan::Tensor slow_tensor = std::move(received_slow).value();
    EXPECT_EQ(slow_tensor.data(), data[index]);
    slow_tensor = holoscan::Tensor{};
  }
  EXPECT_FALSE(slow_subscriber->has_samples().value());

  auto received_fast = fast_subscriber->receive();
  ASSERT_TRUE(received_fast) << holoscan::runtime_error_message(received_fast.error());
  holoscan::Tensor fast_tensor = std::move(received_fast).value();
  EXPECT_EQ(fast_tensor.data(), data[2]);
  fast_tensor = holoscan::Tensor{};

  EXPECT_TRUE(wait_until(
      [&destroyed_counters]() {
        return destroyed_counters[1]->load(std::memory_order_relaxed) == 1 &&
               destroyed_counters[2]->load(std::memory_order_relaxed) == 1;
      },
      std::chrono::seconds(2)));
}

// Stresses concurrent depth-one drop-oldest emit/receive with two subscribers. The publisher
// advances the sidecar store while receivers race to acquire descriptors; every emitted Tensor
// must eventually be destroyed exactly once.
//
//   publisher (thread) --[drop-oldest, fanout=2, depth=1]--> subscriber_a (thread)
//                                                         \--> subscriber_b (thread)
//   10000 iterations; destroyed_counter must equal emitted_count after all threads join.
TEST(ChannelTensorTest, DropOldestDepthOneConcurrentEmitReceiveReleasesSidecarRecords) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.expected_recipient_count = 2U;
  options.subscriber_queue_depth = 1U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&subscriber_a.value(), &subscriber_b.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  constexpr std::uint32_t kIterations = 10000U;
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  std::atomic<bool> start{false};
  std::atomic<bool> publisher_done{false};
  std::atomic<bool> publisher_failed{false};
  std::atomic<bool> subscriber_failed{false};
  std::atomic<int> publisher_error_code{0};
  std::atomic<int> subscriber_error_code{0};
  std::atomic<std::uint32_t> emitted_count{0U};
  std::atomic<std::uint32_t> received_count{0U};

  auto receive_until_drained = [&](auto& subscriber) {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!publisher_done.load(std::memory_order_acquire) || subscriber.has_samples().value()) {
      auto received = subscriber.receive();
      if (received) {
        const holoscan::Tensor tensor = std::move(received).value();
        if (tensor.nbytes() != 6) {
          subscriber_failed.store(true, std::memory_order_release);
        }
        received_count.fetch_add(1U, std::memory_order_relaxed);
        continue;
      }
      if (received.error().code == holoscan::ErrorCode::kNotReady) {
        std::this_thread::yield();
        continue;
      }
      subscriber_error_code.store(static_cast<int>(received.error().code),
                                  std::memory_order_release);
      subscriber_failed.store(true, std::memory_order_release);
      break;
    }
  };

  std::thread receive_a([&]() { receive_until_drained(subscriber_a.value()); });
  std::thread receive_b([&]() { receive_until_drained(subscriber_b.value()); });
  std::thread publish_thread([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::uint32_t index = 0U; index < kIterations; ++index) {
      void* data = nullptr;
      auto emitted = publisher->emit(
          make_counting_tensor(destroyed_counter, &data, static_cast<std::uint8_t>(index & 0xffU)));
      if (!emitted || emitted.value() != options.expected_recipient_count) {
        publisher_error_code.store(emitted ? -1 : static_cast<int>(emitted.error().code),
                                   std::memory_order_release);
        publisher_failed.store(true, std::memory_order_release);
        break;
      }
      emitted_count.fetch_add(1U, std::memory_order_relaxed);
      if ((index & 0x3fU) == 0U) {
        std::this_thread::yield();
      }
    }
    publisher_done.store(true, std::memory_order_release);
  });

  start.store(true, std::memory_order_release);
  publish_thread.join();
  receive_a.join();
  receive_b.join();

  EXPECT_FALSE(publisher_failed.load(std::memory_order_acquire))
      << "publisher error code " << publisher_error_code.load(std::memory_order_acquire);
  EXPECT_FALSE(subscriber_failed.load(std::memory_order_acquire))
      << "subscriber error code " << subscriber_error_code.load(std::memory_order_acquire);
  EXPECT_EQ(emitted_count.load(std::memory_order_relaxed), kIterations);
  EXPECT_GT(received_count.load(std::memory_order_relaxed), 0U);
  EXPECT_TRUE(wait_until(
      [&destroyed_counter, &emitted_count]() {
        return destroyed_counter->load(std::memory_order_relaxed) ==
               static_cast<int>(emitted_count.load(std::memory_order_relaxed));
      },
      std::chrono::seconds(2)));
}

// Verifies warmed-up 1:1 depth-one drop-oldest replacement stays host-allocation-free while
// skipping old unread descriptors.
TEST(ChannelTensorTest, DropOldestDepthOneReplacementHotPathDoesNotAllocateAfterWarmup) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 1U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  std::array<std::shared_ptr<std::atomic<int>>, 4> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 4> data{};
  std::array<holoscan::Tensor, 4> tensors{
      make_counting_tensor(destroyed_counters[0], &data[0], 1U),
      make_counting_tensor(destroyed_counters[1], &data[1], 2U),
      make_counting_tensor(destroyed_counters[2], &data[2], 3U),
      make_counting_tensor(destroyed_counters[3], &data[3], 4U),
  };

  auto warmup_emit = publisher->emit(holoscan::Tensor(tensors[0]));
  ASSERT_TRUE(warmup_emit) << holoscan::runtime_error_message(warmup_emit.error());
  auto warmup = subscriber->receive();
  ASSERT_TRUE(warmup) << holoscan::runtime_error_message(warmup.error());
  holoscan::Tensor warmup_tensor = std::move(warmup).value();
  ASSERT_EQ(warmup_tensor.data(), data[0]);
  warmup_tensor = holoscan::Tensor{};

  std::size_t allocation_count = 0U;
  bool all_cycles_succeeded = true;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t index = 1U; index < tensors.size(); ++index) {
      auto emitted = publisher->emit(holoscan::Tensor(tensors[index]));
      all_cycles_succeeded = all_cycles_succeeded && emitted.has_value() && emitted.value() == 1U;
    }

    auto received = subscriber->receive();
    all_cycles_succeeded = all_cycles_succeeded && received.has_value();
    if (received) {
      const holoscan::Tensor latest_tensor = std::move(received).value();
      all_cycles_succeeded = all_cycles_succeeded && latest_tensor.data() == data.back();
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  EXPECT_EQ(allocation_count, 0U);
}

// Verifies warmed-up fanout depth-one drop-oldest replacement remains host-allocation-free while
// the slow subscriber receives only the newest Tensor.
TEST(ChannelTensorTest, DropOldestDepthOneFanoutReplacementHotPathDoesNotAllocateAfterWarmup) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 1U;
  options.expected_recipient_count = 2U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto fast_subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(fast_subscriber) << holoscan::runtime_error_message(fast_subscriber.error());
  auto slow_subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(slow_subscriber) << holoscan::runtime_error_message(slow_subscriber.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&fast_subscriber.value(), &slow_subscriber.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  std::array<std::shared_ptr<std::atomic<int>>, 4> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 4> data{};
  std::array<holoscan::Tensor, 4> tensors{
      make_counting_tensor(destroyed_counters[0], &data[0], 1U),
      make_counting_tensor(destroyed_counters[1], &data[1], 2U),
      make_counting_tensor(destroyed_counters[2], &data[2], 3U),
      make_counting_tensor(destroyed_counters[3], &data[3], 4U),
  };

  auto warmup_emit = publisher->emit(holoscan::Tensor(tensors[0]));
  ASSERT_TRUE(warmup_emit) << holoscan::runtime_error_message(warmup_emit.error());
  ASSERT_EQ(warmup_emit.value(), 2U);
  auto warmup_fast = fast_subscriber->receive();
  ASSERT_TRUE(warmup_fast) << holoscan::runtime_error_message(warmup_fast.error());
  auto warmup_slow = slow_subscriber->receive();
  ASSERT_TRUE(warmup_slow) << holoscan::runtime_error_message(warmup_slow.error());
  holoscan::Tensor warmup_fast_tensor = std::move(warmup_fast).value();
  holoscan::Tensor warmup_slow_tensor = std::move(warmup_slow).value();
  ASSERT_EQ(warmup_fast_tensor.data(), data[0]);
  ASSERT_EQ(warmup_slow_tensor.data(), data[0]);
  warmup_fast_tensor = holoscan::Tensor{};
  warmup_slow_tensor = holoscan::Tensor{};

  std::size_t allocation_count = 0U;
  bool all_cycles_succeeded = true;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t index = 1U; index < tensors.size(); ++index) {
      auto emitted = publisher->emit(holoscan::Tensor(tensors[index]));
      all_cycles_succeeded = all_cycles_succeeded && emitted.has_value() && emitted.value() == 2U;

      auto received_fast = fast_subscriber->receive();
      all_cycles_succeeded = all_cycles_succeeded && received_fast.has_value();
      if (received_fast) {
        const holoscan::Tensor fast_tensor = std::move(received_fast).value();
        all_cycles_succeeded = all_cycles_succeeded && fast_tensor.data() == data[index];
      }
    }

    auto received_slow = slow_subscriber->receive();
    all_cycles_succeeded = all_cycles_succeeded && received_slow.has_value();
    if (received_slow) {
      const holoscan::Tensor latest_tensor = std::move(received_slow).value();
      all_cycles_succeeded = all_cycles_succeeded && latest_tensor.data() == data.back();
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  EXPECT_EQ(allocation_count, 0U);
}

// Verifies same-process Tensor sidecar lookup fails closed for descriptors with invalid ABI,
// payload kind, process id, token, or generation.
TEST(ChannelTensorTest, RejectsInvalidSameProcessDescriptors) {
  holoscan::TensorSidecarStore store(/*capacity=*/1U, /*expected_recipient_count=*/1U);

  auto expect_tensor_receive_error = [&](const holoscan::ChannelSampleDescriptor& descriptor) {
    auto received = store.acquire_sample(descriptor, /*recipient_index=*/0U);
    ASSERT_FALSE(received);
    EXPECT_EQ(received.error().code, holoscan::ErrorCode::kReceiveError);
  };

  auto unexpected_payload = make_descriptor(21U);
  unexpected_payload.payload_kind = holoscan::ChannelPayloadKind::kInlinePayload;
  expect_tensor_receive_error(unexpected_payload);

  auto wrong_process = make_descriptor(22U);
  wrong_process.process_id = static_cast<std::uint64_t>(::getpid()) + 1U;
  expect_tensor_receive_error(wrong_process);

  auto empty_token = make_descriptor(23U);
  empty_token.sidecar_token = 0U;
  expect_tensor_receive_error(empty_token);

  auto stale_generation = make_descriptor(24U);
  stale_generation.sidecar_token = 1U;
  stale_generation.generation = 99U;
  expect_tensor_receive_error(stale_generation);

  auto unsupported_abi = make_descriptor(25U);
  unsupported_abi.abi_version = holoscan::kChannelSampleDescriptorAbiVersion + 1U;
  expect_tensor_receive_error(unsupported_abi);
}

// Verifies the same-process sidecar accepts an empty Tensor without inventing a backing address,
// rejects a nonempty Tensor with the same null address, and returns its finite record after each
// terminal receive/release path.
TEST(ChannelTensorTest, SidecarAcceptsZeroByteNullDataAndRejectsNonemptyNullData) {
  const auto make_null_data_tensor = [](std::vector<std::int64_t> shape,
                                        DLDataType dtype = DLDataType{kDLUInt, 8, 1}) {
    auto context = std::make_shared<holoscan::DLManagedTensorContext>();
    context->dl_shape = std::move(shape);
    context->tensor.dl_tensor.data = nullptr;
    context->tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
    context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(context->dl_shape.size());
    context->tensor.dl_tensor.dtype = dtype;
    context->tensor.dl_tensor.shape = context->dl_shape.data();
    context->tensor.dl_tensor.strides = nullptr;
    context->tensor.dl_tensor.byte_offset = 0U;
    return holoscan::Tensor(std::move(context));
  };

  holoscan::TensorSidecarStore store(/*capacity=*/1U, /*expected_recipient_count=*/1U);
  holomq::DeliveryRecipientId recipient{};
  recipient.bytes.front() = 1U;
  ASSERT_TRUE(store.register_recipient(recipient));

  auto rejected = store.insert(make_null_data_tensor({1}));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument);

  std::array malformed_tensors{
      holoscan::Tensor{},
      make_null_data_tensor({0}, DLDataType{kDLUInt, 0, 1}),
      make_null_data_tensor({-1}),
      make_null_data_tensor({std::numeric_limits<std::int64_t>::max(), 2}),
  };
  for (auto& malformed : malformed_tensors) {
    auto malformed_result = store.insert(std::move(malformed));
    ASSERT_FALSE(malformed_result);
    EXPECT_EQ(malformed_result.error().code, holoscan::ErrorCode::kInvalidArgument);
  }

  auto published = store.insert(make_null_data_tensor({0}));
  ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());
  EXPECT_EQ(published->byte_size, 0U);
  auto received = store.acquire_sample(*published, /*recipient_index=*/0U);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->metadata.byte_size, 0U);
  EXPECT_EQ(received->data.nbytes(), 0);
  EXPECT_EQ(received->data.data(), nullptr);

  auto reused = store.insert(make_null_data_tensor({0}));
  ASSERT_TRUE(reused) << holoscan::runtime_error_message(reused.error());
  store.release_all(*reused);
  auto reused_after_release = store.insert(make_null_data_tensor({0}));
  ASSERT_TRUE(reused_after_release)
      << holoscan::runtime_error_message(reused_after_release.error());
  store.release_all(*reused_after_release);
}

// Verifies sidecar capacity exhaustion reports kResourceExhausted and that released records can be
// reused afterward.
TEST(ChannelTensorTest, SidecarCapacityExhaustionReportsResourceExhausted) {
  // Key setup: isolate the fixed two-record sidecar domain from transport queue scheduling.
  holoscan::TensorSidecarStore store(/*capacity=*/2U, /*expected_recipient_count=*/1U);
  holomq::DeliveryRecipientId recipient{};
  recipient.bytes.front() = 1U;
  ASSERT_TRUE(store.register_recipient(recipient));

  auto destroyed_1 = std::make_shared<std::atomic<int>>(0);
  auto destroyed_2 = std::make_shared<std::atomic<int>>(0);
  auto destroyed_3 = std::make_shared<std::atomic<int>>(0);
  auto destroyed_4 = std::make_shared<std::atomic<int>>(0);
  void* data_1 = nullptr;
  void* data_2 = nullptr;
  void* data_3 = nullptr;
  void* data_4 = nullptr;

  auto published_1 = store.insert(make_counting_tensor(destroyed_1, &data_1, 1U));
  ASSERT_TRUE(published_1) << holoscan::runtime_error_message(published_1.error());
  auto published_2 = store.insert(make_counting_tensor(destroyed_2, &data_2, 2U));
  ASSERT_TRUE(published_2) << holoscan::runtime_error_message(published_2.error());

  // Key assertion: true sidecar exhaustion is reported before a third record can be retained.
  auto exhausted = store.insert(make_counting_tensor(destroyed_3, &data_3, 3U));
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_EQ(destroyed_3->load(std::memory_order_relaxed), 1);

  auto received_1 = store.acquire_sample(*published_1, /*recipient_index=*/0U);
  ASSERT_TRUE(received_1) << holoscan::runtime_error_message(received_1.error());
  holoscan::Tensor tensor_1 = std::move(received_1).value().data;
  EXPECT_EQ(tensor_1.data(), data_1);
  EXPECT_EQ(destroyed_1->load(std::memory_order_relaxed), 0);

  auto recovered = store.insert(make_counting_tensor(destroyed_4, &data_4, 4U));
  ASSERT_TRUE(recovered) << holoscan::runtime_error_message(recovered.error());

  tensor_1 = holoscan::Tensor{};
  EXPECT_EQ(destroyed_1->load(std::memory_order_relaxed), 1);

  auto received_2 = store.acquire_sample(*published_2, /*recipient_index=*/0U);
  ASSERT_TRUE(received_2) << holoscan::runtime_error_message(received_2.error());
  holoscan::Tensor tensor_2 = std::move(received_2).value().data;
  EXPECT_EQ(tensor_2.data(), data_2);

  auto received_4 = store.acquire_sample(*recovered, /*recipient_index=*/0U);
  ASSERT_TRUE(received_4) << holoscan::runtime_error_message(received_4.error());
  holoscan::Tensor tensor_4 = std::move(received_4).value().data;
  EXPECT_EQ(tensor_4.data(), data_4);

  tensor_2 = holoscan::Tensor{};
  tensor_4 = holoscan::Tensor{};
  EXPECT_EQ(destroyed_2->load(std::memory_order_relaxed), 1);
  EXPECT_EQ(destroyed_4->load(std::memory_order_relaxed), 1);
}

// Verifies a capacity-one sidecar store can publish again after the receiver drops each Tensor
// handle, without exposing sidecar token/generation internals through receive_sample().
TEST(ChannelTensorTest, SidecarCapacityOneReusesFreedRecord) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.subscriber_queue_depth = 1U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  for (std::uint8_t fill = 1U; fill <= 4U; ++fill) {
    auto destroyed = std::make_shared<std::atomic<int>>(0);
    void* data = nullptr;

    auto emitted = publisher->emit(make_counting_tensor(destroyed, &data, fill));
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
    EXPECT_EQ(emitted.value(), 1U);

    auto sample = subscriber->receive_sample();
    ASSERT_TRUE(sample) << holoscan::runtime_error_message(sample.error());
    EXPECT_EQ(sample->data.data(), data);
    EXPECT_EQ(static_cast<const std::uint8_t*>(sample->data.data())[0], fill);
    EXPECT_EQ(sample->metadata.sequence, fill);
    EXPECT_EQ(sample->metadata.byte_size, 6U);
    EXPECT_EQ(destroyed->load(std::memory_order_relaxed), 0);

    // Clear the received handle so the single sidecar record can be returned to the free list.
    sample->data = holoscan::Tensor{};
    EXPECT_EQ(destroyed->load(std::memory_order_relaxed), 1);
  }
}

// Verifies end-to-end GPU reuse-safety through the Channel: a device Tensor allocated with the
// compiled fanout release-stream capacity is emitted to two
// subscribers, and the single backing device block is not returned to the pool until BOTH
// recipients drop their handles AND their deallocation-stream events complete. The slow consumer's
// release stream is blocked behind a host function so a single-stream release would wrongly reuse
// the block early.
//
//   publisher --[blocking, fanout=2, device pool, max_fanout_release_streams=2]-->
//       subscriber_a (fast_stream)
//   \-> subscriber_b (slow_stream, host-fn blocked)
//   Single device block must not be reused until BOTH consumer streams complete.
TEST(ChannelTensorTest, BlockingFanoutDeviceTensorWaitsForAllConsumerReleaseStreams) {
  holoscan::test::skip_without_cuda_device();

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 2U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_tensor_recipients(publisher.value(),
                                         {&subscriber_a.value(), &subscriber_b.value()},
                                         2U,
                                         std::chrono::seconds(2)));

  // Single-block device pool so block reuse is directly observable.
  auto pool =
      std::make_shared<holoscan::BlockMemoryPool>(holoscan::MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  holoscan::test::UniqueCudaStream slow_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(slow_stream), cudaSuccess);
  holoscan::test::UniqueCudaStream fast_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(fast_stream), cudaSuccess);

  // Key setup: single-block device pool with max_fanout_release_streams=2 so both consumer
  // streams must complete before the block is returned. Apply the compiled fanout bound
  // directly to storage now that catalog publishers no longer expose caller-selected helpers.
  const std::array<int64_t, 2> shape{2, 3};
  auto maybe_tensor =
      holoscan::make_tensor(shape,
                            DLDataType{kDLFloat, 32, 1},
                            std::nullopt,
                            holoscan::MemoryStorageType::kDevice,
                            pool,
                            holoscan::MemoryBuffer::Options{.max_fanout_release_streams = 2U});
  ASSERT_TRUE(maybe_tensor) << holoscan::runtime_error_message(maybe_tensor.error());
  auto tensor = std::move(maybe_tensor).value();
  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << holoscan::runtime_error_message(writable_data.error());
  void* const expected_data = *writable_data;

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 2U);

  auto received_a = subscriber_a->receive();
  ASSERT_TRUE(received_a) << holoscan::runtime_error_message(received_a.error());
  auto tensor_a = std::move(received_a).value();
  EXPECT_EQ(tensor_a.data(), expected_data);
  auto received_b = subscriber_b->receive();
  ASSERT_TRUE(received_b) << holoscan::runtime_error_message(received_b.error());
  auto tensor_b = std::move(received_b).value();
  EXPECT_EQ(tensor_b.data(), expected_data);

  // The sidecar dropped its copy after both receives; the two recipient handles keep the single
  // block checked out.
  EXPECT_FALSE(pool->allocate(1, holoscan::MemoryStorageType::kDevice));

  // Block the slow consumer's stream behind a host function so its release event cannot complete.
  holoscan::test::BlockingHostState slow_state;
  ASSERT_EQ(
      cudaLaunchHostFunc(slow_stream.get(), holoscan::test::blocking_host_callback, &slow_state),
      cudaSuccess);
  if (!holoscan::test::wait_until_started(slow_state)) {
    holoscan::test::release_blocking_host(slow_state);
    EXPECT_EQ(cudaStreamSynchronize(slow_stream.get()), cudaSuccess);
    FAIL() << "Timed out waiting for CUDA host function to start";
  }

  ASSERT_EQ(cudaMemsetAsync(expected_data, 0x2A, 24, fast_stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(fast_stream.get()), cudaSuccess);

  EXPECT_TRUE(tensor_a.set_deallocation_stream(fast_stream.get()));
  EXPECT_TRUE(tensor_b.set_deallocation_stream(slow_stream.get()));
  tensor_a = holoscan::Tensor();
  tensor_b = holoscan::Tensor();

  // Key assertion: pool block unavailable after both handles dropped while slow_stream blocked.
  // Fanout release must wait for every consumer release stream before the block can be reused.
  EXPECT_FALSE(pool->allocate(1, holoscan::MemoryStorageType::kDevice));

  holoscan::test::release_blocking_host(slow_state);
  ASSERT_EQ(cudaStreamSynchronize(slow_stream.get()), cudaSuccess);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pool->stats().available == 1) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto reused = pool->allocate(1, holoscan::MemoryStorageType::kDevice);
  ASSERT_TRUE(reused) << holoscan::runtime_error_message(reused.error());
  EXPECT_EQ(reused.value(), expected_data);
  EXPECT_TRUE(pool->free(reused.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies a device Tensor received through the Channel survives the sidecar round-trip with its
// stream-aware native memory reference intact: a DLPack export of the received Tensor keeps the
// backing device block checked out until the export deleter runs.
//
//   publisher --[blocking, device pool]--> subscriber
//   DLPack export holds an independent ref; C++ handle drop must not return the block.
TEST(ChannelTensorTest, ReceivedDeviceTensorDLPackExportRetainsBlockUntilDeleter) {
  holoscan::test::skip_without_cuda_device();

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 1U;

  auto publisher = create_tensor_publisher(runtime.value(), channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto pool =
      std::make_shared<holoscan::BlockMemoryPool>(holoscan::MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  const std::array<int64_t, 2> shape{2, 3};
  auto maybe_tensor =
      holoscan::make_tensor(shape,
                            DLDataType{kDLFloat, 32, 1},
                            std::nullopt,
                            holoscan::MemoryStorageType::kDevice,
                            pool,
                            holoscan::MemoryBuffer::Options{.max_fanout_release_streams = 1U});
  ASSERT_TRUE(maybe_tensor) << holoscan::runtime_error_message(maybe_tensor.error());
  auto tensor = std::move(maybe_tensor).value();
  const void* const expected_data = tensor.data();
  ASSERT_NE(expected_data, nullptr);

  auto emitted = publisher->emit(std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto received_tensor = std::move(received).value();
  EXPECT_EQ(received_tensor.data(), expected_data);

  // Export the received Tensor to DLPack; the export holds an independent reference to storage.
  auto exported_result = received_tensor.to_dlpack();
  ASSERT_TRUE(exported_result) << holoscan::runtime_error_message(exported_result.error());
  DLManagedTensor* exported = exported_result.value();
  ASSERT_NE(exported->deleter, nullptr);

  // Key assertion: dropping the C++ handle must not return the block while DLPack is alive.
  received_tensor = holoscan::Tensor();
  EXPECT_FALSE(pool->allocate(1, holoscan::MemoryStorageType::kDevice));

  // Running the export deleter releases the last reference and returns the block to the pool.
  exported->deleter(exported);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pool->stats().available == 1) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto reused = pool->allocate(1, holoscan::MemoryStorageType::kDevice);
  ASSERT_TRUE(reused) << holoscan::runtime_error_message(reused.error());
  EXPECT_EQ(reused.value(), expected_data);
  EXPECT_TRUE(pool->free(reused.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}
