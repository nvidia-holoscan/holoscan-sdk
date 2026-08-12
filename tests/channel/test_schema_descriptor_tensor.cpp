/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/schema/image_metadata.hpp>
#include <holoscan/schema/inline_flatbuffer_payload.hpp>
#include <holoscan/schema/tensor_materialization.hpp>

#include "../../src/channel/holomq_tensor.hpp"
#include "../../src/channel/payload_descriptor_wire.hpp"
#include "../../src/runtime/channel_runtime_binding_access.hpp"
#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/verifier.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "channel_holomq_test_support.hpp"
#include "channel_test_support.hpp"
// Registers the generated CustomControl owning type with schema concepts.
#include "custom_control_schema_traits.hpp"  // NOLINT(misc-include-cleaner)
#include "header_generated.h"
#include "image_generated.h"
#include "tensor_generated.h"
#include "time_generated.h"

#include "schema_test_types.hpp"

namespace {

using namespace holoscan::channel_schema_test;
using holoscan::channel_test::ChildProcessCleanup;
using holoscan::channel_test::HoloMQTestMemoryKind;
using holoscan::channel_test::make_options;
using holoscan::channel_test::make_unique_channel_name;
using holoscan::channel_test::read_exact;
using holoscan::channel_test::try_acquire_buffer;
using holoscan::channel_test::verify_host_pattern;
using holoscan::channel_test::wait_for_subscribers;
using holoscan::channel_test::wait_until;
using holoscan::channel_test::write_exact;
using holoscan::channel_test::write_host_pattern;

#ifndef HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE
#define HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE ""
#endif

// Returns a structurally valid BufferHandlePayloadReference built from the fixed 8-byte
// sentinel "HSDK5EA1". Simulates a cross-process handle shape without a live HoloMQ
// export; used in descriptor field round-trip tests that do not import a real buffer.
[[nodiscard]] holoscan::BufferHandlePayloadReference make_structural_opaque_reference() {
  constexpr std::array<std::byte, 8U> kOpaqueCarrier{std::byte{0x48},
                                                     std::byte{0x53},
                                                     std::byte{0x44},
                                                     std::byte{0x4B},
                                                     std::byte{0x35},
                                                     std::byte{0x45},
                                                     std::byte{0x41},
                                                     std::byte{0x31}};
  return holoscan::channel_internal::OpaquePayloadReferenceAccess::from_bytes(kOpaqueCarrier);
}
template <typename T>
concept HasTransportDescriptor = requires(T value) { value.descriptor; };

static_assert(holoscan::detail::HasDescriptorTensorSchemaNativeTraits<holoscan::schema::ImageT>);
static_assert(!holoscan::detail::HasDescriptorTensorSchemaNativeTraits<
              holoscan::UnsupportedDescriptorTensorSchemaNative>);
static_assert(holoscan::ChannelInlinePayload<DescriptorBackedImagePayload>);
static_assert(DescriptorBackedImagePayload::metadata_capacity == 512U);
static_assert(!HasTransportDescriptor<ResolvedImageMetadata>);
static_assert(std::is_standard_layout_v<ResolvedImageMetadata>);
static_assert(std::is_trivially_copyable_v<ResolvedImageMetadata>);
static_assert(offsetof(ResolvedImageMetadata, endpoint_generation) == 24U);
static_assert(sizeof(ResolvedImageMetadata) == 544U);

constexpr std::size_t kHoloMQSchemaTensorBufferBytes = 64U;
constexpr std::uint32_t kHoloMQSchemaTensorBufferCount = 1U;
constexpr std::uint32_t kHoloMQSchemaTensorPatternSeed = 0x4d510000U;

struct MultiTensorStagingEnvelope {
  holoscan::SampleMetadata metadata{};
  MultiTensorStagingPayload payload{};
};
[[nodiscard]] std::unique_ptr<holoscan::channel_test::HoloMQPoolFixture> make_schema_pool_fixture(
    const HoloMQTestMemoryKind memory_kind) {
  return holoscan::channel_test::make_holomq_pool_fixture(
      {.buffer_bytes = kHoloMQSchemaTensorBufferBytes,
       .buffer_count = kHoloMQSchemaTensorBufferCount,
       .socket_prefix = "hsdk_channel_schema_holomq",
       .memory_kind = memory_kind,
       .fd_export = memory_kind == HoloMQTestMemoryKind::kCudaDevice});
}

[[nodiscard]] std::unique_ptr<holoscan::channel_test::HoloMQPoolFixture>
make_holomq_pinned_host_pool_fixture() {
  return make_schema_pool_fixture(HoloMQTestMemoryKind::kPinnedHost);
}

[[nodiscard]] std::unique_ptr<holoscan::channel_test::HoloMQPoolFixture>
make_holomq_device_pool_fixture() {
  return make_schema_pool_fixture(HoloMQTestMemoryKind::kCudaDevice);
}

[[nodiscard]] holomq::expected<holomq::BufferPool, holomq::Error> make_holomq_pinned_host_pool(
    holomq::Context& context) {
  return holoscan::channel_test::make_holomq_pinned_host_pool(
      context, kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorBufferCount);
}

// TensorMemoryReference that increments an atomic counter when destroyed. Lets tests
// assert that Tensor storage is released exactly once at the right point in the pipeline.
// Representative test: DescriptorBackedTensorSchemaFanoutKeepsTensorAlive
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

// Builds a 2×3 uint8 CPU Tensor whose CountingTensorStorage increments destroyed_counter
// on destruction. Returns the raw data pointer via data_out so tests can verify that
// the correct sidecar record was materialized and the pointer matches the original Tensor.
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
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 8, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0U;
  return holoscan::Tensor(std::move(context));
}

// Like wait_for_schema_payload_recipients but for descriptor-backed payloads using
// emit_payload(). Tracks emit/receive diagnostic counters and calls ADD_FAILURE with
// the full diagnostics on timeout rather than returning false silently.
template <typename Publisher, typename Payload, typename... Subscribers>
[[nodiscard]] bool wait_for_tensor_schema_payload_recipients(
    Publisher& publisher, const std::uint32_t expected_count, const Payload& payload,
    const std::chrono::milliseconds timeout, Subscribers&... subscribers) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint64_t attempts = 0U;
  std::uint64_t emit_successes = 0U;
  std::uint64_t emit_not_ready = 0U;
  std::uint64_t emit_errors = 0U;
  std::uint64_t receive_successes = 0U;
  std::uint64_t receive_not_ready = 0U;
  std::uint64_t receive_errors = 0U;
  std::string last_emit_error;
  std::string last_receive_error;
  while (std::chrono::steady_clock::now() < deadline) {
    ++attempts;
    static_cast<void>(publisher.update_connections());
    auto emitted = publisher.emit_payload(payload);
    if (emitted) {
      ++emit_successes;
    } else if (emitted.error().code == holoscan::ErrorCode::kNotReady) {
      ++emit_not_ready;
      last_emit_error = holoscan::runtime_error_message(emitted.error());
    } else {
      ++emit_errors;
      last_emit_error = holoscan::runtime_error_message(emitted.error());
    }
    auto receive_one = [&](auto& subscriber) {
      auto received = subscriber.receive();
      if (received) {
        ++receive_successes;
      } else if (received.error().code == holoscan::ErrorCode::kNotReady) {
        ++receive_not_ready;
      } else {
        ++receive_errors;
        last_receive_error = holoscan::runtime_error_message(received.error());
      }
    };
    (receive_one(subscribers), ...);
    if (emitted && emitted.value() >= expected_count) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  ADD_FAILURE() << "timed out waiting for descriptor-backed schema payload delivery"
                << " attempts=" << attempts << " emit_successes=" << emit_successes
                << " emit_not_ready=" << emit_not_ready << " emit_errors=" << emit_errors
                << " receive_successes=" << receive_successes
                << " receive_not_ready=" << receive_not_ready
                << " receive_errors=" << receive_errors
                << " has_subscribers=" << publisher.has_subscribers()
                << " last_emit_error=" << last_emit_error
                << " last_receive_error=" << last_receive_error;
  return false;
}

[[nodiscard]] std::vector<std::byte> copy_flatbuffer_builder_bytes(
    const flatbuffers::FlatBufferBuilder& builder) {
  const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  return {data, data + builder.GetSize()};
}

[[nodiscard]] std::vector<std::byte> finish_image_metadata(
    flatbuffers::FlatBufferBuilder& builder,
    const flatbuffers::Offset<holoscan::schema::Tensor> tensor) {
  const auto image = holoscan::schema::CreateImage(builder,
                                                   /*header=*/0,
                                                   /*height=*/2,
                                                   /*width=*/3,
                                                   holoscan::schema::ImageEncoding_MONO8,
                                                   tensor);
  holoscan::schema::FinishImageBuffer(builder, image);
  return copy_flatbuffer_builder_bytes(builder);
}

[[nodiscard]] std::vector<std::byte> make_image_metadata_without_tensor() {
  flatbuffers::FlatBufferBuilder builder;
  return finish_image_metadata(builder, /*tensor=*/0);
}

[[nodiscard]] std::vector<std::byte> make_image_metadata_with_tensor_fields(
    const std::vector<int64_t>& shape, const std::uint32_t ndim,
    const holoscan::schema::DLDataType* dtype, const holoscan::schema::DLDevice* device,
    const std::optional<std::size_t> data_size = std::nullopt) {
  flatbuffers::FlatBufferBuilder builder;
  const auto shape_offset = builder.CreateVector(shape);
  flatbuffers::Offset<flatbuffers::Vector<std::uint8_t>> data_offset{};
  if (data_size.has_value()) {
    const std::vector<std::uint8_t> data(data_size.value(), 0U);
    data_offset = builder.CreateVector(data);
  }
  const auto tensor = holoscan::schema::CreateTensor(builder,
                                                     data_offset,
                                                     shape_offset,
                                                     dtype,
                                                     device,
                                                     ndim,
                                                     /*strides=*/0);
  return finish_image_metadata(builder, tensor);
}

void expect_generic_tensor_metadata_only(const holoscan::Tensor& tensor) {
  flatbuffers::FlatBufferBuilder builder;
  auto tensor_metadata = holoscan::schema::create_tensor_metadata(builder, tensor);
  ASSERT_TRUE(tensor_metadata) << holoscan::runtime_error_message(tensor_metadata.error());
  holoscan::schema::FinishTensorBuffer(builder, *tensor_metadata);

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(holoscan::schema::VerifyTensorBuffer(verifier));
  const auto* encoded = holoscan::schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(encoded, nullptr);
  EXPECT_EQ(encoded->data(), nullptr);
  ASSERT_NE(encoded->shape(), nullptr);
  EXPECT_EQ(encoded->shape()->size(), 2U);
  EXPECT_EQ(encoded->shape()->Get(0), 3);
  EXPECT_EQ(encoded->shape()->Get(1), 2);
  ASSERT_NE(encoded->strides(), nullptr);
  EXPECT_EQ(encoded->strides()->size(), 2U);
  EXPECT_EQ(encoded->strides()->Get(0), 4);
  EXPECT_EQ(encoded->strides()->Get(1), 1);
}

[[nodiscard]] holoscan::Tensor make_scalar_family_cpu_tensor(const DLDataType dtype) {
  static std::array<std::max_align_t, 2U> data{};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {1};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;
  return holoscan::Tensor(std::move(context));
}

void expect_generic_tensor_dtype_round_trip(const DLDataType dtype) {
  const holoscan::Tensor tensor = make_scalar_family_cpu_tensor(dtype);
  flatbuffers::FlatBufferBuilder builder;
  auto tensor_metadata = holoscan::schema::create_tensor_metadata(builder, tensor);
  ASSERT_TRUE(tensor_metadata) << holoscan::runtime_error_message(tensor_metadata.error());
  holoscan::schema::FinishTensorBuffer(builder, *tensor_metadata);

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(holoscan::schema::VerifyTensorBuffer(verifier));
  const auto* encoded = holoscan::schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(encoded, nullptr);
  ASSERT_NE(encoded->dtype(), nullptr);
  EXPECT_EQ(static_cast<std::uint8_t>(encoded->dtype()->code()), dtype.code);
  EXPECT_EQ(encoded->dtype()->bits(), dtype.bits);
  EXPECT_EQ(encoded->dtype()->lanes(), dtype.lanes);

  auto decoded = holoscan::schema::tensor_view_metadata(*encoded);
  ASSERT_TRUE(decoded) << holoscan::runtime_error_message(decoded.error());
  EXPECT_EQ(decoded->dtype.code, dtype.code);
  EXPECT_EQ(decoded->dtype.bits, dtype.bits);
  EXPECT_EQ(decoded->dtype.lanes, dtype.lanes);
}

// Returns a 3×2 float32 CPU Tensor with explicit strides {4, 1} to verify that
// non-trivial stride encoding survives the Image metadata FlatBuffer round-trip.
[[nodiscard]] holoscan::Tensor make_strided_cpu_tensor_3x2() {
  static std::vector<float> data{1.0F, 2.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F, 5.0F, 6.0F};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {3, 2};
  context->dl_strides = {4, 1};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0;
  return holoscan::Tensor(std::move(context));
}

[[nodiscard]] holoscan::Tensor make_cpu_tensor_with_layout(std::vector<std::int64_t> shape,
                                                           std::vector<std::int64_t> strides) {
  static std::array<std::max_align_t, 2U> data{};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = std::move(shape);
  context->dl_strides = std::move(strides);
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(context->dl_shape.size());
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0U;
  return holoscan::Tensor(std::move(context));
}

// Builds a ChannelSampleDescriptor with kLocalBufferDescriptor kind, sentinel sidecar
// token/generation values, and the schema::Tensor identity. Used in structural descriptor
// round-trip tests that do not resolve a live sidecar record from the store.
[[nodiscard]] holoscan::ChannelSampleDescriptor make_tensor_data_descriptor(
    const holoscan::Tensor& tensor, const std::uint64_t sequence = 1U) {
  return holoscan::ChannelSampleDescriptor{
      .magic = holoscan::kChannelSampleDescriptorMagic,
      .abi_version = holoscan::kChannelSampleDescriptorAbiVersion,
      .payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor,
      .flags = 0U,
      .expected_recipient_count = 1U,
      .process_id = static_cast<std::uint64_t>(::getpid()),
      .sidecar_token = 0x7000U + sequence,
      .generation = 0x9000U + sequence,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = sequence,
      .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
  };
}

// Serializes a complete schema::Image FlatBuffer (header, dimensions, encoding, Tensor
// metadata) from a live Tensor without embedding pixel bytes. Used as the metadata_view
// argument to tensor_schema_publisher::emit() in materialization tests.
[[nodiscard]] std::vector<std::byte> make_image_metadata_bytes(const holoscan::Tensor& tensor,
                                                               std::string_view frame_id,
                                                               const std::int64_t time_ns) {
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 2,
      .width = 3,
      .encoding = holoscan::schema::ImageEncoding_MONO8,
      .time_ns = time_ns,
      .acqtime_ns = time_ns + 17,
      .frame_id = std::string(frame_id)};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  if (!metadata) {
    return {};
  }
  return metadata.value();
}

// Like wait_for_schema_payload_recipients but emits a fresh CountingTensor each attempt;
// primes the sidecar store and warms up subscriber discovery before the single asserting
// emit in tensor-schema materialization tests.
template <typename Publisher, typename... Subscribers>
[[nodiscard]] bool wait_for_tensor_schema_recipients(Publisher& publisher,
                                                     std::span<const std::byte> metadata,
                                                     const std::uint32_t expected_count,
                                                     const std::chrono::milliseconds timeout,
                                                     Subscribers&... subscribers) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto destroyed = std::make_shared<std::atomic<int>>(0);
    void* data = nullptr;
    auto emitted = publisher.emit(metadata, make_counting_tensor(destroyed, &data, 0U));
    ((void)subscribers.receive(), ...);
    if (emitted && emitted.value() >= expected_count) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

// Allocates a device Tensor sized to the publisher's current tensor_memory_options; used
// for device fanout and DLPack export retention tests that need a real GPU allocation.
template <typename Publisher>
[[nodiscard]] holoscan::expected<holoscan::Tensor, holoscan::Error> make_device_tensor_for_schema(
    const Publisher& publisher, const std::shared_ptr<holoscan::Allocator>& allocator) {
  auto memory_options = publisher.tensor_memory_options();
  if (!memory_options) {
    return holoscan::make_unexpected(std::move(memory_options).error());
  }
  const std::array<int64_t, 2> shape{2, 3};
  return holoscan::make_tensor(shape,
                               DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1},
                               std::nullopt,
                               holoscan::MemoryStorageType::kDevice,
                               allocator,
                               memory_options.value());
}

}  // namespace

TEST(ChannelSchemaPayloadTest, PayloadDescriptorWireEncodingZerosNonSemanticBytes) {
  const holoscan::PayloadDescriptor descriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor,
      .flags = 0x12345678U,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = 17U,
      .byte_size = 4096U,
      .local =
          holoscan::LocalPayloadReference{
              .process_id = 91U,
              .token = 92U,
              .generation = 93U,
              .expected_recipient_count = 2U,
          },
  };
  // Key setup: encode the same local descriptor over two distinct nonzero destination patterns.
  std::array<std::byte, sizeof(holoscan::PayloadDescriptor)> first{};
  std::array<std::byte, sizeof(holoscan::PayloadDescriptor)> second{};
  first.fill(std::byte{0xA5});
  second.fill(std::byte{0x5A});
  ASSERT_TRUE(holoscan::channel_internal::encode_payload_descriptor_wire(descriptor, first));
  ASSERT_TRUE(holoscan::channel_internal::encode_payload_descriptor_wire(descriptor, second));

  // Key assertion: initial storage and inactive-union bytes cannot affect the canonical frame.
  EXPECT_EQ(first, second);
  auto decoded = holoscan::channel_internal::decode_payload_descriptor_wire(first);
  ASSERT_TRUE(decoded) << holoscan::runtime_error_message(decoded.error());
  EXPECT_EQ(decoded->payload_kind, descriptor.payload_kind);
  EXPECT_EQ(decoded->flags, descriptor.flags);
  EXPECT_EQ(decoded->schema_id, descriptor.schema_id);
  EXPECT_EQ(decoded->sequence, descriptor.sequence);
  EXPECT_EQ(decoded->byte_size, descriptor.byte_size);
  EXPECT_EQ(decoded->local.process_id, descriptor.local.process_id);
  EXPECT_EQ(decoded->local.token, descriptor.local.token);
  EXPECT_EQ(decoded->local.generation, descriptor.local.generation);
  EXPECT_EQ(decoded->local.expected_recipient_count, descriptor.local.expected_recipient_count);

  std::array<bool, sizeof(holoscan::PayloadDescriptor)> semantic{};
  const auto mark = [&semantic](const std::size_t offset, const std::size_t size) {
    for (std::size_t index = offset; index < offset + size; ++index) {
      semantic[index] = true;
    }
  };
  mark(offsetof(holoscan::PayloadDescriptor, payload_kind),
       sizeof(std::underlying_type_t<holoscan::ChannelPayloadKind>));
  mark(offsetof(holoscan::PayloadDescriptor, flags), sizeof(descriptor.flags));
  mark(offsetof(holoscan::PayloadDescriptor, schema_id), sizeof(descriptor.schema_id));
  mark(offsetof(holoscan::PayloadDescriptor, sequence), sizeof(descriptor.sequence));
  mark(offsetof(holoscan::PayloadDescriptor, byte_size), sizeof(descriptor.byte_size));
  constexpr std::size_t kLocalOffset = offsetof(holoscan::PayloadDescriptor, local);
  mark(kLocalOffset + offsetof(holoscan::LocalPayloadReference, process_id),
       sizeof(descriptor.local.process_id));
  mark(kLocalOffset + offsetof(holoscan::LocalPayloadReference, token),
       sizeof(descriptor.local.token));
  mark(kLocalOffset + offsetof(holoscan::LocalPayloadReference, generation),
       sizeof(descriptor.local.generation));
  mark(kLocalOffset + offsetof(holoscan::LocalPayloadReference, expected_recipient_count),
       sizeof(descriptor.local.expected_recipient_count));
  for (std::size_t index = 0U; index < first.size(); ++index) {
    if (!semantic[index]) {
      EXPECT_EQ(first[index], std::byte{}) << "non-canonical byte at offset " << index;
    }
  }
}
// Verifies descriptor-backed Image metadata and its structurally valid opaque external-buffer
// reference can cross a real process boundary before Tensor materialization is involved.
TEST(ChannelSchemaPayloadTest, BlockingPublishesDescriptorBackedImageFlatBufferToChildProcess) {
  const char* const child_exe = HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE not configured";

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();

  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const auto metadata = make_image_metadata_bytes(tensor, "ipc_descriptor_image", 424000);
  ASSERT_FALSE(metadata.empty());
  const holoscan::PayloadDescriptor payload_descriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = 101U,
      .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
      .buffer = make_structural_opaque_reference()};
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata.data(), metadata.size()), payload_descriptor);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  const DescriptorBackedImagePayload payload_value = std::move(payload).value();

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
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_NAME", channel_name.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_MODE", "descriptor", 1));
    ::execl(child_exe, "channel_schema_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
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

  std::string last_emit_error;
  bool child_exited_early = false;
  int child_status = 0;
  std::uint8_t early_child_code = 0xffU;
  const bool delivered = wait_until(
      [&publisher,
       &payload_value,
       &last_emit_error,
       child,
       &child_exited_early,
       &child_status,
       &early_child_code,
       &child_cleanup] {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit_payload(payload_value);
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
  ASSERT_TRUE(delivered) << "failed to publish descriptor-backed Image schema payload to child: "
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

// Verifies explicit Channel endpoints carry Image metadata without materializing Tensor bytes.
TEST(ChannelSchemaPayloadTest, PublisherSubscriberCarryImageMetadataFlatBuffer) {
  //   publisher --[kBlocking, depth=8]--> subscriber (Image metadata only; no pixel bytes)
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 3,
      .width = 2,
      .encoding = holoscan::schema::ImageEncoding_MONO32F,
      .time_ns = 987654321,
      .acqtime_ns = 987654399,
      .frame_id = "camera_frame"};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  ASSERT_TRUE(metadata) << holoscan::runtime_error_message(metadata.error());
  auto payload =
      ImagePayload::copy_from(std::span<const std::byte>(metadata->data(), metadata->size()));
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<ImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<ImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(3)));

  auto emitted = publisher->emit(payload.value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->schema_id, holoscan::schema_identity<holoscan::schema::Image>().id);
  auto valid = received->validate();
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());

  auto parsed = holoscan::schema::parse_image_metadata(received->bytes());
  ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());
  const holoscan::schema::Image* image = parsed->image;
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->height(), 3);
  EXPECT_EQ(image->width(), 2);
  EXPECT_EQ(image->encoding(), holoscan::schema::ImageEncoding_MONO32F);
  ASSERT_NE(image->header(), nullptr);
  ASSERT_NE(image->header()->stamp(), nullptr);
  EXPECT_EQ(image->header()->stamp()->time_ns(), 987654321);
  ASSERT_NE(image->header()->frame_id(), nullptr);
  EXPECT_EQ(image->header()->frame_id()->string_view(), std::string_view("camera_frame"));
  ASSERT_NE(image->data(), nullptr);
  if (image->data()->data() != nullptr) {
    EXPECT_EQ(image->data()->data()->size(), 0U);
  }
}

// Verifies Image metadata retains the embedded Tensor shape/type/device information needed to
// pair a schema message with an external Tensor payload descriptor in a later materialization step.
TEST(ChannelSchemaPayloadTest, ImageMetadataExposesTensorDescriptorMetadata) {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const auto metadata = make_image_metadata_bytes(tensor, "metadata_camera", 111000);
  ASSERT_FALSE(metadata.empty());

  auto parsed = holoscan::schema::parse_image_metadata(
      std::span<const std::byte>(metadata.data(), metadata.size()));
  ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());

  auto tensor_metadata = holoscan::schema::image_tensor_metadata(parsed.value());
  ASSERT_TRUE(tensor_metadata) << holoscan::runtime_error_message(tensor_metadata.error());
  EXPECT_EQ(tensor_metadata->rank, 2U);
  EXPECT_EQ(tensor_metadata->dtype.code, kDLFloat);
  EXPECT_EQ(tensor_metadata->dtype.bits, 32U);
  EXPECT_EQ(tensor_metadata->dtype.lanes, 1U);
  EXPECT_EQ(tensor_metadata->device.device_type, kDLCPU);
  EXPECT_EQ(tensor_metadata->device.device_id, 0);
  EXPECT_TRUE(tensor_metadata->has_explicit_strides);
  EXPECT_EQ(tensor_metadata->shape[0], 3);
  EXPECT_EQ(tensor_metadata->shape[1], 2);
  EXPECT_EQ(tensor_metadata->dlpack_strides[0], 4);
  EXPECT_EQ(tensor_metadata->dlpack_strides[1], 1);
  EXPECT_EQ(tensor_metadata->byte_size, static_cast<std::uint64_t>(tensor.nbytes()));
}

// The public Tensor-member encoder is metadata-only regardless of ambient object-API scopes. This
// is the schema-generic primitive used by descriptor-backed schema callbacks.
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(ChannelSchemaPayloadTest, TensorMetadataEncoderIgnoresAmbientMaterializationScopes) {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  {
    const holoscan::schema::WithTensorMaterialization scope;
    expect_generic_tensor_metadata_only(tensor);
  }
  {
    const holoscan::schema::WithTensorGather scope(1024U);
    expect_generic_tensor_metadata_only(tensor);
  }
}

// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(ChannelSchemaPayloadTest, TensorMetadataEncoderAcceptsCoreDLPackScalarFamilies) {
  // Key setup: encode every scalar-family dtype admitted by the Core Tensor layout contract.
  constexpr std::array<DLDataType, 7U> kDTypes{
      DLDataType{kDLInt, 32U, 1U},
      DLDataType{kDLUInt, 8U, 1U},
      DLDataType{kDLFloat, 32U, 1U},
      DLDataType{kDLOpaqueHandle, 64U, 1U},
      DLDataType{kDLBfloat, 16U, 1U},
      DLDataType{kDLComplex, 64U, 1U},
      DLDataType{kDLBool, 8U, 1U},
  };

  // Key assertion: schema encode and decode preserve every valid DLPack dtype exactly.
  for (const auto dtype : kDTypes) {
    expect_generic_tensor_dtype_round_trip(dtype);
  }
}

// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(ChannelSchemaPayloadTest, TensorMetadataHelpersRejectInvalidPhysicalLayouts) {
  // Key setup: exercise overlapping, zero-stride, and arithmetic-overflow layouts through both the
  // live-Tensor encoder and verifier-valid reflected Tensor metadata.
  const std::array<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>, 3U>
      invalid_layouts{
          std::pair{std::vector<std::int64_t>{2, 2}, std::vector<std::int64_t>{1, 1}},
          std::pair{std::vector<std::int64_t>{2, 2}, std::vector<std::int64_t>{0, 1}},
          std::pair{std::vector<std::int64_t>{3, 2},
                    std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max(), 1}},
      };
  const holoscan::schema::DLDataType dtype{holoscan::schema::DLDataTypeCode_kDLUInt,
                                           /*bits=*/8U,
                                           /*lanes=*/1U};
  const holoscan::schema::DLDevice device{holoscan::schema::DLDeviceType_kDLCPU,
                                          /*device_id=*/0};

  // Key assertion: neither exported helper accepts a layout that Core Tensor access rejects.
  for (const auto& [shape, strides] : invalid_layouts) {
    const holoscan::Tensor source = make_cpu_tensor_with_layout(shape, strides);
    flatbuffers::FlatBufferBuilder encode_builder;
    auto encoded = holoscan::schema::create_tensor_metadata(encode_builder, source);
    ASSERT_FALSE(encoded);
    EXPECT_EQ(encoded.error().code, holoscan::ErrorCode::kInvalidArgument);

    flatbuffers::FlatBufferBuilder decode_builder;
    const auto shape_offset = decode_builder.CreateVector(shape);
    const auto strides_offset = decode_builder.CreateVector(strides);
    const auto tensor = holoscan::schema::CreateTensor(decode_builder,
                                                       /*data=*/0,
                                                       shape_offset,
                                                       &dtype,
                                                       &device,
                                                       static_cast<std::uint32_t>(shape.size()),
                                                       strides_offset);
    holoscan::schema::FinishTensorBuffer(decode_builder, tensor);
    const auto* root = holoscan::schema::GetTensor(decode_builder.GetBufferPointer());
    ASSERT_NE(root, nullptr);
    auto decoded = holoscan::schema::tensor_view_metadata(*root);
    ASSERT_FALSE(decoded);
    EXPECT_EQ(decoded.error().code, holoscan::ErrorCode::kCodecError);
  }
}

// Image metadata must admit the same DLPack scalar families as the shared Tensor metadata codec.
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(ChannelSchemaPayloadTest, ImageMetadataAcceptsCoreDLPackScalarFamilies) {
  // Key setup: encode Image metadata for every scalar-family dtype admitted by Core Tensor.
  constexpr std::array<DLDataType, 7U> kDTypes{
      DLDataType{kDLInt, 32U, 1U},
      DLDataType{kDLUInt, 8U, 1U},
      DLDataType{kDLFloat, 32U, 1U},
      DLDataType{kDLOpaqueHandle, 64U, 1U},
      DLDataType{kDLBfloat, 16U, 1U},
      DLDataType{kDLComplex, 64U, 1U},
      DLDataType{kDLBool, 8U, 1U},
  };
  const holoscan::schema::ImageMetadataOptions options{
      .height = 1,
      .width = 1,
      .encoding = holoscan::schema::ImageEncoding_CUSTOM,
  };

  // Key assertion: both the Image API and common descriptor adapter preserve every dtype.
  for (const auto dtype : kDTypes) {
    const holoscan::Tensor tensor = make_scalar_family_cpu_tensor(dtype);
    auto metadata = holoscan::schema::build_image_metadata(tensor, options);
    ASSERT_TRUE(metadata) << holoscan::runtime_error_message(metadata.error());
    auto parsed = holoscan::schema::parse_image_metadata(*metadata);
    ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());

    auto image_metadata = holoscan::schema::image_tensor_metadata(*parsed);
    ASSERT_TRUE(image_metadata) << holoscan::runtime_error_message(image_metadata.error());
    EXPECT_EQ(image_metadata->dtype.code, dtype.code);
    EXPECT_EQ(image_metadata->dtype.bits, dtype.bits);
    EXPECT_EQ(image_metadata->dtype.lanes, dtype.lanes);

    auto descriptor_metadata = holoscan::schema::DescriptorFlatBufferTensorViewTraits<
        holoscan::schema::Image>::tensor_view_metadata(*parsed->image);
    ASSERT_TRUE(descriptor_metadata)
        << holoscan::runtime_error_message(descriptor_metadata.error());
    EXPECT_EQ(descriptor_metadata->dtype.code, dtype.code);
    EXPECT_EQ(descriptor_metadata->dtype.bits, dtype.bits);
    EXPECT_EQ(descriptor_metadata->dtype.lanes, dtype.lanes);
  }
}

// Descriptor receive must reject a verifier-valid schema carrying embedded Tensor bytes before a
// materializing subscriber can expose a typed sample or attempt descriptor resolution.
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(ChannelSchemaPayloadTest,
     DescriptorBackedTensorSchemaRejectsEmbeddedBytesBeforeTypedExposure) {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const holoscan::schema::DLDataType dtype{holoscan::schema::DLDataTypeCode_kDLFloat,
                                           /*bits=*/32,
                                           /*lanes=*/1};
  const holoscan::schema::DLDevice device{holoscan::schema::DLDeviceType_kDLCPU,
                                          /*device_id=*/0};
  const auto metadata = make_image_metadata_with_tensor_fields(
      {3, 2}, /*ndim=*/2U, &dtype, &device, /*data_size=*/1U);
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata.data(), metadata.size()),
      make_tensor_data_descriptor(tensor, 29U));
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(3)));

  auto emitted = publisher->emit(*payload);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto received = subscriber->receive();
  ASSERT_FALSE(received);
  EXPECT_EQ(received.error().code, holoscan::ErrorCode::kCodecError);
}

// Verifies verifier-valid shape metadata cannot wrap either its element count or its final
// item-size product into a smaller logical payload size.
TEST(ChannelSchemaPayloadTest, ImageMetadataRejectsInferredByteSizeOverflow) {
  const holoscan::schema::DLDataType uint8_dtype{holoscan::schema::DLDataTypeCode_kDLUInt,
                                                 /*bits=*/8,
                                                 /*lanes=*/1};
  const holoscan::schema::DLDataType uint32_dtype{holoscan::schema::DLDataTypeCode_kDLUInt,
                                                  /*bits=*/32,
                                                  /*lanes=*/1};
  const holoscan::schema::DLDevice cpu_device{holoscan::schema::DLDeviceType_kDLCPU,
                                              /*device_id=*/0};
  const std::array overflow_cases{
      std::pair{std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 3}, &uint8_dtype},
      std::pair{std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 2}, &uint32_dtype}};

  for (const auto& [shape, dtype] : overflow_cases) {
    const auto metadata = make_image_metadata_with_tensor_fields(
        shape, static_cast<std::uint32_t>(shape.size()), dtype, &cpu_device);
    auto parsed = holoscan::schema::parse_image_metadata(
        std::span<const std::byte>(metadata.data(), metadata.size()));
    ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());

    auto tensor_metadata = holoscan::schema::image_tensor_metadata(parsed.value());
    ASSERT_FALSE(tensor_metadata);
    EXPECT_EQ(tensor_metadata.error().code, holoscan::ErrorCode::kCodecError);
  }
}

// Verifies the generic descriptor-backed payload wrapper can pair Image metadata with Tensor
// descriptor metadata without materializing tensor bytes into the FlatBuffer. This inline-wrapper
// test intentionally uses a fabricated descriptor; materializing Tensor receive uses the dedicated
// tensor-schema endpoint tests below.
TEST(ChannelSchemaPayloadTest,
     DescriptorBackedImageCarriesTensorDescriptorWithoutMaterializingData) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 3,
      .width = 2,
      .encoding = holoscan::schema::ImageEncoding_MONO32F,
      .time_ns = 555000,
      .acqtime_ns = 555123,
      .frame_id = "descriptor_camera"};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  ASSERT_TRUE(metadata) << holoscan::runtime_error_message(metadata.error());

  const auto tensor_descriptor = make_tensor_data_descriptor(tensor, 17U);
  const holoscan::PayloadDescriptor tensor_payload_descriptor =
      holoscan::payload_descriptor_from_descriptor(tensor_descriptor);
  // This payload reference has same-process shape but no live sidecar record behind it yet.
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata->data(), metadata->size()), tensor_payload_descriptor);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(3)));

  auto emitted = publisher->emit(payload.value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto valid = received->validate();
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
  EXPECT_EQ(received->schema_id, holoscan::schema_identity<holoscan::schema::Image>().id);
  ASSERT_NE(received->descriptor_at(0U), nullptr);
  EXPECT_EQ(received->descriptor_at(0U)->schema_id,
            holoscan::schema_identity<holoscan::schema::Tensor>().id);
  EXPECT_EQ(received->descriptor_at(0U)->payload_kind,
            holoscan::ChannelPayloadKind::kLocalBufferDescriptor);
  EXPECT_EQ(received->descriptor_at(0U)->local.token, tensor_descriptor.sidecar_token);
  EXPECT_EQ(received->descriptor_at(0U)->local.generation, tensor_descriptor.generation);
  EXPECT_EQ(received->descriptor_at(0U)->byte_size, static_cast<std::uint64_t>(tensor.nbytes()));

  auto parsed = holoscan::schema::parse_image_metadata(received->bytes());
  ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());
  const holoscan::schema::Image* image = parsed->image;
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->height(), 3);
  EXPECT_EQ(image->width(), 2);
  ASSERT_NE(image->header(), nullptr);
  ASSERT_NE(image->header()->frame_id(), nullptr);
  EXPECT_EQ(image->header()->frame_id()->string_view(), std::string_view("descriptor_camera"));
  ASSERT_NE(image->data(), nullptr);
  if (image->data()->data() != nullptr) {
    EXPECT_EQ(image->data()->data()->size(), 0U);
  }
}

// Verifies descriptor-backed schema wrappers preserve the production PayloadDescriptor shape,
// including the cross-process opaque-reference arm. This generic inline wrapper only copies the
// descriptor; the private cross-process materializer validates and imports its transport carrier.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImagePreservesBufferHandlePayloadDescriptor) {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const auto metadata = make_image_metadata_bytes(tensor, "buffer_handle_camera", 667000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());

  const holoscan::PayloadDescriptor payload_descriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = 73U,
      .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
      .buffer = make_structural_opaque_reference()};

  auto payload = DescriptorBackedImagePayload::copy_from(metadata_view, payload_descriptor);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  auto valid = payload->validate();
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
  ASSERT_NE(payload->descriptor_at(0U), nullptr);
  EXPECT_EQ(payload->descriptor_at(0U)->payload_kind, holoscan::ChannelPayloadKind::kBufferHandle);
  EXPECT_EQ(payload->descriptor_at(0U)->schema_id,
            holoscan::schema_identity<holoscan::schema::Tensor>().id);
  EXPECT_EQ(payload->descriptor_at(0U)->sequence, 73U);
  EXPECT_EQ(payload->descriptor_at(0U)->byte_size, static_cast<std::uint64_t>(tensor.nbytes()));
}

// Verifies descriptor-backed schema payload construction rejects descriptors that cannot safely
// identify external tensor data.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImageRejectsInvalidTensorDescriptor) {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 3,
      .width = 2,
      .encoding = holoscan::schema::ImageEncoding_MONO32F,
      .time_ns = 777000,
      .acqtime_ns = 777123,
      .frame_id = "descriptor_camera"};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  ASSERT_TRUE(metadata) << holoscan::runtime_error_message(metadata.error());
  const auto metadata_bytes = std::span<const std::byte>(metadata->data(), metadata->size());

  auto empty_token = make_tensor_data_descriptor(tensor, 21U);
  empty_token.sidecar_token = 0U;
  auto empty_token_payload = DescriptorBackedImagePayload::copy_from(metadata_bytes, empty_token);
  ASSERT_FALSE(empty_token_payload);
  EXPECT_EQ(empty_token_payload.error().code, holoscan::ErrorCode::kCodecError);

  auto wrong_kind = make_tensor_data_descriptor(tensor, 22U);
  wrong_kind.payload_kind = holoscan::ChannelPayloadKind::kInlinePayload;
  auto wrong_kind_payload = DescriptorBackedImagePayload::copy_from(metadata_bytes, wrong_kind);
  ASSERT_FALSE(wrong_kind_payload);
  EXPECT_EQ(wrong_kind_payload.error().code, holoscan::ErrorCode::kCodecError);

  auto unsupported_abi = make_tensor_data_descriptor(tensor, 23U);
  unsupported_abi.abi_version = holoscan::kChannelSampleDescriptorAbiVersion + 1U;
  auto unsupported_abi_payload =
      DescriptorBackedImagePayload::copy_from(metadata_bytes, unsupported_abi);
  ASSERT_FALSE(unsupported_abi_payload);
  EXPECT_EQ(unsupported_abi_payload.error().code, holoscan::ErrorCode::kCodecError);

  auto wrong_process = make_tensor_data_descriptor(tensor, 24U);
  wrong_process.process_id = static_cast<std::uint64_t>(::getpid()) + 1U;
  auto wrong_process_payload =
      DescriptorBackedImagePayload::copy_from(metadata_bytes, wrong_process);
  // The installed schema wrapper performs only portable structural checks. The private sidecar
  // materializer owns the authoritative operating-system process-identity comparison (covered by
  // TensorSidecarStoreTest), so constructing transport metadata here remains valid.
  ASSERT_TRUE(wrong_process_payload)
      << holoscan::runtime_error_message(wrong_process_payload.error());

  auto buffer_handle = make_tensor_data_descriptor(tensor, 25U);
  buffer_handle.payload_kind = holoscan::ChannelPayloadKind::kBufferHandle;
  auto buffer_handle_payload =
      DescriptorBackedImagePayload::copy_from(metadata_bytes, buffer_handle);
  ASSERT_FALSE(buffer_handle_payload);
  EXPECT_EQ(buffer_handle_payload.error().code, holoscan::ErrorCode::kCodecError);
}

// Verifies the production-direction descriptor-backed schema endpoint mints a real sidecar
// descriptor on emit and resolves that descriptor into a Tensor handle on receive. The Image
// FlatBuffer remains metadata-only; pixel storage is not materialized into the schema bytes.
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaMaterializesTensorFromSidecar) {
  //   tensor_schema_publisher --[kBlocking]--> tensor_schema_subscriber
  //   (Image metadata carried inline; Tensor bytes resolved via sidecar on receive)
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 11U);
  const auto metadata = make_image_metadata_bytes(tensor, "materialized_camera", 123000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(
      publisher.value(), metadata_view, 1U, std::chrono::seconds(3), subscriber.value()));

  auto emitted = publisher->emit(metadata_view, std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->tensor.data(), expected_data);
  EXPECT_EQ(received->metadata.schema_id, DescriptorBackedImagePayload::expected_schema_id);
  EXPECT_EQ(received->metadata.schema_version,
            DescriptorBackedImagePayload::expected_schema_version);
  EXPECT_NE(received->payload.local.token, 0U);
  EXPECT_EQ(received->payload.payload_kind, holoscan::ChannelPayloadKind::kLocalBufferDescriptor);
  EXPECT_EQ(received->payload.schema_id, DescriptorBackedImagePayload::expected_schema_id);
  EXPECT_EQ(received->payload.byte_size, 6U);
  EXPECT_TRUE(received->payload.is_local_buffer());

  const holoscan::schema::Image* image = received->root();
  ASSERT_NE(image, nullptr);
  ASSERT_NE(image->header(), nullptr);
  ASSERT_NE(image->header()->frame_id(), nullptr);
  EXPECT_EQ(image->header()->frame_id()->string_view(), std::string_view("materialized_camera"));
  ASSERT_NE(image->data(), nullptr);
  if (image->data()->data() != nullptr) {
    EXPECT_EQ(image->data()->data()->size(), 0U);
  }
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  received->tensor = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies malformed metadata is rejected before Tensor sidecar reservation, so a failed emit
// neither retains the Tensor nor consumes the only sidecar record needed by the next valid emit.
TEST(ChannelSchemaPayloadTest,
     DescriptorBackedTensorSchemaRejectsInvalidMetadataBeforeReservation) {
  //   tensor_schema_publisher --[kBlocking, depth=1]--> tensor_schema_subscriber
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 1U;
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  auto invalid_destroyed = std::make_shared<std::atomic<int>>(0);
  void* invalid_data = nullptr;
  const std::array<std::byte, 8U> invalid_metadata{};
  auto invalid_emit = publisher->emit(invalid_metadata,
                                      make_counting_tensor(invalid_destroyed, &invalid_data, 71U));
  ASSERT_FALSE(invalid_emit);
  EXPECT_EQ(invalid_emit.error().code, holoscan::ErrorCode::kCodecError);
  EXPECT_EQ(invalid_destroyed->load(std::memory_order_relaxed), 1);

  auto valid_destroyed = std::make_shared<std::atomic<int>>(0);
  void* valid_data = nullptr;
  auto valid_tensor = make_counting_tensor(valid_destroyed, &valid_data, 72U);
  const auto valid_metadata =
      make_image_metadata_bytes(valid_tensor, "valid_after_rejection", 124000);
  ASSERT_FALSE(valid_metadata.empty());
  const auto valid_metadata_view =
      std::span<const std::byte>(valid_metadata.data(), valid_metadata.size());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(
      publisher.value(), valid_metadata_view, 1U, std::chrono::seconds(3), subscriber.value()));

  auto valid_emit = publisher->emit(valid_metadata_view, std::move(valid_tensor));
  ASSERT_TRUE(valid_emit) << holoscan::runtime_error_message(valid_emit.error());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->tensor.data(), valid_data);
}

// A release-callback allocation failure occurs before HoloMQ submits the sample. The adapter must
// release the unpublished sidecar claim so the finite record is immediately reusable by a later
// logical sample.
// REQ: CHAN-067
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaRollsBackReleaseRegistrationFailure) {
  //   tensor_schema_publisher --[kBlocking, depth=1]--> tensor_schema_subscriber
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 1U;
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto failed_counter = std::make_shared<std::atomic<int>>(0);
  void* failed_data = nullptr;
  auto failed_tensor = make_counting_tensor(failed_counter, &failed_data, 81U);
  const auto failed_metadata =
      make_image_metadata_bytes(failed_tensor, "callback_registration_failure", 125000);
  ASSERT_FALSE(failed_metadata.empty());
  holoscan::expected<std::uint32_t, holoscan::Error> failed;
  {
    // Key setup: ScopedAllocationFailure injects an allocation error so the adapter must
    //   roll back the sidecar claim before returning kResourceExhausted.
    const holoscan::test::ScopedAllocationFailure fail_next_allocation{0U};
    failed = publisher->emit(failed_metadata, std::move(failed_tensor));
  }
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_EQ(failed_counter->load(std::memory_order_relaxed), 1);

  auto retry_counter = std::make_shared<std::atomic<int>>(0);
  void* retry_data = nullptr;
  auto retry_tensor = make_counting_tensor(retry_counter, &retry_data, 82U);
  const auto retry_metadata =
      make_image_metadata_bytes(retry_tensor, "after_callback_registration_failure", 126000);
  ASSERT_FALSE(retry_metadata.empty());
  auto retried = publisher->emit(retry_metadata, std::move(retry_tensor));
  ASSERT_TRUE(retried) << holoscan::runtime_error_message(retried.error());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  // Key assertion: retry resolves successfully, confirming rollback left the sidecar reusable.
  EXPECT_EQ(received->tensor.data(), retry_data);
  received->tensor = holoscan::Tensor{};
  EXPECT_EQ(retry_counter->load(std::memory_order_relaxed), 1);
}

// Verifies descriptor-backed schema fanout reuses the Tensor sidecar fanout lifetime contract:
// every recipient materializes the same Tensor pointer, and storage is freed only after all
// recipients drop their resolved handles.
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaFanoutKeepsTensorAlive) {
  //   tensor_schema_publisher --[kBlocking, recipients=2]--> subscriber_a
  //                                                      --> subscriber_b
  //   Same Tensor sidecar shared across both; storage freed only after both drop handles.
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto tensor = make_counting_tensor(destroyed_counter, &expected_data, 22U);
  const auto metadata = make_image_metadata_bytes(tensor, "fanout_camera", 223000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking, 2U);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(publisher.value(),
                                                metadata_view,
                                                2U,
                                                std::chrono::seconds(3),
                                                subscriber_a.value(),
                                                subscriber_b.value()));

  auto emitted = publisher->emit(metadata_view, std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 2U);

  auto received_a = subscriber_a->receive();
  ASSERT_TRUE(received_a) << holoscan::runtime_error_message(received_a.error());
  auto sample_a = std::move(received_a).value();
  auto received_b = subscriber_b->receive();
  ASSERT_TRUE(received_b) << holoscan::runtime_error_message(received_b.error());
  auto sample_b = std::move(received_b).value();
  EXPECT_EQ(sample_a.tensor.data(), expected_data);
  EXPECT_EQ(sample_b.tensor.data(), expected_data);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  sample_a.tensor = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  sample_b.tensor = holoscan::Tensor{};
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Verifies descriptor-backed Image metadata can be paired with a real HoloMQ BufferHandle and
// received as a read-only Tensor without resolving a same-process sidecar record.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImageBufferHandleMaterializesImportedTensor) {
  constexpr std::uint64_t kUnboundAuthoringGeneration = 37U;
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  write_host_pattern(buffer.ptr(), kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorPatternSeed);
  const auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))};
  auto source_tensor = holoscan::make_tensor(shape,
                                             DLDataType{kDLUInt, 32, 1},
                                             std::nullopt,
                                             holoscan::MemoryStorageType::kHost,
                                             buffer.ptr());
  ASSERT_TRUE(source_tensor) << holoscan::runtime_error_message(source_tensor.error());
  const auto metadata =
      make_image_metadata_bytes(source_tensor.value(), "buffer_handle_image", 923000);
  ASSERT_FALSE(metadata.empty());

  const holoscan::PayloadDescriptor payload_descriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = 94U,
      .byte_size = kHoloMQSchemaTensorBufferBytes,
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      .buffer = holoscan::channel_internal::make_holomq_payload_reference(*handle)};
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata.data(), metadata.size()),
      payload_descriptor,
      kUnboundAuthoringGeneration);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  const DescriptorBackedImagePayload payload_value = std::move(payload).value();
  // HoloMQ subscriber discovery can lag under rapid repeated test create/destroy cycles. Probe
  // actual payload delivery, not just the publisher's connection count, before measuring the real
  // descriptor-backed materialization behavior below.
  ASSERT_TRUE(wait_for_tensor_schema_payload_recipients(
      publisher.value(), 1U, payload_value, std::chrono::seconds(2), subscriber.value()));
  auto published = publisher->emit_payload(payload_value);
  ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());
  EXPECT_EQ(published.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto sample = std::move(received).value();
  EXPECT_EQ(sample.metadata.endpoint_generation,
            holoscan::channel_internal::kCompatibilitySchemaEndpointGeneration);
  EXPECT_EQ(sample.payload.payload_kind, holoscan::ChannelPayloadKind::kBufferHandle);
  EXPECT_TRUE(sample.tensor.is_read_only());
  EXPECT_EQ(sample.tensor.device().device_type, kDLCUDAHost);
  EXPECT_EQ(sample.tensor.nbytes(), static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes));
  EXPECT_TRUE(verify_host_pattern(
      sample.tensor.data(), kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorPatternSeed));

  // The imported Tensor owns a read-only HoloMQ lease, so the producer pool slot is still checked
  // out until the imported Tensor is released.
  EXPECT_FALSE(fixture->pool.acquire().has_value());
  sample.tensor = holoscan::Tensor{};
  buffer = holomq::Buffer{};
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies normal Image schema emit in BufferHandle mode retains its source Tensor while the
// transport sample is queued, holds that transport loan through buffer import, and releases the
// pool slot only after the imported Tensor is released.
TEST(ChannelSchemaPayloadTest,
     DescriptorBackedImageBufferHandleEmitRetainsOwnerUntilSubscriberReleasesSample) {
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

  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))};
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer trigger_source_buffer = std::move(*trigger_buffer);
  auto trigger_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(trigger_source_buffer));
  ASSERT_TRUE(trigger_tensor) << holoscan::runtime_error_message(trigger_tensor.error());
  const auto trigger_metadata =
      make_image_metadata_bytes(trigger_tensor.value(), "buffer_handle_trigger", 925000);

  auto buffer = fixture->pool.acquire();
  ASSERT_TRUE(buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer source_buffer = std::move(*buffer);
  write_host_pattern(
      source_buffer.ptr(), kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorPatternSeed);
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(source_buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());
  const auto metadata = make_image_metadata_bytes(tensor.value(), "buffer_handle_retained", 926000);

  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto emitted = publisher->emit(metadata, std::move(tensor).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_TRUE(received->tensor.is_read_only());
  EXPECT_TRUE(verify_host_pattern(
      received->tensor.data(), kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorPatternSeed));
  EXPECT_FALSE(fixture->pool.acquire().has_value());
  received->tensor = holoscan::Tensor{};
  auto trigger_emitted = publisher->emit(trigger_metadata, std::move(trigger_tensor).value());
  ASSERT_TRUE(trigger_emitted) << holoscan::runtime_error_message(trigger_emitted.error());

  std::optional<holomq::Buffer> reused;
  EXPECT_TRUE(
      wait_until([&fixture, &reused]() { return try_acquire_buffer(fixture->pool, reused); },
                 std::chrono::seconds(2)));
}

// Verifies normal Image BufferHandle publication preserves asynchronous producer readiness through
// materialization and defers pool reuse until explicit consumer-stream work completes.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImageBufferHandleOrdersProducerAndConsumerStreams) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  auto trigger_pool = fixture->context.create_device_buffer_pool(kHoloMQSchemaTensorBufferBytes,
                                                                 kHoloMQSchemaTensorBufferCount);
  ASSERT_TRUE(trigger_pool.has_value());
  auto trigger_buffer = trigger_pool->acquire();
  ASSERT_TRUE(trigger_buffer.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer trigger_source_buffer = std::move(*trigger_buffer);
  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))};
  auto trigger_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(trigger_source_buffer));
  ASSERT_TRUE(trigger_tensor) << holoscan::runtime_error_message(trigger_tensor.error());
  const auto trigger_metadata =
      make_image_metadata_bytes(trigger_tensor.value(), "buffer_handle_stream_trigger", 928000);

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
  constexpr std::uint8_t kPattern = 0xa5U;
  ASSERT_EQ(cudaMemsetAsync(
                *tensor_data, kPattern, kHoloMQSchemaTensorBufferBytes, producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(tensor->set_readiness_stream(producer_stream.get()));
  const auto metadata =
      make_image_metadata_bytes(tensor.value(), "buffer_handle_stream_ordered", 927000);

  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto emitted = publisher->emit(metadata, std::move(tensor).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto sample = std::move(received).value();
  EXPECT_TRUE(sample.tensor.is_read_only());
  EXPECT_TRUE(sample.tensor.has_readiness_stream_contract());
  auto synchronized = sample.tensor.synchronize_for_consumer_stream(consumer_stream.get());
  ASSERT_TRUE(synchronized) << holoscan::runtime_error_message(synchronized.error());

  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, kHoloMQSchemaTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> host(
      static_cast<std::uint8_t*>(raw_host));
  ASSERT_EQ(cudaMemcpyAsync(host.get(),
                            sample.tensor.data(),
                            kHoloMQSchemaTensorBufferBytes,
                            cudaMemcpyDeviceToHost,
                            consumer_stream.get()),
            cudaSuccess);

  holoscan::test::BlockingHostState consumer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                consumer_stream.get(), holoscan::test::blocking_host_callback, &consumer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard consumer_guard(consumer_state);
  ASSERT_TRUE(sample.tensor.set_deallocation_stream(consumer_stream.get()));
  sample.tensor = holoscan::Tensor{};
  auto trigger_emitted = publisher->emit(trigger_metadata, std::move(trigger_tensor).value());
  ASSERT_TRUE(trigger_emitted) << holoscan::runtime_error_message(trigger_emitted.error());
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(consumer_state));
  for (std::size_t i = 0; i < kHoloMQSchemaTensorBufferBytes; ++i) {
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

// REQ: CHAN-126
// REQ: CHAN-127
TEST(ChannelSchemaPayloadTest,
     MultiTensorStagingJoinsDeviceReadinessAndPreservesHeterogeneousFields) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  auto second_source_pool = fixture->context.create_device_buffer_pool(
      kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorBufferCount);
  ASSERT_TRUE(second_source_pool);
  auto first_source_buffer = fixture->pool.acquire();
  auto second_source_buffer = second_source_pool->acquire();
  ASSERT_TRUE(first_source_buffer);
  ASSERT_TRUE(second_source_buffer);

  holoscan::test::UniqueCudaStream first_producer_stream;
  holoscan::test::UniqueCudaStream second_producer_stream;
  holoscan::test::UniqueCudaStream consumer_stream;
  holoscan::test::UniqueCudaStream unfenced_probe_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(first_producer_stream), cudaSuccess);
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(second_producer_stream), cudaSuccess);
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(unfenced_probe_stream), cudaSuccess);
  holoscan::test::BlockingHostState first_producer_state;
  ASSERT_EQ(cudaLaunchHostFunc(first_producer_stream.get(),
                               holoscan::test::blocking_host_callback,
                               &first_producer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard first_producer_guard(first_producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(first_producer_state));

  constexpr std::uint8_t kFirstPattern = 0x3cU;
  constexpr std::uint8_t kSecondPattern = 0xa7U;
  const std::array<std::int64_t, 1U> device_shape{
      static_cast<std::int64_t>(kHoloMQSchemaTensorBufferBytes)};
  auto first_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      device_shape,
      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      std::nullopt,
      std::move(*first_source_buffer));
  auto second_tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      device_shape,
      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      std::nullopt,
      std::move(*second_source_buffer));
  ASSERT_TRUE(first_tensor);
  ASSERT_TRUE(second_tensor);
  auto first_data = first_tensor->mutable_data();
  auto second_data = second_tensor->mutable_data();
  ASSERT_TRUE(first_data);
  ASSERT_TRUE(second_data);
  ASSERT_EQ(
      cudaMemsetAsync(
          *first_data, kFirstPattern, kHoloMQSchemaTensorBufferBytes, first_producer_stream.get()),
      cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(*second_data,
                            kSecondPattern,
                            kHoloMQSchemaTensorBufferBytes,
                            second_producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(first_tensor->set_readiness_stream(first_producer_stream.get()));
  ASSERT_TRUE(second_tensor->set_readiness_stream(second_producer_stream.get()));
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  void* host_data{};
  const holoscan::Tensor host_tensor = make_counting_tensor(destroyed, &host_data, 0x51U);

  constexpr std::size_t kDescriptorOffset =
      offsetof(MultiTensorStagingEnvelope, payload) +
      offsetof(MultiTensorStagingPayload, descriptor) +
      offsetof(MultiTensorStagingPayload::descriptor_carriage_type, descriptors);
  const std::array descriptor_offsets{
      kDescriptorOffset,
      kDescriptorOffset + sizeof(holoscan::PayloadDescriptor),
      kDescriptorOffset + 2U * sizeof(holoscan::PayloadDescriptor),
  };
  constexpr DLDataType kUInt8{.code = kDLUInt, .bits = 8U, .lanes = 1U};
  const std::array staging{
      holoscan::channel_internal::TensorSchemaStagingOptions{
          .max_byte_span = kHoloMQSchemaTensorBufferBytes,
          .slot_count = 2U,
          .dtype = kUInt8,
          .rank = 1U,
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .device = holoscan::DeviceId{0},
          .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      },
      holoscan::channel_internal::TensorSchemaStagingOptions{
          .max_byte_span = 6U,
          .slot_count = 2U,
          .dtype = kUInt8,
          .rank = 2U,
          .memory_kind = holoscan::MemoryKind::kHost,
          .device = {},
          .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      },
      holoscan::channel_internal::TensorSchemaStagingOptions{
          .max_byte_span = kHoloMQSchemaTensorBufferBytes,
          .slot_count = 2U,
          .dtype = kUInt8,
          .rank = 1U,
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .device = holoscan::DeviceId{0},
          .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      },
  };
  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_publisher(
      *runtime,
      channel_name,
      sizeof(MultiTensorStagingEnvelope),
      descriptor_offsets,
      options,
      staging);
  auto subscriber = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_subscriber(
      *runtime, channel_name, sizeof(MultiTensorStagingEnvelope), 3U, options);
  ASSERT_TRUE(publisher);
  ASSERT_TRUE(subscriber);
  ASSERT_TRUE(wait_for_subscribers(**publisher, std::chrono::seconds(2)));

  // Key setup: stage two CUDA fields produced on distinct streams around one ordinary host field;
  // the first producer remains blocked while the second can finish independently.
  const std::array tensors{*first_tensor, host_tensor, *second_tensor};
  auto direct_only = (*publisher)->reserve_tensors(tensors, true);
  // Key assertion: one non-exportable field rejects an all-direct transaction before staging.
  ASSERT_FALSE(direct_only);
  auto reservation = (*publisher)->reserve_tensors(tensors);
  ASSERT_TRUE(reservation) << holoscan::runtime_error_message(reservation.error());
  ASSERT_EQ(reservation->field_count, tensors.size());
  MultiTensorStagingEnvelope sent{};
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    auto* field = reservation->field_at(index);
    ASSERT_NE(field, nullptr);
    field->descriptor.schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id;
    *sent.payload.descriptor_at(index) = field->descriptor;
  }
  const auto sent_bytes = std::as_bytes(std::span<const MultiTensorStagingEnvelope, 1U>(&sent, 1U));
  auto published = (*publisher)->publish_payload(sent_bytes, std::move(*reservation));
  ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());

  MultiTensorStagingEnvelope received{};
  auto received_loan = (*subscriber)
                           ->receive_payload(std::as_writable_bytes(
                               std::span<MultiTensorStagingEnvelope, 1U>(&received, 1U)));
  ASSERT_TRUE(received_loan) << holoscan::runtime_error_message(received_loan.error());

  const holoscan::TensorViewMetadata device_metadata{
      .rank = 1U,
      .dtype = kUInt8,
      .device = DLDevice{.device_type = kDLCUDA, .device_id = 0},
      .has_explicit_byte_strides = false,
      .byte_size = kHoloMQSchemaTensorBufferBytes,
      .shape = {static_cast<std::int64_t>(kHoloMQSchemaTensorBufferBytes)},
  };
  const holoscan::TensorViewMetadata host_metadata{
      .rank = 2U,
      .dtype = kUInt8,
      .device = DLDevice{.device_type = kDLCPU, .device_id = 0},
      .has_explicit_byte_strides = true,
      .byte_size = 6U,
      .shape = {2, 3},
      .byte_strides = {3, 1},
  };
  auto first_received =
      (*subscriber)
          ->import_buffer_handle_tensor(*received.payload.descriptor_at(0U), device_metadata, 0U);
  auto host_received =
      (*subscriber)
          ->import_buffer_handle_tensor(*received.payload.descriptor_at(1U), host_metadata, 1U);
  auto second_received =
      (*subscriber)
          ->import_buffer_handle_tensor(*received.payload.descriptor_at(2U), device_metadata, 2U);
  ASSERT_TRUE(first_received);
  ASSERT_TRUE(host_received);
  ASSERT_TRUE(second_received);

  auto second_synchronized =
      second_received->synchronize_for_consumer_stream(unfenced_probe_stream.get());
  ASSERT_TRUE(second_synchronized);
  EXPECT_EQ(cudaStreamQuery(unfenced_probe_stream.get()), cudaErrorNotReady);

  // Key assertion: field two independently waits for the joined copy stream; the heterogeneous
  // host field retains its planned host storage and exact bytes.
  EXPECT_EQ(host_received->device().device_type, kDLCPU);
  EXPECT_TRUE(std::equal(static_cast<const std::uint8_t*>(host_received->data()),
                         static_cast<const std::uint8_t*>(host_received->data()) + 6U,
                         static_cast<const std::uint8_t*>(host_data)));
  auto first_synchronized = first_received->synchronize_for_consumer_stream(consumer_stream.get());
  ASSERT_TRUE(first_synchronized);
  // Pinned destinations keep D2H enqueue asynchronous; pageable storage can block before the
  // producer gate below is released and deadlock this readiness test.
  void* first_raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&first_raw_host, kHoloMQSchemaTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> first_host(
      static_cast<std::uint8_t*>(first_raw_host));
  void* second_raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&second_raw_host, kHoloMQSchemaTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> second_host(
      static_cast<std::uint8_t*>(second_raw_host));
  ASSERT_EQ(cudaMemcpyAsync(first_host.get(),
                            first_received->data(),
                            kHoloMQSchemaTensorBufferBytes,
                            cudaMemcpyDeviceToHost,
                            consumer_stream.get()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(second_host.get(),
                            second_received->data(),
                            kHoloMQSchemaTensorBufferBytes,
                            cudaMemcpyDeviceToHost,
                            consumer_stream.get()),
            cudaSuccess);
  first_producer_guard.dismiss();
  holoscan::test::release_blocking_host(first_producer_state);
  ASSERT_EQ(cudaStreamSynchronize(unfenced_probe_stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  EXPECT_TRUE(std::ranges::all_of(
      std::span<const std::uint8_t>(first_host.get(), kHoloMQSchemaTensorBufferBytes),
      [](const auto value) { return value == kFirstPattern; }));
  EXPECT_TRUE(std::ranges::all_of(
      std::span<const std::uint8_t>(second_host.get(), kHoloMQSchemaTensorBufferBytes),
      [](const auto value) { return value == kSecondPattern; }));
}

// An incomplete-fanout rejection retires its unsent HoloMQ loan synchronously, but the staging
// D2D copy remains local CUDA work. The sole slot must stay backpressured until that copy
// completes. REQ: CHAN-126
TEST(ChannelSchemaPayloadTest, DeviceStagingSlotWaitsForCopyAfterIncompleteFanout) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);

  holoscan::test::UniqueCudaStream producer_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  holoscan::test::BlockingHostState producer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                producer_stream.get(), holoscan::test::blocking_host_callback, &producer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard producer_guard(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(producer_state));

  auto source_pool = std::make_shared<holoscan::BlockMemoryPool>(
      holoscan::MemoryStorageType::kDevice, kHoloMQSchemaTensorBufferBytes, 1U, 0);
  ASSERT_TRUE(source_pool->initialize_pool());
  const std::array<std::int64_t, 1U> shape{
      static_cast<std::int64_t>(kHoloMQSchemaTensorBufferBytes)};
  auto source = holoscan::make_tensor(shape,
                                      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kDevice,
                                      source_pool,
                                      producer_stream.get());
  ASSERT_TRUE(source) << holoscan::runtime_error_message(source.error());
  auto source_data = source->mutable_data();
  ASSERT_TRUE(source_data) << holoscan::runtime_error_message(source_data.error());
  ASSERT_EQ(
      cudaMemsetAsync(*source_data, 0x6d, kHoloMQSchemaTensorBufferBytes, producer_stream.get()),
      cudaSuccess);
  ASSERT_TRUE(source->set_readiness_stream(producer_stream.get()));

  constexpr std::array<std::size_t, 1U> kDescriptorOffsets{0U};
  constexpr std::array staging{holoscan::channel_internal::TensorSchemaStagingOptions{
      .max_byte_span = kHoloMQSchemaTensorBufferBytes,
      .slot_count = 1U,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .rank = 1U,
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .device = holoscan::DeviceId{0},
      .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
  }};
  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_publisher(
      *runtime,
      make_unique_channel_name(),
      sizeof(holoscan::PayloadDescriptor),
      kDescriptorOffsets,
      options,
      staging);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  const std::array tensors{*source};
  auto first = (*publisher)->reserve_tensors(tensors);
  ASSERT_TRUE(first) << holoscan::runtime_error_message(first.error());
  std::array<std::byte, sizeof(holoscan::PayloadDescriptor)> payload{};
  auto rejected = (*publisher)->publish_payload(payload, std::move(*first));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kNotReady);

  // Key assertion: synchronous callback retirement alone cannot recycle the cold-bound slot.
  auto blocked = (*publisher)->reserve_tensors(tensors);
  ASSERT_FALSE(blocked);
  EXPECT_EQ(blocked.error().code, holoscan::ErrorCode::kBackpressured);

  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  ASSERT_EQ(cudaStreamSynchronize(producer_stream.get()), cudaSuccess);
  auto recycled = (*publisher)->reserve_tensors(tensors);
  ASSERT_TRUE(recycled) << holoscan::runtime_error_message(recycled.error());
  (*publisher)->release_tensor(std::move(*recycled));
  ASSERT_EQ(cudaStreamSynchronize(producer_stream.get()), cudaSuccess);
}

// REQ: CHAN-126
TEST(ChannelSchemaPayloadTest, MultiTensorReservationHotPathDoesNotAllocateAfterWarmup) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kSameProcessSidecar;
  constexpr std::size_t kDescriptorOffset =
      offsetof(MultiTensorStagingEnvelope, payload) +
      offsetof(MultiTensorStagingPayload, descriptor) +
      offsetof(MultiTensorStagingPayload::descriptor_carriage_type, descriptors);
  constexpr std::array descriptor_offsets{
      kDescriptorOffset,
      kDescriptorOffset + sizeof(holoscan::PayloadDescriptor),
      kDescriptorOffset + 2U * sizeof(holoscan::PayloadDescriptor),
  };
  const std::string channel_name = make_unique_channel_name();
  auto publisher = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_publisher(
      *runtime,
      channel_name,
      sizeof(MultiTensorStagingEnvelope),
      descriptor_offsets,
      options,
      std::span<const holoscan::channel_internal::TensorSchemaStagingOptions>{});
  auto subscriber = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_subscriber(
      *runtime, channel_name, sizeof(MultiTensorStagingEnvelope), 3U, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(**publisher, std::chrono::seconds(2)));

  auto destroyed = std::make_shared<std::atomic<int>>(0);
  void* data{};
  const holoscan::Tensor tensor = make_counting_tensor(destroyed, &data, 0x6dU);
  const std::array tensors{tensor, tensor, tensor};
  const auto reserve_once = [&]() {
    auto reservation = (*publisher)->reserve_tensors(tensors);
    if (!reservation) {
      return false;
    }
    bool fields_valid = true;
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      const auto* field = reservation->field_at(index);
      if (field == nullptr ||
          field->descriptor.payload_kind != holoscan::ChannelPayloadKind::kLocalBufferDescriptor) {
        fields_valid = false;
        break;
      }
    }
    (*publisher)->release_tensor(std::move(*reservation));
    return fields_valid;
  };

  // Key setup: one complete three-field sidecar reservation primes the fixed reservation storage.
  ASSERT_TRUE(reserve_once());
  bool all_cycles_succeeded = true;
  std::size_t allocation_count = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
      all_cycles_succeeded = reserve_once() && all_cycles_succeeded;
    }
    allocation_count = allocations.count();
  }

  // Key assertion: the cold-preallocated multi-field reservation path performs no host heap
  // allocation after warmup or between reservation releases.
  EXPECT_TRUE(all_cycles_succeeded);
  EXPECT_EQ(allocation_count, 0U);
}

// REQ: CHAN-126
TEST(ChannelSchemaPayloadTest, MultiTensorReservationReapRetryPreservesHotPathAndDiagnostics) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);
  auto source_buffer = fixture->pool.acquire();
  ASSERT_TRUE(source_buffer);
  const std::array<std::int64_t, 1U> shape{
      static_cast<std::int64_t>(kHoloMQSchemaTensorBufferBytes)};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape,
      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      std::nullopt,
      std::move(*source_buffer));
  ASSERT_TRUE(tensor) << holoscan::runtime_error_message(tensor.error());

  constexpr std::size_t kDescriptorOffset =
      offsetof(MultiTensorStagingEnvelope, payload) +
      offsetof(MultiTensorStagingPayload, descriptor) +
      offsetof(MultiTensorStagingPayload::descriptor_carriage_type, descriptors);
  constexpr std::array descriptor_offsets{
      kDescriptorOffset,
      kDescriptorOffset + sizeof(holoscan::PayloadDescriptor),
      kDescriptorOffset + 2U * sizeof(holoscan::PayloadDescriptor),
  };
  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 1U;
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  auto publisher = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_publisher(
      *runtime,
      channel_name,
      sizeof(MultiTensorStagingEnvelope),
      descriptor_offsets,
      options,
      std::span<const holoscan::channel_internal::TensorSchemaStagingOptions>{});
  auto subscriber = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_subscriber(
      *runtime, channel_name, sizeof(MultiTensorStagingEnvelope), 3U, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(**publisher, std::chrono::seconds(2)));

  const std::array tensors{*tensor, *tensor, *tensor};
  auto pending = (*publisher)->reserve_tensors(tensors, true);
  ASSERT_TRUE(pending) << holoscan::runtime_error_message(pending.error());
  MultiTensorStagingEnvelope payload{};
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    auto* field = pending->field_at(index);
    ASSERT_NE(field, nullptr);
    field->descriptor.schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id;
    *payload.payload.descriptor_at(index) = field->descriptor;
  }
  const auto payload_bytes =
      std::as_bytes(std::span<const MultiTensorStagingEnvelope, 1U>(&payload, 1U));
  auto published = (*publisher)->publish_payload(payload_bytes, std::move(*pending));
  ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());
  {
    MultiTensorStagingEnvelope received{};
    auto returned = (*subscriber)
                        ->receive_payload(std::as_writable_bytes(
                            std::span<MultiTensorStagingEnvelope, 1U>(&received, 1U)));
    ASSERT_TRUE(returned) << holoscan::runtime_error_message(returned.error());
  }

  // Retain the two free metadata slots while the returned publication's third slot awaits the
  // publisher-loan callback that the pressure-recovery path dispatches.
  std::array<holoscan::channel_internal::TensorSchemaPayloadReservation, 2U> held{};
  for (auto& reservation : held) {
    auto reserved = (*publisher)->reserve_tensors(tensors, true);
    ASSERT_TRUE(reserved) << holoscan::runtime_error_message(reserved.error());
    reservation = std::move(*reserved);
  }

  holoscan::channel_internal::TensorSchemaPayloadReservation recovered_reservation;
  std::size_t allocation_count = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    auto recovered = (*publisher)->reserve_tensors(tensors, true);
    allocation_count = allocations.count();
    ASSERT_TRUE(recovered) << holoscan::runtime_error_message(recovered.error());
    recovered_reservation = std::move(*recovered);
  }

  // Transient pressure, callback reaping, and the successful retry are allocation-free.
  EXPECT_EQ(allocation_count, 0U);

  // All three reservation slots are now genuinely retained and no returned callback can release
  // one. The reap-and-retry sequence must fail with useful terminal context.
  auto exhausted = (*publisher)->reserve_tensors(tensors, true);
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_EQ(exhausted.error().message,
            "Tensor schema reservation pool remains exhausted after release-callback reap retry");

  (*publisher)->release_tensor(std::move(recovered_reservation));
  for (auto& reservation : held) {
    (*publisher)->release_tensor(std::move(reservation));
  }
}

// REQ: CHAN-126
TEST(ChannelSchemaPayloadTest, MultiTensorStagingRecordsPerFieldPoolSetupCostBaseline) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  constexpr DLDataType kUInt8{.code = kDLUInt, .bits = 8U, .lanes = 1U};

  // Key setup: construct one independent host staging pool per field for N = 1, 3, and 5,
  // sampling each complete endpoint setup three times.
  for (const std::size_t field_count : {1U, 3U, 5U}) {
    std::vector<std::size_t> descriptor_offsets(field_count);
    std::vector<holoscan::channel_internal::TensorSchemaStagingOptions> staging(field_count);
    for (std::size_t index = 0U; index < field_count; ++index) {
      descriptor_offsets[index] = index * sizeof(holoscan::PayloadDescriptor);
      staging[index] = holoscan::channel_internal::TensorSchemaStagingOptions{
          .max_byte_span = kHoloMQSchemaTensorBufferBytes,
          .slot_count = 1U,
          .dtype = kUInt8,
          .rank = 1U,
          .memory_kind = holoscan::MemoryKind::kHost,
          .device = {},
          .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      };
    }
    std::array<std::int64_t, 3U> samples{};
    for (std::size_t iteration = 0U; iteration < samples.size(); ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      auto publisher =
          holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_publisher(
              *runtime,
              make_unique_channel_name(),
              field_count * sizeof(holoscan::PayloadDescriptor),
              descriptor_offsets,
              options,
              staging);
      const auto stop = std::chrono::steady_clock::now();
      ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
      samples[iteration] =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }
    std::ranges::sort(samples);
    // Key assertion: publish each median as a hardware-specific baseline for any later pool
    // grouping optimization; this test deliberately imposes no machine-dependent threshold.
    RecordProperty("staging_setup_us_n" + std::to_string(field_count), samples[1U]);
  }
}

// Verifies multi-Tensor publisher setup rejects fields that alias one descriptor slot.
TEST(ChannelSchemaPayloadTest, MultiTensorPublisherRejectsDuplicateDescriptorOffsets) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  constexpr std::size_t kPayloadSize = 2U * sizeof(holoscan::PayloadDescriptor);
  constexpr std::array<std::size_t, 2U> kDuplicateOffsets{0U, 0U};

  // Key setup: two declared Tensor fields alias the same descriptor bytes.
  auto publisher = holoscan::detail::RuntimeChannelBindingAccess::create_tensor_schema_publisher(
      *runtime,
      make_unique_channel_name(),
      kPayloadSize,
      kDuplicateOffsets,
      options,
      std::span<const holoscan::channel_internal::TensorSchemaStagingOptions>{});

  // Key assertion: setup rejects aliasing before a publisher or replacement callback can exist.
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies relaying a descriptor-backed Image preserves both its metadata and the original
// producer fence. The downstream Tensor must wait for S0 on S1 and keep the source pool slot
// unavailable until S1's consumer work completes.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImageBufferHandleRelayPreservesStreamOrdering) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto upstream_trigger_pool = fixture->context.create_device_buffer_pool(
      kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorBufferCount);
  ASSERT_TRUE(upstream_trigger_pool.has_value());
  auto upstream_trigger_buffer = upstream_trigger_pool->acquire();
  ASSERT_TRUE(upstream_trigger_buffer.has_value());
  auto downstream_trigger_pool = fixture->context.create_device_buffer_pool(
      kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorBufferCount);
  ASSERT_TRUE(downstream_trigger_pool.has_value());
  auto downstream_trigger_buffer = downstream_trigger_pool->acquire();
  ASSERT_TRUE(downstream_trigger_buffer.has_value());

  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))};
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer upstream_trigger_source = std::move(*upstream_trigger_buffer);
  auto upstream_trigger = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(upstream_trigger_source));
  ASSERT_TRUE(upstream_trigger) << holoscan::runtime_error_message(upstream_trigger.error());
  const auto upstream_trigger_metadata = make_image_metadata_bytes(
      upstream_trigger.value(), "buffer_handle_relay_upstream_trigger", 929000);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer downstream_trigger_source = std::move(*downstream_trigger_buffer);
  auto downstream_trigger = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, std::move(downstream_trigger_source));
  ASSERT_TRUE(downstream_trigger) << holoscan::runtime_error_message(downstream_trigger.error());
  const auto downstream_trigger_metadata = make_image_metadata_bytes(
      downstream_trigger.value(), "buffer_handle_relay_downstream_trigger", 930000);

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
  constexpr std::uint8_t kPattern = 0x6dU;
  ASSERT_EQ(cudaMemsetAsync(
                *source_data, kPattern, kHoloMQSchemaTensorBufferBytes, producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(source->set_readiness_stream(producer_stream.get()));
  const auto metadata =
      make_image_metadata_bytes(source.value(), "buffer_handle_relay_ordered", 931000);

  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  //   upstream_publisher --[kBufferHandle, kBlocking]--> upstream_subscriber
  //   upstream_subscriber (relay) --> downstream_publisher
  //   downstream_publisher --[kBufferHandle, kBlocking]--> downstream_subscriber
  //   Producer stream S0 blocked; consumer stream S1 synchronizes on S0 before D2H copy.
  const std::string upstream_name = make_unique_channel_name();
  auto upstream_publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(upstream_name, options);
  ASSERT_TRUE(upstream_publisher) << holoscan::runtime_error_message(upstream_publisher.error());
  auto upstream_subscriber = runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(
      upstream_name, options);
  ASSERT_TRUE(upstream_subscriber) << holoscan::runtime_error_message(upstream_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(upstream_publisher.value(), std::chrono::seconds(2)));

  const std::string downstream_name = make_unique_channel_name();
  auto downstream_publisher = runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(
      downstream_name, options);
  ASSERT_TRUE(downstream_publisher)
      << holoscan::runtime_error_message(downstream_publisher.error());
  auto downstream_subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(downstream_name,
                                                                             options);
  ASSERT_TRUE(downstream_subscriber)
      << holoscan::runtime_error_message(downstream_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(downstream_publisher.value(), std::chrono::seconds(2)));

  auto emitted = upstream_publisher->emit(metadata, std::move(source).value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto imported = upstream_subscriber->receive();
  ASSERT_TRUE(imported) << holoscan::runtime_error_message(imported.error());
  auto upstream_sample = std::move(imported).value();
  EXPECT_TRUE(upstream_sample.tensor.is_read_only());
  EXPECT_TRUE(upstream_sample.tensor.has_readiness_stream_contract());
  auto relayed = downstream_publisher->emit(upstream_sample.metadata_bytes(),
                                            std::move(upstream_sample.tensor));
  ASSERT_TRUE(relayed) << holoscan::runtime_error_message(relayed.error());
  auto upstream_trigger_emitted =
      upstream_publisher->emit(upstream_trigger_metadata, std::move(upstream_trigger).value());
  ASSERT_TRUE(upstream_trigger_emitted)
      << holoscan::runtime_error_message(upstream_trigger_emitted.error());

  auto downstream = downstream_subscriber->receive();
  ASSERT_TRUE(downstream) << holoscan::runtime_error_message(downstream.error());
  auto downstream_sample = std::move(downstream).value();
  EXPECT_TRUE(downstream_sample.tensor.is_read_only());
  EXPECT_TRUE(downstream_sample.tensor.has_readiness_stream_contract());
  const holoscan::schema::Image* downstream_image = downstream_sample.root();
  ASSERT_NE(downstream_image, nullptr);
  ASSERT_NE(downstream_image->header(), nullptr);
  ASSERT_NE(downstream_image->header()->stamp(), nullptr);
  EXPECT_EQ(downstream_image->header()->stamp()->time_ns(), 931000);
  ASSERT_NE(downstream_image->header()->frame_id(), nullptr);
  EXPECT_EQ(downstream_image->header()->frame_id()->string_view(),
            std::string_view("buffer_handle_relay_ordered"));

  auto synchronized =
      downstream_sample.tensor.synchronize_for_consumer_stream(consumer_stream.get());
  ASSERT_TRUE(synchronized) << holoscan::runtime_error_message(synchronized.error());
  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, kHoloMQSchemaTensorBufferBytes), cudaSuccess);
  const std::unique_ptr<std::uint8_t, holoscan::test::CudaPinnedHostDeleter> host(
      static_cast<std::uint8_t*>(raw_host));
  ASSERT_EQ(cudaMemcpyAsync(host.get(),
                            downstream_sample.tensor.data(),
                            kHoloMQSchemaTensorBufferBytes,
                            cudaMemcpyDeviceToHost,
                            consumer_stream.get()),
            cudaSuccess);

  holoscan::test::BlockingHostState consumer_state;
  ASSERT_EQ(cudaLaunchHostFunc(
                consumer_stream.get(), holoscan::test::blocking_host_callback, &consumer_state),
            cudaSuccess);
  holoscan::test::BlockingHostReleaseGuard consumer_guard(consumer_state);
  ASSERT_TRUE(downstream_sample.tensor.set_deallocation_stream(consumer_stream.get()));
  downstream_sample.tensor = holoscan::Tensor{};
  auto downstream_trigger_emitted = downstream_publisher->emit(
      downstream_trigger_metadata, std::move(downstream_trigger).value());
  ASSERT_TRUE(downstream_trigger_emitted)
      << holoscan::runtime_error_message(downstream_trigger_emitted.error());
  EXPECT_FALSE(fixture->pool.acquire().has_value());

  producer_guard.dismiss();
  holoscan::test::release_blocking_host(producer_state);
  ASSERT_TRUE(holoscan::test::wait_until_started(consumer_state));
  for (std::size_t i = 0; i < kHoloMQSchemaTensorBufferBytes; ++i) {
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

// Verifies the descriptor-backed Image BufferHandle path across a real process boundary: the
// parent publishes metadata plus a HoloMQ BufferHandle and an exec'd child process receives through
// ChannelTensorSchemaSubscriber, imports the buffer, and verifies the materialized read-only
// Tensor.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImageBufferHandleMaterializesImportedTensorInChild) {
  const char* const child_exe = HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE not configured";

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
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
    const std::string expected_bytes = std::to_string(kHoloMQSchemaTensorBufferBytes);
    const std::string expected_seed = std::to_string(kHoloMQSchemaTensorPatternSeed);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_NAME", channel_name.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_MODE", "image", 1));
    static_cast<void>(
        ::setenv("HOLOSCAN_CHANNEL_SCHEMA_EXPECTED_BYTES", expected_bytes.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_EXPECTED_SEED", expected_seed.c_str(), 1));
    ::execl(child_exe, "channel_schema_interprocess_child", static_cast<char*>(nullptr));
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

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  write_host_pattern(buffer.ptr(), kHoloMQSchemaTensorBufferBytes, kHoloMQSchemaTensorPatternSeed);
  const auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))};
  auto source_tensor = holoscan::make_tensor(shape,
                                             DLDataType{kDLUInt, 32, 1},
                                             std::nullopt,
                                             holoscan::MemoryStorageType::kHost,
                                             buffer.ptr());
  ASSERT_TRUE(source_tensor) << holoscan::runtime_error_message(source_tensor.error());
  const auto metadata =
      make_image_metadata_bytes(source_tensor.value(), "buffer_handle_child_image", 924000);
  ASSERT_FALSE(metadata.empty());

  const holoscan::PayloadDescriptor payload_descriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = 95U,
      .byte_size = kHoloMQSchemaTensorBufferBytes,
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      .buffer = holoscan::channel_internal::make_holomq_payload_reference(*handle)};
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata.data(), metadata.size()), payload_descriptor);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  const DescriptorBackedImagePayload payload_value = std::move(payload).value();

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<DescriptorBackedImagePayload>(channel_name, options);
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

  std::string last_emit_error;
  bool child_exited_early = false;
  int child_status = 0;
  std::uint8_t early_child_code = 0xffU;
  const bool delivered = wait_until(
      [&publisher,
       &payload_value,
       &last_emit_error,
       child,
       &child_exited_early,
       &child_status,
       &early_child_code,
       &child_cleanup] {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit(payload_value);
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
  ASSERT_TRUE(delivered) << "failed to publish descriptor-backed Image payload to child: "
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

  buffer = holomq::Buffer{};
  EXPECT_TRUE(fixture->pool.acquire().has_value());
}

// Verifies descriptor-backed Image buffer-handle materialization rejects malformed schema
// metadata or descriptors before exposing an imported Tensor to runtime code.
TEST(ChannelSchemaPayloadTest, DescriptorBackedImageBufferHandleRejectsInvalidMaterialization) {
  holoscan::test::skip_without_cuda_device();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto fixture = make_holomq_pinned_host_pool_fixture();
  ASSERT_NE(fixture, nullptr);

  auto buffer_or = fixture->pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  holomq::Buffer buffer = std::move(*buffer_or);
  const auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  const holoscan::schema::DLDataType uint32_dtype{holoscan::schema::DLDataTypeCode_kDLUInt,
                                                  /*bits=*/32,
                                                  /*lanes=*/1};
  const holoscan::schema::DLDataType invalid_dtype{holoscan::schema::DLDataTypeCode_kDLUInt,
                                                   /*bits=*/0,
                                                   /*lanes=*/1};
  const holoscan::schema::DLDevice host_device{holoscan::schema::DLDeviceType_kDLCUDAHost,
                                               /*device_id=*/0};
  const holoscan::schema::DLDevice cuda_device{holoscan::schema::DLDeviceType_kDLCUDA,
                                               /*device_id=*/0};

  auto descriptor_for_handle = [&](const std::uint64_t byte_size,
                                   const holomq::BufferHandle& selected_handle) {
    return holoscan::PayloadDescriptor{
        .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
        .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
        .sequence = 95U,
        .byte_size = byte_size,
        .buffer = holoscan::channel_internal::make_holomq_payload_reference(selected_handle)};
  };
  auto valid_descriptor = [&](const std::uint64_t byte_size) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
    return descriptor_for_handle(byte_size, *handle);
  };
  const auto valid_metadata = make_image_metadata_with_tensor_fields(
      {static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))},
      /*ndim=*/1U,
      &uint32_dtype,
      &host_device);
  auto valid_payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(valid_metadata.data(), valid_metadata.size()),
      valid_descriptor(kHoloMQSchemaTensorBufferBytes));
  ASSERT_TRUE(valid_payload) << holoscan::runtime_error_message(valid_payload.error());
  // Probe actual descriptor-backed payload delivery before sending malformed samples. A connection
  // count alone can briefly race HoloMQ subscriber discovery during repeated test runs.
  ASSERT_TRUE(wait_for_tensor_schema_payload_recipients(
      publisher.value(), 1U, valid_payload.value(), std::chrono::seconds(2), subscriber.value()));

  auto emit_and_expect_error = [&](std::vector<std::byte> metadata,
                                   holoscan::PayloadDescriptor descriptor,
                                   holoscan::ErrorCode expected_code) {
    auto payload = DescriptorBackedImagePayload::copy_from(
        std::span<const std::byte>(metadata.data(), metadata.size()), descriptor);
    ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
    auto published = publisher->emit_payload(payload.value());
    ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());
    auto received = subscriber->receive();
    ASSERT_FALSE(received);
    EXPECT_EQ(received.error().code, expected_code)
        << holoscan::runtime_error_message(received.error());
  };

  emit_and_expect_error(make_image_metadata_without_tensor(),
                        valid_descriptor(kHoloMQSchemaTensorBufferBytes),
                        holoscan::ErrorCode::kCodecError);
  emit_and_expect_error(
      make_image_metadata_with_tensor_fields(
          {static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))},
          /*ndim=*/1U,
          &invalid_dtype,
          &host_device),
      valid_descriptor(kHoloMQSchemaTensorBufferBytes),
      holoscan::ErrorCode::kCodecError);
  emit_and_expect_error(make_image_metadata_with_tensor_fields(std::vector<int64_t>(9U, 1),
                                                               /*ndim=*/9U,
                                                               &uint32_dtype,
                                                               &host_device),
                        valid_descriptor(kHoloMQSchemaTensorBufferBytes),
                        holoscan::ErrorCode::kCodecError);
  const auto malformed_reference_metadata = make_image_metadata_with_tensor_fields(
      {static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))},
      /*ndim=*/1U,
      &uint32_dtype,
      &host_device);
  auto malformed_reference = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(malformed_reference_metadata.data(),
                                 malformed_reference_metadata.size()),
      holoscan::PayloadDescriptor{
          .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
          .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
          .sequence = 96U,
          .byte_size = kHoloMQSchemaTensorBufferBytes,
          .buffer = holoscan::BufferHandlePayloadReference{}});
  ASSERT_FALSE(malformed_reference);
  EXPECT_EQ(malformed_reference.error().code, holoscan::ErrorCode::kCodecError);
  emit_and_expect_error(
      make_image_metadata_with_tensor_fields(
          {static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))},
          /*ndim=*/1U,
          &uint32_dtype,
          &cuda_device),
      valid_descriptor(kHoloMQSchemaTensorBufferBytes),
      holoscan::ErrorCode::kInvalidArgument);
  emit_and_expect_error(
      make_image_metadata_with_tensor_fields(
          {static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))},
          /*ndim=*/1U,
          &uint32_dtype,
          &host_device),
      valid_descriptor(kHoloMQSchemaTensorBufferBytes / 2U),
      holoscan::ErrorCode::kInvalidArgument);

  auto device_fixture = make_holomq_device_pool_fixture();
  ASSERT_NE(device_fixture, nullptr);
  auto device_buffer_or = device_fixture->pool.acquire();
  ASSERT_TRUE(device_buffer_or.has_value());
  auto device_handle = device_buffer_or->export_handle_unsynchronized();
  ASSERT_TRUE(device_handle.has_value());
  const holoscan::schema::DLDevice mismatched_cuda_device{
      holoscan::schema::DLDeviceType_kDLCUDA,
      holoscan::test::invalid_cuda_device_id(),
  };
  // Key assertion: imported-buffer validation compares the physical CUDA ordinal against the
  // schema descriptor instead of trusting independently frozen peer metadata.
  emit_and_expect_error(
      make_image_metadata_with_tensor_fields(
          {static_cast<int64_t>(kHoloMQSchemaTensorBufferBytes / sizeof(std::uint32_t))},
          /*ndim=*/1U,
          &uint32_dtype,
          &mismatched_cuda_device),
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
      descriptor_for_handle(kHoloMQSchemaTensorBufferBytes, *device_handle),
      holoscan::ErrorCode::kInvalidArgument);
}

// Verifies descriptor-backed drop-oldest delivery releases the oldest sidecar record and retains
// the newest configured number of materialized Tensors in FIFO order.
// REQ: CHAN-124
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaDropOldestRetainsNewestTensors) {
  //   tensor_schema_publisher --[kDropOldest, depth=2]--> subscriber
  //   (publishes tensors[0..2]; tensors[0] evicted; subscriber receives tensors[1] and [2])
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  std::array<std::shared_ptr<std::atomic<int>>, 3> destroyed_counters{
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
      std::make_shared<std::atomic<int>>(0),
  };
  std::array<void*, 3> data{};
  std::array<holoscan::Tensor, 3> tensors{
      make_counting_tensor(destroyed_counters[0], &data[0], 31U),
      make_counting_tensor(destroyed_counters[1], &data[1], 32U),
      make_counting_tensor(destroyed_counters[2], &data[2], 33U),
  };
  const auto metadata = make_image_metadata_bytes(tensors.front(), "drop_oldest_camera", 323100);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());

  auto options = make_options(holoscan::ChannelQueuePolicy::kDropOldest);
  options.subscriber_queue_depth = 2U;
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(
      publisher.value(), metadata_view, 1U, std::chrono::seconds(3), subscriber.value()));

  for (auto& tensor : tensors) {
    auto emitted = publisher->emit(metadata_view, std::move(tensor));
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  }

  EXPECT_TRUE(wait_until(
      [&destroyed_counters]() {
        return destroyed_counters[0]->load(std::memory_order_relaxed) == 1;
      },
      std::chrono::seconds(2)));
  EXPECT_EQ(destroyed_counters[1]->load(std::memory_order_relaxed), 0);
  EXPECT_EQ(destroyed_counters[2]->load(std::memory_order_relaxed), 0);

  for (std::size_t index = 1U; index < data.size(); ++index) {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    EXPECT_EQ(received->tensor.data(), data[index]);
    received->tensor = holoscan::Tensor{};
    EXPECT_EQ(destroyed_counters[index]->load(std::memory_order_relaxed), 1);
  }
  EXPECT_FALSE(subscriber->has_samples().value());
}

// Verifies the materializing receive path fails closed when descriptor-backed schema metadata
// carries a descriptor that does not resolve through this channel's TensorSidecarStore.
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaRejectsUnresolvableDescriptors) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const auto metadata = make_image_metadata_bytes(tensor, "bad_descriptor_camera", 423000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());
  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto inline_publisher =
      runtime->create_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(inline_publisher) << holoscan::runtime_error_message(inline_publisher.error());
  auto materializing_subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(materializing_subscriber)
      << holoscan::runtime_error_message(materializing_subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(inline_publisher.value(), std::chrono::seconds(3)));

  auto emit_and_expect_receive_error = [&](DescriptorBackedImagePayload payload,
                                           holoscan::ErrorCode expected_code) {
    auto emitted = inline_publisher->emit(payload);
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
    auto received = materializing_subscriber->receive();
    ASSERT_FALSE(received);
    EXPECT_EQ(received.error().code, expected_code);
  };
  auto emit_and_expect_publish_error = [&](DescriptorBackedImagePayload payload,
                                           holoscan::ErrorCode expected_code) {
    auto emitted = inline_publisher->emit(payload);
    ASSERT_FALSE(emitted);
    EXPECT_EQ(emitted.error().code, expected_code);
  };

  auto base_payload = DescriptorBackedImagePayload::copy_from(
      metadata_view, make_tensor_data_descriptor(tensor, 61U));
  ASSERT_TRUE(base_payload) << holoscan::runtime_error_message(base_payload.error());
  emit_and_expect_receive_error(base_payload.value(), holoscan::ErrorCode::kReceiveError);

  auto wrong_process = base_payload.value();
  ASSERT_NE(wrong_process.descriptor_at(0U), nullptr);
  wrong_process.descriptor_at(0U)->local.process_id = static_cast<std::uint64_t>(::getpid()) + 1U;
  emit_and_expect_receive_error(wrong_process, holoscan::ErrorCode::kReceiveError);

  auto empty_token = base_payload.value();
  ASSERT_NE(empty_token.descriptor_at(0U), nullptr);
  empty_token.descriptor_at(0U)->local.token = 0U;
  emit_and_expect_publish_error(empty_token, holoscan::ErrorCode::kCodecError);

  auto buffer_handle = base_payload.value();
  ASSERT_NE(buffer_handle.descriptor_at(0U), nullptr);
  *buffer_handle.descriptor_at(0U) = holoscan::PayloadDescriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = 62U,
      .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
      .buffer = holoscan::BufferHandlePayloadReference{}};
  emit_and_expect_publish_error(buffer_handle, holoscan::ErrorCode::kCodecError);
}

// Verifies the warmed-up descriptor-backed schema materialization path avoids host heap allocation
// across metadata payload construction, sidecar descriptor transport, and Tensor resolution.
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaHotPathDoesNotAllocateAfterWarmup) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* data = nullptr;
  const auto metadata_tensor = make_counting_tensor(destroyed_counter, &data, 55U);
  const auto metadata = make_image_metadata_bytes(metadata_tensor, "alloc_camera", 523000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(
      publisher.value(), metadata_view, 1U, std::chrono::seconds(3), subscriber.value()));

  auto warmup_destroyed = std::make_shared<std::atomic<int>>(0);
  void* warmup_data = nullptr;
  // Key setup: one warmup emit/receive primes all internal caches before the measured loop.
  auto warmup_emit =
      publisher->emit(metadata_view, make_counting_tensor(warmup_destroyed, &warmup_data, 66U));
  ASSERT_TRUE(warmup_emit) << holoscan::runtime_error_message(warmup_emit.error());
  auto warmup_receive = subscriber->receive();
  ASSERT_TRUE(warmup_receive) << holoscan::runtime_error_message(warmup_receive.error());

  std::array<std::shared_ptr<std::atomic<int>>, 32> destroyed_counters{};
  std::array<void*, 32> expected_data{};
  std::array<holoscan::Tensor, 32> tensors{};
  for (std::size_t i = 0U; i < tensors.size(); ++i) {
    destroyed_counters[i] = std::make_shared<std::atomic<int>>(0);
    tensors[i] = make_counting_tensor(
        destroyed_counters[i], &expected_data[i], static_cast<std::uint8_t>(i));
  }

  bool all_cycles_succeeded = true;
  std::size_t allocation_count = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t i = 0U; i < tensors.size(); ++i) {
      auto emitted = publisher->emit(metadata_view, std::move(tensors[i]));
      all_cycles_succeeded = all_cycles_succeeded && emitted.has_value() && emitted.value() == 1U;
      auto received = subscriber->receive();
      all_cycles_succeeded = all_cycles_succeeded && received.has_value() &&
                             received->tensor.data() == expected_data[i] &&
                             received->metadata.bytes().size() == metadata.size();
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  // Key assertion: zero allocations across 32 emit/receive cycles on the descriptor hot path.
  EXPECT_EQ(allocation_count, 0U);
}

// Verifies descriptor-backed schema materialization preserves stream-ordered device-pool reuse:
// a single device block is not returned to the pool until every resolved Tensor handle is released
// and every recorded consumer release stream has completed.
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaDeviceFanoutWaitsForReleaseStreams) {
  //   tensor_schema_publisher --[kBlocking, recipients=2]--> subscriber_a (fast_stream)
  //                                                      --> subscriber_b (slow_stream)
  //   Pool slot held until both release streams complete; slow_stream is the gate.
  holoscan::test::skip_without_cuda_device();

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking, 2U);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());

  auto pool =
      std::make_shared<holoscan::BlockMemoryPool>(holoscan::MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_tensor = make_device_tensor_for_schema(publisher.value(), pool);
  ASSERT_TRUE(maybe_tensor) << holoscan::runtime_error_message(maybe_tensor.error());
  auto tensor = std::move(maybe_tensor).value();
  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << holoscan::runtime_error_message(writable_data.error());
  void* const expected_data = *writable_data;

  const auto metadata = make_image_metadata_bytes(tensor, "device_fanout_camera", 623000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(publisher.value(),
                                                metadata_view,
                                                2U,
                                                std::chrono::seconds(3),
                                                subscriber_a.value(),
                                                subscriber_b.value()));

  auto emitted = publisher->emit(metadata_view, std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 2U);

  auto received_a = subscriber_a->receive();
  ASSERT_TRUE(received_a) << holoscan::runtime_error_message(received_a.error());
  auto sample_a = std::move(received_a).value();
  EXPECT_EQ(sample_a.tensor.data(), expected_data);
  auto received_b = subscriber_b->receive();
  ASSERT_TRUE(received_b) << holoscan::runtime_error_message(received_b.error());
  auto sample_b = std::move(received_b).value();
  EXPECT_EQ(sample_b.tensor.data(), expected_data);

  EXPECT_FALSE(pool->allocate(1, holoscan::MemoryStorageType::kDevice));

  holoscan::test::UniqueCudaStream fast_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(fast_stream), cudaSuccess);
  holoscan::test::UniqueCudaStream slow_stream;
  ASSERT_EQ(holoscan::test::create_non_blocking_cuda_stream(slow_stream), cudaSuccess);

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

  // Key setup: subscriber_a dealloc on fast_stream (already synchronized); subscriber_b
  //   dealloc on slow_stream (held by blocking host callback until explicitly released).
  EXPECT_TRUE(sample_a.tensor.set_deallocation_stream(fast_stream.get()));
  EXPECT_TRUE(sample_b.tensor.set_deallocation_stream(slow_stream.get()));
  sample_a.tensor = holoscan::Tensor{};
  sample_b.tensor = holoscan::Tensor{};

  // Key assertion: pool slot unavailable until slow_stream is released and synchronizes.
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

// Verifies a descriptor-backed schema receive preserves the Tensor's DLPack storage ownership: a
// DLPack export of the resolved device Tensor keeps the pool block checked out until the export
// deleter runs, even after the C++ Tensor handle is dropped.
TEST(ChannelSchemaPayloadTest, DescriptorBackedTensorSchemaDeviceDLPackExportRetainsBlock) {
  //   tensor_schema_publisher --[kBlocking]--> tensor_schema_subscriber
  //   (DLPack export keeps pool block checked out past C++ Tensor handle drop)
  holoscan::test::skip_without_cuda_device();

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  auto pool =
      std::make_shared<holoscan::BlockMemoryPool>(holoscan::MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_tensor = make_device_tensor_for_schema(publisher.value(), pool);
  ASSERT_TRUE(maybe_tensor) << holoscan::runtime_error_message(maybe_tensor.error());
  auto tensor = std::move(maybe_tensor).value();
  const void* const expected_data = tensor.data();
  ASSERT_NE(expected_data, nullptr);

  const auto metadata = make_image_metadata_bytes(tensor, "device_dlpack_camera", 723000);
  ASSERT_FALSE(metadata.empty());
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());
  ASSERT_TRUE(wait_for_tensor_schema_recipients(
      publisher.value(), metadata_view, 1U, std::chrono::seconds(3), subscriber.value()));

  auto emitted = publisher->emit(metadata_view, std::move(tensor));
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto sample = std::move(received).value();
  EXPECT_EQ(sample.tensor.data(), expected_data);

  // Key setup: DLPack export transfers storage ownership so the C++ Tensor can be reset
  //   while the pool block stays checked out through the DLManagedTensor deleter.
  auto exported_result = sample.tensor.to_dlpack();
  ASSERT_TRUE(exported_result) << holoscan::runtime_error_message(exported_result.error());
  DLManagedTensor* exported = exported_result.value();
  ASSERT_NE(exported->deleter, nullptr);

  sample.tensor = holoscan::Tensor{};
  EXPECT_FALSE(pool->allocate(1, holoscan::MemoryStorageType::kDevice));

  // Key assertion: pool slot reclaimed only after the DLPack deleter runs.
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
