/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/verifier.h>
#include <gtest/gtest.h>

#include "channel/channel_payload_limits.hpp"
#include "core/domain/tensor_access.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/schema/flatbuffer_schema_package.hpp"
#include "holoscan/schema/image_metadata.hpp"
#include "holoscan/schema/tensor_materialization.hpp"

#include "header_generated.h"
#include "image_generated.h"
#include "imu_generated.h"
#include "string_generated.h"
#include "tensor_generated.h"
#include "time_generated.h"

namespace holoscan {

struct IndexedDescriptorSchemaNative {
  std::array<std::shared_ptr<Tensor>, 3U> tensors;
};

struct InvalidMultiTensorSingularTraitsNative {
  std::shared_ptr<Tensor> tensor;
};

struct InvalidDuplicateTensorFieldIdsNative {};
struct InvalidDescendingTensorFieldIdsNative {};

template <>
struct DescriptorTensorSchemaNativeTraits<IndexedDescriptorSchemaNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 3U;
  static constexpr std::array<std::uint16_t, tensor_count> tensor_field_ids{2U, 5U, 9U};

  [[nodiscard]] static const Tensor* tensor_at(const IndexedDescriptorSchemaNative& value,
                                               const std::size_t index) noexcept {
    return index < tensor_count ? value.tensors[index].get() : nullptr;
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<InvalidMultiTensorSingularTraitsNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 3U;

  [[nodiscard]] static const Tensor* tensor(
      const InvalidMultiTensorSingularTraitsNative& value) noexcept {
    return value.tensor.get();
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<InvalidDuplicateTensorFieldIdsNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, tensor_count> tensor_field_ids{2U, 2U};
  [[nodiscard]] static const Tensor* tensor_at(const InvalidDuplicateTensorFieldIdsNative&,
                                               std::size_t) noexcept {
    return nullptr;
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<InvalidDescendingTensorFieldIdsNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, tensor_count> tensor_field_ids{9U, 2U};
  [[nodiscard]] static const Tensor* tensor_at(const InvalidDescendingTensorFieldIdsNative&,
                                               std::size_t) noexcept {
    return nullptr;
  }
};

// Smoke test: flatc produced a usable string_generated.h from public/schemas/string.fbs, and a
// String message round-trips through build -> verify -> read. This exercises the whole FlatBuffers
// codegen + runtime-link path end to end.
TEST(Schema, StringSchemaRoundTrips) {
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(schema::CreateStringDirect(builder, "hello flatbuffers"));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyStringBuffer(verifier));

  const schema::String* parsed = schema::GetString(builder.GetBufferPointer());
  ASSERT_NE(parsed, nullptr);
  ASSERT_NE(parsed->data(), nullptr);
  EXPECT_EQ(parsed->data()->str(), "hello flatbuffers");
}

TEST(Schema, TimeSchemaRoundTrips) {
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(schema::CreateTime(builder, 123456789, 123456790));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyTimeBuffer(verifier));

  const schema::Time* time = schema::GetTime(builder.GetBufferPointer());
  ASSERT_NE(time, nullptr);
  EXPECT_EQ(time->time_ns(), 123456789);
  EXPECT_EQ(time->acqtime_ns(), 123456790);
}

TEST(Schema, HeaderSchemaRoundTrips) {
  flatbuffers::FlatBufferBuilder builder;
  const auto stamp = schema::CreateTime(builder, 100, 200);
  const auto frame_id = builder.CreateString("camera_0");
  builder.Finish(schema::CreateHeader(builder, stamp, frame_id));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyHeaderBuffer(verifier));

  const schema::Header* header = schema::GetHeader(builder.GetBufferPointer());
  ASSERT_NE(header, nullptr);
  ASSERT_NE(header->stamp(), nullptr);
  EXPECT_EQ(header->stamp()->time_ns(), 100);
  ASSERT_NE(header->frame_id(), nullptr);
  EXPECT_EQ(header->frame_id()->string_view(), std::string_view("camera_0"));
}

TEST(Schema, ImuSchemaRoundTrips) {
  flatbuffers::FlatBufferBuilder builder;
  const auto stamp = schema::CreateTime(builder, 1, 2);
  const auto frame_id = builder.CreateString("imu_link");
  const auto header = schema::CreateHeader(builder, stamp, frame_id);
  const schema::Quaternionf orientation{1.0F, 0.0F, 0.0F, 0.0F};
  const schema::Vector3f angular_velocity{0.1F, 0.2F, 0.3F};
  const schema::Vector3f linear_acceleration{9.8F, 0.0F, -0.1F};
  builder.Finish(
      schema::CreateImu(builder, header, &orientation, &angular_velocity, &linear_acceleration));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyImuBuffer(verifier));

  const schema::Imu* imu = schema::GetImu(builder.GetBufferPointer());
  ASSERT_NE(imu, nullptr);
  ASSERT_NE(imu->header(), nullptr);
  ASSERT_NE(imu->header()->frame_id(), nullptr);
  EXPECT_EQ(imu->header()->frame_id()->string_view(), std::string_view("imu_link"));
  ASSERT_NE(imu->orientation(), nullptr);
  EXPECT_FLOAT_EQ(imu->orientation()->w(), 1.0F);
  ASSERT_NE(imu->angular_velocity(), nullptr);
  EXPECT_FLOAT_EQ(imu->angular_velocity()->y(), 0.2F);
  ASSERT_NE(imu->linear_acceleration(), nullptr);
  EXPECT_FLOAT_EQ(imu->linear_acceleration()->x(), 9.8F);
}

// REQ: OPERATOR-035
TEST(Schema, ImageDescriptorPortOpsPreserveLegacyTableShape) {
  using Payload = schema::DescriptorFlatBufferPayload<
      schema::Image,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::Image>>;
  struct Envelope {
    SampleMetadata metadata{};
    Payload payload{};
  };

  const detail::RuntimePortTypeOps* const ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(ops->payload_size, sizeof(schema::ImageT));
  EXPECT_EQ(ops->payload_alignment, alignof(schema::ImageT));
  EXPECT_EQ(ops->wire_capacity, sizeof(Envelope));
  EXPECT_EQ(ops->sample_size, sizeof(Sample<schema::ImageT>));
  EXPECT_EQ(ops->sample_alignment, alignof(Sample<schema::ImageT>));
  EXPECT_EQ(ops->predicate_view_size, sizeof(PredicateInputView<schema::ImageT>));
  EXPECT_EQ(ops->predicate_view_alignment, alignof(PredicateInputView<schema::ImageT>));
  EXPECT_EQ(ops->codec_family, detail::RuntimePayloadCodecFamily::kDescriptorTensorSchema);
  EXPECT_EQ(ops->payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_FALSE(ops->inline_pod);
  EXPECT_FALSE(ops->generic_loan_supported);
  EXPECT_TRUE(ops->persistent_sample_storage);
  EXPECT_EQ(ops->generic_loan_slot_size_bytes, 0U);
  EXPECT_EQ(ops->descriptor_offset,
            offsetof(Envelope, payload) + offsetof(Payload, descriptor) +
                offsetof(typename Payload::descriptor_carriage_type, descriptors));

  const std::array<bool, 28U> populated_slots{
      ops->encode_payload != nullptr,
      ops->construct_sample != nullptr,
      ops->destroy_sample != nullptr,
      ops->prepare_sample_storage != nullptr,
      ops->reset_sample_storage != nullptr,
      ops->sample_payload != nullptr,
      ops->sample_metadata != nullptr,
      ops->descriptor_tensor != nullptr,
      ops->descriptor_tensor_count != nullptr,
      ops->descriptor_tensor_at != nullptr,
      ops->encode_descriptor_tensor_sample != nullptr,
      ops->encode_descriptor_tensor_loan != nullptr,
      ops->descriptor_tensor_sample_metadata != nullptr,
      ops->descriptor_tensor_view_metadata != nullptr,
      ops->descriptor_tensor_view_metadata_at != nullptr,
      ops->assign_descriptor_tensor_sample != nullptr,
      ops->assign_descriptor_tensors_sample != nullptr,
      ops->construct_predicate_view != nullptr,
      ops->destroy_predicate_view != nullptr,
      ops->construct_value != nullptr,
      ops->destroy_value != nullptr,
      ops->bind_input != nullptr,
      ops->unbind_input != nullptr,
      ops->bind_output != nullptr,
      ops->unbind_output != nullptr,
      ops->publish_tensor_loan != nullptr,
      ops->bind_tensor_output != nullptr,
      ops->unbind_tensor_output != nullptr,
  };
  constexpr std::array<bool, 28U> kExpectedPopulatedSlots{
      false, false, true, true, true, true,  true,  true, true, true, true, true, true, true,
      true,  true,  true, true, true, false, false, true, true, true, true, true, true, true,
  };
  EXPECT_EQ(populated_slots, kExpectedPopulatedSlots);

  schema::ImageT image{};
  image.data = std::make_shared<Tensor>();
  EXPECT_EQ(ops->descriptor_tensor_count(&image), 1U);
  EXPECT_EQ(ops->descriptor_tensor(&image), ops->descriptor_tensor_at(&image, 0U));
  EXPECT_EQ(ops->descriptor_tensor_at(&image, 1U), nullptr);
}

// The second SDK descriptor-backed schema must be admitted through the generic factory rather than
// an Image-only branch.
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(Schema, TensorPacketUsesGenericDescriptorTensorOperations) {
  static_assert(detail::HasDescriptorTensorSchemaNativeTraits<schema::TensorPacketT>);
  static_assert(!detail::HasGeneratedSchemaNativeTraits<schema::TensorPacketT>);
  static_assert(schema::DescriptorFlatBufferTensorViewTraits<schema::TensorPacket>::tensor_count ==
                1U);

  // Key setup: every SDK package whose binary schema embeds tensor.fbs uses the current artifact.
  const auto image_package_status = validate_schema_package(schema_package<schema::Image>());
  ASSERT_TRUE(image_package_status) << runtime_error_message(image_package_status.error());
  const auto tensor_package_status = validate_schema_package(schema_package<schema::Tensor>());
  ASSERT_TRUE(tensor_package_status) << runtime_error_message(tensor_package_status.error());
  const auto package_status = validate_schema_package(schema_package<schema::TensorPacket>());
  // Key assertion: the root and transitive Tensor artifact digests match their generated BFBS.
  ASSERT_TRUE(package_status) << runtime_error_message(package_status.error());

  const detail::RuntimePortTypeOps* const ops =
      detail::runtime_port_type_ops<schema::TensorPacketT>();
  ASSERT_NE(ops, nullptr);
  EXPECT_EQ(ops->payload_size, sizeof(schema::TensorPacketT));
  EXPECT_EQ(ops->payload_alignment, alignof(schema::TensorPacketT));
  EXPECT_EQ(ops->codec_family, detail::RuntimePayloadCodecFamily::kDescriptorTensorSchema);
  EXPECT_EQ(ops->payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_TRUE(ops->persistent_sample_storage);
  const std::array<bool, 28U> populated_slots{
      ops->encode_payload != nullptr,
      ops->construct_sample != nullptr,
      ops->destroy_sample != nullptr,
      ops->prepare_sample_storage != nullptr,
      ops->reset_sample_storage != nullptr,
      ops->sample_payload != nullptr,
      ops->sample_metadata != nullptr,
      ops->descriptor_tensor != nullptr,
      ops->descriptor_tensor_count != nullptr,
      ops->descriptor_tensor_at != nullptr,
      ops->encode_descriptor_tensor_sample != nullptr,
      ops->encode_descriptor_tensor_loan != nullptr,
      ops->descriptor_tensor_sample_metadata != nullptr,
      ops->descriptor_tensor_view_metadata != nullptr,
      ops->descriptor_tensor_view_metadata_at != nullptr,
      ops->assign_descriptor_tensor_sample != nullptr,
      ops->assign_descriptor_tensors_sample != nullptr,
      ops->construct_predicate_view != nullptr,
      ops->destroy_predicate_view != nullptr,
      ops->construct_value != nullptr,
      ops->destroy_value != nullptr,
      ops->bind_input != nullptr,
      ops->unbind_input != nullptr,
      ops->bind_output != nullptr,
      ops->unbind_output != nullptr,
      ops->publish_tensor_loan != nullptr,
      ops->bind_tensor_output != nullptr,
      ops->unbind_tensor_output != nullptr,
  };
  constexpr std::array<bool, 28U> kExpectedPopulatedSlots{
      false, false, true, true, true, true,  true,  true, true, true, true, true, true, true,
      true,  true,  true, true, true, false, false, true, true, true, true, true, true, true,
  };
  EXPECT_EQ(populated_slots, kExpectedPopulatedSlots);
}

// REQ: OPERATOR-035
TEST(Schema, IndexedDescriptorTraitsExposeThreeTensorTypeOps) {
  // Key setup: a three-field native trait exposes only indexed access while a malformed
  //   three-field trait deliberately retains only the legacy singular accessor.
  // Key assertion: indexed count/read operations preserve canonical order, transactional
  //   assignment receives all three handles, and the malformed singular trait is not admitted.
  static_assert(detail::HasDescriptorTensorSchemaNativeTraits<IndexedDescriptorSchemaNative>);
  static_assert(
      !detail::HasDescriptorTensorSchemaNativeTraits<InvalidMultiTensorSingularTraitsNative>);
  static_assert(
      !detail::HasDescriptorTensorSchemaNativeTraits<InvalidDuplicateTensorFieldIdsNative>);
  static_assert(
      !detail::HasDescriptorTensorSchemaNativeTraits<InvalidDescendingTensorFieldIdsNative>);
  static_assert(detail::descriptor_tensor_schema_tensor_count_v<IndexedDescriptorSchemaNative> ==
                3U);

  const detail::RuntimePortTypeOps ops{
      .descriptor_tensor_count = [](const void* payload) noexcept -> std::uint32_t {
        return payload == nullptr ? 0U : 3U;
      },
      .descriptor_tensor_at = [](const void* payload,
                                 const std::uint32_t index) noexcept -> const Tensor* {
        return payload == nullptr
                   ? nullptr
                   : detail::descriptor_tensor_schema_tensor_at(
                         *static_cast<const IndexedDescriptorSchemaNative*>(payload), index);
      },
      .assign_descriptor_tensors_sample = [](void* storage,
                                             std::span<const std::byte>,
                                             std::span<Tensor>
                                                 tensors,
                                             SampleMetadata) noexcept -> expected<void, Error> {
        if (storage == nullptr || tensors.size() != 3U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& value = *static_cast<IndexedDescriptorSchemaNative*>(storage);
        for (std::size_t index = 0U; index < tensors.size(); ++index) {
          value.tensors[index] = std::make_shared<Tensor>(std::move(tensors[index]));
        }
        return {};
      },
  };

  IndexedDescriptorSchemaNative source{};
  source.tensors = {
      std::make_shared<Tensor>(),
      std::make_shared<Tensor>(),
      std::make_shared<Tensor>(),
  };
  EXPECT_EQ(ops.descriptor_tensor_count(&source), 3U);
  EXPECT_EQ(ops.descriptor_tensor, nullptr);
  for (std::uint32_t index = 0U; index < source.tensors.size(); ++index) {
    EXPECT_EQ(ops.descriptor_tensor_at(&source, index), source.tensors[index].get());
  }
  EXPECT_EQ(ops.descriptor_tensor_at(&source, 3U), nullptr);

  IndexedDescriptorSchemaNative destination{};
  std::array<Tensor, 3U> assigned_tensors{};
  auto assigned = ops.assign_descriptor_tensors_sample(&destination, {}, assigned_tensors, {});
  ASSERT_TRUE(assigned) << runtime_error_message(assigned.error());
  for (const auto& tensor : destination.tensors) {
    EXPECT_NE(tensor, nullptr);
  }
}

// REQ: CHAN-124
// REQ: CHAN-125
// REQ: CHAN-120
// REQ: OPERATOR-035
template <typename Payload>
void expect_version_three_descriptor_layout() {
  static_assert(std::is_standard_layout_v<Payload>);
  EXPECT_EQ(sizeof(Payload), 880U);
  EXPECT_EQ(offsetof(Payload, envelope_abi_version), 0U);
  EXPECT_EQ(offsetof(Payload, envelope_header_size), 4U);
  EXPECT_EQ(offsetof(Payload, schema_id), 8U);
  EXPECT_EQ(offsetof(Payload, schema_version), 24U);
  EXPECT_EQ(offsetof(Payload, metadata_size), 28U);
  EXPECT_EQ(offsetof(Payload, endpoint_generation), 32U);
  EXPECT_EQ(offsetof(Payload, descriptor), 40U);
  EXPECT_EQ(offsetof(Payload, metadata), 368U);
}

TEST(Schema, SingularDescriptorSchemasUseVersionThreeCarriage) {
  // Key setup: instantiate both SDK descriptor-backed schemas with one Tensor field.
  // Key assertion: both use the same explicit version-3 count-and-mask carriage as N-field schemas.
  using ImagePayload = schema::DescriptorFlatBufferPayload<
      schema::Image,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::Image>>;
  using TensorPacketPayload = schema::DescriptorFlatBufferPayload<
      schema::TensorPacket,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>>;

  EXPECT_EQ(kSchemaEnvelopeABIVersion, 2U);
  EXPECT_EQ(kDescriptorTensorSchemaEnvelopeABIVersion, 3U);
  EXPECT_EQ(kSchemaEnvelopeHeaderSize, 40U);
  expect_version_three_descriptor_layout<ImagePayload>();
  expect_version_three_descriptor_layout<TensorPacketPayload>();
  EXPECT_EQ(ImagePayload{}.descriptor_count(), 1U);
  EXPECT_EQ(ImagePayload{}.tensor_presence_mask(), 1U);
  EXPECT_EQ(ImagePayload{}.tensor_descriptor_mask(), 1U);
}

// REQ: CHAN-125
TEST(Schema, DescriptorTensorCarriageStructurallySupportsSixtyFourFields) {
  // Key setup: instantiate the public envelope at its exact structural field-count ceiling.
  using MaximumPayload = schema::DescriptorFlatBufferPayload<schema::Image, 1U, 64U>;

  // Key assertion: all 64 descriptor positions and mask bits are representable; transport-specific
  // byte limits are enforced separately during physical compilation.
  const MaximumPayload payload{};
  EXPECT_EQ(payload.descriptor_count(), 64U);
  EXPECT_EQ(payload.tensor_presence_mask(), std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(payload.tensor_descriptor_mask(), std::numeric_limits<std::uint64_t>::max());
  EXPECT_NE(payload.descriptor_at(63U), nullptr);
  EXPECT_EQ(payload.descriptor_at(64U), nullptr);
  EXPECT_LE(sizeof(MaximumPayload), channel_internal::kMaximumInlinePayloadBytes);
}

// REQ: CHAN-124
// REQ: CHAN-125
TEST(Schema, DescriptorTensorEnvelopeUsesUniformVersionThreeArrayAndExplicitMasks) {
  // Key setup: encode the same metadata into one- and three-Tensor descriptor envelopes, then
  //   deliberately stamp the superseded early-access ABI version.
  // Key assertion: both envelopes carry explicit count-and-mask state under version 3 and reject
  //   version 2 before typed access.
  constexpr std::size_t kMetadataCapacity = 512U;
  using SingularPayload = schema::DescriptorFlatBufferPayload<schema::Image, kMetadataCapacity>;
  using MultiPayload = schema::DescriptorFlatBufferPayload<schema::Image, kMetadataCapacity, 3U>;

  static_assert(std::is_standard_layout_v<MultiPayload>);
  static_assert(std::is_trivially_copyable_v<MultiPayload>);
  static_assert(sizeof(SingularPayload) == 880U);
  static_assert(offsetof(SingularPayload, metadata) == 368U);
  static_assert(offsetof(MultiPayload, envelope_abi_version) == 0U);
  static_assert(offsetof(MultiPayload, descriptor) == kSchemaEnvelopeHeaderSize);

  flatbuffers::FlatBufferBuilder builder;
  schema::FinishImageBuffer(builder,
                            schema::CreateImage(builder,
                                                /*header=*/0,
                                                /*height=*/2,
                                                /*width=*/3,
                                                schema::ImageEncoding_MONO8,
                                                /*data=*/0));
  const auto make_descriptor = [](std::uint64_t token) {
    return PayloadDescriptor{
        .payload_kind = ChannelPayloadKind::kLocalBufferDescriptor,
        .schema_id = schema::flatbuffer_schema_identity_v<schema::Image>.id,
        .local =
            LocalPayloadReference{
                .process_id = 1U,
                .token = token,
                .generation = 1U,
                .expected_recipient_count = 1U,
            },
    };
  };
  const std::array descriptors{
      make_descriptor(1U),
      make_descriptor(2U),
      make_descriptor(3U),
  };
  auto multi = MultiPayload::copy_from_builder(builder, descriptors, 9U);
  ASSERT_TRUE(multi) << runtime_error_message(multi.error());
  EXPECT_EQ(multi->envelope_abi_version, kDescriptorTensorSchemaEnvelopeABIVersion);
  EXPECT_EQ(multi->envelope_header_size, kSchemaEnvelopeHeaderSize);
  EXPECT_EQ(multi->descriptor_count(), 3U);
  EXPECT_EQ(multi->tensor_presence_mask(), 0b111U);
  EXPECT_EQ(multi->tensor_descriptor_mask(), 0b111U);
  for (std::size_t index = 0U; index < descriptors.size(); ++index) {
    ASSERT_NE(multi->descriptor_at(index), nullptr);
    EXPECT_EQ(multi->descriptor_at(index)->local.token, index + 1U);
  }
  EXPECT_EQ(multi->descriptor_at(descriptors.size()), nullptr);
  EXPECT_TRUE(multi->validate());

  auto invalid_count = *multi;
  invalid_count.descriptor.tensor_count = 65U;
  EXPECT_EQ(invalid_count.descriptor_count(), 3U);
  EXPECT_FALSE(invalid_count.validate());
  auto invalid_reserved = *multi;
  invalid_reserved.descriptor.reserved = 1U;
  EXPECT_FALSE(invalid_reserved.validate());
  auto invalid_presence = *multi;
  invalid_presence.descriptor.presence_mask = 0b011U;
  EXPECT_FALSE(invalid_presence.validate());
  auto invalid_carriage = *multi;
  invalid_carriage.descriptor.descriptor_mask = 0b101U;
  EXPECT_FALSE(invalid_carriage.validate());

  const auto& package = schema_package<schema::Image>();
  EXPECT_TRUE(multi->verified_root(package, 9U));
  auto wrong_multi_version = *multi;
  wrong_multi_version.envelope_abi_version = kSchemaEnvelopeABIVersion;
  auto rejected_multi = wrong_multi_version.verified_root(package, 9U);
  ASSERT_FALSE(rejected_multi);
  EXPECT_EQ(rejected_multi.error().code, ErrorCode::kProtocolError);

  auto singular = SingularPayload::copy_from_builder(builder, descriptors.front(), 9U);
  ASSERT_TRUE(singular) << runtime_error_message(singular.error());
  EXPECT_EQ(singular->envelope_abi_version, kDescriptorTensorSchemaEnvelopeABIVersion);
  EXPECT_EQ(singular->descriptor_count(), 1U);
  EXPECT_EQ(singular->tensor_presence_mask(), 1U);
  EXPECT_EQ(singular->tensor_descriptor_mask(), 1U);
  EXPECT_TRUE(singular->verified_root(package, 9U));
  singular->envelope_abi_version = kSchemaEnvelopeABIVersion;
  auto rejected_singular = singular->verified_root(package, 9U);
  ASSERT_FALSE(rejected_singular);
  EXPECT_EQ(rejected_singular.error().code, ErrorCode::kProtocolError);
}

namespace {

// A 2x3 contiguous float32 CPU tensor (24 bytes) backed by static storage.
Tensor make_cpu_tensor_2x3() {
  static std::vector<float> data{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {2, 3};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;
  return Tensor(std::move(context));
}

// A 3x2 float32 CPU tensor with explicit row strides {4, 1} backed by static storage.
Tensor make_strided_cpu_tensor_3x2() {
  static std::vector<float> data{1.0F, 2.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F, 5.0F, 6.0F};
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {3, 2};
  context->dl_strides = {4, 1};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0;
  return Tensor(std::move(context));
}

// Packs a 2x3 MONO32F Image wrapping the given Tensor via the flatc object API.
std::vector<std::uint8_t> pack_image_object_api(const Tensor& tensor) {
  auto image = std::make_shared<schema::ImageT>();
  image->height = 2;
  image->width = 3;
  image->encoding = schema::ImageEncoding_MONO32F;
  image->data = std::make_shared<Tensor>(tensor);

  flatbuffers::FlatBufferBuilder builder;
  schema::FinishImageBuffer(builder, schema::Image::Pack(builder, image.get()));
  const auto* data = builder.GetBufferPointer();
  return {data, data + builder.GetSize()};
}

struct FailingTensorPacketPrepareCodec {
  [[nodiscard]] expected<void, Error> prepare(schema::TensorPacketT&) const {
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  [[nodiscard]] expected<flatbuffers::Offset<schema::TensorPacket>, Error> encode_root(
      flatbuffers::FlatBufferBuilder&, const schema::TensorPacketT&,
      std::span<const flatbuffers::Offset<schema::Tensor>>) const {
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  [[nodiscard]] expected<void, Error> assign_root(schema::TensorPacketT&,
                                                  const schema::TensorPacket&) const {
    return {};
  }
};

struct MissingTensorPacketCodecMethods {};

static_assert(schema::detail::StandardDescriptorTensorCodec<
              FailingTensorPacketPrepareCodec, schema::TensorPacketT, schema::TensorPacket>);
static_assert(!schema::detail::StandardDescriptorTensorCodec<
              MissingTensorPacketCodecMethods, schema::TensorPacketT, schema::TensorPacket>);
static_assert(schema::detail::StandardDescriptorTensorMember<schema::TensorPacketT,
                                                             &schema::TensorPacketT::data>);
static_assert(!schema::detail::StandardDescriptorTensorMember<schema::TensorPacketT,
                                                              &schema::TensorPacketT::source>);

}  // namespace

// The standard adapter owns persistent storage, bounded Tensor metadata encoding, authoritative
// Tensor validation, handle assignment, and SampleMetadata bookkeeping.
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(Schema, StandardDescriptorTensorAdapterOwnsCommonLifecycle) {
  using Native = schema::TensorPacketT;
  using Payload = schema::DescriptorFlatBufferPayload<
      schema::TensorPacket,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>>;
  struct Envelope {
    SampleMetadata metadata{};
    Payload payload{};
  };

  const detail::RuntimePortTypeOps* const ops = detail::runtime_port_type_ops<Native>();
  ASSERT_NE(ops, nullptr);

  Native source{};
  source.sequence = 42U;
  source.source = "third-party-camera";
  source.scale = 2.5;
  source.data = std::make_shared<Tensor>(make_cpu_tensor_2x3());
  const SampleMetadata metadata{.sequence = 19U, .byte_size = 24U};
  // Key setup: encode with a non-initial endpoint attachment generation.
  constexpr std::uint64_t kEndpointGeneration = 7U;
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> wire{};
  wire.fill(std::byte{0xA5});
  auto encoded = ops->encode_descriptor_tensor_sample(&source, metadata, kEndpointGeneration, wire);
  ASSERT_TRUE(encoded) << runtime_error_message(encoded.error());
  EXPECT_LT(*encoded, wire.size());
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> repeated_wire{};
  repeated_wire.fill(std::byte{0x5A});
  auto repeated =
      ops->encode_descriptor_tensor_sample(&source, metadata, kEndpointGeneration, repeated_wire);
  ASSERT_TRUE(repeated) << runtime_error_message(repeated.error());
  ASSERT_EQ(*repeated, *encoded);
  EXPECT_TRUE(std::ranges::equal(std::span<const std::byte>{wire}.first(*encoded),
                                 std::span<const std::byte>{repeated_wire}.first(*repeated)));
  // Key assertion: the envelope retains the exact endpoint attachment generation.
  Envelope encoded_envelope{};
  std::memcpy(&encoded_envelope, wire.data(), sizeof(encoded_envelope));
  EXPECT_EQ(encoded_envelope.payload.endpoint_generation, kEndpointGeneration);
  PayloadDescriptor* const descriptor = encoded_envelope.payload.descriptor_at(0U);
  ASSERT_NE(descriptor, nullptr);
  descriptor->payload_kind = ChannelPayloadKind::kLocalBufferDescriptor;
  descriptor->schema_id = schema_identity<Native>().id;
  descriptor->byte_size = metadata.byte_size;
  descriptor->local.token = 1U;
  descriptor->local.generation = 1U;
  descriptor->local.expected_recipient_count = 1U;
  std::memcpy(wire.data(), &encoded_envelope, sizeof(encoded_envelope));

  alignas(Sample<Native>) std::array<std::byte, sizeof(Sample<Native>)> storage{};
  auto prepared = ops->prepare_sample_storage(storage.data());
  ASSERT_TRUE(prepared) << runtime_error_message(prepared.error());
  auto* sample = reinterpret_cast<Sample<Native>*>(storage.data());
  ASSERT_NE(sample->data.data, nullptr);
  Tensor* const retained_tensor_holder = sample->data.data.get();

  const auto encoded_prefix = std::span<const std::byte>{wire}.first(*encoded);
  auto decoded_metadata =
      ops->descriptor_tensor_sample_metadata(encoded_prefix, 0U, kEndpointGeneration);
  ASSERT_TRUE(decoded_metadata) << runtime_error_message(decoded_metadata.error());
  auto tensor_metadata = ops->descriptor_tensor_view_metadata(encoded_prefix, kEndpointGeneration);
  ASSERT_TRUE(tensor_metadata) << runtime_error_message(tensor_metadata.error());
  EXPECT_EQ(tensor_metadata->rank, 2U);
  EXPECT_EQ(tensor_metadata->shape[0], 2);
  EXPECT_EQ(tensor_metadata->shape[1], 3);
  EXPECT_EQ(tensor_metadata->dtype.code, kDLFloat);
  auto truncated = ops->descriptor_tensor_sample_metadata(
      encoded_prefix.first(encoded_prefix.size() - 1U), 0U, kEndpointGeneration);
  ASSERT_FALSE(truncated);
  EXPECT_EQ(truncated.error().code, ErrorCode::kProtocolError);
  Envelope malformed_envelope = encoded_envelope;
  malformed_envelope.payload.metadata[0] = std::byte{0xFF};
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> malformed_wire{};
  std::memcpy(malformed_wire.data(), &malformed_envelope, sizeof(malformed_envelope));
  auto malformed_metadata =
      ops->descriptor_tensor_view_metadata(malformed_wire, kEndpointGeneration);
  ASSERT_FALSE(malformed_metadata);
  EXPECT_EQ(malformed_metadata.error().code, ErrorCode::kProtocolError);
  auto invalid_carriage_mask =
      ops->descriptor_tensor_sample_metadata(wire, 0b10U, kEndpointGeneration);
  ASSERT_FALSE(invalid_carriage_mask);
  EXPECT_EQ(invalid_carriage_mask.error().code, ErrorCode::kProtocolError);
  // Key assertion: decode rejects an envelope from a stale endpoint attachment.
  auto stale_metadata = ops->descriptor_tensor_sample_metadata(wire, 0U, kEndpointGeneration + 1U);
  ASSERT_FALSE(stale_metadata);
  EXPECT_EQ(stale_metadata.error().code, ErrorCode::kProtocolError);
  auto assigned =
      ops->assign_descriptor_tensor_sample(storage.data(), wire, make_cpu_tensor_2x3(), metadata);
  ASSERT_TRUE(assigned) << runtime_error_message(assigned.error());
  EXPECT_EQ(sample->data.sequence, 42U);
  EXPECT_EQ(sample->data.source, "third-party-camera");
  EXPECT_DOUBLE_EQ(sample->data.scale, 2.5);
  EXPECT_EQ(sample->data.data.get(), retained_tensor_holder);
  EXPECT_EQ(sample->data.data->shape(), (std::vector<std::int64_t>{2, 3}));
  EXPECT_EQ(sample->metadata.sequence, 19U);

  ops->reset_sample_storage(storage.data());
  EXPECT_EQ(sample->data.data.get(), retained_tensor_holder);
  EXPECT_EQ(sample->data.sequence, 0U);
  EXPECT_TRUE(sample->data.source.empty());
  EXPECT_DOUBLE_EQ(sample->data.scale, 1.0);
  EXPECT_EQ(sample->metadata.sequence, 0U);
  ops->destroy_sample(storage.data());
}

TEST(Schema, StandardDescriptorTensorAdapterRejectsMismatchBeforeAssignment) {
  using Native = schema::TensorPacketT;
  using Payload = schema::DescriptorFlatBufferPayload<
      schema::TensorPacket,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>>;
  struct Envelope {
    SampleMetadata metadata{};
    Payload payload{};
  };

  const detail::RuntimePortTypeOps* const ops = detail::runtime_port_type_ops<Native>();
  ASSERT_NE(ops, nullptr);
  Native source{};
  source.sequence = 99U;
  source.source = "must-not-commit";
  source.data = std::make_shared<Tensor>(make_cpu_tensor_2x3());
  const SampleMetadata metadata{.sequence = 3U};
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> wire{};
  ASSERT_TRUE(ops->encode_descriptor_tensor_sample(&source, metadata, 1U, wire));

  alignas(Sample<Native>) std::array<std::byte, sizeof(Sample<Native>)> storage{};
  ASSERT_TRUE(ops->prepare_sample_storage(storage.data()));
  auto* sample = reinterpret_cast<Sample<Native>*>(storage.data());
  auto assigned = ops->assign_descriptor_tensor_sample(
      storage.data(), wire, make_strided_cpu_tensor_3x2(), metadata);
  ASSERT_FALSE(assigned);
  EXPECT_EQ(assigned.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(sample->data.sequence, 0U);
  EXPECT_TRUE(sample->data.source.empty());
  EXPECT_EQ(sample->metadata.sequence, 0U);
  ops->destroy_sample(storage.data());
}

TEST(Schema, StandardDescriptorTensorAdapterBoundsEncodingAndForwardsPrepareFailure) {
  using Native = schema::TensorPacketT;
  constexpr std::size_t kMetadataCapacity =
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>;
  using Payload = schema::DescriptorFlatBufferPayload<schema::TensorPacket, kMetadataCapacity>;
  struct Envelope {
    SampleMetadata metadata{};
    Payload payload{};
  };

  const detail::RuntimePortTypeOps* const ops = detail::runtime_port_type_ops<Native>();
  ASSERT_NE(ops, nullptr);
  Native oversized{};
  oversized.source.assign(kMetadataCapacity * 2U, 'x');
  oversized.data = std::make_shared<Tensor>(make_cpu_tensor_2x3());
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> wire{};
  auto encoded = ops->encode_descriptor_tensor_sample(&oversized, {}, 1U, wire);
  ASSERT_FALSE(encoded);
  EXPECT_EQ(encoded.error().code, ErrorCode::kResourceExhausted);

  const auto failing_ops =
      schema::make_standard_descriptor_tensor_port_type_ops<Native,
                                                            schema::TensorPacket,
                                                            kMetadataCapacity,
                                                            &Native::data>(
          FailingTensorPacketPrepareCodec{});
  alignas(Sample<Native>) std::array<std::byte, sizeof(Sample<Native>)> storage{};
  auto prepared = failing_ops.prepare_sample_storage(storage.data());
  ASSERT_FALSE(prepared);
  EXPECT_EQ(prepared.error().code, ErrorCode::kFailure);
}

// By default schema::Tensor::Pack serializes only metadata (shape/dtype/device), not the bytes.
TEST(Schema, TensorPacksMetadataOnlyByDefault) {
  const Tensor tensor = make_cpu_tensor_2x3();
  flatbuffers::FlatBufferBuilder builder;
  schema::FinishTensorBuffer(builder, schema::Tensor::Pack(builder, &tensor));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyTensorBuffer(verifier));

  const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(packed, nullptr);
  EXPECT_EQ(packed->ndim(), 2U);
  ASSERT_NE(packed->shape(), nullptr);
  EXPECT_EQ(packed->shape()->size(), 2U);
  EXPECT_EQ(packed->shape()->Get(0), 2);
  EXPECT_EQ(packed->shape()->Get(1), 3);
  ASSERT_NE(packed->dtype(), nullptr);
  EXPECT_EQ(packed->dtype()->code(), schema::DLDataTypeCode_kDLFloat);
  EXPECT_EQ(packed->data(), nullptr);  // metadata-only: no bytes serialized
}

// WithTensorMaterialization forces the bytes into Tensor.data, and UnPackTo reconstructs an owned
// (copied) Tensor with the same data.
TEST(Schema, TensorMaterializesAndUnpacksUnderScope) {
  const Tensor tensor = make_cpu_tensor_2x3();
  flatbuffers::FlatBufferBuilder builder;
  {
    // Key setup: materialization scope active during Pack forces bytes into Tensor.data.
    const schema::WithTensorMaterialization scope;
    schema::FinishTensorBuffer(builder, schema::Tensor::Pack(builder, &tensor));
  }

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyTensorBuffer(verifier));

  const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(packed, nullptr);
  ASSERT_NE(packed->data(), nullptr);
  EXPECT_EQ(packed->data()->size(), 6 * sizeof(float));

  // Key assertion: UnPackTo creates an owned copy with the original 2x3 shape and byte values.
  Tensor unpacked;
  packed->UnPackTo(&unpacked);
  ASSERT_TRUE(detail::TensorAccess::context(unpacked));
  EXPECT_EQ(unpacked.ndim(), 2);
  EXPECT_EQ(unpacked.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_TRUE(unpacked.is_dlpack_copied());
  ASSERT_NE(unpacked.data(), nullptr);
  const float* values = static_cast<const float*>(unpacked.data());
  EXPECT_FLOAT_EQ(values[0], 1.0F);
  EXPECT_FLOAT_EQ(values[5], 6.0F);
}

// UnPackTo accepts verifier-valid Tensor metadata when rank, shape, and explicit strides agree.
TEST(Schema, TensorUnpackAcceptsConsistentWireMetadata) {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<std::uint8_t, 6U> data{1U, 2U, 3U, 4U, 5U, 6U};
  const std::array<int64_t, 2U> shape{2, 3};
  const std::array<int64_t, 2U> strides{3, 1};
  const auto data_offset = builder.CreateVector(data.data(), data.size());
  const auto shape_offset = builder.CreateVector(shape.data(), shape.size());
  const auto strides_offset = builder.CreateVector(strides.data(), strides.size());
  const schema::DLDataType dtype{schema::DLDataTypeCode_kDLUInt, 8U, 1U};
  const schema::DLDevice device{schema::DLDeviceType_kDLCPU, 0};
  schema::FinishTensorBuffer(builder,
                             schema::CreateTensor(builder,
                                                  data_offset,
                                                  shape_offset,
                                                  &dtype,
                                                  &device,
                                                  /*ndim=*/2U,
                                                  strides_offset));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyTensorBuffer(verifier));
  const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(packed, nullptr);

  Tensor unpacked;
  packed->UnPackTo(&unpacked);
  EXPECT_EQ(unpacked.ndim(), 2);
  EXPECT_EQ(unpacked.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(unpacked.strides(), (std::vector<int64_t>{3, 1}));
  EXPECT_TRUE(unpacked.is_dlpack_copied());
  ASSERT_NE(unpacked.data(), nullptr);
  const std::uint8_t* values = static_cast<const std::uint8_t*>(unpacked.data());
  EXPECT_EQ(values[0], 1U);
  EXPECT_EQ(values[5], 6U);
}

// FlatBuffers verification checks wire structure rather than cross-field Tensor invariants.
// UnPackTo must reject inconsistent metadata before allocating or copying embedded Tensor bytes.
TEST(Schema, TensorUnpackRejectsInconsistentWireMetadata) {
  struct TestCase {
    std::string_view name;
    std::uint32_t ndim;
    std::vector<int64_t> shape;
    std::optional<std::vector<int64_t>> strides;
    std::string_view expected_error;
  };
  const std::array<TestCase, 4U> test_cases{
      TestCase{"ndim greater than shape",
               3U,
               {2, 3},
               std::nullopt,
               "cannot unpack FlatBuffer Tensor whose ndim does not match shape size"},
      TestCase{"ndim less than shape",
               1U,
               {2, 3},
               std::nullopt,
               "cannot unpack FlatBuffer Tensor whose ndim does not match shape size"},
      TestCase{"ndim exceeds int32",
               std::numeric_limits<std::uint32_t>::max(),
               {1},
               std::nullopt,
               "cannot unpack FlatBuffer Tensor whose ndim exceeds int32"},
      TestCase{"strides length differs from ndim",
               2U,
               {2, 3},
               std::vector<int64_t>{3},
               "cannot unpack FlatBuffer Tensor whose strides size does not match ndim"}};

  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    flatbuffers::FlatBufferBuilder builder;
    const std::array<std::uint8_t, 1U> data{0U};
    const auto data_offset = builder.CreateVector(data.data(), data.size());
    const auto shape_offset = builder.CreateVector(test_case.shape);
    flatbuffers::Offset<flatbuffers::Vector<std::int64_t>> strides_offset = 0;
    if (test_case.strides.has_value()) {
      strides_offset = builder.CreateVector(*test_case.strides);
    }
    const schema::DLDataType dtype{schema::DLDataTypeCode_kDLUInt, 8U, 1U};
    const schema::DLDevice device{schema::DLDeviceType_kDLCPU, 0};
    schema::FinishTensorBuffer(
        builder,
        schema::CreateTensor(
            builder, data_offset, shape_offset, &dtype, &device, test_case.ndim, strides_offset));

    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(schema::VerifyTensorBuffer(verifier));
    const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
    ASSERT_NE(packed, nullptr);

    Tensor unpacked;
    try {
      packed->UnPackTo(&unpacked);
      FAIL() << "expected inconsistent Tensor metadata to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_EQ(std::string_view{error.what()}, test_case.expected_error);
    }
    EXPECT_FALSE(detail::TensorAccess::context(unpacked));
  }
}

// WithTensorGather materializes only tensors strictly smaller than the byte threshold.
TEST(Schema, TensorGatherRespectsSizeThreshold) {
  const Tensor tensor = make_cpu_tensor_2x3();  // 24 bytes

  {
    // Key setup: threshold 16 is below the 24-byte tensor; metadata-only serialization expected.
    flatbuffers::FlatBufferBuilder builder;
    const schema::WithTensorGather scope(16);  // below 24 -> metadata only
    schema::FinishTensorBuffer(builder, schema::Tensor::Pack(builder, &tensor));
    EXPECT_EQ(schema::GetTensor(builder.GetBufferPointer())->data(), nullptr);
  }

  {
    // Key setup: threshold 64 is above the 24-byte tensor; full byte serialization expected.
    flatbuffers::FlatBufferBuilder builder;
    const schema::WithTensorGather scope(64);  // above 24 -> materialized
    schema::FinishTensorBuffer(builder, schema::Tensor::Pack(builder, &tensor));
    const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
    ASSERT_NE(packed->data(), nullptr);
    EXPECT_EQ(packed->data()->size(), 24U);
  }
}

// build_image_metadata encodes the Header (timestamp/frame id), encoding, and explicit Tensor
// strides, and parse_image_metadata reads them back. Local-channel metadata stays metadata-only:
// no Tensor bytes are materialized into the FlatBuffer.
TEST(Schema, ImageMetadataHelperRoundTrips) {
  const Tensor tensor = make_strided_cpu_tensor_3x2();
  const schema::ImageMetadataOptions options{.height = 3,
                                             .width = 2,
                                             .encoding = schema::ImageEncoding_MONO32F,
                                             .time_ns = 987654321,
                                             .acqtime_ns = 987654399,
                                             .frame_id = "imu_camera_frame"};

  auto metadata = schema::build_image_metadata(tensor, options);
  ASSERT_TRUE(metadata.has_value());

  auto parsed =
      schema::parse_image_metadata(std::span<const std::byte>(metadata->data(), metadata->size()));
  ASSERT_TRUE(parsed.has_value());
  const schema::Image* image = parsed->image;
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->height(), 3);
  EXPECT_EQ(image->width(), 2);
  EXPECT_EQ(image->encoding(), schema::ImageEncoding_MONO32F);
  ASSERT_NE(image->header(), nullptr);
  ASSERT_NE(image->header()->stamp(), nullptr);
  EXPECT_EQ(image->header()->stamp()->time_ns(), 987654321);
  EXPECT_EQ(image->header()->stamp()->acqtime_ns(), 987654399);
  ASSERT_NE(image->header()->frame_id(), nullptr);
  EXPECT_EQ(image->header()->frame_id()->string_view(), std::string_view("imu_camera_frame"));

  ASSERT_NE(image->data(), nullptr);
  ASSERT_NE(image->data()->shape(), nullptr);
  EXPECT_EQ(image->data()->shape()->size(), 2U);
  EXPECT_EQ(image->data()->shape()->Get(0), 3);
  EXPECT_EQ(image->data()->shape()->Get(1), 2);
  ASSERT_NE(image->data()->strides(), nullptr);
  EXPECT_EQ(image->data()->strides()->size(), 2U);
  EXPECT_EQ(image->data()->strides()->Get(0), 4);
  EXPECT_EQ(image->data()->strides()->Get(1), 1);
  if (image->data()->data() != nullptr) {
    EXPECT_EQ(image->data()->data()->size(), 0U);  // metadata-only: no bytes serialized
  }
}

// The descriptor-backed Image encoder is independent of the opt-in object-API materialization
// scopes. Pixel bytes remain in the separately retained Tensor even when ordinary Tensor::Pack
// calls in the same thread would gather them into a FlatBuffer.
// REQ: TENSOR-SCHEMA-005
TEST(Schema, ImageMetadataHelperIgnoresAmbientTensorMaterialization) {
  const Tensor tensor = make_cpu_tensor_2x3();
  const schema::ImageMetadataOptions options{.height = 2,
                                             .width = 3,
                                             .encoding = schema::ImageEncoding_MONO32F,
                                             .time_ns = 1234,
                                             .acqtime_ns = 1240,
                                             .frame_id = "bounded_camera"};

  const auto assert_metadata_only = [&](const expected<std::vector<std::byte>, Error>& metadata) {
    ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
    auto parsed = schema::parse_image_metadata(*metadata);
    ASSERT_TRUE(parsed) << runtime_error_message(parsed.error());
    ASSERT_NE(parsed->image->data(), nullptr);
    EXPECT_EQ(parsed->image->data()->data(), nullptr);
    ASSERT_NE(parsed->image->header(), nullptr);
    ASSERT_NE(parsed->image->header()->stamp(), nullptr);
    EXPECT_EQ(parsed->image->header()->stamp()->time_ns(), options.time_ns);
    ASSERT_NE(parsed->image->header()->frame_id(), nullptr);
    EXPECT_EQ(parsed->image->header()->frame_id()->string_view(), options.frame_id);
  };

  {
    const schema::WithTensorMaterialization scope;
    assert_metadata_only(schema::build_image_metadata(tensor, options));
  }
  {
    const schema::WithTensorGather scope(1024U);
    assert_metadata_only(schema::build_image_metadata(tensor, options));
  }
}

// Image metadata uses the descriptor envelope's finite metadata capacity. It rejects missing
// Tensor ownership and oversized metadata rather than falling back to an allocating object pack.
// REQ: TENSOR-SCHEMA-005
TEST(Schema, DescriptorImageMetadataCodecIsBoundedAndRejectsMissingTensor) {
  const schema::ImageMetadataOptions options{
      .height = 2, .width = 3, .encoding = schema::ImageEncoding_MONO32F};
  auto missing = schema::build_image_metadata(Tensor{}, options);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kInvalidArgument);

  auto oversized_options = options;
  oversized_options.frame_id.assign(512U, 'x');
  auto oversized = schema::build_image_metadata(make_cpu_tensor_2x3(), oversized_options);
  ASSERT_FALSE(oversized);
  EXPECT_EQ(oversized.error().code, ErrorCode::kResourceExhausted);
}

// Verifier-invalid bytes and verifier-valid Tensor metadata whose rank disagrees with shape are
// rejected before a descriptor-backed Tensor view can be exposed.
// REQ: TENSOR-SCHEMA-005
TEST(Schema, ImageMetadataRejectsMalformedAndMismatchedTensorMetadata) {
  // Key setup: 8 zero bytes cannot form a valid FlatBuffer; parse must return kCodecError.
  const std::array<std::byte, 8U> malformed{};
  auto malformed_result = schema::parse_image_metadata(malformed);
  ASSERT_FALSE(malformed_result);
  EXPECT_EQ(malformed_result.error().code, ErrorCode::kCodecError);

  // Key setup: construct a valid FlatBuffer with ndim=3 but a 2-element shape vector so that
  //            image_tensor_metadata detects the rank/shape disagreement and returns kCodecError.
  flatbuffers::FlatBufferBuilder builder;
  const std::array<int64_t, 2U> shape{2, 3};
  const auto shape_offset = builder.CreateVector(shape.data(), shape.size());
  const schema::DLDataType dtype{schema::DLDataTypeCode_kDLFloat, 32U, 1U};
  const schema::DLDevice device{schema::DLDeviceType_kDLCPU, 0};
  const auto tensor = schema::CreateTensor(builder,
                                           /*data=*/0,
                                           shape_offset,
                                           &dtype,
                                           &device,
                                           /*ndim=*/3U,
                                           /*strides=*/0);
  schema::FinishImageBuffer(builder,
                            schema::CreateImage(builder,
                                                /*header=*/0,
                                                /*height=*/2,
                                                /*width=*/3,
                                                schema::ImageEncoding_MONO32F,
                                                tensor));
  const auto metadata = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize());
  auto parsed = schema::parse_image_metadata(metadata);
  ASSERT_TRUE(parsed) << runtime_error_message(parsed.error());
  auto mismatched = schema::image_tensor_metadata(*parsed);
  ASSERT_FALSE(mismatched);
  EXPECT_EQ(mismatched.error().code, ErrorCode::kCodecError);

  // Key setup: Image packed with embedded bytes (via WithTensorMaterialization + object API) must
  //            be rejected by the descriptor path because it does not support inline-bytes images.
  std::vector<std::uint8_t> materialized;
  {
    const schema::WithTensorMaterialization scope;
    materialized = pack_image_object_api(make_cpu_tensor_2x3());
  }
  auto materialized_view = schema::parse_image_metadata(std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(materialized.data()), materialized.size()));
  ASSERT_TRUE(materialized_view) << runtime_error_message(materialized_view.error());
  auto hidden_bytes = schema::image_tensor_metadata(*materialized_view);
  ASSERT_FALSE(hidden_bytes);
  EXPECT_EQ(hidden_bytes.error().code, ErrorCode::kCodecError);
}

// Descriptor metadata rejects an element-stride to byte-stride conversion that cannot be
// represented by the public signed byte-stride model.
// REQ: TENSOR-SCHEMA-005
TEST(Schema, ImageMetadataRejectsByteStrideOverflow) {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<int64_t, 1U> shape{1};
  const std::array<int64_t, 1U> strides{std::numeric_limits<int64_t>::max()};
  const auto shape_offset = builder.CreateVector(shape.data(), shape.size());
  const auto strides_offset = builder.CreateVector(strides.data(), strides.size());
  const schema::DLDataType dtype{schema::DLDataTypeCode_kDLUInt, 16U, 1U};
  const schema::DLDevice device{schema::DLDeviceType_kDLCPU, 0};
  const auto tensor = schema::CreateTensor(builder,
                                           /*data=*/0,
                                           shape_offset,
                                           &dtype,
                                           &device,
                                           /*ndim=*/1U,
                                           strides_offset);
  schema::FinishImageBuffer(builder,
                            schema::CreateImage(builder,
                                                /*header=*/0,
                                                /*height=*/1,
                                                /*width=*/1,
                                                schema::ImageEncoding_CUSTOM,
                                                tensor));
  const auto metadata = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize());

  auto result =
      schema::DescriptorFlatBufferTensorViewTraits<schema::Image>::tensor_view_metadata(metadata);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kCodecError);
}

// build_image_metadata rejects non-positive dimensions.
TEST(Schema, ImageMetadataHelperRejectsInvalidDimensions) {
  const Tensor tensor = make_cpu_tensor_2x3();
  const schema::ImageMetadataOptions options{.height = 0, .width = 3};
  auto metadata = schema::build_image_metadata(tensor, options);
  ASSERT_FALSE(metadata.has_value());
  EXPECT_EQ(metadata.error().code, ErrorCode::kInvalidArgument);
}

// The embedded Tensor follows the same materialization rule as a root Tensor: metadata-only by
// default, full bytes when packed inside a WithTensorMaterialization scope.
TEST(Schema, ImageObjectApiMaterializesEmbeddedTensorInsideScope) {
  const Tensor tensor = make_cpu_tensor_2x3();

  const auto metadata_only = pack_image_object_api(tensor);
  const schema::Image* metadata_image = schema::GetImage(metadata_only.data());
  ASSERT_NE(metadata_image, nullptr);
  ASSERT_NE(metadata_image->data(), nullptr);
  if (metadata_image->data()->data() != nullptr) {
    EXPECT_EQ(metadata_image->data()->data()->size(), 0U);
  }

  std::vector<std::uint8_t> materialized;
  {
    const schema::WithTensorMaterialization scope;
    materialized = pack_image_object_api(tensor);
  }
  flatbuffers::Verifier verifier(materialized.data(), materialized.size());
  ASSERT_TRUE(schema::VerifyImageBuffer(verifier));

  const schema::Image* image = schema::GetImage(materialized.data());
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->encoding(), schema::ImageEncoding_MONO32F);
  ASSERT_NE(image->data(), nullptr);
  ASSERT_NE(image->data()->data(), nullptr);
  EXPECT_EQ(image->data()->data()->size(), 6U * sizeof(float));
  const auto* const image_data = image->data()->data()->data();
  float first_value = 0.0F;
  float last_value = 0.0F;
  std::memcpy(&first_value, image_data, sizeof(first_value));
  std::memcpy(&last_value, image_data + 5U * sizeof(float), sizeof(last_value));
  EXPECT_FLOAT_EQ(first_value, 1.0F);
  EXPECT_FLOAT_EQ(last_value, 6.0F);
}

}  // namespace holoscan
