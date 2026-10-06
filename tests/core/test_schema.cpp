/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
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

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/schema/tensor_materialization.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include "channel/channel_payload_limits.hpp"
#include "core/domain/tensor_access.hpp"

#include "header_generated.h"
#include "image_generated.h"
#include "imu_generated.h"
#include "string_generated.h"
#include "tensor_generated.h"
#include "tensor_packet_generated.h"
#include "time_generated.h"

namespace holoscan {

template <typename T>
concept HasGeneratedFlatBufferName = requires {
  { T::GetFullyQualifiedName() } -> std::convertible_to<const char*>;
};

using DescriptorTensorAssign =
    decltype(detail::RuntimePortTypeOps{}.assign_descriptor_tensors_sample);
static_assert(std::is_invocable_r_v<expected<void, Error>, DescriptorTensorAssign, void*,
                                    std::span<const std::byte>, std::span<Tensor>,
                                    detail::DescriptorTensorWireMetadata,
                                    detail::DescriptorTensorSampleMetadata>);
static_assert(!std::is_invocable_r_v<expected<void, Error>, DescriptorTensorAssign, void*,
                                     std::span<const std::byte>, std::span<Tensor>,
                                     detail::DescriptorTensorSampleMetadata,
                                     detail::DescriptorTensorWireMetadata>);

// REQ: SCHEMA-001
TEST(Schema, GeneratedNamesAndRootNativeMappingsUseShippedCodegen) {
  using Root = schema::TensorPacket;
  using Native = schema::TensorPacketT;

  static_assert(HasGeneratedFlatBufferName<Root>);
  static_assert(HasGeneratedFlatBufferName<Native>);
  static_assert(HasGeneratedFlatBufferName<schema::Tensor>);
  static_assert(std::same_as<typename Native::TableType, Root>);
  static_assert(std::same_as<typename Root::NativeTableType, Native>);
  static_assert(std::same_as<typename schema::Tensor::NativeTableType, Tensor>);
  static_assert(std::string_view(Root::GetFullyQualifiedName()) == "holoscan.schema.TensorPacket");
  static_assert(std::string_view(Native::GetFullyQualifiedName()) ==
                "holoscan.schema.TensorPacketT");
  static_assert(std::string_view(schema::Tensor::GetFullyQualifiedName()) ==
                "holoscan.schema.Tensor");
}

struct IndexedDescriptorSchemaNative {
  std::array<std::shared_ptr<Tensor>, 3U> tensors;
};

struct InvalidMissingIndexedTensorAccessNative {
  std::shared_ptr<Tensor> tensor;
};
struct InvalidMissingSingleTensorFieldIdNative {};

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
struct DescriptorTensorSchemaNativeTraits<InvalidMissingIndexedTensorAccessNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 3U;

  [[nodiscard]] static const Tensor* tensor(
      const InvalidMissingIndexedTensorAccessNative& value) noexcept {
    return value.tensor.get();
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<InvalidMissingSingleTensorFieldIdNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 1U;

  [[nodiscard]] static const Tensor* tensor_at(const InvalidMissingSingleTensorFieldIdNative&,
                                               std::size_t) noexcept {
    return nullptr;
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
  schema::FinishStringBuffer(builder, schema::CreateStringDirect(builder, "hello flatbuffers"));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyStringBuffer(verifier));

  const schema::String* parsed = schema::GetString(builder.GetBufferPointer());
  ASSERT_NE(parsed, nullptr);
  ASSERT_NE(parsed->data(), nullptr);
  EXPECT_EQ(parsed->data()->str(), "hello flatbuffers");
}

TEST(Schema, TimeSchemaRoundTrips) {
  flatbuffers::FlatBufferBuilder builder;
  schema::FinishTimeBuffer(builder, schema::CreateTime(builder, 123456789, 123456790));

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
  schema::FinishHeaderBuffer(builder, schema::CreateHeader(builder, stamp, frame_id, 300, 5U));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyHeaderBuffer(verifier));

  const schema::Header* header = schema::GetHeader(builder.GetBufferPointer());
  ASSERT_NE(header, nullptr);
  ASSERT_NE(header->stamp(), nullptr);
  EXPECT_EQ(header->stamp()->time_ns(), 100);
  EXPECT_EQ(header->capture_timestamp_ns(), 300);
  EXPECT_EQ(header->device_sequence(), 5U);
  ASSERT_NE(header->frame_id(), nullptr);
  EXPECT_EQ(header->frame_id()->string_view(), std::string_view("camera_0"));
}

TEST(Schema, ImuSchemaRoundTrips) {
  flatbuffers::FlatBufferBuilder builder;
  const auto stamp = schema::CreateTime(builder, 1, 2);
  const auto frame_id = builder.CreateString("imu_link");
  const auto header = schema::CreateHeader(builder, stamp, frame_id);
  const schema::Quaternionf orientation{0.0F, 0.0F, 0.0F, 1.0F};
  const schema::Vector3f angular_velocity{0.1F, 0.2F, 0.3F};
  const schema::Vector3f linear_acceleration{9.8F, 0.0F, -0.1F};
  schema::FinishImuBuffer(
      builder,
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

TEST(Schema, GeometryDoublePrecisionStructsMatchFloatFieldOrder) {
  const schema::Quaterniond orientation{0.0, 0.0, 0.0, 1.0};
  EXPECT_DOUBLE_EQ(orientation.x(), 0.0);
  EXPECT_DOUBLE_EQ(orientation.w(), 1.0);

  const schema::Vector3d position{1.5, -2.5, 3.5};
  EXPECT_DOUBLE_EQ(position.x(), 1.5);
  EXPECT_DOUBLE_EQ(position.y(), -2.5);
  EXPECT_DOUBLE_EQ(position.z(), 3.5);

  double covariance[9] = {};
  covariance[0] = 1.0;
  covariance[4] = 1.0;
  covariance[8] = 1.0;
  const schema::Matrix3d identity_covariance{covariance};
  EXPECT_DOUBLE_EQ(identity_covariance.m()->Get(0), 1.0);
  EXPECT_DOUBLE_EQ(identity_covariance.m()->Get(4), 1.0);
  EXPECT_DOUBLE_EQ(identity_covariance.m()->Get(8), 1.0);
}

// REQ: OPERATOR-035
TEST(Schema, ImageDescriptorPortOpsPreserveLegacyTableShape) {
  using Payload = schema::DescriptorFlatBufferPayload<
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>;
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
  // Key setup: a three-field native trait exposes the mandatory indexed inventory while malformed
  //   traits omit indexed access or provide invalid field IDs.
  // Key assertion: indexed count/read operations preserve canonical order, transactional
  //   assignment receives all three handles, and malformed traits are not admitted.
  static_assert(detail::HasDescriptorTensorSchemaNativeTraits<IndexedDescriptorSchemaNative>);
  static_assert(
      !detail::HasDescriptorTensorSchemaNativeTraits<InvalidMissingIndexedTensorAccessNative>);
  static_assert(
      !detail::HasDescriptorTensorSchemaNativeTraits<InvalidMissingSingleTensorFieldIdNative>);
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
      .assign_descriptor_tensors_sample =
          [](void* storage,
             std::span<const std::byte>,
             std::span<Tensor>
                 tensors,
             detail::DescriptorTensorWireMetadata,
             detail::DescriptorTensorSampleMetadata) noexcept -> expected<void, Error> {
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
  auto assigned = ops.assign_descriptor_tensors_sample(&destination,
                                                       {},
                                                       assigned_tensors,
                                                       detail::DescriptorTensorWireMetadata{},
                                                       detail::DescriptorTensorSampleMetadata{});
  ASSERT_TRUE(assigned) << runtime_error_message(assigned.error());
  for (const auto& tensor : destination.tensors) {
    EXPECT_NE(tensor, nullptr);
  }
}

// REQ: CHAN-145
// REQ: CHAN-120
// REQ: OPERATOR-035
template <typename Payload>
void expect_current_descriptor_layout() {
  static_assert(std::is_standard_layout_v<Payload>);
  EXPECT_EQ(sizeof(Payload), 880U);
  EXPECT_EQ(offsetof(Payload, envelope_abi_version), 0U);
  EXPECT_EQ(offsetof(Payload, envelope_header_size), 4U);
  EXPECT_EQ(offsetof(Payload, schema_family_id), 8U);
  EXPECT_EQ(offsetof(Payload, metadata_size), 24U);
  EXPECT_EQ(offsetof(Payload, transport_profile_tag), 28U);
  EXPECT_EQ(offsetof(Payload, endpoint_generation), 32U);
  EXPECT_EQ(offsetof(Payload, descriptor), 40U);
  EXPECT_EQ(offsetof(Payload, metadata), 368U);
}

TEST(Schema, SingularDescriptorSchemasUseCurrentCarriage) {
  // Key setup: instantiate both SDK descriptor-backed schemas with one Tensor field.
  // Key assertion: both use the same explicit count-and-mask carriage as N-field schemas.
  using ImagePayload = schema::DescriptorFlatBufferPayload<
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>;
  using TensorPacketPayload = schema::DescriptorFlatBufferPayload<
      schema::TensorPacket,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::TensorPacket>>;

  EXPECT_EQ(kSchemaEnvelopeABIVersion, 1U);
  EXPECT_EQ(kDescriptorTensorSchemaEnvelopeABIVersion, 1U);
  EXPECT_EQ(kSchemaEnvelopeHeaderSize, 40U);
  expect_current_descriptor_layout<ImagePayload>();
  expect_current_descriptor_layout<TensorPacketPayload>();
  EXPECT_EQ(ImagePayload{}.descriptor_count(), 1U);
  EXPECT_EQ(ImagePayload{}.tensor_presence_mask(), 1U);
  EXPECT_EQ(ImagePayload{}.tensor_descriptor_mask(), 1U);
}

// REQ: SCHEMA-004
TEST(Schema, DescriptorTransportProfileTracksGeneratedTensorInventoryAndCarriage) {
  constexpr auto image_profile =
      detail::descriptor_tensor_schema_transport_profile_v<schema::ImageT>;
  constexpr auto packet_profile =
      detail::descriptor_tensor_schema_transport_profile_v<schema::TensorPacketT>;
  constexpr auto indexed_profile =
      detail::descriptor_tensor_schema_transport_profile_v<IndexedDescriptorSchemaNative>;
  constexpr std::array<std::uint16_t, 2U> expanded_field_ids{4U, 7U};
  constexpr auto expanded_profile_tag = derive_descriptor_tensor_transport_profile_tag(
      std::span{expanded_field_ids}, image_profile.metadata_capacity_bytes);

  static_assert(schema_transport_profile_tag(image_profile) != 0U);
  static_assert(image_profile.tensor_field_ids.size() == 1U);
  static_assert(image_profile.tensor_field_ids.front() == 4U);
  static_assert(schema_transport_profile_field_mask(image_profile) == 1U);
  static_assert(schema_transport_profile_tag(image_profile) !=
                schema_transport_profile_tag(packet_profile));
  static_assert(schema_transport_profile_tag(image_profile) != expanded_profile_tag);
  static_assert(indexed_profile.tensor_field_ids.size() == 3U);
  static_assert(indexed_profile.tensor_field_ids[0] == 2U);
  static_assert(indexed_profile.tensor_field_ids[1] == 5U);
  static_assert(indexed_profile.tensor_field_ids[2] == 9U);
  static_assert(schema_transport_profile_field_mask(indexed_profile) == 7U);
  static_assert(schema_transport_profile_tag(indexed_profile) !=
                schema_transport_profile_tag(image_profile));
  EXPECT_EQ(schema_transport_profile_tag(schema_package<schema::Image>().transport_profile()),
            schema_transport_profile_tag(image_profile));
  EXPECT_EQ(detail::operator_spec_payload_identity<schema::ImageT>().transport_profile_tag,
            schema_transport_profile_tag(image_profile));
  EXPECT_EQ(detail::operator_spec_payload_identity<schema::StringT>().transport_profile_tag, 0U);
}

// REQ: CHAN-145
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

// REQ: CHAN-145
// REQ: SCHEMA-004
TEST(Schema, DescriptorTensorEnvelopeUsesUniformArrayAndExplicitMasks) {
  // Key setup: encode the same metadata into one- and three-Tensor descriptor envelopes, then
  //   deliberately stamp an unsupported ABI version.
  // Key assertion: both envelopes carry explicit count-and-mask state and reject an unsupported
  //   version before typed access.
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
        .schema_id = schema::flatbuffer_schema_identity_v<schema::Image>.family_id,
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
  const auto& package = schema_package<schema::Image>();
  // The generated Image profile has one Tensor field. A manually enlarged wrapper is rejected
  // even though its structural descriptor array is otherwise representable.
  auto incompatible_multi = MultiPayload::copy_from_builder(builder, descriptors, 9U);
  ASSERT_FALSE(incompatible_multi);
  EXPECT_EQ(incompatible_multi.error().code, ErrorCode::kCodecError);
  const auto metadata = std::span<const std::byte>{
      reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize()};
  auto incompatible_trusted = MultiPayload::copy_from_trusted_metadata(metadata, descriptors, 9U);
  ASSERT_FALSE(incompatible_trusted);
  EXPECT_EQ(incompatible_trusted.error().code, ErrorCode::kCodecError);

  auto singular = SingularPayload::copy_from_builder(builder, descriptors.front(), 9U);
  ASSERT_TRUE(singular) << runtime_error_message(singular.error());
  EXPECT_EQ(singular->envelope_abi_version, kDescriptorTensorSchemaEnvelopeABIVersion);
  EXPECT_EQ(singular->descriptor_count(), 1U);
  EXPECT_EQ(singular->tensor_presence_mask(), 1U);
  EXPECT_EQ(singular->tensor_descriptor_mask(), 1U);
  EXPECT_TRUE(singular->verified_root(package, 9U));
  auto wrong_family = *singular;
  wrong_family.schema_family_id.value.front() ^= std::byte{0xFF};
  auto rejected_family_validation = wrong_family.validate();
  ASSERT_FALSE(rejected_family_validation);
  EXPECT_EQ(rejected_family_validation.error().message, "FlatBuffer schema family mismatch");
  auto wrong_profile = *singular;
  ++wrong_profile.transport_profile_tag;
  auto rejected_profile_validation = wrong_profile.validate();
  ASSERT_FALSE(rejected_profile_validation);
  EXPECT_EQ(rejected_profile_validation.error().message,
            "FlatBuffer transport profile tag mismatch");
  auto rejected_profile = wrong_profile.verified_root(package, 9U);
  ASSERT_FALSE(rejected_profile);
  EXPECT_EQ(rejected_profile.error().code, ErrorCode::kProtocolError);
  singular->envelope_abi_version = kDescriptorTensorSchemaEnvelopeABIVersion + 1U;
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

// A 1-D uint8 CPU tensor of the requested byte length, backed by storage that outlives the caller.
//
// The shape a multi-planar producer actually publishes. Plane records are byte ranges inside one
// allocation, so the Tensor an Image carries has to span the bytes those ranges name; a Tensor
// sized for something else cannot contain them and is rejected. A deque because each tensor holds
// a pointer into its own element and deque never relocates the elements already in it.
Tensor make_cpu_byte_tensor(const std::int64_t byte_count) {
  static std::deque<std::vector<std::uint8_t>> storage;
  auto& bytes = storage.emplace_back(static_cast<std::size_t>(byte_count), std::uint8_t{0});
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {byte_count};
  context->tensor.dl_tensor.data = bytes.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 8, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;
  return Tensor(std::move(context));
}

// A 1-D uint16 CPU tensor spanning the requested byte length, backed by storage that outlives the
// caller.
//
// The carriage a producer uses for an encoding whose container is wider than a byte. Alongside
// make_cpu_byte_tensor it is what distinguishes the two admissible carriages -- a typed view and an
// opaque byte buffer -- from an element type that contradicts the encoding outright.
Tensor make_cpu_uint16_tensor(const std::int64_t byte_count) {
  static std::deque<std::vector<std::uint16_t>> storage;
  auto& elements = storage.emplace_back(
      static_cast<std::size_t>(byte_count) / sizeof(std::uint16_t), std::uint16_t{0});
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {static_cast<std::int64_t>(elements.size())};
  context->tensor.dl_tensor.data = elements.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 16, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;
  return Tensor(std::move(context));
}

// A 1-D uint8 CPU tensor whose element groups the requested number of lanes, backed by storage that
// outlives the caller.
//
// The carriage the Image element rule refuses: a vectorized element states a component organization
// the encoding has already fixed. Sized in elements rather than bytes so the caller controls the
// span independently of the lane count, which is what lets a case reject on the lane rule instead
// of on a frame that does not fit.
Tensor make_cpu_lane_tensor(const std::int64_t element_count, const std::uint16_t lanes) {
  static std::deque<std::vector<std::uint8_t>> storage;
  auto& bytes =
      storage.emplace_back(static_cast<std::size_t>(element_count) * lanes, std::uint8_t{0});
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {element_count};
  context->tensor.dl_tensor.data = bytes.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 8, .lanes = lanes};
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
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::TensorPacket>>;
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
  descriptor->schema_id = schema_identity<Native>().family_id;
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
  SampleMetadata normalized_metadata = metadata;
  normalized_metadata.capture_timestamp_ns = 123U;
  normalized_metadata.source_clock_id = ClockId{9U};
  auto assigned = ops->assign_descriptor_tensor_sample(
      storage.data(),
      wire,
      make_cpu_tensor_2x3(),
      detail::DescriptorTensorWireMetadata{metadata},
      detail::DescriptorTensorSampleMetadata{normalized_metadata});
  ASSERT_TRUE(assigned) << runtime_error_message(assigned.error());
  EXPECT_EQ(sample->data.sequence, 42U);
  EXPECT_EQ(sample->data.source, "third-party-camera");
  EXPECT_DOUBLE_EQ(sample->data.scale, 2.5);
  EXPECT_EQ(sample->data.data.get(), retained_tensor_holder);
  EXPECT_EQ(sample->data.data->shape(), (std::vector<std::int64_t>{2, 3}));
  // Key assertion: singular assignment validates against wire metadata while exposing the
  // independently normalized plan-clock coordinate.
  EXPECT_EQ(sample->metadata.sequence, 19U);
  EXPECT_EQ(sample->metadata.capture_timestamp_ns, 123U);
  EXPECT_EQ(sample->metadata.source_clock_id, ClockId{9U});

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
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::TensorPacket>>;
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
  auto assigned =
      ops->assign_descriptor_tensor_sample(storage.data(),
                                           wire,
                                           make_strided_cpu_tensor_3x2(),
                                           detail::DescriptorTensorWireMetadata{metadata},
                                           detail::DescriptorTensorSampleMetadata{metadata});
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
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::TensorPacket>;
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

// Embedded Tensor.data must cover the storage span implied by shape/dtype/strides. Undersized
// payloads would otherwise produce a Tensor whose nbytes()/strided indexing reads out of bounds.
TEST(Schema, TensorUnpackRejectsUndersizedDataForLayout) {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<std::uint8_t, 1U> data{0U};  // far smaller than a 2x3 uint8 span
  const std::array<int64_t, 2U> shape{2, 3};
  const auto data_offset = builder.CreateVector(data.data(), data.size());
  const auto shape_offset = builder.CreateVector(shape.data(), shape.size());
  const schema::DLDataType dtype{schema::DLDataTypeCode_kDLUInt, 8U, 1U};
  const schema::DLDevice device{schema::DLDeviceType_kDLCPU, 0};
  schema::FinishTensorBuffer(
      builder,
      schema::CreateTensor(
          builder, data_offset, shape_offset, &dtype, &device, /*ndim=*/2U, /*strides=*/0));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyTensorBuffer(verifier));
  const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(packed, nullptr);

  Tensor unpacked;
  try {
    packed->UnPackTo(&unpacked);
    FAIL() << "expected undersized Tensor data to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string_view{error.what()},
              "cannot unpack FlatBuffer Tensor whose data size is smaller than the layout span");
  }
  EXPECT_FALSE(detail::TensorAccess::context(unpacked));
}

TEST(Schema, TensorUnpackRejectsInvalidLayoutMetadata) {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<std::uint8_t, 6U> data{1U, 2U, 3U, 4U, 5U, 6U};
  const std::array<int64_t, 2U> shape{2, -3};  // negative extent
  const auto data_offset = builder.CreateVector(data.data(), data.size());
  const auto shape_offset = builder.CreateVector(shape.data(), shape.size());
  const schema::DLDataType dtype{schema::DLDataTypeCode_kDLUInt, 8U, 1U};
  const schema::DLDevice device{schema::DLDeviceType_kDLCPU, 0};
  schema::FinishTensorBuffer(
      builder,
      schema::CreateTensor(
          builder, data_offset, shape_offset, &dtype, &device, /*ndim=*/2U, /*strides=*/0));

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(schema::VerifyTensorBuffer(verifier));
  const schema::Tensor* packed = schema::GetTensor(builder.GetBufferPointer());
  ASSERT_NE(packed, nullptr);

  Tensor unpacked;
  try {
    packed->UnPackTo(&unpacked);
    FAIL() << "expected invalid Tensor layout to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string_view{error.what()},
              "cannot unpack FlatBuffer Tensor with invalid layout metadata");
  }
  EXPECT_FALSE(detail::TensorAccess::context(unpacked));
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
                                             .capture_timestamp_ns = 987654321,
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
  EXPECT_EQ(image->header()->capture_timestamp_ns(), 987654321);
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

// Round-trip coverage for the descriptor fields that complete the camera contract. The round-trip
// above covers geometry, header times, and Tensor layout; these are the fields a consumer needs in
// order to interpret the samples, and each is read back through parse_image_metadata so that a
// field missed by either the encoder or the decoder fails here rather than silently arriving as
// zero at a consumer.
//
// The sequence is an unsigned source identifier. Precision has encoding-specific bounds and an
// explicit unknown sentinel, covered by the validation tests below.
TEST(Schema, ImageMetadataHelperRoundTripsTheCameraDescriptorFields) {
  // Spans the two planes below, since a Tensor that cannot contain them is rejected before any
  // field reaches the wire to be read back.
  const Tensor tensor = make_cpu_byte_tensor(192);
  // P010 because a plane layout is only meaningful against a multi-planar encoding: a luma plane
  // followed by an interleaved chroma plane at half height. It is also the encoding
  // significant_bits exists for, carrying 10 bits of precision in a 16-bit container, so the two
  // fields under test here are exercised against an encoding that admits both rather than against
  // one where 10 bits would not fit the component.
  const schema::ImageMetadataOptions options{
      .height = 2,
      .width = 3,
      .encoding = schema::ImageEncoding_P010,
      .device_sequence = 4242U,
      .color_space = schema::ColorSpace_REC2020_PQ,
      .significant_bits = 10U,
      .roi_offset_x = 16,
      .roi_offset_y = 480,
      .plane_layouts = {schema::ImagePlaneLayout{0U, 64U, 128U},
                        schema::ImagePlaneLayout{128U, 64U, 64U}},
      .exposure_time_ns = 8'000'000,
      .gain = 2.5F};

  auto metadata = schema::build_image_metadata(tensor, options);
  ASSERT_TRUE(metadata.has_value());
  auto parsed =
      schema::parse_image_metadata(std::span<const std::byte>(metadata->data(), metadata->size()));
  ASSERT_TRUE(parsed.has_value());
  const schema::Image* image = parsed->image;
  ASSERT_NE(image, nullptr);

  EXPECT_EQ(image->encoding(), schema::ImageEncoding_P010);
  EXPECT_EQ(image->color_space(), schema::ColorSpace_REC2020_PQ);
  EXPECT_EQ(image->significant_bits(), 10U);
  EXPECT_EQ(image->roi_offset_x(), 16);
  EXPECT_EQ(image->roi_offset_y(), 480);
  EXPECT_EQ(image->exposure_time_ns(), 8'000'000);
  EXPECT_FLOAT_EQ(image->gain(), 2.5F);
  // The device counter moved to the Header, so it reads back from there rather than off the Image.
  ASSERT_NE(image->header(), nullptr);
  EXPECT_EQ(image->header()->device_sequence(), 4242U);
  // Each plane has to survive as a whole: an offset that round-tripped without its extent would
  // still leave a consumer unable to bound its reads.
  ASSERT_NE(image->plane_layouts(), nullptr);
  ASSERT_EQ(image->plane_layouts()->size(), 2U);
  EXPECT_EQ(image->plane_layouts()->Get(0)->offset_bytes(), 0U);
  EXPECT_EQ(image->plane_layouts()->Get(0)->row_stride_bytes(), 64U);
  EXPECT_EQ(image->plane_layouts()->Get(0)->size_bytes(), 128U);
  EXPECT_EQ(image->plane_layouts()->Get(1)->offset_bytes(), 128U);
  EXPECT_EQ(image->plane_layouts()->Get(1)->row_stride_bytes(), 64U);
  EXPECT_EQ(image->plane_layouts()->Get(1)->size_bytes(), 64U);
}

// A producer that reports none of the optional descriptor facts must be distinguishable from one
// that reported zeros, and an absent plane layout vector must stay absent. This is the half a
// round-trip test cannot show on its own: a decoder that fabricates a default would satisfy the
// test above.
TEST(Schema, ImageMetadataHelperLeavesUnreportedCameraFieldsAbsent) {
  const Tensor tensor = make_cpu_tensor_2x3();
  // MONO32F rather than MONO16, because the Tensor is float32 and a known encoding may not
  // contradict the element type carrying it.
  const schema::ImageMetadataOptions options{
      .height = 2, .width = 3, .encoding = schema::ImageEncoding_MONO32F};

  auto metadata = schema::build_image_metadata(tensor, options);
  ASSERT_TRUE(metadata.has_value());
  auto parsed =
      schema::parse_image_metadata(std::span<const std::byte>(metadata->data(), metadata->size()));
  ASSERT_TRUE(parsed.has_value());
  const schema::Image* image = parsed->image;
  ASSERT_NE(image, nullptr);

  EXPECT_EQ(image->color_space(), schema::ColorSpace_UNSPECIFIED);
  EXPECT_EQ(image->significant_bits(), 0U);
  EXPECT_EQ(image->roi_offset_x(), 0);
  EXPECT_EQ(image->roi_offset_y(), 0);
  EXPECT_EQ(image->exposure_time_ns(), 0);
  EXPECT_FLOAT_EQ(image->gain(), 0.0F);
  ASSERT_NE(image->header(), nullptr);
  EXPECT_EQ(image->header()->device_sequence(), 0U);
  // An empty vector is not encoded at all, so the field is genuinely missing rather than present
  // and empty. Either reads as "no pitches", so both are accepted.
  if (image->plane_layouts() != nullptr) {
    EXPECT_EQ(image->plane_layouts()->size(), 0U);
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
                                             .capture_timestamp_ns = 1234,
                                             .frame_id = "bounded_camera"};

  const auto assert_metadata_only = [&](const expected<std::vector<std::byte>, Error>& metadata) {
    ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
    auto parsed = schema::parse_image_metadata(*metadata);
    ASSERT_TRUE(parsed) << runtime_error_message(parsed.error());
    ASSERT_NE(parsed->image->data(), nullptr);
    EXPECT_EQ(parsed->image->data()->data(), nullptr);
    ASSERT_NE(parsed->image->header(), nullptr);
    EXPECT_EQ(parsed->image->header()->capture_timestamp_ns(), options.capture_timestamp_ns);
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
  // Key setup: 8 zero bytes cannot form a valid FlatBuffer; parse must return kCodecError. Aligned
  // so the content is what fails: std::byte has no alignment of its own, and an unaligned automatic
  // array would be refused before verification and pass this test for the wrong reason.
  alignas(std::max_align_t) const std::array<std::byte, 8U> malformed{};
  auto malformed_result = schema::parse_image_metadata(malformed);
  ASSERT_FALSE(malformed_result);
  EXPECT_EQ(malformed_result.error().code, ErrorCode::kCodecError);

  // Key setup: a correctly formed metadata buffer offset by one byte. Verification would accept the
  // same bytes at an aligned address, so this is the caller mistake the guard exists for.
  auto well_formed = schema::build_image_metadata(
      make_cpu_tensor_2x3(),
      schema::ImageMetadataOptions{
          .height = 2, .width = 3, .encoding = schema::ImageEncoding_MONO32F});
  ASSERT_TRUE(well_formed) << runtime_error_message(well_formed.error());
  alignas(std::max_align_t) std::array<std::byte, 512U> shifted{};
  ASSERT_LT(well_formed->size() + 1U, shifted.size());
  std::memcpy(shifted.data() + 1U, well_formed->data(), well_formed->size());
  auto unaligned_result = schema::parse_image_metadata(
      std::span<const std::byte>{shifted.data() + 1U, well_formed->size()});
  ASSERT_FALSE(unaligned_result);
  EXPECT_EQ(unaligned_result.error().code, ErrorCode::kInvalidArgument);
  // The same bytes at an aligned address parse, so the rejection above is the alignment and not the
  // content.
  EXPECT_TRUE(schema::parse_image_metadata(*well_formed).has_value());

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

// The signed and enumerated descriptor fields can each hold a state the schema gives no meaning
// to. The receive path copies them through verbatim, so a value rejected here is one that would
// otherwise reach a consumer looking exactly like a measurement.
// REQ: SENSOR-IO-010
TEST(Schema, ImageMetadataHelperRejectsFieldsTheSchemaGivesNoMeaning) {
  const Tensor tensor = make_cpu_tensor_2x3();
  const auto rejects = [&tensor](const schema::ImageMetadataOptions& options) {
    auto metadata = schema::build_image_metadata(tensor, options);
    ASSERT_FALSE(metadata.has_value());
    EXPECT_EQ(metadata.error().code, ErrorCode::kInvalidArgument);
  };

  rejects({.height = 2,
           .width = 3,
           .color_space = static_cast<schema::ColorSpace>(schema::ColorSpace_MAX + 1)});
  rejects({.height = 2, .width = 3, .roi_offset_x = -1});
  rejects({.height = 2, .width = 3, .roi_offset_y = -1});
  rejects({.height = 2, .width = 3, .exposure_time_ns = -1});
  rejects({.height = 2, .width = 3, .gain = -1.0F});
  // A gain that is not a real number compares false against every bound, so the check has to be
  // ordered to catch it rather than to let it fall through as though it were in range.
  rejects({.height = 2, .width = 3, .gain = std::numeric_limits<float>::quiet_NaN()});
  rejects({.height = 2, .width = 3, .gain = std::numeric_limits<float>::infinity()});
  // Every case here carries the default CUSTOM encoding, which states no component organization
  // for a figure to contradict. What remains is the bound that holds without one: no component of
  // any element type is wider than 64 bits, so a larger figure describes no sample anywhere. The
  // per-encoding bounds are the subject of their own test below.
  rejects({.height = 2, .width = 3, .significant_bits = 65U});
  rejects({.height = 2, .width = 3, .significant_bits = 254U});

  const auto admits = [&tensor](const schema::ImageMetadataOptions& options) {
    EXPECT_TRUE(schema::build_image_metadata(tensor, options).has_value());
  };
  admits({.height = 2, .width = 3, .significant_bits = 10U});
  admits({.height = 2, .width = 3, .significant_bits = 64U});
  // Zero stays the way a producer says the precision equals the container width.
  admits({.height = 2, .width = 3, .significant_bits = 0U});
}

// significant_bits is precision inside a container, so the container bounds it. The bound has to be
// the one the encoding names rather than the Tensor element type: P010 is defined to carry 10-bit
// samples in 16-bit words and is routinely delivered as a flat byte buffer, so measuring against
// the element type would cap the encoding at 8 and reject its own definition. Measuring against the
// widest component any type provides, which is what this used to do, bounds nothing in practice and
// accepts 64 bits of precision in an 8-bit component.
// REQ: SENSOR-IO-010
TEST(Schema, ImageMetadataHelperBoundsSignificantBitsByTheEncodingsComponent) {
  const auto build = [](const Tensor& tensor,
                        const schema::ImageEncoding encoding,
                        const std::uint8_t significant_bits) {
    return schema::build_image_metadata(tensor,
                                        schema::ImageMetadataOptions{
                                            .height = 2,
                                            .width = 3,
                                            .encoding = encoding,
                                            .significant_bits = significant_bits,
                                        });
  };
  const auto rejects = [&build](const Tensor& tensor,
                                const schema::ImageEncoding encoding,
                                const std::uint8_t significant_bits) {
    auto result = build(tensor, encoding, significant_bits);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // An 8-bit encoding cannot hold nine bits of precision, whatever the producer reports.
  const Tensor bytes = make_cpu_byte_tensor(64);
  EXPECT_TRUE(build(bytes, schema::ImageEncoding_MONO8, 8U).has_value());
  rejects(bytes, schema::ImageEncoding_MONO8, 9U);
  // The 16-bit MONO entry admits what the 8-bit one refuses, so the bound follows the encoding and
  // is not one constant shared by the family.
  EXPECT_TRUE(build(bytes, schema::ImageEncoding_MONO16, 16U).has_value());
  rejects(bytes, schema::ImageEncoding_MONO16, 17U);

  // The case the encoding-relative bound exists for. P010's samples are 16-bit and its carriage
  // here is a byte buffer, so a bound read off the element type would have stopped at 8.
  const std::vector<schema::ImagePlaneLayout> p010_planes{schema::ImagePlaneLayout{0U, 12U, 24U},
                                                          schema::ImagePlaneLayout{24U, 12U, 12U}};
  const auto build_p010 = [&bytes, &p010_planes](const std::uint8_t significant_bits) {
    schema::ImageMetadataOptions options{.height = 2,
                                         .width = 3,
                                         .encoding = schema::ImageEncoding_P010,
                                         .significant_bits = significant_bits};
    options.plane_layouts = p010_planes;
    return schema::build_image_metadata(bytes, options);
  };
  EXPECT_TRUE(build_p010(10U).has_value());
  EXPECT_TRUE(build_p010(16U).has_value());
  auto beyond_container = build_p010(17U);
  ASSERT_FALSE(beyond_container.has_value());
  EXPECT_EQ(beyond_container.error().code, ErrorCode::kInvalidArgument);

  // No component of a 10:10:10:2 word is wider than ten bits, so the bound is the component and not
  // the 32-bit word the four of them are packed into.
  EXPECT_TRUE(build(bytes, schema::ImageEncoding_RGBA1010102, 10U).has_value());
  rejects(bytes, schema::ImageEncoding_RGBA1010102, 11U);

  // A Bayer encoding states no container and takes the Tensor's, so the same figure is admissible
  // against a 16-bit element type and not against an 8-bit one.
  EXPECT_TRUE(build(make_cpu_uint16_tensor(64), schema::ImageEncoding_BAYER_RGGB, 12U).has_value());
  rejects(bytes, schema::ImageEncoding_BAYER_RGGB, 12U);
}

// REQ: SENSOR-IO-010
TEST(Schema, ImageUnknownPrecisionRoundTripsWithoutClaimingContainerWidth) {
  const Tensor tensor = make_cpu_byte_tensor(64);
  for (const auto encoding : {schema::ImageEncoding_MONO8,
                              schema::ImageEncoding_MONO16,
                              schema::ImageEncoding_BAYER_RGGB,
                              schema::ImageEncoding_CUSTOM}) {
    const schema::ImageMetadataOptions options{
        .height = 2, .width = 3, .encoding = encoding, .significant_bits = 255U};
    auto metadata = schema::build_image_metadata(tensor, options);
    ASSERT_TRUE(metadata) << metadata.error().message;
    const auto* image = schema::GetImage(metadata->data());
    ASSERT_NE(image, nullptr);
    EXPECT_EQ(image->significant_bits(), 255U);
    EXPECT_TRUE(schema::DescriptorFlatBufferTensorViewTraits<schema::Image>::tensor_view_metadata(
        *metadata));
  }
}

namespace {

constexpr std::uint64_t kConsecutivePlaneBytes = 64U;

/// @brief One plane starting where the previous one ended, so a count test never trips the
/// overlap rule it is not testing.
[[nodiscard]] std::vector<schema::ImagePlaneLayout> consecutive_planes(const std::size_t count) {
  std::vector<schema::ImagePlaneLayout> layouts;
  layouts.reserve(count);
  for (std::size_t plane = 0U; plane < count; ++plane) {
    layouts.emplace_back(plane * kConsecutivePlaneBytes, 32U, kConsecutivePlaneBytes);
  }
  return layouts;
}

/// @brief Bytes a Tensor needs to contain \p count consecutive planes, so a count test never trips
/// the containment bound it is not testing either.
[[nodiscard]] constexpr std::int64_t consecutive_planes_span_bytes(const std::size_t count) {
  return static_cast<std::int64_t>(count * kConsecutivePlaneBytes);
}

}  // namespace

// A plane layout is meaningful only against the encoding's plane count, so the two have to agree.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRejectsPlaneLayoutsContradictingTheEncoding) {
  // Large enough to contain the most planes any case below declares, so every rejection here is
  // the count rule this test is about rather than the containment bound.
  const Tensor tensor = make_cpu_byte_tensor(consecutive_planes_span_bytes(4U));
  const auto build = [&tensor](const schema::ImageEncoding encoding, const std::size_t planes) {
    schema::ImageMetadataOptions options{.height = 2, .width = 3, .encoding = encoding};
    options.plane_layouts = consecutive_planes(planes);
    return schema::build_image_metadata(tensor, options);
  };

  // A single-plane encoding has no second plane to describe; the Tensor strides are authoritative.
  auto single_plane = build(schema::ImageEncoding_RGB8, 1U);
  ASSERT_FALSE(single_plane.has_value());
  EXPECT_EQ(single_plane.error().code, ErrorCode::kInvalidArgument);

  // Two-plane NV12 given three layouts names a plane that does not exist.
  auto wrong_count = build(schema::ImageEncoding_NV12, 3U);
  ASSERT_FALSE(wrong_count.has_value());
  EXPECT_EQ(wrong_count.error().code, ErrorCode::kInvalidArgument);

  EXPECT_TRUE(build(schema::ImageEncoding_NV12, 2U).has_value());
  EXPECT_TRUE(build(schema::ImageEncoding_I420, 3U).has_value());
  EXPECT_TRUE(build(schema::ImageEncoding_RGB8, 0U).has_value());
  // A custom encoding names no layout, so no plane count can contradict it.
  EXPECT_TRUE(build(schema::ImageEncoding_CUSTOM, 2U).has_value());
}

// A multi-planar encoding cannot decline to describe its planes. Omitting the vector used to mean
// that the Tensor strides were authoritative instead, which cannot be true of more than one plane:
// one Tensor has one base address, so the offsets the deferral promised were never anywhere.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRequiresPlaneLayoutsForEveryMultiPlanarEncoding) {
  const Tensor tensor = make_cpu_tensor_2x3();
  const auto build_without_layouts = [&tensor](const schema::ImageEncoding encoding) {
    const schema::ImageMetadataOptions options{.height = 2, .width = 3, .encoding = encoding};
    return schema::build_image_metadata(tensor, options);
  };

  for (const schema::ImageEncoding encoding : {schema::ImageEncoding_NV12,
                                               schema::ImageEncoding_I420,
                                               schema::ImageEncoding_P010,
                                               schema::ImageEncoding_R8_G8_B8,
                                               schema::ImageEncoding_B8_G8_R8}) {
    auto result = build_without_layouts(encoding);
    ASSERT_FALSE(result.has_value()) << "encoding " << static_cast<int>(encoding);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  }

  // A single-plane encoding describes its one plane through the Tensor, and a custom encoding
  // states no plane count for an empty vector to contradict. MONO32F because the Tensor here is
  // float32 and a single-plane encoding has to agree with the element type carrying it.
  EXPECT_TRUE(build_without_layouts(schema::ImageEncoding_MONO32F).has_value());
  EXPECT_TRUE(build_without_layouts(schema::ImageEncoding_CUSTOM).has_value());
}

// A plane's packed row width follows from the top-level width, the encoding, and the plane index,
// so a stride that cannot reach the end of a row and an extent that cannot reach the last row are
// both derivable errors rather than something to leave for the consumer to discover by reading past
// the plane.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRejectsPlanesTooSmallForTheEncodedGeometry) {
  // Sized for the padded case at the end, which reserves the most. Every rejection below is then a
  // geometry minimum rather than the containment bound, which the dedicated test covers.
  const Tensor tensor = make_cpu_byte_tensor(1536);
  const auto build =
      [&tensor](
          const schema::ImageEncoding encoding, std::vector<schema::ImagePlaneLayout> layouts) {
        // 6x4 so that every plane of every encoding below has an even extent, keeping this test
        // about the minimums rather than about the rounding rule the next test covers.
        schema::ImageMetadataOptions options{.height = 4, .width = 6, .encoding = encoding};
        options.plane_layouts = std::move(layouts);
        return schema::build_image_metadata(tensor, options);
      };
  const auto rejects =
      [&build](
          const schema::ImageEncoding encoding, std::vector<schema::ImagePlaneLayout> layouts) {
        auto result = build(encoding, std::move(layouts));
        ASSERT_FALSE(result.has_value()) << "encoding=" << static_cast<int>(encoding);
        EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
      };

  // NV12 at 6x4: a 6-byte luma row over 4 rows, then a chroma row of 3 U/V pairs over 2 rows. The
  // exact minimum is a stride of 6 and extents of 3 * 6 + 6 and 1 * 6 + 6.
  EXPECT_TRUE(build(schema::ImageEncoding_NV12,
                    {schema::ImagePlaneLayout{0U, 6U, 24U}, schema::ImagePlaneLayout{24U, 6U, 12U}})
                  .has_value());
  // One byte short of a luma row.
  rejects(schema::ImageEncoding_NV12,
          {schema::ImagePlaneLayout{0U, 5U, 24U}, schema::ImagePlaneLayout{24U, 6U, 12U}});
  // One byte short of the last luma row.
  rejects(schema::ImageEncoding_NV12,
          {schema::ImagePlaneLayout{0U, 6U, 23U}, schema::ImagePlaneLayout{24U, 6U, 12U}});
  // One byte short of a chroma row, which is twice the chroma column count and not the luma width.
  rejects(schema::ImageEncoding_NV12,
          {schema::ImagePlaneLayout{0U, 6U, 24U}, schema::ImagePlaneLayout{24U, 5U, 12U}});
  // One byte short of the last chroma row.
  rejects(schema::ImageEncoding_NV12,
          {schema::ImagePlaneLayout{0U, 6U, 24U}, schema::ImagePlaneLayout{24U, 6U, 11U}});

  // I420 at 6x4 splits the chroma into two planes of 3 bytes over 2 rows each.
  EXPECT_TRUE(build(schema::ImageEncoding_I420,
                    {schema::ImagePlaneLayout{0U, 6U, 24U},
                     schema::ImagePlaneLayout{24U, 3U, 6U},
                     schema::ImagePlaneLayout{30U, 3U, 6U}})
                  .has_value());
  rejects(schema::ImageEncoding_I420,
          {schema::ImagePlaneLayout{0U, 6U, 24U},
           schema::ImagePlaneLayout{24U, 2U, 6U},
           schema::ImagePlaneLayout{30U, 3U, 6U}});

  // P010 carries every sample in a 16-bit container, so each row is twice as wide as NV12's. A
  // layout sized for NV12 is therefore rejected, which is the case a check on the plane count
  // alone could never have caught.
  EXPECT_TRUE(
      build(schema::ImageEncoding_P010,
            {schema::ImagePlaneLayout{0U, 12U, 48U}, schema::ImagePlaneLayout{48U, 12U, 24U}})
          .has_value());
  rejects(schema::ImageEncoding_P010,
          {schema::ImagePlaneLayout{0U, 6U, 24U}, schema::ImagePlaneLayout{24U, 6U, 12U}});

  // Three co-sited full-frame planes, with no subsampling to reason about.
  for (const auto encoding : {schema::ImageEncoding_R8_G8_B8, schema::ImageEncoding_B8_G8_R8}) {
    const std::vector<schema::ImagePlaneLayout> valid{
        {0U, 6U, 24U}, {24U, 6U, 24U}, {48U, 6U, 24U}};
    EXPECT_TRUE(build(encoding, valid).has_value());
    for (std::size_t plane = 0; plane < valid.size(); ++plane) {
      SCOPED_TRACE(plane);
      auto undersized = valid;
      undersized[plane] = schema::ImagePlaneLayout{valid[plane].offset_bytes(), 6U, 23U};
      rejects(encoding, undersized);
      auto short_row = valid;
      short_row[plane] = schema::ImagePlaneLayout{valid[plane].offset_bytes(), 5U, 24U};
      rejects(encoding, short_row);
    }
  }

  // A generously padded layout stays admissible: the minimums are lower bounds, and a producer
  // aligning its rows or reserving trailing padding owes no justification for either.
  EXPECT_TRUE(build(schema::ImageEncoding_NV12,
                    {schema::ImagePlaneLayout{0U, 256U, 1024U},
                     schema::ImagePlaneLayout{1024U, 256U, 512U}})
                  .has_value());
}

// image.fbs fixes the odd-extent rule for a subsampled plane as rounding up, so a 5x3 NV12 frame
// has a chroma plane of three U/V columns over two rows. Validation has to apply the same rule: had
// it rounded down, a producer following the schema would be rejected for declaring the plane that
// describes the last luma column and row, and one that dropped those samples would be accepted.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRoundsSubsampledChromaExtentsUp) {
  const Tensor tensor = make_cpu_byte_tensor(64);
  const auto build = [&tensor](std::vector<schema::ImagePlaneLayout> layouts) {
    schema::ImageMetadataOptions options{
        .height = 3, .width = 5, .encoding = schema::ImageEncoding_NV12};
    options.plane_layouts = std::move(layouts);
    return schema::build_image_metadata(tensor, options);
  };

  // Luma: 5 bytes over 3 rows. Chroma: ceil(5 / 2) = 3 U/V pairs, so 6 bytes, over
  // ceil(3 / 2) = 2 rows.
  EXPECT_TRUE(build({schema::ImagePlaneLayout{0U, 5U, 15U}, schema::ImagePlaneLayout{15U, 6U, 12U}})
                  .has_value());

  // A chroma row sized as though the width had been rounded down to two U/V pairs.
  auto narrow_chroma =
      build({schema::ImagePlaneLayout{0U, 5U, 15U}, schema::ImagePlaneLayout{15U, 4U, 12U}});
  ASSERT_FALSE(narrow_chroma.has_value());
  EXPECT_EQ(narrow_chroma.error().code, ErrorCode::kInvalidArgument);

  // A chroma extent sized as though the height had been rounded down to one row.
  auto short_chroma =
      build({schema::ImagePlaneLayout{0U, 5U, 15U}, schema::ImagePlaneLayout{15U, 6U, 6U}});
  ASSERT_FALSE(short_chroma.has_value());
  EXPECT_EQ(short_chroma.error().code, ErrorCode::kInvalidArgument);
}

// The count rule above is about the encoding. These are about the layout itself, and so hold for
// ImageEncoding_CUSTOM too: a producer-defined layout is still a claim about bytes, and none of
// these states is a claim at all. The receive path copies the vector through verbatim, so a
// consumer that trusted one of them would compute an address from nothing.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRejectsPlaneLayoutsThatDescribeNoMemory) {
  // Wide enough for the gapped pair below, which reaches furthest, so every rejection here is the
  // layout rule it names rather than the containment bound.
  const Tensor tensor = make_cpu_byte_tensor(512);
  const auto build = [&tensor](std::vector<schema::ImagePlaneLayout> layouts) {
    schema::ImageMetadataOptions options{
        .height = 2, .width = 3, .encoding = schema::ImageEncoding_CUSTOM};
    options.plane_layouts = std::move(layouts);
    return schema::build_image_metadata(tensor, options);
  };
  const auto rejects = [&build](std::vector<schema::ImagePlaneLayout> layouts) {
    auto result = build(std::move(layouts));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // A stride of zero advances through no rows, so the plane holds at most one.
  rejects({schema::ImagePlaneLayout{0U, 0U, 64U}});
  // An extent of zero reserves nothing, which no consumer can bound a read against.
  rejects({schema::ImagePlaneLayout{0U, 32U, 0U}});
  // An offset plus an extent that wraps names no range at all.
  rejects({schema::ImagePlaneLayout{std::numeric_limits<std::uint64_t>::max(), 32U, 64U}});
  // Two planes cannot both own the same bytes: a consumer told to write one would corrupt the
  // other, and nothing on the wire says which claim was the intended one.
  rejects({schema::ImagePlaneLayout{0U, 32U, 128U}, schema::ImagePlaneLayout{64U, 32U, 128U}});
  // Overlap is rejected regardless of declaration order, so a producer need not sort its planes.
  rejects({schema::ImagePlaneLayout{64U, 32U, 128U}, schema::ImagePlaneLayout{0U, 32U, 128U}});
  // More planes than the receive path reserves capacity for; a layout that builds must decode.
  rejects(consecutive_planes(schema::kMaximumImagePlanes + 1U));

  // Abutting planes are not overlapping: the second begins exactly where the first ends.
  EXPECT_TRUE(
      build({schema::ImagePlaneLayout{0U, 32U, 64U}, schema::ImagePlaneLayout{64U, 32U, 64U}})
          .has_value());
  // A gap between planes is the producer's to leave, for alignment it does not have to justify.
  EXPECT_TRUE(
      build({schema::ImagePlaneLayout{0U, 32U, 64U}, schema::ImagePlaneLayout{256U, 32U, 64U}})
          .has_value());
  // The bound itself is inclusive.
  EXPECT_TRUE(build(consecutive_planes(schema::kMaximumImagePlanes)).has_value());
}

// image.fbs states containment in the single allocation backing Image.data as a restriction on the
// format, so a plane reaching past the Tensor is a descriptor the schema gives no meaning to rather
// than one a careful consumer works around: the Image carries one Tensor and therefore one base
// address, and no offset from it names a second allocation. The tensor here holds 24 bytes.
//
// Both paths take this verdict from one function, so what this pins for the authoring encoder holds
// for the receive assignment too, and a descriptor one accepts is one the other accepts.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperBoundsPlaneExtentsAgainstTheTensor) {
  const Tensor tensor = make_cpu_byte_tensor(24);
  const auto build = [&tensor](std::vector<schema::ImagePlaneLayout> layouts) {
    schema::ImageMetadataOptions options{
        .height = 2, .width = 3, .encoding = schema::ImageEncoding_CUSTOM};
    options.plane_layouts = std::move(layouts);
    return schema::build_image_metadata(tensor, options);
  };
  const auto rejects = [&build](std::vector<schema::ImagePlaneLayout> layouts) {
    auto result = build(std::move(layouts));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // The last row ends on the Tensor's final byte. An exact fit is inside, not one past: the bound
  // is on the end of the extent, so treating this as a violation would reject the ordinary case of
  // a plane that occupies its allocation exactly.
  EXPECT_TRUE(build({schema::ImagePlaneLayout{0U, 12U, 24U}}).has_value());
  // One byte beyond that end. The case worth pinning, because it is the off-by-one a producer
  // computing its own extents actually makes, and the only difference from the line above.
  rejects({schema::ImagePlaneLayout{0U, 12U, 25U}});
  // A plane beginning past the Tensor entirely, which addresses none of it.
  rejects({schema::ImagePlaneLayout{1024U, 12U, 24U}});
  // Contained on its own, but carried past the end by its offset. Neither field is out of range by
  // itself, so this is the case a bound on either one alone would miss.
  rejects({schema::ImagePlaneLayout{16U, 12U, 16U}});
  // An offset plus an extent that wraps is reported as the overflow it is rather than as a plane
  // outside the Tensor, so the diagnostic still names the arithmetic that failed.
  rejects({schema::ImagePlaneLayout{std::numeric_limits<std::uint64_t>::max(), 12U, 24U}});
  // Every plane is bounded, not just the first: a second plane that leaves the Tensor is rejected
  // even where the first is contained and the two do not overlap.
  rejects({schema::ImagePlaneLayout{0U, 12U, 12U}, schema::ImagePlaneLayout{12U, 12U, 64U}});
}

// The bound is the Tensor's addressable extent and not the product of its extents, and the two
// differ for exactly the layouts plane records exist to describe. A producer that pads its rows
// has a backing wider than its logical size, and bounding against the logical size would reject
// the aligned images the multi-planar encodings advertise.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperBoundsPlaneExtentsAgainstTheStrideAwareTensorSpan) {
  // 3x2 float32 with explicit element strides {4, 1}: 24 logical bytes, but the row padding puts
  // the last addressable byte at (3 - 1) * 16 + (2 - 1) * 4 + 4 = 40.
  const Tensor tensor = make_strided_cpu_tensor_3x2();
  const auto build = [&tensor](std::vector<schema::ImagePlaneLayout> layouts) {
    schema::ImageMetadataOptions options{
        .height = 3, .width = 2, .encoding = schema::ImageEncoding_CUSTOM};
    options.plane_layouts = std::move(layouts);
    return schema::build_image_metadata(tensor, options);
  };

  // Past the 24 logical bytes and inside the 40 addressable ones. Admitted, which is the whole
  // point: a padded backing is bounded by what it can address.
  EXPECT_TRUE(build({schema::ImagePlaneLayout{0U, 16U, 40U}}).has_value());
  // One byte past the addressable extent is still rejected, so the wider bound is a different
  // bound and not an absent one.
  auto beyond = build({schema::ImagePlaneLayout{0U, 16U, 41U}});
  ASSERT_FALSE(beyond.has_value());
  EXPECT_EQ(beyond.error().code, ErrorCode::kInvalidArgument);
}

// A single-plane encoding is required to declare no plane records, so the containment loop above
// never runs for one and its geometry would otherwise never meet the Tensor at all. Where the
// encoding also fixes its bytes per pixel the frame's minimum size follows from the extent alone,
// and a backing too small to hold it makes the descriptor self-contradictory rather than merely
// unusual: a consumer is told the encoding determines the byte organization, so it computes
// addresses from the validated geometry and reaches past the allocation.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperBoundsFixedEncodingFramesAgainstTheTensor) {
  const auto build = [](const Tensor& tensor,
                        const std::int32_t height,
                        const std::int32_t width,
                        const schema::ImageEncoding encoding) {
    return schema::build_image_metadata(
        tensor,
        schema::ImageMetadataOptions{.height = height, .width = width, .encoding = encoding});
  };
  const auto rejects = [&build](const Tensor& tensor,
                                const std::int32_t height,
                                const std::int32_t width,
                                const schema::ImageEncoding encoding) {
    auto result = build(tensor, height, width, encoding);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // The case the bound exists for: three bytes per pixel across 4096x4096 needs 48 MiB, and the
  // descriptor offers one byte of backing to hold it.
  rejects(make_cpu_byte_tensor(1), 4096, 4096, schema::ImageEncoding_RGB8);
  // A frame that occupies its allocation exactly is inside the bound, not one past it.
  EXPECT_TRUE(build(make_cpu_byte_tensor(18), 2, 3, schema::ImageEncoding_RGB8).has_value());
  // One byte short of that, which is the off-by-one a producer sizing its own buffer makes.
  rejects(make_cpu_byte_tensor(17), 2, 3, schema::ImageEncoding_RGB8);
  // The bound is on the minimum: a backing larger than a packed frame needs stays admissible,
  // because row padding is a producer's choice and not a contradiction.
  EXPECT_TRUE(build(make_cpu_byte_tensor(64), 2, 3, schema::ImageEncoding_RGB8).has_value());
  // The component count is part of the derivation and not a constant: the same extent that fits in
  // 18 bytes as RGB8 needs 24 as RGBA8.
  EXPECT_TRUE(build(make_cpu_byte_tensor(24), 2, 3, schema::ImageEncoding_RGBA8).has_value());
  rejects(make_cpu_byte_tensor(23), 2, 3, schema::ImageEncoding_RGBA8);
  // So is the component width, which the floating-point encodings carry at four bytes each.
  EXPECT_TRUE(build(make_cpu_byte_tensor(72), 2, 3, schema::ImageEncoding_RGB32F).has_value());
  rejects(make_cpu_byte_tensor(71), 2, 3, schema::ImageEncoding_RGB32F);
  // Packed 4:2:2 rounds an odd width up to a whole chroma group rather than truncating it, so a
  // three-wide YUYV row is two four-byte groups and not one and a half.
  EXPECT_TRUE(build(make_cpu_byte_tensor(16), 2, 3, schema::ImageEncoding_YUYV).has_value());
  rejects(make_cpu_byte_tensor(15), 2, 3, schema::ImageEncoding_YUYV);
  // An extent whose byte count does not fit a uint64 is the frame being unrepresentable. It has to
  // be reported as exceeding the Tensor rather than wrapping to a small figure that fits inside it.
  rejects(make_cpu_byte_tensor(24),
          std::numeric_limits<std::int32_t>::max(),
          std::numeric_limits<std::int32_t>::max(),
          schema::ImageEncoding_RGBA32F);
}

// A Bayer encoding states no sample width of its own, but the Image carries one: image.fbs defers
// the container to the Tensor element type, so the width is stated somewhere and the frame bound
// can be reached through it. Resolving it is what keeps the deferral from behaving like an absence
// of any claim, which is how a 16-mebibyte mosaic in a one-byte allocation used to be admitted.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperResolvesDeferredSampleWidthsFromTheTensor) {
  const auto build = [](const Tensor& tensor,
                        const std::int32_t height,
                        const std::int32_t width,
                        const schema::ImageEncoding encoding) {
    return schema::build_image_metadata(
        tensor,
        schema::ImageMetadataOptions{.height = height, .width = width, .encoding = encoding});
  };
  const auto rejects = [&build](const Tensor& tensor,
                                const std::int32_t height,
                                const std::int32_t width,
                                const schema::ImageEncoding encoding) {
    auto result = build(tensor, height, width, encoding);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // One byte per pixel once the element type is read, so a 4096x4096 mosaic needs 16 MiB and gets
  // one byte.
  rejects(make_cpu_byte_tensor(1), 4096, 4096, schema::ImageEncoding_BAYER_RGGB);
  EXPECT_TRUE(build(make_cpu_byte_tensor(6), 2, 3, schema::ImageEncoding_BAYER_RGGB).has_value());
  rejects(make_cpu_byte_tensor(5), 2, 3, schema::ImageEncoding_BAYER_GBRG);
  // The same extent against a wider element type needs twice the bytes, which is the whole point of
  // resolving the width rather than assuming one.
  EXPECT_TRUE(
      build(make_cpu_uint16_tensor(12), 2, 3, schema::ImageEncoding_BAYER_BGGR).has_value());
  rejects(make_cpu_uint16_tensor(10), 2, 3, schema::ImageEncoding_BAYER_GRBG);

  // Each MONO entry names its own width, so none of them defers and all of them are bounded. This
  // is the half of the schema text that was wrong: MONO8 and MONO16 differ by exactly the width
  // they were said to leave to the Tensor. Every width is listed rather than a representative one,
  // because the bound is derived per entry and a new entry added to the wrong group would otherwise
  // be unbounded silently.
  EXPECT_TRUE(build(make_cpu_byte_tensor(6), 2, 3, schema::ImageEncoding_MONO8).has_value());
  rejects(make_cpu_byte_tensor(5), 2, 3, schema::ImageEncoding_MONO8);
  EXPECT_TRUE(build(make_cpu_byte_tensor(12), 2, 3, schema::ImageEncoding_MONO16).has_value());
  rejects(make_cpu_byte_tensor(11), 2, 3, schema::ImageEncoding_MONO16);
  // The two 32-bit entries carry the same width and differ only in how a component is interpreted,
  // so they reach an identical bound here. Which element types may carry them is a separate rule
  // with its own coverage; a byte buffer is the opaque carriage both admit.
  EXPECT_TRUE(build(make_cpu_byte_tensor(24), 2, 3, schema::ImageEncoding_MONO32).has_value());
  rejects(make_cpu_byte_tensor(23), 2, 3, schema::ImageEncoding_MONO32);
  EXPECT_TRUE(build(make_cpu_byte_tensor(24), 2, 3, schema::ImageEncoding_MONO32F).has_value());
  rejects(make_cpu_byte_tensor(23), 2, 3, schema::ImageEncoding_MONO32F);

  // CUSTOM is the one encoding left unbounded, because it is the one that genuinely states no
  // organization for a bound to be derived from. Pinned so that stays a deliberate scope.
  EXPECT_TRUE(build(make_cpu_byte_tensor(1), 4096, 4096, schema::ImageEncoding_CUSTOM).has_value());
}

// Two carriages are admissible for an encoding that names its own element type, and they are the
// two a producer has: the typed view a consumer indexes directly, and the opaque byte buffer it
// casts. An element type that is neither states the sample organization twice and states it
// differently, and nothing downstream can decide which of the two claims to believe.
// REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRejectsElementTypesContradictingTheEncoding) {
  const auto build = [](const Tensor& tensor, const schema::ImageEncoding encoding) {
    return schema::build_image_metadata(
        tensor, schema::ImageMetadataOptions{.height = 2, .width = 3, .encoding = encoding});
  };
  const auto rejects = [&build](const Tensor& tensor, const schema::ImageEncoding encoding) {
    auto result = build(tensor, encoding);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // The contradiction from the review: MONO16 is one unsigned 16-bit component and the Tensor
  // carrying it is float32.
  rejects(make_cpu_tensor_2x3(), schema::ImageEncoding_MONO16);
  // The typed carriage of the same encoding.
  EXPECT_TRUE(build(make_cpu_uint16_tensor(12), schema::ImageEncoding_MONO16).has_value());
  // The byte carriage of it, where the Tensor claims a length and nothing about the samples.
  EXPECT_TRUE(build(make_cpu_byte_tensor(12), schema::ImageEncoding_MONO16).has_value());
  // The converse pairing, so the rule is agreement between the two and not a preference for
  // integers: a float encoding rejects the unsigned element type that MONO16 requires.
  rejects(make_cpu_uint16_tensor(24), schema::ImageEncoding_MONO32F);
  EXPECT_TRUE(build(make_cpu_tensor_2x3(), schema::ImageEncoding_MONO32F).has_value());

  // A Bayer encoding defers its container, so no element type can contradict it and both widths
  // are admissible.
  EXPECT_TRUE(build(make_cpu_byte_tensor(6), schema::ImageEncoding_BAYER_RGGB).has_value());
  EXPECT_TRUE(build(make_cpu_uint16_tensor(12), schema::ImageEncoding_BAYER_RGGB).has_value());
  // CUSTOM states no organization at all.
  EXPECT_TRUE(build(make_cpu_tensor_2x3(), schema::ImageEncoding_CUSTOM).has_value());

  // A multi-planar encoding is exempt, because for it the Tensor is a byte-addressed backing that
  // plane_layouts carves up rather than a typed view of the samples. Accepting a float32 Tensor
  // here is what lets P010's 16-bit samples travel in a flat buffer at all.
  schema::ImageMetadataOptions p010{
      .height = 2, .width = 3, .encoding = schema::ImageEncoding_P010};
  p010.plane_layouts = {schema::ImagePlaneLayout{0U, 6U, 12U},
                        schema::ImagePlaneLayout{12U, 8U, 8U}};
  EXPECT_TRUE(schema::build_image_metadata(make_cpu_tensor_2x3(), p010).has_value());
}

// The same contradiction reached through the lane count instead of the type. Every encoding but
// CUSTOM already fixes the component organization of a sample, so a vectorized element states it a
// second time and nothing in the descriptor relates the two: a four-lane uint8 element labelled
// RGB8 leaves both the frame width and the meaning of those lanes undecided. Channels are a
// trailing shape axis in this format, which is the form the frame and plane bounds are derived
// against. REQ: SENSOR-IO-008
TEST(Schema, ImageMetadataHelperRejectsVectorizedElementsForOrganizedEncodings) {
  const auto build = [](const Tensor& tensor, const schema::ImageEncoding encoding) {
    return schema::build_image_metadata(
        tensor, schema::ImageMetadataOptions{.height = 2, .width = 3, .encoding = encoding});
  };
  const auto rejects = [&build](const Tensor& tensor, const schema::ImageEncoding encoding) {
    auto result = build(tensor, encoding);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };

  // Six four-lane elements span 24 bytes and a 2x3 RGB8 frame needs 18, so the frame fits and the
  // lane count is what the refusal is about.
  rejects(make_cpu_lane_tensor(6, 4), schema::ImageEncoding_RGB8);
  // The scalar element of the same width is the admissible carriage.
  EXPECT_TRUE(build(make_cpu_lane_tensor(24, 1), schema::ImageEncoding_RGB8).has_value());

  // An encoding that defers its sample width defers it to a container, and a vector of containers
  // is not one, so the deferral has no single answer to read.
  rejects(make_cpu_lane_tensor(6, 2), schema::ImageEncoding_BAYER_RGGB);
  // A multi-planar encoding states organization too, even though no element type contradicts it.
  schema::ImageMetadataOptions nv12{
      .height = 2, .width = 3, .encoding = schema::ImageEncoding_NV12};
  nv12.plane_layouts = {schema::ImagePlaneLayout{0U, 3U, 6U}, schema::ImagePlaneLayout{6U, 4U, 4U}};
  auto vectorized_nv12 = schema::build_image_metadata(make_cpu_lane_tensor(6, 4), nv12);
  ASSERT_FALSE(vectorized_nv12.has_value());
  EXPECT_EQ(vectorized_nv12.error().code, ErrorCode::kInvalidArgument);

  // CUSTOM states no organization for a lane count to contradict, and keeps the latitude it already
  // has over plane count and frame extent.
  EXPECT_TRUE(build(make_cpu_lane_tensor(6, 4), schema::ImageEncoding_CUSTOM).has_value());

  // Bytes were never what was at stake. A vectorized element counts at its full width, so six
  // four-lane elements address 24 bytes and not six. Read through CUSTOM's plane records, the one
  // place a lane count survives to reach a bound at all.
  schema::ImageMetadataOptions custom_plane{
      .height = 2, .width = 3, .encoding = schema::ImageEncoding_CUSTOM};
  custom_plane.plane_layouts = {schema::ImagePlaneLayout{0U, 12U, 24U}};
  EXPECT_TRUE(schema::build_image_metadata(make_cpu_lane_tensor(6, 4), custom_plane).has_value());
  custom_plane.plane_layouts = {schema::ImagePlaneLayout{0U, 12U, 25U}};
  EXPECT_FALSE(schema::build_image_metadata(make_cpu_lane_tensor(6, 4), custom_plane).has_value());
}

namespace {

using ReceivedImagePayload = schema::DescriptorFlatBufferPayload<
    schema::Image, schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>;

/// @brief Wire layout the descriptor-backed Image port type ops encode into.
struct ReceivedImageEnvelope {
  SampleMetadata metadata{};
  ReceivedImagePayload payload{};
};

/// @brief Image fields a forged peer message sets, defaulted to values the schema admits.
///
/// MONO32F because every message here is carried by the 2x3 float32 Tensor below, and a known
/// encoding has to agree with the element type carrying it. A default that did not would make each
/// case below fail on the element type before reaching the field it is about.
struct ForgedImageFields {
  std::int32_t height = 2;
  std::int32_t width = 3;
  schema::ImageEncoding encoding = schema::ImageEncoding_MONO32F;
  schema::ColorSpace color_space = schema::ColorSpace_SRGB;
  std::uint8_t significant_bits = 0;
  std::int32_t roi_offset_x = 0;
  std::int32_t roi_offset_y = 0;
  std::int64_t exposure_time_ns = 0;
  float gain = 1.0F;
  std::vector<schema::ImagePlaneLayout> plane_layouts;
};

/// @brief Serialize one Image root the way a producer outside this SDK would.
///
/// build_image_metadata refuses every value these receive-path tests are about, which is the point
/// of the authoring checks and the reason the forgery cannot go through it. CreateImage writes
/// whatever it is handed, so this stands in for a peer that never ran the encoder.
[[nodiscard]] std::vector<std::byte> forge_image_metadata(const Tensor& tensor,
                                                          const ForgedImageFields& fields) {
  flatbuffers::FlatBufferBuilder builder;
  auto tensor_metadata = schema::create_tensor_metadata(builder, tensor);
  EXPECT_TRUE(tensor_metadata);
  const auto tensor_offset =
      tensor_metadata ? *tensor_metadata : flatbuffers::Offset<schema::Tensor>{};
  const auto plane_layouts =
      fields.plane_layouts.empty()
          ? flatbuffers::Offset<flatbuffers::Vector<const schema::ImagePlaneLayout*>>{}
          : builder.CreateVectorOfStructs(fields.plane_layouts);
  schema::FinishImageBuffer(builder,
                            schema::CreateImage(builder,
                                                /*header=*/0,
                                                fields.height,
                                                fields.width,
                                                fields.encoding,
                                                tensor_offset,
                                                fields.color_space,
                                                fields.significant_bits,
                                                fields.roi_offset_x,
                                                fields.roi_offset_y,
                                                plane_layouts,
                                                fields.exposure_time_ns,
                                                fields.gain));
  const auto* bytes = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  return {bytes, bytes + builder.GetSize()};
}

/// @brief Replace an encoded envelope's metadata region with peer-supplied Image bytes.
///
/// The surrounding envelope stays exactly as the local encoder wrote it -- same schema identity,
/// same endpoint generation, same sample metadata, same Tensor descriptor -- so the only thing
/// under test is what the assignment hook does with a verifier-valid root it did not produce.
void splice_forged_metadata(const std::span<std::byte> wire,
                            const std::span<const std::byte> forged) {
  ReceivedImageEnvelope envelope{};
  ASSERT_EQ(wire.size(), sizeof(envelope));
  std::memcpy(&envelope, wire.data(), sizeof(envelope));
  ASSERT_LE(forged.size(), envelope.payload.metadata.size());
  envelope.payload.metadata.fill(std::byte{});
  std::memcpy(envelope.payload.metadata.data(), forged.data(), forged.size());
  envelope.payload.metadata_size = static_cast<std::uint32_t>(forged.size());
  std::memcpy(wire.data(), &envelope, sizeof(envelope));
}

}  // namespace

// REQ: SENSOR-IO-010
TEST(Schema, ImageRGB16AndRGBA16ValidateAndRoundTrip) {
  const auto* ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(ops, nullptr);
  for (const auto encoding : {schema::ImageEncoding_RGB16, schema::ImageEncoding_RGBA16}) {
    SCOPED_TRACE(schema::EnumNameImageEncoding(encoding));
    const std::int64_t frame_bytes = encoding == schema::ImageEncoding_RGB16 ? 36 : 48;
    for (const auto& tensor :
         {make_cpu_uint16_tensor(frame_bytes), make_cpu_byte_tensor(frame_bytes)}) {
      SCOPED_TRACE(tensor.dtype().bits);
      for (const std::uint8_t precision : {0, 12, 16, 255}) {
        SCOPED_TRACE(precision);
        schema::ImageMetadataOptions options{
            .height = 2, .width = 3, .encoding = encoding, .significant_bits = precision};
        const auto built = schema::build_image_metadata(tensor, options);
        ASSERT_TRUE(built) << runtime_error_message(built.error());
        const auto parsed = schema::parse_image_metadata(*built);
        ASSERT_TRUE(parsed);
        EXPECT_EQ(parsed->image->encoding(), encoding);
        EXPECT_EQ(parsed->image->significant_bits(), precision);

        schema::ImageT source{};
        source.height = 2;
        source.width = 3;
        source.encoding = encoding;
        source.significant_bits = precision;
        source.data = std::make_shared<Tensor>(tensor);
        const SampleMetadata metadata{.sequence = 11U,
                                      .byte_size = static_cast<std::uint64_t>(frame_bytes)};
        alignas(std::max_align_t) std::array<std::byte, sizeof(ReceivedImageEnvelope)> wire{};
        const auto encoded = ops->encode_descriptor_tensor_sample(&source, metadata, 1U, wire);
        ASSERT_TRUE(encoded) << runtime_error_message(encoded.error());
        alignas(Sample<schema::ImageT>) std::array<std::byte, sizeof(Sample<schema::ImageT>)>
            storage{};
        ASSERT_TRUE(ops->prepare_sample_storage(storage.data()));
        const auto assigned =
            ops->assign_descriptor_tensor_sample(storage.data(),
                                                 wire,
                                                 tensor,
                                                 detail::DescriptorTensorWireMetadata{metadata},
                                                 detail::DescriptorTensorSampleMetadata{metadata});
        EXPECT_TRUE(assigned);
        const auto* sample = reinterpret_cast<const Sample<schema::ImageT>*>(storage.data());
        EXPECT_EQ(sample->data.encoding, encoding);
        EXPECT_EQ(sample->data.significant_bits, precision);
        EXPECT_EQ(sample->data.height, 2);
        EXPECT_EQ(sample->data.width, 3);
        if (sample->data.data) {
          EXPECT_EQ(sample->data.data->dtype().bits, tensor.dtype().bits);
        } else {
          ADD_FAILURE() << "Received image has no Tensor";
        }
        ops->destroy_sample(storage.data());
      }
    }
  }
}

// REQ: SENSOR-IO-010
TEST(Schema, ImageRGB16AndRGBA16RejectInvalidStorageAndMetadata) {
  const auto* ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(ops, nullptr);
  for (const auto encoding : {schema::ImageEncoding_RGB16, schema::ImageEncoding_RGBA16}) {
    SCOPED_TRACE(schema::EnumNameImageEncoding(encoding));
    const std::int64_t frame_bytes = encoding == schema::ImageEncoding_RGB16 ? 36 : 48;
    const auto rejects = [&](const Tensor& tensor, const schema::ImageMetadataOptions& options) {
      const auto built = schema::build_image_metadata(tensor, options);
      ASSERT_FALSE(built);
      EXPECT_EQ(built.error().code, ErrorCode::kInvalidArgument);

      // A peer may bypass the authoring validator, so exercise the same facts on receive.
      schema::ImageT source{};
      source.height = 1;
      source.width = 1;
      source.encoding = schema::ImageEncoding_CUSTOM;
      source.data = std::make_shared<Tensor>(tensor);
      const SampleMetadata metadata{.sequence = 11U};
      alignas(std::max_align_t) std::array<std::byte, sizeof(ReceivedImageEnvelope)> wire{};
      ASSERT_TRUE(ops->encode_descriptor_tensor_sample(&source, metadata, 1U, wire));
      splice_forged_metadata(wire,
                             forge_image_metadata(tensor,
                                                  {.height = options.height,
                                                   .width = options.width,
                                                   .encoding = options.encoding,
                                                   .significant_bits = options.significant_bits,
                                                   .plane_layouts = options.plane_layouts}));
      alignas(Sample<schema::ImageT>) std::array<std::byte, sizeof(Sample<schema::ImageT>)>
          storage{};
      ASSERT_TRUE(ops->prepare_sample_storage(storage.data()));
      const auto assigned =
          ops->assign_descriptor_tensor_sample(storage.data(),
                                               wire,
                                               tensor,
                                               detail::DescriptorTensorWireMetadata{metadata},
                                               detail::DescriptorTensorSampleMetadata{metadata});
      EXPECT_FALSE(assigned);
      if (!assigned) {
        EXPECT_EQ(assigned.error().code, ErrorCode::kProtocolError);
      }
      const auto* sample = reinterpret_cast<const Sample<schema::ImageT>*>(storage.data());
      EXPECT_EQ(sample->data.height, 0);
      EXPECT_EQ(sample->metadata.sequence, 0U);
      ops->destroy_sample(storage.data());
    };
    const schema::ImageMetadataOptions options{.height = 2, .width = 3, .encoding = encoding};
    rejects(make_cpu_byte_tensor(frame_bytes - 1), options);
    rejects(make_cpu_uint16_tensor(frame_bytes - 2), options);
    rejects(make_cpu_tensor_2x3(), {.height = 1, .width = 1, .encoding = encoding});
    rejects(make_cpu_lane_tensor(frame_bytes, 2), options);
    auto invalid = options;
    invalid.significant_bits = 17;
    rejects(make_cpu_uint16_tensor(frame_bytes), invalid);
    rejects(make_cpu_byte_tensor(frame_bytes), invalid);
    invalid = options;
    invalid.plane_layouts = {schema::ImagePlaneLayout{
        0U, static_cast<std::uint32_t>(frame_bytes / 2), static_cast<std::uint64_t>(frame_bytes)}};
    rejects(make_cpu_uint16_tensor(frame_bytes), invalid);
  }
}

// FlatBuffers verification proves an envelope is structurally an Image and says nothing about
// whether its values mean anything, so the receive path has to take the same semantic verdict the
// encoder takes. Each case below is a state a Channel or external producer can put on the wire and
// no consumer can interpret, and each has to be refused before the sample slot is touched: a
// half-updated slot is one a consumer reads as a measurement.
// REQ: SENSOR-IO-010
TEST(Schema, ImageReceivePathRejectsDescriptorFieldsTheSchemaGivesNoMeaning) {
  const detail::RuntimePortTypeOps* const ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(ops, nullptr);

  schema::ImageT source{};
  source.height = 2;
  source.width = 3;
  source.encoding = schema::ImageEncoding_MONO32F;
  source.data = std::make_shared<Tensor>(make_cpu_tensor_2x3());
  const SampleMetadata metadata{.sequence = 11U, .byte_size = 24U};
  alignas(std::max_align_t) std::array<std::byte, sizeof(ReceivedImageEnvelope)> wire{};
  auto encoded = ops->encode_descriptor_tensor_sample(&source, metadata, 1U, wire);
  ASSERT_TRUE(encoded) << runtime_error_message(encoded.error());

  alignas(Sample<schema::ImageT>) std::array<std::byte, sizeof(Sample<schema::ImageT>)> storage{};
  auto prepared = ops->prepare_sample_storage(storage.data());
  ASSERT_TRUE(prepared) << runtime_error_message(prepared.error());
  auto* sample = reinterpret_cast<Sample<schema::ImageT>*>(storage.data());

  const auto rejects = [&](const ForgedImageFields& fields) {
    alignas(std::max_align_t) std::array<std::byte, sizeof(ReceivedImageEnvelope)> forged_wire{};
    std::memcpy(forged_wire.data(), wire.data(), wire.size());
    splice_forged_metadata(forged_wire, forge_image_metadata(make_cpu_tensor_2x3(), fields));
    // The wire metadata is the envelope's own, so a rejection below is the semantic Image verdict
    // and not the envelope-match check refusing metadata the encoder never wrote.
    auto assigned =
        ops->assign_descriptor_tensor_sample(storage.data(),
                                             forged_wire,
                                             make_cpu_tensor_2x3(),
                                             detail::DescriptorTensorWireMetadata{metadata},
                                             detail::DescriptorTensorSampleMetadata{metadata});
    ASSERT_FALSE(assigned);
    EXPECT_EQ(assigned.error().code, ErrorCode::kProtocolError);
    // Key assertion: the slot still holds what reset left in it, so nothing partial is observable.
    EXPECT_EQ(sample->data.height, 0);
    EXPECT_EQ(sample->data.width, 0);
    EXPECT_EQ(sample->metadata.sequence, 0U);
  };

  rejects({.height = -2});
  rejects({.width = 0});
  rejects({.encoding = static_cast<schema::ImageEncoding>(schema::ImageEncoding_MAX + 1)});
  rejects({.color_space = static_cast<schema::ColorSpace>(schema::ColorSpace_MAX + 1)});
  rejects({.roi_offset_x = -1});
  rejects({.roi_offset_y = -1});
  rejects({.exposure_time_ns = -1});
  rejects({.gain = -1.0F});
  rejects({.gain = std::numeric_limits<float>::quiet_NaN()});
  rejects({.gain = std::numeric_limits<float>::infinity()});
  // A multi-planar encoding whose planes are described nowhere.
  rejects({.encoding = schema::ImageEncoding_NV12});
  // A fixed-organization single-plane encoding whose extent needs more bytes than the Tensor the
  // envelope carries. The peer declares no planes, so this is refused by the frame bound rather
  // than by plane containment, and refusing it here is what makes that bound a property of the
  // wire and not just of the local encoder.
  rejects({.height = 64, .width = 64, .encoding = schema::ImageEncoding_RGB8});
  // A known encoding contradicting the element type of the Tensor this message resolved to. The
  // peer states the sample organization twice and states it differently, and the receive path is
  // the only place that can be caught for a producer that never ran the local encoder.
  rejects({.encoding = schema::ImageEncoding_MONO16});
  // A Bayer encoding takes its container from that same element type, so what it cannot do is
  // report more precision than the container holds. Four bytes per pixel here covers the frame, so
  // the refusal is the precision and not the extent.
  rejects({.encoding = schema::ImageEncoding_BAYER_RGGB, .significant_bits = 33U});
  // A single-plane encoding carrying a layout that contradicts the Tensor it defers to. Kept inside
  // the 24-byte Tensor so the rejection is that contradiction and not the containment bound.
  rejects({.encoding = schema::ImageEncoding_MONO32F,
           .plane_layouts = {schema::ImagePlaneLayout{0U, 12U, 24U}}});
  // A luma stride that cannot reach the end of one 3-byte row, likewise contained so that the
  // stride is what fails.
  rejects({.encoding = schema::ImageEncoding_NV12,
           .plane_layouts = {schema::ImagePlaneLayout{0U, 2U, 6U},
                             schema::ImagePlaneLayout{6U, 4U, 4U}}});
  // A plane reaching one byte past the Tensor this message resolved to. The receive path is where
  // containment earns its keep: a peer that computed its offsets against a different allocation
  // than the one it sent cannot be caught any later than here, and the sample would otherwise
  // carry an unreadable range to a consumer looking exactly like a measurement.
  rejects({.encoding = schema::ImageEncoding_CUSTOM,
           .plane_layouts = {schema::ImagePlaneLayout{0U, 12U, 25U}}});
  // More planes than the slot reserved capacity for, which the assignment loop must never reach.
  rejects({.encoding = schema::ImageEncoding_CUSTOM,
           .plane_layouts = consecutive_planes(schema::kMaximumImagePlanes + 1U)});

  // The same forged envelope with admissible fields is accepted, so the rejections above are the
  // semantic checks and not an artifact of splicing the metadata region.
  alignas(std::max_align_t) std::array<std::byte, sizeof(ReceivedImageEnvelope)> accepted_wire{};
  std::memcpy(accepted_wire.data(), wire.data(), wire.size());
  splice_forged_metadata(accepted_wire,
                         forge_image_metadata(make_cpu_tensor_2x3(), ForgedImageFields{}));
  auto assigned =
      ops->assign_descriptor_tensor_sample(storage.data(),
                                           accepted_wire,
                                           make_cpu_tensor_2x3(),
                                           detail::DescriptorTensorWireMetadata{metadata},
                                           detail::DescriptorTensorSampleMetadata{metadata});
  ASSERT_TRUE(assigned) << runtime_error_message(assigned.error());
  EXPECT_EQ(sample->data.height, 2);
  EXPECT_EQ(sample->data.width, 3);
  EXPECT_EQ(sample->data.color_space, schema::ColorSpace_SRGB);
  EXPECT_FLOAT_EQ(sample->data.gain, 1.0F);
  EXPECT_EQ(sample->metadata.sequence, 11U);
  ops->destroy_sample(storage.data());
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
