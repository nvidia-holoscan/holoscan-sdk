/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/tensor_materialization.hpp>

#include "descriptor_frame_multi_order_traits.hpp"

namespace {

using Native = app3p::MultiDescriptorFrameT;
using Root = app3p::MultiDescriptorFrame;
using Payload = holoscan::schema::DescriptorFlatBufferPayloadFor<Root>;
static_assert(Payload::metadata_capacity ==
              holoscan::DescriptorTensorSchemaNativeTraits<Native>::metadata_capacity);
static_assert(Payload::max_tensors == 3U);

struct Envelope {
  holoscan::SampleMetadata metadata{};
  Payload payload{};
};

holoscan::Tensor make_first_tensor() {
  static std::vector<float> data{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {2, 3};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  return holoscan::Tensor{std::move(context)};
}

holoscan::Tensor make_second_tensor() {
  static std::vector<std::uint8_t> data{3U, 5U, 8U, 13U};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {4};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 8, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  return holoscan::Tensor{std::move(context)};
}

holoscan::Tensor make_third_tensor() {
  static std::vector<std::int16_t> data{21, 34};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {1, 2};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLInt, .bits = 16, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  return holoscan::Tensor{std::move(context)};
}

Native make_native() {
  Native value{};
  value.sequence = 42U;
  value.label = "three descriptor fields";
  value.exposure = 2.5F;
  value.header = std::make_unique<app3p::FrameHeaderT>();
  value.header->source = "stereo-camera";
  value.header->frame_id = 17U;
  value.first = std::make_shared<holoscan::Tensor>(make_first_tensor());
  value.second = std::make_shared<holoscan::Tensor>(make_second_tensor());
  value.third = std::make_shared<holoscan::Tensor>(make_third_tensor());
  return value;
}

std::vector<std::byte> hand_encode(const Native& value) {
  flatbuffers::FlatBufferBuilder builder;
  auto first = holoscan::schema::create_tensor_metadata(builder, *value.first);
  auto second = holoscan::schema::create_tensor_metadata(builder, *value.second);
  auto third = holoscan::schema::create_tensor_metadata(builder, *value.third);
  if (!first || !second || !third) {
    return {};
  }
  const auto label = builder.CreateString(value.label);
  const auto header_source = builder.CreateString(value.header->source);
  const auto header = app3p::CreateFrameHeader(builder, header_source, value.header->frame_id);
  app3p::MultiDescriptorFrameBuilder root_builder{builder};
  root_builder.add_sequence(value.sequence);
  root_builder.add_first(*first);
  root_builder.add_label(label);
  root_builder.add_second(*second);
  root_builder.add_exposure(value.exposure);
  root_builder.add_third(*third);
  root_builder.add_header(header);
  app3p::FinishMultiDescriptorFrameBuffer(builder, root_builder.Finish());
  const auto bytes =
      std::as_bytes(std::span<const std::uint8_t>{builder.GetBufferPointer(), builder.GetSize()});
  return {bytes.begin(), bytes.end()};
}

holoscan::PayloadDescriptor make_descriptor(const std::uint64_t token) {
  return holoscan::PayloadDescriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor,
      .schema_id = Payload::expected_schema_family_id,
      .local =
          holoscan::LocalPayloadReference{
              .process_id = 1U,
              .token = token,
              .generation = 1U,
              .expected_recipient_count = 1U,
          },
  };
}

bool metadata_has_no_data(const std::span<const std::byte> bytes) {
  const Root* root = Payload::root_from_metadata(bytes);
  if (root == nullptr || root->first() == nullptr || root->second() == nullptr ||
      root->third() == nullptr) {
    return false;
  }
  const auto data_absent = [](const holoscan::schema::Tensor* tensor) {
    return tensor->data() == nullptr || tensor->data()->empty();
  };
  return data_absent(root->first()) && data_absent(root->second()) && data_absent(root->third());
}

}  // namespace

// REQ: CHAN-145
// REQ: OPERATOR-035
int main() {
  // Key setup: encode three generated Tensor fields in canonical field-id order through the
  // generic Runtime operations and through an equivalent hand-authored FlatBuffers builder.
  const holoscan::detail::RuntimePortTypeOps* const ops =
      holoscan::detail::runtime_port_type_ops<Native>();
  if (ops == nullptr || ops->descriptor_tensor_count == nullptr ||
      ops->descriptor_tensor_at == nullptr || ops->descriptor_tensor_view_metadata_at == nullptr ||
      ops->assign_descriptor_tensors_sample == nullptr) {
    return 1;
  }
  Native source = make_native();
  if (ops->descriptor_tensor(&source) != nullptr || ops->descriptor_tensor_count(&source) != 3U ||
      ops->descriptor_tensor_at(&source, 0U) != source.first.get() ||
      ops->descriptor_tensor_at(&source, 1U) != source.second.get() ||
      ops->descriptor_tensor_at(&source, 2U) != source.third.get() ||
      ops->descriptor_tensor_at(&source, 3U) != nullptr) {
    return 2;
  }

  const holoscan::SampleMetadata metadata{
      .sequence = 7U,
      .byte_size = static_cast<std::uint64_t>(source.first->nbytes() + source.second->nbytes() +
                                              source.third->nbytes()),
      .payload_kind = holoscan::PayloadKind::kDescriptorBacked,
  };
  constexpr std::uint64_t kEndpointGeneration = 7U;
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> wire{};
  auto encoded = ops->encode_descriptor_tensor_sample(&source, metadata, kEndpointGeneration, wire);
  if (!encoded || *encoded == 0U || *encoded >= wire.size()) {
    return 3;
  }
  const auto wire_prefix = std::span<const std::byte>{wire}.first(*encoded);
  Envelope envelope{};
  std::memcpy(&envelope, wire.data(), wire.size());
  const std::vector<std::byte> expected = hand_encode(source);

  // Key assertion: generic bytes equal the hand encoding, all three Tensor data vectors remain
  // absent even under ambient materialization, and receive assignment commits all fields at once.
  if (expected.size() != envelope.payload.bytes().size() ||
      !std::equal(expected.begin(), expected.end(), envelope.payload.bytes().begin()) ||
      !metadata_has_no_data(envelope.payload.bytes())) {
    return 4;
  }

  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> ambient_wire{};
  {
    const holoscan::schema::WithTensorMaterialization ambient_scope;
    auto ambient = ops->encode_descriptor_tensor_sample(
        &source, metadata, kEndpointGeneration, std::span<std::byte>{ambient_wire});
    if (!ambient || *ambient != *encoded) {
      return 5;
    }
  }
  Envelope ambient_envelope{};
  std::memcpy(&ambient_envelope, ambient_wire.data(), ambient_wire.size());
  if (ambient_envelope.payload.bytes().size() != envelope.payload.bytes().size() ||
      !std::equal(ambient_envelope.payload.bytes().begin(),
                  ambient_envelope.payload.bytes().end(),
                  envelope.payload.bytes().begin()) ||
      !metadata_has_no_data(ambient_envelope.payload.bytes())) {
    return 6;
  }

  Envelope carried = envelope;
  for (std::size_t index = 0U; index < 3U; ++index) {
    *carried.payload.descriptor_at(index) = make_descriptor(index + 1U);
  }
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> carried_wire{};
  std::memcpy(carried_wire.data(), &carried, carried_wire.size());
  const auto carried_prefix = std::span<const std::byte>{carried_wire}.first(*encoded);
  for (std::uint32_t index = 0U; index < 3U; ++index) {
    if (!ops->descriptor_tensor_view_metadata_at(carried_prefix, index, kEndpointGeneration)) {
      return 7;
    }
  }
  if (ops->descriptor_tensor_view_metadata_at(carried_prefix, 3U, kEndpointGeneration) ||
      ops->descriptor_tensor_view_metadata(carried_prefix, kEndpointGeneration)) {
    return 8;
  }

  alignas(holoscan::Sample<Native>) std::array<std::byte, sizeof(holoscan::Sample<Native>)>
      storage{};
  if (!ops->prepare_sample_storage(storage.data())) {
    return 9;
  }
  std::array<holoscan::Tensor, 3U> tensors{
      *source.first,
      *source.second,
      *source.third,
  };
  auto assigned = ops->assign_descriptor_tensors_sample(
      storage.data(),
      wire_prefix,
      tensors,
      holoscan::detail::DescriptorTensorWireMetadata{metadata},
      holoscan::detail::DescriptorTensorSampleMetadata{metadata});
  auto* sample = reinterpret_cast<holoscan::Sample<Native>*>(storage.data());
  const bool assigned_all =
      assigned && sample->data.sequence == source.sequence && sample->data.label == source.label &&
      sample->data.exposure == source.exposure && sample->data.header != nullptr &&
      sample->data.header->source == source.header->source &&
      sample->data.header->frame_id == source.header->frame_id && sample->data.first != nullptr &&
      sample->data.second != nullptr && sample->data.third != nullptr &&
      sample->data.first->data() == source.first->data() &&
      sample->data.second->data() == source.second->data() &&
      sample->data.third->data() == source.third->data() &&
      sample->metadata.sequence == metadata.sequence &&
      sample->metadata.byte_size == metadata.byte_size &&
      sample->metadata.payload_kind == metadata.payload_kind;
  ops->destroy_sample(storage.data());
  if (!assigned_all) {
    return 10;
  }

  if (!ops->descriptor_tensor_sample_metadata(carried_prefix, 0b111U, kEndpointGeneration) ||
      ops->descriptor_tensor_sample_metadata(carried_prefix, 0b001U, kEndpointGeneration)) {
    return 11;
  }
  return 0;
}
