/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <span>

#include <flatbuffers/flatbuffer_builder.h>

#include <holoscan/schema/flatbuffer_schema_package.hpp>

namespace holoscan::detail {
namespace {

struct TensorPacketCodec {
  [[nodiscard]] expected<void, Error> prepare(schema::TensorPacketT& packet) const {
    packet.source.reserve(
        schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>);
    return {};
  }

  void reset(schema::TensorPacketT& packet) const noexcept {
    packet.sequence = 0U;
    packet.source.clear();
    packet.scale = 1.0;
  }

  [[nodiscard]] expected<flatbuffers::Offset<schema::TensorPacket>, Error> encode_root(
      flatbuffers::FlatBufferBuilder& builder, const schema::TensorPacketT& value,
      const std::span<const flatbuffers::Offset<schema::Tensor>> tensor_offsets) const {
    if (tensor_offsets.size() != 1U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto source = value.source.empty() ? flatbuffers::Offset<flatbuffers::String>{}
                                             : builder.CreateString(value.source);
    return schema::CreateTensorPacket(
        builder, value.sequence, source, value.scale, tensor_offsets[0]);
  }

  [[nodiscard]] expected<void, Error> assign_root(schema::TensorPacketT& value,
                                                  const schema::TensorPacket& packet) const {
    const auto* source = packet.source();
    if (source != nullptr &&
        source->size() >
            schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    value.sequence = packet.sequence();
    value.scale = packet.scale();
    if (source == nullptr) {
      value.source.clear();
    } else {
      value.source.assign(source->c_str(), source->size());
    }
    return {};
  }
};

}  // namespace

template <>
const RuntimePortTypeOps* runtime_port_type_ops<schema::TensorPacketT>() noexcept {
  static const RuntimePortTypeOps ops = schema::make_standard_descriptor_tensor_port_type_ops<
      schema::TensorPacketT,
      schema::TensorPacket,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::TensorPacket>,
      &schema::TensorPacketT::data>(TensorPacketCodec{});
  return &ops;
}

}  // namespace holoscan::detail
