/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>

#include <holoscan/channel/tensor_schema_endpoints.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/tensor_materialization.hpp>

#include "external_frame_schema_traits.hpp"

namespace descriptor_tensor_external_topic {

using Frame = tensor_topic::ExternalFrameT;
using Fields = tensor_topic::ExternalFrameTensorFields;
using Root = tensor_topic::ExternalFrame;
using Payload = holoscan::schema::DescriptorFlatBufferPayloadFor<Root>;

static_assert(Payload::max_tensors == 2U);

inline constexpr std::uint64_t kDataSequence = 1U;
inline constexpr std::uint64_t kAckSequence = 2U;
inline constexpr std::string_view kClientSource = "standalone-client";
inline constexpr std::string_view kGraphSource = "graph-relay";
inline constexpr DLDataType kPixelsDtype{kDLUInt, 8U, 1U};
inline constexpr DLDataType kCalibrationDtype{kDLFloat, 32U, 1U};
inline constexpr std::array<std::int64_t, 2U> kPixelsShape{2, 4};
inline constexpr std::array<std::int64_t, 1U> kCalibrationShape{3};
inline constexpr std::array<std::uint8_t, 8U> kPixels{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
inline constexpr std::array<float, 3U> kCalibration{0.5F, 1.0F, -0.5F};
inline constexpr holoscan::TensorRepresentation kPixelsRepresentation{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kPixelsDtype,
    .rank = 2U,
    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
};
inline constexpr holoscan::TensorRepresentation kCalibrationRepresentation{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kCalibrationDtype,
    .rank = 1U,
    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
};

[[nodiscard]] inline std::string import_endpoint_name(const std::string_view instance) {
  return "descriptor-tensor-import-" + std::string{instance};
}

[[nodiscard]] inline std::string export_endpoint_name(const std::string_view instance) {
  return "descriptor-tensor-export-" + std::string{instance};
}

[[nodiscard]] inline holoscan::DescriptorTensorTopicContract<Frame> topic_contract(
    const holoscan::InterfaceTensorEmitPolicy policy) {
  holoscan::DescriptorTensorTopicContract<Frame> result;
  result
      .field(Fields::pixels,
             holoscan::TensorBoundaryContract{
                 .representation = kPixelsRepresentation,
                 .bounds = holoscan::tensor_bounds(kPixelsShape),
                 .transfer = holoscan::TensorTransferPolicy{.foreign_emit_policy = policy},
             })
      .field(Fields::calibration,
             holoscan::TensorBoundaryContract{
                 .representation = kCalibrationRepresentation,
                 .bounds = holoscan::tensor_bounds(kCalibrationShape),
                 .transfer = holoscan::TensorTransferPolicy{.foreign_emit_policy = policy},
             });
  return result;
}

[[nodiscard]] inline holoscan::TopicOptions import_topic_options(
    const std::string_view endpoint_name) {
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.attach = holoscan::TopicAttachPolicy{
      .peer = holoscan::PeerRequirement::kRequired,
      .max_wait = std::chrono::seconds{10},
  };
  options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 2U};
  return options;
}

[[nodiscard]] inline holoscan::TopicOptions export_topic_options(
    const std::string_view endpoint_name) {
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.attach = holoscan::TopicAttachPolicy{
      .peer = holoscan::PeerRequirement::kRequired,
      .max_wait = std::chrono::seconds{10},
  };
  options.bounds.subscriber =
      holoscan::SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  return options;
}

template <typename T, std::size_t N>
class HostArrayReference final : public holoscan::TensorMemoryReference {
 public:
  explicit HostArrayReference(const std::array<T, N>& source) : values(source) {}

  std::array<T, N> values;
};

template <typename T, std::size_t N, std::size_t Rank>
[[nodiscard]] holoscan::expected<holoscan::Tensor, holoscan::Error> make_host_tensor(
    const std::array<T, N>& values, const std::array<std::int64_t, Rank>& shape,
    const DLDataType dtype) {
  auto backing = std::make_shared<HostArrayReference<T, N>>(values);
  void* const data = backing->values.data();
  return holoscan::make_tensor(
      shape, dtype, std::nullopt, holoscan::MemoryStorageType::kSystem, data, std::move(backing));
}

[[nodiscard]] inline holoscan::expected<std::array<holoscan::Tensor, 2U>, holoscan::Error>
make_client_tensors() {
  auto pixels = make_host_tensor(kPixels, kPixelsShape, kPixelsDtype);
  if (!pixels) {
    return holoscan::make_unexpected(std::move(pixels).error());
  }
  auto calibration = make_host_tensor(kCalibration, kCalibrationShape, kCalibrationDtype);
  if (!calibration) {
    return holoscan::make_unexpected(std::move(calibration).error());
  }
  return std::array<holoscan::Tensor, 2U>{std::move(*pixels), std::move(*calibration)};
}

[[nodiscard]] inline holoscan::expected<std::vector<std::byte>, holoscan::Error> build_metadata(
    const std::uint64_t sequence, const std::string_view source,
    const std::span<const holoscan::Tensor> tensors) {
  if (tensors.size() != Payload::max_tensors) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "metadata Tensor inventory is incomplete"});
  }
  flatbuffers::FlatBufferBuilder builder;
  auto pixels = holoscan::schema::create_tensor_metadata(builder, tensors[0]);
  if (!pixels) {
    return holoscan::make_unexpected(std::move(pixels).error());
  }
  auto calibration = holoscan::schema::create_tensor_metadata(builder, tensors[1]);
  if (!calibration) {
    return holoscan::make_unexpected(std::move(calibration).error());
  }
  const auto source_offset = builder.CreateString(source);
  const auto root =
      tensor_topic::CreateExternalFrame(builder, sequence, source_offset, *pixels, *calibration);
  tensor_topic::FinishExternalFrameBuffer(builder, root);
  const auto* const first =
      reinterpret_cast<const std::byte*>(builder.GetBufferPointer());  // NOLINT
  return std::vector<std::byte>{first, first + builder.GetSize()};
}

[[nodiscard]] inline bool tensors_match(const std::span<const holoscan::Tensor> tensors) {
  if (tensors.size() != Payload::max_tensors) {
    return false;
  }
  const auto dtype_matches = [](const DLDataType actual, const DLDataType expected) {
    return actual.code == expected.code && actual.bits == expected.bits &&
           actual.lanes == expected.lanes;
  };
  if (tensors[0].ndim() != static_cast<std::int32_t>(kPixelsShape.size()) ||
      tensors[1].ndim() != static_cast<std::int32_t>(kCalibrationShape.size()) ||
      !std::equal(kPixelsShape.begin(), kPixelsShape.end(), tensors[0].shape_span().begin()) ||
      !std::equal(
          kCalibrationShape.begin(), kCalibrationShape.end(), tensors[1].shape_span().begin()) ||
      !dtype_matches(tensors[0].dtype(), kPixelsDtype) ||
      !dtype_matches(tensors[1].dtype(), kCalibrationDtype) ||
      tensors[0].device().device_type != kDLCPU || tensors[0].device().device_id != 0 ||
      tensors[1].device().device_type != kDLCPU || tensors[1].device().device_id != 0) {
    return false;
  }
  const auto pixels_ready = tensors[0].synchronize_for_host();
  const auto calibration_ready = tensors[1].synchronize_for_host();
  const auto* const pixels = tensors[0].data_as<std::uint8_t>();
  const auto* const calibration = tensors[1].data_as<float>();
  return pixels_ready && calibration_ready && pixels != nullptr && calibration != nullptr &&
         tensors[0].nbytes() == static_cast<std::int64_t>(kPixels.size()) &&
         tensors[1].nbytes() ==
             static_cast<std::int64_t>(kCalibration.size() * sizeof(kCalibration.front())) &&
         std::equal(kPixels.begin(), kPixels.end(), pixels) &&
         std::equal(kCalibration.begin(), kCalibration.end(), calibration);
}

}  // namespace descriptor_tensor_external_topic
