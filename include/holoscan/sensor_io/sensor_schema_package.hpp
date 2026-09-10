/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "holoscan/core/schema_identity.hpp"
#include "holoscan/holoscan_sensor_io_export.h"
#include "holoscan/schema/flatbuffer_schema_package.hpp"
#include "holoscan/sensor_io/sensor_schema_traits.hpp"

#include "image_bfbs_generated.h"
#include "image_generated.h"
#include "imu_bfbs_generated.h"
#include "imu_generated.h"

/// @file
/// @brief Port-type registration for the sensor payload schemas this library owns.
///
/// Including this header is what makes \c holoscan::schema::ImageT and \c holoscan::schema::ImuT
/// legal on a Holoscan port. Core registers the transport primitives the same way for its own
/// types; nothing here is a private extension mechanism, and a vendor schema registers itself
/// identically.
///
/// The source evidence below names Core-owned includes -- geometry.fbs, header.fbs, time.fbs,
/// tensor.fbs -- by logical path and content hash, both of which are independent of where the
/// including schema lives. That is why moving image.fbs did not disturb them.

namespace holoscan::schema::detail {

inline constexpr std::array<SchemaPackageSource, 3> kImageIncludes{
    kTensorSource,
    kHeaderSource,
    kTimeSource,
};
inline constexpr std::array<SchemaPackageSource, 3> kImuIncludes{
    kGeometrySource,
    kHeaderSource,
    kTimeSource,
};

}  // namespace holoscan::schema::detail

namespace holoscan {

/// @brief Describe the SDK Image owning type as descriptor metadata plus one Core Tensor.
///
/// Image metadata is bounded independently from Tensor storage. This specialization deliberately
/// does not opt `ImageT` into `GeneratedSchemaNativeTraits`: ordinary generated-schema encoding
/// would serialize or materialize Tensor bytes and bypass the plan-owned Tensor lifetime.
template <>
struct DescriptorTensorSchemaNativeTraits<schema::ImageT> {
  using root_type = schema::Image;  ///< Generated metadata table root.
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_default_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 1U;  ///< Number of direct Tensor fields.
  static constexpr std::array<std::uint16_t, 1U> tensor_field_ids{4U};

  /// @brief Return the embedded Tensor without extending its lifetime.
  /// @param value Owning generated Image value.
  /// @return Embedded Tensor, or `nullptr` when the Image has no Tensor member.
  [[nodiscard]] static const Tensor* tensor(const schema::ImageT& value) noexcept {
    return value.data.get();
  }

  /// @brief Return the indexed embedded Tensor without extending its lifetime.
  [[nodiscard]] static const Tensor* tensor_at(const schema::ImageT& value,
                                               const std::size_t index) noexcept {
    return index == 0U ? tensor(value) : nullptr;
  }
};

/// @brief Describe the SDK Imu owning generated type and finite serialized capacity.
template <>
struct GeneratedSchemaNativeTraits<schema::ImuT> {
  using root_type = schema::Imu;  ///< Generated table root used only for package codec evidence.
  static constexpr std::size_t max_serialized_size =
      schema::flatbuffer_default_inline_capacity_v<root_type>;  ///< Maximum serialized bytes.
};

/// @brief Return the SDK Image semantic identity for its descriptor-backed owning type.
template <>
consteval SchemaIdentity schema_identity<schema::ImageT>() {
  return schema::FlatBufferSchemaTraits<schema::Image>::identity;
}

/// @brief Return the SDK Imu semantic identity for its owning generated type.
/// @return Complete stable identity shared with the generated Imu root package.
template <>
consteval SchemaIdentity schema_identity<schema::ImuT>() {
  return schema::FlatBufferSchemaTraits<schema::Imu>::identity;
}

namespace detail {

/// @brief Return the descriptor-backed Runtime operations for generated Image values.
/// @return Process-local operations whose address never enters a plan or transport record.
template <>
[[nodiscard]] HOLOSCAN_SENSOR_IO_EXPORT const RuntimePortTypeOps*
runtime_port_type_ops<schema::ImageT>() noexcept;

}  // namespace detail

/// @brief Return the generated SDK Image package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::Image>& schema_package<schema::Image>() noexcept {
  static const SchemaPackageDescriptor<schema::Image> package =
      schema::detail::make_sdk_flatbuffer_package<schema::Image, schema::ImageBinarySchema>(
          "image.fbs", std::span{schema::detail::kImageIncludes});
  return package;
}

/// @brief Return the generated SDK Imu package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::Imu>& schema_package<schema::Imu>() noexcept {
  static const SchemaPackageDescriptor<schema::Imu> package =
      schema::detail::make_sdk_flatbuffer_package<schema::Imu, schema::ImuBinarySchema>(
          "imu.fbs", std::span{schema::detail::kImuIncludes});
  return package;
}

/// @brief Return the generated SDK Imu package for its owning native type.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::ImuT>& schema_package<schema::ImuT>() noexcept {
  static const SchemaPackageDescriptor<schema::ImuT> package = schema::detail::
      make_sdk_flatbuffer_native_package<schema::ImuT, schema::Imu, schema::ImuBinarySchema>(
          "imu.fbs", std::span{schema::detail::kImuIncludes});
  return package;
}

}  // namespace holoscan
