/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file
/// @brief Typed access to the embedded FlatBuffers binary schema (BFBS) reflection bytes.
///
/// BFBS bytes describe a generated schema to tools that resolve fields dynamically at runtime, such
/// as a recorder that publishes a self-describing schema beside every logged message. They are
/// build provenance only: they never enter schema identity, family compatibility, hashing, plan or
/// catalog records, or any wire format, and \ref validate_schema_package never requires them.
///
/// A schema carries these bytes only when its build opted in through the `BFBS_EMBED` argument to
/// `holoscan_add_flatbuffer_schema()`. Because the bytes are reached through the same generated
/// binding that supplies \ref schema_package, a caller cannot pair one schema's reflection bytes
/// with another schema's payloads.

#include <concepts>
#include <cstddef>
#include <span>

#include "holoscan/schema/flatbuffer_schema_traits.hpp"

namespace holoscan::detail {

/// Check that a generated root binding names embedded BFBS reflection bytes.
template <typename T>
concept HasEmbeddedBinarySchemaRootBinding = HasFlatBufferSchemaRootBinding<T> && requires {
  typename schema::FlatBufferSchemaBinding<T>::binary_schema_type;
};

/// Check that a generated owning type reaches embedded BFBS reflection bytes through its root.
template <typename T>
concept HasEmbeddedBinarySchemaNativeBinding =
    HasFlatBufferSchemaNativeBinding<T> &&
    HasEmbeddedBinarySchemaRootBinding<typename T::TableType>;

/// Return the reflection bytes published by one flatc `--bfbs-gen-embed` type.
template <typename Embedded>
[[nodiscard]] inline std::span<const std::byte> embedded_binary_schema() noexcept {
  return {reinterpret_cast<const std::byte*>(Embedded::data()), Embedded::size()};
}

}  // namespace holoscan::detail

namespace holoscan {

/// @brief Return the embedded BFBS reflection bytes for a generated FlatBuffers root.
/// @tparam Root Generated table root whose build opted into `BFBS_EMBED`.
/// @return Process-static reflection bytes describing exactly \p Root.
/// @note The returned view has static storage duration and stays valid for the process lifetime.
template <typename Root>
  requires detail::HasEmbeddedBinarySchemaRootBinding<Root>
[[nodiscard]] inline std::span<const std::byte> binary_schema() noexcept {
  return detail::embedded_binary_schema<
      typename schema::FlatBufferSchemaBinding<Root>::binary_schema_type>();
}

/// @brief Return a generated owning type's embedded BFBS reflection bytes through its root binding.
/// @tparam Native Generated owning object-API type whose build opted into `BFBS_EMBED`.
/// @return Process-static reflection bytes describing \p Native's generated root.
/// @note The returned view has static storage duration and stays valid for the process lifetime.
template <typename Native>
  requires detail::HasEmbeddedBinarySchemaNativeBinding<Native>
[[nodiscard]] inline std::span<const std::byte> binary_schema() noexcept {
  return detail::embedded_binary_schema<
      typename schema::FlatBufferSchemaBinding<typename Native::TableType>::binary_schema_type>();
}

/// @brief Check whether semantic type `T` carries embedded BFBS reflection bytes.
/// @tparam T Generated root or owning object-API type.
///
/// A generic consumer uses this to stay compilable across schemas that did and did not opt into
/// `BFBS_EMBED`, rather than making reflection bytes a requirement of every logged type.
template <typename T>
concept HasBinarySchema = requires {
  { binary_schema<T>() } -> std::same_as<std::span<const std::byte>>;
};

}  // namespace holoscan
