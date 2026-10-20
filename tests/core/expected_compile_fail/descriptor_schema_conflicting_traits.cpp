/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <holoscan/core/port.hpp>

struct ConflictingDescriptorTraitsRoot {};

struct ConflictingDescriptorTraits {
  holoscan::Tensor tensor;
};

template <>
consteval holoscan::SchemaIdentity holoscan::schema_identity<ConflictingDescriptorTraits>() {
  return holoscan::make_schema_identity(
      holoscan::make_schema_family_identity("holoscan.test.ConflictingDescriptorTraits"),
      holoscan::make_flatbuffer_file_identifier("CDTR"));
}

template <>
struct holoscan::DescriptorTensorSchemaNativeTraits<ConflictingDescriptorTraits> {
  using root_type = ConflictingDescriptorTraitsRoot;
  static constexpr std::size_t metadata_capacity = 256U;
  static constexpr std::size_t tensor_count = 1U;
  static constexpr std::array<std::uint16_t, tensor_count> tensor_field_ids{1U};

  [[nodiscard]] static const holoscan::Tensor* tensor_at(const ConflictingDescriptorTraits& value,
                                                         const std::size_t index) noexcept {
    return index == 0U ? &value.tensor : nullptr;
  }
};

template <>
struct holoscan::GeneratedSchemaNativeTraits<ConflictingDescriptorTraits> {
  using root_type = ConflictingDescriptorTraitsRoot;
  static constexpr std::size_t max_serialized_size = 256U;
};

int main() {
  static_cast<void>(holoscan::detail::runtime_port_type_ops<ConflictingDescriptorTraits>());
}
