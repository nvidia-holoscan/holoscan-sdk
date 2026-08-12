/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>

#include <holoscan/core/port.hpp>

struct ConflictingDescriptorTraitsRoot {};

struct ConflictingDescriptorTraits {
  holoscan::Tensor tensor;
};

template <>
consteval holoscan::SchemaIdentity holoscan::schema_identity<ConflictingDescriptorTraits>() {
  return holoscan::SchemaIdentity{
      .canonical_name = "nvidia.holoscan.test.ConflictingDescriptorTraits",
      .version = 1U,
      .schema_sha256 = {std::byte{0x42}},
      .id = holoscan::SchemaId{.value = {std::byte{0x24}}},
  };
}

template <>
struct holoscan::DescriptorTensorSchemaNativeTraits<ConflictingDescriptorTraits> {
  using root_type = ConflictingDescriptorTraitsRoot;
  static constexpr std::size_t metadata_capacity = 256U;

  [[nodiscard]] static const holoscan::Tensor* tensor(
      const ConflictingDescriptorTraits& value) noexcept {
    return &value.tensor;
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
