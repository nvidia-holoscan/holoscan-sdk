/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>

#include <holoscan/core/port.hpp>

struct MissingDescriptorTypeOpsRoot {};

struct MissingDescriptorTypeOps {
  holoscan::Tensor tensor;
};

template <>
consteval holoscan::SchemaIdentity holoscan::schema_identity<MissingDescriptorTypeOps>() {
  return holoscan::SchemaIdentity{
      .canonical_name = "nvidia.holoscan.test.MissingDescriptorTypeOps",
      .version = 1U,
      .schema_sha256 = {std::byte{0x42}},
      .id = holoscan::SchemaId{.value = {std::byte{0x24}}},
  };
}

template <>
struct holoscan::DescriptorTensorSchemaNativeTraits<MissingDescriptorTypeOps> {
  using root_type = MissingDescriptorTypeOpsRoot;
  static constexpr std::size_t metadata_capacity = 256U;

  [[nodiscard]] static const holoscan::Tensor* tensor(
      const MissingDescriptorTypeOps& value) noexcept {
    return &value.tensor;
  }
};

int main() {
  static_cast<void>(holoscan::detail::runtime_port_type_ops<MissingDescriptorTypeOps>());
}
