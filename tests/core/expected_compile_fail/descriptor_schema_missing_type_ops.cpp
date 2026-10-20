/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <holoscan/core/port.hpp>

struct MissingDescriptorTypeOpsRoot {};

struct MissingDescriptorTypeOps {
  holoscan::Tensor tensor;
};

template <>
consteval holoscan::SchemaIdentity holoscan::schema_identity<MissingDescriptorTypeOps>() {
  return holoscan::make_schema_identity(
      holoscan::make_schema_family_identity("holoscan.test.MissingDescriptorTypeOps"),
      holoscan::make_flatbuffer_file_identifier("MDOP"));
}

template <>
struct holoscan::DescriptorTensorSchemaNativeTraits<MissingDescriptorTypeOps> {
  using root_type = MissingDescriptorTypeOpsRoot;
  static constexpr std::size_t metadata_capacity = 256U;
  static constexpr std::size_t tensor_count = 1U;
  static constexpr std::array<std::uint16_t, tensor_count> tensor_field_ids{1U};

  [[nodiscard]] static const holoscan::Tensor* tensor_at(const MissingDescriptorTypeOps& value,
                                                         const std::size_t index) noexcept {
    return index == 0U ? &value.tensor : nullptr;
  }
};

int main() {
  static_cast<void>(holoscan::detail::runtime_port_type_ops<MissingDescriptorTypeOps>());
}
