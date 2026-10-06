/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan {

// Descriptor-backed schema carrying more than one Tensor, so no field is implied by the type.
struct AmbiguousProjectionSchema {
  std::shared_ptr<Tensor> left;
  std::shared_ptr<Tensor> right;
};

template <>
consteval SchemaIdentity schema_identity<AmbiguousProjectionSchema>() {
  return schema_identity<schema::ImageT>();
}

template <>
struct DescriptorTensorSchemaNativeTraits<AmbiguousProjectionSchema> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, 2U> tensor_field_ids{1U, 3U};

  [[nodiscard]] static const Tensor* tensor_at(const AmbiguousProjectionSchema& value,
                                               const std::size_t index) noexcept {
    return index == 0U ? value.left.get() : value.right.get();
  }
};

}  // namespace holoscan

// REQ: GRAPH-043
int main() {
  holoscan::Graph graph;
  holoscan::Output<holoscan::AmbiguousProjectionSchema> output{"output"};
  holoscan::Input<holoscan::Tensor> input{"input"};
  // Omitting .project() leaves the field ambiguous, which must be named rather than guessed.
  (void)graph.add_flow(output, input);
}
