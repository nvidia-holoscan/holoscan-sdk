/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/graph.hpp>

struct CarrierPayload {
  std::uint32_t value{};
};

template <>
consteval holoscan::SchemaIdentity holoscan::schema_identity<CarrierPayload>() {
  return holoscan::make_schema_identity(
      holoscan::make_schema_family_identity("holoscan.test.CarrierPayload"),
      holoscan::make_flatbuffer_file_identifier("CRPL"));
}

// REQ: GRAPH-008
int main() {
  holoscan::Graph graph;
  holoscan::Output<CarrierPayload> output{"output"};
  graph.configure(output).tensor_allocation(
      holoscan::TensorPoolBounds{.max_rank = 1, .max_byte_span = 4});
}
