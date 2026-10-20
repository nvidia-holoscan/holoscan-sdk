/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <iostream>
#include <string_view>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>

#include "descriptor_topic_mismatch_schema_traits.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: descriptor_topic_mismatch_peer CATALOG ENDPOINT\n";
    return 2;
  }

  auto catalog = holoscan::ChannelEndpointCatalog::load_json(std::filesystem::path{argv[1]});
  if (!catalog) {
    std::cerr << "failed to load catalog: " << holoscan::runtime_error_message(catalog.error())
              << '\n';
    return 3;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "failed to create runtime: " << holoscan::runtime_error_message(runtime.error())
              << '\n';
    return 4;
  }

  using Payload = holoscan::schema::DescriptorFlatBufferPayloadFor<tensor_topic::ExternalFrame>;
  auto publisher = runtime->create_tensor_schema_publisher<Payload>(*catalog, argv[2]);
  if (publisher) {
    std::cerr << "incompatible descriptor Tensor profile was accepted\n";
    return 5;
  }
  if (publisher.error().code != holoscan::ErrorCode::kIncompatible) {
    std::cerr << "unexpected rejection: " << holoscan::runtime_error_message(publisher.error())
              << '\n';
    return 6;
  }

  std::cout << "descriptor Tensor profile mismatch rejected\n";
  return 0;
}
