/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "registry_builtin_payloads.hpp"

#include <cstdint>
#include <vector>

#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>

namespace holoscan::authoring::detail {

std::vector<RegisteredPayloadType> make_builtin_integer_payloads() {
  std::vector<RegisteredPayloadType> rows;
  rows.reserve(8U);
  rows.push_back(make_registered_payload<std::uint8_t>("holoscan.uint8"));
  rows.push_back(make_registered_payload<std::int8_t>("holoscan.int8"));
  rows.push_back(make_registered_payload<std::uint16_t>("holoscan.uint16"));
  rows.push_back(make_registered_payload<std::int16_t>("holoscan.int16"));
  rows.push_back(make_registered_payload<std::uint32_t>("holoscan.uint32"));
  rows.push_back(make_registered_payload<std::int32_t>("holoscan.int32"));
  rows.push_back(make_registered_payload<std::uint64_t>("holoscan.uint64"));
  rows.push_back(make_registered_payload<std::int64_t>("holoscan.int64"));
  return rows;
}

}  // namespace holoscan::authoring::detail
