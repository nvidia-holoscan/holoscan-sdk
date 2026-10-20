/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

struct Params {
  std::vector<std::string> values{};
};

// REQ: PARAM-003
HOLOSCAN_PARAM_SCHEMA(Params,
                      holoscan::param_schema("holoscan.test.ElementUtf8BoundWithoutEncoding", 1),
                      holoscan::field<&Params::values>("values").max_element_utf8_bytes(16));

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
