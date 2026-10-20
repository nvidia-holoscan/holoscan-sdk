/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

struct Params {
  std::vector<std::optional<int>> values{};
};

// REQ: PARAM-054
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("holoscan.test.OptionalSequenceElement", 1),
                      holoscan::field<&Params::values>("values"));

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
