/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

struct PartialParams {
  int declared{};
  int unlisted{};
};

HOLOSCAN_PARAM_SCHEMA(PartialParams,
                      holoscan::param_schema("holoscan.test.PartialRef", 1).partial(),
                      holoscan::field<&PartialParams::declared>("declared"));

// REQ: PARAM-002
constexpr auto kRef = holoscan::param_ref<&PartialParams::unlisted>();

int main() {}
