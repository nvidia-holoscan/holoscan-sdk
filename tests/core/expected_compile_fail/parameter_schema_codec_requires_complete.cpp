/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/params/config.hpp>

// REQ: PARAM-002
// REQ: PARAM-038
struct PartialParams {
  int value{};
};

HOLOSCAN_PARAM_SCHEMA(PartialParams,
                      holoscan::param_schema("nvidia.holoscan.test.PartialCodec", 1).partial(),
                      holoscan::field<&PartialParams::value>("value"));

template <>
struct holoscan::config::ConfigCodec<PartialParams>
    : holoscan::params::DefaultedParamSchemaCodec<PartialParams> {};

int main() {}
