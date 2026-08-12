/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

struct Params {
  int value{};
};

// REQ: PARAM-003
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("nvidia.holoscan.test.PersistenceConflict", 1),
                      holoscan::field<&Params::value>("value").persisted().transient().transient());

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
