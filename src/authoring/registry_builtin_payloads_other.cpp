/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "registry_builtin_payloads.hpp"

#include <vector>

#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/domain/tensor.hpp>

namespace holoscan::authoring::detail {

std::vector<RegisteredPayloadType> make_builtin_other_payloads() {
  std::vector<RegisteredPayloadType> rows;
  rows.reserve(4U);
  rows.push_back(make_registered_payload<bool>("holoscan.boolean"));
  rows.push_back(make_registered_payload<float>("holoscan.float32"));
  rows.push_back(make_registered_payload<double>("holoscan.float64"));
  rows.push_back(make_registered_payload<Tensor>("holoscan.Tensor"));
  return rows;
}

}  // namespace holoscan::authoring::detail
