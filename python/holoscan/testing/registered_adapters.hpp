// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "adapters.hpp"

namespace holoscan::testing::python {

struct RegisteredSnapshot {
  // Keep converters and the schema facade alive until all owned capture data is destroyed.
  std::shared_ptr<const BindingOwner> owner;
  GilSafeObject schema_type;
  authoring::ForeignSchemaValue metadata;
  SampleMetadata sample_metadata;
  std::vector<std::pair<std::string, std::optional<TensorSnapshot>>> tensors;
};

std::shared_ptr<const PayloadAdapter> registered_payload_adapter(
    const PortRecord& port, std::shared_ptr<const BindingOwner> owner);
void init_schema_snapshot(nanobind::module_& module);

}  // namespace holoscan::testing::python
