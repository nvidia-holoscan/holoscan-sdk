/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/prepared_schema_storage.hpp>

#include "prepared_descriptor_schema_traits.hpp"

namespace holoscan::test::prepared_descriptor {

using Packet = prepared_schema_test::PacketT;
using Fields = prepared_schema_test::PacketTensorFields;
using Scratch = schema::DescriptorMetadataScratch<Packet>;
inline constexpr std::array<std::int64_t, 1> kShape{1};
inline constexpr std::size_t kPatterns = 48;

inline Packet make_metadata(std::size_t pattern) {
  const auto mask = pattern / 3;
  const std::size_t elements = std::array<std::size_t, 3>{0, 1, 4}[pattern % 3];
  const std::size_t length = elements == 4 ? 128 : elements == 1 ? 6 : 0;
  Packet value;
  value.label.assign(length, 'p');
  value.readings.assign(elements * 2, 4.F);
  value.tags.assign(elements, std::string(length, 't'));
  if (mask & 1U) {
    value.metadata = std::make_shared<prepared_schema_test::MetadataT>();
    value.metadata->label.assign(length, 'm');
    value.metadata->tags.assign(elements == 4 ? 3 : elements, std::string(length, 'n'));
    if (mask & 2U)
      value.metadata->calibration = std::make_shared<prepared_schema_test::Calibration>(2.F, 3);
    if (mask & 4U) {
      value.metadata->child = std::make_shared<prepared_schema_test::ChildT>();
      value.metadata->child->label.assign(length, 'c');
    }
  }
  if (mask & 8U)
    value.calibration = std::make_shared<prepared_schema_test::Calibration>(4.F, 5);
  return value;
}

inline void edit_metadata(Packet& value) {
  if (!value.label.empty())
    value.label.front() = 'e';
  if (!value.readings.empty())
    value.readings.front() = 14.F;
  if (!value.tags.empty() && !value.tags.front().empty())
    value.tags.front().front() = 'e';
  if (value.metadata && !value.metadata->label.empty())
    value.metadata->label.front() = 'e';
  if (value.metadata && value.metadata->child && !value.metadata->child->label.empty())
    value.metadata->child->label.front() = 'e';
}

inline bool same_calibration(const std::shared_ptr<prepared_schema_test::Calibration>& lhs,
                             const std::shared_ptr<prepared_schema_test::Calibration>& rhs) {
  return static_cast<bool>(lhs) == static_cast<bool>(rhs) &&
         (!lhs || (lhs->scale() == rhs->scale() && lhs->revision() == rhs->revision()));
}

// Ignore Tensor owners and sequence; compare every ordinary metadata field without allocating.
inline bool same_metadata(const Packet& lhs, const Packet& rhs) {
  if (lhs.label != rhs.label || lhs.readings != rhs.readings || lhs.tags != rhs.tags ||
      !same_calibration(lhs.calibration, rhs.calibration) ||
      static_cast<bool>(lhs.metadata) != static_cast<bool>(rhs.metadata))
    return false;
  if (!lhs.metadata)
    return true;
  const auto& a = *lhs.metadata;
  const auto& b = *rhs.metadata;
  return a.label == b.label && a.tags == b.tags && same_calibration(a.calibration, b.calibration) &&
         static_cast<bool>(a.child) == static_cast<bool>(b.child) &&
         (!a.child || a.child->label == b.child->label);
}

inline void declare_input(OperatorSpec& spec, Input<Packet>& input, MemoryKind kind) {
  auto& builder = spec.input(input);
  builder.queue_depth(1);
  const TensorInputSpec tensor{
      .representation = {.memory_kind = kind, .dtype = DLDataType{kDLUInt, 64, 1}, .rank = 1}};
  builder.tensor_field(Fields::first, tensor);
  builder.tensor_field(Fields::second, tensor);
}

inline void declare_output(OperatorSpec& spec, Output<Packet>& output, MemoryKind kind) {
  auto& builder = spec.output(output);
  builder.max_emits_per_compute(1);
  const TensorOutputSpec tensor{
      .representation = {.memory_kind = kind, .dtype = DLDataType{kDLUInt, 64, 1}, .rank = 1},
      .bounds = tensor_bounds(8),
      .storage = TensorOutputStorage::kRuntimePool};
  builder.tensor_field(Fields::first, tensor);
  builder.tensor_field(Fields::second, tensor);
}

inline auto shape_requests() {
  return std::array{tensor_shape(Fields::first, kShape), tensor_shape(Fields::second, kShape)};
}

}  // namespace holoscan::test::prepared_descriptor
