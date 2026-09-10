/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_catalog_digest_internal.hpp"

#include <cstddef>
#include <cstdint>

namespace holoscan::detail {
namespace {

void add_hash_bytes(StableHashBuilder& hash, const ParamSchemaHash& value) noexcept {
  for (const std::byte byte : value.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
}

void add_constraints(StableHashBuilder& hash, const ParamConstraints& value) noexcept {
  hash.add(value.has_range);
  hash.add(value.range_min);
  hash.add(value.range_max);
  hash.add(value.allowed_value_count);
  for (std::size_t index = 0U; index < value.allowed_value_count; ++index) {
    hash.add(value.allowed_values[index]);
  }
  hash.add_canonical_size(value.max_utf8_bytes);
  hash.add_canonical_size(value.max_element_utf8_bytes);
  hash.add_canonical_size(value.max_elements);
  hash.add_canonical_size(value.max_encoded_bytes);
}

}  // namespace

void add_parameter_catalog_records(StableHashBuilder& hash,
                                   std::span<const OperatorParamRecord> operators,
                                   std::span<const ParamFieldCatalogRecord> fields) noexcept {
  hash.add_canonical_size(operators.size());
  for (const auto& op : operators) {
    hash.add(op.operator_instance.id);
    hash.add(op.operator_instance.plan_generation);
    hash.add(op.operator_instance.run_generation);
    hash.add_canonical_size(op.path.segments().size());
    for (const auto& segment : op.path.segments()) {
      hash.add(segment);
    }
    hash.add(op.effective_profile.has_value());
    if (op.effective_profile) {
      hash.add(*op.effective_profile);
    }
    hash.add(op.schema_availability);
    hash.add(op.schema_disposition);
    hash.add(op.materialization);
    hash.add(op.schema_id);
    hash.add(op.schema_version);
    add_hash_bytes(hash, op.structural_hash);
    add_hash_bytes(hash, op.contract_hash);
    add_hash_bytes(hash, op.metadata_hash);
    hash.add(op.schema_coverage);
    hash.add(op.schema_proof);
  }
  hash.add_canonical_size(fields.size());
  for (const auto& field : fields) {
    hash.add(field.instance);
    add_hash_bytes(hash, field.instance_identity_digest);
    hash.add(field.address.operator_instance.id);
    hash.add(field.address.operator_instance.plan_generation);
    hash.add(field.address.operator_instance.run_generation);
    hash.add(field.address.field);
    add_hash_bytes(hash, field.full_id);
    hash.add_canonical_size(field.path.segments().size());
    for (const auto& segment : field.path.segments()) {
      hash.add(segment);
    }
    hash.add(field.logical_type);
    hash.add(field.type_shape.type);
    hash.add(field.type_shape.bit_width);
    hash.add(field.type_shape.element_or_rep_type);
    hash.add(field.type_shape.element_or_rep_bit_width);
    hash.add(field.type_shape.duration_period_numerator);
    hash.add(field.type_shape.duration_period_denominator);
    hash.add(field.type_shape.has_stable_shape);
    hash.add(field.mutability);
    hash.add(field.earliest_invalidated_stage);
    hash.add(field.persistence);
    hash.add(field.visibility);
    hash.add(field.external_access);
    hash.add(field.sensitivity);
    hash.add(field.required_on_load);
    hash.add(field.alias);
    hash.add(field.validator_id);
    add_constraints(hash, field.constraints);
    hash.add(field.string_encoding);
    hash.add(field.element_string_encoding);
    hash.add(field.value_route);
    hash.add(field.storage.owns_dynamic_storage);
    hash.add(field.storage.fixed_object_extent);
    hash.add(field.storage.has_portable_codec);
    hash.add(field.storage.supports_bounded_materialization);
    hash.add_canonical_size(field.storage.inline_bytes);
    hash.add(field.materialization);
    hash.add(field.projection);
    hash.add(field.update_disposition);
    hash.add(field.adapter_eligible);
  }
}

}  // namespace holoscan::detail
