/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_identity.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../core/parameter_catalog_digest_internal.hpp"

namespace holoscan::detail {
namespace {

template <typename Digest>
void add_hash_bytes(StableHashBuilder& hash, Digest value) noexcept {
  for (const std::byte byte : value.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
}

template <typename T, typename AddValue>
void add_optional(StableHashBuilder& hash, const std::optional<T>& value,
                  AddValue add_value) noexcept {
  hash.add(value.has_value());
  if (value) {
    add_value(hash, *value);
  }
}

void add_thread_placement(StableHashBuilder& hash, const ThreadPlacement& placement) {
  hash.add(placement.affinity.has_value());
  if (placement.affinity) {
    CPUAffinity canonical_affinity = *placement.affinity;
    canonical_affinity.normalize();
    hash.add_canonical_size(canonical_affinity.cores.size());
    for (const std::uint32_t core : canonical_affinity.cores) {
      hash.add(core);
    }
  }
  hash.add(placement.scheduling.policy);
  add_optional(hash, placement.scheduling.priority, [](StableHashBuilder& target, int priority) {
    target.add(priority);
  });
  add_optional(hash,
               placement.scheduling.deadline,
               [](StableHashBuilder& target, const DeadlineScheduling& deadline) {
                 target.add(deadline.runtime_ns);
                 target.add(deadline.deadline_ns);
                 target.add(deadline.period_ns);
               });
  hash.add(placement.wake_mode);
  hash.add(placement.irq_isolation);
  hash.add(placement.probe_failure_policy);
}

void add_profile_admission_input(StableHashBuilder& hash, const ProfileManifest& profile) noexcept {
  hash.add(profile.disposition);
  add_optional(hash, profile.requested_profile, [](StableHashBuilder& target, Profile value) {
    target.add(value);
  });
  add_optional(hash, profile.effective_graph_profile, [](StableHashBuilder& target, Profile value) {
    target.add(value);
  });
  hash.add_canonical_size(profile.effective_regions.size());
  for (const auto& region : profile.effective_regions) {
    hash.add(region.region_id.value);
    add_optional(hash, region.requested_profile, [](StableHashBuilder& target, Profile value) {
      target.add(value);
    });
    add_optional(hash, region.effective_profile, [](StableHashBuilder& target, Profile value) {
      target.add(value);
    });
    hash.add(region.origin);
  }
  hash.add_canonical_size(profile.scope_resolutions.size());
  for (const auto& scope : profile.scope_resolutions) {
    hash.add(scope.object_id);
    hash.add(scope.object_kind);
    hash.add(scope.effective_profile);
    hash.add_canonical_size(scope.contributing_regions.size());
    for (const RegionId region : scope.contributing_regions) {
      hash.add(region.value);
    }
  }
  hash.add(profile.profile_policy_id.major);
  hash.add(profile.profile_policy_id.minor);
  add_hash_bytes(hash, profile.profile_policy_id.digest);
  hash.add(profile.plan_schema_version);
  hash.add(profile.runtime_abi_version);
  hash.add(profile.target_triple);
  hash.add_canonical_size(profile.required_capabilities.size());
  for (const auto& capability : profile.required_capabilities) {
    hash.add(capability.capability_id);
    hash.add(capability.version);
    hash.add(capability.evidence_id);
  }
  hash.add_canonical_size(profile.evidence_references.size());
  for (const auto& evidence : profile.evidence_references) {
    hash.add(evidence.id);
    hash.add(evidence.kind);
    hash.add(evidence.version);
  }
}

template <typename Range, typename Append>
void add_canonical_deployment_records(StableHashBuilder& hash, const Range& records,
                                      Append append) {
  std::vector<ParamSchemaHash> record_digests;
  record_digests.reserve(records.size());
  for (const auto& record : records) {
    StableHashBuilder record_hash;
    append(record_hash, record);
    record_digests.push_back(record_hash.finish());
  }
  std::sort(record_digests.begin(), record_digests.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.value < rhs.value;
  });
  hash.add_canonical_size(record_digests.size());
  for (const ParamSchemaHash digest : record_digests) {
    add_hash_bytes(hash, digest);
  }
}

void add_deployment_binding(StableHashBuilder& hash, const DeploymentBinding& deployment) {
  add_canonical_deployment_records(
      hash, deployment.thread_placements(), [](auto& record_hash, const auto& binding) {
        record_hash.add("thread_placement");
        record_hash.add(binding.partition_name);
        add_thread_placement(record_hash, binding.placement);
      });
  add_canonical_deployment_records(
      hash, deployment.cuda_green_contexts(), [](auto& record_hash, const auto& binding) {
        record_hash.add("cuda_green_context");
        record_hash.add(binding.partition_name);
        record_hash.add(binding.placement.device.value);
        record_hash.add(binding.placement.sm_count);
      });
  add_canonical_deployment_records(
      hash, deployment.generated_endpoint_names(), [](auto& record_hash, const auto& pin) {
        record_hash.add("generated_endpoint_name");
        record_hash.add(pin.final_name);
        record_hash.add(pin.algorithm);
        record_hash.add(pin.version);
      });
  add_canonical_deployment_records(
      hash, deployment.tensor_output_devices(), [](auto& record_hash, const auto& placement) {
        record_hash.add("tensor_output_device");
        record_hash.add(placement.operator_path);
        record_hash.add(placement.output_port);
        record_hash.add(placement.device.value);
      });
}

void add_scalar(StableHashBuilder& hash, ParamScalarValue value) noexcept {
  hash.add(value.kind);
  hash.add(value.bit_width);
  hash.add(value.raw_bits);
}

}  // namespace

ParameterProjectionCollision classify_parameter_field_projection(
    std::span<const ParameterFieldProjectionIdentity> existing, ParamFieldId field,
    ParamSchemaHash full_field) noexcept {
  const auto duplicate = std::find_if(existing.begin(), existing.end(), [&](const auto& item) {
    return item.first == field || item.second == full_field;
  });
  return duplicate == existing.end() ? ParameterProjectionCollision::kNone
                                     : ParameterProjectionCollision::kFieldId;
}

ParameterProjectionCollision classify_parameter_instance_projection(
    std::span<const ParameterInstanceProjectionIdentity> existing, ParamInstanceId instance,
    ParamSchemaHash full_instance) noexcept {
  if (instance == 0U) {
    return ParameterProjectionCollision::kInstanceId;
  }
  const auto duplicate = std::find_if(
      existing.begin(), existing.end(), [&](const auto& item) { return item.first == instance; });
  if (duplicate == existing.end()) {
    return ParameterProjectionCollision::kNone;
  }
  return duplicate->second == full_instance ? ParameterProjectionCollision::kFieldId
                                            : ParameterProjectionCollision::kInstanceId;
}

std::string_view parameter_projection_collision_token(
    ParameterProjectionCollision collision) noexcept {
  switch (collision) {
    case ParameterProjectionCollision::kNone:
      return {};
    case ParameterProjectionCollision::kFieldId:
      return "PARAMETER_FIELD_ID_COLLISION";
    case ParameterProjectionCollision::kInstanceId:
      return "PARAMETER_INSTANCE_ID_COLLISION";
  }
  return "PARAMETER_INSTANCE_ID_COLLISION";
}

bool canonicalize_parameter_scalar(ParamScalarValue& value) noexcept {
  switch (value.kind) {
    case ParamScalarKind::kBool:
      return value.bit_width == 1U && value.raw_bits <= 1U;
    case ParamScalarKind::kSignedInteger:
    case ParamScalarKind::kUnsignedInteger:
      if (value.bit_width != 8U && value.bit_width != 16U && value.bit_width != 32U &&
          value.bit_width != 64U) {
        return false;
      }
      return value.bit_width == 64U || value.raw_bits < (std::uint64_t{1} << value.bit_width);
    case ParamScalarKind::kFloatingPoint:
      if (value.bit_width == 32U) {
        if (value.raw_bits > std::numeric_limits<std::uint32_t>::max()) {
          return false;
        }
        const auto bits = static_cast<std::uint32_t>(value.raw_bits);
        if ((bits & 0x7F800000U) == 0x7F800000U) {
          return false;
        }
        if (bits == 0x80000000U) {
          value.raw_bits = 0U;
        }
        return true;
      }
      if (value.bit_width == 64U) {
        if ((value.raw_bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) {
          return false;
        }
        if (value.raw_bits == 0x8000000000000000ULL) {
          value.raw_bits = 0U;
        }
        return true;
      }
      return false;
    case ParamScalarKind::kDurationNanoseconds:
      return value.bit_width == 64U;
    case ParamScalarKind::kNone:
      return false;
  }
  return false;
}

ParameterProtectedValueDigest parameter_field_value_digest(const ParamSchemaDescriptor& schema,
                                                           const ParamFieldDescriptor& field,
                                                           const void* aggregate) noexcept {
  if (field.local_operations.read_member_as_scalar == nullptr) {
    return {.disposition = ParameterValueDigestDisposition::kLocalTypedField};
  }
  ParamScalarValue scalar;
  if (!field.local_operations.read_member_as_scalar(aggregate, scalar) ||
      !canonicalize_parameter_scalar(scalar)) {
    return {.disposition = ParameterValueDigestDisposition::kInvalidScalar};
  }
  StableHashBuilder hash;
  hash.add("holoscan.parameter.field_value.v1");
  hash.add(schema.stable_id);
  hash.add(schema.version);
  add_hash_bytes(hash, schema.structural_hash);
  add_hash_bytes(hash, schema.contract_hash);
  add_hash_bytes(hash, field.full_id);
  add_scalar(hash, scalar);
  return {
      .disposition = ParameterValueDigestDisposition::kCanonicalScalarFields,
      .digest = hash.finish(),
  };
}

ParamSchemaHash parameter_instance_identity(const OperatorPath& path,
                                            ParamSchemaHash field_full_id) noexcept {
  StableHashBuilder hash;
  hash.add("holoscan.param.instance.v1");
  hash.add_canonical_size(path.segments().size());
  for (const auto& segment : path.segments()) {
    hash.add(segment);
  }
  add_hash_bytes(hash, field_full_id);
  return hash.finish();
}

ParamInstanceId compact_parameter_instance_id(ParamSchemaHash identity) noexcept {
  ParamInstanceId result{};
  for (std::size_t index = 0U; index < sizeof(result); ++index) {
    result = (result << 8U) |
             static_cast<ParamInstanceId>(std::to_integer<std::uint8_t>(identity.value[index]));
  }
  return result;
}

ParamCatalogDigest parameter_catalog_digest(
    std::span<const OperatorParamRecord> operators,
    std::span<const ParamFieldCatalogRecord> fields) noexcept {
  StableHashBuilder hash;
  hash.add("holoscan.parameter_catalog.v1");
  add_parameter_catalog_records(hash, operators, fields);
  return ParamCatalogDigest{.value = hash.finish().value};
}

ParamSchemaHash empty_parameter_migration_catalog_digest() noexcept {
  StableHashBuilder hash;
  hash.add("holoscan.param.migration_catalog.v1");
  hash.add_canonical_size(0U);
  return hash.finish();
}

ParamSchemaHash parameter_admission_context_digest(const ProfileManifest& profile,
                                                   const DeploymentBinding& deployment,
                                                   ParamSchemaHash migration_catalog_digest) {
  StableHashBuilder hash;
  hash.add("holoscan.param.admission_context.v1");
  hash.add("profile_admission_input");
  add_profile_admission_input(hash, profile);
  hash.add("deployment_binding");
  add_deployment_binding(hash, deployment);
  hash.add("migration_catalog");
  add_hash_bytes(hash, migration_catalog_digest);
  return hash.finish();
}

ParameterProtectedValueDigest parameter_value_digest(const ParamTypeDescriptor& type,
                                                     const ParamSchemaDescriptor* schema,
                                                     const void* value, bool no_params) noexcept {
  if (no_params) {
    return {.disposition = ParameterValueDigestDisposition::kNoParams};
  }
  if (schema == nullptr) {
    return {.disposition = ParameterValueDigestDisposition::kOpaqueSchema};
  }
  if (schema->coverage != ParamSchemaCoverage::kComplete) {
    return {.disposition = ParameterValueDigestDisposition::kPartialSchema};
  }
  if (value == nullptr || schema->local_type != &type) {
    return {.disposition = ParameterValueDigestDisposition::kInvalidScalar};
  }
  StableHashBuilder hash;
  hash.add("holoscan.parameter.value.v1");
  hash.add(schema->stable_id);
  hash.add(schema->version);
  add_hash_bytes(hash, schema->structural_hash);
  add_hash_bytes(hash, schema->contract_hash);
  hash.add_canonical_size(schema->fields.size());
  for (const ParamFieldDescriptor& field : schema->fields) {
    if (field.local_operations.read_member_as_scalar == nullptr) {
      return {.disposition = ParameterValueDigestDisposition::kLocalTypedField};
    }
    ParamScalarValue scalar;
    if (!field.local_operations.read_member_as_scalar(value, scalar) ||
        !canonicalize_parameter_scalar(scalar)) {
      return {.disposition = ParameterValueDigestDisposition::kInvalidScalar};
    }
    add_hash_bytes(hash, field.full_id);
    add_scalar(hash, scalar);
  }
  return {
      .disposition = ParameterValueDigestDisposition::kCanonicalScalarFields,
      .digest = hash.finish(),
  };
}

}  // namespace holoscan::detail
