/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_package_capabilities.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <holoscan/core/parameter_schema.hpp>

#include "compile_platform.hpp"
#include "profile_admission.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool policy_equal(const ProfilePolicyId& lhs, const ProfilePolicyId& rhs) noexcept {
  return lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.digest == rhs.digest;
}

[[nodiscard]] bool policy_less(const ProfilePolicyId& lhs, const ProfilePolicyId& rhs) noexcept {
  return std::tuple{lhs.major, lhs.minor, lhs.digest.value} <
         std::tuple{rhs.major, rhs.minor, rhs.digest.value};
}

void append_json_string(std::string& output, std::string_view value) {
  constexpr std::string_view kHex = "0123456789abcdef";
  output.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(kHex[(byte >> 4U) & 0x0FU]);
          output.push_back(kHex[byte & 0x0FU]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  output.push_back('"');
}

template <typename Values, typename Append>
void append_array(std::string& output, const Values& values, Append append) {
  output.push_back('[');
  bool first = true;
  for (const auto& value : values) {
    if (!first) {
      output.push_back(',');
    }
    first = false;
    append(output, value);
  }
  output.push_back(']');
}

[[nodiscard]] std::string digest_text(const ParamSchemaHash& digest) {
  constexpr std::string_view kHex = "0123456789abcdef";
  std::string result{"sha256:"};
  result.reserve(71U);
  for (const std::byte byte : digest.value) {
    const auto value = std::to_integer<std::uint8_t>(byte);
    result.push_back(kHex[(value >> 4U) & 0x0FU]);
    result.push_back(kHex[value & 0x0FU]);
  }
  return result;
}

[[nodiscard]] std::string policy_digest_text(const ProfilePolicyId& policy) {
  return digest_text(ParamSchemaHash{.value = policy.digest.value});
}

[[nodiscard]] std::string hash_text(std::string_view bytes) {
  StableHashBuilder hash;
  for (const unsigned char byte : bytes) {
    hash.add_byte(byte);
  }
  return digest_text(hash.finish());
}

template <typename Values, typename Equal>
[[nodiscard]] bool has_duplicate(const Values& values, Equal equal) {
  return std::adjacent_find(values.begin(), values.end(), equal) != values.end();
}

[[nodiscard]] Error malformed_error() {
  return Error{ErrorCode::kInvalidArgument, "PROFILE_PACKAGE_MANIFEST_MALFORMED"};
}

[[nodiscard]] bool valid_digest(std::string_view digest) noexcept {
  return digest.size() == 71U && digest.starts_with("sha256:") &&
         std::all_of(digest.begin() + 7, digest.end(), [](char value) {
           return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
         });
}

[[nodiscard]] RuntimePackageCapabilityManifest make_generated_manifest() {
  RuntimePackageCapabilityManifest manifest;
  manifest.package_id = "holoscan-core";
  manifest.build_id = "holoscan-core-5.0";
  manifest.supported_plan_schemas = {"holoscan.execution_plan.private.v1"};
  manifest.profile_policies = {profile_policy_id_v1()};
  manifest.runtime_abi_versions = {"holoscan.runtime_abi.v1"};
  manifest.target_triples = {std::string{compile_target_triple()}};
  manifest.offers = {
      {"holoscan.profile.robotics.finite_evidence",
       1U,
       "holoscan.profile.robotics.finite_evidence/v1"},
      {"holoscan.profile.robotics_critical.bounded_memory",
       1U,
       "holoscan.profile.robotics_critical.bounded_memory/v1"},
      {"holoscan.profile.robotics_critical.latency_budget",
       1U,
       "holoscan.profile.robotics_critical.latency_budget/v1"},
      {"holoscan.profile.regulated.complete_evidence",
       1U,
       "holoscan.profile.regulated.unqualified_evidence/v1"},
      {"holoscan.runtime.channel.generated_schema", 1U, "holoscan.channel.generated_schema/v1"},
      {"holoscan.runtime.channel.holomq_transport", 1U, "holoscan.channel.holomq_transport/v1"},
      {"holoscan.runtime.channel.pod_codec", 1U, "holoscan.channel.pod_codec/v1"},
      {"holoscan.runtime.channel_endpoint_catalog", 1U, "holoscan.compile.endpoint_catalog/v1"},
      {"holoscan.runtime.channel_service_attach", 1U, "holoscan.compile.channel_service_plan/v1"},
      {"holoscan.runtime.execution_plan", 1U, "holoscan.compile.execution_plan.private.v1"},
      {"holoscan.runtime.generic_output_loans", 1U, "holoscan.compile.generic_output_loan_plan/v1"},
      {"holoscan.runtime.payload_edges", 1U, "holoscan.compile.physical_plan/v1"},
      {"holoscan.runtime.readiness_sources", 1U, "holoscan.compile.readiness_plan/v1"},
      {"holoscan.runtime.ready.deterministic_serial",
       1U,
       "holoscan.compile.scheduler.ready_policy/v1"},
      {"holoscan.runtime.temporal_contracts", 1U, "holoscan.compile.temporal_plan/v1"},
      {"holoscan.runtime.target.current", 1U, "holoscan.runtime.target.current/v1"},
      {"holoscan.runtime.tensor_device_stream_materialization",
       1U,
       "holoscan.compile.tensor_device_resource_plan/v1"},
      {std::string{kTensorInlineRecordCapabilityId},
       kTensorInlineRecordCapabilityVersion,
       "holoscan.runtime.tensor_inline_record/generated/v1"},
      {"holoscan.runtime.tensor_pool_materialization", 1U, "holoscan.compile.tensor_pool_plan/v1"},
      {"holoscan.runtime.tensor_transfer_matrix", 1U, "holoscan.compile.tensor_transfer_plan/v1"},
      {"holoscan.runtime.thread_placement.apply_readback",
       1U,
       "holoscan.runtime.thread_placement.apply_readback/v1"},
      {"holoscan.runtime.trace_context", 1U, "holoscan.runtime.trace_context/v1"},
  };
  manifest.tensor_inline_record = RuntimeTensorInlineRecordCapability{
      .provider_id = "holoscan-core.generated.tensor-record-provider",
      .service_family_id = "holoscan.channel.tensor-record-service",
      .capability_id = std::string{kTensorInlineRecordCapabilityId},
      .capability_version = kTensorInlineRecordCapabilityVersion,
      .evidence_id = "holoscan.runtime.tensor_inline_record/generated/v1",
      .record_format_id = std::string{kTensorRecordFormatId},
      .record_format_version = kTensorRecordFormatVersion,
      .inline_discriminator = std::string{kTensorInlineDiscriminator},
      .opaque_descriptor_discriminator = std::string{kTensorOpaqueDescriptorDiscriminator},
      .fixed_metadata_overhead_bytes = 512U,
      .max_tensor_rank = 32U,
      .max_inline_logical_bytes = 65'536U,
      .max_fixed_record_bytes = 66'048U,
      .inline_arm_schema_id = std::string{kTensorInlineArmSchemaId},
      .opaque_descriptor_arm_schema_id = std::string{kTensorOpaqueDescriptorArmSchemaId},
  };
  const auto sealed = seal_runtime_package_capability_manifest(manifest);
  if (!sealed) {
    // All fields above are compile-time-controlled. Returning an empty digest
    // makes the subsequent mandatory validation fail closed if this fixture is
    // accidentally changed into a malformed generated artifact.
    manifest.digest.clear();
  }
  return manifest;
}

}  // namespace

bool RuntimePackageCapabilityView::supports_plan_schema(std::string_view value) const noexcept {
  return std::binary_search(
      manifest_->supported_plan_schemas.begin(), manifest_->supported_plan_schemas.end(), value);
}

bool RuntimePackageCapabilityView::supports_profile_policy(
    const ProfilePolicyId& value) const noexcept {
  return std::binary_search(
      manifest_->profile_policies.begin(), manifest_->profile_policies.end(), value, policy_less);
}

bool RuntimePackageCapabilityView::supports_runtime_abi(std::string_view value) const noexcept {
  return std::binary_search(
      manifest_->runtime_abi_versions.begin(), manifest_->runtime_abi_versions.end(), value);
}

bool RuntimePackageCapabilityView::supports_target(std::string_view value) const noexcept {
  return std::binary_search(
      manifest_->target_triples.begin(), manifest_->target_triples.end(), value);
}

const RuntimePackageCapabilityOffer* RuntimePackageCapabilityView::find_offer(
    std::string_view id, std::uint32_t version) const noexcept {
  const auto found = std::lower_bound(manifest_->offers.begin(),
                                      manifest_->offers.end(),
                                      std::pair{id, version},
                                      [](const auto& lhs, const auto& rhs) {
                                        return std::pair<std::string_view, std::uint32_t>{
                                                   lhs.id, lhs.version} < rhs;
                                      });
  return found != manifest_->offers.end() && found->id == id && found->version == version ? &*found
                                                                                          : nullptr;
}

const RuntimePackageCapabilityOffer* RuntimePackageCapabilityView::find_predicate_recomputation(
    std::string_view qualified_key, SemanticVersion semantic_version,
    std::string_view implementation_fingerprint) const {
  const auto* offer =
      find_offer(runtime_predicate_capability_id(qualified_key, semantic_version), 1U);
  return offer != nullptr && offer->evidence_id == implementation_fingerprint ? offer : nullptr;
}

std::string runtime_predicate_capability_id(std::string_view qualified_key,
                                            SemanticVersion semantic_version) {
  std::string id{"holoscan.runtime.predicate_recompute/v1/key:"};
  id += std::to_string(qualified_key.size());
  id.push_back(':');
  id.append(qualified_key);
  id += "/semantic:" + std::to_string(semantic_version.major) + '.' +
        std::to_string(semantic_version.minor) + '.' + std::to_string(semantic_version.patch);
  return id;
}

std::string runtime_package_capability_canonical_json_v1(
    const RuntimePackageCapabilityManifest& manifest, bool include_digest) {
  auto plan_schemas = manifest.supported_plan_schemas;
  auto policies = manifest.profile_policies;
  auto runtime_abis = manifest.runtime_abi_versions;
  auto targets = manifest.target_triples;
  auto offers = manifest.offers;
  std::sort(plan_schemas.begin(), plan_schemas.end());
  std::sort(policies.begin(), policies.end(), policy_less);
  std::sort(runtime_abis.begin(), runtime_abis.end());
  std::sort(targets.begin(), targets.end());
  std::sort(offers.begin(), offers.end(), [](const auto& lhs, const auto& rhs) {
    return std::pair{lhs.id, lhs.version} < std::pair{rhs.id, rhs.version};
  });

  std::string output{"{\"build_id\":"};
  append_json_string(output, manifest.build_id);
  output += ",\"capability_offers\":";
  append_array(output, offers, [](std::string& destination, const auto& offer) {
    destination += "{\"evidence_id\":";
    append_json_string(destination, offer.evidence_id);
    destination += ",\"id\":";
    append_json_string(destination, offer.id);
    destination += ",\"version\":" + std::to_string(offer.version) + "}";
  });
  if (include_digest) {
    output += ",\"digest\":";
    append_json_string(output, manifest.digest);
  }
  output += ",\"package_id\":";
  append_json_string(output, manifest.package_id);
  output += ",\"profile_policies\":";
  append_array(output, policies, [](std::string& destination, const auto& policy) {
    destination += "{\"digest\":";
    append_json_string(destination, policy_digest_text(policy));
    destination += ",\"major\":" + std::to_string(policy.major) +
                   ",\"minor\":" + std::to_string(policy.minor) + "}";
  });
  output += ",\"runtime_abi_versions\":";
  append_array(output, runtime_abis, [](std::string& destination, const auto& value) {
    append_json_string(destination, value);
  });
  output += ",\"schema\":";
  append_json_string(output, manifest.schema);
  output += ",\"schema_version\":" + std::to_string(manifest.schema_version);
  output += ",\"supported_plan_schemas\":";
  append_array(output, plan_schemas, [](std::string& destination, const auto& value) {
    append_json_string(destination, value);
  });
  output += ",\"target_triples\":";
  append_array(output, targets, [](std::string& destination, const auto& value) {
    append_json_string(destination, value);
  });
  output += ",\"tensor_inline_record_capability\":";
  if (manifest.tensor_inline_record) {
    const auto& record = *manifest.tensor_inline_record;
    output += "{\"capability_id\":";
    append_json_string(output, record.capability_id);
    output += ",\"capability_version\":" + std::to_string(record.capability_version);
    output += ",\"evidence_id\":";
    append_json_string(output, record.evidence_id);
    output += ",\"fixed_metadata_overhead_bytes\":" +
              std::to_string(record.fixed_metadata_overhead_bytes);
    output += ",\"inline_arm_schema_id\":";
    append_json_string(output, record.inline_arm_schema_id);
    output += ",\"inline_discriminator\":";
    append_json_string(output, record.inline_discriminator);
    output += ",\"max_fixed_record_bytes\":" + std::to_string(record.max_fixed_record_bytes);
    output += ",\"max_inline_logical_bytes\":" + std::to_string(record.max_inline_logical_bytes);
    output += ",\"max_tensor_rank\":" + std::to_string(record.max_tensor_rank);
    output += ",\"opaque_descriptor_arm_schema_id\":";
    append_json_string(output, record.opaque_descriptor_arm_schema_id);
    output += ",\"opaque_descriptor_discriminator\":";
    append_json_string(output, record.opaque_descriptor_discriminator);
    output += ",\"provider_id\":";
    append_json_string(output, record.provider_id);
    output += ",\"record_format_id\":";
    append_json_string(output, record.record_format_id);
    output += ",\"record_format_version\":" + std::to_string(record.record_format_version);
    output += ",\"service_family_id\":";
    append_json_string(output, record.service_family_id);
    output.push_back('}');
  } else {
    output += "null";
  }
  output.push_back('}');
  return output;
}

expected<void, Error> seal_runtime_package_capability_manifest(
    RuntimePackageCapabilityManifest& manifest) {
  std::sort(manifest.supported_plan_schemas.begin(), manifest.supported_plan_schemas.end());
  std::sort(manifest.profile_policies.begin(), manifest.profile_policies.end(), policy_less);
  std::sort(manifest.runtime_abi_versions.begin(), manifest.runtime_abi_versions.end());
  std::sort(manifest.target_triples.begin(), manifest.target_triples.end());
  std::sort(manifest.offers.begin(), manifest.offers.end(), [](const auto& lhs, const auto& rhs) {
    return std::pair{lhs.id, lhs.version} < std::pair{rhs.id, rhs.version};
  });
  if (manifest.schema != kRuntimePackageCapabilityCodecV1 || manifest.schema_version != 1U ||
      manifest.package_id.empty() || manifest.build_id.empty() ||
      manifest.supported_plan_schemas.empty() || manifest.profile_policies.empty() ||
      manifest.runtime_abi_versions.empty() || manifest.target_triples.empty() ||
      has_duplicate(manifest.supported_plan_schemas, std::equal_to<>{}) ||
      has_duplicate(manifest.profile_policies, policy_equal) ||
      has_duplicate(manifest.runtime_abi_versions, std::equal_to<>{}) ||
      has_duplicate(manifest.target_triples, std::equal_to<>{}) ||
      has_duplicate(manifest.offers,
                    [](const auto& lhs, const auto& rhs) {
                      return lhs.id == rhs.id && lhs.version == rhs.version;
                    }) ||
      std::any_of(manifest.offers.begin(), manifest.offers.end(), [](const auto& offer) {
        return offer.id.empty() || offer.version == 0U || offer.evidence_id.empty();
      })) {
    return make_unexpected(malformed_error());
  }
  if (manifest.tensor_inline_record) {
    const auto& record = *manifest.tensor_inline_record;
    const bool sum_overflows =
        record.max_inline_logical_bytes >
        std::numeric_limits<std::uint64_t>::max() - record.fixed_metadata_overhead_bytes;
    const auto offer =
        std::find_if(manifest.offers.begin(), manifest.offers.end(), [&](const auto& candidate) {
          return candidate.id == record.capability_id &&
                 candidate.version == record.capability_version;
        });
    if (record.provider_id.empty() || record.service_family_id.empty() ||
        record.capability_id.empty() || record.capability_version == 0U ||
        record.evidence_id.empty() || record.record_format_id.empty() ||
        record.record_format_version == 0U || record.inline_discriminator.empty() ||
        record.opaque_descriptor_discriminator.empty() ||
        record.inline_discriminator == record.opaque_descriptor_discriminator ||
        record.fixed_metadata_overhead_bytes == 0U || record.max_tensor_rank == 0U ||
        record.max_inline_logical_bytes == 0U || sum_overflows ||
        record.max_fixed_record_bytes <
            record.max_inline_logical_bytes + record.fixed_metadata_overhead_bytes ||
        record.inline_arm_schema_id.empty() || record.opaque_descriptor_arm_schema_id.empty() ||
        offer == manifest.offers.end() || offer->evidence_id != record.evidence_id) {
      return make_unexpected(malformed_error());
    }
  }
  manifest.digest = hash_text(runtime_package_capability_canonical_json_v1(manifest, false));
  return {};
}

expected<RuntimePackageCapabilityView, Error> validate_runtime_package_capability_manifest(
    const RuntimePackageCapabilityManifest& manifest) {
  RuntimePackageCapabilityManifest canonical = manifest;
  const auto sealed = seal_runtime_package_capability_manifest(canonical);
  if (!sealed || !valid_digest(manifest.digest) || canonical.digest != manifest.digest ||
      canonical.supported_plan_schemas != manifest.supported_plan_schemas ||
      canonical.runtime_abi_versions != manifest.runtime_abi_versions ||
      canonical.target_triples != manifest.target_triples ||
      canonical.tensor_inline_record != manifest.tensor_inline_record ||
      canonical.offers.size() != manifest.offers.size() ||
      !std::equal(canonical.profile_policies.begin(),
                  canonical.profile_policies.end(),
                  manifest.profile_policies.begin(),
                  policy_equal) ||
      !std::equal(canonical.offers.begin(),
                  canonical.offers.end(),
                  manifest.offers.begin(),
                  [](const auto& lhs, const auto& rhs) {
                    return lhs.id == rhs.id && lhs.version == rhs.version &&
                           lhs.evidence_id == rhs.evidence_id;
                  })) {
    return make_unexpected(malformed_error());
  }
  return RuntimePackageCapabilityView{manifest};
}

const RuntimePackageCapabilityManifest& runtime_package_capability_manifest() {
  static const RuntimePackageCapabilityManifest manifest = make_generated_manifest();
  return manifest;
}

RuntimePackageCapabilityView runtime_package_capability_view() {
  const auto view =
      validate_runtime_package_capability_manifest(runtime_package_capability_manifest());
  if (view) {
    return view.value();
  }
  // Construction can only fail after an invalid source-controlled generated
  // fixture change. Preserve a valid borrowed object and let matching reject
  // every requirement rather than introducing mutable recovery or search.
  static const RuntimePackageCapabilityManifest fail_closed = [] {
    RuntimePackageCapabilityManifest manifest;
    manifest.package_id = "invalid-generated-package";
    manifest.build_id = "invalid-generated-build";
    manifest.supported_plan_schemas = {"invalid"};
    manifest.profile_policies = {profile_policy_id_v1()};
    manifest.runtime_abi_versions = {"invalid"};
    manifest.target_triples = {"invalid"};
    manifest.offers = {{"invalid", 1U, "invalid"}};
    (void)seal_runtime_package_capability_manifest(manifest);
    return manifest;
  }();
  return validate_runtime_package_capability_manifest(fail_closed).value();
}

RuntimeCompileEvidenceView runtime_compile_evidence_view() {
  // EA1-017 replaces this empty generated qualification set with the selected
  // package's sealed deployment evidence. Keeping it in the captured view now
  // prevents compiler passes from consulting a mutable provider independently.
  return RuntimeCompileEvidenceView{
      .package_capabilities = runtime_package_capability_view(),
      .endpoint_clock_health = {},
      .tensor_device_bindings = {},
  };
}

}  // namespace holoscan::detail
