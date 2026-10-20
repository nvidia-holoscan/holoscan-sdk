/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

#include "process_local_fanout_experiment.hpp"

namespace holoscan::detail {

inline constexpr std::string_view kRuntimePackageCapabilityCodecV1 =
    "holoscan.runtime_package_capabilities.canonical_json.v1";
inline constexpr std::string_view kTensorInlineRecordCapabilityId =
    "holoscan.runtime.tensor_inline_record";
inline constexpr std::uint32_t kTensorInlineRecordCapabilityVersion = 1U;
inline constexpr std::string_view kTensorRecordFormatId = "holoscan.channel.tensor_record";
inline constexpr std::uint32_t kTensorRecordFormatVersion = 1U;
inline constexpr std::string_view kTensorInlineDiscriminator = "inline";
inline constexpr std::string_view kTensorOpaqueDescriptorDiscriminator = "opaque_descriptor";
inline constexpr std::string_view kTensorInlineArmSchemaId =
    "holoscan.channel.tensor_record.inline/v1";
inline constexpr std::string_view kTensorOpaqueDescriptorArmSchemaId =
    "holoscan.channel.tensor_record.opaque_descriptor/v1";

/// @brief Describe one exact capability offered by the executing package.
struct RuntimePackageCapabilityOffer {
  std::string id;
  std::uint32_t version{};
  std::string evidence_id;
};

/// @brief Describe the one generated bounded inline Tensor record provider.
struct RuntimeTensorInlineRecordCapability {
  std::string provider_id;
  std::string service_family_id;
  std::string capability_id;
  std::uint32_t capability_version{};
  std::string evidence_id;
  std::string record_format_id;
  std::uint32_t record_format_version{};
  std::string inline_discriminator;
  std::string opaque_descriptor_discriminator;
  std::uint32_t fixed_metadata_overhead_bytes{};
  std::uint32_t max_tensor_rank{};
  std::uint64_t max_inline_logical_bytes{};
  std::uint64_t max_fixed_record_bytes{};
  std::string inline_arm_schema_id;
  std::string opaque_descriptor_arm_schema_id;

  friend bool operator==(const RuntimeTensorInlineRecordCapability&,
                         const RuntimeTensorInlineRecordCapability&) = default;
};

/// @brief Hold one generated, immutable package capability artifact.
///
/// This source-private model contains only stable package identities and
/// capability evidence. It deliberately excludes paths, raw build flags,
/// secrets, pointers, backend objects, and observed startup match results.
struct RuntimePackageCapabilityManifest {
  std::string schema{std::string{kRuntimePackageCapabilityCodecV1}};
  std::uint32_t schema_version{1U};
  std::string package_id;
  std::string build_id;
  std::string digest;
  std::vector<std::string> supported_plan_schemas;
  std::vector<ProfilePolicyId> profile_policies;
  std::vector<std::string> runtime_abi_versions;
  std::vector<std::string> target_triples;
  std::vector<RuntimePackageCapabilityOffer> offers;
  std::optional<RuntimeTensorInlineRecordCapability> tensor_inline_record;
};

/// @brief Borrow one validated immutable package capability artifact.
class RuntimePackageCapabilityView {
 public:
  [[nodiscard]] const RuntimePackageCapabilityManifest& manifest() const noexcept {
    return *manifest_;
  }

  [[nodiscard]] bool supports_plan_schema(std::string_view value) const noexcept;
  [[nodiscard]] bool supports_profile_policy(const ProfilePolicyId& value) const noexcept;
  [[nodiscard]] bool supports_runtime_abi(std::string_view value) const noexcept;
  [[nodiscard]] bool supports_target(std::string_view value) const noexcept;
  [[nodiscard]] const RuntimePackageCapabilityOffer* find_offer(
      std::string_view id, std::uint32_t version) const noexcept;
  [[nodiscard]] const RuntimePackageCapabilityOffer* find_predicate_recomputation(
      std::string_view qualified_key, SemanticVersion semantic_version,
      std::string_view implementation_fingerprint) const;
  [[nodiscard]] const RuntimeTensorInlineRecordCapability* tensor_inline_record() const noexcept {
    return manifest_->tensor_inline_record ? &*manifest_->tensor_inline_record : nullptr;
  }

 private:
  explicit RuntimePackageCapabilityView(const RuntimePackageCapabilityManifest& manifest) noexcept
      : manifest_(&manifest) {}

  const RuntimePackageCapabilityManifest* manifest_{};
  friend expected<RuntimePackageCapabilityView, Error> validate_runtime_package_capability_manifest(
      const RuntimePackageCapabilityManifest&);
};

/// @brief Carry one package-bound endpoint monitor-path qualification into compilation.
///
/// The exact endpoint name is resolved to a plan-local port only after the
/// endpoint catalog has been normalized. Package and build identity prevent
/// evidence issued for another generated Runtime package from being reused.
struct RuntimeEndpointClockHealthQualification {
  std::string endpoint_name;
  std::string package_id;
  std::string build_id;
  std::string package_digest;
  std::chrono::nanoseconds monitor_period{};
  std::chrono::nanoseconds observation_to_call{};
  std::chrono::nanoseconds delivery_delay{};
  std::chrono::nanoseconds jitter_margin{};
  bool bounded_transition_delivery{};
  bool trace_loss_fail_closed{};
  bool unspecified_reason_qualified{};
  std::string evidence_reference;
};

/// @brief Bind one device-visible Tensor port to an explicit CUDA device.
///
/// This source-private compile input is deployment evidence, not public
/// authoring authority. Device-visible physical lowering rejects a missing,
/// duplicate, foreign, or unresolved binding rather than inferring device
/// zero. The compiler copies only the resolved identity into the immutable
/// plan and retains no reference to this record.
struct TensorDeviceBindingRecord {
  PortId port{};    ///< Canonical plan-local Tensor port identity.
  DeviceId device;  ///< Explicit nonnegative CUDA device identity.
};

/// @brief Supply deterministic target Green Context capability to compiler admission.
struct CudaGreenContextCapabilityEvidence {
  DeviceId device;                           ///< Process-local target ordinal.
  std::string deployment_device_id;          ///< Stable deployment physical identity.
  std::uint32_t total_sm_count{};            ///< Qualified total SM count.
  std::uint32_t min_sm_partition_size{};     ///< Driver minimum structured size.
  std::uint32_t sm_coscheduled_alignment{};  ///< Driver co-scheduling alignment.
  /// Qualified driver-simulated maximum group count; may be below total/quantum.
  std::uint32_t max_structured_context_count{};
  std::string evidence_id;  ///< Versioned evidence provenance.
  PlacementProbeStatus probe_status{
      PlacementProbeStatus::kRuntimeRequired};  ///< Local-pass or deferred target recheck.
};

/// @brief Borrow all immutable generated evidence captured by one compilation.
///
/// This source-private value contains no mutable provider, lookup service, path, backend object, or
/// Runtime handle.
/// Every span borrows caller-owned storage that must remain alive through the compile call. The
/// compiler copies the evidence into plan records and retains no reference to that storage.
struct RuntimeCompileEvidenceView {
  RuntimePackageCapabilityView package_capabilities;
  std::span<const RuntimeEndpointClockHealthQualification> endpoint_clock_health;
  std::span<const TensorDeviceBindingRecord> tensor_device_bindings;
  std::span<const CudaGreenContextCapabilityEvidence> cuda_green_context_capabilities;
  /// Source-private diagnostic seam that preserves the prior HoloMQ path for exact A/B checks.
  bool force_holomq_process_local_inline{};
  /// Source-private diagnostic seam that preserves event attachment for exact HoloMQ A/B checks.
  bool force_holomq_process_local_event_wake{};
  /// Source-private A/B seam: serve every process-local HoloMQ service from the pump, as before
  /// delivery inside emit() existed. Never selected by compile().
  bool force_process_local_pump_delivery{};
  /// Source-private A/B seam: keep drop-oldest fanout families on HoloMQ, as before direct fanout
  /// existed, while fanout-one edges keep direct retention. Never selected by compile().
  bool force_holomq_process_local_fanout{};
  /// Restricted, source-private fanout experiment; production defaults remain unchanged.
  ProcessLocalFanoutExperiment fanout_experiment{ProcessLocalFanoutExperiment::kDisabled};
};

/// @brief Return canonical package bytes with or without the digest field.
///
/// Arrays are sorted by their schema-v1 keys in the returned bytes without
/// mutating the source artifact. Duplicate keys are rejected by validation.
[[nodiscard]] std::string runtime_package_capability_canonical_json_v1(
    const RuntimePackageCapabilityManifest& manifest, bool include_digest);

/// @brief Return the exact capability id for one independently keyed predicate version.
[[nodiscard]] std::string runtime_predicate_capability_id(std::string_view qualified_key,
                                                          SemanticVersion semantic_version);

/// @brief Canonicalize and seal an owning generated package artifact.
///
/// This is a cold package-generation helper. It sorts every set-like record,
/// rejects duplicate or contradictory keys, and replaces `digest` with the
/// SHA-256 identity of canonical bytes that omit the digest field.
[[nodiscard]] expected<void, Error> seal_runtime_package_capability_manifest(
    RuntimePackageCapabilityManifest& manifest);

/// @brief Validate and borrow one immutable generated artifact.
[[nodiscard]] expected<RuntimePackageCapabilityView, Error>
validate_runtime_package_capability_manifest(const RuntimePackageCapabilityManifest& manifest);

/// @brief Return the build-generated package artifact used by Core compilation.
///
/// HSDK5-017 replaces this source-local generated fixture with the exactly-one
/// embedded or fixed-install provider. The returned model and matching rules
/// are already the provider-independent HSDK5-009 contract.
[[nodiscard]] const RuntimePackageCapabilityManifest& runtime_package_capability_manifest();

/// @brief Return the validated immutable view injected into the compiler.
[[nodiscard]] RuntimePackageCapabilityView runtime_package_capability_view();

/// @brief Return the generated package and qualification evidence for compilation.
/// @return Immutable source-private evidence borrowed for the duration of compilation.
[[nodiscard]] RuntimeCompileEvidenceView runtime_compile_evidence_view();

}  // namespace holoscan::detail
