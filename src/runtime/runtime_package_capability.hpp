/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

#include "../compiler/runtime_package_capabilities.hpp"

namespace holoscan::detail {

inline constexpr std::size_t kRuntimePackageCapabilityMaximumBytes = 64U * 1024U;
inline constexpr std::size_t kRuntimePackageCapabilityMaximumTokenBytes = 512U;
inline constexpr std::size_t kRuntimePackageCapabilityMaximumSetEntries = 256U;

/// @brief Identify the exact generated provider selected for one process.
enum class RuntimePackageCapabilityProviderKind : std::uint8_t {
  kEmbedded = 0,
  kFixedInstalled = 1,
  kIdenticalDual = 2,
};

/// @brief Supply the only two permitted generated package-manifest sources.
///
/// The installed bytes must already have been read from the one build-generated
/// fixed location. This carrier accepts no path, environment variable, current
/// working directory, or search callback.
struct RuntimePackageCapabilityProviderInput {
  std::optional<std::string_view> embedded_bytes;
  std::optional<std::string_view> fixed_installed_bytes;
};

/// @brief Own one resolved, validated executing-package capability manifest.
struct ResolvedRuntimePackageCapability {
  RuntimePackageCapabilityManifest manifest;
  std::string canonical_bytes;
  RuntimePackageCapabilityProviderKind provider_kind{
      RuntimePackageCapabilityProviderKind::kEmbedded};
};

/// @brief Classify one exact plan-to-package startup compatibility result.
enum class RuntimePackageCapabilityMatchResult : std::uint8_t {
  kMatched = 0,
  kPlanSchemaUnsupported = 1,
  kProfilePolicyUnsupported = 2,
  kRuntimeABIIncompatible = 3,
  kTargetIncompatible = 4,
  kCapabilityUnsatisfied = 5,
  kEvidenceUnsatisfied = 6,
};

/// @brief Bind private match evidence to one startup attempt and run identity.
struct RuntimePackageCapabilityMatchContext {
  std::uint64_t startup_attempt{};
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
};

/// @brief Preserve one bounded required-versus-observed capability match.
struct RuntimePackageCapabilityMatchRecord {
  RuntimePackageCapabilityMatchContext context;
  std::string package_id;
  std::string package_build_id;
  std::string package_manifest_digest;
  std::string required_capability_id;
  std::uint32_t required_version{};
  std::string required_evidence_id;
  std::optional<std::uint32_t> observed_version;
  std::optional<std::string> observed_evidence_id;
  RuntimePackageCapabilityMatchResult result{RuntimePackageCapabilityMatchResult::kMatched};

  friend bool operator==(const RuntimePackageCapabilityMatchRecord&,
                         const RuntimePackageCapabilityMatchRecord&) = default;
};

/// @brief Own the deterministic startup package-compatibility evidence.
struct RuntimePackageCapabilityMatchReport {
  RuntimePackageCapabilityMatchContext context;
  std::string package_id;
  std::string package_build_id;
  std::string package_manifest_digest;
  RuntimePackageCapabilityMatchResult result{RuntimePackageCapabilityMatchResult::kMatched};
  std::vector<RuntimePackageCapabilityMatchRecord> capability_matches;

  [[nodiscard]] bool compatible() const noexcept {
    return result == RuntimePackageCapabilityMatchResult::kMatched;
  }
};

/// @brief Decode and validate exact canonical package-manifest bytes.
///
/// Decoding is bounded by the constants above and then verified by canonical
/// byte-for-byte re-encoding and digest recomputation. Noncanonical JSON,
/// duplicate or unknown fields, an incorrect codec, and a malformed digest all
/// return `PROFILE_PACKAGE_MANIFEST_MALFORMED`.
[[nodiscard]] expected<RuntimePackageCapabilityManifest, Error>
decode_runtime_package_capability_manifest(std::string_view canonical_bytes);

/// @brief Resolve exactly one embedded or fixed-installed logical provider.
///
/// Identical dual providers are accepted as one logical provider. Distinct
/// valid dual providers fail with `PROFILE_PACKAGE_MANIFEST_AMBIGUOUS`.
[[nodiscard]] expected<ResolvedRuntimePackageCapability, Error>
resolve_runtime_package_capability_provider(const RuntimePackageCapabilityProviderInput& input);

/// @brief Return the stable registry token for one compatibility failure.
/// @return An empty view for \ref RuntimePackageCapabilityMatchResult::kMatched.
[[nodiscard]] std::string_view runtime_package_capability_match_token(
    RuntimePackageCapabilityMatchResult result) noexcept;

/// @brief Match one immutable plan manifest against the resolved executing package.
///
/// Header compatibility is checked before capability lookup. Capability
/// requirements are matched exactly and emitted in canonical id/version order;
/// neither producer build identity nor unrelated package offers participate.
[[nodiscard]] expected<RuntimePackageCapabilityMatchReport, Error>
match_runtime_package_capabilities(const ProfileManifest& required,
                                   const ResolvedRuntimePackageCapability& package,
                                   RuntimePackageCapabilityMatchContext context);

}  // namespace holoscan::detail
