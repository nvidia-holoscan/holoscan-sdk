/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler/runtime_package_capabilities.hpp"
#include "runtime/runtime_package_capability.hpp"

namespace holoscan {
namespace {

// Builds the embedded ResolvedRuntimePackageCapability from the compiled-in manifest.
// Used as the reference runtime package in nearly all capability matching tests.
[[nodiscard]] detail::ResolvedRuntimePackageCapability generated_package() {
  const auto& manifest = detail::runtime_package_capability_manifest();
  return detail::ResolvedRuntimePackageCapability{
      .manifest = manifest,
      .canonical_bytes = detail::runtime_package_capability_canonical_json_v1(manifest, true),
      .provider_kind = detail::RuntimePackageCapabilityProviderKind::kEmbedded,
  };
}

// Builds a minimal ProfileManifest that satisfies every header field in `manifest`.
// Uses a different producer_build_id to show that field is diagnostic-only, not matched.
[[nodiscard]] ProfileManifest matching_requirements(
    const detail::RuntimePackageCapabilityManifest& manifest) {
  ProfileManifest required;
  required.plan_schema_version = manifest.supported_plan_schemas.front();
  required.profile_policy_id = manifest.profile_policies.front();
  required.runtime_abi_version = manifest.runtime_abi_versions.front();
  required.target_triple = manifest.target_triples.front();
  required.producer_build_id = "a-different-compiler-build-is-diagnostic-only";
  return required;
}

// Extends matching_requirements() with the three capability IDs required by the
// interprocess channel stack (HoloMQ transport, endpoint catalog, service attach).
// Used in InterprocessChannelUsesEveryFixedProviderForm.
[[nodiscard]] ProfileManifest interprocess_channel_requirements(
    const detail::RuntimePackageCapabilityManifest& manifest) {
  ProfileManifest required = matching_requirements(manifest);
  for (const std::string_view capability_id : {
           "holoscan.runtime.channel.holomq_transport",
           "holoscan.runtime.channel_endpoint_catalog",
           "holoscan.runtime.channel_service_attach",
       }) {
    const auto offer = std::find_if(
        manifest.offers.begin(), manifest.offers.end(), [capability_id](const auto& item) {
          return item.id == capability_id && item.version == 1U;
        });
    EXPECT_NE(offer, manifest.offers.end()) << capability_id;
    if (offer != manifest.offers.end()) {
      required.required_capabilities.push_back(ProfileCapabilityRequirementRecord{
          .capability_id = offer->id,
          .version = offer->version,
          .evidence_id = offer->evidence_id,
      });
    }
  }
  return required;
}

// Assertion helper: checks that `result` holds an error whose message equals `token`.
// Reduces repetitive ASSERT_FALSE/EXPECT_EQ pairs in the rejection tests below.
template <typename Value>
void expect_error_token(const expected<Value, Error>& result, std::string_view token) {
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, token);
}

// Replaces the first occurrence of `from` with `to` in `bytes`. Used to inject
// targeted JSON mutations for the canonical-bytes rejection test.
[[nodiscard]] std::string replace_once(std::string bytes, std::string_view from,
                                       std::string_view to) {
  const std::size_t position = bytes.find(from);
  EXPECT_NE(position, std::string::npos);
  if (position != std::string::npos) {
    bytes.replace(position, from.size(), to);
  }
  return bytes;
}

// Reorders the first two JSON objects in capability_offers[]. Used to verify that the
// canonical decoder rejects non-canonical offer array ordering.
[[nodiscard]] std::string swap_first_two_offers(std::string bytes) {
  const std::string_view prefix = "\"capability_offers\":[";
  const std::size_t array = bytes.find(prefix);
  EXPECT_NE(array, std::string::npos);
  if (array == std::string::npos) {
    return bytes;
  }
  const std::size_t first_begin = bytes.find('{', array + prefix.size());
  const std::size_t first_end = bytes.find('}', first_begin);
  const std::size_t second_begin = bytes.find('{', first_end + 1U);
  const std::size_t second_end = bytes.find('}', second_begin);
  EXPECT_NE(first_begin, std::string::npos);
  EXPECT_NE(first_end, std::string::npos);
  EXPECT_NE(second_begin, std::string::npos);
  EXPECT_NE(second_end, std::string::npos);
  if (first_begin == std::string::npos || first_end == std::string::npos ||
      second_begin == std::string::npos || second_end == std::string::npos) {
    return bytes;
  }

  const std::string first = bytes.substr(first_begin, first_end - first_begin + 1U);
  const std::string second = bytes.substr(second_begin, second_end - second_begin + 1U);
  bytes.replace(first_begin, second_end - first_begin + 1U, second + ',' + first);
  return bytes;
}

// Asserts that matching `required` against `package` yields `expected_result` with an
// empty capability_matches list. Header mismatches short-circuit capability lookup.
void expect_header_mismatch(const ProfileManifest& required,
                            const detail::ResolvedRuntimePackageCapability& package,
                            detail::RuntimePackageCapabilityMatchResult expected_result,
                            std::string_view expected_token) {
  const auto matched = detail::match_runtime_package_capabilities(
      required,
      package,
      detail::RuntimePackageCapabilityMatchContext{
          .startup_attempt = 3U, .plan_generation = 7U, .run_generation = 11U});
  ASSERT_TRUE(matched) << matched.error().message;
  EXPECT_EQ(matched.value().result, expected_result);
  EXPECT_TRUE(matched.value().capability_matches.empty());
  EXPECT_EQ(detail::runtime_package_capability_match_token(matched.value().result), expected_token);
}

// REQ: CHAN-077
// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, DecodesExactGeneratedCanonicalBytes) {
  const auto package = generated_package();
  const auto decoded = detail::decode_runtime_package_capability_manifest(package.canonical_bytes);
  ASSERT_TRUE(decoded) << decoded.error().message;

  EXPECT_EQ(decoded.value().package_id, package.manifest.package_id);
  EXPECT_EQ(decoded.value().build_id, package.manifest.build_id);
  EXPECT_EQ(decoded.value().digest, package.manifest.digest);
  EXPECT_EQ(decoded.value().offers.size(), package.manifest.offers.size());
  EXPECT_EQ(detail::runtime_package_capability_canonical_json_v1(decoded.value(), true),
            package.canonical_bytes);
}

// REQ: TENSOR-CORE-013
// REQ: RUNTIME-045
TEST(RuntimePackageCapabilityTest, GeneratedPackageOffersCudaComputeStreamMaterialization) {
  // Key setup: validate the generated Runtime package capability manifest and query both CUDA
  // materialization offers by their public capability identities.
  const auto& manifest = detail::runtime_package_capability_manifest();
  const auto view = detail::validate_runtime_package_capability_manifest(manifest);
  ASSERT_TRUE(view) << view.error().message;

  const auto* offer = view->find_offer("holoscan.runtime.cuda_compute_stream_materialization", 1U);
  // Key assertion: compute-stream and Green Context offers carry their matching compiler evidence.
  ASSERT_NE(offer, nullptr);
  EXPECT_EQ(offer->evidence_id, "holoscan.compile.cuda_compute_stream_plan/v1");
  offer = view->find_offer("holoscan.runtime.cuda_green_context_materialization", 1U);
  ASSERT_NE(offer, nullptr);
  EXPECT_EQ(offer->evidence_id, "holoscan.compile.cuda_green_context_plan/v1");
}

// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, ResolvesOnlyTheTwoFixedProviderForms) {
  const auto package = generated_package();

  const auto embedded = detail::resolve_runtime_package_capability_provider(
      {.embedded_bytes = package.canonical_bytes});
  ASSERT_TRUE(embedded) << embedded.error().message;
  EXPECT_EQ(embedded.value().provider_kind,
            detail::RuntimePackageCapabilityProviderKind::kEmbedded);

  const auto installed = detail::resolve_runtime_package_capability_provider(
      {.fixed_installed_bytes = package.canonical_bytes});
  ASSERT_TRUE(installed) << installed.error().message;
  EXPECT_EQ(installed.value().provider_kind,
            detail::RuntimePackageCapabilityProviderKind::kFixedInstalled);

  const auto identical = detail::resolve_runtime_package_capability_provider(
      {.embedded_bytes = package.canonical_bytes,
       .fixed_installed_bytes = package.canonical_bytes});
  ASSERT_TRUE(identical) << identical.error().message;
  EXPECT_EQ(identical.value().provider_kind,
            detail::RuntimePackageCapabilityProviderKind::kIdenticalDual);

  expect_error_token(detail::resolve_runtime_package_capability_provider({}),
                     "PROFILE_PACKAGE_MANIFEST_MISSING");

  auto different_manifest = package.manifest;
  different_manifest.build_id = "other-build";
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(different_manifest));
  const std::string different_bytes =
      detail::runtime_package_capability_canonical_json_v1(different_manifest, true);
  expect_error_token(
      detail::resolve_runtime_package_capability_provider(
          {.embedded_bytes = package.canonical_bytes, .fixed_installed_bytes = different_bytes}),
      "PROFILE_PACKAGE_MANIFEST_AMBIGUOUS");

  const std::string malformed_bytes = ' ' + package.canonical_bytes;
  expect_error_token(
      detail::resolve_runtime_package_capability_provider({.embedded_bytes = malformed_bytes}),
      "PROFILE_PACKAGE_MANIFEST_MALFORMED");
}

// REQ: CHAN-077
// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, InterprocessChannelUsesEveryFixedProviderForm) {
  const auto generated = generated_package();
  const ProfileManifest required = interprocess_channel_requirements(generated.manifest);

  // Key setup: same canonical bytes presented as embedded, installed, and dual-mode provider.
  const std::array providers{
      detail::RuntimePackageCapabilityProviderInput{
          .embedded_bytes = generated.canonical_bytes,
      },
      detail::RuntimePackageCapabilityProviderInput{
          .fixed_installed_bytes = generated.canonical_bytes,
      },
      detail::RuntimePackageCapabilityProviderInput{
          .embedded_bytes = generated.canonical_bytes,
          .fixed_installed_bytes = generated.canonical_bytes,
      },
  };
  for (const auto& provider : providers) {
    const auto resolved = detail::resolve_runtime_package_capability_provider(provider);
    ASSERT_TRUE(resolved) << resolved.error().message;
    const auto matched = detail::match_runtime_package_capabilities(required, resolved.value(), {});
    ASSERT_TRUE(matched) << matched.error().message;
    EXPECT_TRUE(matched->compatible());
    ASSERT_EQ(matched->capability_matches.size(), required.required_capabilities.size());
    EXPECT_TRUE(std::all_of(matched->capability_matches.begin(),
                            matched->capability_matches.end(),
                            [](const auto& item) {
                              return item.result ==
                                     detail::RuntimePackageCapabilityMatchResult::kMatched;
                            }));
  }

  // Key assertion: all three provider forms satisfy channel requirements; cross-build with
  //   same sealed capabilities is also compatible; bumped version or wrong evidence rejected.
  // Build identity is diagnostic only. A differently built package with the
  // same sealed requirements is compatible; arbitrary newer capability
  // versions remain rejected by the existing exact-match path.
  auto compatible_cross_build = generated.manifest;
  compatible_cross_build.build_id = "compatible-cross-build";
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(compatible_cross_build));
  const std::string compatible_bytes =
      detail::runtime_package_capability_canonical_json_v1(compatible_cross_build, true);
  const auto installed = detail::resolve_runtime_package_capability_provider(
      {.fixed_installed_bytes = compatible_bytes});
  ASSERT_TRUE(installed) << installed.error().message;
  const auto matched_cross_build =
      detail::match_runtime_package_capabilities(required, installed.value(), {});
  ASSERT_TRUE(matched_cross_build) << matched_cross_build.error().message;
  EXPECT_TRUE(matched_cross_build->compatible());

  ProfileManifest wrong_version = required;
  const auto transport =
      std::find_if(wrong_version.required_capabilities.begin(),
                   wrong_version.required_capabilities.end(),
                   [](const auto& item) {
                     return item.capability_id == "holoscan.runtime.channel.holomq_transport";
                   });
  ASSERT_NE(transport, wrong_version.required_capabilities.end());
  ++transport->version;
  const auto rejected_version =
      detail::match_runtime_package_capabilities(wrong_version, generated, {});
  ASSERT_TRUE(rejected_version) << rejected_version.error().message;
  EXPECT_EQ(rejected_version->result,
            detail::RuntimePackageCapabilityMatchResult::kCapabilityUnsatisfied);

  ProfileManifest wrong_evidence = required;
  const auto evidence =
      std::find_if(wrong_evidence.required_capabilities.begin(),
                   wrong_evidence.required_capabilities.end(),
                   [](const auto& item) {
                     return item.capability_id == "holoscan.runtime.channel.holomq_transport";
                   });
  ASSERT_NE(evidence, wrong_evidence.required_capabilities.end());
  evidence->evidence_id += ".wrong";
  const auto rejected_evidence =
      detail::match_runtime_package_capabilities(wrong_evidence, generated, {});
  ASSERT_TRUE(rejected_evidence) << rejected_evidence.error().message;
  EXPECT_EQ(rejected_evidence->result,
            detail::RuntimePackageCapabilityMatchResult::kEvidenceUnsatisfied);
}

// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, RejectsNoncanonicalOrUnboundedBytes) {
  const auto package = generated_package();
  std::vector<std::string> malformed;
  malformed.push_back(' ' + package.canonical_bytes);
  malformed.push_back(package.canonical_bytes + '\n');
  malformed.push_back("\xEF\xBB\xBF" + package.canonical_bytes);
  malformed.push_back(
      replace_once(package.canonical_bytes, "\"schema_version\":1", "\"schema_version\":01"));
  malformed.push_back(
      replace_once(package.canonical_bytes, "\"schema_version\":1", "\"schema_version\":1.0"));
  malformed.push_back(replace_once(package.canonical_bytes,
                                   "holoscan.runtime_package_capabilities.canonical_json.v1",
                                   "holoscan.profile_policy.canonical_json.v1"));
  malformed.push_back(replace_once(
      package.canonical_bytes, "{\"build_id\":", "{\"unknown\":\"value\",\"build_id\":"));
  malformed.push_back(replace_once(package.canonical_bytes,
                                   ",\"capability_offers\":",
                                   ",\"build_id\":\"conflicting-build\","
                                   "\"capability_offers\":"));
  malformed.push_back(swap_first_two_offers(package.canonical_bytes));

  std::string wrong_digest = package.canonical_bytes;
  const std::size_t digest = wrong_digest.find("\"digest\":\"sha256:");
  ASSERT_NE(digest, std::string::npos);
  const std::size_t first_hex = digest + std::string_view{"\"digest\":\"sha256:"}.size();
  wrong_digest[first_hex] = wrong_digest[first_hex] == '0' ? '1' : '0';
  malformed.push_back(std::move(wrong_digest));

  auto no_tensor = package.manifest;
  no_tensor.tensor_inline_record.reset();
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(no_tensor));
  malformed.push_back(detail::runtime_package_capability_canonical_json_v1(no_tensor, true));
  malformed.emplace_back(detail::kRuntimePackageCapabilityMaximumBytes + 1U, 'x');

  for (const auto& bytes : malformed) {
    expect_error_token(detail::decode_runtime_package_capability_manifest(bytes),
                       "PROFILE_PACKAGE_MANIFEST_MALFORMED");
  }
}

// REQ: CHAN-077
// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, MatchesExactFactsAndRecordsCanonicalStartupEvidence) {
  const auto package = generated_package();
  ASSERT_GE(package.manifest.offers.size(), 2U);
  ProfileManifest required = matching_requirements(package.manifest);
  const auto& first = package.manifest.offers.front();
  const auto& last = package.manifest.offers.back();
  required.required_capabilities = {
      {.capability_id = last.id, .version = last.version, .evidence_id = last.evidence_id},
      {.capability_id = first.id, .version = first.version, .evidence_id = first.evidence_id},
  };
  const detail::RuntimePackageCapabilityMatchContext context{
      .startup_attempt = 19U, .plan_generation = 23U, .run_generation = 29U};

  const auto matched = detail::match_runtime_package_capabilities(required, package, context);
  ASSERT_TRUE(matched) << matched.error().message;
  ASSERT_TRUE(matched.value().compatible());
  EXPECT_EQ(matched.value().context.startup_attempt, 19U);
  EXPECT_EQ(matched.value().context.plan_generation, 23U);
  EXPECT_EQ(matched.value().context.run_generation, 29U);
  EXPECT_EQ(matched.value().package_id, package.manifest.package_id);
  EXPECT_EQ(matched.value().package_build_id, package.manifest.build_id);
  EXPECT_EQ(matched.value().package_manifest_digest, package.manifest.digest);
  ASSERT_EQ(matched.value().capability_matches.size(), 2U);
  EXPECT_LT(matched.value().capability_matches.front().required_capability_id,
            matched.value().capability_matches.back().required_capability_id);
  for (const auto& record : matched.value().capability_matches) {
    EXPECT_EQ(record.context.startup_attempt, context.startup_attempt);
    EXPECT_EQ(record.context.plan_generation, context.plan_generation);
    EXPECT_EQ(record.context.run_generation, context.run_generation);
    EXPECT_EQ(record.package_manifest_digest, package.manifest.digest);
    EXPECT_EQ(record.result, detail::RuntimePackageCapabilityMatchResult::kMatched);
    ASSERT_TRUE(record.observed_version.has_value());
    ASSERT_TRUE(record.observed_evidence_id.has_value());
    EXPECT_EQ(record.observed_version, record.required_version);
    EXPECT_EQ(record.observed_evidence_id, record.required_evidence_id);
  }
}

// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, RejectsHeaderFactsBeforeCapabilityLookup) {
  const auto package = generated_package();
  ProfileManifest required = matching_requirements(package.manifest);
  const auto& offer = package.manifest.offers.front();
  required.required_capabilities = {
      {.capability_id = offer.id, .version = offer.version, .evidence_id = offer.evidence_id}};

  auto mismatched = required;
  mismatched.plan_schema_version += ".unsupported";
  expect_header_mismatch(mismatched,
                         package,
                         detail::RuntimePackageCapabilityMatchResult::kPlanSchemaUnsupported,
                         "PROFILE_PLAN_SCHEMA_UNSUPPORTED");

  mismatched = required;
  mismatched.profile_policy_id.major += 1000U;
  expect_header_mismatch(mismatched,
                         package,
                         detail::RuntimePackageCapabilityMatchResult::kProfilePolicyUnsupported,
                         "PROFILE_POLICY_UNSUPPORTED");

  mismatched = required;
  mismatched.runtime_abi_version += ".unsupported";
  expect_header_mismatch(mismatched,
                         package,
                         detail::RuntimePackageCapabilityMatchResult::kRuntimeABIIncompatible,
                         "PROFILE_RUNTIME_ABI_INCOMPATIBLE");

  mismatched = required;
  mismatched.target_triple += ".unsupported";
  expect_header_mismatch(mismatched,
                         package,
                         detail::RuntimePackageCapabilityMatchResult::kTargetIncompatible,
                         "PROFILE_TARGET_INCOMPATIBLE");
}

// A private artifact carries one exact schema identifier. Any other identifier is rejected before
// capability lookup; there is no reader, migration path, or compatibility mode for it.
// REQ: COMPILE-069
// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, RejectsIncorrectPlanSchemaIdentifierBeforeAdmission) {
  auto manifest = detail::runtime_package_capability_manifest();
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(manifest));
  const detail::ResolvedRuntimePackageCapability package{
      .manifest = manifest,
      .canonical_bytes = detail::runtime_package_capability_canonical_json_v1(manifest, true),
      .provider_kind = detail::RuntimePackageCapabilityProviderKind::kEmbedded,
  };
  ProfileManifest required = matching_requirements(manifest);
  required.plan_schema_version += ".incorrect";
  const auto& offer = manifest.offers.front();
  required.required_capabilities = {
      {.capability_id = offer.id, .version = offer.version, .evidence_id = offer.evidence_id}};

  expect_header_mismatch(required,
                         package,
                         detail::RuntimePackageCapabilityMatchResult::kPlanSchemaUnsupported,
                         "PROFILE_PLAN_SCHEMA_UNSUPPORTED");
}

// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, RecordsEveryCapabilityAndEvidenceFailureDeterministically) {
  const auto package = generated_package();
  ASSERT_GE(package.manifest.offers.size(), 2U);
  ProfileManifest required = matching_requirements(package.manifest);
  const auto& first = package.manifest.offers.front();
  const auto& last = package.manifest.offers.back();
  // Key setup: four requirements in unsorted order — missing, wrong evidence, bumped
  //   version, and a correct match — to verify deterministic sorting and classification.
  required.required_capabilities = {
      {.capability_id = "zz.missing.capability", .version = 1U, .evidence_id = "missing/evidence"},
      {.capability_id = first.id, .version = first.version, .evidence_id = "wrong/evidence"},
      {.capability_id = last.id, .version = last.version + 1000U, .evidence_id = last.evidence_id},
      {.capability_id = last.id, .version = last.version, .evidence_id = last.evidence_id},
  };

  const auto matched = detail::match_runtime_package_capabilities(
      required, package, {.startup_attempt = 31U, .plan_generation = 37U, .run_generation = 41U});
  ASSERT_TRUE(matched) << matched.error().message;
  ASSERT_FALSE(matched.value().compatible());
  ASSERT_EQ(matched.value().capability_matches.size(), required.required_capabilities.size());
  // Key assertion: per-record results are sorted by (id, version, evidence_id).
  EXPECT_TRUE(std::is_sorted(
      matched.value().capability_matches.begin(),
      matched.value().capability_matches.end(),
      [](const auto& lhs, const auto& rhs) {
        return std::tuple{
                   lhs.required_capability_id, lhs.required_version, lhs.required_evidence_id} <
               std::tuple{
                   rhs.required_capability_id, rhs.required_version, rhs.required_evidence_id};
      }));

  const auto first_failure =
      std::find_if(matched.value().capability_matches.begin(),
                   matched.value().capability_matches.end(),
                   [](const auto& record) {
                     return record.result != detail::RuntimePackageCapabilityMatchResult::kMatched;
                   });
  ASSERT_NE(first_failure, matched.value().capability_matches.end());
  EXPECT_EQ(matched.value().result, first_failure->result);

  const auto evidence_failure =
      std::find_if(matched.value().capability_matches.begin(),
                   matched.value().capability_matches.end(),
                   [&](const auto& record) {
                     return record.required_capability_id == first.id &&
                            record.required_evidence_id == "wrong/evidence";
                   });
  ASSERT_NE(evidence_failure, matched.value().capability_matches.end());
  EXPECT_EQ(evidence_failure->result,
            detail::RuntimePackageCapabilityMatchResult::kEvidenceUnsatisfied);
  EXPECT_EQ(evidence_failure->observed_evidence_id, first.evidence_id);

  const auto version_failure = std::find_if(matched.value().capability_matches.begin(),
                                            matched.value().capability_matches.end(),
                                            [&](const auto& record) {
                                              return record.required_capability_id == last.id &&
                                                     record.required_version != last.version;
                                            });
  ASSERT_NE(version_failure, matched.value().capability_matches.end());
  EXPECT_EQ(version_failure->result,
            detail::RuntimePackageCapabilityMatchResult::kCapabilityUnsatisfied);
  EXPECT_TRUE(version_failure->observed_version.has_value());

  const auto missing = std::find_if(
      matched.value().capability_matches.begin(),
      matched.value().capability_matches.end(),
      [](const auto& record) { return record.required_capability_id == "zz.missing.capability"; });
  ASSERT_NE(missing, matched.value().capability_matches.end());
  EXPECT_FALSE(missing->observed_version.has_value());
  EXPECT_FALSE(missing->observed_evidence_id.has_value());
}

// REQ: RUNTIME-010
TEST(RuntimePackageCapabilityTest, BoundsPlanRequirementsBeforeMatching) {
  const auto package = generated_package();
  ProfileManifest required = matching_requirements(package.manifest);
  const auto& offer = package.manifest.offers.front();
  const ProfileCapabilityRequirementRecord requirement{
      .capability_id = offer.id, .version = offer.version, .evidence_id = offer.evidence_id};
  required.required_capabilities.assign(detail::kRuntimePackageCapabilityMaximumSetEntries + 1U,
                                        requirement);
  expect_error_token(detail::match_runtime_package_capabilities(required, package, {}),
                     "PROFILE_CAPABILITY_UNSATISFIED");

  required.required_capabilities = {requirement};
  required.required_capabilities.front().capability_id.assign(
      detail::kRuntimePackageCapabilityMaximumTokenBytes + 1U, 'x');
  expect_error_token(detail::match_runtime_package_capabilities(required, package, {}),
                     "PROFILE_CAPABILITY_UNSATISFIED");

  auto oversized_package = package;
  oversized_package.manifest.package_id.assign(
      detail::kRuntimePackageCapabilityMaximumTokenBytes + 1U, 'x');
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(oversized_package.manifest));
  oversized_package.canonical_bytes =
      detail::runtime_package_capability_canonical_json_v1(oversized_package.manifest, true);
  expect_error_token(detail::match_runtime_package_capabilities(
                         matching_requirements(oversized_package.manifest), oversized_package, {}),
                     "PROFILE_PACKAGE_MANIFEST_MALFORMED");
}

}  // namespace
}  // namespace holoscan
