/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_package_capability.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan::detail {
namespace {

[[nodiscard]] Error malformed_error() {
  return Error{ErrorCode::kInvalidArgument, "PROFILE_PACKAGE_MANIFEST_MALFORMED"};
}

[[nodiscard]] Error missing_error() {
  return Error{ErrorCode::kNotFound, "PROFILE_PACKAGE_MANIFEST_MISSING"};
}

[[nodiscard]] Error ambiguous_error() {
  return Error{ErrorCode::kIncompatible, "PROFILE_PACKAGE_MANIFEST_AMBIGUOUS"};
}

[[nodiscard]] bool canonical_token_byte(unsigned char value) noexcept {
  return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
         (value >= '0' && value <= '9') || value == '_' || value == '.' || value == ':' ||
         value == '/' || value == '-';
}

[[nodiscard]] std::optional<std::uint8_t> hex_nibble(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  return std::nullopt;
}

[[nodiscard]] bool parse_digest(std::string_view text, PlanDigest& digest) noexcept {
  if (text.size() != 71U || !text.starts_with("sha256:")) {
    return false;
  }
  for (std::size_t index = 0U; index < digest.value.size(); ++index) {
    const auto high = hex_nibble(text[7U + index * 2U]);
    const auto low = hex_nibble(text[8U + index * 2U]);
    if (!high || !low) {
      return false;
    }
    digest.value[index] = static_cast<std::byte>((*high << 4U) | *low);
  }
  return true;
}

class RuntimePackageCapabilityReader {
 public:
  explicit RuntimePackageCapabilityReader(std::string_view bytes) : bytes_(bytes) {}

  [[nodiscard]] bool parse(RuntimePackageCapabilityManifest& manifest) {
    if (!expect("{\"build_id\":")) {
      return false;
    }
    if (!parse_token(manifest.build_id) || !expect(",\"capability_offers\":")) {
      return false;
    }
    if (!parse_offers(manifest.offers) || !expect(",\"digest\":")) {
      return false;
    }
    if (!parse_token(manifest.digest) || !expect(",\"package_id\":")) {
      return false;
    }
    if (!parse_token(manifest.package_id) || !expect(",\"profile_policies\":")) {
      return false;
    }
    if (!parse_policies(manifest.profile_policies) || !expect(",\"runtime_abi_versions\":")) {
      return false;
    }
    if (!parse_token_array(manifest.runtime_abi_versions) || !expect(",\"schema\":")) {
      return false;
    }
    if (!parse_token(manifest.schema) || !expect(",\"schema_version\":")) {
      return false;
    }
    if (!parse_unsigned(manifest.schema_version) || !expect(",\"supported_plan_schemas\":")) {
      return false;
    }
    if (!parse_token_array(manifest.supported_plan_schemas) || !expect(",\"target_triples\":")) {
      return false;
    }
    if (!parse_token_array(manifest.target_triples) ||
        !expect(",\"tensor_inline_record_capability\":")) {
      return false;
    }
    RuntimeTensorInlineRecordCapability tensor;
    if (!parse_tensor_inline_record(tensor)) {
      return false;
    }
    manifest.tensor_inline_record = std::move(tensor);
    return consume('}') && position_ == bytes_.size();
  }

 private:
  [[nodiscard]] bool expect(std::string_view expected) {
    if (position_ > bytes_.size() || expected.size() > bytes_.size() - position_ ||
        bytes_.substr(position_, expected.size()) != expected) {
      return false;
    }
    position_ += expected.size();
    return true;
  }

  [[nodiscard]] bool consume(char expected) {
    if (position_ >= bytes_.size() || bytes_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  [[nodiscard]] bool parse_token(std::string& output) {
    if (!consume('"')) {
      return false;
    }
    const std::size_t begin = position_;
    while (position_ < bytes_.size() && bytes_[position_] != '"') {
      const auto byte = static_cast<unsigned char>(bytes_[position_]);
      if (!canonical_token_byte(byte) ||
          position_ - begin >= kRuntimePackageCapabilityMaximumTokenBytes) {
        return false;
      }
      ++position_;
    }
    if (position_ == begin || !consume('"')) {
      return false;
    }
    output.assign(bytes_.substr(begin, position_ - begin - 1U));
    return true;
  }

  template <typename Unsigned>
  [[nodiscard]] bool parse_unsigned(Unsigned& output) {
    static_assert(std::is_unsigned_v<Unsigned>);
    if (position_ >= bytes_.size() || bytes_[position_] < '0' || bytes_[position_] > '9') {
      return false;
    }
    if (bytes_[position_] == '0' && position_ + 1U < bytes_.size() &&
        bytes_[position_ + 1U] >= '0' && bytes_[position_ + 1U] <= '9') {
      return false;
    }
    std::uint64_t value = 0U;
    do {
      const std::uint64_t digit = static_cast<std::uint64_t>(bytes_[position_] - '0');
      if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
        return false;
      }
      value = value * 10U + digit;
      ++position_;
    } while (position_ < bytes_.size() && bytes_[position_] >= '0' && bytes_[position_] <= '9');
    if (value > std::numeric_limits<Unsigned>::max()) {
      return false;
    }
    output = static_cast<Unsigned>(value);
    return true;
  }

  template <typename ParseElement>
  [[nodiscard]] bool parse_array(ParseElement parse_element) {
    if (!consume('[')) {
      return false;
    }
    std::size_t count = 0U;
    if (position_ < bytes_.size() && bytes_[position_] == ']') {
      ++position_;
      return true;
    }
    while (true) {
      if (count == kRuntimePackageCapabilityMaximumSetEntries || !parse_element()) {
        return false;
      }
      ++count;
      if (consume(']')) {
        return true;
      }
      if (!consume(',')) {
        return false;
      }
    }
  }

  [[nodiscard]] bool parse_token_array(std::vector<std::string>& output) {
    return parse_array([&] {
      std::string value;
      if (!parse_token(value)) {
        return false;
      }
      output.push_back(std::move(value));
      return true;
    });
  }

  [[nodiscard]] bool parse_offers(std::vector<RuntimePackageCapabilityOffer>& output) {
    return parse_array([&] {
      RuntimePackageCapabilityOffer offer;
      if (!expect("{\"evidence_id\":") || !parse_token(offer.evidence_id) || !expect(",\"id\":") ||
          !parse_token(offer.id) || !expect(",\"version\":") || !parse_unsigned(offer.version) ||
          !consume('}')) {
        return false;
      }
      output.push_back(std::move(offer));
      return true;
    });
  }

  [[nodiscard]] bool parse_policies(std::vector<ProfilePolicyId>& output) {
    return parse_array([&] {
      ProfilePolicyId policy;
      std::string digest;
      if (!expect("{\"digest\":") || !parse_token(digest) || !expect(",\"major\":") ||
          !parse_unsigned(policy.major) || !expect(",\"minor\":") ||
          !parse_unsigned(policy.minor) || !consume('}') || !parse_digest(digest, policy.digest)) {
        return false;
      }
      output.push_back(std::move(policy));
      return true;
    });
  }

  [[nodiscard]] bool parse_tensor_inline_record(RuntimeTensorInlineRecordCapability& record) {
    return expect("{\"capability_id\":") && parse_token(record.capability_id) &&
           expect(",\"capability_version\":") && parse_unsigned(record.capability_version) &&
           expect(",\"evidence_id\":") && parse_token(record.evidence_id) &&
           expect(",\"fixed_metadata_overhead_bytes\":") &&
           parse_unsigned(record.fixed_metadata_overhead_bytes) &&
           expect(",\"inline_arm_schema_id\":") && parse_token(record.inline_arm_schema_id) &&
           expect(",\"inline_discriminator\":") && parse_token(record.inline_discriminator) &&
           expect(",\"max_fixed_record_bytes\":") &&
           parse_unsigned(record.max_fixed_record_bytes) &&
           expect(",\"max_inline_logical_bytes\":") &&
           parse_unsigned(record.max_inline_logical_bytes) && expect(",\"max_tensor_rank\":") &&
           parse_unsigned(record.max_tensor_rank) &&
           expect(",\"opaque_descriptor_arm_schema_id\":") &&
           parse_token(record.opaque_descriptor_arm_schema_id) &&
           expect(",\"opaque_descriptor_discriminator\":") &&
           parse_token(record.opaque_descriptor_discriminator) && expect(",\"provider_id\":") &&
           parse_token(record.provider_id) && expect(",\"record_format_id\":") &&
           parse_token(record.record_format_id) && expect(",\"record_format_version\":") &&
           parse_unsigned(record.record_format_version) && expect(",\"service_family_id\":") &&
           parse_token(record.service_family_id) && consume('}');
  }

  std::string_view bytes_;
  std::size_t position_{};
};

[[nodiscard]] bool same_logical_provider(const RuntimePackageCapabilityManifest& lhs,
                                         const RuntimePackageCapabilityManifest& rhs,
                                         std::string_view lhs_bytes,
                                         std::string_view rhs_bytes) noexcept {
  return lhs_bytes == rhs_bytes && lhs.package_id == rhs.package_id &&
         lhs.build_id == rhs.build_id && lhs.digest == rhs.digest;
}

[[nodiscard]] RuntimePackageCapabilityMatchReport make_match_report(
    const RuntimePackageCapabilityManifest& package, RuntimePackageCapabilityMatchContext context) {
  return RuntimePackageCapabilityMatchReport{
      .context = context,
      .package_id = package.package_id,
      .package_build_id = package.build_id,
      .package_manifest_digest = package.digest,
  };
}

[[nodiscard]] const RuntimePackageCapabilityOffer* find_offer_with_id(
    const RuntimePackageCapabilityManifest& package, std::string_view id) noexcept {
  const auto found = std::lower_bound(
      package.offers.begin(),
      package.offers.end(),
      id,
      [](const auto& offer, std::string_view candidate) { return offer.id < candidate; });
  return found != package.offers.end() && found->id == id ? &*found : nullptr;
}

[[nodiscard]] bool bounded_requirement(const ProfileCapabilityRequirementRecord& requirement) {
  return !requirement.capability_id.empty() &&
         requirement.capability_id.size() <= kRuntimePackageCapabilityMaximumTokenBytes &&
         requirement.version != 0U && !requirement.evidence_id.empty() &&
         requirement.evidence_id.size() <= kRuntimePackageCapabilityMaximumTokenBytes;
}

}  // namespace

expected<RuntimePackageCapabilityManifest, Error> decode_runtime_package_capability_manifest(
    std::string_view canonical_bytes) {
  if (canonical_bytes.empty() || canonical_bytes.size() > kRuntimePackageCapabilityMaximumBytes) {
    return make_unexpected(malformed_error());
  }

  RuntimePackageCapabilityManifest manifest;
  RuntimePackageCapabilityReader reader{canonical_bytes};
  if (!reader.parse(manifest)) {
    return make_unexpected(malformed_error());
  }
  const auto validated = validate_runtime_package_capability_manifest(manifest);
  if (!validated ||
      runtime_package_capability_canonical_json_v1(manifest, true) != canonical_bytes) {
    return make_unexpected(malformed_error());
  }
  return manifest;
}

expected<ResolvedRuntimePackageCapability, Error> resolve_runtime_package_capability_provider(
    const RuntimePackageCapabilityProviderInput& input) {
  const bool has_embedded = input.embedded_bytes.has_value();
  const bool has_installed = input.fixed_installed_bytes.has_value();
  if (!has_embedded && !has_installed) {
    return make_unexpected(missing_error());
  }
  const std::string_view embedded_bytes =
      has_embedded ? input.embedded_bytes.value() : std::string_view{};
  const std::string_view installed_bytes =
      has_installed ? input.fixed_installed_bytes.value() : std::string_view{};

  std::optional<RuntimePackageCapabilityManifest> embedded;
  std::optional<RuntimePackageCapabilityManifest> installed;
  if (has_embedded) {
    auto decoded = decode_runtime_package_capability_manifest(embedded_bytes);
    if (!decoded) {
      return make_unexpected(decoded.error());
    }
    embedded = std::move(decoded.value());
  }
  if (has_installed) {
    auto decoded = decode_runtime_package_capability_manifest(installed_bytes);
    if (!decoded) {
      return make_unexpected(decoded.error());
    }
    installed = std::move(decoded.value());
  }

  if (embedded && installed) {
    if (!same_logical_provider(*embedded, *installed, embedded_bytes, installed_bytes)) {
      return make_unexpected(ambiguous_error());
    }
    return ResolvedRuntimePackageCapability{
        .manifest = std::move(*embedded),
        .canonical_bytes = std::string{embedded_bytes},
        .provider_kind = RuntimePackageCapabilityProviderKind::kIdenticalDual,
    };
  }

  if (embedded) {
    return ResolvedRuntimePackageCapability{
        .manifest = std::move(*embedded),
        .canonical_bytes = std::string{embedded_bytes},
        .provider_kind = RuntimePackageCapabilityProviderKind::kEmbedded,
    };
  }
  return ResolvedRuntimePackageCapability{
      .manifest = std::move(*installed),
      .canonical_bytes = std::string{installed_bytes},
      .provider_kind = RuntimePackageCapabilityProviderKind::kFixedInstalled,
  };
}

std::string_view runtime_package_capability_match_token(
    RuntimePackageCapabilityMatchResult result) noexcept {
  switch (result) {
    case RuntimePackageCapabilityMatchResult::kMatched:
      return {};
    case RuntimePackageCapabilityMatchResult::kPlanSchemaUnsupported:
      return "PROFILE_PLAN_SCHEMA_UNSUPPORTED";
    case RuntimePackageCapabilityMatchResult::kProfilePolicyUnsupported:
      return "PROFILE_POLICY_UNSUPPORTED";
    case RuntimePackageCapabilityMatchResult::kRuntimeABIIncompatible:
      return "PROFILE_RUNTIME_ABI_INCOMPATIBLE";
    case RuntimePackageCapabilityMatchResult::kTargetIncompatible:
      return "PROFILE_TARGET_INCOMPATIBLE";
    case RuntimePackageCapabilityMatchResult::kCapabilityUnsatisfied:
      return "PROFILE_CAPABILITY_UNSATISFIED";
    case RuntimePackageCapabilityMatchResult::kEvidenceUnsatisfied:
      return "PROFILE_EVIDENCE_UNSATISFIED";
  }
  return "PROFILE_PACKAGE_MANIFEST_MALFORMED";
}

expected<RuntimePackageCapabilityMatchReport, Error> match_runtime_package_capabilities(
    const ProfileManifest& required, const ResolvedRuntimePackageCapability& package,
    RuntimePackageCapabilityMatchContext context) {
  if (package.canonical_bytes.empty() ||
      package.canonical_bytes.size() > kRuntimePackageCapabilityMaximumBytes) {
    return make_unexpected(malformed_error());
  }
  const auto decoded = decode_runtime_package_capability_manifest(package.canonical_bytes);
  const auto validated = validate_runtime_package_capability_manifest(package.manifest);
  if (!decoded || !validated ||
      runtime_package_capability_canonical_json_v1(package.manifest, true) !=
          package.canonical_bytes) {
    return make_unexpected(malformed_error());
  }
  if (required.required_capabilities.size() > kRuntimePackageCapabilityMaximumSetEntries ||
      std::any_of(required.required_capabilities.begin(),
                  required.required_capabilities.end(),
                  [](const auto& requirement) { return !bounded_requirement(requirement); })) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted, "PROFILE_CAPABILITY_UNSATISFIED"});
  }

  const auto view = validated.value();
  RuntimePackageCapabilityMatchReport report = make_match_report(package.manifest, context);
  if (!view.supports_plan_schema(required.plan_schema_version)) {
    report.result = RuntimePackageCapabilityMatchResult::kPlanSchemaUnsupported;
    return report;
  }
  if (!view.supports_profile_policy(required.profile_policy_id)) {
    report.result = RuntimePackageCapabilityMatchResult::kProfilePolicyUnsupported;
    return report;
  }
  if (!view.supports_runtime_abi(required.runtime_abi_version)) {
    report.result = RuntimePackageCapabilityMatchResult::kRuntimeABIIncompatible;
    return report;
  }
  if (!view.supports_target(required.target_triple)) {
    report.result = RuntimePackageCapabilityMatchResult::kTargetIncompatible;
    return report;
  }

  std::vector<const ProfileCapabilityRequirementRecord*> requirements;
  requirements.reserve(required.required_capabilities.size());
  for (const auto& requirement : required.required_capabilities) {
    requirements.push_back(&requirement);
  }
  std::sort(requirements.begin(), requirements.end(), [](const auto* lhs, const auto* rhs) {
    return std::tie(lhs->capability_id, lhs->version, lhs->evidence_id) <
           std::tie(rhs->capability_id, rhs->version, rhs->evidence_id);
  });
  report.capability_matches.reserve(requirements.size());

  for (const auto* requirement : requirements) {
    const auto* exact = view.find_offer(requirement->capability_id, requirement->version);
    const auto* same_id = find_offer_with_id(package.manifest, requirement->capability_id);
    RuntimePackageCapabilityMatchResult result = RuntimePackageCapabilityMatchResult::kMatched;
    if (exact == nullptr) {
      result = RuntimePackageCapabilityMatchResult::kCapabilityUnsatisfied;
    } else if (exact->evidence_id != requirement->evidence_id) {
      result = RuntimePackageCapabilityMatchResult::kEvidenceUnsatisfied;
    }
    if (report.result == RuntimePackageCapabilityMatchResult::kMatched &&
        result != RuntimePackageCapabilityMatchResult::kMatched) {
      report.result = result;
    }

    RuntimePackageCapabilityMatchRecord record{
        .context = context,
        .package_id = package.manifest.package_id,
        .package_build_id = package.manifest.build_id,
        .package_manifest_digest = package.manifest.digest,
        .required_capability_id = requirement->capability_id,
        .required_version = requirement->version,
        .required_evidence_id = requirement->evidence_id,
        .result = result,
    };
    const auto* observed = exact != nullptr ? exact : same_id;
    if (observed != nullptr) {
      record.observed_version = observed->version;
      record.observed_evidence_id = observed->evidence_id;
    }
    report.capability_matches.push_back(std::move(record));
  }
  return report;
}

}  // namespace holoscan::detail
