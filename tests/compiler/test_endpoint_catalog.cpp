/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/channel_common.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/endpoint_catalog_lowering.hpp"
#include "compiler/endpoint_clock_health_lowering.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {
namespace {

constexpr std::string_view kGeneratedImportName = "__holoscan_auto_endpoint/v1/%root/import/0";
constexpr std::size_t kCatalogSourceLimit = 16U * 1024U * 1024U;
constexpr std::size_t kJsonDepthLimit = 64U;
constexpr std::size_t kJsonNodeLimit = 100'000U;
constexpr std::size_t kJsonObjectMemberLimit = 4'096U;
constexpr std::size_t kJsonArrayElementLimit = 65'536U;
constexpr std::size_t kEndpointLimit = 4'096U;
constexpr std::size_t kClockLimit = 1'024U;
constexpr std::size_t kClockConversionLimit = 4'096U;
constexpr std::size_t kPodFieldLimit = 1'024U;
constexpr std::size_t kTensorAxisLimit = 32U;

[[nodiscard]] std::string null_array(std::size_t count) {
  std::string result{"["};
  if (count != 0U) {
    result.reserve(2U + (count * 5U));
    result += "null";
    for (std::size_t index = 1U; index < count; ++index) {
      result += ",null";
    }
  }
  result.push_back(']');
  return result;
}

[[nodiscard]] std::string null_object(std::size_t count) {
  std::string result{"{"};
  for (std::size_t index = 0U; index < count; ++index) {
    if (index != 0U) {
      result.push_back(',');
    }
    result += "\"k" + std::to_string(index) + "\":null";
  }
  result.push_back('}');
  return result;
}

[[nodiscard]] std::string nested_null(std::size_t depth) {
  std::string result(depth - 1U, '[');
  result += "null";
  result.append(depth - 1U, ']');
  return result;
}

[[nodiscard]] std::string json_with_node_count(std::size_t nodes) {
  const std::size_t leaves = nodes - 3U;
  const std::size_t first = leaves / 2U;
  const std::size_t second = leaves - first;
  return "[" + null_array(first) + "," + null_array(second) + "]";
}

[[nodiscard]] std::string replace_empty_array(std::string source, std::string_view field,
                                              std::size_t elements) {
  const std::string needle = "\"" + std::string(field) + "\":[]";
  const auto position = source.find(needle);
  if (position == std::string::npos) {
    return {};
  }
  source.replace(position, needle.size(), "\"" + std::string(field) + "\":" + null_array(elements));
  return source;
}

void expect_catalog_resource_limit(const expected<ChannelEndpointCatalog, Error>& result,
                                   std::string_view resource) {
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(result.error().message.starts_with("ENDPOINT_CATALOG_RESOURCE_LIMIT_EXCEEDED|"));
  EXPECT_NE(result.error().message.find("|resource=" + std::string(resource) + "|"),
            std::string::npos);
  EXPECT_NE(result.error().message.find("|limit="), std::string::npos);
  EXPECT_NE(result.error().message.find("|actual="), std::string::npos);
}

[[nodiscard]] std::size_t endpoint_count(const ChannelEndpointCatalog& catalog) {
  const auto* state = detail::ChannelEndpointCatalogAccess::state(catalog);
  return state == nullptr ? 0U : state->endpoints.size();
}

// Minimal embedded Graph that imports one HoloMQ topic via import_topic.
// Used by SyntheticRootMarkerCannotCollideWithAuthoredPathSegment to verify
// that the percent-encoded "%25root" marker disambiguates auto-generated names
// from the synthetic "%root" path segment reserved by the compiler.
class EndpointChildGraph final : public Graph {
 protected:
  void compose() override { (void)import_topic<std::uint32_t>("holomq://nested"); }
};

// Operator that triggers on a 1 ms clock and accepts its input with a 5 ms
// freshness window. Used by CrossDomainTemporalUseRequiresDeclaredMonitoredHealth
// and related clock-health tests to exercise cross-domain LatestWithin contracts
// that require declared clock-health monitoring.
class CrossDomainLatestOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .input(input, LatestWithin{.max_age = std::chrono::milliseconds{5}})
        .on_missing(input, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

[[nodiscard]] bool has_endpoint_diagnostic(const detail::EndpointCatalogLoweringResult& result,
                                           std::string_view token) {
  return std::any_of(result.errors.begin(), result.errors.end(), [&](const auto& diagnostic) {
    return diagnostic.token == token;
  });
}

[[nodiscard]] bool has_clock_health_diagnostic(
    const detail::EndpointClockHealthLoweringResult& result, std::string_view token) {
  return std::any_of(result.errors.begin(), result.errors.end(), [&](const auto& diagnostic) {
    return diagnostic.token == token;
  });
}

// Returns a Graph snapshot whose sole external topic carries the given endpoint
// name. Convenience factory for name-validation lowering tests that need only
// the name, not a full compile pass.
[[nodiscard]] detail::GraphSnapshotData explicit_topic_snapshot(std::string_view name) {
  Graph graph{"endpoint-shape"};
  TopicOptions options;
  options.endpoint_name = name;
  (void)graph.import_topic<std::uint32_t>("holomq://endpoint-shape", options);
  return *detail::GraphCompilerAccess::canonical_snapshot(graph);
}

// Constructs a minimal ChannelEndpointRecord for normalization tests that
// assert on payload-identity admission (POD, schema) rather than graph topology.
[[nodiscard]] detail::ChannelEndpointRecord identity_endpoint(PayloadIdentityRecord identity) {
  return detail::ChannelEndpointRecord{
      .name = "identity-endpoint",
      .uri = "holomq://identity-endpoint",
      .direction = TopicDirection::kImport,
      .name_origin = EndpointNameOrigin::kExplicit,
      .payload_identity = std::move(identity),
  };
}

// Returns a fully portable POD identity for a single uint32 field.
// Paired with identity_endpoint in PodAdmissionRequires... and related POD tests.
[[nodiscard]] PayloadIdentityRecord valid_pod_identity() {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kPod,
      .payload_kind = PayloadKind::kInlineTyped,
      .pod =
          PodIdentityRecord{
              .canonical_name = "nvidia.holoscan.test.EndpointPod",
              .version = 1U,
              .object_size = 4U,
              .object_alignment = 4U,
              .fields = {{.name = "value",
                          .byte_offset = 0U,
                          .scalar_kind = PodScalarKind::kUInt32,
                          .element_count = 1U}},
          },
  };
}

// Payload type selector for PropertyEndpoint; used by add_property_endpoint to
// generate per-seed endpoint sets in SeededPermutationsRoundTripAndOrderCompatibilityEvidence.
enum class PropertyPayload : std::uint8_t {
  kUInt32,
  kInt64,
  kFloat64,
};

// Lightweight endpoint descriptor parameterizing direction, ordinal, and payload
// type for the seeded-permutation round-trip test.
struct PropertyEndpoint {
  std::size_t ordinal{};
  TopicDirection direction{TopicDirection::kImport};
  PropertyPayload payload{PropertyPayload::kUInt32};
};

// Adds one typed endpoint (import or export) to builder using ordinal-derived
// names and bounds. The incompatible flag silently flips the payload type on
// every third endpoint to synthesize mismatched catalogs for the
// SeededPermutationsRoundTripAndOrderCompatibilityEvidence test.
template <typename T>
[[nodiscard]] expected<void, Error> add_property_endpoint(ChannelEndpointCatalog::Builder& builder,
                                                          const PropertyEndpoint& endpoint,
                                                          const TopicOptions& options) {
  const std::string suffix = std::to_string(endpoint.ordinal);
  const std::string name = "endpoint-" + suffix;
  const std::string uri = "holomq://property/" + suffix;
  if (endpoint.direction == TopicDirection::kImport) {
    return builder.import_topic<T>(name, uri, options);
  }
  return builder.export_topic<T>(name, uri, options);
}

[[nodiscard]] expected<void, Error> add_property_endpoint(ChannelEndpointCatalog::Builder& builder,
                                                          const PropertyEndpoint& endpoint,
                                                          bool incompatible) {
  TopicOptions options;
  options.attach = TopicAttachPolicy{
      .peer = endpoint.ordinal % 2U == 0U ? PeerRequirement::kOptional : PeerRequirement::kRequired,
      .max_wait = std::chrono::milliseconds{100 + static_cast<std::int64_t>(endpoint.ordinal)},
  };
  if (endpoint.direction == TopicDirection::kImport) {
    options.bounds.publisher = PublisherEndpointBounds{
        .max_loaned_samples = static_cast<std::uint32_t>(1U + endpoint.ordinal % 4U)};
  } else {
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = static_cast<std::uint32_t>(2U + endpoint.ordinal % 7U),
        .max_borrowed_samples = static_cast<std::uint32_t>(1U + endpoint.ordinal % 2U),
    };
  }

  PropertyPayload payload = endpoint.payload;
  if (incompatible && endpoint.ordinal % 3U == 0U) {
    payload =
        payload == PropertyPayload::kUInt32 ? PropertyPayload::kInt64 : PropertyPayload::kUInt32;
  }
  switch (payload) {
    case PropertyPayload::kUInt32:
      return add_property_endpoint<std::uint32_t>(builder, endpoint, options);
    case PropertyPayload::kInt64:
      return add_property_endpoint<std::int64_t>(builder, endpoint, options);
    case PropertyPayload::kFloat64:
      return add_property_endpoint<double>(builder, endpoint, options);
  }
  return make_unexpected(Error{ErrorCode::kFailure});
}

// Populates graph with the canonical cross-domain sensor path used by all
// CrossDomain* and CompilerCatalog* clock-health tests:
//   sensor_clock (ManualClock) --[AffineConversion]--> runtime_clock (RealtimeClock, default)
//   [holomq://cross-domain-sensor, cap_time=sensor_clock] --[depth=8]--> target->input
void add_cross_domain_latest_path(Graph& graph, ClockHealthOptions health) {
  const ClockKey capture_clock = graph.add_clock<ManualClock>("sensor_clock", std::int64_t{0});
  const ClockKey reference_clock = graph.add_clock<RealtimeClock>("runtime_clock");
  graph.set_default_clock(reference_clock);
  graph.add_clock_conversion(capture_clock, reference_clock, AffineClockConversion{});

  TopicOptions topic;
  topic.endpoint_name = "cross_domain_sensor";
  topic.capture_time = RequireCaptureTime{.clock = capture_clock};
  topic.clock_health = health;
  auto imported = graph.import_topic<std::uint32_t>("holomq://cross-domain-sensor", topic);
  auto target = graph.op<CrossDomainLatestOperator>("target");
  graph.add_flow(imported.output(), target->input, ConnectionOptions{.queue_depth = 8U});
}

// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, ExplicitNameProducesLookupableImmutableRecord) {
  Graph graph{"explicit-endpoint"};
  TopicOptions topic;
  topic.endpoint_name = "camera_frames";
  (void)graph.import_topic<std::uint32_t>("holomq://camera", topic);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok());
  ASSERT_EQ(endpoint_count(plan.endpoint_catalog()), 1U);
  const auto endpoint = plan.endpoint_catalog().endpoint("camera_frames");
  ASSERT_TRUE(endpoint);
  EXPECT_EQ(endpoint->name(), "camera_frames");
  EXPECT_EQ(endpoint->uri(), "holomq://camera");
  EXPECT_EQ(plan.endpoint_catalog().scope(), "explicit-endpoint");
  EXPECT_EQ(endpoint->service(), "camera_frames.service");
  EXPECT_EQ(endpoint->direction(), TopicDirection::kImport);
  EXPECT_EQ(endpoint->name_origin(), EndpointNameOrigin::kExplicit);
  EXPECT_FALSE(endpoint->generated_name().has_value());
  EXPECT_EQ(endpoint->memory_conversion(), MemoryConversionPolicy::kDefault);
  EXPECT_EQ(endpoint->attach_policy().peer, PeerRequirement::kOptional);
  EXPECT_EQ(endpoint->attach_policy().max_wait, std::chrono::seconds{2});
  ASSERT_TRUE(endpoint->bounds().publisher.has_value());
  EXPECT_EQ(endpoint->bounds().publisher->max_loaned_samples, 1U);
  EXPECT_FALSE(endpoint->bounds().subscriber.has_value());
  EXPECT_FALSE(endpoint->tensor_requirements().has_value());
  ASSERT_EQ(plan.graph().external_topics().size(), 1U);
  EXPECT_TRUE(noexcept(plan.endpoint_catalog().endpoint("missing")));
  const auto missing = plan.endpoint_catalog().endpoint("missing");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kNotFound);
}

// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, AttachBoundDefaultIsProfileAwareAndConstrainedProfilesRequireAuthorship) {
  Graph graph{"attach-profile"};
  TopicOptions options;
  options.endpoint_name = "endpoint";
  (void)graph.import_topic<std::uint32_t>("holomq://attach-profile", options);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);

  const auto standard = detail::lower_endpoint_catalog(
      *snapshot, DeploymentBinding{}, Profile::kStandard, std::nullopt);
  ASSERT_TRUE(standard.accepted());
  const auto standard_endpoint = standard.catalog.endpoint("endpoint");
  ASSERT_TRUE(standard_endpoint);
  EXPECT_EQ(standard_endpoint->attach_policy().max_wait, std::chrono::seconds{2});

  const auto regulated = detail::lower_endpoint_catalog(
      *snapshot, DeploymentBinding{}, Profile::kRegulated, std::nullopt);
  EXPECT_FALSE(regulated.accepted());
  EXPECT_TRUE(has_endpoint_diagnostic(regulated, "ENDPOINT_ATTACH_BOUND_REQUIRED"));

  options.attach.max_wait = std::chrono::milliseconds{750};
  Graph explicit_graph{"attach-profile-explicit"};
  (void)explicit_graph.import_topic<std::uint32_t>("holomq://attach-profile", options);
  const auto explicit_snapshot = detail::GraphCompilerAccess::canonical_snapshot(explicit_graph);
  const auto admitted = detail::lower_endpoint_catalog(
      *explicit_snapshot, DeploymentBinding{}, Profile::kRegulated, std::nullopt);
  ASSERT_TRUE(admitted.accepted());
  const auto admitted_endpoint = admitted.catalog.endpoint("endpoint");
  ASSERT_TRUE(admitted_endpoint);
  EXPECT_EQ(admitted_endpoint->attach_policy().max_wait, std::chrono::milliseconds{750});
}

// REQ: COMPILE-007
// REQ: COMPILE-012
TEST(EndpointCatalog, GeneratedStandardNameWarnsButRemainsDeployable) {
  Graph graph{"generated-endpoint"};
  (void)graph.import_topic<std::uint32_t>("holomq://generated");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "ENDPOINT_NAME_GENERATED_PIN_RECOMMENDED"));
  const auto warning = std::find_if(
      plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& diagnostic) {
        return diagnostic.token == "ENDPOINT_NAME_GENERATED_PIN_RECOMMENDED";
      });
  ASSERT_NE(warning, plan.diagnostics().end());
  EXPECT_EQ(warning->severity, DiagnosticSeverity::kWarning);
  EXPECT_TRUE(warning->id);
  EXPECT_NE(warning->source.graph_generation, 0U);
  EXPECT_TRUE(warning->source.graph);
  EXPECT_TRUE(warning->source.vertex);
  EXPECT_TRUE(warning->source.port);
  ASSERT_EQ(endpoint_count(plan.endpoint_catalog()), 1U);
  const auto endpoint = plan.endpoint_catalog().endpoint(kGeneratedImportName);
  ASSERT_TRUE(endpoint);
  EXPECT_EQ(endpoint->name_origin(), EndpointNameOrigin::kGenerated);
  const auto generated_name = endpoint->generated_name();
  if (!generated_name.has_value()) {
    FAIL() << "generated endpoint omitted derivation evidence";
    return;
  }
  EXPECT_EQ(generated_name->algorithm, "holoscan.auto_endpoint_name");
  EXPECT_EQ(generated_name->version, 1U);
}

// REQ: COMPILE-007
// REQ: COMPILE-012
TEST(EndpointCatalog, ConstrainedGeneratedNameRequiresExactPinAndRetainsAttemptedCatalog) {
  Graph graph{"constrained-endpoint"};
  TopicOptions topic;
  topic.attach.max_wait = std::chrono::seconds{2};
  (void)graph.import_topic<std::uint32_t>("holomq://constrained", topic);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);

  // Key setup: kRegulated profile demands an explicit pin; without one the
  // lowering rejects the catalog but still populates it for diagnostics.
  const auto rejected = detail::lower_endpoint_catalog(
      *snapshot, DeploymentBinding{}, Profile::kRegulated, std::nullopt);
  EXPECT_FALSE(rejected.accepted());
  EXPECT_TRUE(has_endpoint_diagnostic(rejected, "ENDPOINT_NAME_GENERATED_PIN_REQUIRED"));
  ASSERT_EQ(endpoint_count(rejected.catalog), 1U);
  EXPECT_TRUE(rejected.catalog.endpoint(kGeneratedImportName));
  EXPECT_FALSE(rejected.catalog.json().empty());

  DeploymentBinding pinned;
  pinned.allow_generated_endpoint_name(
      GeneratedEndpointNamePin{.final_name = kGeneratedImportName});
  const auto accepted =
      detail::lower_endpoint_catalog(*snapshot, pinned, Profile::kRegulated, std::nullopt);
  // Key assertion: providing the exact generated name as a pin lifts the rejection.
  ASSERT_TRUE(accepted.accepted());
  ASSERT_EQ(endpoint_count(accepted.catalog), 1U);
  EXPECT_TRUE(accepted.catalog.endpoint(kGeneratedImportName));
}

// REQ: COMPILE-003
// REQ: COMPILE-007
//
//   [holomq://effective-boundary-profile (auto-named)] --[depth=1]--> target
//   region "regulated-target": contains(target), profile(kRegulated)
//   Graph-level profile is kStandard; effective boundary for the imported topic
//   is kRegulated because its only consumer sits inside the regulated region.
TEST(EndpointCatalog, UsesTheBoundarysEffectiveRegionProfileForGeneratedNameAdmission) {
  Graph graph{"effective-boundary-profile"};
  auto imported = graph.import_topic<std::uint32_t>("holomq://effective-boundary-profile");
  auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(imported.output(), target->input);
  graph.region("regulated-target").add(target).profile(Profile::kRegulated);

  CompileOptions options;
  options.profile = Profile::kStandard;
  const ExecutionPlan plan = compile(graph, options);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "ENDPOINT_NAME_GENERATED_PIN_REQUIRED"));
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "ENDPOINT_NAME_GENERATED_PIN_REQUIRED";
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  EXPECT_EQ(diagnostic->severity, DiagnosticSeverity::kError);
  ASSERT_EQ(endpoint_count(plan.endpoint_catalog()), 1U);
  EXPECT_TRUE(plan.endpoint_catalog().endpoint(kGeneratedImportName));
}

// REQ: COMPILE-007
TEST(EndpointCatalog, DuplicateOrMalformedDeploymentPinsRetainAttemptedCatalog) {
  Graph graph{"invalid-pins"};
  (void)graph.import_topic<std::uint32_t>("holomq://invalid-pins");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);

  DeploymentBinding duplicate;
  duplicate.allow_generated_endpoint_name(
      GeneratedEndpointNamePin{.final_name = kGeneratedImportName});
  duplicate.allow_generated_endpoint_name(
      GeneratedEndpointNamePin{.final_name = kGeneratedImportName});
  const auto duplicate_result =
      detail::lower_endpoint_catalog(*snapshot, duplicate, Profile::kStandard, std::nullopt);
  EXPECT_FALSE(duplicate_result.accepted());
  EXPECT_TRUE(has_endpoint_diagnostic(duplicate_result, "ENDPOINT_NAME_GENERATED_PIN_DUPLICATE"));
  EXPECT_EQ(endpoint_count(duplicate_result.catalog), 1U);

  DeploymentBinding malformed;
  malformed.allow_generated_endpoint_name(GeneratedEndpointNamePin{
      .final_name = " invalid", .algorithm = "unsupported", .version = 7U});
  const auto malformed_result =
      detail::lower_endpoint_catalog(*snapshot, malformed, Profile::kStandard, std::nullopt);
  EXPECT_FALSE(malformed_result.accepted());
  EXPECT_TRUE(has_endpoint_diagnostic(malformed_result, "ENDPOINT_NAME_GENERATED_PIN_INVALID"));
  EXPECT_TRUE(has_endpoint_diagnostic(malformed_result, "ENDPOINT_NAME_GENERATED_PIN_MISMATCH"));
  EXPECT_EQ(endpoint_count(malformed_result.catalog), 1U);
}

// REQ: COMPILE-022
TEST(EndpointCatalog, SuppliedCatalogMustMatchTheDerivedSetExactly) {
  Graph generated{"catalog-authority"};
  (void)generated.import_topic<std::uint32_t>("holomq://authority");
  const ExecutionPlan authority = compile(generated);
  ASSERT_TRUE(authority.ok());

  CompileOptions exact_options;
  exact_options.endpoint_catalog = authority.endpoint_catalog();
  const ExecutionPlan exact = compile(generated, exact_options);
  ASSERT_TRUE(exact.ok()) << exact.json();
  EXPECT_FALSE(test::has_diagnostic(exact, "ENDPOINT_NAME_GENERATED_PIN_REQUIRED"));
  EXPECT_FALSE(test::has_diagnostic(exact, "ENDPOINT_NAME_GENERATED_PIN_RECOMMENDED"));
  EXPECT_EQ(exact.endpoint_catalog().digest(), authority.endpoint_catalog().digest());

  Graph different{"different-catalog"};
  TopicOptions topic;
  topic.endpoint_name = "different";
  (void)different.import_topic<std::uint32_t>("holomq://different", topic);
  CompileOptions mismatch_options;
  mismatch_options.endpoint_catalog = authority.endpoint_catalog();
  const ExecutionPlan mismatch = compile(different, mismatch_options);
  EXPECT_FALSE(mismatch.ok());
  ASSERT_TRUE(test::has_diagnostic(mismatch, "ENDPOINT_CATALOG_MISMATCH"));
  ASSERT_EQ(endpoint_count(mismatch.endpoint_catalog()), 1U);
  EXPECT_TRUE(mismatch.endpoint_catalog().endpoint("different"));
  const auto diagnostic = std::find_if(
      mismatch.diagnostics().begin(), mismatch.diagnostics().end(), [](const auto& item) {
        return item.token == "ENDPOINT_CATALOG_MISMATCH";
      });
  ASSERT_NE(diagnostic, mismatch.diagnostics().end());
  EXPECT_TRUE(diagnostic->id);
  EXPECT_NE(diagnostic->source.graph_generation, 0U);
  EXPECT_TRUE(diagnostic->source.graph);

  Graph different_scope{"different-scope"};
  (void)different_scope.import_topic<std::uint32_t>("holomq://authority");
  CompileOptions scope_options;
  scope_options.endpoint_catalog = authority.endpoint_catalog();
  const ExecutionPlan scope_mismatch = compile(different_scope, scope_options);
  EXPECT_FALSE(scope_mismatch.ok());
  EXPECT_TRUE(test::has_diagnostic(scope_mismatch, "ENDPOINT_CATALOG_MISMATCH"));
  EXPECT_EQ(scope_mismatch.endpoint_catalog().scope(), "different-scope");
}

// REQ: COMPILE-007
// REQ: COMPILE-066
TEST(EndpointCatalog, LoweringRejectsUnicodeControlsAndEdgeWhitespace) {
  const auto reject_name = [](std::string name) {
    auto snapshot = explicit_topic_snapshot("valid");
    snapshot.external_topics.front().endpoint_name = std::move(name);
    const auto result = detail::lower_endpoint_catalog(
        snapshot, DeploymentBinding{}, Profile::kStandard, std::nullopt);
    EXPECT_FALSE(result.accepted());
    EXPECT_TRUE(has_endpoint_diagnostic(result, "ENDPOINT_NAME_INVALID"));
    EXPECT_EQ(endpoint_count(result.catalog), 0U);
    EXPECT_FALSE(result.catalog.json().empty());
    ASSERT_FALSE(result.errors.empty());
    EXPECT_NE(result.errors.front().source.graph_generation, 0U);
    EXPECT_TRUE(result.errors.front().source.graph);
    EXPECT_TRUE(result.errors.front().source.vertex);
    EXPECT_TRUE(result.errors.front().source.port);
  };

  reject_name(std::string("valid") + "\xC2\x80" + "control");
  reject_name(std::string("\xC2\x85") + "control-and-space");
  reject_name(std::string("\xC2\xA0") + "nonbreaking-space");
  reject_name(std::string("em-space") + "\xE2\x80\x83");
  reject_name(std::string("\xE3\x80\x80") + "ideographic-space");

  auto interior = explicit_topic_snapshot("valid");
  interior.external_topics.front().endpoint_name =
      std::string("interior") + "\xE2\x80\x83" + "space";
  const auto accepted = detail::lower_endpoint_catalog(
      interior, DeploymentBinding{}, Profile::kStandard, std::nullopt);
  EXPECT_TRUE(accepted.accepted());
  EXPECT_EQ(endpoint_count(accepted.catalog), 1U);
}

// REQ: COMPILE-007
TEST(EndpointCatalog, LoweringRejectsReservedAndDuplicateFinalNamesWithoutRepair) {
  auto reserved = explicit_topic_snapshot("valid");
  reserved.external_topics.front().endpoint_name = "__holoscan_auto_endpoint/v1/forged";
  const auto reserved_result = detail::lower_endpoint_catalog(
      reserved, DeploymentBinding{}, Profile::kStandard, std::nullopt);
  EXPECT_FALSE(reserved_result.accepted());
  EXPECT_TRUE(has_endpoint_diagnostic(reserved_result, "ENDPOINT_NAME_RESERVED_PREFIX"));
  EXPECT_EQ(endpoint_count(reserved_result.catalog), 0U);
  EXPECT_FALSE(reserved_result.catalog.json().empty());

  // Graph topology:
  //   import(first)    export(second)  (disconnected external endpoints)
  Graph graph{"duplicate-final-name"};
  TopicOptions first;
  first.endpoint_name = "first";
  TopicOptions second;
  second.endpoint_name = "second";
  (void)graph.import_topic<std::uint32_t>("holomq://first", first);
  (void)graph.export_topic<std::uint32_t>("holomq://second", second);
  // Key setup: force the later declaration to reuse the first declaration's final name.
  auto duplicate = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_EQ(duplicate.external_topics.size(), 2U);
  duplicate.external_topics.back().endpoint_name = duplicate.external_topics.front().endpoint_name;
  const auto& later_topic = duplicate.external_topics.back();
  const auto duplicate_result = detail::lower_endpoint_catalog(
      duplicate, DeploymentBinding{}, Profile::kStandard, std::nullopt);
  EXPECT_FALSE(duplicate_result.accepted());
  EXPECT_TRUE(has_endpoint_diagnostic(duplicate_result, "ENDPOINT_NAME_DUPLICATE"));
  const auto duplicate_diagnostic = std::find_if(
      duplicate_result.errors.begin(), duplicate_result.errors.end(), [](const auto& diagnostic) {
        return diagnostic.token == "ENDPOINT_NAME_DUPLICATE";
      });
  ASSERT_NE(duplicate_diagnostic, duplicate_result.errors.end());
  // Key assertion: sorting equal endpoint names preserves authoring order and attributes the
  // duplicate to the later declaration rather than resolving the first matching name again.
  EXPECT_EQ(duplicate_diagnostic->source.vertex, later_topic.vertex);
  EXPECT_EQ(duplicate_diagnostic->source.port, PortId{later_topic.graph_port.value});
  // Duplicate stable names cannot inhabit a valid immutable catalog. The
  // lowering result retains deterministic diagnostics, but publishes no
  // catalog value whose lookup semantics would be ambiguous.
  EXPECT_EQ(endpoint_count(duplicate_result.catalog), 0U);
}

// REQ: COMPILE-007
// REQ: COMPILE-012
//
//   parent graph: [holomq://root] (auto-named)     → %root/import/0
//   embedded "%root" subgraph: [holomq://nested]   → %25root/import/0
//   The compiler percent-encodes authored path segments that happen to equal
//   the synthetic "%root" marker so the two names remain distinct.
TEST(EndpointCatalog, SyntheticRootMarkerCannotCollideWithAuthoredPathSegment) {
  Graph graph{"root-marker-collision"};
  (void)graph.import_topic<std::uint32_t>("holomq://root");
  (void)graph.embed<EndpointChildGraph>("%root");

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto lowered = detail::lower_endpoint_catalog(
      *snapshot, DeploymentBinding{}, Profile::kStandard, std::nullopt);
  ASSERT_TRUE(lowered.accepted());
  ASSERT_EQ(endpoint_count(lowered.catalog), 2U);
  EXPECT_TRUE(lowered.catalog.endpoint(kGeneratedImportName));
  constexpr std::string_view kEscapedNestedImport = "__holoscan_auto_endpoint/v1/%25root/import/0";
  EXPECT_TRUE(lowered.catalog.endpoint(kEscapedNestedImport));
}

// REQ: COMPILE-007
// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, StandaloneBuilderRoundTripsTheCompleteCanonicalCatalog) {
  using namespace std::chrono_literals;

  TopicOptions imported_options;
  imported_options.capture_time = RequireCaptureTime{.clock = ClockKey{"sensor_clock"}};
  imported_options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5ms,
  };
  imported_options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  TopicOptions exported_options;
  exported_options.capture_time = RequireCaptureTime{.clock = ClockKey{"sensor_clock"}};
  exported_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 4U,
      .max_borrowed_samples = 2U,
  };

  // Key setup: one import with capture_time+clock_health+publisher bounds; one export
  // with capture_time and subscriber bounds. Both share a named sensor_clock reference.
  auto builder = ChannelEndpointCatalog::builder("standalone-roundtrip");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("camera", "holomq://camera", imported_options));
  ASSERT_TRUE(builder.export_topic<std::uint32_t>("result", "holomq://result", exported_options));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);
  EXPECT_EQ(built->json().find("boundary_port"), std::string::npos);
  EXPECT_NE(built->json().find("\"clocks\":[{"), std::string::npos);
  EXPECT_NE(built->json().find("\"services\":[{"), std::string::npos);

  // Key assertion: JSON round-trip through parse_json produces identical bytes
  // and digest; internal state exposes the same clock name, bounds, and
  // service-static fields that were authored via the builder.
  auto parsed = ChannelEndpointCatalog::parse_json(built->json());
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->digest(), built->digest());
  EXPECT_EQ(parsed->json(), built->json());
  ASSERT_TRUE(parsed->raw_source_sha256().has_value());
  ASSERT_TRUE(parsed->raw_source_size_bytes().has_value());
  EXPECT_EQ(*parsed->raw_source_size_bytes(), built->json().size());
  const auto camera = parsed->endpoint("camera");
  ASSERT_TRUE(camera);
  ASSERT_TRUE(camera->capture_time().has_value());
  EXPECT_EQ(camera->capture_time()->clock().name(), "sensor_clock");
  EXPECT_EQ(camera->clock_health().capability(), ClockHealthCapability::kStatusReports);
  EXPECT_EQ(camera->clock_health().requirement(), ClockHealthRequirement::kFreshHealthyReport);
  EXPECT_EQ(camera->clock_health().max_report_age(), 5ms);
  const auto result_endpoint = parsed->endpoint("result");
  ASSERT_TRUE(result_endpoint);
  ASSERT_TRUE(result_endpoint->capture_time().has_value());
  EXPECT_EQ(result_endpoint->capture_time()->clock().name(), "sensor_clock");

  const auto* state = detail::ChannelEndpointCatalogAccess::state(*parsed);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->endpoints.size(), 2U);
  const auto& camera_record = state->endpoints.front();
  ASSERT_EQ(camera_record.name, "camera");
  EXPECT_EQ(camera_record.service_static.id, "camera.service");
  EXPECT_EQ(camera_record.service_static.payload_identity.kind, PayloadIdentityKind::kScalar);
  EXPECT_EQ(camera_record.service_static.max_subscriber_depth, 1U);
  EXPECT_EQ(camera_record.service_static.max_subscriber_borrowed, 1U);
  EXPECT_EQ(camera_record.service_static.publisher_history_depth, 0U);
  EXPECT_TRUE(camera_record.service_static.event_wake);
  ASSERT_TRUE(camera_record.publisher_request.has_value());
  EXPECT_EQ(camera_record.publisher_request->max_loaned_samples, 2U);
  EXPECT_EQ(camera_record.publisher_request->capture_time_clock->name(), "sensor_clock");
  EXPECT_FALSE(camera_record.subscriber_request.has_value());

  const auto& result_record = state->endpoints.back();
  ASSERT_EQ(result_record.name, "result");
  EXPECT_FALSE(result_record.publisher_request.has_value());
  ASSERT_TRUE(result_record.capture_time_clock.has_value());
  EXPECT_EQ(result_record.capture_time_clock->name(), "sensor_clock");
  ASSERT_TRUE(result_record.subscriber_request.has_value());
  EXPECT_EQ(result_record.subscriber_request->queue_depth, 4U);
  EXPECT_EQ(result_record.subscriber_request->max_borrowed_samples, 2U);
  EXPECT_EQ(result_record.service_static.max_subscriber_depth, 4U);
  EXPECT_EQ(result_record.service_static.max_subscriber_borrowed, 2U);
  EXPECT_EQ(parsed->json().find("\"profile\""), std::string::npos);
}

// REQ: COMPILE-007
// REQ: COMPILE-022
// REQ: EXTERNAL-010
// REQ: SCHEMA-003
// REQ: SCHEMA-004
TEST(EndpointCatalog, DescriptorTensorSchemaFieldsRoundTripAsServiceIdentity) {
  constexpr std::array<std::uint16_t, 2U> kTensorFieldIds{0U, 4U};
  const auto kTransportProfile =
      make_descriptor_tensor_transport_profile(std::span{kTensorFieldIds}, 256U);
  SchemaIdentityRecord schema{
      .canonical_name = "nvidia.holoscan.test.ExternalDescriptorSchema",
      .compatibility_epoch = 1U,
      .family_id = derive_schema_family_id("nvidia.holoscan.test.ExternalDescriptorSchema", 1U),
      .file_identifier = make_flatbuffer_file_identifier("EDSC"),
  };
  auto endpoint = identity_endpoint(PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kSchema,
      .payload_kind = PayloadKind::kDescriptorBacked,
      .transport_profile_tag = schema_transport_profile_tag(kTransportProfile),
      .schema = schema,
  });
  endpoint.schema_writer_max_serialized_bytes = 256U;
  endpoint.schema_identity = schema;
  // Key setup: encode two descriptor Tensor fields with distinct IDs, indices, memory kinds,
  // byte bounds, and memory-conversion policies into one service identity.
  endpoint.tensor_fields = {
      DescriptorTensorTopicFieldRecord{
          .field_id = 0U,
          .tensor_index = 0U,
          .tensor =
              TensorBoundaryContractRecord{
                  .memory_kind = MemoryKind::kCudaDevice,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
                  .rank = 1U,
                  .max_rank = 1U,
                  .max_byte_span = 4096U,
                  .strides = TensorStridePolicy::kContiguousOnly,
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable,
                  .memory_conversion = MemoryConversionPolicy::kDisable,
              },
      },
      DescriptorTensorTopicFieldRecord{
          .field_id = 4U,
          .tensor_index = 1U,
          .tensor =
              TensorBoundaryContractRecord{
                  .memory_kind = MemoryKind::kPinnedHost,
                  .dtype = DLDataType{kDLUInt, 16U, 1U},
                  .rank = 2U,
                  .max_rank = 2U,
                  .max_byte_span = 8192U,
                  .strides = TensorStridePolicy::kContiguousOnly,
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable,
                  .memory_conversion = MemoryConversionPolicy::kAllowIfSafe,
              },
      },
  };

  auto normalized = detail::ChannelEndpointCatalogAccess::normalize("descriptor-tensor-fields",
                                                                    {std::move(endpoint)});
  ASSERT_TRUE(normalized) << normalized.error().message;
  EXPECT_NE(normalized->json().find("\"descriptor_tensor_fields\":[{"), std::string::npos);
  EXPECT_NE(normalized->json().find("\"memory_conversion\":\"disable\""), std::string::npos);

  auto parsed = ChannelEndpointCatalog::parse_json(normalized->json());
  ASSERT_TRUE(parsed) << parsed.error().message;
  // Key assertion: canonical JSON round-trip retains every per-field contract value and rejects
  // a duplicate field ID rather than normalizing the inventory ambiguously.
  EXPECT_EQ(parsed->json(), normalized->json());
  EXPECT_EQ(parsed->digest(), normalized->digest());
  const auto* state = detail::ChannelEndpointCatalogAccess::state(*parsed);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->endpoints.size(), 1U);
  const auto expected_metadata_bucket = channel_internal::inline_payload_bucket_size(256U);
  ASSERT_TRUE(expected_metadata_bucket);
  EXPECT_EQ(state->endpoints.front().service_static.required_data_frame_bytes, 256U);
  EXPECT_EQ(state->endpoints.front().service_static.selected_inline_bucket_bytes,
            *expected_metadata_bucket);
  EXPECT_EQ(state->endpoints.front().service_static.bucket_usable_serialized_bytes,
            *expected_metadata_bucket);
  EXPECT_EQ(state->endpoints.front().service_static.payload_identity.transport_profile_tag,
            schema_transport_profile_tag(kTransportProfile));
  const auto& fields = state->endpoints.front().service_static.tensor_fields;
  ASSERT_EQ(fields.size(), 2U);
  EXPECT_EQ(fields[0].field_id, 0U);
  EXPECT_EQ(fields[0].tensor.memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(fields[0].tensor.memory_conversion, MemoryConversionPolicy::kDisable);
  EXPECT_EQ(fields[1].field_id, 4U);
  EXPECT_EQ(fields[1].tensor_index, 1U);
  EXPECT_EQ(fields[1].tensor.max_byte_span, 8192U);
  EXPECT_EQ(fields[1].tensor.memory_conversion, MemoryConversionPolicy::kAllowIfSafe);

  auto incompatible_profile = state->endpoints.front();
  ++incompatible_profile.payload_identity.transport_profile_tag;
  auto profile_catalog = detail::ChannelEndpointCatalogAccess::normalize(
      "descriptor-tensor-fields", {std::move(incompatible_profile)});
  ASSERT_TRUE(profile_catalog) << profile_catalog.error().message;
  EXPECT_FALSE(normalized->compatibility_with(*profile_catalog).compatible());

  auto duplicate = state->endpoints.front();
  duplicate.tensor_fields[1].field_id = duplicate.tensor_fields[0].field_id;
  auto rejected = detail::ChannelEndpointCatalogAccess::normalize("descriptor-tensor-fields",
                                                                  {std::move(duplicate)});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().message, "ENDPOINT_TENSOR_POLICY_INVALID");

  std::string mismatched_bucket = normalized->json();
  const std::string bucket_field =
      "\"selected_inline_bucket_bytes\":" + std::to_string(*expected_metadata_bucket);
  const auto bucket_offset = mismatched_bucket.find(bucket_field);
  ASSERT_NE(bucket_offset, std::string::npos);
  mismatched_bucket.replace(
      bucket_offset,
      bucket_field.size(),
      "\"selected_inline_bucket_bytes\":" + std::to_string(*expected_metadata_bucket / 2U));
  auto rejected_bucket = ChannelEndpointCatalog::parse_json(mismatched_bucket);
  ASSERT_FALSE(rejected_bucket);
  EXPECT_EQ(rejected_bucket.error().message, "ENDPOINT_SERVICE_INVALID");
}

// REQ: COMPILE-007
// REQ: COMPILE-022
// REQ: CHAN-115
TEST(EndpointCatalog, DropOldestQueuePolicyRoundTripsAsServiceStaticIdentity) {
  auto drop_oldest_endpoint = identity_endpoint(PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kScalar,
      .payload_kind = PayloadKind::kInlineTyped,
      .scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt32},
  });
  drop_oldest_endpoint.name = "drop-oldest-endpoint";
  drop_oldest_endpoint.uri = "holomq://drop-oldest-endpoint";
  drop_oldest_endpoint.service = "drop-oldest-endpoint.service";
  drop_oldest_endpoint.queue_policy = QueuePolicy::kDropOldest;
  auto drop_oldest = detail::ChannelEndpointCatalogAccess::normalize(
      "drop-oldest-queue-policy", {std::move(drop_oldest_endpoint)});
  ASSERT_TRUE(drop_oldest) << drop_oldest.error().message;
  EXPECT_NE(drop_oldest->json().find("\"policy\":\"drop_oldest\""), std::string::npos);

  auto parsed = ChannelEndpointCatalog::parse_json(drop_oldest->json());
  ASSERT_TRUE(parsed) << parsed.error().message;
  EXPECT_EQ(parsed->digest(), drop_oldest->digest());
  EXPECT_EQ(parsed->json(), drop_oldest->json());
  const auto* parsed_state = detail::ChannelEndpointCatalogAccess::state(*parsed);
  ASSERT_NE(parsed_state, nullptr);
  ASSERT_EQ(parsed_state->endpoints.size(), 1U);
  EXPECT_EQ(parsed_state->endpoints.front().service_static.queue_policy, QueuePolicy::kDropOldest);

  std::string latest_json = drop_oldest->json();
  const auto policy_position = latest_json.find("\"policy\":\"drop_oldest\"");
  ASSERT_NE(policy_position, std::string::npos);
  latest_json.replace(policy_position,
                      std::string_view{"\"policy\":\"drop_oldest\""}.size(),
                      "\"policy\":\"latest\"");
  const auto latest = ChannelEndpointCatalog::parse_json(latest_json);
  ASSERT_FALSE(latest);
  EXPECT_EQ(latest.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(latest.error().message, "ENDPOINT_SERVICE_INVALID");

  auto blocking_endpoint = identity_endpoint(PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kScalar,
      .payload_kind = PayloadKind::kInlineTyped,
      .scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt32},
  });
  blocking_endpoint.name = "drop-oldest-endpoint";
  blocking_endpoint.uri = "holomq://drop-oldest-endpoint";
  blocking_endpoint.service = "drop-oldest-endpoint.service";
  auto blocking = detail::ChannelEndpointCatalogAccess::normalize("drop-oldest-queue-policy",
                                                                  {std::move(blocking_endpoint)});
  ASSERT_TRUE(blocking) << blocking.error().message;
  const auto drop_oldest_view = drop_oldest->endpoint("drop-oldest-endpoint");
  const auto blocking_view = blocking->endpoint("drop-oldest-endpoint");
  ASSERT_TRUE(drop_oldest_view);
  ASSERT_TRUE(blocking_view);
  EXPECT_NE(detail::ChannelEndpointCatalogAccess::backend_service_binding(*drop_oldest_view),
            detail::ChannelEndpointCatalogAccess::backend_service_binding(*blocking_view));

  const auto compatibility = drop_oldest->compatibility_with(*blocking);
  EXPECT_FALSE(compatibility.compatible());
  EXPECT_NE(compatibility.json().find("ENDPOINT_SERVICE_QUEUE_POLICY_MISMATCH"), std::string::npos);
  EXPECT_NE(compatibility.json().find("\"required\":\"drop_oldest\""), std::string::npos);
}

// Logical catalog service identities retain the same 255-byte limit under both queue policies.
// Runtime derives the protected drop-oldest terminal identity from the fixed physical binding,
// while the reserved suffix remains unavailable to every public logical service.
// REQ: CHAN-132
// REQ: EXTERNAL-009
TEST(EndpointCatalog, LogicalServiceIdentityRetainsFullCatalogLimit) {
  constexpr std::size_t kLogicalServiceNameCapacity = 255U;
  const auto make_endpoint = [](std::string service, const QueuePolicy policy) {
    auto endpoint = identity_endpoint(PayloadIdentityRecord{
        .kind = PayloadIdentityKind::kScalar,
        .payload_kind = PayloadKind::kInlineTyped,
        .scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt32},
    });
    endpoint.name = "terminal-service-identity";
    endpoint.uri = "holomq://terminal-service-identity";
    endpoint.service = std::move(service);
    endpoint.queue_policy = policy;
    return endpoint;
  };

  // Key setup: exercise the full logical limit and its first invalid length under both policies.
  for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    auto maximum = detail::ChannelEndpointCatalogAccess::normalize(
        "full-logical-service-name",
        {make_endpoint(std::string(kLogicalServiceNameCapacity, 's'), policy)});
    ASSERT_TRUE(maximum) << maximum.error().message;
    const auto endpoint = maximum->endpoint("terminal-service-identity");
    ASSERT_TRUE(endpoint);
    EXPECT_EQ(endpoint->service().size(), kLogicalServiceNameCapacity);
    EXPECT_EQ(detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint).size(), 30U);

    auto oversized = detail::ChannelEndpointCatalogAccess::normalize(
        "oversized-logical-service-name",
        {make_endpoint(std::string(kLogicalServiceNameCapacity + 1U, 's'), policy)});
    ASSERT_FALSE(oversized);
    EXPECT_EQ(oversized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(oversized.error().message, "ENDPOINT_SERVICE_INVALID");

    auto reserved = detail::ChannelEndpointCatalogAccess::normalize(
        "reserved-terminal-identity", {make_endpoint("public.hsdk-terminal-v1", policy)});

    // Key assertion: the logical maximum normalizes to the fixed short physical binding, while
    // an oversized or reserved logical identity rejects before catalog publication.
    ASSERT_FALSE(reserved);
    EXPECT_EQ(reserved.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(reserved.error().message, "ENDPOINT_SERVICE_INVALID");
  }
}

// REQ: COMPILE-022
// REQ: COMPILE-024
TEST(EndpointCatalog, SeededPermutationsRoundTripAndOrderCompatibilityEvidence) {
  constexpr std::size_t kSeedCount = 64U;
  const auto build_catalog = [](std::span<const PropertyEndpoint> endpoints,
                                std::span<const std::size_t>
                                    order,
                                bool incompatible) -> expected<ChannelEndpointCatalog, Error> {
    auto builder = ChannelEndpointCatalog::builder("property-catalog");
    for (const std::size_t index : order) {
      auto added = add_property_endpoint(builder, endpoints[index], incompatible);
      if (!added) {
        return make_unexpected(std::move(added).error());
      }
    }
    return std::move(builder).build();
  };

  // Key setup: for each seed, build a catalog in authored order and again in
  // reversed order. The catalog normalizes endpoint order canonically, so both
  // must produce identical JSON and digest. Incompatible variants (flipped type
  // on every third endpoint) must be stably ordered for compatibility reports.
  for (std::size_t seed = 0U; seed < kSeedCount; ++seed) {
    const std::size_t count = 1U + seed % 16U;
    std::vector<PropertyEndpoint> endpoints;
    endpoints.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
      endpoints.push_back(PropertyEndpoint{
          .ordinal = index,
          .direction =
              (seed + index) % 2U == 0U ? TopicDirection::kImport : TopicDirection::kExport,
          .payload = static_cast<PropertyPayload>((seed + index) % 3U),
      });
    }

    std::vector<std::size_t> authored_order(count);
    std::iota(authored_order.begin(), authored_order.end(), 0U);
    std::mt19937_64 generator{0x484F4C4F5343414EULL + seed};
    std::shuffle(authored_order.begin(), authored_order.end(), generator);
    std::vector<std::size_t> reverse_order = authored_order;
    std::reverse(reverse_order.begin(), reverse_order.end());

    auto authored = build_catalog(endpoints, authored_order, false);
    ASSERT_TRUE(authored) << "seed=" << seed << ": " << authored.error().message;
    auto reversed = build_catalog(endpoints, reverse_order, false);
    ASSERT_TRUE(reversed) << "seed=" << seed << ": " << reversed.error().message;
    // Key assertion: canonical JSON is insertion-order-independent.
    EXPECT_EQ(authored->json(), reversed->json()) << "seed=" << seed;
    EXPECT_EQ(authored->digest(), reversed->digest()) << "seed=" << seed;

    auto parsed = ChannelEndpointCatalog::parse_json(authored->json());
    ASSERT_TRUE(parsed) << "seed=" << seed << ": " << parsed.error().message;
    EXPECT_EQ(parsed->json(), authored->json()) << "seed=" << seed;
    EXPECT_EQ(parsed->digest(), authored->digest()) << "seed=" << seed;

    auto incompatible_authored = build_catalog(endpoints, authored_order, true);
    ASSERT_TRUE(incompatible_authored)
        << "seed=" << seed << ": " << incompatible_authored.error().message;
    auto incompatible_reversed = build_catalog(endpoints, reverse_order, true);
    ASSERT_TRUE(incompatible_reversed)
        << "seed=" << seed << ": " << incompatible_reversed.error().message;
    const auto authored_report = authored->compatibility_with(*incompatible_authored);
    const auto reversed_report = reversed->compatibility_with(*incompatible_reversed);
    EXPECT_EQ(authored_report.json(), reversed_report.json()) << "seed=" << seed;
    EXPECT_FALSE(authored_report.compatible()) << "seed=" << seed;
  }
}

// REQ: COMPILE-022
TEST(EndpointCatalog, StandaloneBuilderAcceptsTheExactEndpointLimit) {
  auto builder = ChannelEndpointCatalog::builder("standalone-limit");
  for (std::size_t index = 0U; index < kEndpointLimit; ++index) {
    const std::string suffix = std::to_string(index);
    auto added =
        builder.import_topic<std::uint32_t>("endpoint-" + suffix, "holomq://endpoint-" + suffix);
    ASSERT_TRUE(added) << "index=" << index << ": " << added.error().message;
  }

  auto over_limit =
      builder.import_topic<std::uint32_t>("endpoint-over-limit", "holomq://endpoint-over-limit");
  ASSERT_FALSE(over_limit);
  EXPECT_EQ(over_limit.error().code, ErrorCode::kResourceExhausted);
  EXPECT_NE(over_limit.error().message.find("resource=normalized_endpoints"), std::string::npos);

  auto built = std::move(builder).build();
  ASSERT_TRUE(built) << built.error().message;
  const auto* state = detail::ChannelEndpointCatalogAccess::state(*built);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->endpoints.size(), kEndpointLimit);
}

// REQ: COMPILE-007
// REQ: COMPILE-023
//
//   capture-clock (ManualClock) --[AffineConversion]--> health-age-clock (RealtimeClock, default)
//   [holomq://ordinary,   no cap_time, health=kUnavailable]
//   [holomq://monitored,  cap_time=capture-clock, health=kStatusReports+kFreshHealthyReport, 5ms]
TEST(EndpointCatalog, ClockHealthModesNormalizeAndCompileWithExplicitCapabilityEvidence) {
  using namespace std::chrono_literals;

  Graph graph{"clock-health-plan"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey age_clock = graph.add_clock<RealtimeClock>("health-age-clock");
  graph.set_default_clock(age_clock);

  TopicOptions ordinary;
  ordinary.endpoint_name = "ordinary";
  const auto ordinary_import = graph.import_topic<std::uint32_t>("holomq://ordinary", ordinary);

  TopicOptions monitored;
  monitored.endpoint_name = "monitored";
  monitored.capture_time = RequireCaptureTime{.clock = capture_clock};
  monitored.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5ms,
  };
  const auto monitored_import = graph.import_topic<std::uint32_t>("holomq://monitored", monitored);
  const auto ordinary_sink = graph.op<test::CompileTestOperator>("ordinary-sink");
  const auto monitored_sink = graph.op<test::CompileTestOperator>("monitored-sink");
  graph.add_flow(ordinary_import.output(), ordinary_sink->input);
  graph.add_flow(monitored_import.output(), monitored_sink->input);
  graph.partition("ordinary").add(ordinary_sink);
  graph.partition("monitored").add(monitored_sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.endpoint_clock_health().size(), 2U);
  const auto* catalog_state = detail::ChannelEndpointCatalogAccess::state(plan.endpoint_catalog());
  ASSERT_NE(catalog_state, nullptr);
  const auto endpoint_port = [&](std::string_view name) {
    const auto found = std::find_if(catalog_state->endpoints.begin(),
                                    catalog_state->endpoints.end(),
                                    [&](const auto& endpoint) { return endpoint.name == name; });
    return found == catalog_state->endpoints.end() ? PortId{} : found->boundary_port;
  };
  const auto ordinary_record =
      std::find_if(plan.endpoint_clock_health().begin(),
                   plan.endpoint_clock_health().end(),
                   [&](const auto& row) { return row.endpoint == endpoint_port("ordinary"); });
  ASSERT_NE(ordinary_record, plan.endpoint_clock_health().end());
  EXPECT_EQ(ordinary_record->capability, ClockHealthCapability::kUnavailable);
  EXPECT_EQ(ordinary_record->requirement, ClockHealthRequirement::kNotRequired);
  EXPECT_FALSE(ordinary_record->max_report_age.has_value());
  EXPECT_FALSE(ordinary_record->health_age_clock.valid());
  EXPECT_EQ(ordinary_record->fixed_report_ingress_capacity, 0U);
  EXPECT_EQ(ordinary_record->trace_loss_disposition,
            EndpointClockHealthTraceLossDisposition::kNotApplicable);

  const auto monitored_record =
      std::find_if(plan.endpoint_clock_health().begin(),
                   plan.endpoint_clock_health().end(),
                   [&](const auto& row) { return row.endpoint == endpoint_port("monitored"); });
  ASSERT_NE(monitored_record, plan.endpoint_clock_health().end());
  EXPECT_EQ(monitored_record->capability, ClockHealthCapability::kStatusReports);
  EXPECT_EQ(monitored_record->requirement, ClockHealthRequirement::kFreshHealthyReport);
  EXPECT_EQ(monitored_record->max_report_age, 5ms);
  EXPECT_TRUE(monitored_record->capture_clock.valid());
  EXPECT_TRUE(monitored_record->health_age_clock.valid());
  // Key assertion: monitored endpoint has two distinct clocks (capture vs health-age)
  // and its witness event ID is the canonical holoscan clock-health update token.
  EXPECT_NE(monitored_record->capture_clock, monitored_record->health_age_clock);
  EXPECT_EQ(monitored_record->fixed_report_ingress_capacity, 1U);
  EXPECT_EQ(monitored_record->fixed_update_capacity, 4U);
  EXPECT_EQ(monitored_record->per_dispatch_report_budget, 1U);
  EXPECT_EQ(monitored_record->producer_epoch_source,
            EndpointProducerEpochSource::kAttachmentGeneration);
  EXPECT_EQ(monitored_record->monitor_path_kind,
            EndpointClockHealthMonitorPathKind::kPublisherSerialized);
  EXPECT_EQ(monitored_record->trace_loss_disposition,
            EndpointClockHealthTraceLossDisposition::kFailClosed);
  EXPECT_EQ(monitored_record->clock_health_update_witness_id,
            "holoscan.event.clock-health-update.v1");

  constexpr std::array<std::string_view, 10> kReasonTokens{
      "none",
      "synchronization_lock_lost",
      "phase_error_limit_exceeded",
      "rate_error_limit_exceeded",
      "calibration_expired",
      "clock_stalled",
      "timestamp_regression",
      "timestamp_source_fault",
      "monitor_fault",
      "unspecified",
  };
  for (const auto token : kReasonTokens) {
    EXPECT_NE(plan.json().find(std::string("\"") + std::string(token) + "\""),
              std::string_view::npos)
        << token;
  }
}

// REQ: COMPILE-007
// REQ: COMPILE-023
TEST(EndpointCatalog, InvalidClockHealthPairsAndRolesFailWithoutNormalization) {
  using namespace std::chrono_literals;

  const auto expect_invalid = [](TopicDirection direction, ClockHealthOptions health) {
    TopicOptions options;
    options.clock_health = health;
    auto builder = ChannelEndpointCatalog::builder("invalid-clock-health");
    auto added =
        direction == TopicDirection::kImport
            ? builder.import_topic<std::uint32_t>("endpoint", "holomq://endpoint", options)
            : builder.export_topic<std::uint32_t>("endpoint", "holomq://endpoint", options);
    ASSERT_FALSE(added);
    const std::string expected =
        "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID|path=/builder/endpoints/endpoint/clock_health|"
        "required=unavailable+not_required(age=absent,any_role)|status_reports+fresh_healthy_"
        "report(age=positive,external_publisher)|provided=capability=" +
        std::string(health.capability == ClockHealthCapability::kStatusReports ? "status_reports"
                                                                               : "unavailable") +
        ";requirement=" +
        std::string(health.requirement == ClockHealthRequirement::kFreshHealthyReport
                        ? "fresh_healthy_report"
                        : "not_required") +
        ";max_report_age_ns=" +
        (health.max_report_age ? std::to_string(health.max_report_age->count()) : "absent") +
        ";role=" +
        (direction == TopicDirection::kImport ? "external_publisher" : "external_subscriber");
    EXPECT_EQ(added.error().message, expected);
    auto unchanged = std::move(builder).build();
    ASSERT_TRUE(unchanged);
    const auto absent = unchanged->endpoint("endpoint");
    ASSERT_FALSE(absent);
    EXPECT_EQ(absent.error().code, ErrorCode::kNotFound);
  };

  expect_invalid(TopicDirection::kImport,
                 ClockHealthOptions{
                     .capability = ClockHealthCapability::kStatusReports,
                     .requirement = ClockHealthRequirement::kNotRequired,
                 });
  expect_invalid(TopicDirection::kImport,
                 ClockHealthOptions{
                     .capability = ClockHealthCapability::kStatusReports,
                     .requirement = ClockHealthRequirement::kNotRequired,
                     .max_report_age = 5ms,
                 });
  expect_invalid(TopicDirection::kImport,
                 ClockHealthOptions{
                     .capability = ClockHealthCapability::kUnavailable,
                     .requirement = ClockHealthRequirement::kFreshHealthyReport,
                 });
  expect_invalid(TopicDirection::kImport,
                 ClockHealthOptions{
                     .capability = ClockHealthCapability::kUnavailable,
                     .requirement = ClockHealthRequirement::kFreshHealthyReport,
                     .max_report_age = 5ms,
                 });
  expect_invalid(TopicDirection::kImport,
                 ClockHealthOptions{
                     .capability = ClockHealthCapability::kUnavailable,
                     .requirement = ClockHealthRequirement::kNotRequired,
                     .max_report_age = 5ms,
                 });
  expect_invalid(TopicDirection::kExport,
                 ClockHealthOptions{
                     .capability = ClockHealthCapability::kStatusReports,
                     .requirement = ClockHealthRequirement::kFreshHealthyReport,
                     .max_report_age = 5ms,
                 });
}

// REQ: COMPILE-023
TEST(EndpointCatalog, InvalidClockHealthDiagnosticsPreservePathAndPresentedFacts) {
  using namespace std::chrono_literals;

  auto builder = ChannelEndpointCatalog::builder("raw-clock-health-diagnostic");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);
  std::string raw = built->json();
  const auto capability = raw.find("\"capability\":\"unavailable\"");
  ASSERT_NE(capability, std::string::npos);
  raw.replace(capability,
              std::string_view{"\"capability\":\"unavailable\""}.size(),
              "\"capability\":\"status_reports\"");
  const auto parsed = ChannelEndpointCatalog::parse_json(raw);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(
      parsed.error().message,
      "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID|path=/endpoints/0/clock_health|endpoint=sensor|"
      "required=unavailable+not_required(age=absent,any_role)|status_reports+fresh_healthy_"
      "report(age=positive,external_publisher)|provided=capability=status_reports;requirement="
      "not_required;max_report_age_ns=absent;role=external_publisher");

  Graph graph{"compile-clock-health-diagnostic"};
  TopicOptions options;
  options.endpoint_name = "sensor";
  (void)graph.import_topic<std::uint32_t>("holomq://sensor", options);
  auto snapshot = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  snapshot.external_topics.front().clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kNotRequired,
      .max_report_age = 5ms,
  };
  std::vector<std::optional<Profile>> profiles(snapshot.vertices.size(), Profile::kStandard);
  const auto lowered =
      detail::lower_endpoint_clock_health(snapshot, PlanGenerationId{17U}, {}, {}, profiles);
  const auto diagnostic =
      std::find_if(lowered.errors.begin(), lowered.errors.end(), [](const auto& candidate) {
        return candidate.token == "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID";
      });
  ASSERT_NE(diagnostic, lowered.errors.end());
  EXPECT_EQ(diagnostic->json_pointer, "/endpoints/0/clock_health");
  EXPECT_EQ(diagnostic->required,
            "unavailable+not_required(age=absent,any_role)|status_reports+fresh_healthy_report("
            "age=positive,external_publisher)");
  EXPECT_EQ(diagnostic->provided,
            "capability=status_reports;requirement=not_required;max_report_age_ns=5000000;role="
            "external_publisher");
  EXPECT_EQ(snapshot.external_topics.front().clock_health.max_report_age, 5ms);
}

// REQ: COMPILE-007
// REQ: COMPILE-023
TEST(EndpointCatalog, ConstrainedMonitoredHealthRequiresQualifiedMonitorEvidence) {
  using namespace std::chrono_literals;

  Graph graph{"constrained-clock-health"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey age_clock = graph.add_clock<RealtimeClock>("health-age-clock");
  graph.set_default_clock(age_clock);
  TopicOptions options;
  options.endpoint_name = "monitored";
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5ms,
  };
  (void)graph.import_topic<std::uint32_t>("holomq://monitored", options);

  CompileOptions compile_options;
  compile_options.profile = Profile::kRegulated;
  const ExecutionPlan plan = compile(graph, compile_options);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PROFILE_EVIDENCE_UNSATISFIED")) << plan.json();
  ASSERT_EQ(plan.endpoint_clock_health().size(), 1U);
  EXPECT_EQ(plan.endpoint_clock_health().front().capability, ClockHealthCapability::kStatusReports);
  EXPECT_TRUE(plan.endpoint_clock_health().front().monitor_evidence_reference.empty());
}

// REQ: COMPILE-023
// REQ: COMPILE-030
// Exercise package-bound monitor qualification through the complete compiler,
// rather than calling the clock-health lowerer directly.
//
//   capture-clock (ManualClock) --[AffineConversion]--> health-age-clock (RealtimeClock, default)
//   [holomq://monitored, cap_time=capture-clock, health=status_reports+fresh_healthy, max_age=50ms]
TEST(EndpointCatalog, CompileIngressAcceptsOnlyExactPackageBoundMonitorQualification) {
  using namespace std::chrono_literals;

  Graph graph{"qualified-clock-health-ingress"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey age_clock = graph.add_clock<RealtimeClock>("health-age-clock");
  graph.set_default_clock(age_clock);
  TopicOptions topic;
  topic.endpoint_name = "monitored";
  topic.capture_time = RequireCaptureTime{.clock = capture_clock};
  topic.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 50ms,
  };
  topic.attach.max_wait = 1s;
  (void)graph.import_topic<std::uint32_t>("holomq://monitored", topic);

  const auto package = detail::runtime_package_capability_view();
  const auto& package_manifest = package.manifest();
  const detail::RuntimeEndpointClockHealthQualification qualification{
      .endpoint_name = "monitored",
      .package_id = package_manifest.package_id,
      .build_id = package_manifest.build_id,
      .package_digest = package_manifest.digest,
      .monitor_period = 10ms,
      .observation_to_call = 5ms,
      .delivery_delay = 5ms,
      .jitter_margin = 5ms,
      .bounded_transition_delivery = true,
      .trace_loss_fail_closed = true,
      .unspecified_reason_qualified = true,
      .evidence_reference = "test.clock-health.compile-ingress/v1",
  };
  CompileOptions options;
  options.profile = Profile::kRobotics;
  const auto compile_with =
      [&](std::span<const detail::RuntimeEndpointClockHealthQualification> qualifications) {
        return detail::compile_with_runtime_evidence(graph,
                                                     options,
                                                     detail::RuntimeCompileEvidenceView{
                                                         .package_capabilities = package,
                                                         .endpoint_clock_health = qualifications,
                                                     });
      };

  // Key setup: qualification matches the exact package_digest, all timing budgets
  // within the max_report_age (50ms = 10+5+5+5+margin), and bounded transitions.
  const std::array valid{qualification};
  const ExecutionPlan accepted = compile_with(valid);
  ASSERT_TRUE(accepted.ok()) << accepted.json();
  ASSERT_EQ(accepted.endpoint_clock_health().size(), 1U);
  const auto& health = accepted.endpoint_clock_health().front();
  EXPECT_EQ(health.max_monitor_period, 10ms);
  EXPECT_EQ(health.max_observation_to_call_delay, 5ms);
  EXPECT_EQ(health.max_report_delivery_delay, 5ms);
  EXPECT_EQ(health.admitted_jitter_margin, 5ms);
  EXPECT_EQ(health.monitor_evidence_reference, "test.clock-health.compile-ingress/v1");
  EXPECT_NE(accepted.json().find("test.clock-health.compile-ingress/v1"), std::string::npos);

  auto changed = qualification;
  changed.jitter_margin = 6ms;
  const std::array changed_qualifications{changed};
  const ExecutionPlan changed_plan = compile_with(changed_qualifications);
  ASSERT_TRUE(changed_plan.ok()) << changed_plan.json();
  EXPECT_NE(changed_plan.digest(), accepted.digest());

  const auto expect_rejected = [&](auto qualifications) {
    const ExecutionPlan rejected = compile_with(qualifications);
    EXPECT_FALSE(rejected.ok());
    EXPECT_TRUE(test::has_diagnostic(rejected, "PROFILE_EVIDENCE_UNSATISFIED")) << rejected.json();
  };

  // Key assertion: wrong endpoint name, excessive delivery delay, duplicate
  // qualification entries, and wrong package digest each independently reject.
  auto wrong_endpoint = qualification;
  wrong_endpoint.endpoint_name = "another-endpoint";
  expect_rejected(std::array{wrong_endpoint});

  auto excessive = qualification;
  excessive.delivery_delay = 30ms;
  expect_rejected(std::array{excessive});

  expect_rejected(std::array{qualification, qualification});

  auto wrong_package = qualification;
  wrong_package.package_digest =
      "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  expect_rejected(std::array{wrong_package});
}

// REQ: COMPILE-023
TEST(EndpointCatalog, CrossDomainTemporalUseRequiresDeclaredMonitoredHealth) {
  using namespace std::chrono_literals;

  // Key setup: without clock-health options the cross-domain LatestWithin
  // path cannot satisfy the runtime capability requirement for temporal selection.
  Graph ordinary_graph{"q61-cross-domain-ordinary"};
  add_cross_domain_latest_path(ordinary_graph, ClockHealthOptions{});
  const ExecutionPlan ordinary = compile(ordinary_graph);
  EXPECT_FALSE(ordinary.ok());
  EXPECT_TRUE(test::has_diagnostic(ordinary, "PROFILE_CAPABILITY_UNSATISFIED")) << ordinary.json();
  ASSERT_EQ(ordinary.endpoint_clock_health().size(), 1U);
  EXPECT_EQ(ordinary.endpoint_clock_health().front().capability,
            ClockHealthCapability::kUnavailable);
  EXPECT_EQ(ordinary.endpoint_clock_health().front().requirement,
            ClockHealthRequirement::kNotRequired);

  Graph monitored_graph{"q61-cross-domain-monitored"};
  add_cross_domain_latest_path(monitored_graph,
                               ClockHealthOptions{
                                   .capability = ClockHealthCapability::kStatusReports,
                                   .requirement = ClockHealthRequirement::kFreshHealthyReport,
                                   .max_report_age = 50ms,
                               });
  const ExecutionPlan monitored = compile(monitored_graph);
  ASSERT_TRUE(monitored.ok()) << monitored.json();
  ASSERT_EQ(monitored.endpoint_clock_health().size(), 1U);
  EXPECT_EQ(monitored.endpoint_clock_health().front().capability,
            ClockHealthCapability::kStatusReports);
  ASSERT_EQ(monitored.input_selections().size(), 1U);
  // Key assertion: the health-age clock used for freshness tracking equals
  // the reference clock in the LatestWithin input selection record.
  EXPECT_EQ(monitored.endpoint_clock_health().front().health_age_clock,
            monitored.input_selections().front().reference_clock);
  EXPECT_EQ(monitored.endpoint_clock_health().front().capture_conversion_id,
            monitored.input_selections().front().conversion_id);
  EXPECT_EQ(monitored.endpoint_clock_health().front().plan_generation,
            monitored.clock_catalog().plan_generation);
  EXPECT_EQ(monitored.endpoint_clock_health().front().conversion_generation,
            monitored.clock_catalog().plan_generation);
}

// REQ: COMPILE-023
TEST(EndpointCatalog, ClockHealthReasonSchemaPreservesFrozenIdsAndTokens) {
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kNone) == 0U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kSynchronizationLockLost) == 1U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kPhaseErrorLimitExceeded) == 2U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kRateErrorLimitExceeded) == 3U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kCalibrationExpired) == 4U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kClockStalled) == 5U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kTimestampRegression) == 6U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kTimestampSourceFault) == 7U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kMonitorFault) == 8U);
  static_assert(static_cast<std::uint8_t>(ClockHealthReason::kUnspecified) == 9U);

  Graph graph{"clock-health-reason-schema"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  constexpr std::string_view kExpected{
      "\"clock_health_reason_schema\":[{\"id\":0,\"token\":\"none\"},"
      "{\"id\":1,\"token\":\"synchronization_lock_lost\"},"
      "{\"id\":2,\"token\":\"phase_error_limit_exceeded\"},"
      "{\"id\":3,\"token\":\"rate_error_limit_exceeded\"},"
      "{\"id\":4,\"token\":\"calibration_expired\"},"
      "{\"id\":5,\"token\":\"clock_stalled\"},"
      "{\"id\":6,\"token\":\"timestamp_regression\"},"
      "{\"id\":7,\"token\":\"timestamp_source_fault\"},"
      "{\"id\":8,\"token\":\"monitor_fault\"},"
      "{\"id\":9,\"token\":\"unspecified\"}]"};
  EXPECT_NE(plan.json().find(kExpected), std::string::npos);
}

// REQ: COMPILE-023
TEST(EndpointCatalog, CompilerCatalogProjectsOnlyAuthoritativePlanClockFacts) {
  using namespace std::chrono_literals;

  Graph graph{"catalog-clock-join"};
  add_cross_domain_latest_path(graph,
                               ClockHealthOptions{
                                   .capability = ClockHealthCapability::kStatusReports,
                                   .requirement = ClockHealthRequirement::kFreshHealthyReport,
                                   .max_report_age = 50ms,
                               });
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* catalog = detail::ChannelEndpointCatalogAccess::state(plan.endpoint_catalog());
  ASSERT_NE(catalog, nullptr);
  ASSERT_EQ(catalog->clocks.size(), 2U);
  ASSERT_EQ(catalog->clock_conversions.size(), 1U);
  EXPECT_TRUE(std::all_of(catalog->clocks.begin(), catalog->clocks.end(), [](const auto& clock) {
    return clock.kind != "external" && clock.epoch == "plan_coordinate" && clock.id != 0U;
  }));
  ASSERT_EQ(plan.clock_conversions().size(), 1U);
  const auto& catalog_conversion = catalog->clock_conversions.front();
  const auto& plan_conversion = plan.clock_conversions().front();
  EXPECT_EQ(catalog_conversion.scale_numerator, plan_conversion.scale.numerator);
  EXPECT_EQ(catalog_conversion.scale_denominator, plan_conversion.scale.denominator);
  EXPECT_EQ(catalog_conversion.offset_ns, plan_conversion.offset_ns);
  EXPECT_LE(catalog_conversion.admitted_min, 0);
  EXPECT_GE(catalog_conversion.admitted_max, 0);
  EXPECT_EQ(catalog_conversion.provenance,
            "holoscan.execution-plan.clock-conversion.v1/" + std::to_string(plan_conversion.id));
}

// REQ: COMPILE-023
TEST(EndpointCatalog, ClockProjectionClosesTransitiveConversionsIndependentOfRecordOrder) {
  // Graph topology:
  //   import(capture clock)    capture --> intermediate --> reference
  Graph graph{"catalog-clock-transitive-closure"};
  const ClockKey capture = graph.add_clock<ManualClock>("capture", std::int64_t{0});
  const ClockKey intermediate = graph.add_clock<ManualClock>("intermediate", std::int64_t{0});
  const ClockKey reference = graph.add_clock<ManualClock>("reference", std::int64_t{0});
  graph.set_default_clock(reference);
  graph.add_clock_conversion(capture, intermediate, AffineClockConversion{});
  graph.add_clock_conversion(intermediate, reference, AffineClockConversion{});
  TopicOptions options;
  options.endpoint_name = "transitive-clock-import";
  options.capture_time = RequireCaptureTime{.clock = capture};
  (void)graph.import_topic<std::uint32_t>("holomq://transitive-clock-import", std::move(options));

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_conversions().size(), 2U);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  std::vector<ClockConversionRecord> reversed_conversions{plan.clock_conversions().begin(),
                                                          plan.clock_conversions().end()};
  std::reverse(reversed_conversions.begin(), reversed_conversions.end());

  // Key setup: present B->C before A->B even though only A is named by the endpoint.
  const auto lowered = detail::lower_endpoint_catalog(*snapshot,
                                                      DeploymentBinding{},
                                                      Profile::kStandard,
                                                      std::nullopt,
                                                      plan.clock_catalog().clocks,
                                                      reversed_conversions);

  // Key assertion: reachability closes to a fixpoint and retains both conversion rows and all
  // three clocks regardless of conversion declaration order.
  ASSERT_TRUE(lowered.accepted());
  const auto* catalog = detail::ChannelEndpointCatalogAccess::state(lowered.catalog);
  ASSERT_NE(catalog, nullptr);
  EXPECT_EQ(catalog->clocks.size(), 3U);
  EXPECT_EQ(catalog->clock_conversions.size(), 2U);
}

// REQ: COMPILE-017
TEST(EndpointCatalog, CrossDomainSelectionFamiliesJoinTheImportHealthContract) {
  using namespace std::chrono_literals;

  Graph graph{"q61-selection-matrix"};
  add_cross_domain_latest_path(graph,
                               ClockHealthOptions{
                                   .capability = ClockHealthCapability::kStatusReports,
                                   .requirement = ClockHealthRequirement::kFreshHealthyReport,
                                   .max_report_age = 50ms,
                               });
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.input_selections().size(), 1U);
  ASSERT_NE(plan.input_selections().front().conversion_id, 0U);

  auto snapshot = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  snapshot.external_topics.front().clock_health = ClockHealthOptions{};
  std::vector<std::optional<Profile>> profiles(snapshot.vertices.size(), Profile::kStandard);
  const std::array<InputSelectorKind, 3U> timing_sensitive{
      InputSelectorKind::kExact,
      InputSelectorKind::kLatestWithin,
      InputSelectorKind::kBracketedWindow,
  };
  for (const auto selector : timing_sensitive) {
    InputSelectionPlanRecord selection = plan.input_selections().front();
    selection.selector = selector;
    const std::array<InputSelectionPlanRecord, 1U> selections{selection};
    const auto result = detail::lower_endpoint_clock_health(snapshot,
                                                            plan.clock_catalog().plan_generation,
                                                            plan.clock_catalog().clocks,
                                                            plan.clock_conversions(),
                                                            profiles,
                                                            selections);
    EXPECT_FALSE(result.accepted()) << static_cast<unsigned>(selector);
    EXPECT_TRUE(has_clock_health_diagnostic(result, "PROFILE_CAPABILITY_UNSATISFIED"))
        << static_cast<unsigned>(selector);
  }

  // OnAligned lowers each member to Exact with a direct conversion. The Exact
  // cell above therefore covers aligned members without trigger-name coupling.
  InputSelectionPlanRecord drain = plan.input_selections().front();
  drain.selector = InputSelectorKind::kDrainAll;
  const std::array<InputSelectionPlanRecord, 1U> drain_selection{drain};
  const auto drain_result =
      detail::lower_endpoint_clock_health(snapshot,
                                          plan.clock_catalog().plan_generation,
                                          plan.clock_catalog().clocks,
                                          plan.clock_conversions(),
                                          profiles,
                                          drain_selection);
  EXPECT_TRUE(drain_result.accepted());
}

// REQ: COMPILE-023
TEST(EndpointCatalog, ConstrainedMonitorQualificationIsFiniteExactAndInspectable) {
  using namespace std::chrono_literals;

  Graph graph{"q61-qualified-monitor"};
  add_cross_domain_latest_path(graph,
                               ClockHealthOptions{
                                   .capability = ClockHealthCapability::kStatusReports,
                                   .requirement = ClockHealthRequirement::kFreshHealthyReport,
                                   .max_report_age = 50ms,
                               });
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_FALSE(snapshot->external_topics.empty());
  std::vector<std::optional<Profile>> profiles(snapshot->vertices.size(), Profile::kRobotics);

  // Key setup: Profile::kRobotics requires monitor qualification evidence;
  // calling the lowerer without it yields PROFILE_EVIDENCE_UNSATISFIED.
  const auto missing = detail::lower_endpoint_clock_health(*snapshot,
                                                           plan.clock_catalog().plan_generation,
                                                           plan.clock_catalog().clocks,
                                                           plan.clock_conversions(),
                                                           profiles,
                                                           plan.input_selections());
  EXPECT_FALSE(missing.accepted());
  EXPECT_TRUE(has_clock_health_diagnostic(missing, "PROFILE_EVIDENCE_UNSATISFIED"));

  detail::EndpointClockHealthQualificationRecord qualification{
      .endpoint = PortId{snapshot->external_topics.front().graph_port.value},
      .monitor_period = 10ms,
      .observation_to_call = 5ms,
      .delivery_delay = 5ms,
      .jitter_margin = 5ms,
      .bounded_transition_delivery = true,
      .trace_loss_fail_closed = true,
      .unspecified_reason_qualified = true,
      .evidence_reference = "test.clock-health.monitor-path/v1",
  };
  const std::array<detail::EndpointClockHealthQualificationRecord, 1U> qualifications{
      qualification};
  const auto admitted = detail::lower_endpoint_clock_health(*snapshot,
                                                            plan.clock_catalog().plan_generation,
                                                            plan.clock_catalog().clocks,
                                                            plan.clock_conversions(),
                                                            profiles,
                                                            plan.input_selections(),
                                                            qualifications);
  // Key assertion: all timing fields from the qualification record are preserved
  // verbatim; an excessive delivery_delay (30ms > max_report_age budget) rejects.
  ASSERT_TRUE(admitted.accepted());
  ASSERT_EQ(admitted.records.size(), 1U);
  EXPECT_EQ(admitted.records.front().monitor_evidence_reference,
            "test.clock-health.monitor-path/v1");
  EXPECT_EQ(admitted.records.front().max_monitor_period, 10ms);
  EXPECT_EQ(admitted.records.front().max_observation_to_call_delay, 5ms);
  EXPECT_EQ(admitted.records.front().max_report_delivery_delay, 5ms);
  EXPECT_EQ(admitted.records.front().admitted_jitter_margin, 5ms);
  EXPECT_EQ(admitted.records.front().unspecified_reason_disposition,
            ClockHealthUnspecifiedDisposition::kQualifiedOnly);

  qualification.delivery_delay = 30ms;
  const std::array<detail::EndpointClockHealthQualificationRecord, 1U> excessive{qualification};
  const auto rejected = detail::lower_endpoint_clock_health(*snapshot,
                                                            plan.clock_catalog().plan_generation,
                                                            plan.clock_catalog().clocks,
                                                            plan.clock_conversions(),
                                                            profiles,
                                                            plan.input_selections(),
                                                            excessive);
  EXPECT_FALSE(rejected.accepted());
  EXPECT_TRUE(has_clock_health_diagnostic(rejected, "PROFILE_EVIDENCE_UNSATISFIED"));
}

// REQ: COMPILE-023
TEST(EndpointCatalog, UnspecifiedClockHealthReasonUsesTheExactFiveProfileMatrixWithoutMutation) {
  using namespace std::chrono_literals;

  Graph graph{"q61-profile-matrix"};
  add_cross_domain_latest_path(graph,
                               ClockHealthOptions{
                                   .capability = ClockHealthCapability::kStatusReports,
                                   .requirement = ClockHealthRequirement::kFreshHealthyReport,
                                   .max_report_age = 50ms,
                               });
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_FALSE(snapshot->external_topics.empty());
  const auto authored_health = snapshot->external_topics.front().clock_health;

  detail::EndpointClockHealthQualificationRecord qualification{
      .endpoint = PortId{snapshot->external_topics.front().graph_port.value},
      .monitor_period = 10ms,
      .observation_to_call = 5ms,
      .delivery_delay = 5ms,
      .jitter_margin = 5ms,
      .bounded_transition_delivery = true,
      .trace_loss_fail_closed = true,
      .unspecified_reason_qualified = false,
      .evidence_reference = "test.clock-health.profile-matrix/v1",
  };
  const auto lower = [&](Profile profile, bool qualify_unspecified) {
    std::vector<std::optional<Profile>> profiles(snapshot->vertices.size(), profile);
    auto selected_qualification = qualification;
    selected_qualification.unspecified_reason_qualified = qualify_unspecified;
    const std::array<detail::EndpointClockHealthQualificationRecord, 1U> qualifications{
        selected_qualification};
    return detail::lower_endpoint_clock_health(*snapshot,
                                               plan.clock_catalog().plan_generation,
                                               plan.clock_catalog().clocks,
                                               plan.clock_conversions(),
                                               profiles,
                                               plan.input_selections(),
                                               qualifications);
  };

  for (const Profile profile : {Profile::kStandard, Profile::kResearch}) {
    const auto result = lower(profile, false);
    ASSERT_TRUE(result.accepted());
    ASSERT_EQ(result.records.size(), 1U);
    EXPECT_EQ(result.records.front().unspecified_reason_disposition,
              ClockHealthUnspecifiedDisposition::kAdmitted);
  }
  for (const Profile profile : {Profile::kRobotics, Profile::kRealTime}) {
    const auto result = lower(profile, false);
    ASSERT_TRUE(result.accepted());
    ASSERT_EQ(result.records.size(), 1U);
    EXPECT_EQ(result.records.front().unspecified_reason_disposition,
              ClockHealthUnspecifiedDisposition::kQualifiedOnly);
  }
  const auto regulated_rejected = lower(Profile::kRegulated, false);
  ASSERT_TRUE(regulated_rejected.accepted());
  ASSERT_EQ(regulated_rejected.records.size(), 1U);
  EXPECT_EQ(regulated_rejected.records.front().unspecified_reason_disposition,
            ClockHealthUnspecifiedDisposition::kRejected);
  const auto regulated_qualified = lower(Profile::kRegulated, true);
  ASSERT_TRUE(regulated_qualified.accepted());
  ASSERT_EQ(regulated_qualified.records.size(), 1U);
  EXPECT_EQ(regulated_qualified.records.front().unspecified_reason_disposition,
            ClockHealthUnspecifiedDisposition::kQualifiedOnly);

  EXPECT_EQ(snapshot->external_topics.front().clock_health.capability, authored_health.capability);
  EXPECT_EQ(snapshot->external_topics.front().clock_health.requirement,
            authored_health.requirement);
  EXPECT_EQ(snapshot->external_topics.front().clock_health.max_report_age,
            authored_health.max_report_age);
  EXPECT_FALSE(qualification.unspecified_reason_qualified);
}

// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, TensorCatalogRoundTripPreservesEveryLogicalBoundaryConstraint) {
  // Key setup: full HD CUDA-device uint8 tensor (1080×1920, bounded strided,
  // copy-if-needed) with per-axis stride bounds; all fields must survive JSON
  // round-trip without coercion or zero-filling.
  const std::array<TensorAxisBounds, 2> axes{{
      {.max_extent = 1080U, .max_abs_stride_bytes = 7680U},
      {.max_extent = 1920U, .max_abs_stride_bytes = 4U},
  }};
  TopicOptions options;
  options.tensor = TensorBoundaryContract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kCudaDevice,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .rank = 2U,
              .strides = TensorStridePolicy::kBoundedStrided,
          },
      .bounds =
          TensorBounds{
              .max_rank = 2U,
              .max_byte_span = 8'294'400U,
              .axes = axes,
          },
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              .max_staging_byte_span = 4'147'200U,
          },
      .memory_conversion = MemoryConversionPolicy::kDisable,
  };

  auto builder = ChannelEndpointCatalog::builder("tensor-roundtrip");
  ASSERT_TRUE(builder.import_topic<Tensor>("frames", "holomq://frames", options));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);
  EXPECT_NE(built->json().find("\"memory_kind\":\"cuda_device\""), std::string::npos);
  EXPECT_NE(built->json().find("\"tensor\":{\"role\":\"external_boundary\",\"representation\":"),
            std::string::npos);
  EXPECT_NE(built->json().find("\"bounds\":{\"max_rank\":2"), std::string::npos);
  EXPECT_NE(built->json().find("\"transfer\":{\"foreign_emit_policy\":\"copy_if_needed\""),
            std::string::npos);
  EXPECT_NE(built->json().find("\"max_staging_byte_span\":4147200"), std::string::npos);
  EXPECT_NE(built->json().find("\"memory_conversion\":\"disable\""), std::string::npos);
  EXPECT_EQ(built->json().find("\"origins\":"), std::string::npos);
  EXPECT_NE(built->json().find("\"dtype\":{\"code\":1,\"bits\":8,\"lanes\":1}"), std::string::npos);
  EXPECT_NE(built->json().find("\"rank\":2"), std::string::npos);
  EXPECT_NE(built->json().find("\"max_extent\":1080"), std::string::npos);

  auto parsed = ChannelEndpointCatalog::parse_json(built->json());
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->json(), built->json());
  EXPECT_EQ(parsed->digest(), built->digest());
  const auto endpoint = parsed->endpoint("frames");
  ASSERT_TRUE(endpoint);
  const auto tensor_view = endpoint->tensor_requirements();
  ASSERT_TRUE(tensor_view.has_value());
  EXPECT_EQ(tensor_view->memory_kind(), MemoryKind::kCudaDevice);
  ASSERT_TRUE(tensor_view->dtype().has_value());
  EXPECT_EQ(tensor_view->dtype()->code, kDLUInt);
  EXPECT_EQ(tensor_view->rank(), 2U);
  EXPECT_EQ(tensor_view->max_rank(), 2U);
  EXPECT_EQ(tensor_view->max_byte_span(), 8'294'400U);
  EXPECT_EQ(tensor_view->stride_policy(), TensorStridePolicy::kBoundedStrided);
  EXPECT_EQ(tensor_view->foreign_emit_policy(), InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_FALSE(tensor_view->max_inline_bytes().has_value());
  EXPECT_EQ(tensor_view->max_staging_byte_span(), 4'147'200U);
  ASSERT_EQ(tensor_view->axes().size(), 2U);
  EXPECT_EQ(tensor_view->axes()[0].max_extent, 1080U);
  const auto* state = detail::ChannelEndpointCatalogAccess::state(*parsed);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->endpoints.size(), 1U);
  ASSERT_TRUE(state->endpoints.front().tensor.has_value());
  const auto& tensor = *state->endpoints.front().tensor;
  EXPECT_EQ(tensor.memory_kind, MemoryKind::kCudaDevice);
  ASSERT_TRUE(tensor.dtype.has_value());
  EXPECT_EQ(tensor.dtype->code, kDLUInt);
  EXPECT_EQ(tensor.dtype->bits, 8U);
  EXPECT_EQ(tensor.dtype->lanes, 1U);
  EXPECT_EQ(tensor.rank, 2U);
  EXPECT_EQ(tensor.max_rank, 2U);
  EXPECT_EQ(tensor.max_byte_span, 8'294'400U);
  EXPECT_EQ(tensor.max_staging_byte_span, 4'147'200U);
  EXPECT_EQ(tensor.memory_conversion, MemoryConversionPolicy::kDisable);
  ASSERT_EQ(tensor.axes.size(), 2U);
  EXPECT_EQ(tensor.axes[0].max_extent, 1080U);
  EXPECT_EQ(tensor.axes[0].max_abs_stride_bytes, 7680U);
  EXPECT_FALSE(tensor.max_inline_bytes.has_value());
}

// REQ: COMPILE-007
// REQ: COMPILE-022
TEST(EndpointCatalog, OmittedTensorRequirementsRemainAbsentInsteadOfBecomingZeroBounds) {
  auto builder = ChannelEndpointCatalog::builder("tensor-optional");
  ASSERT_TRUE(builder.import_topic<Tensor>("frames", "holomq://frames"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);
  EXPECT_NE(built->json().find("\"tensor\":null"), std::string::npos);

  auto parsed = ChannelEndpointCatalog::parse_json(built->json());
  ASSERT_TRUE(parsed);
  const auto* state = detail::ChannelEndpointCatalogAccess::state(*parsed);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->endpoints.size(), 1U);
  EXPECT_FALSE(state->endpoints.front().tensor.has_value());
}

// REQ: COMPILE-007
// REQ: COMPILE-022
TEST(EndpointCatalog, StrictParserRejectsAmbiguousOrLossyCatalogDocuments) {
  auto builder = ChannelEndpointCatalog::builder("strict-parser");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);

  const auto expect_failure = [](std::string_view source, std::string_view token) {
    auto parsed = ChannelEndpointCatalog::parse_json(source);
    ASSERT_FALSE(parsed);
    EXPECT_EQ(parsed.error().message, token);
  };

  std::string duplicate = built->json();
  duplicate.insert(1U, "\"format\":\"holoscan.channel-endpoints\",");
  expect_failure(std::move(duplicate), "ENDPOINT_CATALOG_DUPLICATE_KEY");

  std::string unknown = built->json();
  unknown.insert(1U, "\"unknown\":0,");
  expect_failure(std::move(unknown), "ENDPOINT_CATALOG_FORMAT_UNSUPPORTED");

  std::string unsupported_version = built->json();
  const auto version = unsupported_version.find("\"version\":2");
  ASSERT_NE(version, std::string::npos);
  unsupported_version.replace(version, std::string_view{"\"version\":2"}.size(), "\"version\":3");
  expect_failure(std::move(unsupported_version), "ENDPOINT_CATALOG_FORMAT_UNSUPPORTED");

  std::string floating_version = built->json();
  const auto floating = floating_version.find("\"version\":2");
  ASSERT_NE(floating, std::string::npos);
  floating_version.replace(floating, std::string_view{"\"version\":2"}.size(), "\"version\":2.0");
  expect_failure(std::move(floating_version), "ENDPOINT_CATALOG_CANONICALIZATION_UNSUPPORTED");

  std::string invalid_utf8 = built->json();
  const auto scope = invalid_utf8.find("\"scope\":\"");
  ASSERT_NE(scope, std::string::npos);
  invalid_utf8.insert(scope + std::string_view{"\"scope\":\""}.size(), 1U, static_cast<char>(0xFF));
  expect_failure(std::move(invalid_utf8), "ENDPOINT_CATALOG_PARSE_FAILED");

  std::string wrong_digest = built->json();
  const auto digest = wrong_digest.find("\"digest\":\"sha256:");
  ASSERT_NE(digest, std::string::npos);
  const auto first_digit = digest + std::string_view{"\"digest\":\"sha256:"}.size();
  wrong_digest[first_digit] = wrong_digest[first_digit] == '0' ? '1' : '0';
  expect_failure(std::move(wrong_digest), "ENDPOINT_CATALOG_DIGEST_MISMATCH");
}

// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, ParserEnforcesFixedSchemaV2ResourceLimitsAtTheExactBoundary) {
  const auto expect_within_parser_limit = [](std::string_view source) {
    const auto parsed = ChannelEndpointCatalog::parse_json(source);
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().code, ErrorCode::kResourceExhausted) << parsed.error().message;
  };

  expect_within_parser_limit(nested_null(kJsonDepthLimit));
  expect_catalog_resource_limit(
      ChannelEndpointCatalog::parse_json(nested_null(kJsonDepthLimit + 1U)), "nesting_depth");

  expect_within_parser_limit(json_with_node_count(kJsonNodeLimit));
  expect_catalog_resource_limit(
      ChannelEndpointCatalog::parse_json(json_with_node_count(kJsonNodeLimit + 1U)), "json_nodes");

  expect_within_parser_limit(null_object(kJsonObjectMemberLimit));
  expect_catalog_resource_limit(
      ChannelEndpointCatalog::parse_json(null_object(kJsonObjectMemberLimit + 1U)),
      "object_members");

  expect_within_parser_limit(null_array(kJsonArrayElementLimit));
  expect_catalog_resource_limit(
      ChannelEndpointCatalog::parse_json(null_array(kJsonArrayElementLimit + 1U)),
      "array_elements");

  const auto empty_catalog =
      detail::ChannelEndpointCatalogAccess::normalize("collection-limits", {});
  ASSERT_TRUE(empty_catalog);
  const auto expect_collection_boundary =
      [&](std::string_view field, std::size_t limit, std::string_view resource) {
        const std::string exact = replace_empty_array(empty_catalog->json(), field, limit);
        ASSERT_FALSE(exact.empty());
        expect_within_parser_limit(exact);
        const std::string over = replace_empty_array(empty_catalog->json(), field, limit + 1U);
        ASSERT_FALSE(over.empty());
        expect_catalog_resource_limit(ChannelEndpointCatalog::parse_json(over), resource);
      };
  expect_collection_boundary("clocks", kClockLimit, "normalized_clocks");
  expect_collection_boundary(
      "clock_conversions", kClockConversionLimit, "normalized_clock_conversions");
  expect_collection_boundary("services", kEndpointLimit, "normalized_services");
  expect_collection_boundary("endpoints", kEndpointLimit, "normalized_endpoints");

  std::string exact_source(kCatalogSourceLimit, ' ');
  expect_within_parser_limit(exact_source);
  exact_source.push_back(' ');
  expect_catalog_resource_limit(ChannelEndpointCatalog::parse_json(exact_source), "source_bytes");

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("holoscan_endpoint_catalog_oversized_" + std::to_string(unique) + ".json");
  {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(stream);
    stream.write(exact_source.data(), static_cast<std::streamsize>(exact_source.size()));
    ASSERT_TRUE(stream);
  }
  expect_catalog_resource_limit(ChannelEndpointCatalog::load_json(path), "source_bytes");
  std::error_code cleanup_error;
  std::filesystem::remove(path, cleanup_error);
  EXPECT_FALSE(cleanup_error);
}

// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, NormalizationEnforcesFixedSchemaV2CollectionLimits) {
  const auto make_scalar_endpoint = [](std::size_t index) {
    return detail::ChannelEndpointRecord{
        .name = "endpoint-" + std::to_string(index),
        .uri = "holomq://endpoint-" + std::to_string(index),
        .direction = TopicDirection::kImport,
        .name_origin = EndpointNameOrigin::kExplicit,
        .payload_identity =
            PayloadIdentityRecord{
                .kind = PayloadIdentityKind::kScalar,
                .payload_kind = PayloadKind::kInlineTyped,
                .scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt32},
            },
    };
  };
  std::vector<detail::ChannelEndpointRecord> endpoints;
  endpoints.reserve(kEndpointLimit + 1U);
  for (std::size_t index = 0U; index < kEndpointLimit; ++index) {
    endpoints.push_back(make_scalar_endpoint(index));
  }
  auto exact_endpoints =
      detail::ChannelEndpointCatalogAccess::normalize("endpoint-limit", endpoints);
  ASSERT_TRUE(exact_endpoints) << exact_endpoints.error().message;
  endpoints.push_back(make_scalar_endpoint(kEndpointLimit));
  expect_catalog_resource_limit(
      detail::ChannelEndpointCatalogAccess::normalize("endpoint-limit", std::move(endpoints)),
      "normalized_endpoints");

  expect_catalog_resource_limit(
      detail::ChannelEndpointCatalogAccess::normalize(
          "clock-limit", {}, std::vector<detail::CatalogClockRecord>(kClockLimit + 1U)),
      "normalized_clocks");
  expect_catalog_resource_limit(
      detail::ChannelEndpointCatalogAccess::normalize(
          "conversion-limit",
          {},
          {},
          std::vector<detail::CatalogClockConversionRecord>(kClockConversionLimit + 1U)),
      "normalized_clock_conversions");

  const auto make_pod_endpoint = [](std::size_t field_count) {
    auto endpoint = identity_endpoint(valid_pod_identity());
    endpoint.payload_identity.pod->canonical_name = "nvidia.holoscan.test.BoundedPod";
    endpoint.payload_identity.pod->object_size = static_cast<std::uint32_t>(field_count);
    endpoint.payload_identity.pod->object_alignment = 1U;
    endpoint.payload_identity.pod->fields.clear();
    endpoint.payload_identity.pod->fields.reserve(field_count);
    for (std::size_t index = 0U; index < field_count; ++index) {
      endpoint.payload_identity.pod->fields.push_back(PodFieldIdentityRecord{
          .name = "field-" + std::to_string(index),
          .byte_offset = static_cast<std::uint32_t>(index),
          .scalar_kind = PodScalarKind::kUInt8,
          .element_count = 1U,
      });
    }
    return endpoint;
  };
  auto exact_fields = detail::ChannelEndpointCatalogAccess::normalize(
      "pod-field-limit", {make_pod_endpoint(kPodFieldLimit)});
  ASSERT_TRUE(exact_fields) << exact_fields.error().message;
  expect_catalog_resource_limit(detail::ChannelEndpointCatalogAccess::normalize(
                                    "pod-field-limit", {make_pod_endpoint(kPodFieldLimit + 1U)}),
                                "normalized_pod_fields");

  const auto make_tensor_endpoint = [&make_scalar_endpoint](std::size_t axis_count) {
    auto endpoint = make_scalar_endpoint(0U);
    endpoint.name = "tensor-endpoint";
    endpoint.uri = "holomq://tensor-endpoint";
    endpoint.payload_identity = PayloadIdentityRecord{
        .kind = PayloadIdentityKind::kTensor,
        .payload_kind = PayloadKind::kDescriptorBacked,
    };
    endpoint.tensor = TensorBoundaryContractRecord{
        .memory_kind = MemoryKind::kHost,
        .rank = static_cast<std::uint32_t>(axis_count),
        .max_rank = static_cast<std::uint32_t>(axis_count),
        .max_byte_span = 1U,
        .axes = std::vector<TensorAxisBounds>(axis_count,
                                              TensorAxisBounds{
                                                  .max_extent = 1U,
                                                  .max_abs_stride_bytes = 1U,
                                              }),
        .strides = TensorStridePolicy::kBoundedStrided,
        .foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable,
    };
    return endpoint;
  };
  auto exact_axes = detail::ChannelEndpointCatalogAccess::normalize(
      "tensor-axis-limit", {make_tensor_endpoint(kTensorAxisLimit)});
  ASSERT_TRUE(exact_axes) << exact_axes.error().message;
  expect_catalog_resource_limit(
      detail::ChannelEndpointCatalogAccess::normalize(
          "tensor-axis-limit", {make_tensor_endpoint(kTensorAxisLimit + 1U)}),
      "normalized_tensor_axes");
}

// REQ: COMPILE-022
// REQ: COMPILE-066
// REQ: CHAN-074
TEST(EndpointCatalog, PodAdmissionRequiresCompletePortableLayoutAndVerifiesDerivedDigests) {
  auto valid = detail::ChannelEndpointCatalogAccess::normalize(
      "pod-validation", {identity_endpoint(valid_pod_identity())});
  ASSERT_TRUE(valid) << valid.error().message;
  const auto endpoint = valid->endpoint("identity-endpoint");
  ASSERT_TRUE(endpoint);
  const auto pod = endpoint->pod_type_identity();
  ASSERT_TRUE(pod.has_value());
  EXPECT_EQ(pod->name(), "nvidia.holoscan.test.EndpointPod");
  EXPECT_EQ(pod->object_size(), 4U);
  ASSERT_EQ(pod->fields().size(), 1U);
  EXPECT_EQ(pod->fields().front().name(), "value");

  const auto expect_invalid = [](PayloadIdentityRecord identity) {
    auto result = detail::ChannelEndpointCatalogAccess::normalize(
        "pod-validation", {identity_endpoint(std::move(identity))});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().message, "ENDPOINT_PAYLOAD_IDENTITY_INVALID");
  };

  auto bad_alignment = valid_pod_identity();
  bad_alignment.pod->object_alignment = 3U;
  expect_invalid(std::move(bad_alignment));
  auto gap = valid_pod_identity();
  gap.pod->fields.front().byte_offset = 1U;
  expect_invalid(std::move(gap));
  auto zero_count = valid_pod_identity();
  zero_count.pod->fields.front().element_count = 0U;
  expect_invalid(std::move(zero_count));
  auto invalid_name = valid_pod_identity();
  invalid_name.pod->canonical_name = std::string{"invalid\0name", 12U};
  expect_invalid(std::move(invalid_name));

  std::string corrupt_digest = valid->json();
  const auto digest = corrupt_digest.find("\"layout_digest\":\"sha256:");
  ASSERT_NE(digest, std::string::npos);
  const auto first_digit = digest + std::string_view{"\"layout_digest\":\"sha256:"}.size();
  corrupt_digest[first_digit] = corrupt_digest[first_digit] == '0' ? '1' : '0';
  const auto parsed = ChannelEndpointCatalog::parse_json(corrupt_digest);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().message, "ENDPOINT_PAYLOAD_IDENTITY_INVALID");
}

// REQ: COMPILE-022
// REQ: CHAN-074
TEST(EndpointCatalog, NormalizationPreservesPodAuthorityCollisionChecks) {
  auto first = identity_endpoint(valid_pod_identity());
  first.name = "first";
  first.uri = "holomq://first";
  auto equivalent = identity_endpoint(valid_pod_identity());
  equivalent.name = "equivalent";
  equivalent.uri = "holomq://equivalent";
  auto accepted =
      detail::ChannelEndpointCatalogAccess::normalize("pod-authority", {first, equivalent});
  ASSERT_TRUE(accepted) << accepted.error().message;

  auto conflicting = identity_endpoint(valid_pod_identity());
  conflicting.name = "conflicting";
  conflicting.uri = "holomq://conflicting";
  conflicting.payload_identity.pod->fields.front().name = "different-layout";
  auto rejected = detail::ChannelEndpointCatalogAccess::normalize(
      "pod-authority", {std::move(first), std::move(conflicting)});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kIncompatible);
  EXPECT_EQ(rejected.error().message, "ENDPOINT_POD_IDENTITY_COLLISION");
}

// REQ: COMPILE-074
TEST(EndpointCatalog, SchemaAdmissionRejectsAFamilyIdThatDoesNotMatchNameAndEpoch) {
  SchemaIdentityRecord schema{
      .canonical_name = "nvidia.holoscan.test.EndpointSchema",
      .compatibility_epoch = 1U,
      .family_id = SchemaId{.value = {std::byte{0xA5}}},
      .file_identifier = make_flatbuffer_file_identifier("EPSC"),
  };
  PayloadIdentityRecord payload{
      .kind = PayloadIdentityKind::kSchema,
      .payload_kind = PayloadKind::kSerializedBytes,
      .schema = schema,
  };
  auto endpoint = identity_endpoint(std::move(payload));
  endpoint.schema_writer_max_serialized_bytes = 1U;
  endpoint.schema_identity = std::move(schema);
  const auto result =
      detail::ChannelEndpointCatalogAccess::normalize("schema-validation", {std::move(endpoint)});
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "ENDPOINT_PAYLOAD_IDENTITY_INVALID");
}

// REQ: COMPILE-022
// REQ: COMPILE-074
// REQ: SCHEMA-005
TEST(EndpointCatalog, ParserRejectsRetiredExactArtifactSchemaIdentityFields) {
  SchemaIdentityRecord schema{
      .canonical_name = "nvidia.holoscan.test.EndpointSchema",
      .compatibility_epoch = 1U,
      .family_id = derive_schema_family_id("nvidia.holoscan.test.EndpointSchema", 1U),
      .file_identifier = make_flatbuffer_file_identifier("EPSC"),
  };
  auto endpoint = identity_endpoint(PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kSchema,
      .payload_kind = PayloadKind::kSerializedBytes,
      .schema = schema,
  });
  endpoint.schema_writer_max_serialized_bytes = 1U;
  endpoint.schema_identity = std::move(schema);
  auto normalized =
      detail::ChannelEndpointCatalogAccess::normalize("schema-validation", {std::move(endpoint)});
  ASSERT_TRUE(normalized) << normalized.error().message;

  constexpr std::string_view kSchemaPrefix{"\"schema_identity\":{"};
  constexpr std::array<std::string_view, 3> kRetiredFields{
      "\"version\":1,",
      "\"schema_sha256\":\"sha256:"
      "0000000000000000000000000000000000000000000000000000000000000000\",",
      "\"id\":\"00000000000000000000000000000000\",",
  };
  for (const std::string_view retired_field : kRetiredFields) {
    std::string retired = normalized->json();
    const auto schema_record = retired.find(kSchemaPrefix);
    ASSERT_NE(schema_record, std::string::npos);
    retired.insert(schema_record + kSchemaPrefix.size(), retired_field);
    const auto parsed = ChannelEndpointCatalog::parse_json(retired);
    ASSERT_FALSE(parsed) << retired_field;
    EXPECT_EQ(parsed.error().message, "ENDPOINT_PAYLOAD_IDENTITY_INVALID") << retired_field;
  }
}

// REQ: COMPILE-007
// REQ: COMPILE-022
TEST(EndpointCatalog, RawSourceProvenanceIsNonsemanticAndNeverFabricated) {
  auto builder = ChannelEndpointCatalog::builder("raw-provenance");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);
  EXPECT_FALSE(built->raw_source_sha256().has_value());
  EXPECT_FALSE(built->raw_source_size_bytes().has_value());

  auto canonical = ChannelEndpointCatalog::parse_json(built->json());
  ASSERT_TRUE(canonical);
  const std::string formatted = " \n" + built->json() + "\n";
  auto reformatted = ChannelEndpointCatalog::parse_json(formatted);
  ASSERT_TRUE(reformatted);
  EXPECT_EQ(reformatted->json(), canonical->json());
  EXPECT_EQ(reformatted->digest(), canonical->digest());
  ASSERT_TRUE(reformatted->raw_source_sha256().has_value());
  ASSERT_TRUE(canonical->raw_source_sha256().has_value());
  EXPECT_NE(*reformatted->raw_source_sha256(), *canonical->raw_source_sha256());
  EXPECT_EQ(*reformatted->raw_source_size_bytes(), formatted.size());
}

// REQ: COMPILE-007
// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, SaveIsAtomicLoadableAndDoesNotRewriteIdenticalCanonicalBytes) {
  auto builder = ChannelEndpointCatalog::builder("save-contract");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("holoscan_endpoint_catalog_" + std::to_string(unique) + ".json");
  std::error_code cleanup_error;
  std::filesystem::remove(path, cleanup_error);
  ASSERT_TRUE(built->save_json(path));
  const auto first_write_time = std::filesystem::last_write_time(path);
  ASSERT_TRUE(built->save_json(path));
  EXPECT_EQ(std::filesystem::last_write_time(path), first_write_time);

  auto loaded = ChannelEndpointCatalog::load_json(path);
  ASSERT_TRUE(loaded);
  EXPECT_EQ(loaded->json(), built->json());
  EXPECT_EQ(loaded->digest(), built->digest());
  ASSERT_TRUE(loaded->raw_source_size_bytes().has_value());
  EXPECT_EQ(*loaded->raw_source_size_bytes(), built->json().size());
  std::filesystem::remove(path, cleanup_error);
  EXPECT_FALSE(cleanup_error);
}

// REQ: COMPILE-007
// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, SaveRepairsAnOversizedExistingDestination) {
  auto builder = ChannelEndpointCatalog::builder("save-oversized-destination");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("holoscan_endpoint_catalog_oversized_destination_" + std::to_string(unique) + ".json");
  {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(stream);
    stream.write(built->json().data(), static_cast<std::streamsize>(built->json().size()));
    stream.put('\0');
    ASSERT_TRUE(stream);
  }
  ASSERT_EQ(std::filesystem::file_size(path), built->json().size() + 1U);

  ASSERT_TRUE(built->save_json(path));
  EXPECT_EQ(std::filesystem::file_size(path), built->json().size());
  auto repaired = ChannelEndpointCatalog::load_json(path);
  ASSERT_TRUE(repaired);
  EXPECT_EQ(repaired->json(), built->json());

  std::error_code cleanup_error;
  std::filesystem::remove(path, cleanup_error);
  EXPECT_FALSE(cleanup_error);
}

// REQ: COMPILE-007
// REQ: COMPILE-022
// REQ: COMPILE-066
TEST(EndpointCatalog, SaveRejectsAndCleansAnOversizedStagedFile) {
  auto builder = ChannelEndpointCatalog::builder("save-oversized-stage");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("holoscan_endpoint_catalog_oversized_stage_" + std::to_string(unique) + ".json");
  std::error_code cleanup_error;
  std::filesystem::remove(path, cleanup_error);

  struct SaveFaultReset {
    ~SaveFaultReset() { detail::EndpointCatalogTestAccess::reset_save_faults(); }
  };
  const SaveFaultReset reset;
  detail::EndpointCatalogTestAccess::oversize_next_staged_save();
  const auto result = built->save_json(path);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_EQ(result.error().message, "ENDPOINT_CATALOG_SAVE_FAILED");
  EXPECT_FALSE(std::filesystem::exists(path));

  const std::string temporary_prefix = path.filename().string() + ".tmp-";
  for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
    EXPECT_FALSE(entry.path().filename().string().starts_with(temporary_prefix));
  }
}

// REQ: COMPILE-024
TEST(EndpointCatalog, CompatibilityChecksCompletePayloadAndCapacityContracts) {
  // Key setup: required uses uint32 with queue=4/borrow=3; candidate uses uint64
  // with queue=2/borrow=1. Every capacity and type dimension differs, ensuring
  // the full set of compatibility diagnostics fires in a single report.
  TopicOptions required_options;
  required_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 4U,
      .max_borrowed_samples = 3U,
  };
  auto required_builder = ChannelEndpointCatalog::builder("compatibility");
  ASSERT_TRUE(
      required_builder.export_topic<std::uint32_t>("result", "holomq://result", required_options));
  auto required = std::move(required_builder).build();
  ASSERT_TRUE(required);

  TopicOptions candidate_options;
  candidate_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 2U,
      .max_borrowed_samples = 1U,
  };
  auto candidate_builder = ChannelEndpointCatalog::builder("compatibility");
  ASSERT_TRUE(candidate_builder.export_topic<std::uint64_t>(
      "result", "holomq://result", candidate_options));
  auto candidate = std::move(candidate_builder).build();
  ASSERT_TRUE(candidate);

  const auto report = required->compatibility_with(*candidate);
  EXPECT_FALSE(report.compatible());
  const auto has_token = [&](std::string_view token) {
    return std::any_of(report.diagnostics().begin(),
                       report.diagnostics().end(),
                       [&](const auto& diagnostic) { return diagnostic.token == token; });
  };
  EXPECT_TRUE(has_token("ENDPOINT_TYPE_MISMATCH"));
  EXPECT_TRUE(has_token("ENDPOINT_SERVICE_MAX_QUEUE_INSUFFICIENT"));
  EXPECT_TRUE(has_token("ENDPOINT_SERVICE_MAX_BORROW_INSUFFICIENT"));
  EXPECT_TRUE(has_token("ENDPOINT_SUBSCRIBER_QUEUE_REQUEST_EXCEEDS_SERVICE"));
  EXPECT_TRUE(has_token("ENDPOINT_SUBSCRIBER_BORROW_REQUEST_EXCEEDS_SERVICE"));
  EXPECT_TRUE(has_token("ENDPOINT_BINDING_MISMATCH"));
  // Key assertion: exactly 6 diagnostics in causal order; the final
  // ENDPOINT_BINDING_MISMATCH carries the cause_id of the first type failure.
  ASSERT_EQ(report.diagnostics().size(), 6U);
  for (std::size_t index = 0U; index < report.diagnostics().size(); ++index) {
    EXPECT_EQ(report.diagnostics()[index].id.value, index + 1U);
    EXPECT_EQ(report.diagnostics()[index].service, "result.service");
    EXPECT_EQ(report.diagnostics()[index].endpoint, "result");
    EXPECT_FALSE(report.diagnostics()[index].json_pointer.empty());
    EXPECT_FALSE(report.diagnostics()[index].required.empty());
    EXPECT_FALSE(report.diagnostics()[index].provided.empty());
  }
  ASSERT_TRUE(report.diagnostics().back().cause_id.has_value());
  EXPECT_EQ(report.diagnostics().back().cause_id->value, 1U);
  EXPECT_NE(report.json().find("\"code\":\"ENDPOINT_TYPE_MISMATCH\""), std::string::npos);
  EXPECT_NE(report.json().find("\"json_pointer\":\"/services/0/payload\""), std::string::npos);
  EXPECT_NE(report.json().find("\"cause_id\":1"), std::string::npos);
}

// REQ: COMPILE-024
TEST(EndpointCatalog, CompatibilityKeepsEndpointLocalDifferencesOutOfServiceBinding) {
  TopicOptions required_options;
  required_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = std::chrono::seconds{1},
  };
  auto required_builder = ChannelEndpointCatalog::builder("required-scope");
  ASSERT_TRUE(
      required_builder.import_topic<std::uint32_t>("input", "holomq://required", required_options));
  auto required = std::move(required_builder).build();
  ASSERT_TRUE(required);

  TopicOptions candidate_options;
  candidate_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kOptional,
      .max_wait = std::chrono::seconds{2},
  };
  auto candidate_builder = ChannelEndpointCatalog::builder("candidate-scope");
  ASSERT_TRUE(candidate_builder.import_topic<std::uint32_t>(
      "input", "holomq://candidate", candidate_options));
  auto candidate = std::move(candidate_builder).build();
  ASSERT_TRUE(candidate);

  const auto report = required->compatibility_with(*candidate);
  ASSERT_FALSE(report.compatible());
  ASSERT_GE(report.diagnostics().size(), 4U);
  EXPECT_EQ(report.diagnostics()[0].token, "ENDPOINT_SCOPE_MISMATCH");
  EXPECT_EQ(report.diagnostics()[0].json_pointer, "/scope");
  EXPECT_EQ(report.diagnostics()[1].token, "ENDPOINT_ATTACH_BOUND_EXCEEDED");
  EXPECT_EQ(report.diagnostics()[1].json_pointer, "/endpoints/0/attach/max_wait_ns");
  EXPECT_EQ(report.diagnostics()[2].token, "ENDPOINT_PEER_REQUIREMENT_MISMATCH");
  EXPECT_EQ(report.diagnostics()[2].json_pointer, "/endpoints/0/attach/peer");
  EXPECT_EQ(report.diagnostics()[3].token, "ENDPOINT_URI_MISMATCH");
  EXPECT_EQ(report.diagnostics()[3].json_pointer, "/endpoints/0/uri");
  EXPECT_EQ(std::count_if(report.diagnostics().begin(),
                          report.diagnostics().end(),
                          [](const auto& diagnostic) {
                            return diagnostic.token == "ENDPOINT_BINDING_MISMATCH";
                          }),
            0U);
}

// REQ: COMPILE-024
//
//   required:  [holomq://shared, name="sensor"]          (1 endpoint)
//   candidate: [holomq://shared, name="a"] + [name="b"]  (2 endpoints, same URI)
//   Compatibility: sensor is AMBIGUOUS (2 URI matches); a and b are UNEXPECTED.
TEST(EndpointCatalog, CompatibilityOrdersAmbiguityAndUnexpectedRowsDeterministically) {
  auto required_builder = ChannelEndpointCatalog::builder("lookup-matrix");
  ASSERT_TRUE(required_builder.import_topic<std::uint32_t>("sensor", "holomq://shared"));
  auto required = std::move(required_builder).build();
  ASSERT_TRUE(required);

  auto candidate_builder = ChannelEndpointCatalog::builder("lookup-matrix");
  ASSERT_TRUE(candidate_builder.import_topic<std::uint32_t>("a", "holomq://shared"));
  ASSERT_TRUE(candidate_builder.import_topic<std::uint32_t>("b", "holomq://shared"));
  auto candidate = std::move(candidate_builder).build();
  ASSERT_TRUE(candidate);

  const auto report = required->compatibility_with(*candidate);
  ASSERT_FALSE(report.compatible());
  ASSERT_EQ(report.diagnostics().size(), 3U);
  EXPECT_EQ(report.diagnostics()[0].token, "ENDPOINT_UNEXPECTED");
  EXPECT_EQ(report.diagnostics()[1].token, "ENDPOINT_UNEXPECTED");
  EXPECT_EQ(report.diagnostics()[2].token, "ENDPOINT_AMBIGUOUS");
  constexpr std::string_view kExpected{
      "{\"compatible\":false,\"diagnostics\":["
      "{\"id\":1,\"code\":\"ENDPOINT_UNEXPECTED\",\"severity\":\"error\",\"service\":"
      "\"a.service\",\"endpoint\":\"a\",\"json_pointer\":\"/endpoints/0\",\"required\":"
      "\"absent\",\"provided\":\"a\"},"
      "{\"id\":2,\"code\":\"ENDPOINT_UNEXPECTED\",\"severity\":\"error\",\"service\":"
      "\"b.service\",\"endpoint\":\"b\",\"json_pointer\":\"/endpoints/1\",\"required\":"
      "\"absent\",\"provided\":\"b\"},"
      "{\"id\":3,\"code\":\"ENDPOINT_AMBIGUOUS\",\"severity\":\"error\",\"service\":"
      "\"sensor.service\",\"endpoint\":\"sensor\",\"json_pointer\":\"/endpoints/0\","
      "\"required\":\"sensor\",\"provided\":\"2 uri matches\"}]}"};
  EXPECT_EQ(report.json(), kExpected);
}

// REQ: COMPILE-024
TEST(EndpointCatalog, CompatibilityChainsServiceStaticCausesToTheDerivedBinding) {
  auto builder = ChannelEndpointCatalog::builder("service-static-matrix");
  ASSERT_TRUE(builder.export_topic<std::uint32_t>("result", "holomq://result"));
  auto required = std::move(builder).build();
  ASSERT_TRUE(required);
  const auto* required_state = detail::ChannelEndpointCatalogAccess::state(*required);
  ASSERT_NE(required_state, nullptr);

  // Key setup: mutate publisher_history_depth and event_wake on the candidate's
  // internal state to synthesize service-static mismatches without rebuilding a
  // full catalog via the public builder API.
  auto mutated = std::make_shared<detail::ChannelEndpointCatalogState>(*required_state);
  mutated->endpoints.front().service_static.publisher_history_depth = 1U;
  mutated->endpoints.front().service_static.event_wake = false;
  const auto candidate = detail::ChannelEndpointCatalogAccess::make(std::move(mutated));
  const auto report = required->compatibility_with(candidate);
  ASSERT_FALSE(report.compatible());
  ASSERT_EQ(report.diagnostics().size(), 3U);
  EXPECT_EQ(report.diagnostics()[0].token, "ENDPOINT_SERVICE_READINESS_MISMATCH");
  EXPECT_EQ(report.diagnostics()[1].token, "ENDPOINT_SERVICE_HISTORY_MISMATCH");
  EXPECT_EQ(report.diagnostics()[2].token, "ENDPOINT_BINDING_MISMATCH");
  // Key assertion: ENDPOINT_BINDING_MISMATCH carries the cause_id of the first
  // service-static failure, proving causality chains through the compatibility report.
  ASSERT_TRUE(report.diagnostics()[2].cause_id.has_value());
  EXPECT_EQ(report.diagnostics()[2].cause_id->value, 1U);
}

// REQ: COMPILE-024
TEST(EndpointCatalog, CompatibilityReportsRawProvenanceAsANonsemanticWarning) {
  auto builder = ChannelEndpointCatalog::builder("provenance-matrix");
  ASSERT_TRUE(builder.import_topic<std::uint32_t>("sensor", "holomq://sensor"));
  auto built = std::move(builder).build();
  ASSERT_TRUE(built);
  auto canonical = ChannelEndpointCatalog::parse_json(built->json());
  ASSERT_TRUE(canonical);
  auto reformatted = ChannelEndpointCatalog::parse_json(" \n" + built->json() + "\n");
  ASSERT_TRUE(reformatted);

  const auto report = canonical->compatibility_with(*reformatted);
  ASSERT_TRUE(report.compatible());
  ASSERT_EQ(report.diagnostics().size(), 1U);
  EXPECT_EQ(report.diagnostics().front().token, "ENDPOINT_SOURCE_PROVENANCE_CHANGED");
  EXPECT_EQ(report.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  EXPECT_NE(report.json().find("\"severity\":\"warning\""), std::string::npos);
  EXPECT_EQ(canonical->json().find("\"profile\""), std::string::npos);
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
