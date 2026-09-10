/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/params/error.hpp>
#include <holoscan/params/rebind.hpp>

#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {
namespace {

// Minimal params schema for rebind tests: one required (load-time) field and one transient
// optional field. Exercises field-origin tracking (kCaptured vs kBaseline vs kDefault)
// and projection markers across all ParamRebindMode variants.
struct RebindParams {
  std::int32_t required{1};
  std::int32_t optional{2};
};

HOLOSCAN_PARAM_SCHEMA(RebindParams, param_schema("holoscan.test.RebindParams", 1),
                      field<&RebindParams::required>("required").required_on_load(),
                      field<&RebindParams::optional>("optional").transient());

// No-op operator carrying RebindParams. Used to register operators in test graphs without
// any real compute logic; schema registration is the only requirement here.
class RebindOperator final : public Operator<RebindParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct OpaqueRebindParams {
  std::int32_t value{};
};

// Deliberately has no HOLOSCAN_PARAM_SCHEMA declaration. Narrow target capture must ignore this
// Operator when it is outside the caller-declared destination authority and reject it when
// selected.
class OpaqueRebindOperator final : public Operator<OpaqueRebindParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct FactoryCloneProbe {
  std::shared_ptr<std::atomic<std::uint32_t>> copies;

  FactoryCloneProbe() = default;
  explicit FactoryCloneProbe(std::shared_ptr<std::atomic<std::uint32_t>> value)
      : copies(std::move(value)) {}
  FactoryCloneProbe(const FactoryCloneProbe& other) : copies(other.copies) {
    if (copies) {
      copies->fetch_add(1U, std::memory_order_relaxed);
    }
  }
  FactoryCloneProbe& operator=(const FactoryCloneProbe&) = default;
};

class CloneObservedOpaqueOperator final : public Operator<OpaqueRebindParams> {
 public:
  // The by-value hop deliberately distinguishes factory-capture cloning from construction moves.
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  explicit CloneObservedOpaqueOperator(FactoryCloneProbe probe) : probe_(std::move(probe)) {}

  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }

 private:
  FactoryCloneProbe probe_;
};

// Embedded sub-graph holding a single RebindOperator at path "embedded/op". Used by
// CompleteRebindHonorsRootGraphAndOperatorScopes to exercise graph-scoped snapshot targeting.
class RebindChildGraph final : public Graph {
 protected:
  void compose() override { (void)op<RebindOperator>("op"); }
};

// Builds a ParamSnapshotFieldRecord with a concrete i32 value for the named RebindParams field.
// Returns an empty record if `name` is not found in the schema or fails path parsing.
[[nodiscard]] ParamSnapshotFieldRecord rebind_field(std::string_view name, std::int32_t value) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RebindParams>();
  const auto found = std::ranges::find(schema->fields, name, &ParamFieldDescriptor::canonical_name);
  if (found == schema->fields.end()) {
    return {};
  }
  auto path = ParamPath::parse(name);
  if (!path) {
    return {};
  }
  return ParamSnapshotFieldRecord{
      .address = ParamAddress{.field = found->id},
      .full_id = found->full_id,
      .path = std::move(*path),
      .logical_type = found->logical_type,
      .type =
          ParamSnapshotTypeIdentity{
              .id = "holoscan.i32", .version = 1U, .shape = found->type_shape},
      .codec = ParamSnapshotCodecIdentity{.id = "holoscan.config.i32", .version = 1U},
      .projection = ParamProjectionStatus::kIncluded,
      .value = ParamValue::from(value),
  };
}

// Builds a projection-status-only marker (no codec, no value) for the named field.
// Represents fields excluded from a snapshot — e.g. kExcludedTransient or
// kExcludedUnsupportedCodec — for testing how the rebind layer handles missing values.
[[nodiscard]] ParamSnapshotFieldRecord rebind_marker(std::string_view name,
                                                     ParamProjectionStatus projection) {
  ParamSnapshotFieldRecord marker = rebind_field(name, 0);
  marker.projection = projection;
  marker.codec.reset();
  marker.value.reset();
  return marker;
}

// Constructs a complete ParamSnapshot for a single operator at `path` under `scope`.
// Fields are sorted by field ID before packaging. Pass include_unknown=true to append
// an unrecognised operator entry, which triggers unknown-operator validation in to_overrides.
[[nodiscard]] ParamSnapshot scoped_rebind_snapshot(std::string_view path, ParamSnapshotScope scope,
                                                   bool include_unknown = false,
                                                   std::vector<ParamSnapshotFieldRecord> fields = {
                                                       rebind_field("required", 42)}) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RebindParams>();
  std::ranges::sort(
      fields, {}, [](const ParamSnapshotFieldRecord& field) { return field.address.field; });
  auto operator_path = OperatorPath::parse(path);
  if (!operator_path) {
    return {};
  }
  ParamSnapshotOperatorRecord record{
      .path = std::move(*operator_path),
      .effective_profile = Profile::kStandard,
      .schema_availability = ParamSchemaAvailability::kDeclared,
      .schema_id = std::string{schema->stable_id},
      .schema_version = schema->version,
      .structural_hash = schema->structural_hash,
      .contract_hash = schema->contract_hash,
      .metadata_hash = schema->metadata_hash,
      .schema_coverage = schema->coverage,
      .schema_proof = schema->proof,
      .generation = 1U,
      .accepted_update_sequence = 1U,
      .published_update_sequence = 1U,
      .state = ParamUpdateState::kPublished,
      .filter = ParamSnapshotFilter::kPolicyDefault,
      .value_coverage = ParamSnapshotValueCoverage::kPolicyFiltered,
      .fields = std::move(fields),
  };
  std::vector<ParamSnapshotOperatorRecord> records;
  records.push_back(std::move(record));
  if (include_unknown) {
    auto unknown_path = OperatorPath::parse("unknown");
    if (!unknown_path) {
      return {};
    }
    auto unknown = records.front();
    unknown.path = std::move(*unknown_path);
    records.push_back(std::move(unknown));
  }
  return holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      {},
      {},
      {},
      {},
      ParamReadView::kPublished,
      ParamCaptureConsistency::kRequireStopped,
      std::move(scope),
      ParamUpdateSequence{1U},
      ParamUpdateSequence{1U},
      std::move(records));
}

// Convenience wrapper: root-scoped snapshot for operator "op" with required=42 as the only field.
[[nodiscard]] ParamSnapshot rebind_snapshot(bool include_unknown = false) {
  return scoped_rebind_snapshot("op", ParamSnapshotScope::root(), include_unknown);
}

// REQ: PARAM-024
TEST(ParameterTools, PreservesTypedCoreCause) {
  const auto error =
      from_core_error(Error{ErrorCode::kTimeout, "bounded wait"}, "PARAM_TOOL_CORE_TIMEOUT");
  EXPECT_EQ(error.code(), ParamToolErrorCode::kTimeout);
  EXPECT_EQ(error.token(), "PARAM_TOOL_CORE_TIMEOUT");
  const auto& core_cause = error.core_cause();
  ASSERT_TRUE(core_cause.has_value());
  if (core_cause.has_value()) {
    EXPECT_EQ(core_cause->code, ErrorCode::kTimeout);
    EXPECT_EQ(core_cause->message, "bounded wait");
  }
}

// REQ: PARAM-024
TEST(ParameterTools, PreservesTypedConfigCause) {
  const auto cause = config::ConfigError::make(
      config::ConfigErrorCode::kParseError, "invalid configuration", "/camera/fps");
  const auto error = from_config_error(cause, "PARAM_TOOL_CONFIG_INVALID");
  EXPECT_EQ(error.code(), ParamToolErrorCode::kInvalidArgument);
  EXPECT_EQ(error.token(), "PARAM_TOOL_CONFIG_INVALID");
  const auto& config_cause = error.config_cause();
  ASSERT_TRUE(config_cause.has_value());
  if (config_cause.has_value()) {
    EXPECT_EQ(config_cause->code(), config::ConfigErrorCode::kParseError);
    EXPECT_EQ(config_cause->message(), "invalid configuration");
    EXPECT_EQ(config_cause->path(), "/camera/fps");
  }
}

// REQ: PARAM-024
TEST(ParameterTools, FormatsConfigCauseForHumanReadableLogs) {
  const auto cause = config::ConfigError::make(
      config::ConfigErrorCode::kParseError, "invalid configuration", "/camera/fps");
  const auto error = from_config_error(cause, "PARAM_TOOL_CONFIG_INVALID");

  const std::string formatted = format_error(error);

  EXPECT_EQ(formatted,
            "PARAM_TOOL_CONFIG_INVALID; caused by: CONFIG_PARSE_ERROR: invalid configuration "
            "[path: /camera/fps]");
}

// REQ: PARAM-024
TEST(ParameterTools, FormatsCoreCauseForHumanReadableLogs) {
  const auto error = from_core_error(Error{ErrorCode::kTimeout, "bounded\x1b[2J\nwait"},
                                     "PARAM_TOOL_CORE_TIMEOUT");

  const std::string formatted = format_error(error);

  EXPECT_EQ(formatted, "PARAM_TOOL_CORE_TIMEOUT; caused by: kTimeout: bounded [2J wait");
}

// REQ: PARAM-024
TEST(ParameterTools, OversizedDiagnosticTokenMapsToStableBoundedToken) {
  const std::string oversized(ParamToolError::kTokenCapacity + 1U, 'X');
  const ParamToolError error{ParamToolErrorCode::kInvalidArgument, oversized};
  EXPECT_EQ(error.token(), "PARAM_TOOL_TOKEN_INVALID");
  EXPECT_LE(error.token().size(), ParamToolError::kTokenCapacity);
}

// REQ: PARAM-024
TEST(ParameterTools, MalformedDiagnosticTokensMapToStableBoundedToken) {
  constexpr std::array<std::string_view, 4> invalid_tokens{
      "PARAM TOOL", "PARAM\nTOOL", "param_tool", "1_PARAM_TOOL"};

  for (const std::string_view token : invalid_tokens) {
    SCOPED_TRACE(token);
    const ParamToolError error{ParamToolErrorCode::kInvalidArgument, token};
    EXPECT_EQ(error.token(), "PARAM_TOOL_TOKEN_INVALID");
  }
}

// REQ: PARAM-027
// REQ: PARAM-035
TEST(ParameterTools, StandardAdmissionContextIsExplicit) {
  auto context = ParamAdmissionContext::standard();
  ASSERT_TRUE(context.has_value());
  EXPECT_EQ(context->profile(), Profile::kStandard);
}

// REQ: PARAM-035
// Graph: graph["rebind-public-scopes"]
//   ├─ root<RebindOperator>("root")
//   └─ embedded<RebindChildGraph>
//        └─ op<RebindOperator>("op")
TEST(ParameterTools, PublicTargetReportsEveryExplicitDestinationScopeKind) {
  Graph graph{"rebind-public-scopes"};
  (void)graph.op<RebindOperator>("root");
  auto embedded = graph.embed<RebindChildGraph>("embedded");
  ASSERT_TRUE(embedded);
  auto context = ParamAdmissionContext::standard();
  ASSERT_TRUE(context.has_value());

  auto whole = make_param_rebind_target(graph, *context, ParamRebindTargetScope::whole_graph());
  ASSERT_TRUE(whole.has_value()) << whole.error().detail().token();
  EXPECT_EQ(whole->scope_kind(), ParamRebindTargetScopeKind::kWholeGraph);
  EXPECT_TRUE(whole->scope_paths().empty());

  const auto embedded_path = embedded.path();
  ASSERT_TRUE(embedded_path.has_value());
  // GoogleTest's fatal assertion establishes the precondition but is opaque to clang-tidy.
  auto graph_scope = make_param_rebind_target(
      graph,
      *context,
      ParamRebindTargetScope::graph(
          embedded_path.value()));  // NOLINT(bugprone-unchecked-optional-access)
  ASSERT_TRUE(graph_scope.has_value()) << graph_scope.error().detail().token();
  EXPECT_EQ(graph_scope->scope_kind(), ParamRebindTargetScopeKind::kGraph);
  ASSERT_EQ(graph_scope->scope_paths().size(), 1U);
  EXPECT_EQ(graph_scope->scope_paths().front(), "embedded");

  auto root_path = OperatorPath::parse("root");
  ASSERT_TRUE(root_path.has_value());
  auto operator_scope =
      make_param_rebind_target(graph, *context, ParamRebindTargetScope::operator_(*root_path));
  ASSERT_TRUE(operator_scope.has_value()) << operator_scope.error().detail().token();
  EXPECT_EQ(operator_scope->scope_kind(), ParamRebindTargetScopeKind::kOperator);
  ASSERT_EQ(operator_scope->scope_paths().size(), 1U);
  EXPECT_EQ(operator_scope->scope_paths().front(), "root");

  auto first_path = OperatorPath::parse("embedded/op");
  auto second_path = OperatorPath::parse("root");
  ASSERT_TRUE(first_path.has_value());
  ASSERT_TRUE(second_path.has_value());
  std::vector<OperatorPath> operator_paths;
  operator_paths.push_back(std::move(*first_path));
  operator_paths.push_back(std::move(*second_path));
  auto operator_set_scope = make_param_rebind_target(
      graph, *context, ParamRebindTargetScope::operator_set(operator_paths));
  ASSERT_TRUE(operator_set_scope.has_value()) << operator_set_scope.error().detail().token();
  EXPECT_EQ(operator_set_scope->scope_kind(), ParamRebindTargetScopeKind::kOperatorSet);
  ASSERT_EQ(operator_set_scope->scope_paths().size(), 2U);
  EXPECT_EQ(operator_set_scope->scope_paths()[0], "embedded/op");
  EXPECT_EQ(operator_set_scope->scope_paths()[1], "root");
}

// REQ: PARAM-035
// Graph: graph["rebind-owned-target-lifetime"]
//   └─ op<RebindOperator>("op")
TEST(ParameterTools, CapturedTargetOwnsFactoriesAfterSourceGraphDestruction) {
  auto context = ParamAdmissionContext::standard();
  ASSERT_TRUE(context.has_value());
  std::optional<ParamRebindTarget> target;
  {
    Graph graph{"rebind-owned-target-lifetime"};
    (void)graph.op<RebindOperator>("op");
    auto path = OperatorPath::parse("op");
    ASSERT_TRUE(path.has_value());
    auto captured =
        make_param_rebind_target(graph, *context, ParamRebindTargetScope::operator_(*path));
    ASSERT_TRUE(captured.has_value()) << captured.error().detail().token();
    target.emplace(std::move(*captured));
  }
  ASSERT_TRUE(target.has_value());

  auto overrides = to_overrides(rebind_snapshot(), target.value());
  ASSERT_TRUE(overrides.has_value()) << overrides.error().detail().token();
  ASSERT_EQ(overrides->provenance().target_operators.size(), 1U);
  EXPECT_EQ(overrides->provenance().target_operators.front().operator_path, "op");
}

// REQ: PARAM-035
// REQ: PARAM-036
// Graph: graph["rebind-baseline"]
//   └─ op<RebindOperator>("op")
TEST(ParameterTools, SparseAndCompleteRebindUseDifferentTargetBaselines) {
  // Key setup: operator baseline has required=7, optional=9; snapshot delivers required=42.
  Graph graph{"rebind-baseline"};
  auto op = graph.op<RebindOperator>("op");
  op.set_params(RebindParams{.required = 7, .optional = 9});
  const ParamSnapshot snapshot = rebind_snapshot();

  // Key assertion: sparse keeps optional=9 from baseline; complete falls back to schema default=2.
  auto sparse = to_overrides(snapshot, graph);
  ASSERT_TRUE(sparse.has_value());
  const auto& sparse_state = holoscan::detail::ParamOverrideSetAccess::state(*sparse);
  ASSERT_EQ(sparse_state.operators.size(), 1U);
  const auto* sparse_value =
      static_cast<const RebindParams*>(sparse_state.operators.front().resolved_value.get());
  ASSERT_NE(sparse_value, nullptr);
  EXPECT_EQ(sparse_value->required, 42);
  EXPECT_EQ(sparse_value->optional, 9);
  const auto& sparse_provenance = sparse->provenance();
  EXPECT_EQ(sparse_provenance.mode, ParamRebindMode::kSparseContributions);
  ASSERT_EQ(sparse_provenance.fields.size(), 2U);
  EXPECT_EQ(std::ranges::count(sparse_provenance.fields,
                               ParamFieldRebindOrigin::kCaptured,
                               &ParamRebindFieldProvenance::origin),
            1U);
  EXPECT_EQ(std::ranges::count(sparse_provenance.fields,
                               ParamFieldRebindOrigin::kBaseline,
                               &ParamRebindFieldProvenance::origin),
            1U);

  auto complete =
      to_overrides(snapshot, graph, ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_TRUE(complete.has_value());
  const auto& complete_state = holoscan::detail::ParamOverrideSetAccess::state(*complete);
  ASSERT_EQ(complete_state.operators.size(), 1U);
  const auto* complete_value =
      static_cast<const RebindParams*>(complete_state.operators.front().resolved_value.get());
  ASSERT_NE(complete_value, nullptr);
  EXPECT_EQ(complete_value->required, 42);
  EXPECT_EQ(complete_value->optional, 2);
  const auto& complete_provenance = complete->provenance();
  EXPECT_EQ(complete_provenance.mode, ParamRebindMode::kCompleteObject);
  EXPECT_EQ(std::ranges::count(complete_provenance.fields,
                               ParamFieldRebindOrigin::kDefault,
                               &ParamRebindFieldProvenance::origin),
            1U);
}

// REQ: PARAM-028
// Graph: graph["rebind-no-value-markers"]
//   └─ op<RebindOperator>("op")
TEST(ParameterTools, CompleteRebindDefaultsOptionalMarkersAndRequiresRequiredValues) {
  Graph graph{"rebind-no-value-markers"};
  auto op = graph.op<RebindOperator>("op");
  op.set_params(RebindParams{.required = 7, .optional = 9});

  // Key setup: first sub-case has a real value for "required" and an ExcludedTransient marker
  // for "optional" — expected to succeed, with optional resolved to schema default.
  std::vector<ParamSnapshotFieldRecord> optional_marker_fields;
  optional_marker_fields.push_back(rebind_field("required", 42));
  optional_marker_fields.push_back(
      rebind_marker("optional", ParamProjectionStatus::kExcludedTransient));
  const ParamSnapshot optional_marker_snapshot = scoped_rebind_snapshot(
      "op", ParamSnapshotScope::root(), false, std::move(optional_marker_fields));
  auto accepted = to_overrides(optional_marker_snapshot,
                               graph,
                               ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_TRUE(accepted.has_value()) << accepted.error().detail().token();
  const auto& accepted_state = holoscan::detail::ParamOverrideSetAccess::state(*accepted);
  ASSERT_EQ(accepted_state.operators.size(), 1U);
  const auto* resolved =
      static_cast<const RebindParams*>(accepted_state.operators.front().resolved_value.get());
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->required, 42);
  EXPECT_EQ(resolved->optional, 2);
  EXPECT_EQ(std::ranges::count(accepted->provenance().fields,
                               ParamFieldRebindOrigin::kDefault,
                               &ParamRebindFieldProvenance::origin),
            1U);

  // Key assertion: second sub-case replaces "required" with an UnsupportedCodec marker too;
  // kCompleteObject must reject because a required field has no value to contribute.
  std::vector<ParamSnapshotFieldRecord> required_marker_fields;
  required_marker_fields.push_back(
      rebind_marker("required", ParamProjectionStatus::kExcludedUnsupportedCodec));
  required_marker_fields.push_back(
      rebind_marker("optional", ParamProjectionStatus::kExcludedTransient));
  const ParamSnapshot required_marker_snapshot = scoped_rebind_snapshot(
      "op", ParamSnapshotScope::root(), false, std::move(required_marker_fields));
  auto rejected = to_overrides(required_marker_snapshot,
                               graph,
                               ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code(), ParamRebindErrorCode::kRequiredFieldMissing);
  EXPECT_EQ(rejected.error().detail().token(), "PARAM_REBIND_REQUIRED_FIELD_MISSING");
}

// REQ: PARAM-035
// REQ: PARAM-036
// Graph: graph["rebind-extra-source"]
//   └─ op<RebindOperator>("op")
// Snapshot also contains an "unknown" operator entry (include_unknown=true).
TEST(ParameterTools, CompleteRebindRejectsUnmatchedSourceOperator) {
  Graph graph{"rebind-extra-source"};
  (void)graph.op<RebindOperator>("op");
  auto result = to_overrides(
      rebind_snapshot(true), graph, ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ParamRebindErrorCode::kUnknownOperator);
}

// REQ: PARAM-022
// REQ: PARAM-028
// REQ: PARAM-035
// REQ: PARAM-036
// Three scope variants tested in sequence:
//
// Root-scope — snapshot must cover every operator in the graph:
//   graph["rebind-root-scope"]
//     ├─ op<RebindOperator>("op")
//     └─ op<RebindOperator>("outside")   ← unmatched → error
//
// Operator-scope — explicit target scope addresses "op" only; "outside" remains untouched:
//   graph["rebind-operator-scope"]
//     ├─ op<RebindOperator>("op")
//     └─ op<RebindOperator>("outside")
//
// Graph-scope — snapshot targets the embedded sub-graph only:
//   graph["rebind-graph-scope"]
//     ├─ embed<RebindChildGraph>("embedded")
//     │    └─ op<RebindOperator>("op")   (full path: "embedded/op")
//     └─ op<RebindOperator>("outside")
TEST(ParameterTools, CompleteRebindHonorsRootGraphAndOperatorScopes) {
  Graph root_target{"rebind-root-scope"};
  (void)root_target.op<RebindOperator>("op");
  (void)root_target.op<RebindOperator>("outside");
  auto root_result = to_overrides(
      rebind_snapshot(), root_target, ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_FALSE(root_result.has_value());
  EXPECT_EQ(root_result.error().code(), ParamRebindErrorCode::kUnknownOperator);
  EXPECT_EQ(root_result.error().target_path(), "outside");

  Graph operator_target{"rebind-operator-scope"};
  (void)operator_target.op<RebindOperator>("op");
  (void)operator_target.op<RebindOperator>("outside");
  auto operator_path = OperatorPath::parse("op");
  ASSERT_TRUE(operator_path.has_value());
  const ParamSnapshot operator_snapshot =
      scoped_rebind_snapshot("op", ParamSnapshotScope::operator_(std::move(*operator_path)));
  auto strict_result = to_overrides(operator_snapshot,
                                    operator_target,
                                    ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_FALSE(strict_result.has_value());
  EXPECT_EQ(strict_result.error().code(), ParamRebindErrorCode::kUnknownOperator);
  auto explicit_operator_path = OperatorPath::parse("op");
  ASSERT_TRUE(explicit_operator_path.has_value());
  auto operator_result = to_overrides(operator_snapshot,
                                      operator_target,
                                      ParamRebindTargetScope::operator_(*explicit_operator_path),
                                      ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_TRUE(operator_result.has_value()) << operator_result.error().detail().token();
  const auto& operator_state = holoscan::detail::ParamOverrideSetAccess::state(*operator_result);
  ASSERT_EQ(operator_state.operators.size(), 1U);
  EXPECT_EQ(operator_state.operators.front().operator_path, "op");

  Graph graph_target{"rebind-graph-scope"};
  const auto embedded = graph_target.embed<RebindChildGraph>("embedded");
  ASSERT_TRUE(embedded);
  (void)graph_target.op<RebindOperator>("outside");
  const auto graph_path = embedded.path();
  ASSERT_TRUE(graph_path.has_value());
  const ParamSnapshot graph_snapshot = [&graph_path]() {
    if (graph_path.has_value()) {
      return scoped_rebind_snapshot("embedded/op", ParamSnapshotScope::graph(graph_path.value()));
    }
    return ParamSnapshot{};
  }();
  // GoogleTest's fatal assertion above establishes the precondition but is opaque to clang-tidy.
  auto graph_result =
      to_overrides(graph_snapshot,
                   graph_target,
                   ParamRebindTargetScope::graph(
                       graph_path.value()),  // NOLINT(bugprone-unchecked-optional-access)
                   ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_TRUE(graph_result.has_value()) << graph_result.error().detail().token();
  const auto& graph_state = holoscan::detail::ParamOverrideSetAccess::state(*graph_result);
  ASSERT_EQ(graph_state.operators.size(), 1U);
  EXPECT_EQ(graph_state.operators.front().operator_path, "embedded/op");
}

// REQ: PARAM-022
// REQ: PARAM-028
// Graph: graph["rebind-invalid-scope"]
//   └─ op<RebindOperator>("outside")
// Snapshot scope is locked to operator "op", but the record path is "outside".
TEST(ParameterTools, CompleteRebindRejectsSourceOutsideDeclaredScope) {
  Graph graph{"rebind-invalid-scope"};
  (void)graph.op<RebindOperator>("outside");
  auto scope_path = OperatorPath::parse("op");
  ASSERT_TRUE(scope_path.has_value());
  const ParamSnapshot snapshot =
      scoped_rebind_snapshot("outside", ParamSnapshotScope::operator_(std::move(*scope_path)));

  auto result =
      to_overrides(snapshot, graph, ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ParamRebindErrorCode::kInvalidSnapshot);
  EXPECT_EQ(result.error().source_path(), "outside");
  EXPECT_EQ(result.error().detail().token(), "PARAM_REBIND_SOURCE_OUTSIDE_SCOPE");
}

// REQ: PARAM-035
// REQ: PARAM-036
// Graph: graph["rebind-target-scope"]
//   ├─ op<RebindOperator>("op")
//   └─ opaque<OpaqueRebindOperator>("opaque")
TEST(ParameterTools, ExplicitTargetScopeIgnoresOnlyUnaddressedOpaqueOperators) {
  Graph graph{"rebind-target-scope"};
  const auto selected = graph.op<RebindOperator>("op");
  auto unselected_factory_copies = std::make_shared<std::atomic<std::uint32_t>>(0U);
  const auto opaque =
      graph.op<CloneObservedOpaqueOperator>("opaque", FactoryCloneProbe{unselected_factory_copies});
  graph.partition("selected").add(selected);
  graph.partition("opaque").add(opaque);
  // Ignore prototype/authoring copies. Target capture below is the behavior under test.
  unselected_factory_copies->store(0U, std::memory_order_relaxed);
  auto op_path = OperatorPath::parse("op");
  ASSERT_TRUE(op_path.has_value());

  const GraphAuthoringIdentity before = graph.view().identity();
  auto narrow = to_overrides(rebind_snapshot(), graph, ParamRebindTargetScope::operator_(*op_path));
  ASSERT_TRUE(narrow.has_value()) << narrow.error().detail().token();
  EXPECT_EQ(unselected_factory_copies->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(graph.view().identity(), before);
  EXPECT_EQ(narrow->provenance().target_scope_kind, ParamRebindTargetScopeKind::kOperator);
  ASSERT_EQ(narrow->provenance().target_scope_paths.size(), 1U);
  EXPECT_EQ(narrow->provenance().target_scope_paths.front(), "op");
  ASSERT_EQ(narrow->provenance().target_operators.size(), 1U);
  EXPECT_EQ(narrow->provenance().target_operators.front().operator_path, "op");
  EXPECT_NE(narrow->provenance().target_selection_digest, ParamSchemaHash{});
  CompileOptions accepted_options;
  accepted_options.parameter_overrides = *narrow;
  const ExecutionPlan accepted_plan = compile(graph, accepted_options);
  ASSERT_TRUE(accepted_plan.ok()) << accepted_plan.json();

  ParamRebindProvenance tampered_provenance = narrow->provenance();
  tampered_provenance.target_scope_paths.front() = "opaque";
  tampered_provenance.target_selection_digest =
      holoscan::detail::ParamOverrideSetAccess::target_selection_digest(
          tampered_provenance.target_scope_kind,
          tampered_provenance.target_scope_paths,
          tampered_provenance.target_operators);
  const auto& narrow_state = holoscan::detail::ParamOverrideSetAccess::state(*narrow);
  std::vector<holoscan::detail::ParamOverrideOperatorState> tampered_operators =
      narrow_state.operators;
  CompileOptions tampered_options;
  tampered_options.parameter_overrides =
      holoscan::detail::ParamOverrideSetAccess::make_complete(graph.view().identity(),
                                                              std::move(tampered_operators),
                                                              narrow_state.admission_context_digest,
                                                              std::move(tampered_provenance));
  const ExecutionPlan tampered_plan = compile(graph, tampered_options);
  ASSERT_FALSE(tampered_plan.ok());
  EXPECT_NE(tampered_plan.json().find("PARAMETER_OVERRIDE_TARGET_SELECTION_MISMATCH"),
            std::string::npos);
  EXPECT_EQ(graph.view().identity(), before);

  ParamRebindProvenance cleared_selection = narrow->provenance();
  cleared_selection.target_selection_digest = {};
  cleared_selection.target_scope_paths.clear();
  cleared_selection.target_operators.clear();
  CompileOptions cleared_selection_options;
  cleared_selection_options.parameter_overrides =
      holoscan::detail::ParamOverrideSetAccess::make_complete(graph.view().identity(),
                                                              narrow_state.operators,
                                                              narrow_state.admission_context_digest,
                                                              std::move(cleared_selection));
  const ExecutionPlan cleared_selection_plan = compile(graph, cleared_selection_options);
  ASSERT_FALSE(cleared_selection_plan.ok());
  EXPECT_NE(cleared_selection_plan.json().find("PARAMETER_OVERRIDE_TARGET_SELECTION_MISMATCH"),
            std::string::npos);

  auto context = ParamAdmissionContext::standard();
  ASSERT_TRUE(context.has_value());
  unselected_factory_copies->store(0U, std::memory_order_relaxed);
  auto strict = make_param_rebind_target(graph, *context);
  ASSERT_FALSE(strict.has_value());
  EXPECT_GT(unselected_factory_copies->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(strict.error().code(), ParamRebindErrorCode::kSchemaUnavailable);
  EXPECT_EQ(strict.error().target_path(), "opaque");

  auto opaque_path = OperatorPath::parse("opaque");
  ASSERT_TRUE(opaque_path.has_value());
  unselected_factory_copies->store(0U, std::memory_order_relaxed);
  auto selected_opaque =
      make_param_rebind_target(graph, *context, ParamRebindTargetScope::operator_(*opaque_path));
  ASSERT_FALSE(selected_opaque.has_value());
  EXPECT_GT(unselected_factory_copies->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(selected_opaque.error().code(), ParamRebindErrorCode::kSchemaUnavailable);
  EXPECT_EQ(graph.view().identity(), before);
}

// REQ: PARAM-035
TEST(ParameterTools, CompilerRejectsClearedWholeGraphTargetSelectionEvidence) {
  Graph graph{"rebind-cleared-whole-graph-selection"};
  (void)graph.op<RebindOperator>("op");
  auto overrides = to_overrides(rebind_snapshot(), graph);
  ASSERT_TRUE(overrides) << overrides.error().detail().token();

  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*overrides);
  ParamRebindProvenance provenance = overrides->provenance();
  ASSERT_EQ(provenance.target_scope_kind, ParamRebindTargetScopeKind::kWholeGraph);
  ASSERT_FALSE(provenance.fields.empty());
  provenance.target_selection_digest = {};
  provenance.target_scope_paths.clear();
  provenance.target_operators.clear();

  CompileOptions options;
  options.parameter_overrides =
      holoscan::detail::ParamOverrideSetAccess::make_complete(graph.view().identity(),
                                                              state.operators,
                                                              state.admission_context_digest,
                                                              std::move(provenance));
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_TARGET_SELECTION_MISMATCH"), std::string::npos);
}

// REQ: PARAM-035
TEST(ParameterTools, ExplicitOperatorSetIsCanonicalBoundedAndDuplicateFree) {
  Graph graph{"rebind-target-set"};
  (void)graph.op<RebindOperator>("op");
  (void)graph.op<RebindOperator>("unknown");
  auto context = ParamAdmissionContext::standard();
  ASSERT_TRUE(context.has_value());

  auto op_first = OperatorPath::parse("op");
  auto unknown_first = OperatorPath::parse("unknown");
  auto op_second = OperatorPath::parse("op");
  auto unknown_second = OperatorPath::parse("unknown");
  ASSERT_TRUE(op_first.has_value());
  ASSERT_TRUE(unknown_first.has_value());
  ASSERT_TRUE(op_second.has_value());
  ASSERT_TRUE(unknown_second.has_value());
  std::vector<OperatorPath> reverse_paths;
  reverse_paths.push_back(std::move(*unknown_first));
  reverse_paths.push_back(std::move(*op_first));
  std::vector<OperatorPath> forward_paths;
  forward_paths.push_back(std::move(*op_second));
  forward_paths.push_back(std::move(*unknown_second));
  auto reverse = make_param_rebind_target(
      graph, *context, ParamRebindTargetScope::operator_set(reverse_paths));
  auto forward = make_param_rebind_target(
      graph, *context, ParamRebindTargetScope::operator_set(forward_paths));
  ASSERT_TRUE(reverse.has_value()) << reverse.error().detail().token();
  ASSERT_TRUE(forward.has_value()) << forward.error().detail().token();
  EXPECT_EQ(reverse->selection_digest(), forward->selection_digest());
  ASSERT_EQ(reverse->scope_paths().size(), 2U);
  EXPECT_EQ(reverse->scope_paths()[0], "op");
  EXPECT_EQ(reverse->scope_paths()[1], "unknown");

  auto duplicate_a = OperatorPath::parse("op");
  auto duplicate_b = OperatorPath::parse("op");
  ASSERT_TRUE(duplicate_a.has_value());
  ASSERT_TRUE(duplicate_b.has_value());
  std::vector<OperatorPath> duplicates;
  duplicates.push_back(std::move(*duplicate_a));
  duplicates.push_back(std::move(*duplicate_b));
  auto duplicate =
      make_param_rebind_target(graph, *context, ParamRebindTargetScope::operator_set(duplicates));
  ASSERT_FALSE(duplicate.has_value());
  EXPECT_EQ(duplicate.error().detail().token(), "PARAM_REBIND_TARGET_SCOPE_DUPLICATE");

  auto missing_path = OperatorPath::parse("missing");
  ASSERT_TRUE(missing_path.has_value());
  auto missing =
      make_param_rebind_target(graph, *context, ParamRebindTargetScope::operator_(*missing_path));
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error().code(), ParamRebindErrorCode::kScopeNotFound);

  const std::vector<OperatorPath> empty_paths;
  auto empty =
      make_param_rebind_target(graph, *context, ParamRebindTargetScope::operator_set(empty_paths));
  ASSERT_FALSE(empty.has_value());
  EXPECT_EQ(empty.error().code(), ParamRebindErrorCode::kInvalidTarget);
  EXPECT_EQ(empty.error().detail().token(), "PARAM_REBIND_TARGET_SCOPE_EMPTY");

  constexpr std::size_t kAtLimitPathCount = 65'536U;
  std::vector<OperatorPath> at_limit_paths;
  at_limit_paths.reserve(kAtLimitPathCount);
  for (std::size_t index = 0U; index < kAtLimitPathCount; ++index) {
    auto path = OperatorPath::parse("op");
    ASSERT_TRUE(path.has_value());
    at_limit_paths.push_back(std::move(*path));
  }
  auto at_limit = make_param_rebind_target(
      graph, *context, ParamRebindTargetScope::operator_set(at_limit_paths));
  ASSERT_FALSE(at_limit.has_value());
  EXPECT_EQ(at_limit.error().code(), ParamRebindErrorCode::kInvalidTarget);
  EXPECT_EQ(at_limit.error().detail().token(), "PARAM_REBIND_TARGET_SCOPE_DUPLICATE");

  auto additional_path = OperatorPath::parse("op");
  ASSERT_TRUE(additional_path.has_value());
  at_limit_paths.push_back(std::move(*additional_path));
  auto over_limit = make_param_rebind_target(
      graph, *context, ParamRebindTargetScope::operator_set(at_limit_paths));
  ASSERT_FALSE(over_limit.has_value());
  EXPECT_EQ(over_limit.error().code(), ParamRebindErrorCode::kInvalidTarget);
  EXPECT_EQ(over_limit.error().detail().token(), "PARAM_REBIND_TARGET_SCOPE_LIMIT");
}

// REQ: PARAM-036
TEST(ParameterTools, SparseRebindRejectsEverySourceOutsideExplicitTargetScope) {
  Graph graph{"rebind-sparse-target-scope"};
  (void)graph.op<RebindOperator>("op");
  auto op_path = OperatorPath::parse("op");
  ASSERT_TRUE(op_path.has_value());
  auto result =
      to_overrides(rebind_snapshot(true), graph, ParamRebindTargetScope::operator_(*op_path));
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ParamRebindErrorCode::kUnknownOperator);
  EXPECT_EQ(result.error().stage(), ParamRebindStage::kSparseConstruction);
}

// REQ: PARAM-027
// REQ: PARAM-029
// Graph: graph["rebind-alias-conflict"]
//   └─ op<RebindOperator>("op")
// Alias maps schema->fields[0] → schema->fields[1]; snapshot already has fields[0] directly,
// so the alias would produce two contributions to the same target field.
TEST(ParameterTools, DirectAndAliasFieldMatchesRejectAsAmbiguous) {
  Graph graph{"rebind-alias-conflict"};
  (void)graph.op<RebindOperator>("op");
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RebindParams>();
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->fields.size(), 2U);
  const ParamFieldAlias alias{
      .from_schema_id = std::string{schema->stable_id},
      .from_schema_version = schema->version,
      .from_field_id = schema->fields[0].id,
      .to_schema_id = std::string{schema->stable_id},
      .to_schema_version = schema->version,
      .to_field_id = schema->fields[1].id,
  };
  const std::array aliases{alias};
  auto result =
      to_overrides(rebind_snapshot(),
                   graph,
                   ParamRebindOptions{.field_aliases = std::span<const ParamFieldAlias>{aliases}});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ParamRebindErrorCode::kAliasConflict);
  EXPECT_EQ(result.error().stage(), ParamRebindStage::kFieldAlias);
  EXPECT_EQ(result.error().source_field(), schema->fields[0].id);
  EXPECT_EQ(result.error().target_path(), "op");
}

// REQ: PARAM-029
// REQ: PARAM-036
// Graph: graph["rebind-operator-alias"]
//   └─ op<RebindOperator>("renamed")
// Snapshot references "op"; operator alias maps "op" → "renamed" so the contribution lands.
TEST(ParameterTools, OperatorAliasKeepsDirectFieldOriginAndCompiles) {
  Graph graph{"rebind-operator-alias"};
  (void)graph.op<RebindOperator>("renamed");
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RebindParams>();
  ASSERT_NE(schema, nullptr);
  auto source_path = OperatorPath::parse("op");
  auto target_path = OperatorPath::parse("renamed");
  auto target_scope_path = OperatorPath::parse("renamed");
  ASSERT_TRUE(source_path.has_value());
  ASSERT_TRUE(target_path.has_value());
  ASSERT_TRUE(target_scope_path.has_value());
  const std::array aliases{ParamOperatorAlias{
      .from_path = std::move(*source_path),
      .from_schema_version = schema->version,
      .to_path = std::move(*target_path),
      .to_schema_version = schema->version,
  }};

  // Key setup: alias redirects "op" snapshot entries to the "renamed" operator.
  auto overrides = to_overrides(
      rebind_snapshot(),
      graph,
      ParamRebindTargetScope::operator_(*target_scope_path),
      ParamRebindOptions{.operator_aliases = std::span<const ParamOperatorAlias>{aliases}});
  // Key assertion: aliased field still carries kCaptured origin; plan compiles successfully.
  ASSERT_TRUE(overrides.has_value()) << overrides.error().detail().token();
  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*overrides);
  ASSERT_EQ(state.operators.size(), 1U);
  ASSERT_EQ(state.operators.front().contributions.size(), 1U);
  EXPECT_EQ(state.operators.front().contributions.front().origin,
            holoscan::detail::ParamOverrideContributionOrigin::kCaptured);
  ASSERT_EQ(overrides->provenance().operator_aliases.size(), 1U);
  const auto contributed = std::ranges::find_if(
      overrides->provenance().fields, [&](const ParamRebindFieldProvenance& field) {
        return field.target_field_id == schema->fields.front().id &&
               field.disposition == ParamFieldRebindDisposition::kContributed;
      });
  ASSERT_NE(contributed, overrides->provenance().fields.end());
  EXPECT_EQ(contributed->origin, ParamFieldRebindOrigin::kCaptured);

  CompileOptions options;
  options.parameter_overrides = *overrides;
  const ExecutionPlan plan = compile(graph, options);
  EXPECT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-029
// Graph: graph["rebind-field-alias-scale"]
//   └─ op<RebindOperator>("op")
// 65,536 field aliases are constructed before calling to_overrides with an empty snapshot.
// Verifies the alias-validation path rejects at O(n log n) cost rather than O(n²).
TEST(ParameterTools, ValidatesFieldAliasesAtHardMaximumWithoutQuadraticScan) {
  constexpr std::size_t kAliasCount = 65'536U;
  std::vector<ParamFieldAlias> aliases;
  aliases.reserve(kAliasCount);
  for (std::size_t index = 0; index < kAliasCount; ++index) {
    const ParamFieldId field = static_cast<ParamFieldId>(index + 1U);
    aliases.push_back(ParamFieldAlias{
        .from_schema_id = "holoscan.test.ScaleSource",
        .from_schema_version = 1U,
        .from_field_id = field,
        .to_schema_id = "holoscan.test.ScaleTarget",
        .to_schema_version = 1U,
        .to_field_id = field,
    });
  }

  Graph graph{"rebind-field-alias-scale"};
  (void)graph.op<RebindOperator>("op");
  const auto result =
      to_overrides(ParamSnapshot{},
                   graph,
                   ParamRebindOptions{.field_aliases = std::span<const ParamFieldAlias>{aliases}});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ParamRebindErrorCode::kInvalidSnapshot);
  EXPECT_EQ(result.error().detail().token(), "PARAM_REBIND_SNAPSHOT_EMPTY");
}

// REQ: PARAM-027
// REQ: PARAM-029
// Graph: graph["rebind-migration-limit"]
//   └─ op<RebindOperator>("op")
// Passes 17 migration steps (one above the hard cap of 16) to exercise early rejection
// before any catalog or schema lookup is attempted.
TEST(ParameterTools, RejectsMigrationChainAboveHardMaximumBeforeCatalogLookup) {
  Graph graph{"rebind-migration-limit"};
  (void)graph.op<RebindOperator>("op");
  std::array<ParamMigrationStep, 17> migrations{};
  auto result = to_overrides(
      rebind_snapshot(),
      graph,
      ParamRebindOptions{.migration_chain = std::span<const ParamMigrationStep>{migrations}});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ParamRebindErrorCode::kMigrationChainInvalid);
  EXPECT_EQ(result.error().stage(), ParamRebindStage::kMigration);
}

}  // namespace
}  // namespace holoscan::params
