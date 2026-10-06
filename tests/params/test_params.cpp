/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
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
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>
#include <holoscan/params/rebind.hpp>

#include "compiler/execution_plan_state.hpp"
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

// Constructs a complete ParamSnapshot for a single operator at `path` under `scope`.
// Fields are sorted by field ID before packaging. Pass include_unknown=true to append
// an unrecognised operator entry, which triggers unknown-operator validation during attachment.
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

[[nodiscard]] ParamSnapshot two_operator_rebind_snapshot(bool valid_second) {
  const ParamSnapshot first = rebind_snapshot();
  if (first.operators().empty())
    return {};
  std::vector<ParamSnapshotOperatorRecord> records;
  records.push_back(first.operators().front());
  auto second = records.front();
  auto second_path = OperatorPath::parse("other");
  if (!second_path)
    return {};
  second.path = std::move(*second_path);
  if (!valid_second) {
    second.fields.front().address.field = 9001U;
    second.fields.front().full_id = {};
  }
  records.push_back(std::move(second));
  return holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      {},
      {},
      {},
      {},
      ParamReadView::kPublished,
      ParamCaptureConsistency::kRequireStopped,
      ParamSnapshotScope::root(),
      ParamUpdateSequence{1U},
      ParamUpdateSequence{1U},
      std::move(records));
}

class SnapshotAttachedRebindChildGraph final : public Graph {
 protected:
  void compose() override {
    (void)op<RebindOperator>("op");
    auto attached = set_params(*this,
                               rebind_snapshot(),
                               ParamRebindTargetScope::whole_graph(),
                               ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
    if (!attached) {
      throw RuntimeError{ErrorCode::kFailure, attached.error().detail().token()};
    }
  }
};

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
// REQ: PARAM-028
// REQ: PARAM-035
// REQ: PARAM-036
// REQ: COMPILE-073
TEST(ParameterTools, SnapshotAttachmentUsesFinalSparseBaseline) {
  Graph graph{"snapshot-attachment-final-baseline"};
  auto op = graph.op<RebindOperator>("op");
  op.set_params(RebindParams{.required = 7, .optional = 8});

  auto attached = set_params(graph, rebind_snapshot());
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
  op.set_params(RebindParams{.required = 11, .optional = 12});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const auto* compiled = static_cast<const RebindParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->required, 42);
  EXPECT_EQ(compiled->optional, 12);
}

// REQ: PARAM-039
// REQ: PARAM-048
TEST(ParameterTools, SnapshotContributionReplacesPreparedFieldProvenance) {
  Graph graph{"snapshot-replaces-prepared-provenance"};
  auto op = graph.op<RebindOperator>("op");
  auto required = config::ConfigValue::try_signed_integer(9);
  auto optional = config::ConfigValue::try_signed_integer(8);
  ASSERT_TRUE(required.has_value());
  ASSERT_TRUE(optional.has_value());
  const std::array entries{config::ConfigObjectEntry{"required", *required},
                           config::ConfigObjectEntry{"optional", *optional}};
  auto root = config::ConfigValue::try_object(entries);
  ASSERT_TRUE(root.has_value());
  auto source = config::Document::try_from_value(std::move(*root));
  ASSERT_TRUE(source.has_value());
  auto prepared = try_decode<RebindParams>(*source, config::Path::root());
  ASSERT_TRUE(prepared.has_value()) << prepared.error().token();
  op.set_params(std::move(*prepared));

  auto attached = set_params(graph, rebind_snapshot());
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* schema = param_schema_descriptor<RebindParams>();
  const ParamFieldId required_id = schema->fields[0].id;
  const ParamFieldId optional_id = schema->fields[1].id;
  const auto required_record =
      std::ranges::find(state->parameter_generation_zero_fields,
                        required_id,
                        &holoscan::detail::ParameterGenerationZeroFieldRecord::field);
  const auto optional_record =
      std::ranges::find(state->parameter_generation_zero_fields,
                        optional_id,
                        &holoscan::detail::ParameterGenerationZeroFieldRecord::field);
  ASSERT_NE(required_record, state->parameter_generation_zero_fields.end());
  ASSERT_NE(optional_record, state->parameter_generation_zero_fields.end());
  EXPECT_TRUE(required_record->transaction_contributed);
  EXPECT_FALSE(required_record->config_source.has_value());
  EXPECT_FALSE(optional_record->transaction_contributed);
  ASSERT_TRUE(optional_record->config_source.has_value());
  EXPECT_EQ(optional_record->config_source->source_path, "/optional");
}

// REQ: PARAM-046
TEST(ParameterTools, SnapshotAttachmentComposesLazyGraphBeforeTargetLookup) {
  RebindChildGraph graph;

  auto attached = set_params(graph, rebind_snapshot());
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-028
// REQ: PARAM-046
TEST(ParameterTools, CompleteSnapshotAttachmentUsesDefaults) {
  Graph graph{"complete-snapshot-attachment"};
  auto op = graph.op<RebindOperator>("op");
  op.set_params(RebindParams{.required = 7, .optional = 8});

  auto attached = set_params(graph,
                             rebind_snapshot(),
                             ParamRebindTargetScope::whole_graph(),
                             ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled = static_cast<const RebindParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->required, 42);
  EXPECT_EQ(compiled->optional, 2);
  const ParamFieldId optional_id = param_schema_descriptor<RebindParams>()->fields[1].id;
  const auto optional_record =
      std::ranges::find(state->parameter_generation_zero_fields,
                        optional_id,
                        &holoscan::detail::ParameterGenerationZeroFieldRecord::field);
  ASSERT_NE(optional_record, state->parameter_generation_zero_fields.end());
  EXPECT_EQ(optional_record->origin, holoscan::detail::ParameterGenerationZeroOrigin::kDefault);
  EXPECT_FALSE(optional_record->config_source.has_value());
}

// REQ: PARAM-046
// REQ: COMPILE-073
TEST(ParameterTools, CompleteSnapshotAttachmentRejectsLaterScopeMember) {
  Graph graph{"complete-snapshot-later-member"};
  (void)graph.op<RebindOperator>("op");
  auto attached = set_params(graph,
                             rebind_snapshot(),
                             ParamRebindTargetScope::whole_graph(),
                             ParamRebindOptions{.mode = ParamRebindMode::kCompleteObject});
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();

  (void)graph.op<RebindOperator>("later");
  const ExecutionPlan plan = compile(graph);
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_REBIND_SCOPE_INCOMPLETE"), std::string::npos)
      << plan.json();
}

// REQ: PARAM-046
TEST(ParameterTools, SparseOperatorSnapshotAllowsLaterOutsideScope) {
  Graph graph{"sparse-snapshot-later-outside-scope"};
  (void)graph.op<RebindOperator>("op");
  auto path = OperatorPath::parse("op");
  ASSERT_TRUE(path.has_value());
  auto attached = set_params(graph, rebind_snapshot(), ParamRebindTargetScope::operator_(*path));
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();

  (void)graph.op<OpaqueRebindOperator>("later");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-046
TEST(ParameterTools, SnapshotAttachmentChecksFinalProfile) {
  Graph graph{"snapshot-attachment-final-profile"};
  (void)graph.op<RebindOperator>("op");
  auto attached = set_params(graph, rebind_snapshot());
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();

  CompileOptions options;
  options.profile = Profile::kResearch;
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_REBIND_PROFILE_MISMATCH"), std::string::npos)
      << plan.json();
}

// REQ: PARAM-046
TEST(ParameterTools, EmbeddedSnapshotAttachmentKeepsChildScope) {
  Graph graph{"embedded-snapshot-attachment"};
  auto embedded = graph.embed<SnapshotAttachedRebindChildGraph>("child");
  ASSERT_TRUE(embedded);
  (void)graph.op<OpaqueRebindOperator>("outside");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-028
// REQ: PARAM-029
// Graph: graph["rebind-no-value-markers"]
//   └─ op<RebindOperator>("op")
TEST(ParameterTools, SnapshotAttachmentPreservesOperatorAlias) {
  Graph graph{"snapshot-attachment-operator-alias"};
  (void)graph.op<RebindOperator>("renamed");
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RebindParams>();
  ASSERT_NE(schema, nullptr);
  auto source_path = OperatorPath::parse("op");
  auto target_path = OperatorPath::parse("renamed");
  auto scope_path = OperatorPath::parse("renamed");
  ASSERT_TRUE(source_path.has_value());
  ASSERT_TRUE(target_path.has_value());
  ASSERT_TRUE(scope_path.has_value());
  const std::array aliases{ParamOperatorAlias{
      .from_path = std::move(*source_path),
      .from_schema_version = schema->version,
      .to_path = std::move(*target_path),
      .to_schema_version = schema->version,
  }};

  auto attached = set_params(
      graph,
      rebind_snapshot(),
      ParamRebindTargetScope::operator_(*scope_path),
      ParamRebindOptions{.operator_aliases = std::span<const ParamOperatorAlias>{aliases}});
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
  ASSERT_EQ(attached->operator_aliases.size(), 1U);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-029
// REQ: PARAM-046
TEST(ParameterTools, SnapshotAttachmentPreservesFieldAlias) {
  Graph graph{"snapshot-attachment-field-alias"};
  (void)graph.op<RebindOperator>("op");
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RebindParams>();
  ASSERT_NE(schema, nullptr);

  constexpr ParamFieldId kLegacyRequiredField = 9001U;
  auto legacy_required = rebind_field("required", 42);
  legacy_required.address.field = kLegacyRequiredField;
  legacy_required.full_id = {};
  auto legacy_path = ParamPath::parse("legacy_required");
  ASSERT_TRUE(legacy_path.has_value());
  legacy_required.path = std::move(*legacy_path);
  const std::array aliases{ParamFieldAlias{
      .from_schema_id = std::string{schema->stable_id},
      .from_schema_version = schema->version,
      .from_field_id = kLegacyRequiredField,
      .to_schema_id = std::string{schema->stable_id},
      .to_schema_version = schema->version,
      .to_field_id = schema->fields.front().id,
  }};

  auto attached =
      set_params(graph,
                 scoped_rebind_snapshot("op", ParamSnapshotScope::root(), false, {legacy_required}),
                 ParamRebindTargetScope::whole_graph(),
                 ParamRebindOptions{.field_aliases = std::span<const ParamFieldAlias>{aliases}});
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
  ASSERT_EQ(attached->field_aliases.size(), 1U);
  ASSERT_EQ(attached->fields.size(), 2U);
  EXPECT_EQ(attached->fields.front().origin, ParamFieldRebindOrigin::kAlias);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-029
// REQ: PARAM-046
TEST(ParameterTools, SnapshotAttachmentRejectsUnavailableMigrationWithoutMutatingGraph) {
  Graph graph{"snapshot-attachment-migration"};
  (void)graph.op<RebindOperator>("op");
  const std::array migrations{ParamMigrationStep{}};

  auto rejected = set_params(
      graph,
      rebind_snapshot(),
      ParamRebindTargetScope::whole_graph(),
      ParamRebindOptions{.migration_chain = std::span<const ParamMigrationStep>{migrations}});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code(), ParamRebindErrorCode::kMigrationRequired);
  EXPECT_EQ(rejected.error().stage(), ParamRebindStage::kMigration);
  EXPECT_EQ(rejected.error().detail().token(), "PARAM_REBIND_MIGRATION_CATALOG_UNAVAILABLE");

  auto attached = set_params(graph, rebind_snapshot());
  ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-046
TEST(ParameterTools, FailedMultiOperatorAttachmentPublishesNothing) {
  Graph graph{"snapshot-attachment-atomic"};
  auto first = graph.op<RebindOperator>("op");
  auto second = graph.op<RebindOperator>("other");
  first.set_params(RebindParams{.required = 10, .optional = 11});
  second.set_params(RebindParams{.required = 20, .optional = 21});

  auto rejected = set_params(graph, two_operator_rebind_snapshot(false));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code(), ParamRebindErrorCode::kUnknownField);

  const ExecutionPlan unchanged = compile(graph);
  ASSERT_TRUE(unchanged.ok()) << unchanged.json();
  const auto* unchanged_state = holoscan::detail::ExecutionPlanAccess::state(unchanged);
  ASSERT_NE(unchanged_state, nullptr);
  ASSERT_EQ(unchanged_state->executable_factories.operators.size(), 2U);
  const auto* unchanged_first = static_cast<const RebindParams*>(
      unchanged_state->executable_factories.operators[0].factory->initial_parameter_value());
  const auto* unchanged_second = static_cast<const RebindParams*>(
      unchanged_state->executable_factories.operators[1].factory->initial_parameter_value());
  ASSERT_NE(unchanged_first, nullptr);
  ASSERT_NE(unchanged_second, nullptr);
  EXPECT_EQ(unchanged_first->required, 10);
  EXPECT_EQ(unchanged_second->required, 20);

  {
    auto attached = set_params(graph, two_operator_rebind_snapshot(true));
    ASSERT_TRUE(attached.has_value()) << attached.error().detail().token();
    ASSERT_FALSE(attached->fields.empty());
  }
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 2U);
  const auto* compiled_first = static_cast<const RebindParams*>(
      state->executable_factories.operators[0].factory->initial_parameter_value());
  const auto* compiled_second = static_cast<const RebindParams*>(
      state->executable_factories.operators[1].factory->initial_parameter_value());
  ASSERT_NE(compiled_first, nullptr);
  ASSERT_NE(compiled_second, nullptr);
  EXPECT_EQ(compiled_first->required, 42);
  EXPECT_EQ(compiled_first->optional, 11);
  EXPECT_EQ(compiled_second->required, 42);
  EXPECT_EQ(compiled_second->optional, 21);
}

// REQ: PARAM-046
TEST(ParameterTools, RejectsEmptyOperatorSetScope) {
  Graph graph{"snapshot-empty-operator-set"};
  (void)graph.op<RebindOperator>("op");

  auto attached = set_params(graph, rebind_snapshot(), ParamRebindTargetScope::operator_set({}));

  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().detail().token(), "PARAM_REBIND_TARGET_SCOPE_EMPTY");
}

// REQ: PARAM-046
TEST(ParameterTools, RejectsDuplicateOperatorSetScope) {
  Graph graph{"snapshot-duplicate-operator-set"};
  (void)graph.op<RebindOperator>("op");
  auto path = OperatorPath::parse("op");
  ASSERT_TRUE(path.has_value());

  auto attached =
      set_params(graph,
                 rebind_snapshot(),
                 ParamRebindTargetScope::operator_set(std::vector<OperatorPath>{*path, *path}));

  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().detail().token(), "PARAM_REBIND_TARGET_SCOPE_DUPLICATE");
}

}  // namespace
}  // namespace holoscan::params
