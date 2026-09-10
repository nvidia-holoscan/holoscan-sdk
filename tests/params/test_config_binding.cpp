/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>

#include "core/parameter_catalog_internal.hpp"
#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {

struct ConfigBindingHistoryTerm {
  std::string input_name;
};

struct ComparableOmittedField {
  std::int32_t value{7};
  bool operator==(const ComparableOmittedField&) const noexcept = default;
};

struct AggregateCodecBindingParams {
  std::vector<std::string> labels{"baseline"};
  ComparableOmittedField untouched;
};

struct ThrowingAssignmentValue {
  std::int32_t number{};

  ThrowingAssignmentValue() = default;
  explicit ThrowingAssignmentValue(std::int32_t value) : number(value) {}
  ThrowingAssignmentValue(const ThrowingAssignmentValue&) = default;
  ThrowingAssignmentValue(ThrowingAssignmentValue&&) = default;
  ThrowingAssignmentValue& operator=(const ThrowingAssignmentValue&) = default;
  ThrowingAssignmentValue& operator=(ThrowingAssignmentValue&& other) {
    if (other.number == 42) {
      throw std::bad_alloc{};
    }
    number = other.number;
    return *this;
  }
  bool operator==(const ThrowingAssignmentValue&) const = default;
};

struct ThrowingAssignmentParams {
  ThrowingAssignmentValue value;
};

HOLOSCAN_PARAM_SCHEMA(AggregateCodecBindingParams,
                      param_schema("holoscan.test.AggregateCodecBindingParams", 1),
                      field<&AggregateCodecBindingParams::labels>("labels"),
                      field<&AggregateCodecBindingParams::untouched>("untouched"));

HOLOSCAN_PARAM_SCHEMA(ThrowingAssignmentParams,
                      param_schema("holoscan.test.ThrowingAssignmentParams", 1),
                      field<&ThrowingAssignmentParams::value>("value"));

}  // namespace holoscan::params

template <>
struct holoscan::config::ConfigCodec<holoscan::params::ConfigBindingHistoryTerm> {
  static ConfigResult<holoscan::params::ConfigBindingHistoryTerm> decode(
      ConfigView view, const DecodeOptions& options, DecodeContext& context) {
    auto input_name = ConfigCodec<std::string>::decode(std::move(view), options, context);
    if (!input_name) {
      return tl::make_unexpected(std::move(input_name.error()));
    }
    return holoscan::params::ConfigBindingHistoryTerm{.input_name = std::move(*input_name)};
  }

  static ConfigResult<ConfigValue> encode(const holoscan::params::ConfigBindingHistoryTerm& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ConfigCodec<std::string>::encode(value.input_name, options, context);
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::params::AggregateCodecBindingParams> {
  inline static std::uint32_t decode_calls{};
  inline static bool change_untouched{};

  static ConfigResult<holoscan::params::AggregateCodecBindingParams> decode(
      ConfigView view, const DecodeOptions& options, DecodeContext& context) {
    ++decode_calls;
    ObjectDecoder object{std::move(view), options, context};
    auto labels = object.required<std::vector<std::string>>("labels");
    if (!labels) {
      return tl::make_unexpected(std::move(labels.error()));
    }
    auto untouched = object.required<std::int32_t>("untouched");
    if (!untouched) {
      return tl::make_unexpected(std::move(untouched.error()));
    }
    auto finished = object.finish();
    if (!finished) {
      return tl::make_unexpected(std::move(finished.error()));
    }
    if (change_untouched) {
      ++*untouched;
    }
    return holoscan::params::AggregateCodecBindingParams{.labels = std::move(*labels),
                                                         .untouched = {.value = *untouched}};
  }

  static ConfigResult<ConfigValue> encode(
      const holoscan::params::AggregateCodecBindingParams& value, const EncodeOptions& options,
      EncodeContext& context) {
    auto labels = context.encode_child<std::vector<std::string>>("labels", value.labels, options);
    if (!labels) {
      return tl::make_unexpected(std::move(labels.error()));
    }
    auto untouched =
        context.encode_child<std::int32_t>("untouched", value.untouched.value, options);
    if (!untouched) {
      return tl::make_unexpected(std::move(untouched.error()));
    }
    const std::array entries{ConfigObjectEntry{"labels", *labels},
                             ConfigObjectEntry{"untouched", *untouched}};
    return ConfigValue::object(entries);
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::params::ThrowingAssignmentValue> {
  static ConfigResult<holoscan::params::ThrowingAssignmentValue> decode(
      ConfigView view, const DecodeOptions& options, DecodeContext& context) {
    auto number = ConfigCodec<std::int32_t>::decode(std::move(view), options, context);
    if (!number) {
      return tl::make_unexpected(std::move(number.error()));
    }
    return holoscan::params::ThrowingAssignmentValue{*number};
  }

  static ConfigResult<ConfigValue> encode(const holoscan::params::ThrowingAssignmentValue& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ConfigCodec<std::int32_t>::encode(value.number, options, context);
  }
};

namespace holoscan::params {
namespace {

// Params schema for config-binding tests: "required" carries a 0–100 range constraint and
// an alias "legacy_required"; "optional" is plain. HandleBindingAppliesSparsePatchOverTyped-
// Baseline is the representative test demonstrating sparse-patch application.
struct ConfigBindingParams {
  std::int32_t required{1};
  std::int32_t optional{2};
};

HOLOSCAN_PARAM_SCHEMA(ConfigBindingParams, param_schema("holoscan.test.ConfigBindingParams", 1),
                      field<&ConfigBindingParams::required>("required")
                          .runtime_mutable()
                          .external_tuning()
                          .range(0, 100)
                          .alias("legacy_required"),
                      field<&ConfigBindingParams::optional>("optional"));

// No-op operator carrying ConfigBindingParams. Used to register operators in binding test graphs
// without any real compute logic.
class ConfigBindingOperator final : public Operator<ConfigBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct ContainerBindingParams {
  std::vector<std::string> labels{"baseline"};
  std::map<std::string, double> gains{{"retained", 1.0}};
  std::vector<ConfigBindingHistoryTerm> history{{.input_name = "baseline"}};
  std::int32_t untouched{7};
};

HOLOSCAN_PARAM_SCHEMA(ContainerBindingParams,
                      param_schema("holoscan.test.ContainerBindingParams", 1),
                      field<&ContainerBindingParams::labels>("labels"),
                      field<&ContainerBindingParams::gains>("gains"),
                      field<&ContainerBindingParams::history>("history"),
                      field<&ContainerBindingParams::untouched>("untouched"));

class ContainerBindingOperator final : public Operator<ContainerBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct SourceRetargetBindingParams {
  std::array<std::int32_t, 2> values{{1, 2}};
  std::int32_t marker{3};
};

struct TargetRetargetBindingParams {
  std::int32_t marker{4};
  std::array<std::int32_t, 2> values{{5, 6}};
};

// The shared schema identity makes the local C++ type token the only discriminator.
HOLOSCAN_PARAM_SCHEMA(SourceRetargetBindingParams,
                      param_schema("holoscan.test.RetargetBindingParams", 1),
                      field<&SourceRetargetBindingParams::values>("values"),
                      field<&SourceRetargetBindingParams::marker>("marker"));

HOLOSCAN_PARAM_SCHEMA(TargetRetargetBindingParams,
                      param_schema("holoscan.test.RetargetBindingParams", 1),
                      field<&TargetRetargetBindingParams::values>("values"),
                      field<&TargetRetargetBindingParams::marker>("marker"));

class SourceRetargetBindingOperator final : public Operator<SourceRetargetBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class TargetRetargetBindingOperator final : public Operator<TargetRetargetBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class AggregateCodecBindingOperator final : public Operator<AggregateCodecBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct UnsupportedBindingParams {
  struct Value {
    std::int32_t number{};
  } value;
};

HOLOSCAN_PARAM_SCHEMA(UnsupportedBindingParams,
                      param_schema("holoscan.test.UnsupportedBindingParams", 1),
                      field<&UnsupportedBindingParams::value>("value"));

class UnsupportedBindingOperator final : public Operator<UnsupportedBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class ThrowingAssignmentOperator final : public Operator<ThrowingAssignmentParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Test DSL helpers — thin assertion-wrapped constructors for config values, documents, and paths.
// value_or_null: unwraps a ConfigResult and asserts success; returns null on failure.
// object/array/document/path: build their respective types with inline assertion checks.
[[nodiscard]] config::ConfigValue value_or_null(config::ConfigResult<config::ConfigValue> result) {
  EXPECT_TRUE(result.has_value());
  return result ? std::move(*result) : config::ConfigValue::null();
}

[[nodiscard]] config::ConfigValue object(
    std::initializer_list<std::pair<std::string_view, config::ConfigValue>> values) {
  std::vector<std::string> keys;
  std::vector<config::ConfigValue> owned_values;
  keys.reserve(values.size());
  owned_values.reserve(values.size());
  for (const auto& [key, value] : values) {
    keys.emplace_back(key);
    owned_values.push_back(value);
  }
  std::vector<config::ConfigObjectEntry> entries;
  entries.reserve(values.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    entries.push_back(config::ConfigObjectEntry{.key = keys[index], .value = owned_values[index]});
  }
  return value_or_null(config::ConfigValue::object(entries));
}

[[nodiscard]] config::ConfigValue array(std::initializer_list<config::ConfigValue> values) {
  const std::vector<config::ConfigValue> owned{values};
  return value_or_null(config::ConfigValue::array(owned));
}

[[nodiscard]] config::Document document(config::ConfigValue root) {
  auto result = config::Document::from_value(std::move(root));
  EXPECT_TRUE(result.has_value());
  return result ? std::move(*result)
                : config::Document::from_value(config::ConfigValue::null()).value();
}

[[nodiscard]] config::Path path(std::string_view text) {
  auto result = config::Path::parse(text);
  EXPECT_TRUE(result.has_value());
  return result ? std::move(*result) : config::Path::root();
}

// REQ: PARAM-033
// Graph: graph["config-binding"]
//   └─ op<ConfigBindingOperator>("op")
TEST(ParameterConfigBinding, HandleBindingAppliesSparsePatchOverTypedBaseline) {
  // Key setup: baseline has required=7, optional=9; JSON document patches only "required"→42.
  Graph graph{"config-binding"};
  auto op = graph.op<ConfigBindingOperator>("op");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document source = document(object({
      {"managed", object({{"required", value_or_null(config::ConfigValue::signed_integer(42))}})},
  }));
  const std::array bindings{bind(op, path("/managed"))};

  auto decoded = decode_overrides(source, bindings);

  // Key assertion: required is patched to 42; optional remains at baseline value 9.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*decoded);
  ASSERT_EQ(state.operators.size(), 1U);
  const auto& target = state.operators.front();
  ASSERT_EQ(target.operator_path, "op");
  ASSERT_EQ(target.contributions.size(), 1U);
  const auto* value = static_cast<const ConfigBindingParams*>(target.resolved_value.get());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->required, 42);
  EXPECT_EQ(value->optional, 9);
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingDecodesVectorFieldWithoutAggregateCodec) {
  Graph graph{"config-binding-vector"};
  auto op = graph.op<ContainerBindingOperator>("op");
  op.set_params(ContainerBindingParams{
      .labels = {"baseline"}, .gains = {{"retained", 3.0}}, .untouched = 11});
  const config::Document source = document(object({
      {"managed",
       object({{"labels",
                array({value_or_null(config::ConfigValue::string("left")),
                       value_or_null(config::ConfigValue::string("right"))})}})},
  }));

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*decoded);
  const auto* value =
      static_cast<const ContainerBindingParams*>(state.operators.front().resolved_value.get());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->labels, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(value->gains, (std::map<std::string, double>{{"retained", 3.0}}));
  EXPECT_EQ(value->untouched, 11);
  EXPECT_NE(state.operators.front().baseline_value_digest,
            state.operators.front().resolved_value_digest);
  CompileOptions options;
  options.parameter_overrides = *decoded;
  const ExecutionPlan plan = compile(graph, std::move(options));
  EXPECT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingReplacesMapFieldWithoutAggregateCodec) {
  Graph graph{"config-binding-map"};
  auto op = graph.op<ContainerBindingOperator>("op");
  op.set_params(ContainerBindingParams{.gains = {{"retained", 3.0}, {"removed", 4.0}}});
  const config::Document source = document(object({
      {"managed",
       object({{"gains",
                object({{"replacement", value_or_null(config::ConfigValue::floating(2.5))}})}})},
  }));

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*decoded);
  const auto* value =
      static_cast<const ContainerBindingParams*>(state.operators.front().resolved_value.get());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->gains, (std::map<std::string, double>{{"replacement", 2.5}}));
  EXPECT_EQ(value->labels, (std::vector<std::string>{"baseline"}));
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingComposesVectorWithVisibleElementCodec) {
  Graph graph{"config-binding-composed-vector"};
  auto op = graph.op<ContainerBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"history",
                array({value_or_null(config::ConfigValue::string("state")),
                       value_or_null(config::ConfigValue::string("action"))})}})},
  }));

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*decoded);
  const auto* value =
      static_cast<const ContainerBindingParams*>(state.operators.front().resolved_value.get());
  ASSERT_NE(value, nullptr);
  ASSERT_EQ(value->history.size(), 2U);
  EXPECT_EQ(value->history[0].input_name, "state");
  EXPECT_EQ(value->history[1].input_name, "action");
  CompileOptions options;
  options.parameter_overrides = *decoded;
  const ExecutionPlan plan = compile(graph, std::move(options));
  EXPECT_TRUE(plan.ok()) << plan.json();
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, AggregateCodecRemainsTheNonScalarFallbackWhenAvailable) {
  Graph graph{"config-binding-aggregate-fallback"};
  auto op = graph.op<AggregateCodecBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"labels", array({value_or_null(config::ConfigValue::string("configured"))})}})},
  }));
  config::ConfigCodec<AggregateCodecBindingParams>::decode_calls = 0;
  config::ConfigCodec<AggregateCodecBindingParams>::change_untouched = false;

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  EXPECT_EQ(config::ConfigCodec<AggregateCodecBindingParams>::decode_calls, 1U);
  CompileOptions options;
  options.parameter_overrides = *decoded;
  const ExecutionPlan preserved_plan = compile(graph, std::move(options));
  EXPECT_TRUE(preserved_plan.ok()) << preserved_plan.json();

  config::ConfigCodec<AggregateCodecBindingParams>::change_untouched = true;
  auto changed = decode_overrides(source, {bind(op, path("/managed"))});
  config::ConfigCodec<AggregateCodecBindingParams>::change_untouched = false;

  ASSERT_TRUE(changed.has_value()) << changed.error().token();
  EXPECT_EQ(config::ConfigCodec<AggregateCodecBindingParams>::decode_calls, 2U);
  CompileOptions changed_options;
  changed_options.parameter_overrides = *changed;
  const ExecutionPlan changed_plan = compile(graph, std::move(changed_options));
  EXPECT_FALSE(changed_plan.ok());
  EXPECT_TRUE(std::ranges::any_of(changed_plan.diagnostics(), [](const Diagnostic& diagnostic) {
    return diagnostic.token == "PARAMETER_OVERRIDE_UNDECLARED_CHANGE";
  }));
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, InvalidContainerFieldReturnsNoOverrideSet) {
  Graph graph{"config-binding-invalid-container"};
  auto op = graph.op<ContainerBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"labels", array({value_or_null(config::ConfigValue::signed_integer(1))})}})},
  }));

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_DECODE_FAILED");
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingReportsUnavailableFieldCodec) {
  Graph graph{"config-binding-unsupported-field"};
  auto op = graph.op<UnsupportedBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"value",
                object({{"number", value_or_null(config::ConfigValue::signed_integer(1))}})}})},
  }));

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE");
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingRejectsRoutesCapturedForAnotherSchema) {
  Graph graph{"config-binding-schema-mismatch"};
  auto source_op = graph.op<ThrowingAssignmentOperator>("source");
  graph.op<UnsupportedBindingOperator>("target");
  auto binding = bind(source_op, path("/managed"));
  auto target_path = OperatorPath::parse("target");
  ASSERT_TRUE(target_path.has_value());
  binding.operator_path = std::move(*target_path);
  const config::Document source = document(object({
      {"managed", object({{"value", value_or_null(config::ConfigValue::signed_integer(7))}})},
  }));

  auto decoded = decode_overrides(source, {binding});

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE");
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingRejectsRoutesCapturedForAnotherLocalType) {
  const auto* const source_schema = param_schema_descriptor<SourceRetargetBindingParams>();
  const auto* const target_schema = param_schema_descriptor<TargetRetargetBindingParams>();
  ASSERT_NE(source_schema, nullptr);
  ASSERT_NE(target_schema, nullptr);
  ASSERT_EQ(source_schema->structural_hash, target_schema->structural_hash);

  Graph graph{"config-binding-local-type-mismatch"};
  auto source_op = graph.op<SourceRetargetBindingOperator>("source");
  graph.op<TargetRetargetBindingOperator>("target");
  auto binding = bind(source_op, path("/managed"));
  auto target_path = OperatorPath::parse("target");
  ASSERT_TRUE(target_path.has_value());
  binding.operator_path = std::move(*target_path);
  const config::Document source = document(object({
      {"managed",
       object({{"values",
                array({value_or_null(config::ConfigValue::signed_integer(7)),
                       value_or_null(config::ConfigValue::signed_integer(8))})}})},
  }));

  auto decoded = decode_overrides(source, {binding});

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE");
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, FieldAssignmentAllocationFailureReportsResourceExhaustion) {
  Graph graph{"config-binding-assignment-allocation"};
  auto op = graph.op<ThrowingAssignmentOperator>("op");
  const config::Document source = document(object({
      {"managed", object({{"value", value_or_null(config::ConfigValue::signed_integer(42))}})},
  }));

  auto decoded = decode_overrides(source, {bind(op, path("/managed"))});

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().code(), ParamToolErrorCode::kResourceExhausted);
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_ALLOC_FAILED");
}

// REQ: PARAM-033
// Graph: graph["config-binding-duplicates"]
//   └─ op<ConfigBindingOperator>("op")
// JSON document supplies both "required" (canonical) and "legacy_required" (alias) simultaneously.
TEST(ParameterConfigBinding, CanonicalAndAliasContributionRejectAtomically) {
  Graph graph{"config-binding-duplicates"};
  auto op = graph.op<ConfigBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::signed_integer(42))},
               {"legacy_required", value_or_null(config::ConfigValue::signed_integer(41))}})},
  }));
  const std::array bindings{bind(op, path("/managed"))};

  auto decoded = decode_overrides(source, bindings);

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_DUPLICATE");
}

// REQ: PARAM-033
// Graph: graph["dynamic-config-binding"]
//   └─ op<ConfigBindingOperator>("camera/front")  (OperatorPath-encoded as "camera%2Ffront")
// JSON Pointer "/profiles/camera~1front/1" uses RFC 6901 escaping for the slash in the key name.
TEST(ParameterConfigBinding, DynamicBindingKeepsGraphAndJsonPointerGrammarsSeparate) {
  // Key setup: operator path uses percent-encoding ("camera%2Ffront"); the JSON Pointer
  // uses tilde-encoding ("camera~1front") — the two grammars must be kept separate.
  Graph graph{"dynamic-config-binding"};
  auto op = graph.op<ConfigBindingOperator>("camera/front");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document source = document(object({
      {"profiles",
       object({{"camera/front",
                array({config::ConfigValue::null(),
                       object({{"required",
                                value_or_null(config::ConfigValue::signed_integer(43))}})})}})},
  }));
  auto operator_path = OperatorPath::parse("camera%2Ffront");
  ASSERT_TRUE(operator_path.has_value());
  const std::array bindings{
      ParamConfigBinding{std::move(*operator_path), path("/profiles/camera~1front/1")}};

  auto decoded = decode_overrides(source, graph, bindings);

  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& state = holoscan::detail::ParamOverrideSetAccess::state(*decoded);
  ASSERT_EQ(state.operators.size(), 1U);
  EXPECT_EQ(state.operators.front().operator_path, "camera%2Ffront");
}

// REQ: PARAM-033
// Graph: graph["multi-config-binding"]
//   ├─ op<ConfigBindingOperator>("first")
//   └─ op<ConfigBindingOperator>("second")
// "first" binding succeeds; "second" references unknown field "unknown" → whole decode fails.
TEST(ParameterConfigBinding, MultiOperatorDecodeReturnsNoPartialSet) {
  Graph graph{"multi-config-binding"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::signed_integer(20))}})},
      {"second", object({{"unknown", value_or_null(config::ConfigValue::signed_integer(30))}})},
  }));
  const std::array bindings{bind(first, path("/first")), bind(second, path("/second"))};

  auto decoded = decode_overrides(source, bindings);

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_UNKNOWN");
}

// REQ: PARAM-033
// Graph: graph["config-binding-limits"]
//   ├─ op<ConfigBindingOperator>("first")
//   └─ op<ConfigBindingOperator>("second")
// Two decode limit scenarios: max_fields=0 (invalid options) vs max_bindings=1 (exhausted).
TEST(ParameterConfigBinding, DecodeLimitsDistinguishInvalidOptionsFromCapacityPressure) {
  // Key setup: invalid scenario uses max_fields=0 (semantically nonsensical limit).
  //            constrained scenario uses max_bindings=1 with two bindings in the call.
  Graph graph{"config-binding-limits"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::signed_integer(20))}})},
      {"second", object({{"required", value_or_null(config::ConfigValue::signed_integer(30))}})},
  }));

  ParamConfigDecodeOptions invalid;
  invalid.max_fields = 0U;
  auto invalid_result = decode_overrides(source, {bind(first, path("/first"))}, invalid);
  ASSERT_FALSE(invalid_result.has_value());
  EXPECT_EQ(invalid_result.error().code(), ParamToolErrorCode::kInvalidArgument);
  EXPECT_EQ(invalid_result.error().token(), "PARAM_CONFIG_OPTIONS_INVALID");

  ParamConfigDecodeOptions constrained;
  constrained.max_bindings = 1U;
  auto exhausted = decode_overrides(
      source, {bind(first, path("/first")), bind(second, path("/second"))}, constrained);
  ASSERT_FALSE(exhausted.has_value());
  EXPECT_EQ(exhausted.error().code(), ParamToolErrorCode::kResourceExhausted);
  EXPECT_EQ(exhausted.error().token(), "PARAM_CONFIG_BINDING_CAPACITY");
}

// Builds a synthetic ParameterCatalogView for a single "required" field on operator "op".
// Wires external_access=kTuning and range constraints from the schema descriptor so that
// RuntimeDecodeUsesRestrictedCatalogAndConstraints can exercise out-of-range rejection (>100).
[[nodiscard]] ParameterCatalogView runtime_catalog() {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<ConfigBindingParams>();
  const ParamFieldDescriptor& descriptor = schema->fields.front();
  auto operator_path = OperatorPath::parse("op");
  auto field_path = ParamPath::parse("required");
  EXPECT_TRUE(operator_path.has_value());
  EXPECT_TRUE(field_path.has_value());
  const OperatorRef target{.id = 1U, .plan_generation = 2U, .run_generation = 3U};
  OperatorParamRecord op{
      .operator_instance = target,
      .path = operator_path ? std::move(*operator_path) : OperatorPath{},
      .effective_profile = Profile::kStandard,
      .schema_availability = ParamSchemaAvailability::kDeclared,
      .schema_disposition = ParameterSchemaAdmissionDisposition::kCompleteProven,
      .materialization = ParameterMaterializationStrategy::kFixedObject,
      .schema_id = std::string{schema->stable_id},
      .schema_version = schema->version,
      .structural_hash = schema->structural_hash,
      .contract_hash = schema->contract_hash,
      .metadata_hash = schema->metadata_hash,
      .schema_coverage = schema->coverage,
      .schema_proof = schema->proof,
  };
  ParamFieldCatalogRecord field{
      .instance = 1U,
      .address = ParamAddress{.operator_instance = target, .field = descriptor.id},
      .full_id = descriptor.full_id,
      .path = field_path ? std::move(*field_path) : ParamPath{},
      .logical_type = descriptor.logical_type,
      .type_shape = descriptor.type_shape,
      .mutability = ParamMutability::kRuntime,
      .persistence = descriptor.persistence,
      .visibility = ParamVisibility::kPublic,
      .external_access = ParamExternalAccess::kTuning,
      .sensitivity = ParamSensitivity::kNormal,
      .constraints = descriptor.constraints,
      .value_route = ParamValueRouteDisposition::kPortableScalar,
      .materialization = ParameterMaterializationStrategy::kFixedObject,
      .projection = ParamProjectionStatus::kIncluded,
      .update_disposition = ParameterFieldUpdateDisposition::kAdmitted,
      .adapter_eligible = true,
  };
  ParamCatalogDigest digest{};
  digest.value.front() = std::byte{1U};
  return holoscan::detail::ParameterCatalogAccess::make(
      digest,
      std::vector<OperatorParamRecord>{std::move(op)},
      std::vector<ParamFieldCatalogRecord>{std::move(field)});
}

// REQ: PARAM-033
// No Graph required: operator is looked up by path from the pre-built runtime_catalog().
// Two sub-cases: valid value (42, within 0–100) accepted; out-of-range value (101) rejected.
TEST(ParameterConfigBinding, RuntimeDecodeUsesRestrictedCatalogAndConstraints) {
  const auto catalog = runtime_catalog();
  const config::Document source = document(object({
      {"values", object({{"required", value_or_null(config::ConfigValue::signed_integer(42))}})},
      {"generation", value_or_null(config::ConfigValue::unsigned_integer(7U))},
  }));
  auto operator_path = OperatorPath::parse("op");
  ASSERT_TRUE(operator_path.has_value());
  const RuntimeParamConfigBinding binding{
      .operator_path = std::move(*operator_path),
      .values_path = path("/values"),
      .expected_generation_path = path("/generation"),
  };

  // Key assertion: accepted patch carries expected_generation=7 and value=42.
  auto decoded = decode_runtime_patches(source, catalog, binding);

  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  ASSERT_EQ(decoded->size(), 1U);
  EXPECT_EQ(decoded->front().expected_generation, 7U);
  auto value = decoded->front().value.as<std::int32_t>();
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, 42);

  const config::Document invalid = document(object({
      {"values", object({{"required", value_or_null(config::ConfigValue::signed_integer(101))}})},
      {"generation", value_or_null(config::ConfigValue::unsigned_integer(7U))},
  }));
  auto rejected = decode_runtime_patches(invalid, catalog, binding);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_FIELD_CONSTRAINT");
}

// REQ: PARAM-033
// Single operator "op" built inline (no Graph), with one kIncluded field (required=42) and
// one kExcludedTransient field (optional). Verifies to_config_edit emits a merge-object
// containing only the Included field, with the Excluded field absent.
TEST(ParameterConfigBinding, SnapshotProjectionProducesIncludedFieldMergeOnly) {
  auto operator_path = OperatorPath::parse("op");
  auto required_path = ParamPath::parse("required");
  auto optional_path = ParamPath::parse("optional");
  ASSERT_TRUE(operator_path.has_value());
  ASSERT_TRUE(required_path.has_value());
  ASSERT_TRUE(optional_path.has_value());
  std::vector<ParamSnapshotFieldRecord> fields;
  fields.push_back(ParamSnapshotFieldRecord{.path = std::move(*required_path),
                                            .projection = ParamProjectionStatus::kIncluded,
                                            .value = ParamValue::from(std::int32_t{42})});
  fields.push_back(ParamSnapshotFieldRecord{
      .path = std::move(*optional_path),
      .projection = ParamProjectionStatus::kExcludedTransient,
  });
  ParamSnapshotOperatorRecord op{
      .path = std::move(*operator_path),
      .fields = std::move(fields),
  };
  const ParamSnapshot snapshot = holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      {},
      {},
      {},
      {},
      ParamReadView::kPublished,
      ParamCaptureConsistency::kRequireStopped,
      ParamSnapshotScope::root(),
      std::nullopt,
      std::nullopt,
      std::vector<ParamSnapshotOperatorRecord>{std::move(op)});

  auto edit = to_config_edit(snapshot, path("/managed"));

  ASSERT_TRUE(edit.has_value()) << edit.error().token();
  EXPECT_EQ(edit->kind(), config::EditKind::kMergeObject);
  auto required = edit->value().view().member("required");
  ASSERT_TRUE(required.has_value());
  auto decoded = required->as_signed_integer();
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(*decoded, 42);
  auto optional = edit->value().view().find_member("optional");
  ASSERT_TRUE(optional.has_value());
  EXPECT_FALSE(optional->has_value());
}

}  // namespace
}  // namespace holoscan::params
