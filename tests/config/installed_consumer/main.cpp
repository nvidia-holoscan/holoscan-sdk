/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <map>
#include <ranges>
#include <string>
#include <utility>

#include <holoscan/config/codec.hpp>
#if defined(HOLOSCAN_TEST_USE_CONFIG_IO)
#include <holoscan/config/io.hpp>
#endif
#if defined(HOLOSCAN_TEST_USE_PARAMS)
#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/rebind.hpp>
#endif
#if defined(HOLOSCAN_TEST_USE_PARAMS_IO)
#include <holoscan/params/persistence.hpp>
#endif
#if defined(HOLOSCAN_TEST_USE_TUNER)
#include <holoscan/experimental/params/web_tuner.hpp>
#endif

#if defined(HOLOSCAN_TEST_USE_PARAMS)
namespace installed_consumer {

struct SchemaParams {
  std::int32_t count{1};
  std::string name{"default"};

  bool operator==(const SchemaParams&) const = default;
};

struct WrongParams {
  std::int32_t count{};
};

class SchemaOperator final : public holoscan::Operator<SchemaParams> {
 public:
  void setup(holoscan::OperatorSpec&) override {}
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

HOLOSCAN_PARAM_SCHEMA(SchemaParams,
                      holoscan::param_schema("holoscan.test.installed.SchemaParams", 1),
                      holoscan::field<&SchemaParams::count>("count"),
                      holoscan::field<&SchemaParams::name>("name"));

HOLOSCAN_PARAM_SCHEMA(WrongParams, holoscan::param_schema("holoscan.test.installed.WrongParams", 1),
                      holoscan::field<&WrongParams::count>("count"));

template <typename Input>
concept CanSetSchemaParams = requires(holoscan::OperatorHandle<SchemaOperator> op, Input input) {
  op.set_params(std::move(input));
};

// REQ: PARAM-009
static_assert(CanSetSchemaParams<holoscan::PreparedParams<SchemaParams>>);
static_assert(!CanSetSchemaParams<holoscan::PreparedParams<WrongParams>>);

}  // namespace installed_consumer

template <>
struct holoscan::config::ConfigCodec<installed_consumer::SchemaParams>
    : holoscan::params::DefaultedParamSchemaCodec<installed_consumer::SchemaParams> {};
#endif

int main() {
  constexpr int kSuccess = 0;
  constexpr int kEncodeFailure = 1;
  constexpr int kDocumentFailure = 2;
  constexpr int kRoundTripFailure = 3;
  constexpr int kApiValidationFailure = 4;
#if defined(HOLOSCAN_TEST_USE_PARAMS)
  // REQ: PARAM-024
  // REQ: PARAM-035
  // REQ: PARAM-038
  const holoscan::params::ParamRebindOptions options{};
  const auto config_error = holoscan::config::ConfigError::make(
      holoscan::config::ConfigErrorCode::kParseError, "invalid configuration");
  const auto tooling_error =
      holoscan::params::from_config_error(config_error, "PARAM_TOOL_CONFIG_INVALID");
  const std::string formatted_error = holoscan::params::format_error(tooling_error);
  auto path = holoscan::OperatorPath::parse("camera");
  if (!path) {
    return kApiValidationFailure;
  }
  const auto scope = holoscan::params::ParamRebindTargetScope::operator_(*path);
  const installed_consumer::SchemaParams schema_params{.count = 3, .name = "camera"};
  auto encoded = holoscan::config::try_encode_config(schema_params);
  if (!encoded) {
    return kEncodeFailure;
  }
  auto decoded =
      holoscan::config::try_decode_config<installed_consumer::SchemaParams>(encoded->view());
  if (!decoded || *decoded != schema_params) {
    return kRoundTripFailure;
  }
  auto document = holoscan::config::Document::try_from_value(*encoded);
  if (!document) {
    return kDocumentFailure;
  }
  auto prepared = holoscan::params::try_decode<installed_consumer::SchemaParams>(
      *document, holoscan::config::Path::root(), holoscan::config::DecodeOptions{});
  if (!prepared) {
    return kRoundTripFailure;
  }
  holoscan::Graph graph{"installed-params"};
  auto op = graph.op<installed_consumer::SchemaOperator>("op");
  op.set_params(std::move(*prepared));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return kApiValidationFailure;
  }
  return options.mode == holoscan::ParamRebindMode::kSparseContributions &&
                 scope.kind() == holoscan::ParamRebindTargetScopeKind::kOperator &&
                 scope.paths().size() == 1U &&
                 std::ranges::equal(scope.paths().front(), path->segments()) &&
                 formatted_error.starts_with("PARAM_TOOL_CONFIG_INVALID")
             ? kSuccess
             : kApiValidationFailure;
#elif defined(HOLOSCAN_TEST_USE_PARAMS_IO)
  const holoscan::params::ParamSaveOptions options{};
  return options.safe_projection == holoscan::params::ParamSafeProjection::kPersistedPublicOnly
             ? kSuccess
             : kApiValidationFailure;
#elif defined(HOLOSCAN_TEST_USE_TUNER)
  const holoscan::experimental::params::WebTunerOptions options{};
  return options.max_clients == 1U && options.max_connections == 4U ? kSuccess
                                                                    : kApiValidationFailure;
#else
  const std::string formatted_error = holoscan::config::format_error(
      holoscan::config::ConfigError::make(holoscan::config::ConfigErrorCode::kParseError));
  if (!formatted_error.starts_with("CONFIG_PARSE_ERROR")) {
    return kApiValidationFailure;
  }
  const std::map<std::string, std::int32_t> source{{"camera", 7}};
  auto encoded = holoscan::config::try_encode_config(source);
  if (!encoded) {
    return kEncodeFailure;
  }
  auto document = holoscan::config::Document::try_from_value(*encoded);
  if (!document) {
    return kDocumentFailure;
  }
  auto enabled_path = holoscan::config::Path::try_parse("/enabled");
  if (!enabled_path) {
    return kApiValidationFailure;
  }
  auto updated = document->try_patched(
      holoscan::config::Edit::add(*enabled_path, holoscan::config::ConfigValue::null()));
  if (!updated) {
    return kApiValidationFailure;
  }
  // REQ: CONFIG-020
  try {
    static_cast<void>(holoscan::config::Path::parse("not-rooted"));
    return kApiValidationFailure;
  } catch (const holoscan::config::ConfigException& exception) {
    if (exception.error().code() != holoscan::config::ConfigErrorCode::kPathSyntax) {
      return kApiValidationFailure;
    }
  }
#if defined(HOLOSCAN_TEST_USE_CONFIG_IO)
  std::string text;
  try {
    text = holoscan::config_io::serialize(*document, holoscan::config_io::Format::kJson);
  } catch (const holoscan::config::ConfigException&) {
    return kRoundTripFailure;
  }
  if (text.find("camera") == std::string::npos) {
    return kRoundTripFailure;
  }
  try {
    static_cast<void>(holoscan::config_io::parse("[", holoscan::config_io::Format::kJson));
    return kApiValidationFailure;
  } catch (const holoscan::config::ConfigException& exception) {
    return exception.error().code() == holoscan::config::ConfigErrorCode::kParseError
               ? kSuccess
               : kRoundTripFailure;
  }
#else
  auto decoded = document->try_get<std::map<std::string, std::int32_t>>("");
  return decoded && *decoded == source ? kSuccess : kRoundTripFailure;
#endif
#endif
}
