/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <core/operator_parameter_binding.hpp>
#include <core/operator_test_binding.hpp>

#if defined(HOLOSCAN_PARAMETER_DSO_PARAMS_TESTS)
#include <array>
#include <ranges>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>

#include <compiler/execution_plan_state.hpp>
#include <core/parameter_input_internal.hpp>
#endif

#include "parameter_dso_test_types.hpp"

namespace holoscan::test {
namespace {

// Custom deleter so the DSO-allocated operator is freed via its own module's destructor,
// not the host module's delete — critical for correct cross-DSO memory ownership.
struct DSOOperatorDeleter {
  void operator()(DSOParameterOperator* op) const noexcept { destroy_dso_parameter_operator(op); }
};

#if defined(HOLOSCAN_PARAMETER_DSO_PARAMS_TESTS)
struct DSOGraphDeleter {
  void operator()(DSOParameterGraph* graph) const noexcept { destroy_dso_parameter_graph(graph); }
};

class WrongDSOParameterOperator final : public Operator<DSOParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Authors the same exported projector from the host module.
// ProjectorIdentityAcceptsOneExportedFunctionAcrossModules pairs it with DSOProjectedOperator.
class HostProjectedOperator final : public Operator<DSOProjectorParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_projected_byte_span<&dso_shared_byte_span_projector>(
                "holoscan.test.shared-dso-projector.v1", param_ref<&DSOProjectorParams::bytes>()));
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

[[nodiscard]] expected<std::uint64_t, Error> alternate_dso_byte_span_projector(
    const DSOProjectorParams& params) noexcept {
  return static_cast<std::uint64_t>(params.bytes) + 1U;
}

// Deliberately reuses the DSO projector's semantic ID with a different function.
// ProjectorIdentityRejectsDifferentFunctionsAcrossModules verifies collision rejection.
class AlternateHostProjectedOperator final : public Operator<DSOProjectorParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_projected_byte_span<&alternate_dso_byte_span_projector>(
                "holoscan.test.shared-dso-projector.v1", param_ref<&DSOProjectorParams::bytes>()));
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

[[nodiscard]] config::Document config_document(std::uint32_t value) {
  auto configured = config::ConfigValue::try_unsigned_integer(value);
  EXPECT_TRUE(configured.has_value());
  const std::array fields{config::ConfigObjectEntry{
      "value", configured ? std::move(*configured) : config::ConfigValue::null()}};
  auto object = config::ConfigValue::try_object(fields);
  EXPECT_TRUE(object.has_value());
  const std::array root{config::ConfigObjectEntry{
      "managed", object ? std::move(*object) : config::ConfigValue::null()}};
  auto value_root = config::ConfigValue::try_object(root);
  EXPECT_TRUE(value_root.has_value());
  auto document = config::Document::try_from_value(value_root ? std::move(*value_root)
                                                              : config::ConfigValue::null());
  EXPECT_TRUE(document.has_value());
  return document ? std::move(*document)
                  : config::Document::try_from_value(config::ConfigValue::null()).value();
}

[[nodiscard]] const DSOParams* compiled_params(const ExecutionPlan& plan) {
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  if (state == nullptr) {
    return nullptr;
  }
  const auto vertex = std::ranges::find(state->graph.vertices(), "op", &VertexRecord::name);
  if (vertex == state->graph.vertices().end()) {
    return nullptr;
  }
  const auto factory = std::ranges::find(state->executable_factories.operators,
                                         vertex->id,
                                         &detail::ExecutableOperatorFactoryRecord::vertex);
  return factory == state->executable_factories.operators.end() || !factory->factory
             ? nullptr
             : static_cast<const DSOParams*>(factory->factory->initial_parameter_value());
}
#endif

// REQ: PARAM-013
// When an operator is defined in a DSO, OperatorParameterBinding must resolve the
// type and schema descriptors from the defining module, not the host's local registry.
// This guards against ODR confusion when two modules define the same struct name.
TEST(ParameterDSO, BindingUsesOperatorDefiningModuleIdentity) {
  // Key setup: operator allocated in the DSO; host has its own copy of DSOParams
  // with different local_type_token / local_field_type_token values.
  std::unique_ptr<DSOParameterOperator, DSOOperatorDeleter> op{create_dso_parameter_operator()};
  ASSERT_NE(op, nullptr);

  const auto& host_type = param_type_descriptor<DSOParams>();
  const auto* const host_schema = param_schema_descriptor<DSOParams>();
  ASSERT_NE(host_schema, nullptr);
  EXPECT_NE(host_type.local_type_token, dso_aggregate_type_token());
  EXPECT_NE(host_schema->fields.front().local_operations.local_field_type_token,
            dso_field_type_token());

  // Key assertion: defining_type/defining_schema on the DSO-allocated operator return
  // the DSO's local tokens, which differ from the host-side tokens observed above.
  const auto& bound_type = detail::OperatorParameterBinding::defining_type(*op);
  const auto* const bound_schema = detail::OperatorParameterBinding::defining_schema(*op);
  ASSERT_NE(bound_schema, nullptr);
  EXPECT_EQ(bound_type.local_type_token, dso_aggregate_type_token());
  EXPECT_EQ(bound_schema->fields.front().local_operations.local_field_type_token,
            dso_field_type_token());

  detail::SyntheticOperatorBinding binding;
  ASSERT_TRUE(binding.declare_ports(*op));
  ASSERT_TRUE(binding.invoke_once(*op));
  ASSERT_NE(op->observed_params(), nullptr);
  EXPECT_EQ(op->observed_params(), &op->params());
  EXPECT_EQ(op->observed_params()->value, DSOFieldValue::kDefault);
}

#if defined(HOLOSCAN_PARAMETER_DSO_PARAMS_TESTS)
// REQ: PARAM-013
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterDSO, GraphHandleBindingUsesDefiningModuleIdentity) {
  std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> graph{create_dso_parameter_graph()};
  ASSERT_NE(graph, nullptr);
  auto handle = dso_parameter_operator_handle(graph.get());
  EXPECT_TRUE(handle.graph_valid());
  EXPECT_NE(param_type_descriptor<DSOParams>().local_type_token, dso_aggregate_type_token());
  EXPECT_NE(
      param_schema_descriptor<DSOParams>()->fields.front().local_operations.local_field_type_token,
      dso_field_type_token());

  const config::Document source = config_document(73U);
  auto attached = params::set_params(*graph, source, {params::bind(handle, "/managed")});

  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  const ExecutionPlan plan = compile(*graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const DSOParams* const resolved = compiled_params(plan);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->value, DSOFieldValue::kConfigured);
}

// REQ: PARAM-013
// REQ: PARAM-048
TEST(ParameterDSO, OpaqueParamsAssignmentRejectsCrossModuleTokenWithoutSchema) {
  const std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> graph{create_dso_parameter_graph()};
  ASSERT_NE(graph, nullptr);
  auto handle = dso_opaque_parameter_operator_handle(graph.get());
  EXPECT_NE(param_type_descriptor<DSOOpaqueParams>().local_type_token,
            dso_opaque_aggregate_type_token());
  EXPECT_EQ(param_schema_descriptor<DSOOpaqueParams>(), nullptr);

  try {
    handle.set_params(DSOOpaqueParams{.threshold = 17U});
    FAIL() << "expected schema-less cross-module type rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(make_error(error).message, "GRAPH_PARAMETER_TYPE_MISMATCH");
  }
}

// REQ: PARAM-013
// REQ: PARAM-048
TEST(ParameterDSO, GraphHandleBindingRejectsContractOnlySchemaMismatch) {
  std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> graph{create_dso_parameter_graph()};
  ASSERT_NE(graph, nullptr);
  auto handle = dso_contract_parameter_operator_handle(graph.get());
  auto target = detail::ParameterConfigBindingAccess::target(handle);
  ASSERT_TRUE(target.has_value()) << target.error().message;
  const ParamSchemaDescriptor* const host_schema = param_schema_descriptor<DSOContractParams>();
  ASSERT_NE(host_schema, nullptr);
  ASSERT_NE(target->schema, nullptr);
  EXPECT_EQ(host_schema->stable_id, target->schema->stable_id);
  EXPECT_EQ(host_schema->version, target->schema->version);
  EXPECT_EQ(host_schema->structural_hash, target->schema->structural_hash);
  EXPECT_NE(host_schema->contract_hash, target->schema->contract_hash);

  auto attached =
      params::set_params(*graph, config_document(73U), {params::bind(handle, "/managed")});

  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().token(), "PARAM_CONFIG_SCHEMA_MISMATCH");
}

// REQ: PARAM-013
// REQ: PARAM-048
TEST(ParameterDSO, GraphHandleBindingRejectsWrongOperatorIdentity) {
  const std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> graph{create_dso_parameter_graph()};
  ASSERT_NE(graph, nullptr);
  auto handle = dso_parameter_operator_handle(graph.get());
  auto state = detail::ParameterConfigBindingAccess::state(handle).lock();
  ASSERT_NE(state, nullptr);

  auto target = detail::graph_operator_parameter_target(
      state, 0U, &detail::kGraphOperatorTypeToken<WrongDSOParameterOperator>, nullptr);

  ASSERT_FALSE(target.has_value());
  EXPECT_EQ(target.error().message, "PARAM_CONFIG_HANDLE_INVALID");
}

// REQ: PARAM-013
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterDSO, PreparedParamsAssignmentUsesDefiningModuleIdentity) {
  std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> graph{create_dso_parameter_graph()};
  ASSERT_NE(graph, nullptr);
  auto handle = dso_parameter_operator_handle(graph.get());
  const config::Document source = config_document(81U);
  auto decoded = params::try_decode<DSOParams>(source, "/managed", DSOParams{});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();

  handle.set_params(std::move(*decoded));

  const ExecutionPlan plan = compile(*graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const DSOParams* const resolved = compiled_params(plan);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->value, DSOFieldValue::kPrepared);
}

// REQ: PARAM-048
TEST(ParameterDSO, GraphHandleBindingRejectsWrongGraphAndStaleHandle) {
  std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> source_graph{create_dso_parameter_graph()};
  ASSERT_NE(source_graph, nullptr);
  auto handle = dso_parameter_operator_handle(source_graph.get());
  auto binding = params::bind(handle, "/managed");
  Graph other{"other-dso-parameter-graph"};

  auto wrong_graph = params::set_params(other, config_document(12U), {binding});
  ASSERT_FALSE(wrong_graph.has_value());
  EXPECT_EQ(wrong_graph.error().token(), "PARAM_CONFIG_GRAPH_MISMATCH");

  source_graph.reset();
  try {
    (void)params::bind(handle, "/managed");
    FAIL() << "expected stale handle rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(make_error(error).message, "PARAM_CONFIG_HANDLE_INVALID");
  }
}

// REQ: PARAM-013
// REQ: PARAM-048
TEST(ParameterDSO, PreparedParamsAssignmentRejectsMismatchedEvidence) {
  const std::unique_ptr<DSOParameterGraph, DSOGraphDeleter> graph{create_dso_parameter_graph()};
  ASSERT_NE(graph, nullptr);
  auto handle = dso_parameter_operator_handle(graph.get());
  auto decoded = params::try_decode<DSOParams>(config_document(91U), "/managed", DSOParams{});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& source_evidence = detail::PreparedParamsAccess::evidence(*decoded);
  ASSERT_NE(source_evidence, nullptr);
  auto mismatched = std::make_shared<detail::ParameterInputEvidenceState>(*source_evidence);
  mismatched->schema_id = "holoscan.test.GenuinelyDifferentParams";
  auto forged = detail::PreparedParamsAccess::make(
      std::move(*decoded).take(),
      std::shared_ptr<const detail::ParameterInputEvidenceState>{std::move(mismatched)});

  try {
    handle.set_params(std::move(forged));
    FAIL() << "expected mismatched prepared evidence rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(make_error(error).message, "GRAPH_PARAMETER_INPUT_EVIDENCE_MISMATCH");
  }
}

// REQ: COMPILE-077
TEST(ParameterDSO, ProjectorIdentityAcceptsOneExportedFunctionAcrossModules) {
  // Key setup: setup executes once through each module, while both declarations name the same
  // exported function and exact Params type.
  //
  // plugin_source    host_source
  Graph graph{"shared-projector-across-modules"};
  (void)graph.op<DSOProjectedOperator>("plugin_source");
  (void)graph.op<HostProjectedOperator>("host_source");
  const ExecutionPlan plan = compile(graph);

  // Key assertion: hidden template state does not create a false implementation collision.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.tensor_allocations().size(), 2U);
}

// REQ: COMPILE-077
TEST(ParameterDSO, ProjectorIdentityRejectsDifferentFunctionsAcrossModules) {
  // Key setup: one plugin function and one genuinely different host function reuse one stable ID.
  //
  // plugin_source    alternate_source
  Graph graph{"different-projectors-across-modules"};
  (void)graph.op<DSOProjectedOperator>("plugin_source");
  (void)graph.op<AlternateHostProjectedOperator>("alternate_source");
  const ExecutionPlan plan = compile(graph);

  // Key assertion: canonical callable equality still rejects a real semantic-ID collision.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(plan.tensor_allocations().empty());
  EXPECT_NE(plan.json().find("PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"), std::string::npos);
}
#endif

}  // namespace
}  // namespace holoscan::test
