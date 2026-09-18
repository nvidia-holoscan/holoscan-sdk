/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>

#include "compile_test_graph.hpp"
#include "compiler/compiler_diagnostic.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/tensor_memory_kind_derivation.hpp"
#include "core/diagnostic_context.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_schema_test_access.hpp"

namespace holoscan {

// These fresh types keep the reader specialization confined to this translation unit's fixtures.
enum class MemoryBooleanSelection : std::uint8_t {
  kHost = 1U,
  kPinned = 4U,
  kDevice = 9U,
  // Private replay-only value, deliberately omitted from the authored schema and mapping.
  kUnmapped = 2U,
};
enum class MemoryBooleanReadMode : std::uint8_t { kNative, kReject, kNone, kWrongKind };

struct MemoryBooleanLeafParams {
  MemoryKind memory_kind{MemoryKind::kHost};
};

HOLOSCAN_PARAM_SCHEMA(MemoryBooleanLeafParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.MemoryBooleanLeafParams", 1)),
                      holoscan::field<&MemoryBooleanLeafParams::memory_kind>("memory_kind")
                          .enum_values({param_enum_value("host", MemoryKind::kHost),
                                        param_enum_value("pinned", MemoryKind::kPinnedHost)})
                          .recompile_required());

struct MemoryBooleanIntermediateParams {
  MemoryBooleanLeafParams secret{};
};

HOLOSCAN_PARAM_SCHEMA(
    MemoryBooleanIntermediateParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.MemoryBooleanIntermediateParams", 1)),
    holoscan::field<&MemoryBooleanIntermediateParams::secret>("secret_inner_canary")
        .nested<MemoryBooleanLeafParams>()
        .sensitive());

struct MemoryBooleanProbeParams {
  MemoryKind memory_kind{MemoryKind::kHost};
  MemoryBooleanSelection selection{MemoryBooleanSelection::kHost};
  MemoryBooleanLeafParams secret{};
  MemoryBooleanIntermediateParams intermediate{};
  std::uint64_t private_value{0xfedcba9876543210ULL};
  // Test controls are not Parameter declarations. They live in owned factory aggregates, never
  // in global mutable state. Only the private lowering copy uses a non-native mode.
  MemoryBooleanReadMode read_mode{MemoryBooleanReadMode::kNative};
  mutable std::uint32_t reader_calls{};
};

HOLOSCAN_PARAM_SCHEMA(
    MemoryBooleanProbeParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.MemoryBooleanProbeParams", 1)),
    holoscan::field<&MemoryBooleanProbeParams::memory_kind>("memory_kind")
        .enum_values({param_enum_value("host", MemoryKind::kHost),
                      param_enum_value("pinned", MemoryKind::kPinnedHost)})
        .recompile_required(),
    holoscan::field<&MemoryBooleanProbeParams::selection>("selection")
        .enum_values({param_enum_value("host", MemoryBooleanSelection::kHost),
                      param_enum_value("pinned", MemoryBooleanSelection::kPinned),
                      param_enum_value("device", MemoryBooleanSelection::kDevice)})
        .recompile_required(),
    holoscan::field<&MemoryBooleanProbeParams::secret>("secret_root_canary")
        .nested<MemoryBooleanLeafParams>()
        .sensitive(),
    holoscan::field<&MemoryBooleanProbeParams::intermediate>("intermediate")
        .nested<MemoryBooleanIntermediateParams>(),
    holoscan::field<&MemoryBooleanProbeParams::private_value>("secret_value_canary").sensitive());

namespace detail {
namespace {

[[nodiscard]] bool read_memory_boolean_probe(const MemoryBooleanProbeParams& params,
                                             ParamScalarValue native,
                                             ParamScalarValue& result) noexcept {
  ++params.reader_calls;
  switch (params.read_mode) {
    case MemoryBooleanReadMode::kNative:
      result = native;
      return true;
    case MemoryBooleanReadMode::kReject:
      return false;
    case MemoryBooleanReadMode::kNone:
      result = {};
      return true;
    case MemoryBooleanReadMode::kWrongKind:
      result = ParamScalarValue::from(true);
      return true;
  }
  return false;
}

}  // namespace

// Specialize only the reader for fresh test member paths, before expression instantiation.
// Schema, typed identity, ancestry sensitivity, and production admission predicates stay intact.
template <>
bool TensorMemoryKindParameterDependencyModel<&MemoryBooleanProbeParams::memory_kind>::read_initial(
    const void* aggregate, ParamScalarValue& result) noexcept {
  if (aggregate == nullptr) {
    return false;
  }
  const auto& params = *static_cast<const MemoryBooleanProbeParams*>(aggregate);
  return read_memory_boolean_probe(params, ParamScalarValue::from(params.memory_kind), result);
}

template <>
bool TensorMemoryKindParameterDependencyModel<&MemoryBooleanProbeParams::selection>::read_initial(
    const void* aggregate, ParamScalarValue& result) noexcept {
  if (aggregate == nullptr) {
    return false;
  }
  const auto& params = *static_cast<const MemoryBooleanProbeParams*>(aggregate);
  return read_memory_boolean_probe(params, ParamScalarValue::from(params.selection), result);
}

template <>
bool TensorMemoryKindParameterDependencyModel<
    &MemoryBooleanProbeParams::secret,
    &MemoryBooleanLeafParams::memory_kind>::read_initial(const void* aggregate,
                                                         ParamScalarValue& result) noexcept {
  if (aggregate == nullptr) {
    return false;
  }
  const auto& params = *static_cast<const MemoryBooleanProbeParams*>(aggregate);
  return read_memory_boolean_probe(
      params, ParamScalarValue::from(params.secret.memory_kind), result);
}

template <>
bool TensorMemoryKindParameterDependencyModel<
    &MemoryBooleanProbeParams::intermediate, &MemoryBooleanIntermediateParams::secret,
    &MemoryBooleanLeafParams::memory_kind>::read_initial(const void* aggregate,
                                                         ParamScalarValue& result) noexcept {
  if (aggregate == nullptr) {
    return false;
  }
  const auto& params = *static_cast<const MemoryBooleanProbeParams*>(aggregate);
  return read_memory_boolean_probe(
      params, ParamScalarValue::from(params.intermediate.secret.memory_kind), result);
}

}  // namespace detail

namespace {

enum class MemoryBooleanExpression : std::uint8_t {
  kDirect,
  kMapped,
  kSensitiveRoot,
  kSensitiveInner
};

class MemoryBooleanSource final : public Operator<MemoryBooleanProbeParams> {
 public:
  explicit MemoryBooleanSource(MemoryBooleanExpression expression) : expression_(expression) {}

  void setup(OperatorSpec& spec) override {
    TensorMemoryKindExpression memory;
    switch (expression_) {
      case MemoryBooleanExpression::kDirect:
        memory = parameter_memory_kind(param_ref<&MemoryBooleanProbeParams::memory_kind>());
        break;
      case MemoryBooleanExpression::kMapped:
        memory = parameter_memory_kind(
            param_ref<&MemoryBooleanProbeParams::selection>(),
            {memory_kind_case(MemoryBooleanSelection::kHost, MemoryKind::kHost),
             memory_kind_case(MemoryBooleanSelection::kPinned, MemoryKind::kPinnedHost),
             memory_kind_case(MemoryBooleanSelection::kDevice, MemoryKind::kCudaDevice)});
        break;
      case MemoryBooleanExpression::kSensitiveRoot:
        memory = parameter_memory_kind(
            param_ref<&MemoryBooleanProbeParams::secret, &MemoryBooleanLeafParams::memory_kind>());
        break;
      case MemoryBooleanExpression::kSensitiveInner:
        memory = parameter_memory_kind(param_ref<&MemoryBooleanProbeParams::intermediate,
                                                 &MemoryBooleanIntermediateParams::secret,
                                                 &MemoryBooleanLeafParams::memory_kind>());
        break;
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{.representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U},
                             .bounds = tensor_bounds(4096U),
                             .storage = TensorOutputStorage::kRuntimePool},
            TensorOutputDerivations{.memory_kind = std::move(memory)});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  MemoryBooleanExpression expression_;
};

// Only the copied inventory's value getter differs. Real construction/setup still use the valid
// delegate. The override object has the exact Root type and outlives the entire lowering call.
class MemoryBooleanFactory final : public detail::OperatorFactoryBase {
 public:
  MemoryBooleanFactory(std::shared_ptr<detail::OperatorFactoryBase> delegate,
                       MemoryBooleanProbeParams value, bool null_value)
      : delegate_(std::move(delegate)), value_(value), null_value_(null_value) {}

  expected<std::unique_ptr<OperatorBase>, Error> create(
      detail::DiagnosticCapture* capture) const override {
    return delegate_->create(capture);
  }

  std::shared_ptr<detail::OperatorFactoryBase> clone() const override {
    return std::make_shared<MemoryBooleanFactory>(delegate_->clone(), value_, null_value_);
  }

  expected<std::shared_ptr<detail::OperatorFactoryBase>, Error> clone_with_params(
      const void* params, const void* type_token) const override {
    auto cloned = delegate_->clone_with_params(params, type_token);
    if (!cloned) {
      return make_unexpected(std::move(cloned.error()));
    }
    return std::shared_ptr<detail::OperatorFactoryBase>{
        std::make_shared<MemoryBooleanFactory>(std::move(cloned.value()), value_, null_value_)};
  }

  const ParamTypeDescriptor& parameter_type() const noexcept override {
    return delegate_->parameter_type();
  }
  const ParamSchemaDescriptor* parameter_schema() const noexcept override {
    return delegate_->parameter_schema();
  }
  const void* initial_parameter_value() const noexcept override {
    return null_value_ ? nullptr : &value_;
  }
  bool has_explicit_parameter_baseline() const noexcept override {
    return delegate_->has_explicit_parameter_baseline();
  }
  bool has_no_params() const noexcept override { return delegate_->has_no_params(); }
  LifecycleHookSet lifecycle_hooks() const noexcept override {
    return delegate_->lifecycle_hooks();
  }
  std::uint32_t reader_calls() const noexcept { return value_.reader_calls; }

 private:
  std::shared_ptr<detail::OperatorFactoryBase> delegate_;
  MemoryBooleanProbeParams value_;
  bool null_value_{};
};

class MemoryKindBooleanDiagnostics : public ::testing::Test {
 protected:
  void prepare(MemoryBooleanExpression expression) {
    test::add_compile_test_default_clock(graph_);
    auto source = graph_.op<MemoryBooleanSource>("source", expression);
    // Always bind an owned aggregate: the reader must never mutate a factory's static default.
    source.set_params(MemoryBooleanProbeParams{});
    baseline_ = compile(graph_);
    const bool sensitive = expression == MemoryBooleanExpression::kSensitiveRoot ||
                           expression == MemoryBooleanExpression::kSensitiveInner;
    if (sensitive) {
      ASSERT_FALSE(baseline_.ok());
      ASSERT_TRUE(
          test::has_diagnostic(baseline_, "PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"));
    } else {
      ASSERT_TRUE(baseline_.ok()) << baseline_.json();
    }
    for (const auto& diagnostic : baseline_.diagnostics()) {
      EXPECT_FALSE(std::string_view{diagnostic.token}.starts_with("PARAMETER_"));
      EXPECT_FALSE(std::string_view{diagnostic.token}.starts_with("TEMPORAL_"));
    }
    const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph_);
    ASSERT_NE(snapshot, nullptr);
    snapshot_.emplace(*snapshot);
    const auto* state = detail::ExecutionPlanAccess::state(baseline_);
    ASSERT_NE(state, nullptr);
    factories_ = state->executable_factories;
    for (auto& entry : factories_.operators) {
      ASSERT_NE(entry.factory, nullptr);
      entry.factory = entry.factory->clone();
      ASSERT_NE(entry.factory, nullptr);
    }
    fields_ = state->parameter_fields;
    generation_ = state->plan_generation;
    ASSERT_NE(generation_, 0U);
    const auto port =
        std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& row) {
          return row.direction == PortDirection::kOutput && row.name == "output";
        });
    ASSERT_NE(port, snapshot->ports.end());
    output_ = PortId{port->id.value};
    owner_ = port->owner;
  }

  void lower(MemoryBooleanProbeParams params, bool null_value = false) {
    if (!snapshot_.has_value()) {
      FAIL() << "Expected an unresolved snapshot prepared from the authored graph";
    }
    auto physical = snapshot_.value();
    const auto entry = std::find_if(factories_.operators.begin(),
                                    factories_.operators.end(),
                                    [&](const auto& row) { return row.vertex == owner_; });
    ASSERT_NE(entry, factories_.operators.end());
    probe_ = std::make_shared<MemoryBooleanFactory>(entry->factory, params, null_value);
    entry->factory = probe_;
    lowered_ =
        detail::lower_tensor_memory_kind_derivations(factories_, physical, generation_, fields_);
  }

  void expect_failure(std::string_view token, std::string_view required, std::uint32_t reads) {
    ASSERT_NE(probe_, nullptr);
    EXPECT_EQ(probe_->reader_calls(), reads);
    EXPECT_TRUE(lowered_.records.empty());
    ASSERT_EQ(lowered_.diagnostics.size(), 1U);
    const auto& row = lowered_.diagnostics.front();
    EXPECT_EQ(row.token, token);
    EXPECT_EQ(row.source.graph_generation, 0U);
    EXPECT_EQ(row.source.graph, GraphId{});
    EXPECT_EQ(row.source.vertex, owner_);
    EXPECT_EQ(row.source.port, output_);
    EXPECT_EQ(row.json_pointer,
              "/ports/" + std::to_string(output_.value()) + "/tensor_memory_kind_expression");
    EXPECT_EQ(row.required, required);
    EXPECT_TRUE(row.provided.empty());
    // Explicit guards keep dereferences legible to both readers and optional-access analysis.
    if (!row.detail.has_value()) {
      FAIL() << "Expected supplemental context for the Boolean rejection";
    }
    // Resolve names through the production publication helper and its canonical Graph ID domain.
    detail::ExecutionPlanState publication;
    publication.graph = detail::ExecutionPlanAccess::state(baseline_)->graph;
    detail::publish_derivation_diagnostic(publication, row);
    ASSERT_EQ(publication.diagnostics.size(), 1U);
    const auto& public_row = publication.diagnostics.front();
    ASSERT_TRUE(public_row.detail.has_value());
    if (!public_row.detail.has_value()) {
      FAIL() << "Expected published Boolean rejection context";
    }
    const auto& supplemental = public_row.detail.value();
    EXPECT_EQ(supplemental.operation, DiagnosticOperation::kCompile);
    EXPECT_EQ(supplemental.error.code, ErrorCode::kSuccess);
    EXPECT_TRUE(supplemental.error.message.empty());
    EXPECT_TRUE(supplemental.causes.empty());
    EXPECT_NE(supplemental.operator_path.find("source"), std::string::npos);
    EXPECT_EQ(supplemental.port_name, "output");

    std::string owned = supplemental.operator_path + supplemental.port_name +
                        supplemental.related_operator_path + supplemental.related_port_name +
                        supplemental.error.message;
    for (const auto& mismatch : supplemental.mismatches) {
      owned += mismatch.field + mismatch.required + mismatch.provided;
    }
    // Producer data is inspected before rendering; no assumed new metadata fields or labels.
    EXPECT_LE(owned.size(), detail::DiagnosticCapture::kTextCapacity);
    const std::array diagnostics{public_row};
    const auto single = format_diagnostic(public_row, ExecutionGraphView{});
    const auto report = format_diagnostics(diagnostics, baseline_.graph());
    EXPECT_LE(report.size(), 16384U);
    const std::string all_text = owned + single + report;
    for (const std::string_view forbidden : {"secret_root_canary",
                                             "secret_inner_canary",
                                             "secret_value_canary",
                                             "fedcba9876543210",
                                             "18364758544493064720"}) {
      EXPECT_EQ(all_text.find(forbidden), std::string::npos) << forbidden;
    }
    std::ostringstream factory_address;
    factory_address << static_cast<const void*>(probe_.get());
    EXPECT_EQ(all_text.find(factory_address.str()), std::string::npos);
    if (const void* value = probe_->initial_parameter_value(); value != nullptr) {
      std::ostringstream parameter_address;
      parameter_address << value;
      EXPECT_EQ(all_text.find(parameter_address.str()), std::string::npos);
    }
    const std::array readers{&detail::TensorMemoryKindParameterDependencyModel<
                                 &MemoryBooleanProbeParams::memory_kind>::read_initial,
                             &detail::TensorMemoryKindParameterDependencyModel<
                                 &MemoryBooleanProbeParams::selection>::read_initial,
                             &detail::TensorMemoryKindParameterDependencyModel<
                                 &MemoryBooleanProbeParams::secret,
                                 &MemoryBooleanLeafParams::memory_kind>::read_initial,
                             &detail::TensorMemoryKindParameterDependencyModel<
                                 &MemoryBooleanProbeParams::intermediate,
                                 &MemoryBooleanIntermediateParams::secret,
                                 &MemoryBooleanLeafParams::memory_kind>::read_initial};
    for (const auto reader : readers) {
      std::ostringstream reader_address;
      reader_address << std::hex << reinterpret_cast<std::uintptr_t>(reader);
      EXPECT_EQ(all_text.find(reader_address.str()), std::string::npos);
    }
    EXPECT_EQ(single.find('\x1b'), std::string::npos);
    EXPECT_EQ(report.find('\x1b'), std::string::npos);
  }

  Graph graph_{"memory-boolean-diagnostics"};
  ExecutionPlan baseline_;
  std::optional<detail::GraphSnapshotData> snapshot_;
  detail::ExecutableFactoryInventory factories_;
  std::vector<ParameterFieldPlanRecord> fields_;
  PlanGenerationId generation_{};
  PortId output_{};
  VertexId owner_{};
  std::shared_ptr<MemoryBooleanFactory> probe_;
  detail::TensorMemoryKindDerivationLowering lowered_;
};

// REQ: COMPILE-076
TEST_F(MemoryKindBooleanDiagnostics, AdmittedPrivateInputReadsOnceAndResolves) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kDirect));
  ASSERT_NO_FATAL_FAILURE(lower(MemoryBooleanProbeParams{}));
  ASSERT_NE(probe_, nullptr);
  EXPECT_EQ(probe_->reader_calls(), 1U);
  EXPECT_TRUE(lowered_.diagnostics.empty());
  ASSERT_EQ(lowered_.records.size(), 1U);
  EXPECT_EQ(lowered_.records.front().resolved_memory_kind, MemoryKind::kHost);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, FailedInitialReadKeepsContextWithoutInventingAnError) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kDirect));
  MemoryBooleanProbeParams params;
  params.read_mode = MemoryBooleanReadMode::kReject;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure(
      "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID", "one valid generation-zero scalar", 1U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, NullPrivateInitialValueIsRejectedAfterSuccessfulReplay) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kDirect));
  ASSERT_NO_FATAL_FAILURE(lower(MemoryBooleanProbeParams{}, true));
  expect_failure(
      "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID", "one valid generation-zero scalar", 0U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, EmptySuccessfulReadKeepsTheScalarRejection) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kDirect));
  MemoryBooleanProbeParams params;
  params.read_mode = MemoryBooleanReadMode::kNone;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure(
      "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID", "one valid generation-zero scalar", 1U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, WrongScalarKindReachesTheExactMemoryKindDecode) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kDirect));
  MemoryBooleanProbeParams params;
  params.read_mode = MemoryBooleanReadMode::kWrongKind;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure(
      "PHYSICAL_TENSOR_MEMORY_KIND_PARAMETER_TYPE_INVALID", "exact MemoryKind Parameter", 1U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, MappingMissKeepsTheAdmittedMappingAndRejection) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kMapped));
  MemoryBooleanProbeParams params;
  params.selection = MemoryBooleanSelection::kUnmapped;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure(
      "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID", "one admitted mapped generation-zero value", 1U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, UnknownPrivateValueReachesConcreteKindValidation) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kDirect));
  MemoryBooleanProbeParams params;
  params.memory_kind = MemoryKind::kUnknown;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure(
      "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID", "concrete non-unknown memory kind", 1U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, SensitiveRootRejectsBeforeTheActualReaderRuns) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kSensitiveRoot));
  MemoryBooleanProbeParams params;
  params.read_mode = MemoryBooleanReadMode::kWrongKind;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure("PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID",
                 "non-sensitive memory-kind Parameter path",
                 0U);
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST_F(MemoryKindBooleanDiagnostics, SensitiveIntermediateRejectsBeforeTheActualReaderRuns) {
  ASSERT_NO_FATAL_FAILURE(prepare(MemoryBooleanExpression::kSensitiveInner));
  MemoryBooleanProbeParams params;
  params.read_mode = MemoryBooleanReadMode::kWrongKind;
  ASSERT_NO_FATAL_FAILURE(lower(params));
  expect_failure("PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID",
                 "non-sensitive memory-kind Parameter path",
                 0U);
}

}  // namespace
}  // namespace holoscan
