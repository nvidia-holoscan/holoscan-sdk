/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/readiness_source.hpp>

#include <core/operator_spec_compiler_access.hpp>
#include <core/operator_spec_test_access.hpp>
#include <core/operator_test_binding.hpp>

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

struct OperatorSpecSchemaPayload {
  std::uint32_t value{};
};

template <>
consteval SchemaIdentity schema_identity<OperatorSpecSchemaPayload>() {
  return make_schema_identity(
      make_schema_family_identity("holoscan.test.OperatorSpecSchemaPayload", 7U),
      make_flatbuffer_file_identifier("OSSP"));
}

enum class OperatorSpecMemoryMode : std::uint8_t {
  kHost = 1,
  kPinned = 4,
};

struct OperatorSpecNestedMemoryParams {
  MemoryKind memory_kind{MemoryKind::kPinnedHost};
};

HOLOSCAN_PARAM_SCHEMA(OperatorSpecNestedMemoryParams,
                      holoscan::param_schema("nvidia.holoscan.test.OperatorSpecNestedMemoryParams",
                                             1),
                      holoscan::field<&OperatorSpecNestedMemoryParams::memory_kind>("memory_kind")
                          .enum_values({param_enum_value("host", MemoryKind::kHost),
                                        param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                                        param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                                        param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
                          .recompile_required());

struct OperatorSpecMemoryParams {
  MemoryKind memory_kind{MemoryKind::kHost};
  OperatorSpecMemoryMode mode{OperatorSpecMemoryMode::kHost};
  bool pinned{};
  OperatorSpecNestedMemoryParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecMemoryParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecMemoryParams", 1),
    holoscan::field<&OperatorSpecMemoryParams::memory_kind>("memory_kind")
        .enum_values({param_enum_value("host", MemoryKind::kHost),
                      param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                      param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                      param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
        .recompile_required(),
    holoscan::field<&OperatorSpecMemoryParams::mode>("mode")
        .enum_values({param_enum_value("host", OperatorSpecMemoryMode::kHost),
                      param_enum_value("pinned", OperatorSpecMemoryMode::kPinned)})
        .recompile_required(),
    holoscan::field<&OperatorSpecMemoryParams::pinned>("pinned").recompile_required(),
    holoscan::field<&OperatorSpecMemoryParams::nested>("nested")
        .nested<OperatorSpecNestedMemoryParams>());

struct OperatorSpecUnboundedMemoryKindParams {
  MemoryKind memory_kind{MemoryKind::kHost};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecUnboundedMemoryKindParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecUnboundedMemoryKindParams", 1),
    holoscan::field<&OperatorSpecUnboundedMemoryKindParams::memory_kind>("memory_kind")
        .recompile_required());

struct OperatorSpecProjectionParams {
  std::uint32_t width{16U};
  std::uint32_t height{8U};
  OperatorSpecNestedMemoryParams nested{};
  OperatorSpecNestedMemoryParams alternate_nested{};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecProjectionParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecProjectionParams", 1),
    holoscan::field<&OperatorSpecProjectionParams::width>("width").recompile_required(),
    holoscan::field<&OperatorSpecProjectionParams::height>("height").recompile_required(),
    holoscan::field<&OperatorSpecProjectionParams::nested>("nested")
        .nested<OperatorSpecNestedMemoryParams>(),
    holoscan::field<&OperatorSpecProjectionParams::alternate_nested>("alternate_nested")
        .nested<OperatorSpecNestedMemoryParams>());

struct OperatorSpecRepeatedLeafParams {
  std::uint32_t bytes{1U};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecRepeatedLeafParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecRepeatedLeafParams", 1),
    holoscan::field<&OperatorSpecRepeatedLeafParams::bytes>("bytes").recompile_required());

struct OperatorSpecDependencyLimitParams {
  OperatorSpecRepeatedLeafParams first{};
  OperatorSpecRepeatedLeafParams second{};
  std::uint32_t scalar0{1U};
  std::uint32_t scalar1{1U};
  std::uint32_t scalar2{1U};
  std::uint32_t scalar3{1U};
  std::uint32_t scalar4{1U};
  std::uint32_t scalar5{1U};
  std::uint32_t scalar6{1U};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecDependencyLimitParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecDependencyLimitParams", 1),
    holoscan::field<&OperatorSpecDependencyLimitParams::first>("first")
        .nested<OperatorSpecRepeatedLeafParams>(),
    holoscan::field<&OperatorSpecDependencyLimitParams::second>("second")
        .nested<OperatorSpecRepeatedLeafParams>(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar0>("scalar0").recompile_required(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar1>("scalar1").recompile_required(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar2>("scalar2").recompile_required(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar3>("scalar3").recompile_required(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar4>("scalar4").recompile_required(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar5>("scalar5").recompile_required(),
    holoscan::field<&OperatorSpecDependencyLimitParams::scalar6>("scalar6").recompile_required());

// Provides two different typed member paths with the same dotted display path for
// TensorByteSpanProjectionPreservesTypedMemberBoundaries.
struct OperatorSpecPathLeafParams {
  std::uint32_t bytes{17U};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecPathLeafParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecPathLeafParams", 1),
    holoscan::field<&OperatorSpecPathLeafParams::bytes>("bytes").recompile_required());

struct OperatorSpecPathInnerParams {
  OperatorSpecPathLeafParams b{};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecPathInnerParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecPathInnerParams", 1),
    holoscan::field<&OperatorSpecPathInnerParams::b>("b").nested<OperatorSpecPathLeafParams>());

struct OperatorSpecPathBranchParams {
  OperatorSpecPathLeafParams dotted{};
  OperatorSpecPathInnerParams nested{};
  std::uint32_t scalar0{1U};
  std::uint32_t scalar1{1U};
  std::uint32_t scalar2{1U};
  std::uint32_t scalar3{1U};
  std::uint32_t scalar4{1U};
  std::uint32_t scalar5{1U};
  std::uint32_t scalar6{1U};
};

HOLOSCAN_PARAM_SCHEMA(
    OperatorSpecPathBranchParams,
    holoscan::param_schema("nvidia.holoscan.test.OperatorSpecPathBranchParams", 1),
    holoscan::field<&OperatorSpecPathBranchParams::dotted>("a.b")
        .nested<OperatorSpecPathLeafParams>(),
    holoscan::field<&OperatorSpecPathBranchParams::nested>("a")
        .nested<OperatorSpecPathInnerParams>(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar0>("scalar0").recompile_required(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar1>("scalar1").recompile_required(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar2>("scalar2").recompile_required(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar3>("scalar3").recompile_required(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar4>("scalar4").recompile_required(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar5>("scalar5").recompile_required(),
    holoscan::field<&OperatorSpecPathBranchParams::scalar6>("scalar6").recompile_required());

struct OperatorSpecSharedPathParams {
  OperatorSpecPathBranchParams outer{};
};

HOLOSCAN_PARAM_SCHEMA(OperatorSpecSharedPathParams,
                      holoscan::param_schema("nvidia.holoscan.test.OperatorSpecSharedPathParams",
                                             1),
                      holoscan::field<&OperatorSpecSharedPathParams::outer>("outer")
                          .nested<OperatorSpecPathBranchParams>());

[[nodiscard]] expected<std::uint64_t, Error> operator_spec_projected_bytes(
    const OperatorSpecProjectionParams& params) noexcept {
  return static_cast<std::uint64_t>(params.width) * params.height;
}

[[nodiscard]] expected<std::uint64_t, Error> operator_spec_alternate_projected_bytes(
    const OperatorSpecProjectionParams& params) noexcept {
  return static_cast<std::uint64_t>(params.width) + params.height;
}

[[nodiscard]] expected<std::uint64_t, Error> operator_spec_path_projected_bytes(
    const OperatorSpecPathBranchParams& params) noexcept {
  return static_cast<std::uint64_t>(params.dotted.bytes) + params.nested.b.bytes;
}

[[nodiscard]] expected<std::uint64_t, Error> operator_spec_shared_path_projected_bytes(
    const OperatorSpecSharedPathParams& params) noexcept {
  return static_cast<std::uint64_t>(params.outer.dotted.bytes) + params.outer.nested.b.bytes;
}

namespace {

// Placeholder payload struct with a Tensor-adjacent field; reserved for future
// static-assert checks on schema detection traits.
struct CarrierWithTensorField {
  std::uint64_t frame_id{};
};

template <typename Spec, typename Port>
concept CanAssignInputBuilderToPort = requires(Spec& spec, Port& port) { port = spec.input(port); };

template <typename Spec, typename Port>
concept CanAssignOutputBuilderToPort =
    requires(Spec& spec, Port& port) { port = spec.output(port); };

template <typename Spec, typename Port>
concept HasInputArrayBinding = requires(Spec& spec, Port& port) { spec.input_array(port); };

template <typename Spec, typename Port>
concept HasOutputArrayBinding = requires(Spec& spec, Port& port) { spec.output_array(port); };

template <typename Builder>
concept HasStandaloneTensorAllocation =
    requires(Builder& builder, TensorPoolBounds bounds) { builder.tensor_allocation(bounds); };

using ScalarInput = Input<std::uint32_t>;
using ScalarOutput = Output<std::uint32_t>;
using TensorOutput = Output<Tensor>;

static_assert(detail::operator_spec_payload_kind<std::uint32_t>() == PayloadKind::kInlineTyped);
static_assert(detail::operator_spec_payload_kind<OperatorSpecSchemaPayload>() ==
              PayloadKind::kSerializedBytes);
static_assert(detail::operator_spec_payload_kind<Tensor>() == PayloadKind::kDescriptorBacked);
static_assert(!detail::operator_spec_schema_identity<std::uint32_t>().has_value());
static_assert(!detail::operator_spec_schema_identity<Tensor>().has_value());
static_assert(
    detail::operator_spec_schema_identity<OperatorSpecSchemaPayload>()->compatibility_epoch == 7);
static_assert(detail::operator_spec_schema_identity<OperatorSpecSchemaPayload>()->canonical_name ==
              "holoscan.test.OperatorSpecSchemaPayload");
static_assert(&detail::kOperatorSpecLocalTypeToken<std::uint32_t> !=
              &detail::kOperatorSpecLocalTypeToken<std::int32_t>);
static_assert(&detail::kOperatorSpecLocalTypeToken<OperatorSpecSchemaPayload> !=
              &detail::kOperatorSpecLocalTypeToken<Tensor>);

static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().input(std::declval<ScalarInput&>())),
                 InputSpecBuilder&>);
static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().output(std::declval<ScalarOutput&>())),
                 OutputSpecBuilder&>);
static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().output(std::declval<TensorOutput&>())),
                 TensorOutputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OperatorSpec&>().cuda_graph_capture_eligible()),
                           OperatorSpec&>);

static_assert(!CanAssignInputBuilderToPort<OperatorSpec, ScalarInput>);
static_assert(!CanAssignOutputBuilderToPort<OperatorSpec, ScalarOutput>);
static_assert(!HasInputArrayBinding<OperatorSpec, ScalarInput>);
static_assert(!HasOutputArrayBinding<OperatorSpec, ScalarOutput>);

static_assert(!HasStandaloneTensorAllocation<OutputSpecBuilder>);
static_assert(!HasStandaloneTensorAllocation<TensorOutputSpecBuilder>);
static_assert(std::same_as<decltype(std::declval<TensorOutputSpecBuilder&>()
                                        .max_emits_per_compute(2)
                                        .produces_tensor(TensorOutputSpec{
                                            .representation = {.rank = 4U},
                                            .bounds = tensor_bounds(4096U),
                                            .storage = TensorOutputStorage::kRuntimePool,
                                        })
                                        .late_joiner_history_depth(0)),
                           TensorOutputSpecBuilder&>);

// REQ: OPERATOR-051
TEST(OperatorSpec, CudaGraphCaptureEligibilityDefaultsDisabledAndSupportsFluentOverride) {
  OperatorSpec spec;
  EXPECT_FALSE(detail::OperatorSpecTestAccess::cuda_graph_capture_eligible(spec));

  EXPECT_EQ(&spec.cuda_graph_capture_eligible(), &spec);
  EXPECT_TRUE(detail::OperatorSpecTestAccess::cuda_graph_capture_eligible(spec));

  EXPECT_EQ(&spec.cuda_graph_capture_eligible(), &spec);
  EXPECT_TRUE(detail::OperatorSpecTestAccess::cuda_graph_capture_eligible(spec));

  EXPECT_EQ(&spec.cuda_graph_capture_eligible(false), &spec);
  EXPECT_FALSE(detail::OperatorSpecTestAccess::cuda_graph_capture_eligible(spec));
}

// Operator that registers the same Input<> field under two different names.
// Used by RejectsOnePortDeclarationBoundUnderDifferentNames to verify that a single
// port object cannot be aliased to multiple spec entries (kInvalidArgument expected).
class ReusedPortDeclarationOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_, "first");
    spec.input(input_, "second");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that passes an empty string as the readiness source name.
// Used by RejectsEmptyReadinessSourceName to verify kInvalidArgument is returned
// before any registration takes place.
class EmptyReadinessNameOperator final : public Operator<> {
 public:
  ReadableSource readable_;

  void setup(OperatorSpec& spec) override { spec.readable_source(readable_, ""); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that registers the same NotificationSource under two distinct names.
// Used by RejectsOneReadinessDeclarationBoundUnderDifferentNames.
class ReusedReadinessDeclarationOperator final : public Operator<> {
 public:
  NotificationSource notification_;

  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification_, "first");
    spec.notification_source(notification_, "second");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that registers two different readiness sources with the same name "ready".
// Used by RejectsDuplicateNamesAcrossReadinessSourceKinds (kDuplicateName expected).
class DuplicateReadinessNameOperator final : public Operator<> {
 public:
  ReadableSource readable_;
  DoorbellSource doorbell_;

  void setup(OperatorSpec& spec) override {
    spec.readable_source(readable_, "ready");
    spec.doorbell_source(doorbell_, "ready");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that gives an input port and a readiness source the same name.
// Used by RejectsDuplicateNamesAcrossPortAndReadinessKinds (kDuplicateName expected).
class DuplicatePortAndReadinessNameOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"ready"};
  ReadableSource readable_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.readable_source(readable_, "ready");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator with a clean, non-conflicting set of declarations (one input, one output,
// one readable source, one spin source).  Used by FreshSpecsAcceptTheSameStableDeclarations
// to verify that the same operator can be declared into multiple fresh specs without error.
class ValidDeclarationsOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};
  ReadableSource readable_;
  SpinSource spin_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
    spec.readable_source(readable_, "readable");
    spec.spin_source(spin_, "spin");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator with initially unnamed ports whose stable names are assigned through setup().
// Used by SetupNamesBecomeTheStableNamesOfInitiallyUnnamedPortDeclarations to confirm
// that the name passed to spec.input/output() is stored on the Port object itself.
class SetupNamedPortsOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_;
  Output<std::uint32_t> output_;
  Output<Tensor> tensor_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input");
    spec.output(output_, "output");
    spec.output(tensor_, "tensor");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator covering all three payload kinds in one spec:
//   scalar_ → kInlineTyped (plain C++ value type)
//   schema_ → kSerializedBytes (type with a consteval schema_identity<> specialization)
//   tensor_ → kDescriptorBacked (holoscan::Tensor with descriptor-backed transport)
// Representative test: AdmittedPayloadFamiliesRetainDistinctColdTypeMetadata
class TypedMetadataDeclarationsOperator final : public Operator<> {
 public:
  Input<std::uint32_t> scalar_{"scalar"};
  Output<OperatorSpecSchemaPayload> schema_{"schema"};
  Output<Tensor> tensor_{"tensor"};

  void setup(OperatorSpec& spec) override {
    spec.input(scalar_);
    spec.output(schema_);
    spec.output(tensor_);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that registers all four readiness source kinds and then configures them
// out-of-declaration-order.  This catches any implementation bug where builders are
// associated by a shared insertion index instead of the record's exact family+index.
// Representative test: InterleavedReadinessKindsRetainTheirExactBuilderFamilies
class InterleavedReadinessDeclarationsOperator final : public Operator<> {
 public:
  ReadableSource readable_;
  NotificationSource notification_;
  DoorbellSource doorbell_;
  SpinSource spin_;

  void setup(OperatorSpec& spec) override {
    auto& readable = spec.readable_source(readable_, "readable");
    auto& notification = spec.notification_source(notification_, "notification");
    auto& doorbell = spec.doorbell_source(doorbell_, "doorbell");
    auto& spin = spec.spin_source(spin_, "spin");

    // Exercise every builder after all four records exist. This detects accidental association
    // through one shared insertion index instead of the record's exact builder family and index.
    readable.capacity(1).clock(ClockKey{"readable-clock"});
    notification.capacity(4).sender_reference_capacity(3).clock(ClockKey{"notification-clock"});
    doorbell.capacity(2).clock(ClockKey{"doorbell-clock"});
    spin.capacity(1).clock(ClockKey{"spin-clock"});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Convenience wrapper that constructs a fresh SyntheticOperatorBinding and invokes
// declare_ports on the provided operator.  Returns the Error from the first violation,
// or void on success.  Used by all rejection and acceptance tests in this file.
template <typename Op>
expected<void, Error> declare_once(Op& op) {
  detail::SyntheticOperatorBinding binding;
  return binding.declare_ports(op);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, BindingBuildersAreColdMetadataNotPortReplacementValues) {
  EXPECT_FALSE((std::is_convertible_v<InputSpecBuilder&, ScalarInput>));
  EXPECT_FALSE((std::is_convertible_v<OutputSpecBuilder&, ScalarOutput>));
  EXPECT_FALSE((std::is_convertible_v<TensorOutputSpecBuilder&, TensorOutput>));
}

TEST(OperatorSpec, SetupNamesBecomeTheStableNamesOfInitiallyUnnamedPortDeclarations) {
  SetupNamedPortsOperator op;
  OperatorSpec spec;

  op.setup(spec);

  EXPECT_EQ(op.input_.name(), "input");
  EXPECT_EQ(op.output_.name(), "output");
  EXPECT_EQ(op.tensor_.name(), "tensor");
}

// REQ: OPERATOR-004
TEST(OperatorSpec, TensorBoundsAreFiniteValues) {
  constexpr TensorPoolBounds bounds{
      .max_rank = 8,
      .max_byte_span = 16U * 1024U * 1024U,
  };
  EXPECT_EQ(bounds.max_rank, 8);
  EXPECT_EQ(bounds.max_byte_span, 16U * 1024U * 1024U);
}

// REQ: COMPILE-067
TEST(OperatorSpec, TensorOutputRetainsUnresolvedDerivedByteSpanWithoutEagerAllocation) {
  OperatorSpec spec;
  Output<Tensor> output;

  // Key setup: declare finite rank metadata while leaving slot bytes to a closed expression.
  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          tensor_byte_span(1000U).aligned_to(256U));

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  // Key assertion: setup owns the expression, but physical allocation remains unresolved.
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(metadata.tensor_contract);
  EXPECT_FALSE(metadata.tensor_contract->max_byte_span);
  EXPECT_FALSE(metadata.tensor_allocation);
  ASSERT_TRUE(metadata.tensor_byte_span_expression);
  ASSERT_EQ(metadata.tensor_byte_span_expression->nodes.size(), 2U);
  EXPECT_EQ(metadata.tensor_byte_span_expression->nodes.back().operation,
            TensorByteSpanOperation::kAlignUp);
}

// REQ: COMPILE-076
TEST(OperatorSpec, TensorOutputRetainsCanonicalParameterDerivedMemoryKind) {
  OperatorSpec spec;
  Output<Tensor> output;

  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
              .bounds = tensor_bounds(4096U),
              .storage = TensorOutputStorage::kRuntimePool,
          },
          TensorOutputDerivations{
              .memory_kind =
                  parameter_memory_kind(param_ref<&OperatorSpecMemoryParams::memory_kind>()),
          });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  ASSERT_EQ(metadata.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(metadata.tensor_contract);
  EXPECT_FALSE(metadata.tensor_contract->representation.memory_kind);
  ASSERT_TRUE(metadata.tensor_allocation);
  EXPECT_EQ(metadata.tensor_allocation->max_byte_span, 4096U);
  ASSERT_TRUE(metadata.tensor_memory_kind_expression);
  EXPECT_EQ(metadata.tensor_memory_kind_expression->form, TensorMemoryKindExpressionForm::kDirect);
  EXPECT_EQ(metadata.tensor_memory_kind_expression->parameter_path, "memory_kind");
  EXPECT_TRUE(metadata.tensor_memory_kind_expression->mapping.empty());
}

// REQ: COMPILE-076
TEST(OperatorSpec, TensorMemoryKindMappingNormalizesIndependentOfCallSiteOrder) {
  const auto metadata_for = [](bool reversed) {
    OperatorSpec spec;
    Output<Tensor> output;
    const auto reference = param_ref<&OperatorSpecMemoryParams::mode>();
    const TensorMemoryKindExpression expression =
        reversed
            ? parameter_memory_kind(
                  reference,
                  {memory_kind_case(OperatorSpecMemoryMode::kPinned, MemoryKind::kPinnedHost),
                   memory_kind_case(OperatorSpecMemoryMode::kHost, MemoryKind::kHost)})
            : parameter_memory_kind(
                  reference,
                  {memory_kind_case(OperatorSpecMemoryMode::kHost, MemoryKind::kHost),
                   memory_kind_case(OperatorSpecMemoryMode::kPinned, MemoryKind::kPinnedHost)});
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.rank = 1U},
                .bounds = tensor_bounds(1024U),
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{.memory_kind = expression});
    return detail::OperatorSpecCompilerAccess::port(spec, 0U);
  };

  const auto ordered = metadata_for(false);
  const auto reversed = metadata_for(true);
  ASSERT_EQ(ordered.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_EQ(reversed.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(ordered.tensor_memory_kind_expression);
  ASSERT_TRUE(reversed.tensor_memory_kind_expression);
  EXPECT_EQ(*ordered.tensor_memory_kind_expression, *reversed.tensor_memory_kind_expression);
  ASSERT_EQ(ordered.tensor_memory_kind_expression->mapping.size(), 2U);
  EXPECT_EQ(ordered.tensor_memory_kind_expression->mapping[0].source,
            ParamScalarValue::from(OperatorSpecMemoryMode::kHost));
  EXPECT_EQ(ordered.tensor_memory_kind_expression->mapping[1].source,
            ParamScalarValue::from(OperatorSpecMemoryMode::kPinned));
}

// REQ: COMPILE-076
TEST(OperatorSpec, TensorMemoryKindDerivationSupportsNestedPathsAndSharedCarrier) {
  OperatorSpec spec;
  Output<Tensor> output;
  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.rank = 1U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          TensorOutputDerivations{
              .max_byte_span = tensor_byte_span(2048U),
              .memory_kind =
                  parameter_memory_kind(param_ref<&OperatorSpecMemoryParams::nested,
                                                  &OperatorSpecNestedMemoryParams::memory_kind>()),
          });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  ASSERT_EQ(metadata.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(metadata.tensor_contract);
  EXPECT_FALSE(metadata.tensor_contract->max_byte_span);
  EXPECT_FALSE(metadata.tensor_contract->representation.memory_kind);
  EXPECT_FALSE(metadata.tensor_allocation);
  ASSERT_TRUE(metadata.tensor_byte_span_expression);
  ASSERT_TRUE(metadata.tensor_memory_kind_expression);
  EXPECT_EQ(metadata.tensor_memory_kind_expression->parameter_path, "nested.memory_kind");
}

// REQ: COMPILE-076
TEST(OperatorSpec, TensorMemoryKindDerivationRejectsConflictsAndIncompleteMappings) {
  const TensorOutputSpec derived{
      .representation = {.rank = 1U},
      .bounds = tensor_bounds(1024U),
      .storage = TensorOutputStorage::kRuntimePool,
  };

  OperatorSpec literal;
  Output<Tensor> literal_output;
  auto literal_spec = derived;
  literal_spec.representation.memory_kind = MemoryKind::kHost;
  literal.output(literal_output, "output")
      .produces_tensor(literal_spec,
                       TensorOutputDerivations{
                           .memory_kind = parameter_memory_kind(
                               param_ref<&OperatorSpecMemoryParams::memory_kind>()),
                       });
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(literal, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT");

  OperatorSpec pass_through;
  Output<Tensor> pass_through_output;
  auto pass_through_spec = derived;
  pass_through_spec.storage = TensorOutputStorage::kPassThrough;
  pass_through.output(pass_through_output, "output")
      .produces_tensor(pass_through_spec,
                       TensorOutputDerivations{
                           .memory_kind = parameter_memory_kind(
                               param_ref<&OperatorSpecMemoryParams::memory_kind>()),
                       });
  EXPECT_EQ(
      detail::OperatorSpecCompilerAccess::port(pass_through, 0U).tensor_contract_error.message,
      "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID");

  OperatorSpec unbounded;
  Output<Tensor> unbounded_output;
  unbounded.output(unbounded_output, "output")
      .produces_tensor(derived,
                       TensorOutputDerivations{
                           .memory_kind = parameter_memory_kind(
                               param_ref<&OperatorSpecUnboundedMemoryKindParams::memory_kind>()),
                       });
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(unbounded, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID");

  OperatorSpec incomplete;
  Output<Tensor> incomplete_output;
  incomplete.output(incomplete_output, "output")
      .produces_tensor(
          derived,
          TensorOutputDerivations{
              .memory_kind = parameter_memory_kind(param_ref<&OperatorSpecMemoryParams::pinned>(),
                                                   {memory_kind_case(false, MemoryKind::kHost)}),
          });
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(incomplete, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_MEMORY_KIND_MAPPING_INVALID");

  const auto duplicate = parameter_memory_kind(param_ref<&OperatorSpecMemoryParams::pinned>(),
                                               {memory_kind_case(false, MemoryKind::kHost),
                                                memory_kind_case(false, MemoryKind::kPinnedHost)});
  EXPECT_EQ(duplicate.invalid_reason(), TensorMemoryKindExpressionInvalidReason::kMappingDuplicate);

  const auto unknown_target = parameter_memory_kind(
      param_ref<&OperatorSpecMemoryParams::pinned>(),
      {memory_kind_case(false, MemoryKind::kHost), memory_kind_case(true, MemoryKind::kUnknown)});
  EXPECT_EQ(unknown_target.invalid_reason(),
            TensorMemoryKindExpressionInvalidReason::kMappingTargetInvalid);

  OperatorSpec extra;
  Output<Tensor> extra_output;
  extra.output(extra_output, "output")
      .produces_tensor(
          derived,
          TensorOutputDerivations{
              .memory_kind = parameter_memory_kind(
                  param_ref<&OperatorSpecMemoryParams::mode>(),
                  {memory_kind_case(OperatorSpecMemoryMode::kHost, MemoryKind::kHost),
                   memory_kind_case(OperatorSpecMemoryMode::kPinned, MemoryKind::kPinnedHost),
                   memory_kind_case(static_cast<OperatorSpecMemoryMode>(9U), MemoryKind::kHost)}),
          });
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(extra, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_MEMORY_KIND_MAPPING_INVALID");
}

// REQ: COMPILE-076
TEST(OperatorSpec, TensorMemoryKindMappingRejectsMoreThanSixteenRows) {
  // Key setup: author one more unique source row than the expression-owned mapping limit.
  const auto expression = parameter_memory_kind(
      param_ref<&OperatorSpecMemoryParams::mode>(),
      {memory_kind_case(static_cast<OperatorSpecMemoryMode>(0U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(1U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(2U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(3U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(4U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(5U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(6U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(7U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(8U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(9U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(10U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(11U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(12U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(13U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(14U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(15U), MemoryKind::kHost),
       memory_kind_case(static_cast<OperatorSpecMemoryMode>(16U), MemoryKind::kHost)});

  // Key assertion: expression construction rejects the 17-row table before Graph admission.
  EXPECT_EQ(expression.invalid_reason(),
            TensorMemoryKindExpressionInvalidReason::kMappingLimitExceeded);
}

// REQ: COMPILE-077
TEST(OperatorSpec, TensorByteSpanProjectionRetainsCanonicalPortableIdentity) {
  const auto metadata_for = [](bool reverse_dependencies, std::string_view projection_id) {
    OperatorSpec spec;
    Output<Tensor> output;
    const TensorByteSpanExpression expression =
        reverse_dependencies ? parameter_projected_byte_span<&operator_spec_projected_bytes>(
                                   projection_id,
                                   param_ref<&OperatorSpecProjectionParams::nested,
                                             &OperatorSpecNestedMemoryParams::memory_kind>(),
                                   param_ref<&OperatorSpecProjectionParams::alternate_nested,
                                             &OperatorSpecNestedMemoryParams::memory_kind>(),
                                   param_ref<&OperatorSpecProjectionParams::height>(),
                                   param_ref<&OperatorSpecProjectionParams::width>())
                             : parameter_projected_byte_span<&operator_spec_projected_bytes>(
                                   projection_id,
                                   param_ref<&OperatorSpecProjectionParams::width>(),
                                   param_ref<&OperatorSpecProjectionParams::height>(),
                                   param_ref<&OperatorSpecProjectionParams::nested,
                                             &OperatorSpecNestedMemoryParams::memory_kind>(),
                                   param_ref<&OperatorSpecProjectionParams::alternate_nested,
                                             &OperatorSpecNestedMemoryParams::memory_kind>());
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            expression.aligned_to(256U));
    return detail::OperatorSpecCompilerAccess::port(spec, 0U);
  };

  const auto ordered = metadata_for(false, "nvidia.holoscan.test.projected-bytes.v1");
  const auto reversed = metadata_for(true, "nvidia.holoscan.test.projected-bytes.v1");
  const auto renamed = metadata_for(false, "nvidia.holoscan.test.projected-bytes.v2");

  ASSERT_EQ(ordered.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_EQ(reversed.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(ordered.tensor_byte_span_expression);
  ASSERT_TRUE(reversed.tensor_byte_span_expression);
  ASSERT_TRUE(renamed.tensor_byte_span_expression);
  EXPECT_EQ(*ordered.tensor_byte_span_expression, *reversed.tensor_byte_span_expression);
  EXPECT_NE(*ordered.tensor_byte_span_expression, *renamed.tensor_byte_span_expression);
  ASSERT_EQ(ordered.tensor_byte_span_expression->nodes.size(), 2U);
  const auto& projection = ordered.tensor_byte_span_expression->nodes.front();
  EXPECT_EQ(projection.operation, TensorByteSpanOperation::kParameterProjection);
  EXPECT_EQ(projection.projection_id, "nvidia.holoscan.test.projected-bytes.v1");
  EXPECT_EQ(projection.parameter_schema_id, "nvidia.holoscan.test.OperatorSpecProjectionParams");
  ASSERT_EQ(projection.projection_dependencies.size(), 4U);
  EXPECT_TRUE(std::ranges::any_of(projection.projection_dependencies, [](const auto& dependency) {
    return dependency.parameter_path == "nested.memory_kind";
  }));
  EXPECT_TRUE(std::ranges::any_of(projection.projection_dependencies, [](const auto& dependency) {
    return dependency.parameter_path == "alternate_nested.memory_kind";
  }));
  EXPECT_TRUE(std::ranges::all_of(projection.projection_dependencies, [](const auto& dependency) {
    return !dependency.parameter_member_full_ids.empty() &&
           dependency.parameter_member_full_ids.back() == dependency.parameter_full_id;
  }));
}

// REQ: COMPILE-077
TEST(OperatorSpec, TensorByteSpanProjectionPreservesTypedMemberBoundaries) {
  const auto metadata_for = [](bool shared_root, bool reverse_dependencies) {
    TensorByteSpanExpression expression;
    if (shared_root) {
      const auto dotted = param_ref<&OperatorSpecSharedPathParams::outer,
                                    &OperatorSpecPathBranchParams::dotted,
                                    &OperatorSpecPathLeafParams::bytes>();
      const auto nested = param_ref<&OperatorSpecSharedPathParams::outer,
                                    &OperatorSpecPathBranchParams::nested,
                                    &OperatorSpecPathInnerParams::b,
                                    &OperatorSpecPathLeafParams::bytes>();
      expression = reverse_dependencies
                       ? parameter_projected_byte_span<&operator_spec_shared_path_projected_bytes>(
                             "nvidia.holoscan.test.shared-path-projector.v1", nested, dotted)
                       : parameter_projected_byte_span<&operator_spec_shared_path_projected_bytes>(
                             "nvidia.holoscan.test.shared-path-projector.v1", dotted, nested);
    } else {
      const auto dotted =
          param_ref<&OperatorSpecPathBranchParams::dotted, &OperatorSpecPathLeafParams::bytes>();
      const auto nested = param_ref<&OperatorSpecPathBranchParams::nested,
                                    &OperatorSpecPathInnerParams::b,
                                    &OperatorSpecPathLeafParams::bytes>();
      expression = reverse_dependencies
                       ? parameter_projected_byte_span<&operator_spec_path_projected_bytes>(
                             "nvidia.holoscan.test.path-projector.v1", nested, dotted)
                       : parameter_projected_byte_span<&operator_spec_path_projected_bytes>(
                             "nvidia.holoscan.test.path-projector.v1", dotted, nested);
    }

    OperatorSpec spec;
    Output<Tensor> output;
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            std::move(expression));
    return detail::OperatorSpecCompilerAccess::port(spec, 0U);
  };

  // Key setup: normalize both ambiguous path shapes in both declaration orders.
  const auto different_root = metadata_for(false, false);
  const auto reversed_different_root = metadata_for(false, true);
  const auto shared_root = metadata_for(true, false);
  const auto reversed_shared_root = metadata_for(true, true);

  // Key assertion: display-path collisions retain distinct typed identities and canonical order.
  for (const auto* port : {&different_root, &shared_root}) {
    ASSERT_EQ(port->tensor_contract_error.code, ErrorCode::kSuccess);
    ASSERT_TRUE(port->tensor_byte_span_expression);
    const auto& dependencies =
        port->tensor_byte_span_expression->nodes.front().projection_dependencies;
    ASSERT_EQ(dependencies.size(), 2U);
    EXPECT_EQ(dependencies.front().parameter_path, dependencies.back().parameter_path);
    EXPECT_NE(dependencies.front().parameter_member_full_ids,
              dependencies.back().parameter_member_full_ids);
  }
  EXPECT_EQ(different_root.tensor_byte_span_expression,
            reversed_different_root.tensor_byte_span_expression);
  EXPECT_EQ(shared_root.tensor_byte_span_expression,
            reversed_shared_root.tensor_byte_span_expression);
}

// REQ: COMPILE-077
TEST(OperatorSpec, TensorByteSpanProjectionTypedPathsCountTowardDependencyLimit) {
  const auto projected = parameter_projected_byte_span<&operator_spec_path_projected_bytes>(
      "nvidia.holoscan.test.path-projector-limit.v1",
      param_ref<&OperatorSpecPathBranchParams::dotted, &OperatorSpecPathLeafParams::bytes>(),
      param_ref<&OperatorSpecPathBranchParams::nested,
                &OperatorSpecPathInnerParams::b,
                &OperatorSpecPathLeafParams::bytes>());
  auto within_limit =
      projected
          .maximum_with(parameter_byte_span(param_ref<&OperatorSpecPathBranchParams::scalar0>()))
          .maximum_with(parameter_byte_span(param_ref<&OperatorSpecPathBranchParams::scalar1>()))
          .maximum_with(parameter_byte_span(param_ref<&OperatorSpecPathBranchParams::scalar2>()))
          .maximum_with(parameter_byte_span(param_ref<&OperatorSpecPathBranchParams::scalar3>()))
          .maximum_with(parameter_byte_span(param_ref<&OperatorSpecPathBranchParams::scalar4>()))
          .maximum_with(parameter_byte_span(param_ref<&OperatorSpecPathBranchParams::scalar5>()));
  const TensorOutputSpec tensor_spec{
      .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
      .storage = TensorOutputStorage::kRuntimePool,
  };

  // Key setup: use the two colliding display paths plus six ordinary dependencies.
  OperatorSpec accepted;
  Output<Tensor> accepted_output;
  accepted.output(accepted_output, "output").produces_tensor(tensor_spec, within_limit);
  ASSERT_EQ(detail::OperatorSpecCompilerAccess::port(accepted, 0U).tensor_contract_error.code,
            ErrorCode::kSuccess);

  OperatorSpec rejected;
  Output<Tensor> rejected_output;
  rejected.output(rejected_output, "output")
      .produces_tensor(tensor_spec,
                       within_limit.maximum_with(parameter_byte_span(
                           param_ref<&OperatorSpecPathBranchParams::scalar6>())));

  // Key assertion: both typed paths count independently at the exact eight-dependency boundary.
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(rejected, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
}

// REQ: COMPILE-077
TEST(OperatorSpec, TensorByteSpanProjectionRejectsInvalidIdentityDependenciesAndCollisions) {
  const auto metadata_for = [](TensorByteSpanExpression expression) {
    OperatorSpec spec;
    Output<Tensor> output;
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            std::move(expression));
    return detail::OperatorSpecCompilerAccess::port(spec, 0U);
  };

  const auto empty_id = metadata_for(parameter_projected_byte_span<&operator_spec_projected_bytes>(
      "", param_ref<&OperatorSpecProjectionParams::width>()));
  EXPECT_EQ(empty_id.tensor_contract_error.message, "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID");

  const std::string oversized_id(kMaximumTensorByteSpanProjectionIdBytes + 1U, 'x');
  const auto oversized = metadata_for(parameter_projected_byte_span<&operator_spec_projected_bytes>(
      oversized_id, param_ref<&OperatorSpecProjectionParams::width>()));
  EXPECT_EQ(oversized.tensor_contract_error.message, "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID");

  const auto duplicate = metadata_for(parameter_projected_byte_span<&operator_spec_projected_bytes>(
      "nvidia.holoscan.test.duplicate.v1",
      param_ref<&OperatorSpecProjectionParams::width>(),
      param_ref<&OperatorSpecProjectionParams::width>()));
  EXPECT_EQ(duplicate.tensor_contract_error.message, "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID");

  const auto first = parameter_projected_byte_span<&operator_spec_projected_bytes>(
      "nvidia.holoscan.test.collision.v1", param_ref<&OperatorSpecProjectionParams::width>());
  const auto second = parameter_projected_byte_span<&operator_spec_alternate_projected_bytes>(
      "nvidia.holoscan.test.collision.v1", param_ref<&OperatorSpecProjectionParams::height>());
  const auto collision = metadata_for(first.maximum_with(second));
  EXPECT_EQ(collision.tensor_contract_error.message, "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID");
}

// REQ: COMPILE-067
TEST(OperatorSpec, TensorOutputRejectsConflictingOrMalformedDerivedByteSpanSources) {
  const auto derived_spec = [] {
    return TensorOutputSpec{
        .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
  };

  OperatorSpec literal;
  Output<Tensor> literal_output;
  auto literal_spec = derived_spec();
  literal_spec.bounds = tensor_bounds(4096U);
  // Key setup: mix an authored literal capacity with a derived capacity source.
  literal.output(literal_output, "output").produces_tensor(literal_spec, tensor_byte_span(4096U));
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(literal, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT");

  OperatorSpec pass_through;
  Output<Tensor> pass_through_output;
  auto pass_through_spec = derived_spec();
  pass_through_spec.storage = TensorOutputStorage::kPassThrough;
  pass_through.output(pass_through_output, "output")
      .produces_tensor(pass_through_spec, tensor_byte_span(4096U));
  EXPECT_EQ(
      detail::OperatorSpecCompilerAccess::port(pass_through, 0U).tensor_contract_error.message,
      "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT");

  const auto zero = tensor_byte_span(0U);
  const TensorByteSpanExpression empty;
  const auto bad_alignment = tensor_byte_span(1U).aligned_to(3U);
  OperatorSpec malformed;
  Output<Tensor> malformed_output;
  malformed.output(malformed_output, "output").produces_tensor(derived_spec(), bad_alignment);
  // Key assertion: malformed expressions fail during setup metadata normalization.
  EXPECT_EQ(empty.invalid_reason(), TensorByteSpanExpressionInvalidReason::kEmpty);
  EXPECT_EQ(zero.invalid_reason(), TensorByteSpanExpressionInvalidReason::kNonPositiveConstant);
  EXPECT_EQ(bad_alignment.invalid_reason(),
            TensorByteSpanExpressionInvalidReason::kAlignmentInvalid);
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(malformed, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
}

// REQ: COMPILE-067
TEST(OperatorSpec, TensorOutputRejectsASecondByteSpanSource) {
  OperatorSpec spec;
  Output<Tensor> output;
  const TensorOutputSpec derived{
      .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
      .storage = TensorOutputStorage::kRuntimePool,
  };

  // Key setup: select the derived source before attempting a literal declaration.
  spec.output(output, "output")
      .produces_tensor(derived, tensor_byte_span(4096U))
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(4096U),
          .storage = TensorOutputStorage::kRuntimePool,
      });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  // Key assertion: the first expression remains intact and the duplicate is rejected.
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT");
  EXPECT_TRUE(metadata.tensor_byte_span_expression);
  EXPECT_FALSE(metadata.tensor_allocation);
}

// REQ: COMPILE-067
TEST(OperatorSpec, TensorOutputRejectsExpressionBeyondTheDepthLimit) {
  OperatorSpec spec;
  Output<Tensor> output;
  auto expression = tensor_byte_span(1U);
  // Key setup: one more unary node than the frozen maximum depth is authored.
  for (std::size_t depth = 0U; depth < kMaximumTensorByteSpanExpressionDepth; ++depth) {
    expression = expression.aligned_to(2U);
  }
  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          expression);

  // Key assertion: the bounded expression fails during cold metadata normalization.
  EXPECT_EQ(expression.invalid_reason(),
            TensorByteSpanExpressionInvalidReason::kDepthLimitExceeded);
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
}

// REQ: COMPILE-067
TEST(OperatorSpec, TensorOutputRejectsExpressionBeyondTheNodeLimit) {
  auto expression = tensor_byte_span(1U);
  constexpr std::size_t kMaximumCompositionAttempts = 8U;
  // Key setup: balanced composition grows past the node bound before reaching the depth bound.
  for (std::size_t attempt = 0U;
       attempt < kMaximumCompositionAttempts &&
       expression.invalid_reason() == TensorByteSpanExpressionInvalidReason::kNone;
       ++attempt) {
    expression = expression.maximum_with(expression);
  }

  // Key assertion: finite artifact work is enforced by the distinct node-limit reason.
  EXPECT_EQ(expression.invalid_reason(), TensorByteSpanExpressionInvalidReason::kNodeLimitExceeded);
}

// REQ: COMPILE-067
TEST(OperatorSpec, TensorOutputRejectsExpressionBeyondTheDependencyLimit) {
  OperatorSpec spec;
  Input<Tensor> input0;
  Input<Tensor> input1;
  Input<Tensor> input2;
  Input<Tensor> input3;
  Input<Tensor> input4;
  Input<Tensor> input5;
  Input<Tensor> input6;
  Input<Tensor> input7;
  Input<Tensor> input8;
  Output<Tensor> output;
  spec.input(input0, "input0");
  spec.input(input1, "input1");
  spec.input(input2, "input2");
  spec.input(input3, "input3");
  spec.input(input4, "input4");
  spec.input(input5, "input5");
  spec.input(input6, "input6");
  spec.input(input7, "input7");
  spec.input(input8, "input8");

  // Key setup: combine nine distinct input authorities within the node and depth limits.
  auto expression = input_max_byte_span(input0)
                        .maximum_with(input_max_byte_span(input1))
                        .maximum_with(input_max_byte_span(input2))
                        .maximum_with(input_max_byte_span(input3))
                        .maximum_with(input_max_byte_span(input4))
                        .maximum_with(input_max_byte_span(input5))
                        .maximum_with(input_max_byte_span(input6))
                        .maximum_with(input_max_byte_span(input7))
                        .maximum_with(input_max_byte_span(input8));
  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          std::move(expression));

  // Key assertion: normalized dependency identity is capped independently of expression size.
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 9U).tensor_contract_error.message,
            "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
}

// REQ: COMPILE-067
TEST(OperatorSpec, DistinctNestedParameterPathsCountTowardDependencyLimit) {
  const TensorOutputSpec spec{
      .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
      .storage = TensorOutputStorage::kRuntimePool,
  };
  auto within_limit =
      parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::first,
                                    &OperatorSpecRepeatedLeafParams::bytes>())
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar0>()))
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar1>()))
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar2>()))
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar3>()))
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar4>()))
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar5>()))
          .maximum_with(
              parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::scalar6>()));

  OperatorSpec accepted;
  Output<Tensor> accepted_output;
  accepted.output(accepted_output, "output").produces_tensor(spec, within_limit);
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(accepted, 0U).tensor_contract_error.code,
            ErrorCode::kSuccess);

  auto beyond_limit = within_limit.maximum_with(
      parameter_byte_span(param_ref<&OperatorSpecDependencyLimitParams::second,
                                    &OperatorSpecRepeatedLeafParams::bytes>()));
  OperatorSpec rejected;
  Output<Tensor> rejected_output;
  rejected.output(rejected_output, "output").produces_tensor(spec, std::move(beyond_limit));

  // Key assertion: first.bytes and second.bytes share the nested leaf identity but remain distinct
  // dependencies because the full canonical paths name different owning aggregate members.
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(rejected, 0U).tensor_contract_error.message,
            "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
}

// REQ: COMPILE-067
TEST(OperatorSpec, NormalizedTensorByteSpanExpressionEqualityCoversEveryNodeField) {
  const auto metadata_for = [](std::uint64_t capacity) {
    OperatorSpec spec;
    Output<Tensor> output;
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            tensor_byte_span(capacity).aligned_to(256U));
    return detail::OperatorSpecCompilerAccess::port(spec, 0U).tensor_byte_span_expression;
  };

  // Key setup: normalize two equivalent expressions and one differing only in a node operand.
  const auto first = metadata_for(1000U);
  const auto equivalent = metadata_for(1000U);
  const auto distinct = metadata_for(1001U);

  // Key assertion: record equality is structural and observes the complete node payload.
  EXPECT_EQ(first, equivalent);
  EXPECT_NE(first, distinct);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(OperatorSpec, CommonOutputRejectsASecondTensorTransferSource) {
  OperatorSpec spec;
  Output<OperatorSpecSchemaPayload> output;
  PayloadOptions options;
  options.schema = schema_identity<OperatorSpecSchemaPayload>();
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  spec.output(output, "output")
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      })
      .payload_options(options);

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  ASSERT_TRUE(metadata.tensor_contract);
  ASSERT_TRUE(metadata.tensor_contract->transfer);
  EXPECT_EQ(metadata.tensor_contract->transfer->max_staging_byte_span, 2048U);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(OperatorSpec, CommonOutputRejectsTransferAuthoredBeforeTensorContract) {
  OperatorSpec spec;
  Output<OperatorSpecSchemaPayload> output;
  PayloadOptions options;
  options.schema = schema_identity<OperatorSpecSchemaPayload>();
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  spec.output(output, "output")
      .payload_options(options)
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  EXPECT_FALSE(metadata.tensor_contract);
  ASSERT_TRUE(metadata.payload_options);
  EXPECT_EQ(metadata.payload_options->max_tensor_byte_span, 1024U);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(OperatorSpec, SpecializedTensorOutputRejectsASecondTensorTransferSource) {
  OperatorSpec spec;
  Output<Tensor> output;
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  spec.output(output, "output")
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      })
      .payload_options(options);

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  ASSERT_TRUE(metadata.tensor_contract);
  ASSERT_TRUE(metadata.tensor_contract->transfer);
  EXPECT_EQ(metadata.tensor_contract->transfer->max_staging_byte_span, 2048U);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(OperatorSpec, SpecializedTensorOutputRejectsTransferAuthoredBeforeTensorContract) {
  OperatorSpec spec;
  Output<Tensor> output;
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  spec.output(output, "output")
      .payload_options(options)
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  EXPECT_FALSE(metadata.tensor_contract);
  ASSERT_TRUE(metadata.payload_options);
  EXPECT_EQ(metadata.payload_options->max_tensor_byte_span, 1024U);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
// REQ: COMPILE-067
TEST(OperatorSpec, DerivedTensorOutputRejectsTransferAuthoredBeforeTensorContract) {
  OperatorSpec spec;
  Output<Tensor> output;
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  // Key setup: select payload transfer authority before a derived-capacity declaration.
  spec.output(output, "output")
      .payload_options(options)
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          tensor_byte_span(2048U));

  // Key assertion: the derived overload preserves the first source and rejects the conflict.
  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  EXPECT_FALSE(metadata.tensor_contract);
  EXPECT_FALSE(metadata.tensor_byte_span_expression);
  ASSERT_TRUE(metadata.payload_options);
  EXPECT_EQ(metadata.payload_options->max_tensor_byte_span, 1024U);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsOnePortDeclarationBoundUnderDifferentNames) {
  ReusedPortDeclarationOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsEmptyReadinessSourceName) {
  EmptyReadinessNameOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsOneReadinessDeclarationBoundUnderDifferentNames) {
  ReusedReadinessDeclarationOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsDuplicateNamesAcrossReadinessSourceKinds) {
  DuplicateReadinessNameOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kDuplicateName);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsDuplicateNamesAcrossPortAndReadinessKinds) {
  DuplicatePortAndReadinessNameOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kDuplicateName);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, FreshSpecsAcceptTheSameStableDeclarations) {
  ValidDeclarationsOperator op;
  EXPECT_TRUE(declare_once(op));
  EXPECT_TRUE(declare_once(op));
}

// REQ: OPERATOR-004
TEST(OperatorSpec, AdmittedPayloadFamiliesRetainDistinctColdTypeMetadata) {
  // Key setup: single operator with three ports covering kInlineTyped, kSerializedBytes,
  // and kDescriptorBacked.  Each port's cold metadata must be independently addressable.
  TypedMetadataDeclarationsOperator op;
  OperatorSpec spec;
  op.setup(spec);

  // Key assertion: local_type_token, payload_kind, and schema_identity are all
  // stored per-port and must match the template argument of each Input<>/Output<>.
  ASSERT_EQ(detail::OperatorSpecTestAccess::port_count(spec), 3);
  const auto scalar = detail::OperatorSpecTestAccess::port(spec, 0);
  EXPECT_EQ(scalar.declaration, &op.scalar_);
  EXPECT_EQ(scalar.direction, PortDirection::kInput);
  EXPECT_EQ(scalar.name, "scalar");
  EXPECT_FALSE(scalar.tensor);
  EXPECT_EQ(scalar.builder_index, 0);
  EXPECT_EQ(scalar.local_type_token, &detail::kOperatorSpecLocalTypeToken<std::uint32_t>);
  EXPECT_EQ(scalar.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_FALSE(scalar.schema_identity.has_value());

  const auto schema = detail::OperatorSpecTestAccess::port(spec, 1);
  EXPECT_EQ(schema.declaration, &op.schema_);
  EXPECT_EQ(schema.direction, PortDirection::kOutput);
  EXPECT_EQ(schema.name, "schema");
  EXPECT_FALSE(schema.tensor);
  EXPECT_EQ(schema.builder_index, 0);
  EXPECT_EQ(schema.local_type_token,
            &detail::kOperatorSpecLocalTypeToken<OperatorSpecSchemaPayload>);
  EXPECT_EQ(schema.payload_kind, PayloadKind::kSerializedBytes);
  ASSERT_TRUE(schema.schema_identity.has_value());
  const SchemaIdentity schema_identity_value = schema.schema_identity.value_or(SchemaIdentity{});
  EXPECT_EQ(schema_identity_value.canonical_name, "holoscan.test.OperatorSpecSchemaPayload");
  EXPECT_EQ(schema_identity_value.compatibility_epoch, 7U);
  EXPECT_EQ(schema_identity_value.family_id,
            schema_identity<OperatorSpecSchemaPayload>().family_id);
  EXPECT_EQ(schema_identity_value.file_identifier,
            schema_identity<OperatorSpecSchemaPayload>().file_identifier);

  const auto tensor = detail::OperatorSpecTestAccess::port(spec, 2);
  EXPECT_EQ(tensor.declaration, &op.tensor_);
  EXPECT_EQ(tensor.direction, PortDirection::kOutput);
  EXPECT_EQ(tensor.name, "tensor");
  EXPECT_TRUE(tensor.tensor);
  EXPECT_EQ(tensor.builder_index, 0);
  EXPECT_EQ(tensor.local_type_token, &detail::kOperatorSpecLocalTypeToken<Tensor>);
  EXPECT_EQ(tensor.payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_FALSE(tensor.schema_identity.has_value());
}

// REQ: OPERATOR-006
TEST(OperatorSpec, InterleavedReadinessKindsRetainTheirExactBuilderFamilies) {
  InterleavedReadinessDeclarationsOperator op;
  OperatorSpec spec;
  op.setup(spec);

  using Family = detail::OperatorSpecTestAccess::SourceBuilderFamily;
  ASSERT_EQ(detail::OperatorSpecTestAccess::source_count(spec), 4);
  const auto readable = detail::OperatorSpecTestAccess::source(spec, 0);
  EXPECT_EQ(readable.declaration, &op.readable_);
  EXPECT_EQ(readable.name, "readable");
  EXPECT_EQ(readable.kind, ReadinessSourceKind::kReadable);
  EXPECT_EQ(readable.builder_family, Family::kSingleWitness);
  EXPECT_EQ(readable.builder_index, 0);

  const auto notification = detail::OperatorSpecTestAccess::source(spec, 1);
  EXPECT_EQ(notification.declaration, &op.notification_);
  EXPECT_EQ(notification.name, "notification");
  EXPECT_EQ(notification.kind, ReadinessSourceKind::kNotification);
  EXPECT_EQ(notification.builder_family, Family::kNotification);
  EXPECT_EQ(notification.builder_index, 0);

  const auto doorbell = detail::OperatorSpecTestAccess::source(spec, 2);
  EXPECT_EQ(doorbell.declaration, &op.doorbell_);
  EXPECT_EQ(doorbell.name, "doorbell");
  EXPECT_EQ(doorbell.kind, ReadinessSourceKind::kDoorbell);
  EXPECT_EQ(doorbell.builder_family, Family::kDoorbell);
  EXPECT_EQ(doorbell.builder_index, 0);

  const auto spin = detail::OperatorSpecTestAccess::source(spec, 3);
  EXPECT_EQ(spin.declaration, &op.spin_);
  EXPECT_EQ(spin.name, "spin");
  EXPECT_EQ(spin.kind, ReadinessSourceKind::kSpin);
  EXPECT_EQ(spin.builder_family, Family::kSingleWitness);
  EXPECT_EQ(spin.builder_index, 1);
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
