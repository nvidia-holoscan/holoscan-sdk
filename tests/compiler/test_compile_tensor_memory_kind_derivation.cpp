/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "compile_test_graph.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_generation.hpp"
#include "compiler/tensor_memory_kind_derivation.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_schema_test_access.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

enum class TensorMemorySelection : std::uint8_t {
  kHost = 1,
  kPinned = 4,
  kDevice = 9,
};

struct NestedTensorMemoryParams {
  MemoryKind memory_kind{MemoryKind::kPinnedHost};
};

HOLOSCAN_PARAM_SCHEMA(NestedTensorMemoryParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.NestedTensorMemoryParams", 1)),
                      holoscan::field<&NestedTensorMemoryParams::memory_kind>("memory_kind")
                          .enum_values({param_enum_value("host", MemoryKind::kHost),
                                        param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                                        param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                                        param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
                          .recompile_required());

// Supplies the sensitive intermediate object used by
// RejectsSensitiveNestedAncestorsBeforeReadingValue.
struct IntermediateTensorMemoryParams {
  NestedTensorMemoryParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(IntermediateTensorMemoryParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.IntermediateTensorMemoryParams", 1)),
                      holoscan::field<&IntermediateTensorMemoryParams::nested>("secret")
                          .nested<NestedTensorMemoryParams>()
                          .sensitive());

// Supplies the nested branch of a root-level dotted-path collision.
// Used by PreservesTypedMemberIdentityForCollidingDisplayPaths.
struct TensorMemoryCaptureParams {
  NestedTensorMemoryParams settings{};
};

HOLOSCAN_PARAM_SCHEMA(TensorMemoryCaptureParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.TensorMemoryCaptureParams", 1)),
                      holoscan::field<&TensorMemoryCaptureParams::settings>("settings")
                          .nested<NestedTensorMemoryParams>());

// Supplies the extra nested segment of a shared-root dotted-path collision.
// Used by PreservesTypedMemberIdentityForCollidingDisplayPaths.
struct TensorMemoryDetailParams {
  NestedTensorMemoryParams detail{};
};

HOLOSCAN_PARAM_SCHEMA(TensorMemoryDetailParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.TensorMemoryDetailParams", 1)),
                      holoscan::field<&TensorMemoryDetailParams::detail>("detail")
                          .nested<NestedTensorMemoryParams>());

// Supplies two shared-root branches with the same canonical display path.
// Used by PreservesTypedMemberIdentityForCollidingDisplayPaths.
struct TensorMemorySharedParams {
  NestedTensorMemoryParams dotted_settings{};
  TensorMemoryDetailParams settings{};
};

HOLOSCAN_PARAM_SCHEMA(TensorMemorySharedParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.TensorMemorySharedParams", 1)),
                      holoscan::field<&TensorMemorySharedParams::dotted_settings>("settings.detail")
                          .nested<NestedTensorMemoryParams>(),
                      holoscan::field<&TensorMemorySharedParams::settings>("settings")
                          .nested<TensorMemoryDetailParams>());

// Owns both collision shapes exercised by the normalized dependency identity tests.
// Used by PreservesTypedMemberIdentityForCollidingDisplayPaths.
struct CollidingTensorMemoryPathParams {
  NestedTensorMemoryParams dotted_capture{};
  TensorMemoryCaptureParams capture{};
  TensorMemorySharedParams shared{};
};

HOLOSCAN_PARAM_SCHEMA(
    CollidingTensorMemoryPathParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.CollidingTensorMemoryPathParams", 1)),
    holoscan::field<&CollidingTensorMemoryPathParams::dotted_capture>("capture.settings")
        .nested<NestedTensorMemoryParams>(),
    holoscan::field<&CollidingTensorMemoryPathParams::capture>("capture")
        .nested<TensorMemoryCaptureParams>(),
    holoscan::field<&CollidingTensorMemoryPathParams::shared>("shared")
        .nested<TensorMemorySharedParams>());

struct TensorMemoryKindTestParams {
  MemoryKind memory_kind{MemoryKind::kHost};
  TensorMemorySelection selection{TensorMemorySelection::kHost};
  bool pinned{};
  NestedTensorMemoryParams nested{};
  NestedTensorMemoryParams sensitive_nested{};
  IntermediateTensorMemoryParams intermediate{};
  std::uint64_t capture_sibling{1U};
  MemoryKind runtime_memory_kind{MemoryKind::kHost};
  MemoryKind sensitive_memory_kind{MemoryKind::kHost};
  std::uint64_t bytes{4096U};
};

HOLOSCAN_PARAM_SCHEMA(
    TensorMemoryKindTestParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.TensorMemoryKindTestParams", 1)),
    holoscan::field<&TensorMemoryKindTestParams::memory_kind>("memory_kind")
        .enum_values({param_enum_value("host", MemoryKind::kHost),
                      param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                      param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                      param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
        .recompile_required(),
    holoscan::field<&TensorMemoryKindTestParams::selection>("selection")
        .enum_values({param_enum_value("host", TensorMemorySelection::kHost),
                      param_enum_value("pinned", TensorMemorySelection::kPinned),
                      param_enum_value("device", TensorMemorySelection::kDevice)})
        .recompile_required(),
    holoscan::field<&TensorMemoryKindTestParams::pinned>("pinned").recompile_required(),
    holoscan::field<&TensorMemoryKindTestParams::nested>("capture.settings")
        .nested<NestedTensorMemoryParams>(),
    holoscan::field<&TensorMemoryKindTestParams::sensitive_nested>("sensitive_nested")
        .nested<NestedTensorMemoryParams>()
        .sensitive(),
    holoscan::field<&TensorMemoryKindTestParams::intermediate>("intermediate")
        .nested<IntermediateTensorMemoryParams>(),
    holoscan::field<&TensorMemoryKindTestParams::capture_sibling>("capture"),
    holoscan::field<&TensorMemoryKindTestParams::runtime_memory_kind>("runtime_memory_kind")
        .enum_values({param_enum_value("host", MemoryKind::kHost),
                      param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                      param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                      param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
        .runtime_mutable(),
    holoscan::field<&TensorMemoryKindTestParams::sensitive_memory_kind>("sensitive_memory_kind")
        .enum_values({param_enum_value("host", MemoryKind::kHost),
                      param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                      param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                      param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
        .recompile_required()
        .sensitive(),
    holoscan::field<&TensorMemoryKindTestParams::bytes>("bytes").recompile_required());

struct MemoryKindMultiTensorSchema {
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> calibration;
};

template <>
consteval SchemaIdentity schema_identity<MemoryKindMultiTensorSchema>() {
  return schema_identity<schema::ImageT>();
}

template <>
struct DescriptorTensorSchemaNativeTraits<MemoryKindMultiTensorSchema> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, 2U> tensor_field_ids{1U, 7U};

  [[nodiscard]] static const Tensor* tensor_at(const MemoryKindMultiTensorSchema& value,
                                               std::size_t index) noexcept {
    return index == 0U ? value.image.get() : (index == 1U ? value.calibration.get() : nullptr);
  }
};

namespace detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<MemoryKindMultiTensorSchema>() noexcept {
  static const RuntimePortTypeOps ops = schema::make_descriptor_tensor_port_type_ops<
      MemoryKindMultiTensorSchema,
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>(
      [](void*) noexcept -> expected<void, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*) noexcept {},
      [](const MemoryKindMultiTensorSchema&,
         schema::DescriptorTensorReferences,
         std::span<std::byte>) noexcept -> expected<std::size_t, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*, const schema::Image&, std::span<Tensor>, SampleMetadata) noexcept
          -> expected<void, Error> { return make_unexpected(Error{ErrorCode::kNotImplemented}); });
  return &ops;
}

}  // namespace detail

namespace {

enum class MemoryExpressionCase : std::uint8_t {
  kDirect,
  kMappedEnum,
  kMappedBoolean,
  kNested,
  kSensitiveNestedRoot,
  kSensitiveNestedIntermediate,
  kRuntimeMutable,
  kSensitive,
};

enum class CollidingMemoryPath : std::uint8_t {
  kDottedCapture,
  kNestedCapture,
  kSharedDottedSettings,
  kSharedNestedSettings,
};

// Selects structurally distinct typed paths with colliding canonical display strings.
// Used by PreservesTypedMemberIdentityForCollidingDisplayPaths and
// RejectsCollidingTypedPathChangesDuringReplay.
class CollidingMemoryPathTensorSource final : public Operator<CollidingTensorMemoryPathParams> {
 public:
  static inline bool use_structural_twin{};

  explicit CollidingMemoryPathTensorSource(CollidingMemoryPath path) : path_(path) {}

  void setup(OperatorSpec& spec) override {
    const CollidingMemoryPath effective_path = use_structural_twin ? structural_twin(path_) : path_;
    TensorMemoryKindExpression memory;
    switch (effective_path) {
      case CollidingMemoryPath::kDottedCapture:
        memory = parameter_memory_kind(param_ref<&CollidingTensorMemoryPathParams::dotted_capture,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case CollidingMemoryPath::kNestedCapture:
        memory = parameter_memory_kind(param_ref<&CollidingTensorMemoryPathParams::capture,
                                                 &TensorMemoryCaptureParams::settings,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case CollidingMemoryPath::kSharedDottedSettings:
        memory = parameter_memory_kind(param_ref<&CollidingTensorMemoryPathParams::shared,
                                                 &TensorMemorySharedParams::dotted_settings,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case CollidingMemoryPath::kSharedNestedSettings:
        memory = parameter_memory_kind(param_ref<&CollidingTensorMemoryPathParams::shared,
                                                 &TensorMemorySharedParams::settings,
                                                 &TensorMemoryDetailParams::detail,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
                .bounds = tensor_bounds(4096U),
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{.memory_kind = std::move(memory)});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  [[nodiscard]] static constexpr CollidingMemoryPath structural_twin(
      CollidingMemoryPath path) noexcept {
    switch (path) {
      case CollidingMemoryPath::kDottedCapture:
        return CollidingMemoryPath::kNestedCapture;
      case CollidingMemoryPath::kNestedCapture:
        return CollidingMemoryPath::kDottedCapture;
      case CollidingMemoryPath::kSharedDottedSettings:
        return CollidingMemoryPath::kSharedNestedSettings;
      case CollidingMemoryPath::kSharedNestedSettings:
        return CollidingMemoryPath::kSharedDottedSettings;
    }
    return path;
  }

  CollidingMemoryPath path_;
};

class DerivedMemoryTensorSource final : public Operator<TensorMemoryKindTestParams> {
 public:
  explicit DerivedMemoryTensorSource(
      MemoryExpressionCase expression = MemoryExpressionCase::kDirect, bool reverse_mapping = false,
      bool change_host_target = false)
      : expression_(expression),
        reverse_mapping_(reverse_mapping),
        change_host_target_(change_host_target) {}

  void setup(OperatorSpec& spec) override {
    TensorMemoryKindExpression memory;
    switch (expression_) {
      case MemoryExpressionCase::kDirect:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::memory_kind>());
        break;
      case MemoryExpressionCase::kMappedEnum: {
        const auto host_target = change_host_target_ ? MemoryKind::kPinnedHost : MemoryKind::kHost;
        memory =
            reverse_mapping_
                ? parameter_memory_kind(
                      param_ref<&TensorMemoryKindTestParams::selection>(),
                      {memory_kind_case(TensorMemorySelection::kDevice, MemoryKind::kCudaDevice),
                       memory_kind_case(TensorMemorySelection::kPinned, MemoryKind::kPinnedHost),
                       memory_kind_case(TensorMemorySelection::kHost, host_target)})
                : parameter_memory_kind(
                      param_ref<&TensorMemoryKindTestParams::selection>(),
                      {memory_kind_case(TensorMemorySelection::kHost, host_target),
                       memory_kind_case(TensorMemorySelection::kPinned, MemoryKind::kPinnedHost),
                       memory_kind_case(TensorMemorySelection::kDevice, MemoryKind::kCudaDevice)});
        break;
      }
      case MemoryExpressionCase::kMappedBoolean:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::pinned>(),
                                       {memory_kind_case(false, MemoryKind::kHost),
                                        memory_kind_case(true, MemoryKind::kPinnedHost)});
        break;
      case MemoryExpressionCase::kNested:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::nested,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case MemoryExpressionCase::kSensitiveNestedRoot:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::sensitive_nested,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case MemoryExpressionCase::kSensitiveNestedIntermediate:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::intermediate,
                                                 &IntermediateTensorMemoryParams::nested,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case MemoryExpressionCase::kRuntimeMutable:
        memory =
            parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::runtime_memory_kind>());
        break;
      case MemoryExpressionCase::kSensitive:
        memory =
            parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::sensitive_memory_kind>());
        break;
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
                .bounds = tensor_bounds(4096U),
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{.memory_kind = std::move(memory)});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  MemoryExpressionCase expression_;
  bool reverse_mapping_{};
  bool change_host_target_{};
};

class LiteralMemoryTensorSource final : public Operator<> {
 public:
  explicit LiteralMemoryTensorSource(MemoryKind memory_kind) : memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = memory_kind_,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 2U},
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  MemoryKind memory_kind_;
};

class ExactMemoryTensorSink final : public Operator<> {
 public:
  explicit ExactMemoryTensorSink(MemoryKind memory_kind) : memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = memory_kind_},
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;

 private:
  MemoryKind memory_kind_;
};

class DerivedMemoryImageSource final : public Operator<TensorMemoryKindTestParams> {
 public:
  explicit DerivedMemoryImageSource(MemoryExpressionCase expression = MemoryExpressionCase::kDirect)
      : expression_(expression) {}

  void setup(OperatorSpec& spec) override {
    constexpr TensorFieldSelector<schema::ImageT> kImage{4U, 0U};
    TensorMemoryKindExpression memory;
    switch (expression_) {
      case MemoryExpressionCase::kSensitiveNestedRoot:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::sensitive_nested,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      case MemoryExpressionCase::kSensitiveNestedIntermediate:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::intermediate,
                                                 &IntermediateTensorMemoryParams::nested,
                                                 &NestedTensorMemoryParams::memory_kind>());
        break;
      default:
        memory = parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::memory_kind>());
        break;
    }
    spec.output(output, "output")
        .tensor_field(kImage)
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
                .bounds = tensor_bounds(2048U),
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{
                .memory_kind = std::move(memory),
            });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<schema::ImageT> output;

 private:
  MemoryExpressionCase expression_;
};

class ExactMemoryImageSink final : public Operator<> {
 public:
  explicit ExactMemoryImageSink(MemoryKind memory_kind) : memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    constexpr TensorFieldSelector<schema::ImageT> kImage{4U, 0U};
    spec.input(input, "input")
        .tensor_field(kImage)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = memory_kind_},
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;

 private:
  MemoryKind memory_kind_;
};

class CombinedDerivedTensorSource final : public Operator<TensorMemoryKindTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{
                .max_byte_span =
                    parameter_byte_span(param_ref<&TensorMemoryKindTestParams::bytes>()),
                .memory_kind =
                    parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::memory_kind>()),
            });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class DerivedMemoryMultiTensorSource final : public Operator<TensorMemoryKindTestParams> {
 public:
  static constexpr TensorFieldSelector<MemoryKindMultiTensorSchema> kImage{1U, 0U};
  static constexpr TensorFieldSelector<MemoryKindMultiTensorSchema> kCalibration{7U, 1U};

  void setup(OperatorSpec& spec) override {
    auto& builder = spec.output(output, "output");
    builder.tensor_field(kImage).produces_tensor(
        TensorOutputSpec{
            .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
            .bounds = tensor_bounds(2048U),
            .storage = TensorOutputStorage::kRuntimePool,
        },
        TensorOutputDerivations{
            .memory_kind =
                parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::memory_kind>()),
        });
    builder.tensor_field(kCalibration)
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.dtype = DLDataType{kDLFloat, 32U, 1U}, .rank = 1U},
                .bounds = tensor_bounds(1024U),
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{
                .memory_kind =
                    parameter_memory_kind(param_ref<&TensorMemoryKindTestParams::nested,
                                                    &NestedTensorMemoryParams::memory_kind>()),
            });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<MemoryKindMultiTensorSchema> output;
};

[[nodiscard]] const TensorAllocationPlanRecord& only_allocation(const ExecutionPlan& plan) {
  if (plan.tensor_allocations().size() != 1U) {
    throw std::logic_error{"expected one Tensor allocation"};
  }
  return plan.tensor_allocations().front();
}

[[nodiscard]] ExecutionPlan compile_derived(MemoryExpressionCase expression,
                                            TensorMemoryKindTestParams params = {},
                                            bool reverse_mapping = false,
                                            bool change_host_target = false) {
  Graph graph{"derived-memory-kind"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<DerivedMemoryTensorSource>(
      "source", expression, reverse_mapping, change_host_target);
  source.set_params(std::move(params));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan compile_literal(MemoryKind memory_kind) {
  Graph graph{"literal-memory-kind"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<LiteralMemoryTensorSource>("source", memory_kind);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan compile_colliding_path(CollidingMemoryPath path,
                                                   CollidingTensorMemoryPathParams params) {
  Graph graph{"colliding-memory-kind-path"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<CollidingMemoryPathTensorSource>("source", path);
  source.set_params(std::move(params));
  return compile(graph);
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, CopiesPhysicalSnapshotOnlyWhenDerivationExists) {
  Graph literal_graph{"literal-memory-kind-snapshot"};
  test::add_compile_test_default_clock(literal_graph);
  (void)literal_graph.op<LiteralMemoryTensorSource>("source", MemoryKind::kHost);
  const auto literal_snapshot = detail::GraphCompilerAccess::canonical_snapshot(literal_graph);
  ASSERT_NE(literal_snapshot, nullptr);
  EXPECT_FALSE(detail::has_tensor_memory_kind_derivations(*literal_snapshot));

  Graph derived_graph{"derived-memory-kind-snapshot"};
  test::add_compile_test_default_clock(derived_graph);
  (void)derived_graph.op<DerivedMemoryTensorSource>("source");
  const auto derived_snapshot = detail::GraphCompilerAccess::canonical_snapshot(derived_graph);
  ASSERT_NE(derived_snapshot, nullptr);
  EXPECT_TRUE(detail::has_tensor_memory_kind_derivations(*derived_snapshot));
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, ResolvesDirectGenerationZeroValueIntoPhysicalPlan) {
  const ExecutionPlan host_derived = compile_derived(MemoryExpressionCase::kDirect);
  const ExecutionPlan host_literal = compile_literal(MemoryKind::kHost);
  ASSERT_TRUE(host_derived.ok()) << host_derived.json();
  ASSERT_TRUE(host_literal.ok()) << host_literal.json();
  EXPECT_EQ(only_allocation(host_derived).memory_kind, only_allocation(host_literal).memory_kind);
  EXPECT_EQ(only_allocation(host_derived).bounds.max_rank,
            only_allocation(host_literal).bounds.max_rank);
  EXPECT_EQ(only_allocation(host_derived).bounds.max_byte_span,
            only_allocation(host_literal).bounds.max_byte_span);

  TensorMemoryKindTestParams params;
  params.memory_kind = MemoryKind::kPinnedHost;

  const ExecutionPlan derived = compile_derived(MemoryExpressionCase::kDirect, params);
  const ExecutionPlan literal = compile_literal(MemoryKind::kPinnedHost);

  ASSERT_TRUE(derived.ok()) << derived.json();
  ASSERT_TRUE(literal.ok()) << literal.json();
  const auto& allocation = only_allocation(derived);
  EXPECT_EQ(allocation.memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(allocation.memory_kind, only_allocation(literal).memory_kind);
  EXPECT_EQ(allocation.bounds.max_byte_span, only_allocation(literal).bounds.max_byte_span);
  ASSERT_EQ(derived.tensor_memory_kind_derivations().size(), 1U);
  const auto& evidence = derived.tensor_memory_kind_derivations().front();
  EXPECT_EQ(evidence.output, allocation.output);
  EXPECT_EQ(evidence.resolved_memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(evidence.form, TensorMemoryKindExpressionForm::kDirect);
  EXPECT_EQ(evidence.parameter_path, "memory_kind");
  EXPECT_EQ(evidence.update_disposition,
            TensorMemoryKindDerivationUpdateDisposition::kRecompileRequired);
  EXPECT_EQ(evidence.origin, TensorContractValueOrigin::kCompilerDerivedMemoryKind);
  EXPECT_EQ(derived.find_tensor_memory_kind_derivation("source", "output"), &evidence);
  EXPECT_EQ(derived.find_tensor_memory_kind_derivation("missing", "output"), nullptr);
  EXPECT_NE(derived.json().find("compiler_derived_memory_kind"), std::string_view::npos);
  EXPECT_NE(derived.json().find("tensor_memory_kind_derivations"), std::string_view::npos);
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, ResolvesEnumBooleanAndNestedForms) {
  TensorMemoryKindTestParams enum_params;
  enum_params.selection = TensorMemorySelection::kPinned;
  const ExecutionPlan enum_plan =
      compile_derived(MemoryExpressionCase::kMappedEnum, enum_params, true);
  ASSERT_TRUE(enum_plan.ok()) << enum_plan.json();
  EXPECT_EQ(only_allocation(enum_plan).memory_kind, MemoryKind::kPinnedHost);
  ASSERT_EQ(enum_plan.tensor_memory_kind_derivations().size(), 1U);
  ASSERT_EQ(enum_plan.tensor_memory_kind_derivations().front().mapping.size(), 3U);

  TensorMemoryKindTestParams bool_params;
  bool_params.pinned = true;
  const ExecutionPlan bool_plan =
      compile_derived(MemoryExpressionCase::kMappedBoolean, bool_params);
  ASSERT_TRUE(bool_plan.ok()) << bool_plan.json();
  EXPECT_EQ(only_allocation(bool_plan).memory_kind, MemoryKind::kPinnedHost);

  TensorMemoryKindTestParams nested_params;
  nested_params.nested.memory_kind = MemoryKind::kHost;
  const ExecutionPlan nested_plan = compile_derived(MemoryExpressionCase::kNested, nested_params);
  ASSERT_TRUE(nested_plan.ok()) << nested_plan.json();
  EXPECT_EQ(only_allocation(nested_plan).memory_kind, MemoryKind::kHost);
  ASSERT_EQ(nested_plan.tensor_memory_kind_derivations().size(), 1U);
  EXPECT_EQ(nested_plan.tensor_memory_kind_derivations().front().parameter_path,
            "capture.settings.memory_kind");
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, RejectsSensitiveNestedAncestorsBeforeReadingValue) {
  for (const auto expression : {MemoryExpressionCase::kSensitiveNestedRoot,
                                MemoryExpressionCase::kSensitiveNestedIntermediate}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(expression));

    // Key setup: Compile raw and descriptor outputs backed by the same sensitive nested path.
    const ExecutionPlan raw_plan = compile_derived(expression);
    // Graph:
    //   source (descriptor Tensor output)
    Graph descriptor_graph{"sensitive-nested-memory-kind-field"};
    test::add_compile_test_default_clock(descriptor_graph);
    (void)descriptor_graph.op<DerivedMemoryImageSource>("source", expression);
    const ExecutionPlan descriptor_plan = compile(descriptor_graph);

    // Key assertion: Ancestor sensitivity rejects both forms before exposing derived evidence.
    for (const ExecutionPlan* plan : {&raw_plan, &descriptor_plan}) {
      ASSERT_FALSE(plan->ok()) << plan->json();
      EXPECT_TRUE(test::has_diagnostic(*plan, "PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"))
          << plan->json();
      EXPECT_TRUE(plan->tensor_memory_kind_derivations().empty());
      EXPECT_TRUE(plan->tensor_allocations().empty());
      EXPECT_TRUE(plan->tensor_field_allocations().empty());
    }
  }
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, PreservesDeclarationIdentityAcrossValuesAndMappingOrder) {
  const auto graph_identity =
      [](TensorMemoryKindTestParams params, bool reverse_mapping, bool change_host_target) {
        Graph graph{"memory-kind-identity"};
        test::add_compile_test_default_clock(graph);
        auto source = graph.op<DerivedMemoryTensorSource>(
            "source", MemoryExpressionCase::kMappedEnum, reverse_mapping, change_host_target);
        source.set_params(std::move(params));
        return graph.view().identity();
      };
  TensorMemoryKindTestParams host;
  host.selection = TensorMemorySelection::kHost;
  TensorMemoryKindTestParams pinned;
  pinned.selection = TensorMemorySelection::kPinned;

  const auto ordered = graph_identity(host, false, false);
  const auto reversed = graph_identity(pinned, true, false);
  const auto changed_target = graph_identity(host, false, true);

  EXPECT_EQ(ordered.hash, reversed.hash);
  EXPECT_NE(ordered.hash, changed_target.hash);

  const ExecutionPlan host_plan = compile_derived(MemoryExpressionCase::kMappedEnum, host);
  const ExecutionPlan pinned_plan = compile_derived(MemoryExpressionCase::kMappedEnum, pinned);
  ASSERT_TRUE(host_plan.ok()) << host_plan.json();
  ASSERT_TRUE(pinned_plan.ok()) << pinned_plan.json();
  EXPECT_EQ(only_allocation(host_plan).memory_kind, MemoryKind::kHost);
  EXPECT_EQ(only_allocation(pinned_plan).memory_kind, MemoryKind::kPinnedHost);
  EXPECT_NE(host_plan.digest(), pinned_plan.digest());
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, PreservesTypedMemberIdentityForCollidingDisplayPaths) {
  // Graph topology for each case:
  //   source (isolated derived Tensor output)
  const auto graph_identity = [](CollidingMemoryPath path) {
    Graph graph{"colliding-memory-kind-identity"};
    test::add_compile_test_default_clock(graph);
    (void)graph.op<CollidingMemoryPathTensorSource>("source", path);
    return graph.view().identity();
  };

  // Key setup: pair dotted names with structurally nested paths at the root and below one root.
  CollidingMemoryPathTensorSource::use_structural_twin = false;
  const auto dotted_capture_identity = graph_identity(CollidingMemoryPath::kDottedCapture);
  const auto nested_capture_identity = graph_identity(CollidingMemoryPath::kNestedCapture);
  const auto shared_dotted_identity = graph_identity(CollidingMemoryPath::kSharedDottedSettings);
  const auto shared_nested_identity = graph_identity(CollidingMemoryPath::kSharedNestedSettings);

  CollidingTensorMemoryPathParams params;
  params.dotted_capture.memory_kind = MemoryKind::kHost;
  params.capture.settings.memory_kind = MemoryKind::kPinnedHost;
  params.shared.dotted_settings.memory_kind = MemoryKind::kHost;
  params.shared.settings.detail.memory_kind = MemoryKind::kPinnedHost;
  const ExecutionPlan dotted_capture =
      compile_colliding_path(CollidingMemoryPath::kDottedCapture, params);
  const ExecutionPlan nested_capture =
      compile_colliding_path(CollidingMemoryPath::kNestedCapture, params);
  const ExecutionPlan shared_dotted =
      compile_colliding_path(CollidingMemoryPath::kSharedDottedSettings, params);
  const ExecutionPlan shared_nested =
      compile_colliding_path(CollidingMemoryPath::kSharedNestedSettings, params);

  ASSERT_TRUE(dotted_capture.ok()) << dotted_capture.json();
  ASSERT_TRUE(nested_capture.ok()) << nested_capture.json();
  ASSERT_TRUE(shared_dotted.ok()) << shared_dotted.json();
  ASSERT_TRUE(shared_nested.ok()) << shared_nested.json();
  ASSERT_EQ(dotted_capture.tensor_memory_kind_derivations().size(), 1U);
  ASSERT_EQ(nested_capture.tensor_memory_kind_derivations().size(), 1U);
  ASSERT_EQ(shared_dotted.tensor_memory_kind_derivations().size(), 1U);
  ASSERT_EQ(shared_nested.tensor_memory_kind_derivations().size(), 1U);
  const auto& dotted_capture_evidence = dotted_capture.tensor_memory_kind_derivations().front();
  const auto& nested_capture_evidence = nested_capture.tensor_memory_kind_derivations().front();
  const auto& shared_dotted_evidence = shared_dotted.tensor_memory_kind_derivations().front();
  const auto& shared_nested_evidence = shared_nested.tensor_memory_kind_derivations().front();

  // Key assertion: every typed path remains distinct despite identical display paths and leaves.
  EXPECT_NE(dotted_capture_identity.hash, nested_capture_identity.hash);
  EXPECT_NE(shared_dotted_identity.hash, shared_nested_identity.hash);
  EXPECT_EQ(dotted_capture_evidence.parameter_path, nested_capture_evidence.parameter_path);
  EXPECT_NE(dotted_capture_evidence.parameter_member_full_ids,
            nested_capture_evidence.parameter_member_full_ids);
  EXPECT_EQ(shared_dotted_evidence.parameter_path, shared_nested_evidence.parameter_path);
  EXPECT_NE(shared_dotted_evidence.parameter_member_full_ids,
            shared_nested_evidence.parameter_member_full_ids);
  EXPECT_EQ(dotted_capture_evidence.parameter_member_full_ids.size(), 2U);
  EXPECT_EQ(nested_capture_evidence.parameter_member_full_ids.size(), 3U);
  EXPECT_EQ(shared_dotted_evidence.parameter_member_full_ids.size(), 3U);
  EXPECT_EQ(shared_nested_evidence.parameter_member_full_ids.size(), 4U);
  EXPECT_EQ(only_allocation(dotted_capture).memory_kind, MemoryKind::kHost);
  EXPECT_EQ(only_allocation(nested_capture).memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(only_allocation(shared_dotted).memory_kind, MemoryKind::kHost);
  EXPECT_EQ(only_allocation(shared_nested).memory_kind, MemoryKind::kPinnedHost);
  EXPECT_NE(dotted_capture.json().find("parameter_member_full_ids"), std::string_view::npos);
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, RejectsCollidingTypedPathChangesDuringReplay) {
  for (const auto initial_path :
       {CollidingMemoryPath::kDottedCapture, CollidingMemoryPath::kSharedDottedSettings}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(initial_path));
    // Graph topology:
    //   source (isolated derived Tensor output)
    Graph graph{"colliding-memory-kind-replay"};
    test::add_compile_test_default_clock(graph);
    CollidingMemoryPathTensorSource::use_structural_twin = false;
    auto source = graph.op<CollidingMemoryPathTensorSource>("source", initial_path);
    CollidingTensorMemoryPathParams params;
    params.dotted_capture.memory_kind = MemoryKind::kHost;
    params.capture.settings.memory_kind = MemoryKind::kPinnedHost;
    params.shared.dotted_settings.memory_kind = MemoryKind::kHost;
    params.shared.settings.detail.memory_kind = MemoryKind::kPinnedHost;
    source.set_params(std::move(params));

    // Key setup: replay selects a structural twin with the same canonical display path.
    CollidingMemoryPathTensorSource::use_structural_twin = true;
    const ExecutionPlan plan = compile(graph);
    CollidingMemoryPathTensorSource::use_structural_twin = false;

    // Key assertion: exact normalized record comparison rejects the changed typed dependency.
    ASSERT_FALSE(plan.ok()) << plan.json();
    EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"))
        << plan.json();
    EXPECT_TRUE(plan.tensor_memory_kind_derivations().empty());
    EXPECT_TRUE(plan.tensor_allocations().empty());
  }
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, ResolvesDescriptorTensorFieldWithoutRuntimeExpression) {
  Graph graph{"derived-memory-kind-field"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<DerivedMemoryImageSource>("source");
  TensorMemoryKindTestParams params;
  params.memory_kind = MemoryKind::kPinnedHost;
  source.set_params(params);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_memory_kind_derivations().size(), 1U);
  const auto& evidence = plan.tensor_memory_kind_derivations().front();
  EXPECT_EQ(evidence.field_id, 4U);
  EXPECT_EQ(evidence.tensor_index, 0U);
  EXPECT_EQ(evidence.resolved_memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(plan.find_tensor_memory_kind_derivation("source", "output", 4U, 0U), &evidence);
  EXPECT_EQ(plan.find_tensor_memory_kind_derivation("source", "output"), &evidence);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().memory_kind, MemoryKind::kPinnedHost);
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, ResolvesCombinedFactsAndDescriptorFieldsIndependently) {
  Graph combined_graph{"combined-derived-tensor-facts"};
  test::add_compile_test_default_clock(combined_graph);
  auto combined = combined_graph.op<CombinedDerivedTensorSource>("source");
  TensorMemoryKindTestParams combined_params;
  combined_params.memory_kind = MemoryKind::kPinnedHost;
  combined_params.bytes = 8192U;
  combined.set_params(combined_params);

  const ExecutionPlan combined_plan = compile(combined_graph);
  ASSERT_TRUE(combined_plan.ok()) << combined_plan.json();
  EXPECT_EQ(only_allocation(combined_plan).memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(only_allocation(combined_plan).bounds.max_byte_span, 8192U);
  EXPECT_EQ(combined_plan.tensor_memory_kind_derivations().size(), 1U);
  EXPECT_EQ(combined_plan.tensor_byte_span_derivations().size(), 1U);

  Graph fields_graph{"independent-derived-memory-fields"};
  test::add_compile_test_default_clock(fields_graph);
  auto fields = fields_graph.op<DerivedMemoryMultiTensorSource>("source");
  TensorMemoryKindTestParams field_params;
  field_params.memory_kind = MemoryKind::kHost;
  field_params.nested.memory_kind = MemoryKind::kPinnedHost;
  fields.set_params(field_params);

  const ExecutionPlan fields_plan = compile(fields_graph);
  ASSERT_TRUE(fields_plan.ok()) << fields_plan.json();
  ASSERT_EQ(fields_plan.tensor_field_allocations().size(), 2U);
  EXPECT_EQ(fields_plan.tensor_field_allocations()[0].field_id, 1U);
  EXPECT_EQ(fields_plan.tensor_field_allocations()[0].memory_kind, MemoryKind::kHost);
  EXPECT_EQ(fields_plan.tensor_field_allocations()[1].field_id, 7U);
  EXPECT_EQ(fields_plan.tensor_field_allocations()[1].memory_kind, MemoryKind::kPinnedHost);
  ASSERT_EQ(fields_plan.tensor_memory_kind_derivations().size(), 2U);
  EXPECT_EQ(fields_plan.tensor_memory_kind_derivations()[0].parameter_path, "memory_kind");
  EXPECT_EQ(fields_plan.tensor_memory_kind_derivations()[1].parameter_path,
            "capture.settings.memory_kind");
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, RejectsUnknownAndRuntimeMutableSourcesWithoutFallback) {
  Graph unknown_graph{"unknown-derived-memory-kind"};
  auto unknown_source = unknown_graph.op<DerivedMemoryTensorSource>("source");
  TensorMemoryKindTestParams unknown;
  unknown.memory_kind = MemoryKind::kUnknown;
  try {
    unknown_source.set_params(unknown);
    FAIL() << "expected the direct MemoryKind schema domain to reject kUnknown";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    const Error rejected = make_error(error);
    EXPECT_TRUE(std::string_view{rejected.message}.starts_with(
        "PARAM_VALIDATION_FAILED category=not_allowed"));
  }

  const ExecutionPlan runtime_mutable = compile_derived(MemoryExpressionCase::kRuntimeMutable);
  ASSERT_FALSE(runtime_mutable.ok());
  EXPECT_TRUE(test::has_diagnostic(runtime_mutable,
                                   "PHYSICAL_TENSOR_MEMORY_KIND_PARAMETER_MUTABILITY_INVALID"));
  EXPECT_TRUE(runtime_mutable.tensor_allocations().empty());

  const ExecutionPlan sensitive = compile_derived(MemoryExpressionCase::kSensitive);
  ASSERT_FALSE(sensitive.ok());
  EXPECT_TRUE(test::has_diagnostic(sensitive, "PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"));
  EXPECT_TRUE(sensitive.tensor_allocations().empty());
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, PreservesExistingCudaDeviceBindingFailure) {
  for (const MemoryKind memory_kind : {MemoryKind::kCudaDevice, MemoryKind::kCudaManaged}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(memory_kind));
    TensorMemoryKindTestParams params;
    params.memory_kind = memory_kind;

    const ExecutionPlan derived = compile_derived(MemoryExpressionCase::kDirect, params);
    const ExecutionPlan literal = compile_literal(memory_kind);

    ASSERT_FALSE(derived.ok());
    ASSERT_FALSE(literal.ok());
    EXPECT_TRUE(test::has_diagnostic(derived, "IMPLICIT_DEVICE_ZERO_REJECTED")) << derived.json();
    EXPECT_TRUE(test::has_diagnostic(literal, "IMPLICIT_DEVICE_ZERO_REJECTED")) << literal.json();
    EXPECT_TRUE(derived.tensor_allocations().empty());
    EXPECT_TRUE(literal.tensor_allocations().empty());
  }
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, RunsOrdinaryInputCompatibilityAfterResolution) {
  Graph graph{"derived-memory-kind-edge-compatibility"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<DerivedMemoryTensorSource>("source");
  const auto sink = graph.op<ExactMemoryTensorSink>("sink", MemoryKind::kPinnedHost);
  TensorMemoryKindTestParams params;
  params.memory_kind = MemoryKind::kPinnedHost;
  source.set_params(params);
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().memory_kind, MemoryKind::kPinnedHost);

  Graph mismatch_graph{"derived-memory-kind-edge-mismatch"};
  test::add_compile_test_default_clock(mismatch_graph);
  auto mismatched_source = mismatch_graph.op<DerivedMemoryTensorSource>("source");
  const auto mismatched_sink =
      mismatch_graph.op<ExactMemoryTensorSink>("sink", MemoryKind::kPinnedHost);
  TensorMemoryKindTestParams mismatched_params;
  mismatched_params.memory_kind = MemoryKind::kHost;
  mismatched_source.set_params(mismatched_params);
  mismatch_graph.add_flow(mismatched_source->output, mismatched_sink->input);

  const ExecutionPlan mismatch_plan = compile(mismatch_graph);

  ASSERT_FALSE(mismatch_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(mismatch_plan, "PHYSICAL_TENSOR_LAYOUT_MISMATCH"))
      << mismatch_plan.json();
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, RejectsDescriptorInputMismatchAfterResolution) {
  Graph graph{"derived-descriptor-memory-kind-edge-mismatch"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<DerivedMemoryImageSource>("source");
  const auto sink = graph.op<ExactMemoryImageSink>("sink", MemoryKind::kPinnedHost);
  TensorMemoryKindTestParams params;
  params.memory_kind = MemoryKind::kHost;
  source.set_params(params);
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_LAYOUT_MISMATCH")) << plan.json();
}

// REQ: COMPILE-076
TEST(TensorMemoryKindDerivationCompile, ArtifactIdentityIncludesPortableDerivationEvidence) {
  const ExecutionPlan literal = compile_literal(MemoryKind::kHost);
  ASSERT_TRUE(literal.ok()) << literal.json();
  EXPECT_EQ(literal.json().find("tensor_memory_kind_derivations"), std::string_view::npos);

  const ExecutionPlan plan = compile_derived(MemoryExpressionCase::kMappedEnum);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  auto changed = std::make_shared<detail::ExecutionPlanState>(*state);
  ASSERT_FALSE(changed->tensor_memory_kind_derivations.empty());
  ASSERT_EQ(changed->tensor_memory_kind_derivations.front().mapping.size(), 3U);
  changed->tensor_memory_kind_derivations.front().mapping.back().memory_kind =
      MemoryKind::kCudaManaged;
  ASSERT_TRUE(detail::finalize_plan_artifact(*changed));
  EXPECT_NE(changed->digest, plan.digest());

  auto malformed = std::make_shared<detail::ExecutionPlanState>(*state);
  malformed->tensor_memory_kind_derivations.front().resolved_memory_kind = MemoryKind::kUnknown;
  const auto rejected = detail::finalize_plan_artifact(*malformed);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().message, "TENSOR_MEMORY_KIND_DERIVATION_PLAN_INVALID");

  auto missing_path_identity = std::make_shared<detail::ExecutionPlanState>(*state);
  missing_path_identity->tensor_memory_kind_derivations.front().parameter_member_full_ids.clear();
  const auto missing_identity_rejected = detail::finalize_plan_artifact(*missing_path_identity);
  ASSERT_FALSE(missing_identity_rejected);
  EXPECT_EQ(missing_identity_rejected.error().message,
            "TENSOR_MEMORY_KIND_DERIVATION_PLAN_INVALID");
}

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(TensorMemoryKindDerivationCompile, OwnsBothMismatchEndpointsAfterGraphAndPlanRetire) {
  Diagnostic owned;
  ExecutionGraphView owning_graph;
  {
    Graph graph{"derived-memory-kind-owned-endpoints"};
    test::add_compile_test_default_clock(graph);
    auto source = graph.op<DerivedMemoryTensorSource>("source");
    const auto sink = graph.op<ExactMemoryTensorSink>("sink", MemoryKind::kPinnedHost);
    TensorMemoryKindTestParams params;
    params.memory_kind = MemoryKind::kHost;
    source.set_params(params);
    graph.add_flow(source->output, sink->input);
    const ExecutionPlan plan = compile(graph);
    ASSERT_FALSE(plan.ok());
    const auto rows = plan.diagnostics();
    const auto found =
        std::ranges::find(rows, "PHYSICAL_TENSOR_LAYOUT_MISMATCH", &Diagnostic::token);
    ASSERT_NE(found, rows.end());
    owned = *found;
    owning_graph = plan.graph();
  }
  // Derivation replay, the original Graph, and the entire ExecutionPlan facade have retired.
  // Only the owning diagnostic and the independent compiled-view handle survive.
  if (!owned.detail.has_value()) {
    FAIL() << "Resolved compatibility rejection must retain its two known endpoints";
  }
  const auto& evidence = owned.detail.value();
  EXPECT_EQ(evidence.operation, DiagnosticOperation::kCompile);
  EXPECT_EQ(evidence.error.code, ErrorCode::kIncompatible);
  EXPECT_EQ(evidence.port_name, "input");
  EXPECT_EQ(evidence.related_port_name, "output");
  EXPECT_NE(evidence.operator_path.find("sink"), std::string::npos);
  EXPECT_NE(evidence.related_operator_path.find("source"), std::string::npos);
  ASSERT_TRUE(owned.source.vertex);
  ASSERT_TRUE(owned.source.port);
  ASSERT_TRUE(evidence.related_source.vertex);
  ASSERT_TRUE(evidence.related_source.port);
  EXPECT_NE(owned.source.vertex, evidence.related_source.vertex);
  EXPECT_NE(owned.source.port, evidence.related_source.port);

  const auto assert_endpoint = [&](const DiagnosticSource& identity,
                                   std::string_view vertex_name,
                                   std::string_view port_name) {
    EXPECT_EQ(identity.graph_generation, owning_graph.generation());
    const auto vertices = owning_graph.vertices();
    const auto vertex = std::ranges::find(vertices, identity.vertex, &VertexRecord::id);
    ASSERT_NE(vertex, vertices.end());
    EXPECT_EQ(vertex->name, vertex_name);
    EXPECT_EQ(identity.graph, vertex->graph);
    const auto ports = owning_graph.ports();
    const auto port = std::ranges::find_if(
        ports, [&](const auto& candidate) { return candidate.id.value == identity.port.value(); });
    ASSERT_NE(port, ports.end());
    EXPECT_EQ(port->owner, vertex->id);
    EXPECT_EQ(port->name, port_name);
  };
  assert_endpoint(owned.source, "sink", "input");
  assert_endpoint(evidence.related_source, "source", "output");

  const auto empty_report = format_diagnostic(owned, {});
  EXPECT_NE(empty_report.find("operator=" + evidence.operator_path), std::string::npos);
  EXPECT_NE(empty_report.find("port_name=input"), std::string::npos);
  EXPECT_NE(empty_report.find("related_operator=" + evidence.related_operator_path),
            std::string::npos);
  EXPECT_NE(empty_report.find("related_port=output"), std::string::npos);

  const auto compiled_report = format_diagnostic(owned, owning_graph);
  EXPECT_NE(
      compiled_report.find("vertex=sink, vertex_id=" + std::to_string(owned.source.vertex.value)),
      std::string::npos);
  EXPECT_NE(compiled_report.find("port=" + std::to_string(owned.source.port.value()) +
                                 ", port_name=input"),
            std::string::npos);
  EXPECT_NE(compiled_report.find("related_vertex_id=" +
                                 std::to_string(evidence.related_source.vertex.value)),
            std::string::npos);
  EXPECT_NE(compiled_report.find("related_port_id=" +
                                 std::to_string(evidence.related_source.port.value())),
            std::string::npos);
  EXPECT_EQ(compiled_report.find("captured_port_name="), std::string::npos);
  const auto first_primary_name = compiled_report.find("port_name=input");
  ASSERT_NE(first_primary_name, std::string::npos);
  EXPECT_EQ(compiled_report.find("port_name=input", first_primary_name + 1U), std::string::npos);
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
