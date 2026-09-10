/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "artifact_test_support.hpp"
#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_generation.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_rebind_support.hpp"
#include "core/parameter_schema_test_access.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

struct TensorByteSpanTestParams {
  std::uint64_t bytes{1000U};
  std::uint64_t frozen_bytes{1000U};
  std::uint64_t live_bytes{1000U};
  std::uint64_t lifecycle_bytes{1000U};
  std::int64_t signed_bytes{1000};
  float fractional_bytes{1000.0F};
};

HOLOSCAN_PARAM_SCHEMA(
    TensorByteSpanTestParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.TensorByteSpanTestParams", 1)),
    holoscan::field<&TensorByteSpanTestParams::bytes>("bytes").recompile_required(),
    holoscan::field<&TensorByteSpanTestParams::frozen_bytes>("frozen_bytes"),
    holoscan::field<&TensorByteSpanTestParams::live_bytes>("live_bytes").runtime_mutable(),
    holoscan::field<&TensorByteSpanTestParams::lifecycle_bytes>("lifecycle_bytes")
        .lifecycle_mutable(LifecycleStage::kConfigure),
    holoscan::field<&TensorByteSpanTestParams::signed_bytes>("signed_bytes").recompile_required(),
    holoscan::field<&TensorByteSpanTestParams::fractional_bytes>("fractional_bytes")
        .recompile_required());

struct TensorByteSpanShapeTestParams {
  std::vector<std::uint32_t> shape{20U, 30U};
};

HOLOSCAN_PARAM_SCHEMA(TensorByteSpanShapeTestParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.TensorByteSpanShapeTestParams", 1)),
                      holoscan::field<&TensorByteSpanShapeTestParams::shape>("shape")
                          .max_elements(2U)
                          .recompile_required());

struct UnboundedTensorByteSpanShapeTestParams {
  std::vector<std::uint32_t> shape{20U, 30U};
};

HOLOSCAN_PARAM_SCHEMA(
    UnboundedTensorByteSpanShapeTestParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.UnboundedTensorByteSpanShapeTestParams", 1)),
    holoscan::field<&UnboundedTensorByteSpanShapeTestParams::shape>("shape").recompile_required());

struct SensitiveTensorByteSpanTestParams {
  std::uint64_t bytes{1000U};
};

HOLOSCAN_PARAM_SCHEMA(SensitiveTensorByteSpanTestParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.SensitiveTensorByteSpanTestParams", 1)),
                      holoscan::field<&SensitiveTensorByteSpanTestParams::bytes>("bytes")
                          .recompile_required()
                          .sensitive());

struct DerivedMultiTensorSchema {
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> calibration;
};

template <>
consteval SchemaIdentity schema_identity<DerivedMultiTensorSchema>() {
  return schema_identity<schema::ImageT>();
}

template <>
struct DescriptorTensorSchemaNativeTraits<DerivedMultiTensorSchema> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_default_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, 2U> tensor_field_ids{1U, 7U};

  [[nodiscard]] static const Tensor* tensor_at(const DerivedMultiTensorSchema& value,
                                               std::size_t index) noexcept {
    return index == 0U ? value.image.get() : (index == 1U ? value.calibration.get() : nullptr);
  }
};

namespace detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<DerivedMultiTensorSchema>() noexcept {
  static const RuntimePortTypeOps ops = schema::make_descriptor_tensor_port_type_ops<
      DerivedMultiTensorSchema,
      schema::Image,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::Image>>(
      [](void*) noexcept -> expected<void, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*) noexcept {},
      [](const DerivedMultiTensorSchema&,
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

using namespace std::chrono_literals;

enum class DerivationCase : std::uint8_t {
  kScalar,
  kFrozenScalar,
  kLiveScalar,
  kLifecycleScalar,
  kSignedScalar,
  kInvalidType,
  kCheckedArithmetic,
  kOverflow,
  kAlignOverflow,
};

// Authors raw Tensor pool capacity from selectable scalar expressions.
// Used by ResolvesCheckedParameterExpressionIntoOrdinaryPoolPlan.
class DerivedTensorSource final : public Operator<TensorByteSpanTestParams> {
 public:
  explicit DerivedTensorSource(DerivationCase derivation = DerivationCase::kScalar)
      : derivation_(derivation) {}

  void setup(OperatorSpec& spec) override {
    TensorByteSpanExpression expression;
    switch (derivation_) {
      case DerivationCase::kScalar:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>());
        break;
      case DerivationCase::kFrozenScalar:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::frozen_bytes>());
        break;
      case DerivationCase::kLiveScalar:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::live_bytes>());
        break;
      case DerivationCase::kLifecycleScalar:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::lifecycle_bytes>());
        break;
      case DerivationCase::kSignedScalar:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::signed_bytes>());
        break;
      case DerivationCase::kInvalidType:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::fractional_bytes>());
        break;
      case DerivationCase::kCheckedArithmetic:
        expression = parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>())
                         .multiplied_by(tensor_byte_span(3U))
                         .ceiling_divided_by(tensor_byte_span(2U))
                         .maximum_with(tensor_byte_span(1500U))
                         .aligned_to(256U);
        break;
      case DerivationCase::kOverflow:
        expression =
            parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>())
                .multiplied_by(tensor_byte_span(std::numeric_limits<std::uint64_t>::max()));
        break;
      case DerivationCase::kAlignOverflow:
        expression = tensor_byte_span(std::numeric_limits<std::uint64_t>::max()).aligned_to(256U);
        break;
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation =
                    TensorRepresentation{
                        .memory_kind = MemoryKind::kHost,
                        .dtype = DLDataType{kDLUInt, 8U, 1U},
                        .rank = 2U,
                    },
                .storage = TensorOutputStorage::kRuntimePool,
            },
            std::move(expression));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  DerivationCase derivation_;
};

// Authors an independently bounded copy-staging span for a derived raw Tensor pool.
// Used by PreservesIndependentRawTensorStagingSpans.
class DerivedCopyTensorSource final : public Operator<TensorByteSpanTestParams> {
 public:
  explicit DerivedCopyTensorSource(std::optional<std::uint64_t> staging_byte_span = std::nullopt)
      : staging_byte_span_(staging_byte_span) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
                .transfer =
                    TensorTransferPolicy{
                        .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                        .max_staging_byte_span = staging_byte_span_,
                    },
            },
            parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  std::optional<std::uint64_t> staging_byte_span_;
};

// Couples one Parameter-derived host Tensor allocation to a scalar topology edge so generation-
// zero byte-capacity changes can be proven independent from constrained-automatic partitioning.
class HybridCapacitySource final : public Operator<TensorByteSpanTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(tensor, "tensor")
        .produces_tensor(
            TensorOutputSpec{
                .representation =
                    TensorRepresentation{
                        .memory_kind = MemoryKind::kHost,
                        .dtype = DLDataType{kDLUInt, 8U, 1U},
                        .rank = 1U,
                    },
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>()));
    spec.output(scalar, "scalar").max_emits_per_compute(1U);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> tensor;
  Output<std::uint32_t> scalar;
};

// Omits its derivation during replay to exercise candidate-coverage validation.
// Used by RejectsMissingGenerationZeroDerivationCandidate.
class MissingReplayDerivationSource final : public Operator<TensorByteSpanTestParams> {
 public:
  static inline bool omit_derivation{};

  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec output_spec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
    auto& builder = spec.output(output, "output");
    if (omit_derivation) {
      builder.produces_tensor(output_spec);
    } else {
      builder.produces_tensor(output_spec,
                              parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>()));
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Declares two derived outputs whose replay expressions can diverge from the snapshot.
// Used by RetainsPortIdentityForEachReplayDivergence.
class DivergentReplayDerivationSource final : public Operator<> {
 public:
  static inline bool diverge{};

  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec output_spec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
    const std::uint64_t offset = diverge ? 1U : 0U;
    spec.output(first, "first").produces_tensor(output_spec, tensor_byte_span(1024U + offset));
    spec.output(second, "second").produces_tensor(output_spec, tensor_byte_span(2048U + offset));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> first;
  Output<Tensor> second;
};

// Makes a valid derived output declaration fail during setup replay.
// Used by ReportsReplayDeclarationFailureBeforeCoverageLoss.
class InvalidReplayDerivationSource final : public Operator<> {
 public:
  static inline bool invalidate{};

  void setup(OperatorSpec& spec) override {
    spec.output(output, invalidate ? "" : "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            tensor_byte_span(1024U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Derives raw Tensor bytes from a bounded, fixed-rank shape Parameter.
// Used by ReducesBoundedShapeParameterUsingExactDtypeAndRank.
class ShapeDerivedTensorSource final : public Operator<TensorByteSpanShapeTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation =
                    TensorRepresentation{
                        .memory_kind = MemoryKind::kHost,
                        .dtype = DLDataType{kDLUInt, 8U, 1U},
                        .rank = 2U,
                    },
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_contiguous_byte_span(param_ref<&TensorByteSpanShapeTestParams::shape>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Uses a wide dtype so bounded shape multiplication can overflow.
// Used by EnforcesMutabilityAndCheckedArithmetic.
class WideShapeDerivedTensorSource final : public Operator<TensorByteSpanShapeTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 64U, 1U},
                                   .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_contiguous_byte_span(param_ref<&TensorByteSpanShapeTestParams::shape>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// References an unbounded shape Parameter that cannot prove finite pool capacity.
// Used by RejectsInvalidScalarAndShapeParameterValues.
class UnboundedShapeDerivedTensorSource final
    : public Operator<UnboundedTensorByteSpanShapeTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_contiguous_byte_span(
                param_ref<&UnboundedTensorByteSpanShapeTestParams::shape>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// References a sensitive Parameter that cannot enter public capacity evidence.
// Used by RejectsSensitiveCapacityParameterWithNodeContext.
class SensitiveDerivedTensorSource final : public Operator<SensitiveTensorByteSpanTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_byte_span(param_ref<&SensitiveTensorByteSpanTestParams::bytes>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Authors derived capacity and optional staging policy for an Image Tensor field.
// Used by ResolvesSingularDescriptorFieldIntoExistingPoolPlan.
class DerivedImageSource final : public Operator<TensorByteSpanTestParams> {
 public:
  explicit DerivedImageSource(std::optional<std::uint64_t> staging_byte_span = std::nullopt,
                              bool declare_transfer = true)
      : staging_byte_span_(staging_byte_span), declare_transfer_(declare_transfer) {}

  void setup(OperatorSpec& spec) override {
    constexpr TensorFieldSelector<schema::ImageT> kData{4U, 0U};
    spec.output(output, "output")
        .tensor_field(kData)
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
                .transfer =
                    declare_transfer_
                        ? std::optional<TensorTransferPolicy>{TensorTransferPolicy{
                              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                              .max_staging_byte_span = staging_byte_span_,
                          }}
                        : std::nullopt,
            },
            parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<schema::ImageT> output;

 private:
  std::optional<std::uint64_t> staging_byte_span_;
  bool declare_transfer_{};
};

// Consumes the singular Image Tensor field across a process boundary.
// Used by CrossProcessSingularDescriptorUsesDerivedStagingSpan.
class DerivedImageSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    constexpr TensorFieldSelector<schema::ImageT> kData{4U, 0U};
    spec.input(input, "input")
        .tensor_field(kData)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;
};

// Authors independent derived capacities for two descriptor-backed Tensor fields.
// Used by ResolvesHeterogeneousDescriptorFieldsIndependently.
class DerivedMultiTensorSource final : public Operator<TensorByteSpanTestParams> {
 public:
  static constexpr TensorFieldSelector<DerivedMultiTensorSchema> kImage{1U, 0U};
  static constexpr TensorFieldSelector<DerivedMultiTensorSchema> kCalibration{7U, 1U};

  explicit DerivedMultiTensorSource(
      bool runtime_mutable = false,
      std::optional<std::uint64_t> image_staging_byte_span = std::nullopt,
      std::optional<std::uint64_t> calibration_staging_byte_span = std::nullopt)
      : runtime_mutable_(runtime_mutable),
        image_staging_byte_span_(image_staging_byte_span),
        calibration_staging_byte_span_(calibration_staging_byte_span) {}

  void setup(OperatorSpec& spec) override {
    const TensorOutputSpec image_spec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U},
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                .max_staging_byte_span = image_staging_byte_span_,
            },
    };
    const TensorOutputSpec calibration_spec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLFloat, 32U, 1U},
                           .rank = 1U},
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                .max_staging_byte_span = calibration_staging_byte_span_,
            },
    };
    auto& builder = spec.output(output, "output");
    const TensorByteSpanExpression image_span =
        runtime_mutable_ ? parameter_byte_span(param_ref<&TensorByteSpanTestParams::live_bytes>())
                         : parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>());
    const TensorByteSpanExpression calibration_span =
        runtime_mutable_ ? parameter_byte_span(param_ref<&TensorByteSpanTestParams::live_bytes>())
                         : parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>());
    builder.tensor_field(kImage).produces_tensor(image_spec, image_span);
    builder.tensor_field(kCalibration)
        .produces_tensor(calibration_spec, calibration_span.multiplied_by(tensor_byte_span(2U)));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<DerivedMultiTensorSchema> output;

 private:
  bool runtime_mutable_{};
  std::optional<std::uint64_t> image_staging_byte_span_;
  std::optional<std::uint64_t> calibration_staging_byte_span_;
};

// Provides a literal raw Tensor allocation as input-derived capacity evidence.
// Used by ResolvesInputCapacityFromLiteralAndDerivedProducers.
class LiteralTensorSource final : public Operator<> {
 public:
  explicit LiteralTensorSource(std::uint64_t bytes) : bytes_(bytes) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 2U,
                },
            .bounds = tensor_bounds(bytes_),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  std::uint64_t bytes_;
};

// Derives its output pool capacity from the finite producer bound on its input.
// Used by ResolvesInputCapacityFromLiteralAndDerivedProducers.
class InputDerivedTensorRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation =
                    TensorRepresentation{
                        .memory_kind = MemoryKind::kHost,
                        .dtype = DLDataType{kDLUInt, 8U, 1U},
                        .rank = 2U,
                    },
                .storage = TensorOutputStorage::kRuntimePool,
            },
            input_max_byte_span(input));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Couples Parameter- and input-derived byte bounds to independent rate provenance and queue
// sizing so the two compile-cold DAGs can be proven to meet only at pool allocation lowering.
class RateDerivedTensorSource final : public Operator<TensorByteSpanTestParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .nominal_rate(Rate::hz(60U))
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_byte_span(param_ref<&TensorByteSpanTestParams::bytes>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class RateInputDerivedTensorRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output")
        .follows(input, RateRatio{1U, 4U})
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            input_max_byte_span(input));
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

class RateDerivedTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

// Provides a local raw-Tensor consumer so physical lowering must create an edge service.
// Used by RejectsOverflowedDerivedPoolWithoutPassThroughService.
class RawTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

class InputDerivedTensorChildGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<Tensor> input{"input"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto relay = op<InputDerivedTensorRelay>("relay");
    export_input(input, relay->input);
  }

 private:
  ClockKey clock_import_;
};

class LiteralTensorSourceChildGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Output<Tensor> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto source = op<LiteralTensorSource>("source", 8192U);
    export_output(output, source->output);
  }

 private:
  ClockKey clock_import_;
};

class NestedDerivedTensorGraph final : public Graph {
 protected:
  void compose() override { (void)op<DerivedTensorSource>("source", DerivationCase::kScalar); }
};

// Provides a pass-through raw Tensor output without a finite producer bound.
// Used by RejectsDisconnectedAndUnboundedInputDependencies.
class UnboundedTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost, .rank = 2U}});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Provides a finite setup-time pass-through bound that a Graph override can supersede.
// Used by RejectsInputBoundFromSupersededSetupContract.
class BoundedPassThroughTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

template <typename ParamsT>
[[nodiscard]] ParamSchemaHash required_value_digest(const ParamsT& value) {
  const auto digest = detail::parameter_value_digest(
      param_type_descriptor<ParamsT>(), param_schema_descriptor<ParamsT>(), &value, false);
  if (!digest.available()) {
    throw std::logic_error{"invalid Tensor byte-span Parameter fixture"};
  }
  return digest.digest;
}

[[nodiscard]] ParamOverrideSet tensor_byte_span_override(Graph& graph,
                                                         TensorByteSpanTestParams resolved,
                                                         std::string_view changed_field) {
  const TensorByteSpanTestParams baseline{};
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<TensorByteSpanTestParams>();
  const auto field =
      std::find_if(schema->fields.begin(), schema->fields.end(), [&](const auto& item) {
        return item.canonical_name == changed_field;
      });
  const auto captured = detail::capture_parameter_rebind_graph(graph);
  if (field == schema->fields.end() || !captured) {
    throw std::logic_error{"invalid Tensor byte-span override fixture"};
  }
  const ParamSchemaHash resolved_digest = required_value_digest(resolved);
  std::vector<detail::ParamOverrideOperatorState> operators;
  operators.push_back(detail::ParamOverrideSetAccess::resolved_operator(
      "source",
      std::move(resolved),
      required_value_digest(baseline),
      resolved_digest,
      {{.field = field->id,
        .full_id = field->full_id,
        .origin = detail::ParamOverrideContributionOrigin::kCaptured}}));
  return detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(operators), captured->admission_context_digest);
}

[[nodiscard]] const TensorAllocationPlanRecord& only_allocation(const ExecutionPlan& plan) {
  if (plan.tensor_allocations().size() != 1U) {
    throw std::logic_error{"expected one Tensor allocation"};
  }
  return plan.tensor_allocations().front();
}

[[nodiscard]] std::string derivation_artifact_section(std::string_view artifact) {
  constexpr std::string_view kBegin{"\"tensor_byte_span_derivations\":"};
  constexpr std::string_view kEnd{",\"cuda_compute_streams\":"};
  const std::size_t begin = artifact.find(kBegin);
  const std::size_t end = artifact.find(kEnd, begin);
  if (begin == std::string_view::npos || end == std::string_view::npos) {
    throw std::logic_error{"Tensor byte-span derivation artifact section is missing"};
  }
  return std::string{artifact.substr(begin, end - begin)};
}

[[nodiscard]] std::string read_derivation_artifact_golden() {
  const std::string path =
      std::string(HOLOSCAN_COMPILER_TEST_DATA_DIR) + "/tensor_byte_span_derivation.v2.json";
  std::ifstream stream{path, std::ios::binary};
  if (!stream.is_open()) {
    throw std::logic_error{"Tensor byte-span derivation artifact golden is missing"};
  }
  return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] detail::ProcessPlacementBinding split_source_sink_placement() {
  constexpr std::array source{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array sink{detail::GraphObjectSelector{.graph_path = "sink"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = sink,
                             .launch_group = "consumer-process",
                         });
  return placement;
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ResolvesCheckedParameterExpressionIntoOrdinaryPoolPlan) {
  // Graph topology:
  //   source (isolated derived raw Tensor output)
  Graph graph{"derived-checked-capacity"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedTensorSource>("source", DerivationCase::kCheckedArithmetic);

  // Key setup: the expression computes align_up(max(ceil(1000 * 3 / 2), 1500), 256).
  const GraphAuthoringIdentity identity = graph.view().identity();
  const ExecutionPlan plan = compile(graph);

  // Key assertion: cold derivation becomes the same finite allocation authority Runtime consumes.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(graph.view().identity(), identity);
  EXPECT_EQ(only_allocation(plan).bounds.max_byte_span, 1536U);
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  const auto& evidence = plan.tensor_byte_span_derivations().front();
  EXPECT_EQ(evidence.resolved_max_byte_span, 1536U);
  EXPECT_EQ(evidence.update_disposition,
            TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
  ASSERT_EQ(evidence.nodes.size(), 8U);
  EXPECT_EQ(evidence.nodes.back().operation, TensorByteSpanOperation::kAlignUp);
  EXPECT_EQ(plan.find_tensor_byte_span_derivation("source", "output"), &evidence);
  EXPECT_EQ(plan.find_tensor_byte_span_derivation("missing", "output"), nullptr);
  const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  auto normalized = std::make_shared<detail::ExecutionPlanState>(*state);
  detail::rebind_plan_generation(*normalized, PlanGenerationId{1U});
  ASSERT_TRUE(detail::finalize_plan_artifact(*normalized));
  EXPECT_EQ(detail::artifact_test::normalized_artifact_json(
                "{" + derivation_artifact_section(normalized->json) + "}"),
            detail::artifact_test::normalized_artifact_json(read_derivation_artifact_golden()));
  EXPECT_NE(plan.json().find("compiler_derived_byte_span"), std::string_view::npos);

  auto changed_evidence = std::make_shared<detail::ExecutionPlanState>(*state);
  ASSERT_FALSE(changed_evidence->tensor_byte_span_derivations.front().nodes.empty());
  ++changed_evidence->tensor_byte_span_derivations.front().nodes.front().resolved_value;
  ASSERT_TRUE(detail::finalize_plan_artifact(*changed_evidence));
  // Key assertion: derivation evidence contributes independently to semantic plan identity.
  EXPECT_NE(changed_evidence->digest, plan.digest());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, FindsNestedAndFieldScopedDerivationEvidenceUnambiguously) {
  // Graph topologies:
  //   nested_graph: child/source (isolated nested output)
  //   field_graph: source (isolated two-field descriptor output)
  Graph nested_graph{"derived-nested-lookup"};
  test::add_compile_test_default_clock(nested_graph);
  (void)nested_graph.embed<NestedDerivedTensorGraph>("child");

  // Key setup: compile a derived output beneath an embedded Graph path.
  const ExecutionPlan nested_plan = compile(nested_graph);

  // Key assertion: the authored nested path resolves to exactly one plan-owned record.
  ASSERT_TRUE(nested_plan.ok()) << nested_plan.json();
  ASSERT_EQ(nested_plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(nested_plan.find_tensor_byte_span_derivation("child/source", "output"),
            &nested_plan.tensor_byte_span_derivations().front());
  EXPECT_EQ(nested_plan.find_tensor_byte_span_derivation("child", "output"), nullptr);

  Graph field_graph{"derived-field-lookup"};
  test::add_compile_test_default_clock(field_graph);
  (void)field_graph.op<DerivedMultiTensorSource>("source");
  const ExecutionPlan field_plan = compile(field_graph);

  // Key assertion: multi-Tensor lookup requires one complete, exact field selector.
  ASSERT_TRUE(field_plan.ok()) << field_plan.json();
  ASSERT_EQ(field_plan.tensor_byte_span_derivations().size(), 2U);
  EXPECT_EQ(field_plan.find_tensor_byte_span_derivation("source", "output"), nullptr);
  EXPECT_EQ(field_plan.find_tensor_byte_span_derivation("source", "output", 1U), nullptr);
  EXPECT_EQ(field_plan.find_tensor_byte_span_derivation("source", "output", std::nullopt, 0U),
            nullptr);
  EXPECT_EQ(field_plan.find_tensor_byte_span_derivation("source", "output", 1U, 0U),
            &field_plan.tensor_byte_span_derivations()[0]);
  EXPECT_EQ(field_plan.find_tensor_byte_span_derivation("source", "output", 7U, 1U),
            &field_plan.tensor_byte_span_derivations()[1]);
  EXPECT_EQ(field_plan.find_tensor_byte_span_derivation("source", "output", 9U, 0U), nullptr);
}

// REQ: COMPILE-067
// REQ: COMPILE-068
TEST(TensorByteSpanDerivationCompile, HybridAnchorRetainsFinalOwnerAndDerivedByteSpanIdentity) {
  Graph graph{"hybrid-derived-capacity"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedTensorSource>("source", DerivationCase::kCheckedArithmetic);
  (void)graph.op<test::CompileTestOperator>("automatic-remainder");
  graph.partition("derived-anchor").add(source);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 1U);
  const auto anchor = plan.partitions().lookup_named("derived-anchor");
  ASSERT_TRUE(anchor);
  const auto anchor_record =
      std::ranges::find(plan.execution_partitions(), anchor->id(), &ExecutionPartitionRecord::id);
  ASSERT_NE(anchor_record, plan.execution_partitions().end());
  const auto residual = plan.partitions().lookup_for_path("automatic-remainder");
  ASSERT_TRUE(residual);
  const auto residual_record =
      std::ranges::find(plan.execution_partitions(), residual->id(), &ExecutionPartitionRecord::id);
  ASSERT_NE(residual_record, plan.execution_partitions().end());
  EXPECT_EQ(anchor_record->id, anchor->id());
  EXPECT_EQ(anchor->execution_lane(), anchor_record->execution_lane);
  EXPECT_EQ(residual_record->id, residual->id());
  EXPECT_EQ(residual->execution_lane(), residual_record->execution_lane);
  EXPECT_NE(anchor->id(), residual->id());
  EXPECT_NE(anchor->execution_lane(), residual->execution_lane());
  const auto anchor_lane = std::ranges::find(
      plan.execution_lanes(), anchor->execution_lane(), &ExecutionLanePlanRecord::execution_lane);
  const auto residual_lane = std::ranges::find(
      plan.execution_lanes(), residual->execution_lane(), &ExecutionLanePlanRecord::execution_lane);
  ASSERT_NE(anchor_lane, plan.execution_lanes().end());
  ASSERT_NE(residual_lane, plan.execution_lanes().end());
  EXPECT_EQ(anchor_lane->partition, anchor->id());
  EXPECT_EQ(residual_lane->partition, residual->id());
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  const auto& allocation = plan.tensor_allocations().front();
  const auto& derivation = plan.tensor_byte_span_derivations().front();
  EXPECT_EQ(derivation.output, allocation.output);
  EXPECT_EQ(derivation.plan_generation, allocation.plan_generation);
  EXPECT_EQ(derivation.run_generation, RunGenerationId{});
  EXPECT_EQ(derivation.resolved_max_byte_span, allocation.bounds.max_byte_span);
  EXPECT_EQ(derivation.update_disposition,
            TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
  const auto output = std::ranges::find(plan.ports(), allocation.output, &PortPlanRecord::id);
  ASSERT_NE(output, plan.ports().end());
  EXPECT_EQ(output->direction, PortDirection::kOutput);
  EXPECT_TRUE(std::ranges::find(anchor->members(), output->owner) != anchor->members().end());
  EXPECT_EQ(residual->origin(), ExecutionPartitionOrigin::kAutomatic);
}

// REQ: COMPILE-068
TEST(TensorByteSpanDerivationCompile,
     HybridPartitioningIsInvariantToGenerationZeroDerivedCapacity) {
  Graph graph{"hybrid-generation-zero-capacity-invariance"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<HybridCapacitySource>("source");
  const auto remainder = graph.op<test::CompileTestOperator>("automatic-remainder");
  graph.add_flow(source->scalar, remainder->input);
  graph.partition("derived-anchor").add(source);
  graph.auto_partition_remaining();
  const GraphAuthoringIdentity identity = graph.view().identity();

  const auto compile_capacity = [&](std::uint64_t bytes) {
    TensorByteSpanTestParams resolved;
    resolved.bytes = bytes;
    CompileOptions options;
    options.parameter_overrides = tensor_byte_span_override(graph, resolved, "bytes");
    return compile(graph, std::move(options));
  };

  const ExecutionPlan smaller = compile_capacity(1024U);
  const ExecutionPlan larger = compile_capacity(8192U);

  ASSERT_TRUE(smaller.ok()) << smaller.json();
  ASSERT_TRUE(larger.ok()) << larger.json();
  EXPECT_EQ(graph.view().identity(), identity);
  ASSERT_EQ(smaller.tensor_byte_span_derivations().size(), 1U);
  ASSERT_EQ(larger.tensor_byte_span_derivations().size(), 1U);
  ASSERT_EQ(smaller.tensor_allocations().size(), 1U);
  ASSERT_EQ(larger.tensor_allocations().size(), 1U);
  const auto& smaller_derivation = smaller.tensor_byte_span_derivations().front();
  const auto& larger_derivation = larger.tensor_byte_span_derivations().front();
  const auto& smaller_allocation = smaller.tensor_allocations().front();
  const auto& larger_allocation = larger.tensor_allocations().front();
  EXPECT_EQ(smaller_derivation.resolved_max_byte_span, 1024U);
  EXPECT_EQ(larger_derivation.resolved_max_byte_span, 8192U);
  EXPECT_NE(smaller_derivation.resolved_max_byte_span, larger_derivation.resolved_max_byte_span);
  EXPECT_NE(smaller_allocation.bounds.max_byte_span, larger_allocation.bounds.max_byte_span);
  EXPECT_NE(smaller_allocation.aggregate_byte_capacity, larger_allocation.aggregate_byte_capacity);
  EXPECT_EQ(smaller_allocation.slot_count, larger_allocation.slot_count);

  EXPECT_EQ(smaller.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(larger.partitioning_plan().mode, smaller.partitioning_plan().mode);
  EXPECT_EQ(smaller.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(smaller.partitioning_plan().automatic_partition_count, 1U);
  EXPECT_EQ(larger.partitioning_plan().authored_partition_count,
            smaller.partitioning_plan().authored_partition_count);
  EXPECT_EQ(larger.partitioning_plan().automatic_partition_count,
            smaller.partitioning_plan().automatic_partition_count);

  ASSERT_EQ(smaller.execution_partitions().size(), 2U);
  ASSERT_EQ(larger.execution_partitions().size(), smaller.execution_partitions().size());
  for (std::size_t index = 0U; index < smaller.execution_partitions().size(); ++index) {
    const auto& left = smaller.execution_partitions()[index];
    const auto& right = larger.execution_partitions()[index];
    EXPECT_EQ(left.id, right.id);
    EXPECT_EQ(left.authored_name, right.authored_name);
    EXPECT_EQ(left.members, right.members);
    EXPECT_EQ(left.operator_paths, right.operator_paths);
    EXPECT_EQ(left.execution_lane, right.execution_lane);
  }

  ASSERT_EQ(smaller.partition_decisions().size(), larger.partition_decisions().size());
  for (std::size_t index = 0U; index < smaller.partition_decisions().size(); ++index) {
    const auto& left = smaller.partition_decisions()[index];
    const auto& right = larger.partition_decisions()[index];
    EXPECT_EQ(left.vertex, right.vertex);
    EXPECT_EQ(left.partition, right.partition);
    EXPECT_EQ(left.decision, right.decision);
    EXPECT_EQ(left.reasons, right.reasons);
  }

  ASSERT_EQ(smaller.schedule_edges().size(), 1U);
  ASSERT_EQ(larger.schedule_edges().size(), smaller.schedule_edges().size());
  for (std::size_t index = 0U; index < smaller.schedule_edges().size(); ++index) {
    const auto& left = smaller.schedule_edges()[index];
    const auto& right = larger.schedule_edges()[index];
    EXPECT_EQ(left.edge, right.edge);
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.destination, right.destination);
  }

  ASSERT_EQ(smaller.execution_lanes().size(), larger.execution_lanes().size());
  for (std::size_t index = 0U; index < smaller.execution_lanes().size(); ++index) {
    EXPECT_EQ(smaller.execution_lanes()[index].execution_lane,
              larger.execution_lanes()[index].execution_lane);
    EXPECT_EQ(smaller.execution_lanes()[index].partition,
              larger.execution_lanes()[index].partition);
  }
  for (const ExecutionPlan* const plan : {&smaller, &larger}) {
    const auto& allocation = plan->tensor_allocations().front();
    const auto output = std::ranges::find(plan->ports(), allocation.output, &PortPlanRecord::id);
    ASSERT_NE(output, plan->ports().end());
    const auto owner_partition = std::find_if(
        plan->execution_partitions().begin(),
        plan->execution_partitions().end(),
        [&](const auto& partition) {
          return std::ranges::find(partition.members, output->owner) != partition.members.end();
        });
    ASSERT_NE(owner_partition, plan->execution_partitions().end());
    EXPECT_EQ(owner_partition->authored_name, std::optional<std::string>{"derived-anchor"});
    const auto owner_lane = std::ranges::find(
        plan->execution_lanes(), owner_partition->id, &ExecutionLanePlanRecord::partition);
    ASSERT_NE(owner_lane, plan->execution_lanes().end());
    EXPECT_EQ(owner_lane->execution_lane, owner_partition->execution_lane);
  }
}

// REQ: COMPILE-067
// REQ: COMPILE-069
TEST(TensorByteSpanDerivationCompile,
     ParameterAndInputDerivedBytesComposeWithFixedAndFollowedRateSizing) {
  Graph graph{"rate-and-derived-byte-capacity"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<RateDerivedTensorSource>("source");
  const auto relay = graph.op<RateInputDerivedTensorRelay>("relay");
  const auto sink = graph.op<RateDerivedTensorSink>("sink");
  graph.add_flow(source->output, relay->input);
  graph.add_flow(relay->output, sink->input);
  const GraphAuthoringIdentity identity = graph.view().identity();

  const auto compile_capacity = [&](const std::uint64_t bytes) {
    TensorByteSpanTestParams resolved;
    resolved.bytes = bytes;
    CompileOptions options;
    options.parameter_overrides = tensor_byte_span_override(graph, resolved, "bytes");
    return compile(graph, std::move(options));
  };

  const ExecutionPlan smaller = compile_capacity(1024U);
  const ExecutionPlan larger = compile_capacity(8192U);

  ASSERT_TRUE(smaller.ok()) << smaller.json();
  ASSERT_TRUE(larger.ok()) << larger.json();
  EXPECT_EQ(graph.view().identity(), identity);
  ASSERT_EQ(smaller.tensor_byte_span_derivations().size(), 2U);
  ASSERT_EQ(larger.tensor_byte_span_derivations().size(), 2U);
  ASSERT_EQ(smaller.tensor_allocations().size(), 2U);
  ASSERT_EQ(larger.tensor_allocations().size(), 2U);
  ASSERT_EQ(smaller.queue_sizing_recipes().size(), 2U);
  ASSERT_EQ(larger.queue_sizing_recipes().size(), 2U);

  std::array<std::uint32_t, 2U> relay_depths{};
  std::array<std::uint32_t, 2U> source_slots{};
  std::array<std::uint32_t, 2U> relay_slots{};
  std::array<std::uint64_t, 2U> source_bytes{};
  std::array<std::uint64_t, 2U> relay_bytes{};
  const std::array<const ExecutionPlan*, 2U> plans{&smaller, &larger};
  const std::array<std::uint64_t, 2U> expected_bytes{1024U, 8192U};
  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const auto& plan = *plans[index];
    const auto* const source_derivation = plan.find_tensor_byte_span_derivation("source", "output");
    const auto* const relay_derivation = plan.find_tensor_byte_span_derivation("relay", "output");
    ASSERT_NE(source_derivation, nullptr);
    ASSERT_NE(relay_derivation, nullptr);
    EXPECT_EQ(source_derivation->resolved_max_byte_span, expected_bytes[index]);
    EXPECT_EQ(relay_derivation->resolved_max_byte_span, expected_bytes[index]);
    EXPECT_EQ(source_derivation->update_disposition,
              TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
    EXPECT_EQ(relay_derivation->update_disposition,
              TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);

    const auto source_allocation = std::ranges::find(
        plan.tensor_allocations(), source_derivation->output, &TensorAllocationPlanRecord::output);
    const auto relay_allocation = std::ranges::find(
        plan.tensor_allocations(), relay_derivation->output, &TensorAllocationPlanRecord::output);
    ASSERT_NE(source_allocation, plan.tensor_allocations().end());
    ASSERT_NE(relay_allocation, plan.tensor_allocations().end());
    source_slots[index] = source_allocation->slot_count;
    relay_slots[index] = relay_allocation->slot_count;
    source_bytes[index] = source_allocation->aggregate_byte_capacity;
    relay_bytes[index] = relay_allocation->aggregate_byte_capacity;
    EXPECT_EQ(source_allocation->bounds.max_byte_span, expected_bytes[index]);
    EXPECT_EQ(relay_allocation->bounds.max_byte_span, expected_bytes[index]);
    EXPECT_EQ(source_allocation->aggregate_byte_capacity,
              source_allocation->slot_count * expected_bytes[index]);
    EXPECT_EQ(relay_allocation->aggregate_byte_capacity,
              relay_allocation->slot_count * expected_bytes[index]);

    const auto relay_edge = std::ranges::find(
        plan.payload_edges(), relay_derivation->output, &PayloadEdgePlanRecord::source);
    ASSERT_NE(relay_edge, plan.payload_edges().end());
    relay_depths[index] = relay_edge->queue_depth;
    const auto relay_recipe = std::ranges::find(
        plan.queue_sizing_recipes(), relay_edge->edge, &QueueSizingRecipePlanRecord::edge);
    ASSERT_NE(relay_recipe, plan.queue_sizing_recipes().end());
    EXPECT_EQ(relay_recipe->origin, QueueDepthOrigin::kRateAndContractDerived);

    const auto relay_rate =
        std::ranges::find(plan.port_rates(), relay_derivation->output, &PortRatePlanRecord::port);
    ASSERT_NE(relay_rate, plan.port_rates().end());
    EXPECT_EQ(relay_rate->kind, PortRateKind::kFollowsInput);
    EXPECT_EQ(relay_rate->follows_ratio, (RateRatio{1U, 4U}));

    const auto provider = std::ranges::find(plan.provider_allocation_plans(),
                                            relay_derivation->output,
                                            [](const ProviderAllocationPlanRecord& record) {
                                              return record.output.value_or(PortId{});
                                            });
    ASSERT_NE(provider, plan.provider_allocation_plans().end());
    ASSERT_GT(provider->first_sizing_dependency.value, 0U);
    const std::size_t first = provider->first_sizing_dependency.value - 1U;
    ASSERT_LE(first + provider->sizing_dependency_count, plan.pool_capacity_dependencies().size());
    const auto dependencies =
        plan.pool_capacity_dependencies().subspan(first, provider->sizing_dependency_count);
    EXPECT_TRUE(std::ranges::any_of(dependencies, [&](const auto& dependency) {
      return dependency.pool_id == provider->pool_id &&
             dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe &&
             dependency.queue_sizing_recipe == relay_recipe->id;
    }));
  }

  EXPECT_EQ(source_slots[0], source_slots[1]);
  EXPECT_EQ(relay_slots[0], relay_slots[1]);
  EXPECT_EQ(relay_depths[0], relay_depths[1]);
  EXPECT_EQ(relay_depths[0], 30U);
  EXPECT_NE(source_bytes[0], source_bytes[1]);
  EXPECT_NE(relay_bytes[0], relay_bytes[1]);
}

TEST(TensorByteSpanDerivationCompile, GenerationZeroOverrideChangesCapacityNotGraphIdentity) {
  // Graph topology:
  //   source (isolated derived raw Tensor output)
  Graph graph{"derived-override-capacity"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  const GraphAuthoringIdentity identity = graph.view().identity();

  TensorByteSpanTestParams resolved;
  resolved.bytes = 8192U;
  CompileOptions options;
  options.parameter_overrides = tensor_byte_span_override(graph, resolved, "bytes");
  // Key setup: compile one immutable graph with an accepted generation-zero value override.
  const ExecutionPlan plan = compile(graph, std::move(options));

  // Key assertion: the plan capacity changes without changing declaration identity or rebuilding.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(graph.view().identity(), identity);
  EXPECT_EQ(only_allocation(plan).bounds.max_byte_span, 8192U);
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_byte_span, 8192U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ResolvedValuesDoNotEnterGraphDeclarationHash) {
  // Graph topologies:
  //   baseline: source (1000-byte resolved Parameter)
  //   resized:  source (8192-byte resolved Parameter)
  Graph baseline{"derived-hash-invariance"};
  test::add_compile_test_default_clock(baseline);
  auto baseline_source = baseline.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  baseline_source.set_params(TensorByteSpanTestParams{.bytes = 1000U});

  Graph resized{"derived-hash-invariance"};
  test::add_compile_test_default_clock(resized);
  auto resized_source = resized.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  resized_source.set_params(TensorByteSpanTestParams{.bytes = 8192U});

  // Key setup: two equivalent declarations differ only in their generation-zero Parameter value.
  const ExecutionPlan baseline_plan = compile(baseline);
  const ExecutionPlan resized_plan = compile(resized);

  // Key assertion: resolved capacity affects physical allocation but not declaration identity.
  ASSERT_TRUE(baseline_plan.ok()) << baseline_plan.json();
  ASSERT_TRUE(resized_plan.ok()) << resized_plan.json();
  EXPECT_EQ(baseline.view().identity().hash, resized.view().identity().hash);
  EXPECT_EQ(only_allocation(baseline_plan).bounds.max_byte_span, 1000U);
  EXPECT_EQ(only_allocation(resized_plan).bounds.max_byte_span, 8192U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ReducesBoundedShapeParameterUsingExactDtypeAndRank) {
  // Graph topology:
  //   source (isolated shape-derived raw Tensor output)
  Graph graph{"derived-shape-capacity"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<ShapeDerivedTensorSource>("source");

  TensorByteSpanShapeTestParams resolved;
  resolved.shape = {32U, 48U};
  source.set_params(resolved);
  // Key setup: a bounded two-element shape is reduced with the authored uint8 dtype.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: Phase 1 derives bytes only; it does not fabricate per-axis bounds.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto& allocation = only_allocation(plan);
  EXPECT_EQ(allocation.bounds.max_byte_span, 32U * 48U);
  EXPECT_TRUE(allocation.axes.empty());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, EnforcesMutabilityAndCheckedArithmetic) {
  // Graph topologies (each graph contains one isolated derived source):
  //   frozen | live | lifecycle | overflow | shape_overflow | align_overflow
  Graph frozen{"derived-frozen-capacity"};
  test::add_compile_test_default_clock(frozen);
  (void)frozen.op<DerivedTensorSource>("source", DerivationCase::kFrozenScalar);
  // Key setup: a compile-time immutable field supplies frozen generation-zero geometry.
  const ExecutionPlan frozen_plan = compile(frozen);
  ASSERT_TRUE(frozen_plan.ok()) << frozen_plan.json();
  ASSERT_EQ(frozen_plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(frozen_plan.tensor_byte_span_derivations().front().update_disposition,
            TensorByteSpanDerivationUpdateDisposition::kFrozenCompileTime);

  Graph live{"derived-live-capacity"};
  test::add_compile_test_default_clock(live);
  (void)live.op<DerivedTensorSource>("source", DerivationCase::kLiveScalar);
  // Key setup: Runtime-mutable state cannot define immutable materialized pool geometry.
  const ExecutionPlan live_plan = compile(live);
  ASSERT_FALSE(live_plan.ok());
  EXPECT_TRUE(
      test::has_diagnostic(live_plan, "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID"));

  Graph lifecycle{"derived-lifecycle-capacity"};
  test::add_compile_test_default_clock(lifecycle);
  (void)lifecycle.op<DerivedTensorSource>("source", DerivationCase::kLifecycleScalar);
  const ExecutionPlan lifecycle_plan = compile(lifecycle);
  ASSERT_FALSE(lifecycle_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(lifecycle_plan,
                                   "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID"));

  Graph overflow{"derived-overflow-capacity"};
  test::add_compile_test_default_clock(overflow);
  (void)overflow.op<DerivedTensorSource>("source", DerivationCase::kOverflow);
  const ExecutionPlan overflow_plan = compile(overflow);
  // Key assertion: checked arithmetic fails compilation instead of wrapping pool capacity.
  ASSERT_FALSE(overflow_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(overflow_plan, "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW"));

  Graph shape_overflow{"derived-shape-overflow-capacity"};
  test::add_compile_test_default_clock(shape_overflow);
  auto shape_source = shape_overflow.op<WideShapeDerivedTensorSource>("source");
  shape_source.set_params(TensorByteSpanShapeTestParams{
      .shape = {std::numeric_limits<std::uint32_t>::max(),
                std::numeric_limits<std::uint32_t>::max()},
  });
  const ExecutionPlan shape_overflow_plan = compile(shape_overflow);
  ASSERT_FALSE(shape_overflow_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(shape_overflow_plan, "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW"));

  Graph align_overflow{"derived-align-overflow-capacity"};
  test::add_compile_test_default_clock(align_overflow);
  (void)align_overflow.op<DerivedTensorSource>("source", DerivationCase::kAlignOverflow);
  const ExecutionPlan align_overflow_plan = compile(align_overflow);
  // Key assertion: multiplication, contiguous-shape reduction, and align-up all fail on overflow.
  ASSERT_FALSE(align_overflow_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(align_overflow_plan, "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW"));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsOverflowedDerivedPoolWithoutPassThroughService) {
  // Graph topology:
  //   source --[raw Tensor]--> sink
  Graph graph{"derived-aggregate-overflow"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  const auto sink = graph.op<RawTensorSink>("sink");
  graph.add_flow(source->output, sink->input);
  source.set_params(TensorByteSpanTestParams{.bytes = std::numeric_limits<std::uint64_t>::max()});

  // Key setup: the resolved per-slot span is finite, but aggregate pool bytes overflow.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: rejected direct authority does not become an unpooled service surrogate.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_CAPACITY_OVERFLOW"));
  EXPECT_TRUE(plan.tensor_allocations().empty());
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->process_local_channel_services.empty());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile,
     RejectsOverflowedDerivedDescriptorPoolWithoutPreallocatedService) {
  // Graph topology:
  //   source --[descriptor-backed Image]--> sink
  Graph graph{"derived-descriptor-aggregate-overflow"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<DerivedImageSource>("source");
  const auto sink = graph.op<DerivedImageSink>("sink");
  graph.add_flow(source->output, sink->input);
  source.set_params(TensorByteSpanTestParams{.bytes = std::numeric_limits<std::uint64_t>::max()});

  // Key setup: schema envelope storage is preallocatable, but the derived Tensor pool is not.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: rejected Tensor authority removes every preallocated binding for its edge.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_CAPACITY_OVERFLOW"));
  EXPECT_TRUE(plan.tensor_allocations().empty());
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->payload_edges.empty());
  EXPECT_TRUE(state->channel_locality.empty());
  EXPECT_TRUE(state->process_local_channel_services.empty());
  EXPECT_TRUE(state->process_local_channel_publishers.empty());
  EXPECT_TRUE(state->process_local_channel_subscribers.empty());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsMissingGenerationZeroDerivationCandidate) {
  // Graph topology:
  //   source (isolated output whose replay derivation is omitted)
  Graph graph{"derived-missing-replay-candidate"};
  test::add_compile_test_default_clock(graph);
  MissingReplayDerivationSource::omit_derivation = false;
  (void)graph.op<MissingReplayDerivationSource>("source");

  // Key setup: replay omits the expression retained by the immutable Graph snapshot.
  MissingReplayDerivationSource::omit_derivation = true;
  const ExecutionPlan plan = compile(graph);
  MissingReplayDerivationSource::omit_derivation = false;

  // Key assertion: missing derivation authority is rejected instead of omitting the pool.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID"));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RetainsPortIdentityForEachReplayDivergence) {
  // Graph topology:
  //   source --+--> first  (isolated output)
  //           `--> second (isolated output)
  Graph graph{"derived-replay-divergence-context"};
  test::add_compile_test_default_clock(graph);
  DivergentReplayDerivationSource::diverge = false;
  (void)graph.op<DivergentReplayDerivationSource>("source");

  // Key setup: replay changes both output expressions retained by the immutable snapshot.
  DivergentReplayDerivationSource::diverge = true;
  const ExecutionPlan plan = compile(graph);
  DivergentReplayDerivationSource::diverge = false;

  // Key assertion: the common token remains distinct through each output port identity.
  ASSERT_FALSE(plan.ok());
  std::vector<std::uint64_t> ports;
  for (const auto& diagnostic : plan.diagnostics()) {
    if (diagnostic.token == "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID") {
      ports.push_back(diagnostic.source.port.value());
    }
  }
  std::sort(ports.begin(), ports.end());
  EXPECT_EQ(ports, (std::vector<std::uint64_t>{1U, 2U}));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ReportsReplayDeclarationFailureBeforeCoverageLoss) {
  // Graph topology:
  //   source (isolated output whose replay declaration becomes invalid)
  Graph graph{"derived-replay-declaration-failure"};
  test::add_compile_test_default_clock(graph);
  InvalidReplayDerivationSource::invalidate = false;
  (void)graph.op<InvalidReplayDerivationSource>("source");

  // Key setup: replay records a declaration error after the original snapshot was accepted.
  InvalidReplayDerivationSource::invalidate = true;
  const ExecutionPlan plan = compile(graph);
  InvalidReplayDerivationSource::invalidate = false;

  // Key assertion: setup replay is the sole derivation root cause, not missing-candidate coverage.
  ASSERT_FALSE(plan.ok());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID";
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  EXPECT_EQ(diagnostic->required, "successful Operator setup replay");
  EXPECT_EQ(diagnostic->provided, "operator declaration name is empty");
  EXPECT_EQ(std::count_if(plan.diagnostics().begin(),
                          plan.diagnostics().end(),
                          [](const auto& item) {
                            return item.token == "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID";
                          }),
            1U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsInvalidScalarAndShapeParameterValues) {
  // Graph topologies (each graph contains one isolated invalid derived source):
  //   zero | negative | wrong_type | empty_shape | rank_mismatch | unbounded_shape
  Graph zero{"derived-zero-capacity"};
  test::add_compile_test_default_clock(zero);
  auto zero_source = zero.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  TensorByteSpanTestParams zero_params;
  zero_params.bytes = 0U;
  zero_source.set_params(zero_params);

  Graph negative{"derived-negative-capacity"};
  test::add_compile_test_default_clock(negative);
  auto negative_source = negative.op<DerivedTensorSource>("source", DerivationCase::kSignedScalar);
  TensorByteSpanTestParams negative_params;
  negative_params.signed_bytes = -1;
  negative_source.set_params(negative_params);

  Graph wrong_type{"derived-type-invalid"};
  test::add_compile_test_default_clock(wrong_type);
  (void)wrong_type.op<DerivedTensorSource>("source", DerivationCase::kInvalidType);

  Graph empty_shape{"derived-empty-shape"};
  test::add_compile_test_default_clock(empty_shape);
  auto empty_source = empty_shape.op<ShapeDerivedTensorSource>("source");
  TensorByteSpanShapeTestParams empty_params;
  empty_params.shape.clear();
  empty_source.set_params(empty_params);

  Graph rank_mismatch{"derived-rank-mismatch"};
  test::add_compile_test_default_clock(rank_mismatch);
  auto mismatch_source = rank_mismatch.op<ShapeDerivedTensorSource>("source");
  TensorByteSpanShapeTestParams mismatch_params;
  mismatch_params.shape = {32U};
  mismatch_source.set_params(mismatch_params);

  Graph unbounded_shape{"derived-unbounded-shape"};
  test::add_compile_test_default_clock(unbounded_shape);
  (void)unbounded_shape.op<UnboundedShapeDerivedTensorSource>("source");

  // Key assertion: invalid leaf domains retain phase-specific compiler diagnostics.
  for (Graph* graph : {&zero, &negative, &empty_shape}) {
    const ExecutionPlan plan = compile(*graph);
    ASSERT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_VALUE_INVALID"));
  }
  const ExecutionPlan type_plan = compile(wrong_type);
  ASSERT_FALSE(type_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(type_plan, "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_TYPE_INVALID"));
  const ExecutionPlan rank_plan = compile(rank_mismatch);
  ASSERT_FALSE(rank_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(rank_plan, "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_TYPE_INVALID"));
  const ExecutionPlan unbounded_shape_plan = compile(unbounded_shape);
  ASSERT_FALSE(unbounded_shape_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(unbounded_shape_plan,
                                   "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_TYPE_INVALID"));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsSensitiveCapacityParameterWithNodeContext) {
  // Graph topology:
  //   source (isolated output derived from a sensitive Parameter)
  Graph graph{"derived-sensitive-capacity"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<SensitiveDerivedTensorSource>("source");

  // Key setup: a redacted Parameter is selected as public pool-capacity evidence.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: compilation rejects the dependency without publishing its value.
  ASSERT_FALSE(plan.ok());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID";
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  EXPECT_EQ(diagnostic->json_pointer, "/ports/1/tensor_byte_span_expression/nodes/0");
  EXPECT_EQ(diagnostic->required, "non-sensitive capacity Parameter");
  EXPECT_NE(diagnostic->provided.find("parameter=bytes"), std::string::npos);
  EXPECT_TRUE(plan.tensor_byte_span_derivations().empty());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RetainsDistinctFieldContextForEveryRejectedDependency) {
  // Graph topology:
  //   source (isolated two-field descriptor output with invalid dependencies)
  Graph graph{"derived-multi-field-diagnostics"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedMultiTensorSource>("source", true);

  // Key setup: both descriptor fields depend on the same runtime-mutable Parameter.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: field-scoped JSON pointers prevent diagnostic deduplication.
  ASSERT_FALSE(plan.ok());
  std::vector<std::string> pointers;
  for (const auto& diagnostic : plan.diagnostics()) {
    if (diagnostic.token == "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID") {
      pointers.push_back(diagnostic.json_pointer);
      EXPECT_EQ(diagnostic.required, "compile-time or recompile-required");
      EXPECT_NE(diagnostic.provided.find("parameter=live_bytes"), std::string::npos);
    }
  }
  ASSERT_EQ(pointers.size(), 2U);
  EXPECT_NE(std::find(pointers.begin(),
                      pointers.end(),
                      "/ports/1/tensor_fields/1/tensor_byte_span_expression/nodes/0"),
            pointers.end());
  EXPECT_NE(std::find(pointers.begin(),
                      pointers.end(),
                      "/ports/1/tensor_fields/7/tensor_byte_span_expression/nodes/0"),
            pointers.end());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ResolvesInputCapacityFromLiteralAndDerivedProducers) {
  // Graph topologies:
  //   literal source ----[raw Tensor]--> relay
  //   derived source ----[raw Tensor]--> relay
  //   imported topic ----[raw Tensor]--> relay
  Graph literal_graph{"input-from-literal"};
  test::add_compile_test_default_clock(literal_graph);
  const auto literal = literal_graph.op<LiteralTensorSource>("source", 4096U);
  const auto literal_relay = literal_graph.op<InputDerivedTensorRelay>("relay");
  literal_graph.add_flow(literal->output, literal_relay->input);
  // Key setup: consume the finite allocation fact of a directly connected literal producer.
  const ExecutionPlan literal_plan = compile(literal_graph);
  ASSERT_TRUE(literal_plan.ok()) << literal_plan.json();
  ASSERT_EQ(literal_plan.tensor_allocations().size(), 2U);
  EXPECT_EQ(literal_plan.tensor_allocations()[1].bounds.max_byte_span, 4096U);

  Graph derived_graph{"input-from-derived"};
  test::add_compile_test_default_clock(derived_graph);
  const auto derived = derived_graph.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  const auto derived_relay = derived_graph.op<InputDerivedTensorRelay>("relay");
  derived_graph.add_flow(derived->output, derived_relay->input);
  const ExecutionPlan derived_plan = compile(derived_graph);
  // Key assertion: dependent derivations are evaluated before physical pool lowering.
  ASSERT_TRUE(derived_plan.ok()) << derived_plan.json();
  ASSERT_EQ(derived_plan.tensor_byte_span_derivations().size(), 2U);
  ASSERT_EQ(derived_plan.tensor_allocations().size(), 2U);
  EXPECT_EQ(derived_plan.tensor_allocations()[1].bounds.max_byte_span, 1000U);
  for (const auto& evidence : derived_plan.tensor_byte_span_derivations()) {
    EXPECT_EQ(evidence.update_disposition,
              TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
  }

  Graph imported_graph{"input-from-external-boundary"};
  test::add_compile_test_default_clock(imported_graph);
  TopicOptions options;
  options.endpoint_name = "input-from-external-boundary";
  options.tensor = TensorBoundaryContract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
          },
      .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 2048U},
  };
  const auto imported = imported_graph.import_topic<Tensor>("holomq://input-from-external-boundary",
                                                            std::move(options));
  const auto imported_relay = imported_graph.op<InputDerivedTensorRelay>("relay");
  imported_graph.add_flow(imported.output(), imported_relay->input);
  const ExecutionPlan imported_plan = compile(imported_graph);
  // Key assertion: an imported boundary's normalized finite contract is valid scalar evidence.
  ASSERT_TRUE(imported_plan.ok()) << imported_plan.json();
  ASSERT_EQ(imported_plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(imported_plan.tensor_byte_span_derivations().front().resolved_max_byte_span, 2048U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RetainsSetupAllocationForInputCapacityRefinement) {
  // Graph topology:
  //   literal source --[raw Tensor]--> input-derived relay
  Graph graph{"input-from-retained-setup-allocation"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<LiteralTensorSource>("source", 4096U);
  const auto relay = graph.op<InputDerivedTensorRelay>("relay");
  graph.add_flow(source->output, relay->input);

  // Key setup: a pass-through logical refinement does not replace the setup-owned pool.
  graph.configure(source->output)
      .produces_tensor(TensorOutputSpec{
          .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}},
          .storage = TensorOutputStorage::kPassThrough,
      });
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::ranges::find(snapshot->vertices, std::string{"source"}, &VertexRecord::name);
  ASSERT_NE(source_vertex, snapshot->vertices.end());
  const auto source_port =
      std::ranges::find(snapshot->ports, source_vertex->id, &PortRecord::owner);
  ASSERT_NE(source_port, snapshot->ports.end());
  ASSERT_TRUE(source_port->setup.tensor_contract);
  const auto setup_storage = source_port->setup.tensor_contract->storage;
  EXPECT_EQ(setup_storage, TensorOutputStorage::kRuntimePool);
  ASSERT_TRUE(source_port->setup.tensor_allocation);
  const ExecutionPlan plan = compile(graph);

  // Key assertion: input_max_byte_span still consumes the retained setup allocation authority.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_byte_span, 4096U);
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);
  EXPECT_EQ(plan.tensor_allocations().back().bounds.max_byte_span, 4096U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsInputBoundFromSupersededSetupContract) {
  // Graph topology:
  //   bounded pass-through source --[raw Tensor]--> input-derived relay
  Graph graph{"input-from-superseded-setup"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<BoundedPassThroughTensorSource>("source");
  const auto relay = graph.op<InputDerivedTensorRelay>("relay");
  graph.add_flow(source->output, relay->input);

  // Key setup: the effective Graph override replaces the setup contract and removes its bound.
  graph.configure(source->output)
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
          .storage = TensorOutputStorage::kPassThrough,
      });
  const ExecutionPlan plan = compile(graph);

  // Key assertion: input derivation cannot fall back to superseded setup authority.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_INPUT_UNBOUNDED"));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsDisconnectedAndUnboundedInputDependencies) {
  // Graph topologies:
  //   disconnected: relay (unconnected input)
  //   unbounded: source --[unbounded raw Tensor]--> relay
  Graph disconnected{"input-disconnected"};
  test::add_compile_test_default_clock(disconnected);
  (void)disconnected.op<InputDerivedTensorRelay>("relay");
  // Key setup: the referenced owning input has no incident edge.
  const ExecutionPlan disconnected_plan = compile(disconnected);
  ASSERT_FALSE(disconnected_plan.ok());
  EXPECT_TRUE(
      test::has_diagnostic(disconnected_plan, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"));

  Graph unbounded{"input-unbounded"};
  test::add_compile_test_default_clock(unbounded);
  const auto source = unbounded.op<UnboundedTensorSource>("source");
  const auto relay = unbounded.op<InputDerivedTensorRelay>("relay");
  unbounded.add_flow(source->output, relay->input);
  const ExecutionPlan unbounded_plan = compile(unbounded);
  // Key assertion: consumer declarations alone cannot invent finite producer authority.
  ASSERT_FALSE(unbounded_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(unbounded_plan, "PHYSICAL_TENSOR_BYTE_SPAN_INPUT_UNBOUNDED"));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ResolvesInputDependencyAcrossEmbeddedGraphBoundaries) {
  // Graph topologies:
  //   parent source --[raw Tensor]--> child/input --> child/relay
  //   child/source --> child/output --[raw Tensor]--> parent relay
  Graph parent_to_child{"input-parent-to-child"};
  const ClockKey parent_to_child_clock = test::add_compile_test_default_clock(parent_to_child);
  const auto parent_source = parent_to_child.op<LiteralTensorSource>("source", 4096U);
  const auto child_relay = parent_to_child.embed<InputDerivedTensorChildGraph>("child");
  parent_to_child.bind_clock(child_relay, child_relay->clock_import(), parent_to_child_clock);

  // Key setup: resolve a parent producer through an embedded Graph's public input alias.
  parent_to_child.add_flow(parent_source->output, child_relay->input);
  const ExecutionPlan parent_to_child_plan = compile(parent_to_child);

  // Key assertion: the child Operator derives the concrete parent producer's finite capacity.
  ASSERT_TRUE(parent_to_child_plan.ok()) << parent_to_child_plan.json();
  ASSERT_EQ(parent_to_child_plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(parent_to_child_plan.tensor_byte_span_derivations().front().resolved_max_byte_span,
            4096U);
  EXPECT_EQ(parent_to_child_plan.find_tensor_byte_span_derivation("child/relay", "output"),
            &parent_to_child_plan.tensor_byte_span_derivations().front());

  Graph child_to_parent{"input-child-to-parent"};
  const ClockKey child_to_parent_clock = test::add_compile_test_default_clock(child_to_parent);
  const auto child_source = child_to_parent.embed<LiteralTensorSourceChildGraph>("child");
  child_to_parent.bind_clock(child_source, child_source->clock_import(), child_to_parent_clock);
  const auto parent_relay = child_to_parent.op<InputDerivedTensorRelay>("relay");

  // Key setup: resolve an embedded producer through the child Graph's public output alias.
  child_to_parent.add_flow(child_source->output, parent_relay->input);
  const ExecutionPlan child_to_parent_plan = compile(child_to_parent);

  // Key assertion: the parent Operator derives the concrete child producer's finite capacity.
  ASSERT_TRUE(child_to_parent_plan.ok()) << child_to_parent_plan.json();
  ASSERT_EQ(child_to_parent_plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(child_to_parent_plan.tensor_byte_span_derivations().front().resolved_max_byte_span,
            8192U);
  EXPECT_EQ(child_to_parent_plan.find_tensor_byte_span_derivation("relay", "output"),
            &child_to_parent_plan.tensor_byte_span_derivations().front());
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, MatchesEquivalentLiteralAllocationSemantics) {
  // Graph topologies:
  //   derived_graph: source (derived 1000-byte pool)
  //   literal_graph: source (literal 1000-byte pool)
  Graph derived_graph{"literal-twin"};
  test::add_compile_test_default_clock(derived_graph);
  (void)derived_graph.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  const ExecutionPlan derived = compile(derived_graph);

  Graph literal_graph{"literal-twin"};
  test::add_compile_test_default_clock(literal_graph);
  (void)literal_graph.op<LiteralTensorSource>("source", 1000U);
  const ExecutionPlan literal = compile(literal_graph);

  // Key assertion: derivation changes evidence, not the ordinary physical allocation geometry.
  ASSERT_TRUE(derived.ok()) << derived.json();
  ASSERT_TRUE(literal.ok()) << literal.json();
  const auto& derived_allocation = only_allocation(derived);
  const auto& literal_allocation = only_allocation(literal);
  EXPECT_EQ(derived_allocation.output, literal_allocation.output);
  EXPECT_EQ(derived_allocation.bounds.max_rank, literal_allocation.bounds.max_rank);
  EXPECT_EQ(derived_allocation.bounds.max_byte_span, literal_allocation.bounds.max_byte_span);
  EXPECT_EQ(derived_allocation.axes, literal_allocation.axes);
  ASSERT_EQ(derived_allocation.layout.has_value(), literal_allocation.layout.has_value());
  ASSERT_TRUE(derived_allocation.layout);
  EXPECT_EQ(derived_allocation.layout->memory_kind, literal_allocation.layout->memory_kind);
  ASSERT_EQ(derived_allocation.layout->dtype.has_value(),
            literal_allocation.layout->dtype.has_value());
  ASSERT_TRUE(derived_allocation.layout->dtype);
  EXPECT_EQ(derived_allocation.layout->dtype->code, literal_allocation.layout->dtype->code);
  EXPECT_EQ(derived_allocation.layout->dtype->bits, literal_allocation.layout->dtype->bits);
  EXPECT_EQ(derived_allocation.layout->dtype->lanes, literal_allocation.layout->dtype->lanes);
  EXPECT_EQ(derived_allocation.layout->rank, literal_allocation.layout->rank);
  EXPECT_EQ(derived_allocation.layout->strides, literal_allocation.layout->strides);
  EXPECT_EQ(derived_allocation.memory_kind, literal_allocation.memory_kind);
  EXPECT_EQ(derived_allocation.device, literal_allocation.device);
  EXPECT_EQ(derived_allocation.slot_count, literal_allocation.slot_count);
  EXPECT_EQ(derived_allocation.aggregate_byte_capacity, literal_allocation.aggregate_byte_capacity);
  EXPECT_EQ(derived_allocation.max_outstanding_loans, literal_allocation.max_outstanding_loans);
  EXPECT_EQ(derived_allocation.control_capacity, literal_allocation.control_capacity);
  EXPECT_EQ(derived_allocation.readiness_capacity, literal_allocation.readiness_capacity);
  EXPECT_EQ(derived_allocation.release_capacity, literal_allocation.release_capacity);
  EXPECT_EQ(derived_allocation.retirement_capacity, literal_allocation.retirement_capacity);
  EXPECT_EQ(derived_allocation.quarantine_capacity, literal_allocation.quarantine_capacity);
  EXPECT_EQ(derived_allocation.disposition, literal_allocation.disposition);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ExportCatalogUsesResolvedByteAndStagingSpans) {
  // Graph topology:
  //   source --[raw Tensor]--> exported topic
  Graph graph{"derived-export-capacity"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  TopicOptions options;
  options.endpoint_name = "derived-export-capacity";
  options.tensor = TensorBoundaryContract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
          },
      .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 4096U},
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
          },
  };
  const auto exported = graph.export_topic<Tensor>("holomq://derived-export-capacity", options);
  graph.add_flow(source->output, exported.input());

  // Key setup: the authored boundary is a safe upper bound wider than the resolved producer.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: catalog identity and physical interface storage share the resolved capacity.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint("derived-export-capacity");
  ASSERT_TRUE(endpoint);
  const auto tensor = endpoint->tensor_requirements();
  ASSERT_TRUE(tensor);
  EXPECT_EQ(tensor->max_byte_span(), 1000U);
  EXPECT_EQ(tensor->max_staging_byte_span(), 1000U);
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  EXPECT_EQ(plan.tensor_interface_pools().front().logical.max_byte_span, 1000U);
  EXPECT_EQ(plan.tensor_interface_pools().front().logical.max_staging_byte_span, 1000U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsExportBoundaryNarrowerThanResolvedCapacity) {
  // Graph topology:
  //   source --[raw Tensor]--> narrower exported topic
  Graph graph{"derived-export-conflict"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedTensorSource>("source", DerivationCase::kScalar);
  TopicOptions options;
  options.endpoint_name = "derived-export-conflict";
  options.tensor = TensorBoundaryContract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
          },
      .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 512U},
  };
  const auto exported = graph.export_topic<Tensor>("holomq://derived-export-conflict", options);
  graph.add_flow(source->output, exported.input());

  // Key setup: the external boundary cannot carry every sample admitted by the producer pool.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: compilation fails before publishing incompatible endpoint authority.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT"));
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ResolvesSingularDescriptorFieldIntoExistingPoolPlan) {
  // Graph topology:
  //   source (isolated derived Image output)
  Graph graph{"derived-singular-descriptor"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedImageSource>("source");

  // Key setup: one descriptor field owns an unresolved Parameter-derived capacity.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: singular descriptor lowering keeps its ordinary one-pool Runtime contract.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().field_id, 4U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().tensor_index, 0U);
  EXPECT_EQ(only_allocation(plan).bounds.max_byte_span, 1000U);

  const TensorOutputSpec override{
      .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
      .bounds = tensor_bounds(512U),
      .storage = TensorOutputStorage::kRuntimePool,
  };
  // Key assertion: a Graph override cannot replace the setup-owned derivation authority.
  try {
    graph.configure(source->output).produces_tensor(override);
    FAIL() << "expected derived descriptor allocation override rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    EXPECT_EQ(make_error(error).message, "GRAPH_TENSOR_DERIVED_ALLOCATION_OVERRIDE_INVALID");
  }
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ReplaysOverrideForSingularDescriptorDerivedOutput) {
  // Graph topology:
  //   source (isolated derived Image output)
  Graph graph{"derived-singular-descriptor-override"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedImageSource>("source");
  TensorByteSpanTestParams resolved;
  resolved.bytes = 8192U;
  CompileOptions options;
  options.parameter_overrides = tensor_byte_span_override(graph, resolved, "bytes");

  // Key setup: generation-zero replay re-runs the singular field-scoped declaration.
  const ExecutionPlan plan = compile(graph, std::move(options));

  // Key assertion: replay canonicalizes to the same collapsed port and uses the override value.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().field_id, 4U);
  EXPECT_EQ(only_allocation(plan).bounds.max_byte_span, 8192U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, CrossProcessSingularDescriptorUsesDerivedStagingSpan) {
  // Graph topology:
  //   source --[cross-process Image]--> sink
  Graph graph{"derived-singular-descriptor-cross-process"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedImageSource>("source");
  const auto sink = graph.op<DerivedImageSink>("sink");
  graph.add_flow(source->output, sink->input);

  // Key setup: the singular descriptor edge crosses process groups without a literal staging cap.
  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, split_source_sink_placement());

  // Key assertion: the resolved derivation supplies the exact cross-process staging authority.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  EXPECT_EQ(state->cross_process_channel_services.front().tensor_max_byte_span, 1000U);
  EXPECT_EQ(state->cross_process_channel_services.front().tensor_emit_policy,
            InterfaceTensorEmitPolicy::kCopyIfNeeded);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, PreservesIndependentRawTensorStagingSpans) {
  struct TestCase {
    std::string_view name;
    std::optional<std::uint64_t> authored;
    std::uint64_t expected;
  };
  constexpr std::array cases{
      TestCase{"omitted", std::nullopt, 1000U},
      TestCase{"smaller", 512U, 512U},
      TestCase{"wider", 8192U, 8192U},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    // Graph topology:
    //   source (isolated derived raw Tensor output)
    Graph graph{"derived-raw-independent-staging"};
    test::add_compile_test_default_clock(graph);
    (void)graph.op<DerivedCopyTensorSource>("source", test_case.authored);

    // Key setup: copy staging is omitted, smaller than, or wider than the data span.
    const ExecutionPlan plan = compile(graph);

    // Key assertion: every valid staging authority is preserved independently of slot bytes.
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
    EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_byte_span, 1000U);
    EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_staging_byte_span,
              test_case.expected);
    ASSERT_EQ(plan.ports().size(), 1U);
    ASSERT_TRUE(plan.ports().front().tensor_contract);
    ASSERT_TRUE(plan.ports().front().tensor_contract->transfer);
    EXPECT_EQ(plan.ports().front().tensor_contract->transfer->max_staging_byte_span,
              test_case.expected);
  }
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, GraphRefinesDerivedOutputTransferWithoutReplacingPool) {
  // Graph topology:
  //   source (isolated derived raw Tensor output)
  Graph graph{"derived-graph-transfer-refinement"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedCopyTensorSource>("source", 8192U);
  graph.configure(source->output)
      .produces_tensor(TensorOutputSpec{
          .storage = TensorOutputStorage::kPassThrough,
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                  .max_staging_byte_span = 512U,
              },
      });

  // Key setup: Graph supplies only logical transfer facts for a setup-derived output pool.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: physical lowering keeps the derived pool and the smaller authored staging cap.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_byte_span, 1000U);
  ASSERT_EQ(plan.ports().size(), 1U);
  ASSERT_TRUE(plan.ports().front().tensor_contract);
  EXPECT_EQ(plan.ports().front().tensor_contract->storage, TensorOutputStorage::kRuntimePool);
  ASSERT_TRUE(plan.ports().front().tensor_contract->transfer);
  EXPECT_EQ(plan.ports().front().tensor_contract->transfer->max_staging_byte_span, 512U);
  EXPECT_EQ(plan.ports().front().tensor_contract->origins.max_staging_byte_span,
            TensorContractValueOrigin::kAuthored);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, GraphLayoutRefinementPreservesSetupTransfer) {
  // Graph topology:
  //   source (isolated derived raw Tensor output)
  Graph graph{"derived-graph-layout-refinement"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedCopyTensorSource>("source", 8192U);

  // Key setup: Graph refines only dtype while setup remains the transfer authority.
  graph.configure(source->output)
      .produces_tensor(TensorOutputSpec{
          .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}},
          .storage = TensorOutputStorage::kPassThrough,
      });
  const ExecutionPlan plan = compile(graph);

  // Key assertion: fieldwise resolution retains setup storage, layout, and staging policy.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_staging_byte_span, 8192U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_byte_span, 1000U);
  ASSERT_EQ(plan.ports().size(), 1U);
  ASSERT_TRUE(plan.ports().front().tensor_contract);
  ASSERT_TRUE(plan.ports().front().tensor_contract->transfer);
  EXPECT_EQ(plan.ports().front().tensor_contract->representation.memory_kind, MemoryKind::kHost);
  EXPECT_EQ(plan.ports().front().tensor_contract->representation.rank, 1U);
  EXPECT_EQ(plan.ports().front().tensor_contract->transfer->max_staging_byte_span, 8192U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, PreservesAuthoredDescriptorStagingSpan) {
  // Graph topology:
  //   source (isolated derived Image output)
  Graph graph{"derived-singular-descriptor-authored-staging"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedImageSource>("source", 8192U);

  // Key setup: copy-capable output authoring supplies a staging cap wider than its data span.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: derivation preserves the authored cap and its provenance in the port contract.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_staging_byte_span, 8192U);
  ASSERT_EQ(plan.ports().size(), 1U);
  ASSERT_TRUE(plan.ports().front().tensor_contract);
  ASSERT_TRUE(plan.ports().front().tensor_contract->transfer);
  EXPECT_EQ(plan.ports().front().tensor_contract->transfer->max_staging_byte_span, 8192U);
  EXPECT_EQ(plan.ports().front().tensor_contract->origins.max_staging_byte_span,
            TensorContractValueOrigin::kAuthored);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, PreservesSmallerDescriptorStagingSpan) {
  // Graph topology:
  //   source (isolated derived Image output)
  Graph graph{"derived-singular-descriptor-smaller-staging"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedImageSource>("source", 512U);

  // Key setup: singular descriptor staging is independently smaller than its 1000-byte payload.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: compilation admits and preserves the authored 512-byte staging authority.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(plan.tensor_byte_span_derivations().front().resolved_max_staging_byte_span, 512U);
  ASSERT_EQ(plan.ports().size(), 1U);
  ASSERT_TRUE(plan.ports().front().tensor_contract);
  ASSERT_TRUE(plan.ports().front().tensor_contract->transfer);
  EXPECT_EQ(plan.ports().front().tensor_contract->transfer->max_staging_byte_span, 512U);
  EXPECT_EQ(plan.ports().front().tensor_contract->origins.max_staging_byte_span,
            TensorContractValueOrigin::kAuthored);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ExportCatalogUsesDerivedDescriptorStagingSpan) {
  constexpr TensorFieldSelector<schema::ImageT> kData{4U, 0U};
  // Graph topology:
  //   source --[Image]--> exported topic
  Graph graph{"derived-singular-descriptor-export-staging"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedImageSource>("source", 8192U);

  DescriptorTensorTopicContract<schema::ImageT> tensors;
  tensors.field(kData,
                TensorBoundaryContract{
                    .representation = {.memory_kind = MemoryKind::kHost,
                                       .dtype = DLDataType{kDLUInt, 8U, 1U},
                                       .rank = 1U},
                    .bounds = tensor_bounds(16384U),
                    .transfer =
                        TensorTransferPolicy{
                            .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                            .max_staging_byte_span = 16384U,
                        },
                });
  TopicOptions options;
  options.endpoint_name = "derived-singular-descriptor-export-staging";
  const auto exported =
      graph.export_topic<schema::ImageT>("holomq://derived-singular-descriptor-export-staging",
                                         std::move(options),
                                         std::move(tensors));
  graph.add_flow(source->output, exported.input());

  // Key setup: the derived data span is 1000 bytes while authored staging requires 8192 bytes.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: endpoint identity and interface storage preserve the distinct staging cap.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.channel_services().size(), 1U);
  ASSERT_EQ(plan.channel_services().front().tensor_fields.size(), 1U);
  EXPECT_EQ(plan.channel_services().front().tensor_fields.front().tensor.max_byte_span, 1000U);
  EXPECT_EQ(plan.channel_services().front().tensor_fields.front().tensor.max_staging_byte_span,
            8192U);
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  EXPECT_EQ(plan.tensor_interface_pools().front().logical.max_byte_span, 1000U);
  EXPECT_EQ(plan.tensor_interface_pools().front().logical.max_staging_byte_span, 8192U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ExportCatalogDefaultsDerivedDescriptorStagingSpan) {
  constexpr TensorFieldSelector<schema::ImageT> kData{4U, 0U};
  // Graph topology:
  //   source --[Image]--> exported topic
  Graph graph{"derived-singular-descriptor-export-default-staging"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedImageSource>("source", std::nullopt, false);

  DescriptorTensorTopicContract<schema::ImageT> tensors;
  tensors.field(kData,
                TensorBoundaryContract{
                    .representation = {.memory_kind = MemoryKind::kHost,
                                       .dtype = DLDataType{kDLUInt, 8U, 1U},
                                       .rank = 1U},
                    .bounds = tensor_bounds(16384U),
                    .transfer =
                        TensorTransferPolicy{
                            .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                            .max_staging_byte_span = 16384U,
                        },
                });
  TopicOptions options;
  options.endpoint_name = "derived-singular-descriptor-export-default-staging";
  const auto exported = graph.export_topic<schema::ImageT>(
      "holomq://derived-singular-descriptor-export-default-staging",
      std::move(options),
      std::move(tensors));
  graph.add_flow(source->output, exported.input());

  // Key setup: the setup-derived descriptor field omits a transfer policy.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: copy-capable export defaults staging to the resolved payload span.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.channel_services().size(), 1U);
  ASSERT_EQ(plan.channel_services().front().tensor_fields.size(), 1U);
  EXPECT_EQ(plan.channel_services().front().tensor_fields.front().tensor.max_byte_span, 1000U);
  EXPECT_EQ(plan.channel_services().front().tensor_fields.front().tensor.max_staging_byte_span,
            1000U);
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  EXPECT_EQ(plan.tensor_interface_pools().front().logical.max_byte_span, 1000U);
  EXPECT_EQ(plan.tensor_interface_pools().front().logical.max_staging_byte_span, 1000U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ResolvesHeterogeneousDescriptorFieldsIndependently) {
  // Graph topology:
  //   source (isolated two-field descriptor output)
  Graph graph{"derived-multi-descriptor"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedMultiTensorSource>("source");

  // Key setup: two fields derive different spans from one immutable Parameter generation.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: each field retains an independent pool identity and release capacity.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 2U);
  ASSERT_EQ(plan.tensor_field_allocations().size(), 2U);
  EXPECT_EQ(plan.tensor_field_allocations()[0].field_id, 1U);
  EXPECT_EQ(plan.tensor_field_allocations()[0].bounds.max_byte_span, 1000U);
  EXPECT_EQ(plan.tensor_field_allocations()[1].field_id, 7U);
  EXPECT_EQ(plan.tensor_field_allocations()[1].bounds.max_byte_span, 2000U);
  EXPECT_NE(plan.tensor_field_allocations()[0].pool_id, plan.tensor_field_allocations()[1].pool_id);
  EXPECT_EQ(plan.tensor_field_allocations()[0].release_capacity,
            plan.tensor_field_allocations()[1].release_capacity);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, ExportCatalogNarrowsResolvedDescriptorFieldSpans) {
  // Graph topology:
  //   source --[two-field descriptor]--> exported topic
  Graph graph{"derived-multi-export"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedMultiTensorSource>("source", false, 512U, 4096U);
  DescriptorTensorTopicContract<DerivedMultiTensorSchema> tensors;
  const auto boundary = [](std::uint64_t bytes, DLDataType dtype, std::uint32_t rank) {
    return TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost, .dtype = dtype, .rank = rank},
        .bounds = tensor_bounds(bytes),
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
            },
    };
  };
  tensors.field(DerivedMultiTensorSource::kImage, boundary(4096U, DLDataType{kDLUInt, 8U, 1U}, 2U));
  tensors.field(DerivedMultiTensorSource::kCalibration,
                boundary(4096U, DLDataType{kDLFloat, 32U, 1U}, 1U));
  TopicOptions options;
  options.endpoint_name = "derived-multi-export";
  const auto exported = graph.export_topic<DerivedMultiTensorSchema>(
      "holomq://derived-multi-export", std::move(options), std::move(tensors));
  graph.add_flow(source->output, exported.input());

  // Key setup: external field bounds safely cover both resolved producer capacities.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: catalog identity carries the resolved byte and staging spans per field.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint("derived-multi-export");
  ASSERT_TRUE(endpoint);
  ASSERT_EQ(plan.channel_services().size(), 1U);
  const auto& fields = plan.channel_services().front().tensor_fields;
  ASSERT_EQ(fields.size(), 2U);
  EXPECT_EQ(fields[0].tensor.max_byte_span, 1000U);
  EXPECT_EQ(fields[0].tensor.max_staging_byte_span, 512U);
  EXPECT_EQ(fields[1].tensor.max_byte_span, 2000U);
  EXPECT_EQ(fields[1].tensor.max_staging_byte_span, 4096U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, PreservesIndependentMultiTensorFieldStagingSpans) {
  // Graph topology:
  //   source (isolated two-field descriptor output)
  Graph graph{"derived-multi-independent-staging"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<DerivedMultiTensorSource>("source", false, 512U, 4096U);

  // Key setup: one descriptor field stages below and one above its resolved payload capacity.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: field-scoped staging bounds remain independent and retain authored provenance.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto derivations = plan.tensor_byte_span_derivations();
  ASSERT_EQ(derivations.size(), 2U);
  const auto find_derivation = [&](const auto selector) {
    return std::find_if(derivations.begin(), derivations.end(), [&](const auto& derivation) {
      return derivation.field_id == selector.field_id &&
             derivation.tensor_index == selector.tensor_index;
    });
  };
  const auto image = find_derivation(DerivedMultiTensorSource::kImage);
  const auto calibration = find_derivation(DerivedMultiTensorSource::kCalibration);
  ASSERT_NE(image, derivations.end());
  ASSERT_NE(calibration, derivations.end());
  EXPECT_EQ(image->resolved_max_byte_span, 1000U);
  EXPECT_EQ(image->resolved_max_staging_byte_span, 512U);
  EXPECT_EQ(calibration->resolved_max_byte_span, 2000U);
  EXPECT_EQ(calibration->resolved_max_staging_byte_span, 4096U);
  ASSERT_EQ(plan.tensor_field_allocations().size(), 2U);
}

// REQ: COMPILE-067
TEST(TensorByteSpanDerivationCompile, RejectsDescriptorExportNarrowerThanResolvedFieldSpan) {
  // Graph topology:
  //   source --[two-field descriptor]--> narrower exported topic
  Graph graph{"derived-multi-export-conflict"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<DerivedMultiTensorSource>("source");
  DescriptorTensorTopicContract<DerivedMultiTensorSchema> tensors;
  const auto boundary = [](std::uint64_t bytes, DLDataType dtype, std::uint32_t rank) {
    return TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost, .dtype = dtype, .rank = rank},
        .bounds = tensor_bounds(bytes),
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
            },
    };
  };
  tensors.field(DerivedMultiTensorSource::kImage, boundary(512U, DLDataType{kDLUInt, 8U, 1U}, 2U));
  tensors.field(DerivedMultiTensorSource::kCalibration,
                boundary(4096U, DLDataType{kDLFloat, 32U, 1U}, 1U));
  TopicOptions options;
  options.endpoint_name = "derived-multi-export-conflict";
  const auto exported = graph.export_topic<DerivedMultiTensorSchema>(
      "holomq://derived-multi-export-conflict", std::move(options), std::move(tensors));
  graph.add_flow(source->output, exported.input());

  // Key setup: one field boundary is narrower than its independently resolved producer span.
  const ExecutionPlan plan = compile(graph);

  // Key assertion: the field conflict rejects the entire external endpoint before admission.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT"));
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
