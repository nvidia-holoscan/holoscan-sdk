/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <tuple>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "compile_test_graph.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "core/parameter_schema_test_access.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

struct NestedProjectionParams {
  std::uint32_t channels{4U};
};

HOLOSCAN_PARAM_SCHEMA(
    NestedProjectionParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("holoscan.test.NestedProjectionParams", 1)),
    holoscan::field<&NestedProjectionParams::channels>("channels").recompile_required());

struct TensorByteSpanProjectionParams {
  std::uint32_t width{10U};
  std::uint32_t height{20U};
  std::uint32_t batch{2U};
  std::uint32_t frozen_scale{1U};
  std::uint32_t runtime_scale{1U};
  std::uint32_t lifecycle_scale{1U};
  std::uint32_t sensitive_scale{1U};
  NestedProjectionParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(
    TensorByteSpanProjectionParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("holoscan.test.TensorByteSpanProjectionParams", 1)),
    holoscan::field<&TensorByteSpanProjectionParams::width>("width").recompile_required(),
    holoscan::field<&TensorByteSpanProjectionParams::height>("height").recompile_required(),
    holoscan::field<&TensorByteSpanProjectionParams::batch>("batch").recompile_required(),
    holoscan::field<&TensorByteSpanProjectionParams::frozen_scale>("frozen_scale"),
    holoscan::field<&TensorByteSpanProjectionParams::runtime_scale>("runtime_scale")
        .runtime_mutable(),
    holoscan::field<&TensorByteSpanProjectionParams::lifecycle_scale>("lifecycle_scale")
        .lifecycle_mutable(LifecycleStage::kConfigure),
    holoscan::field<&TensorByteSpanProjectionParams::sensitive_scale>("sensitive_scale")
        .recompile_required()
        .sensitive(),
    holoscan::field<&TensorByteSpanProjectionParams::nested>("image.shape")
        .nested<NestedProjectionParams>());

// Deliberately duplicates TensorByteSpanProjectionParams' portable schema identity while retaining
// a distinct C++ type. RejectsDistinctCppTypeWithCollidingSchemaIdentity verifies that portable
// schema equality alone never authorizes an exact typed projector call.
struct CollidingTensorByteSpanProjectionParams {
  std::uint32_t width{10U};
  std::uint32_t height{20U};
  std::uint32_t batch{2U};
  std::uint32_t frozen_scale{1U};
  std::uint32_t runtime_scale{1U};
  std::uint32_t lifecycle_scale{1U};
  std::uint32_t sensitive_scale{1U};
  NestedProjectionParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(
    CollidingTensorByteSpanProjectionParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("holoscan.test.TensorByteSpanProjectionParams", 1)),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::width>("width").recompile_required(),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::height>("height")
        .recompile_required(),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::batch>("batch").recompile_required(),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::frozen_scale>("frozen_scale"),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::runtime_scale>("runtime_scale")
        .runtime_mutable(),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::lifecycle_scale>("lifecycle_scale")
        .lifecycle_mutable(LifecycleStage::kConfigure),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::sensitive_scale>("sensitive_scale")
        .recompile_required()
        .sensitive(),
    holoscan::field<&CollidingTensorByteSpanProjectionParams::nested>("image.shape")
        .nested<NestedProjectionParams>());

struct SensitiveProjectionContainerParams {
  NestedProjectionParams child{};
};

HOLOSCAN_PARAM_SCHEMA(SensitiveProjectionContainerParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "holoscan.test.SensitiveProjectionContainerParams", 1)),
                      holoscan::field<&SensitiveProjectionContainerParams::child>("child")
                          .nested<NestedProjectionParams>()
                          .sensitive());

struct SensitiveAncestorProjectionParams {
  NestedProjectionParams root_secret{};
  SensitiveProjectionContainerParams container{};
};

HOLOSCAN_PARAM_SCHEMA(
    SensitiveAncestorProjectionParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("holoscan.test.SensitiveAncestorProjectionParams", 1)),
    holoscan::field<&SensitiveAncestorProjectionParams::root_secret>("secret.root")
        .nested<NestedProjectionParams>()
        .sensitive(),
    holoscan::field<&SensitiveAncestorProjectionParams::container>("container")
        .nested<SensitiveProjectionContainerParams>());

struct TensorByteSpanRuntimeProjectionParams {
  std::uint32_t width{10U};
  std::uint32_t height{20U};
  std::uint32_t batch{2U};
};

HOLOSCAN_PARAM_SCHEMA(
    TensorByteSpanRuntimeProjectionParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("holoscan.test.TensorByteSpanRuntimeProjectionParams", 1)),
    holoscan::field<&TensorByteSpanRuntimeProjectionParams::width>("width").recompile_required(),
    holoscan::field<&TensorByteSpanRuntimeProjectionParams::height>("height").recompile_required(),
    holoscan::field<&TensorByteSpanRuntimeProjectionParams::batch>("batch").recompile_required());

// Supplies colliding dotted display paths with distinct typed member boundaries to
// ResolvesCollidingDisplayPathsFromCompleteTypedIdentity.
struct AmbiguousProjectionLeafParams {
  std::uint64_t bytes{17U};
};

HOLOSCAN_PARAM_SCHEMA(
    AmbiguousProjectionLeafParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("holoscan.test.AmbiguousProjectionLeafParams", 1)),
    holoscan::field<&AmbiguousProjectionLeafParams::bytes>("bytes").recompile_required());

struct AmbiguousProjectionInnerParams {
  AmbiguousProjectionLeafParams b{};
};

HOLOSCAN_PARAM_SCHEMA(AmbiguousProjectionInnerParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "holoscan.test.AmbiguousProjectionInnerParams", 1)),
                      holoscan::field<&AmbiguousProjectionInnerParams::b>("b")
                          .nested<AmbiguousProjectionLeafParams>());

struct AmbiguousProjectionBranchParams {
  AmbiguousProjectionLeafParams dotted{};
  AmbiguousProjectionInnerParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(AmbiguousProjectionBranchParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "holoscan.test.AmbiguousProjectionBranchParams", 1)),
                      holoscan::field<&AmbiguousProjectionBranchParams::dotted>("a.b")
                          .nested<AmbiguousProjectionLeafParams>(),
                      holoscan::field<&AmbiguousProjectionBranchParams::nested>("a")
                          .nested<AmbiguousProjectionInnerParams>());

struct AmbiguousProjectionSharedRootParams {
  AmbiguousProjectionBranchParams outer{};
};

HOLOSCAN_PARAM_SCHEMA(AmbiguousProjectionSharedRootParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "holoscan.test.AmbiguousProjectionSharedRootParams", 1)),
                      holoscan::field<&AmbiguousProjectionSharedRootParams::outer>("outer")
                          .nested<AmbiguousProjectionBranchParams>());

namespace {

std::atomic<std::uint32_t> g_projection_invocations{};
std::atomic<std::uint32_t> g_runtime_configures{};
std::atomic<std::uint32_t> g_runtime_computes{};
std::atomic_bool g_runtime_restart_accepted{};

[[nodiscard]] expected<std::uint64_t, Error> checked_projection(
    const TensorByteSpanProjectionParams& params) noexcept {
  g_projection_invocations.fetch_add(1U, std::memory_order_relaxed);
  constexpr std::uint64_t kElementBytes = 4U;
  const auto width = static_cast<std::uint64_t>(params.width);
  const auto height = static_cast<std::uint64_t>(params.height);
  const auto batch = static_cast<std::uint64_t>(params.batch);
  if (height != 0U && width > std::numeric_limits<std::uint64_t>::max() / height) {
    return make_unexpected(
        Error{ErrorCode::kArithmeticOverflow, "private arithmetic overflow detail"});
  }
  const std::uint64_t pixels = width * height;
  if (batch != 0U && pixels > std::numeric_limits<std::uint64_t>::max() / batch) {
    return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
  }
  const std::uint64_t samples = pixels * batch;
  if (samples > std::numeric_limits<std::uint64_t>::max() / kElementBytes) {
    return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
  }
  return samples * kElementBytes;
}

[[nodiscard]] expected<std::uint64_t, Error> colliding_schema_projection(
    const CollidingTensorByteSpanProjectionParams& params) noexcept {
  return static_cast<std::uint64_t>(params.width) * params.height * params.batch;
}

[[nodiscard]] expected<std::uint64_t, Error> runtime_checked_projection(
    const TensorByteSpanRuntimeProjectionParams& params) noexcept {
  g_projection_invocations.fetch_add(1U, std::memory_order_relaxed);
  const auto width = static_cast<std::uint64_t>(params.width);
  const auto height = static_cast<std::uint64_t>(params.height);
  const auto batch = static_cast<std::uint64_t>(params.batch);
  if (height != 0U && width > std::numeric_limits<std::uint64_t>::max() / height) {
    return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
  }
  const std::uint64_t pixels = width * height;
  if (batch != 0U && pixels > std::numeric_limits<std::uint64_t>::max() / batch) {
    return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
  }
  return pixels * batch;
}

[[nodiscard]] expected<std::uint64_t, Error> ambiguous_path_projection(
    const AmbiguousProjectionBranchParams& params) noexcept {
  return params.dotted.bytes + params.nested.b.bytes;
}

[[nodiscard]] expected<std::uint64_t, Error> ambiguous_path_identity_projection(
    const AmbiguousProjectionBranchParams&) noexcept {
  return 17U;
}

[[nodiscard]] expected<std::uint64_t, Error> ambiguous_shared_path_projection(
    const AmbiguousProjectionSharedRootParams& params) noexcept {
  return params.outer.dotted.bytes + params.outer.nested.b.bytes;
}

struct ProjectionFunctions {
  [[nodiscard]] static expected<std::uint64_t, Error> alternate(
      const TensorByteSpanProjectionParams& params) noexcept {
    return static_cast<std::uint64_t>(params.width) + params.height;
  }
};

[[nodiscard]] expected<std::uint64_t, Error> frozen_projection(
    const TensorByteSpanProjectionParams& params) noexcept {
  return params.frozen_scale;
}

[[nodiscard]] expected<std::uint64_t, Error> runtime_projection(
    const TensorByteSpanProjectionParams& params) noexcept {
  return params.runtime_scale;
}

[[nodiscard]] expected<std::uint64_t, Error> lifecycle_projection(
    const TensorByteSpanProjectionParams& params) noexcept {
  return params.lifecycle_scale;
}

[[nodiscard]] expected<std::uint64_t, Error> sensitive_projection(
    const TensorByteSpanProjectionParams& params) noexcept {
  return params.sensitive_scale;
}

[[nodiscard]] expected<std::uint64_t, Error> nested_projection(
    const TensorByteSpanProjectionParams& params) noexcept {
  return static_cast<std::uint64_t>(params.width) * params.nested.channels;
}

[[nodiscard]] expected<std::uint64_t, Error> sensitive_root_projection(
    const SensitiveAncestorProjectionParams& params) noexcept {
  g_projection_invocations.fetch_add(1U, std::memory_order_relaxed);
  return params.root_secret.channels;
}

[[nodiscard]] expected<std::uint64_t, Error> sensitive_intermediate_projection(
    const SensitiveAncestorProjectionParams& params) noexcept {
  g_projection_invocations.fetch_add(1U, std::memory_order_relaxed);
  return params.container.child.channels;
}

[[nodiscard]] expected<std::uint64_t, Error> rejected_projection(
    const TensorByteSpanProjectionParams&) noexcept {
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, "private projector implementation detail"});
}

[[nodiscard]] expected<std::uint64_t, Error> overflow_projection(
    const TensorByteSpanProjectionParams&) noexcept {
  return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
}

[[nodiscard]] expected<std::uint64_t, Error> overflow_message_projection(
    const TensorByteSpanProjectionParams&) noexcept {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW"});
}

[[nodiscard]] expected<std::uint64_t, Error> zero_projection(
    const TensorByteSpanProjectionParams&) noexcept {
  return 0U;
}

enum class ProjectionCase : std::uint8_t {
  kChecked,
  kReversedDependencies,
  kNested,
  kFrozen,
  kRuntimeMutable,
  kLifecycleMutable,
  kSensitive,
  kRejected,
  kOverflow,
  kOverflowMessage,
  kZero,
  kAlternate,
};

class ProjectedTensorSource final : public Operator<TensorByteSpanProjectionParams> {
 public:
  explicit ProjectedTensorSource(ProjectionCase projection = ProjectionCase::kChecked,
                                 std::string_view stable_id = kProjectionId)
      : projection_(projection), stable_id_(stable_id) {}

  void setup(OperatorSpec& spec) override {
    TensorByteSpanExpression expression;
    switch (projection_) {
      case ProjectionCase::kChecked:
        expression = parameter_projected_byte_span<&checked_projection>(
            stable_id_,
            param_ref<&TensorByteSpanProjectionParams::width>(),
            param_ref<&TensorByteSpanProjectionParams::height>(),
            param_ref<&TensorByteSpanProjectionParams::batch>());
        break;
      case ProjectionCase::kReversedDependencies:
        expression = parameter_projected_byte_span<&checked_projection>(
            stable_id_,
            param_ref<&TensorByteSpanProjectionParams::batch>(),
            param_ref<&TensorByteSpanProjectionParams::height>(),
            param_ref<&TensorByteSpanProjectionParams::width>());
        break;
      case ProjectionCase::kNested:
        expression = parameter_projected_byte_span<&nested_projection>(
            stable_id_,
            param_ref<&TensorByteSpanProjectionParams::nested, &NestedProjectionParams::channels>(),
            param_ref<&TensorByteSpanProjectionParams::width>());
        break;
      case ProjectionCase::kFrozen:
        expression = parameter_projected_byte_span<&frozen_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::frozen_scale>());
        break;
      case ProjectionCase::kRuntimeMutable:
        expression = parameter_projected_byte_span<&runtime_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::runtime_scale>());
        break;
      case ProjectionCase::kLifecycleMutable:
        expression = parameter_projected_byte_span<&lifecycle_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::lifecycle_scale>());
        break;
      case ProjectionCase::kSensitive:
        expression = parameter_projected_byte_span<&sensitive_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::sensitive_scale>());
        break;
      case ProjectionCase::kRejected:
        expression = parameter_projected_byte_span<&rejected_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::width>());
        break;
      case ProjectionCase::kOverflow:
        expression = parameter_projected_byte_span<&overflow_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::width>());
        break;
      case ProjectionCase::kOverflowMessage:
        expression = parameter_projected_byte_span<&overflow_message_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::width>());
        break;
      case ProjectionCase::kZero:
        expression = parameter_projected_byte_span<&zero_projection>(
            stable_id_, param_ref<&TensorByteSpanProjectionParams::width>());
        break;
      case ProjectionCase::kAlternate:
        expression = parameter_projected_byte_span<&ProjectionFunctions::alternate>(
            stable_id_,
            param_ref<&TensorByteSpanProjectionParams::width>(),
            param_ref<&TensorByteSpanProjectionParams::height>());
        break;
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            expression.aligned_to(256U).maximum_with(tensor_byte_span(1024U)));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  static constexpr std::string_view kProjectionId =
      "holoscan.test.ProjectedTensorSource.output-bytes.v1";
  Output<Tensor> output;

 private:
  ProjectionCase projection_;
  std::string stable_id_;
};

class LiteralTensorSource final : public Operator<> {
 public:
  explicit LiteralTensorSource(std::uint64_t bytes) : bytes_(bytes) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 2U},
            .bounds = tensor_bounds(bytes_),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  std::uint64_t bytes_{};
};

class ForeignOwnerProjectedSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_projected_byte_span<&checked_projection>(
                "holoscan.test.ForeignOwnerProjectedSource.output-bytes.v1",
                param_ref<&TensorByteSpanProjectionParams::width>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Owns TensorByteSpanProjectionParams but authors a projector for the deliberately colliding type.
// RejectsDistinctCppTypeWithCollidingSchemaIdentity exercises its exact-type rejection.
class CollidingSchemaProjectedSource final : public Operator<TensorByteSpanProjectionParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_projected_byte_span<&colliding_schema_projection>(
                "holoscan.test.CollidingSchemaProjectedSource.output-bytes.v1",
                param_ref<&CollidingTensorByteSpanProjectionParams::width>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class ProjectedImageSource final : public Operator<TensorByteSpanProjectionParams> {
 public:
  explicit ProjectedImageSource(std::string_view stable_id = kProjectionId)
      : stable_id_(stable_id) {}

  void setup(OperatorSpec& spec) override {
    constexpr TensorFieldSelector<schema::ImageT> kImage{4U, 0U};
    spec.output(output, "output")
        .tensor_field(kImage)
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            parameter_projected_byte_span<&nested_projection>(
                stable_id_,
                param_ref<&TensorByteSpanProjectionParams::width>(),
                param_ref<&TensorByteSpanProjectionParams::nested,
                          &NestedProjectionParams::channels>()));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  static constexpr std::string_view kProjectionId =
      "holoscan.test.ProjectedImageSource.image-bytes.v1";
  Output<schema::ImageT> output;

 private:
  std::string stable_id_;
};

// Projects two typed paths with one dotted display path. The bool specialization selects whether
// the ambiguity starts at the Params root or below a shared root; the resolving test uses both.
template <bool SharedRoot>
class AmbiguousPathProjectedSource final
    : public Operator<std::conditional_t<SharedRoot, AmbiguousProjectionSharedRootParams,
                                         AmbiguousProjectionBranchParams>> {
 public:
  explicit AmbiguousPathProjectedSource(bool reverse_dependencies)
      : reverse_dependencies_(reverse_dependencies) {}

  void setup(OperatorSpec& spec) override {
    TensorByteSpanExpression expression;
    if constexpr (SharedRoot) {
      const auto dotted = param_ref<&AmbiguousProjectionSharedRootParams::outer,
                                    &AmbiguousProjectionBranchParams::dotted,
                                    &AmbiguousProjectionLeafParams::bytes>();
      const auto nested = param_ref<&AmbiguousProjectionSharedRootParams::outer,
                                    &AmbiguousProjectionBranchParams::nested,
                                    &AmbiguousProjectionInnerParams::b,
                                    &AmbiguousProjectionLeafParams::bytes>();
      expression = reverse_dependencies_
                       ? parameter_projected_byte_span<&ambiguous_shared_path_projection>(
                             "holoscan.test.ambiguous-shared-path-projector.v1", nested, dotted)
                       : parameter_projected_byte_span<&ambiguous_shared_path_projection>(
                             "holoscan.test.ambiguous-shared-path-projector.v1", dotted, nested);
    } else {
      const auto dotted = param_ref<&AmbiguousProjectionBranchParams::dotted,
                                    &AmbiguousProjectionLeafParams::bytes>();
      const auto nested = param_ref<&AmbiguousProjectionBranchParams::nested,
                                    &AmbiguousProjectionInnerParams::b,
                                    &AmbiguousProjectionLeafParams::bytes>();
      expression = reverse_dependencies_
                       ? parameter_projected_byte_span<&ambiguous_path_projection>(
                             "holoscan.test.ambiguous-path-projector.v1", nested, dotted)
                       : parameter_projected_byte_span<&ambiguous_path_projection>(
                             "holoscan.test.ambiguous-path-projector.v1", dotted, nested);
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            std::move(expression));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  bool reverse_dependencies_{};
};

// Switches one constant projector between typed paths that have the same display path.
// ResolvesCollidingDisplayPathsFromCompleteTypedIdentity uses it to isolate Graph identity.
class AmbiguousPathIdentitySource final : public Operator<AmbiguousProjectionBranchParams> {
 public:
  explicit AmbiguousPathIdentitySource(bool nested) : nested_(nested) {}

  void setup(OperatorSpec& spec) override {
    TensorByteSpanExpression expression;
    if (nested_) {
      expression = parameter_projected_byte_span<&ambiguous_path_identity_projection>(
          "holoscan.test.ambiguous-path-identity-projector.v1",
          param_ref<&AmbiguousProjectionBranchParams::nested,
                    &AmbiguousProjectionInnerParams::b,
                    &AmbiguousProjectionLeafParams::bytes>());
    } else {
      expression = parameter_projected_byte_span<&ambiguous_path_identity_projection>(
          "holoscan.test.ambiguous-path-identity-projector.v1",
          param_ref<&AmbiguousProjectionBranchParams::dotted,
                    &AmbiguousProjectionLeafParams::bytes>());
    }
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost,
                                   .dtype = DLDataType{kDLUInt, 8U, 1U},
                                   .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            std::move(expression));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  bool nested_{};
};

enum class SensitiveAncestorCase : std::uint8_t { kRoot, kIntermediate };

// Authors a raw or descriptor Tensor projection through a sensitive nested ancestor.
// RejectsSensitiveNestedAncestorsBeforeInvokingProjector verifies the compiler rejects it.
class SensitiveAncestorProjectedSource final : public Operator<SensitiveAncestorProjectionParams> {
 public:
  SensitiveAncestorProjectedSource(SensitiveAncestorCase ancestor, bool descriptor)
      : ancestor_(ancestor), descriptor_(descriptor) {}

  void setup(OperatorSpec& spec) override {
    TensorByteSpanExpression expression;
    if (ancestor_ == SensitiveAncestorCase::kRoot) {
      expression = parameter_projected_byte_span<&sensitive_root_projection>(
          "holoscan.test.sensitive-root-projector.v1",
          param_ref<&SensitiveAncestorProjectionParams::root_secret,
                    &NestedProjectionParams::channels>());
    } else {
      expression = parameter_projected_byte_span<&sensitive_intermediate_projection>(
          "holoscan.test.sensitive-intermediate-projector.v1",
          param_ref<&SensitiveAncestorProjectionParams::container,
                    &SensitiveProjectionContainerParams::child,
                    &NestedProjectionParams::channels>());
    }
    const TensorOutputSpec contract{
        .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
        .storage = TensorOutputStorage::kRuntimePool,
    };
    if (descriptor_) {
      constexpr TensorFieldSelector<schema::ImageT> kImage{4U, 0U};
      spec.output(descriptor_output, "descriptor_output")
          .tensor_field(kImage)
          .produces_tensor(contract, expression);
    } else {
      spec.output(raw_output, "raw_output").produces_tensor(contract, expression);
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> raw_output;
  Output<schema::ImageT> descriptor_output;

 private:
  SensitiveAncestorCase ancestor_;
  bool descriptor_{};
};

class ProjectedRuntimeProbeSource final : public Operator<TensorByteSpanRuntimeProjectionParams> {
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
            parameter_projected_byte_span<&runtime_checked_projection>(
                "holoscan.test.ProjectedRuntimeProbeSource.output-bytes.v1",
                param_ref<&TensorByteSpanRuntimeProjectionParams::width>(),
                param_ref<&TensorByteSpanRuntimeProjectionParams::height>(),
                param_ref<&TensorByteSpanRuntimeProjectionParams::batch>()));
    spec.lifecycle().stage(LifecycleStage::kConfigure, &ProjectedRuntimeProbeSource::configure);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::uint32_t invocation = g_runtime_computes.fetch_add(1U, std::memory_order_relaxed);
    if (invocation == 0U) {
      const auto restart = context.request_restart();
      g_runtime_restart_accepted.store(restart.has_value(), std::memory_order_release);
    }
    return {};
  }

  LifecycleStatus configure(LifecycleContext&) noexcept {
    g_runtime_configures.fetch_add(1U, std::memory_order_release);
    return LifecycleStatus::kOk;
  }

  Output<Tensor> output;
};

[[nodiscard]] ExecutionPlan compile_projected(
    ProjectionCase projection = ProjectionCase::kChecked,
    TensorByteSpanProjectionParams params = {},
    std::string_view stable_id = ProjectedTensorSource::kProjectionId) {
  Graph graph{"projected-byte-span"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<ProjectedTensorSource>("source", projection, stable_id);
  source.set_params(std::move(params));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan compile_literal(std::uint64_t bytes) {
  Graph graph{"literal-byte-span"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<LiteralTensorSource>("source", bytes);
  return compile(graph);
}

[[nodiscard]] const TensorAllocationPlanRecord& only_allocation(const ExecutionPlan& plan) {
  if (plan.tensor_allocations().size() != 1U) {
    throw std::logic_error{"expected one Tensor allocation"};
  }
  return plan.tensor_allocations().front();
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, EvaluatesOnceAndMatchesLiteralPhysicalAllocation) {
  g_projection_invocations.store(0U, std::memory_order_relaxed);
  const ExecutionPlan projected = compile_projected();
  const ExecutionPlan literal = compile_literal(1792U);

  ASSERT_TRUE(projected.ok()) << projected.json();
  ASSERT_TRUE(literal.ok()) << literal.json();
  EXPECT_EQ(g_projection_invocations.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(only_allocation(projected).bounds.max_byte_span,
            only_allocation(literal).bounds.max_byte_span);
  ASSERT_EQ(projected.tensor_byte_span_derivations().size(), 1U);
  const auto& evidence = projected.tensor_byte_span_derivations().front();
  EXPECT_EQ(evidence.resolved_max_byte_span, 1792U);
  EXPECT_EQ(evidence.update_disposition,
            TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
  const auto projection = std::ranges::find_if(evidence.nodes, [](const auto& node) {
    return node.operation == TensorByteSpanOperation::kParameterProjection;
  });
  ASSERT_NE(projection, evidence.nodes.end());
  EXPECT_EQ(projection->resolved_value, 1600U);
  EXPECT_EQ(projection->projection_id, ProjectedTensorSource::kProjectionId);
  EXPECT_EQ(projection->parameter_schema_id, "holoscan.test.TensorByteSpanProjectionParams");
  ASSERT_EQ(projection->projection_dependencies.size(), 3U);
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, UsesGenerationZeroValuesWithoutChangingDeclarationIdentity) {
  const auto graph_identity = [](TensorByteSpanProjectionParams params,
                                 ProjectionCase projection,
                                 std::string_view stable_id) {
    Graph graph{"projected-byte-span-identity"};
    test::add_compile_test_default_clock(graph);
    auto source = graph.op<ProjectedTensorSource>("source", projection, stable_id);
    source.set_params(std::move(params));
    return graph.view().identity();
  };

  const TensorByteSpanProjectionParams small;
  TensorByteSpanProjectionParams large;
  large.width = 40U;
  large.height = 30U;
  large.batch = 3U;
  const auto ordered =
      graph_identity(small, ProjectionCase::kChecked, ProjectedTensorSource::kProjectionId);
  const auto reversed = graph_identity(
      large, ProjectionCase::kReversedDependencies, ProjectedTensorSource::kProjectionId);
  const auto changed_id = graph_identity(
      small, ProjectionCase::kChecked, "holoscan.test.ProjectedTensorSource.output-bytes.v2");
  const auto changed_dependencies =
      graph_identity(small, ProjectionCase::kAlternate, ProjectedTensorSource::kProjectionId);
  EXPECT_EQ(ordered.hash, reversed.hash);
  EXPECT_NE(ordered.hash, changed_id.hash);
  EXPECT_NE(ordered.hash, changed_dependencies.hash);

  const ExecutionPlan small_plan = compile_projected(ProjectionCase::kChecked, small);
  const ExecutionPlan large_plan = compile_projected(ProjectionCase::kChecked, large);
  ASSERT_TRUE(small_plan.ok()) << small_plan.json();
  ASSERT_TRUE(large_plan.ok()) << large_plan.json();
  EXPECT_EQ(only_allocation(small_plan).bounds.max_byte_span, 1792U);
  EXPECT_EQ(only_allocation(large_plan).bounds.max_byte_span, 14592U);
  EXPECT_NE(small_plan.digest(), large_plan.digest());
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, ResolvesCollidingDisplayPathsFromCompleteTypedIdentity) {
  const auto compile_source = []<bool SharedRoot>(bool reverse_dependencies) {
    // Graph topology:
    //
    // ambiguous_source
    Graph graph{"ambiguous-projector-path"};
    test::add_compile_test_default_clock(graph);
    (void)graph.op<AmbiguousPathProjectedSource<SharedRoot>>("ambiguous_source",
                                                             reverse_dependencies);
    return compile(graph);
  };  // NOLINT(readability/braces): cpplint misparses a templated lambda terminator.
  const auto graph_identity = [](bool nested) {
    Graph graph{"ambiguous-projector-identity"};
    test::add_compile_test_default_clock(graph);
    (void)graph.op<AmbiguousPathIdentitySource>("ambiguous_source", nested);
    return graph.view().identity();
  };

  // Key setup: compile root-level and shared-root ambiguities in both declaration orders.
  const ExecutionPlan different_root = compile_source.template operator()<false>(false);
  const ExecutionPlan reversed_different_root = compile_source.template operator()<false>(true);
  const ExecutionPlan shared_root = compile_source.template operator()<true>(false);
  const ExecutionPlan reversed_shared_root = compile_source.template operator()<true>(true);

  // Key assertion: both typed inputs survive replay, resolve 34 bytes, and remain portable.
  for (const auto* plan :
       {&different_root, &reversed_different_root, &shared_root, &reversed_shared_root}) {
    ASSERT_TRUE(plan->ok()) << plan->json();
    EXPECT_EQ(only_allocation(*plan).bounds.max_byte_span, 34U);
    const auto derivations = plan->tensor_byte_span_derivations();
    ASSERT_EQ(derivations.size(), 1U);
    ASSERT_FALSE(derivations.front().nodes.empty());
    const auto& node = derivations.front().nodes.front();
    ASSERT_EQ(node.projection_dependencies.size(), 2U);
    EXPECT_EQ(node.projection_dependencies.front().parameter_path,
              node.projection_dependencies.back().parameter_path);
    EXPECT_NE(node.projection_dependencies.front().parameter_member_full_ids,
              node.projection_dependencies.back().parameter_member_full_ids);
    EXPECT_NE(plan->json().find("parameter_member_full_ids"), std::string_view::npos);
  }
  EXPECT_EQ(different_root.digest(), reversed_different_root.digest());
  EXPECT_EQ(shared_root.digest(), reversed_shared_root.digest());
  EXPECT_NE(graph_identity(false).hash, graph_identity(true).hash);
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, IgnoresUnrelatedRuntimeMutableParameterValues) {
  const TensorByteSpanProjectionParams first;
  TensorByteSpanProjectionParams second = first;
  second.runtime_scale = 99U;

  const ExecutionPlan first_plan = compile_projected(ProjectionCase::kChecked, first);
  const ExecutionPlan second_plan = compile_projected(ProjectionCase::kChecked, second);
  ASSERT_TRUE(first_plan.ok()) << first_plan.json();
  ASSERT_TRUE(second_plan.ok()) << second_plan.json();
  ASSERT_EQ(first_plan.tensor_byte_span_derivations().size(), 1U);
  ASSERT_EQ(second_plan.tensor_byte_span_derivations().size(), 1U);
  const auto& first_derivation = first_plan.tensor_byte_span_derivations().front();
  const auto& second_derivation = second_plan.tensor_byte_span_derivations().front();
  EXPECT_EQ(first_derivation.resolved_max_byte_span, second_derivation.resolved_max_byte_span);
  EXPECT_EQ(first_derivation.update_disposition,
            TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
  EXPECT_EQ(second_derivation.update_disposition, first_derivation.update_disposition);
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, RuntimeAndLifecycleRestartNeverInvokeProjector) {
  g_projection_invocations.store(0U, std::memory_order_relaxed);
  g_runtime_configures.store(0U, std::memory_order_relaxed);
  g_runtime_computes.store(0U, std::memory_order_relaxed);
  g_runtime_restart_accepted.store(false, std::memory_order_relaxed);

  Graph graph{"projected-byte-span-runtime-probe"};
  auto source = graph.op<ProjectedRuntimeProbeSource>("source");
  graph.lifecycle().allow_self_restart(source);
  test::add_compile_test_default_clock(graph);
  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(g_projection_invocations.load(std::memory_order_relaxed), 1U);

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (g_runtime_configures.load(std::memory_order_acquire) < 2U && session.is_running() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  session.request_stop();
  session.wait();

  EXPECT_TRUE(g_runtime_restart_accepted.load(std::memory_order_acquire));
  EXPECT_GE(g_runtime_configures.load(std::memory_order_acquire), 2U);
  EXPECT_GE(g_runtime_computes.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(g_projection_invocations.load(std::memory_order_relaxed), 1U);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, SupportsNestedAndDescriptorFieldDependencies) {
  const ExecutionPlan frozen = compile_projected(ProjectionCase::kFrozen);
  ASSERT_TRUE(frozen.ok()) << frozen.json();
  const auto frozen_derivations = frozen.tensor_byte_span_derivations();
  ASSERT_EQ(frozen_derivations.size(), 1U);
  EXPECT_EQ(frozen_derivations.front().update_disposition,
            TensorByteSpanDerivationUpdateDisposition::kFrozenCompileTime);

  TensorByteSpanProjectionParams params;
  params.width = 13U;
  params.nested.channels = 3U;
  const ExecutionPlan nested = compile_projected(ProjectionCase::kNested, params);
  ASSERT_TRUE(nested.ok()) << nested.json();
  EXPECT_EQ(only_allocation(nested).bounds.max_byte_span, 1024U);
  const auto nested_derivations = nested.tensor_byte_span_derivations();
  ASSERT_EQ(nested_derivations.size(), 1U);
  ASSERT_FALSE(nested_derivations.front().nodes.empty());
  const auto& nested_node = nested_derivations.front().nodes.front();
  ASSERT_EQ(nested_node.projection_dependencies.size(), 2U);
  EXPECT_TRUE(std::ranges::any_of(nested_node.projection_dependencies, [](const auto& dependency) {
    return dependency.parameter_path == "image.shape.channels";
  }));

  Graph graph{"projected-descriptor-field"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<ProjectedImageSource>("source");
  source.set_params(params);
  const ExecutionPlan field = compile(graph);
  ASSERT_TRUE(field.ok()) << field.json();
  ASSERT_EQ(field.tensor_byte_span_derivations().size(), 1U);
  EXPECT_EQ(field.tensor_byte_span_derivations().front().field_id, 4U);
  EXPECT_EQ(field.tensor_byte_span_derivations().front().tensor_index, 0U);
  EXPECT_EQ(only_allocation(field).bounds.max_byte_span, 39U);

  for (const std::string& invalid_id :
       {std::string{}, std::string(kMaximumTensorByteSpanProjectionIdBytes + 1U, 'x')}) {
    Graph invalid_graph{"invalid-projected-descriptor-field"};
    test::add_compile_test_default_clock(invalid_graph);
    try {
      (void)invalid_graph.op<ProjectedImageSource>("source", invalid_id);
      FAIL() << "expected malformed descriptor projector identity rejection";
    } catch (const RuntimeError& error) {
      EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
      EXPECT_TRUE(std::string_view{make_error(error).message}.starts_with(
          "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID"));
    }
  }
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, RejectsSensitiveNestedAncestorsBeforeInvokingProjector) {
  for (const SensitiveAncestorCase ancestor :
       {SensitiveAncestorCase::kRoot, SensitiveAncestorCase::kIntermediate}) {
    for (const bool descriptor : {false, true}) {
      g_projection_invocations.store(0U, std::memory_order_relaxed);

      // Key setup: each graph contains one raw or descriptor output whose declared path crosses
      // either a sensitive root or sensitive intermediate field.
      //
      // source
      Graph graph{"sensitive-ancestor-projector"};
      test::add_compile_test_default_clock(graph);
      (void)graph.op<SensitiveAncestorProjectedSource>("source", ancestor, descriptor);
      const ExecutionPlan plan = compile(graph);

      // Key assertion: metadata policy rejection happens before any parameter value is read.
      ASSERT_FALSE(plan.ok());
      EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"));
      EXPECT_EQ(g_projection_invocations.load(std::memory_order_relaxed), 0U);
      EXPECT_TRUE(plan.tensor_allocations().empty());
      EXPECT_TRUE(plan.tensor_byte_span_derivations().empty());
    }
  }
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, RejectsInvalidDependenciesAndProjectorResults) {
  Graph foreign_graph{"foreign-projector-owner"};
  test::add_compile_test_default_clock(foreign_graph);
  (void)foreign_graph.op<ForeignOwnerProjectedSource>("source");
  const ExecutionPlan foreign = compile(foreign_graph);
  ASSERT_FALSE(foreign.ok());
  EXPECT_TRUE(test::has_diagnostic(foreign, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"));

  const ExecutionPlan runtime_mutable = compile_projected(ProjectionCase::kRuntimeMutable);
  ASSERT_FALSE(runtime_mutable.ok());
  EXPECT_TRUE(test::has_diagnostic(runtime_mutable,
                                   "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID"));

  const ExecutionPlan lifecycle_mutable = compile_projected(ProjectionCase::kLifecycleMutable);
  ASSERT_FALSE(lifecycle_mutable.ok());
  EXPECT_TRUE(test::has_diagnostic(lifecycle_mutable,
                                   "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID"));

  const ExecutionPlan sensitive = compile_projected(ProjectionCase::kSensitive);
  ASSERT_FALSE(sensitive.ok());
  EXPECT_TRUE(test::has_diagnostic(sensitive, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"));

  const ExecutionPlan rejected = compile_projected(ProjectionCase::kRejected);
  ASSERT_FALSE(rejected.ok());
  EXPECT_TRUE(test::has_diagnostic(rejected, "PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED"));
  EXPECT_EQ(rejected.json().find("private projector implementation detail"),
            std::string_view::npos);

  const ExecutionPlan overflow = compile_projected(ProjectionCase::kOverflow);
  ASSERT_FALSE(overflow.ok());
  EXPECT_TRUE(test::has_diagnostic(overflow, "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW"));

  const ExecutionPlan overflow_message = compile_projected(ProjectionCase::kOverflowMessage);
  ASSERT_FALSE(overflow_message.ok());
  EXPECT_TRUE(
      test::has_diagnostic(overflow_message, "PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED"));

  const ExecutionPlan zero = compile_projected(ProjectionCase::kZero);
  ASSERT_FALSE(zero.ok());
  EXPECT_TRUE(test::has_diagnostic(zero, "PHYSICAL_TENSOR_BYTE_SPAN_VALUE_INVALID"));
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, RejectsStableIdImplementationCollisionAcrossOutputs) {
  Graph graph{"projector-id-collision"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<ProjectedTensorSource>(
      "first", ProjectionCase::kChecked, "holoscan.test.shared-projector-id.v1");
  (void)graph.op<ProjectedTensorSource>(
      "second", ProjectionCase::kAlternate, "holoscan.test.shared-projector-id.v1");

  const ExecutionPlan plan = compile(graph);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"));
  EXPECT_TRUE(plan.tensor_allocations().empty());
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, RejectsDistinctCppTypeWithCollidingSchemaIdentity) {
  // Key setup: prove the two distinct C++ types deliberately publish identical portable schemas.
  const ParamSchemaDescriptor* const owner_schema =
      param_schema_descriptor<TensorByteSpanProjectionParams>();
  const ParamSchemaDescriptor* const projector_schema =
      param_schema_descriptor<CollidingTensorByteSpanProjectionParams>();
  ASSERT_NE(owner_schema, nullptr);
  ASSERT_NE(projector_schema, nullptr);
  EXPECT_EQ(owner_schema->stable_id, projector_schema->stable_id);
  EXPECT_EQ(owner_schema->version, projector_schema->version);
  EXPECT_EQ(owner_schema->structural_hash, projector_schema->structural_hash);
  EXPECT_EQ(owner_schema->contract_hash, projector_schema->contract_hash);

  Graph graph{"colliding-parameter-schema-types"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<CollidingSchemaProjectedSource>("source");
  const ExecutionPlan plan = compile(graph);

  // Key assertion: portable identity does not authorize a projector for another exact C++ type.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"));
  EXPECT_TRUE(plan.tensor_allocations().empty());
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, ArtifactContainsOnlyPortableProjectionEvidence) {
  const ExecutionPlan literal = compile_literal(1792U);
  ASSERT_TRUE(literal.ok()) << literal.json();
  EXPECT_EQ(literal.json().find("projection_id"), std::string_view::npos);

  const ExecutionPlan plan = compile_projected();
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_NE(plan.json().find("projection_id"), std::string_view::npos);
  EXPECT_NE(plan.json().find(ProjectedTensorSource::kProjectionId), std::string_view::npos);
  EXPECT_NE(plan.json().find("parameter_member_full_ids"), std::string_view::npos);
  EXPECT_EQ(plan.json().find("implementation_token"), std::string_view::npos);

  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  auto changed = std::make_shared<detail::ExecutionPlanState>(*state);
  ASSERT_EQ(changed->tensor_byte_span_derivations.size(), 1U);
  auto& changed_nodes = changed->tensor_byte_span_derivations.front().nodes;
  ASSERT_FALSE(changed_nodes.empty());
  auto& projection = changed_nodes.front();
  projection.projection_id = "holoscan.test.ProjectedTensorSource.output-bytes.v2";
  ASSERT_TRUE(detail::finalize_plan_artifact(*changed));
  EXPECT_NE(changed->digest, plan.digest());

  auto malformed = std::make_shared<detail::ExecutionPlanState>(*state);
  ASSERT_EQ(malformed->tensor_byte_span_derivations.size(), 1U);
  auto& malformed_nodes = malformed->tensor_byte_span_derivations.front().nodes;
  ASSERT_FALSE(malformed_nodes.empty());
  malformed_nodes.front().projection_dependencies.clear();
  const auto rejected = detail::finalize_plan_artifact(*malformed);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().message, "TENSOR_BYTE_SPAN_PROJECTION_PLAN_INVALID");

  auto malformed_member_path = std::make_shared<detail::ExecutionPlanState>(*state);
  ASSERT_EQ(malformed_member_path->tensor_byte_span_derivations.size(), 1U);
  auto& member_path_nodes = malformed_member_path->tensor_byte_span_derivations.front().nodes;
  ASSERT_FALSE(member_path_nodes.empty());
  auto& member_path_dependencies = member_path_nodes.front().projection_dependencies;
  ASSERT_FALSE(member_path_dependencies.empty());
  member_path_dependencies.front().parameter_member_full_ids.clear();
  const auto member_path_rejected = detail::finalize_plan_artifact(*malformed_member_path);
  ASSERT_FALSE(member_path_rejected);
  EXPECT_EQ(member_path_rejected.error().message, "TENSOR_BYTE_SPAN_PROJECTION_PLAN_INVALID");
}

// REQ: COMPILE-077
TEST(TensorByteSpanProjectionCompile, ConcurrentCompilesUseIndependentParameterSnapshots) {
  const auto compile_width = [](std::uint32_t width) {
    TensorByteSpanProjectionParams params;
    params.width = width;
    return compile_projected(ProjectionCase::kChecked, params);
  };
  auto first = std::async(std::launch::async, compile_width, 10U);
  auto second = std::async(std::launch::async, compile_width, 30U);
  const ExecutionPlan first_plan = first.get();
  const ExecutionPlan second_plan = second.get();
  ASSERT_TRUE(first_plan.ok()) << first_plan.json();
  ASSERT_TRUE(second_plan.ok()) << second_plan.json();
  EXPECT_EQ(only_allocation(first_plan).bounds.max_byte_span, 1792U);
  EXPECT_EQ(only_allocation(second_plan).bounds.max_byte_span, 4864U);
}

// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(TensorByteSpanProjectionCompile, RetainsReturnedCodeWithoutPrivateProjectorMessage) {
  for (const auto projection :
       {ProjectionCase::kRejected, ProjectionCase::kOverflow, ProjectionCase::kOverflowMessage}) {
    SCOPED_TRACE(static_cast<unsigned>(projection));
    const ExecutionPlan plan = compile_projected(projection);
    ASSERT_FALSE(plan.ok());
    const bool overflow = projection == ProjectionCase::kOverflow;
    const auto token = overflow ? "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW"
                                : "PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED";
    const auto rows = plan.diagnostics();
    const auto found =
        std::find_if(rows.begin(), rows.end(), [&](const auto& row) { return row.token == token; });
    ASSERT_NE(found, rows.end());
    const auto code = overflow ? ErrorCode::kArithmeticOverflow : ErrorCode::kInvalidArgument;
    EXPECT_EQ(found->provided, error_code_name(code));
    ASSERT_TRUE(found->detail.has_value());
    if (!found->detail.has_value()) {
      FAIL() << "Expected the exact returned projector code";
    }
    const auto& evidence = found->detail.value();
    EXPECT_EQ(evidence.operation, DiagnosticOperation::kCompile);
    EXPECT_EQ(evidence.error.code, code);
    EXPECT_TRUE(evidence.error.message.empty());
    EXPECT_TRUE(evidence.causes.empty());
    const auto projector =
        std::ranges::find(evidence.mismatches, "projector_id", &DiagnosticMismatch::field);
    ASSERT_NE(projector, evidence.mismatches.end());
    EXPECT_EQ(projector->provided, ProjectedTensorSource::kProjectionId);
    EXPECT_EQ(evidence.operator_path, "source");
    EXPECT_EQ(evidence.port_name, "output");
    EXPECT_TRUE(found->source.vertex);
    EXPECT_TRUE(found->source.port);
    const auto rendered = format_diagnostic(*found, plan.graph());
    EXPECT_NE(rendered.find(error_code_name(code)), std::string::npos);
    EXPECT_EQ(rendered.find("private projector implementation detail"), std::string::npos);
    EXPECT_EQ(plan.json().find("private projector implementation detail"), std::string_view::npos);
    EXPECT_EQ(rendered.find("private arithmetic overflow detail"), std::string::npos);
    EXPECT_EQ(plan.json().find("private arithmetic overflow detail"), std::string_view::npos);
    EXPECT_NE(format_diagnostic(*found, {}).find("source"), std::string::npos);
  }
}

// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(TensorByteSpanProjectionCompile, ReturnedErrorDetailIsOwningPrivateAndNonsemantic) {
  const auto structural_key = [](const Diagnostic& row) {
    return std::tuple{row.id.value,
                      row.severity,
                      row.token,
                      row.source.graph_generation,
                      row.source.graph.value,
                      row.source.vertex.value,
                      row.source.port.value(),
                      row.service,
                      row.endpoint,
                      row.json_pointer,
                      row.required,
                      row.provided,
                      row.cause_id};
  };
  ExecutionPlan plan = compile_projected(ProjectionCase::kRejected);
  ASSERT_FALSE(plan.ok());
  // compile_projected's Graph and projector replay temporaries are already gone.
  const auto rows = plan.diagnostics();
  const auto found =
      std::ranges::find(rows, "PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED", &Diagnostic::token);
  ASSERT_NE(found, rows.end());
  const auto index = static_cast<std::size_t>(found - rows.begin());
  // Preserve the owning copy across the later mutation of the plan diagnostic.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const Diagnostic owned = *found;
  if (!owned.detail.has_value()) {
    FAIL() << "Expected the actual returned projector code after final publication";
  }
  EXPECT_EQ(owned.detail->operation, DiagnosticOperation::kCompile);
  EXPECT_EQ(owned.detail->error.code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(owned.detail->error.message.empty());
  EXPECT_TRUE(owned.detail->causes.empty());
  EXPECT_EQ(owned.detail->operator_path, "source");
  EXPECT_EQ(owned.detail->port_name, "output");
  EXPECT_TRUE(owned.source.vertex);
  EXPECT_TRUE(owned.source.port);
  for (const auto& mismatch : owned.detail->mismatches) {
    EXPECT_EQ(mismatch.field.find("private projector implementation detail"), std::string::npos);
    EXPECT_EQ(mismatch.required.find("private projector implementation detail"), std::string::npos);
    EXPECT_EQ(mismatch.provided.find("private projector implementation detail"), std::string::npos);
  }

  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const std::string original_json{plan.json()};
  const PlanDigest original_digest = plan.digest();
  const ExecutionGraphView owning_graph = plan.graph();
  const auto vertices = owning_graph.vertices();
  const auto vertex = std::ranges::find(vertices, owned.source.vertex, &VertexRecord::id);
  ASSERT_NE(vertex, vertices.end());
  EXPECT_EQ(vertex->name, "source");
  EXPECT_EQ(owned.source.graph, vertex->graph);
  EXPECT_EQ(owned.source.graph_generation, owning_graph.generation());
  const auto ports = owning_graph.ports();
  const auto port = std::ranges::find_if(ports, [&](const auto& candidate) {
    return candidate.id.value == owned.source.port.value();
  });
  ASSERT_NE(port, ports.end());
  EXPECT_EQ(port->owner, vertex->id);
  EXPECT_EQ(port->name, "output");

  for (const bool remove_detail : {false, true}) {
    SCOPED_TRACE(remove_detail);
    auto changed = std::make_shared<detail::ExecutionPlanState>(*state);
    auto& supplemental = changed->diagnostics[index].detail;
    if (remove_detail) {
      supplemental.reset();
    } else {
      if (!supplemental.has_value()) {
        FAIL() << "Copied projector detail disappeared";
      }
      // Public context may vary. Preserve the returned-error redaction rule even in this
      // deliberately changed serializer control: never insert the private projector message.
      supplemental->operator_path = "alternate-owned-projector-context";
      supplemental->mismatches.clear();
      supplemental->truncated = true;
    }
    ASSERT_EQ(changed->diagnostics.size(), rows.size());
    for (std::size_t row = 0U; row < rows.size(); ++row) {
      EXPECT_EQ(structural_key(changed->diagnostics[row]), structural_key(rows[row]));
    }
    const auto serialized = detail::finalize_plan_artifact(*changed);
    ASSERT_TRUE(serialized) << serialized.error().message;
    EXPECT_FALSE(changed->admitted);
    EXPECT_EQ(changed->json, original_json);
    EXPECT_EQ(changed->digest, original_digest);
    EXPECT_EQ(changed->json.find("alternate-owned-projector-context"), std::string::npos);
    EXPECT_EQ(changed->json.find("private projector implementation detail"), std::string::npos);
    EXPECT_EQ(changed->json.find("implementation_token"), std::string::npos);
  }

  EXPECT_EQ(original_json.find("private projector implementation detail"), std::string::npos);
  EXPECT_EQ(format_diagnostics(rows, owning_graph).find("private projector implementation detail"),
            std::string::npos);
  plan = {};
  const auto empty_report = format_diagnostic(owned, {});
  EXPECT_NE(empty_report.find("operator=source"), std::string::npos);
  EXPECT_NE(empty_report.find("port_name=output"), std::string::npos);
  EXPECT_EQ(empty_report.find("private projector implementation detail"), std::string::npos);
  const auto compiled_report = format_diagnostic(owned, owning_graph);
  EXPECT_NE(
      compiled_report.find("vertex=source, vertex_id=" + std::to_string(owned.source.vertex.value)),
      std::string::npos);
  EXPECT_NE(compiled_report.find("port=" + std::to_string(owned.source.port.value()) +
                                 ", port_name=output"),
            std::string::npos);
  EXPECT_EQ(compiled_report.find("private projector implementation detail"), std::string::npos);
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
