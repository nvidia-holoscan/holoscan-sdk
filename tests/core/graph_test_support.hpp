/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan {

struct GraphPodPayload {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<GraphPodPayload> {
  static constexpr std::string_view name = "holoscan.test.GraphPodPayload";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<GraphPodPayload> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &GraphPodPayload::value, offsetof(GraphPodPayload, value)));
};

struct GraphPodAliasA {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<GraphPodAliasA> {
  static constexpr std::string_view name = "holoscan.test.GraphPodAlias";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<GraphPodAliasA> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &GraphPodAliasA::value, offsetof(GraphPodAliasA, value)));
};

namespace test::graph_support {

// General-purpose pass-through operator with one uint32_t input and one output.
// Carries an optional label string to distinguish instances in multi-node topologies.
// Used as a topology vertex in the majority of graph-structure tests.
// Representative test: LazyCompositionRunsExactlyOnceAndSnapshotsAreIndependent.
class PassOp : public Operator<> {
 public:
  explicit PassOp(std::string label = {}) : label_(std::move(label)) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2);
    spec.output(output, "output").max_emits_per_compute(1);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  std::string label_;
};

// Source operator emitting GraphPodAliasA; anchors POD type-compatibility tests.
// Representative test: DistinctCppPodTypesInteroperateOnlyByCompleteContract.
class PodSourceOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<GraphPodAliasA> output;
};
// Source operator that declares a bounded Tensor output (max_rank=4, max_byte_span=4096).
// The canonical Tensor producer used in layout, configurator, and rollback tests.
// Representative test: TensorConfiguratorIsStableAndOnlyTightensDeclaredAllocation.
class TensorOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice},
            .bounds = TensorBounds{.max_rank = 4U, .max_byte_span = 4096U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Tensor pass-through operator with device-memory layout annotations on both ports.
// Exercises layout normalization, per-edge conflict detection, and mutation rollback.
// Representative test: TensorEdgeLayoutsRejectRepresentationConflictsAtomically.
class TensorPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice}});
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice}});
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<Tensor> input;
  Output<Tensor> output;
};

// Invokes callable, expects it to throw RuntimeError, and returns the captured Error.
// Allows EXPECT_EQ assertions on error codes and messages without explicit try/catch in tests.
template <typename Callable>
Error graph_error_from(Callable&& callable) {
  try {
    std::forward<Callable>(callable)();
  } catch (const RuntimeError& error) {
    return make_error(error);
  }
  ADD_FAILURE() << "expected RuntimeError";
  return Error{ErrorCode::kSuccess};
}

}  // namespace test::graph_support
}  // namespace holoscan
