/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <benchmark/benchmark.h>

#include <holoscan/core/graph.hpp>

namespace {

class BenchmarkOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;
};

[[nodiscard]] std::size_t operator_count(const benchmark::State& state) noexcept {
  return static_cast<std::size_t>(state.range(0));
}

[[nodiscard]] std::string operator_name(std::size_t index) {
  return "operator_" + std::to_string(index);
}

void author_operators(holoscan::Graph& graph, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    benchmark::DoNotOptimize(graph.op<BenchmarkOperator>(operator_name(index)));
  }
}

void author_linear_chain(holoscan::Graph& graph, std::size_t count) {
  auto prior = graph.op<BenchmarkOperator>(operator_name(0));
  for (std::size_t index = 1; index < count; ++index) {
    auto next = graph.op<BenchmarkOperator>(operator_name(index));
    benchmark::DoNotOptimize(graph.add_flow(prior->output, next->input));
    prior = next;
  }
}

void author_fanout(holoscan::Graph& graph, std::size_t count) {
  auto source = graph.op<BenchmarkOperator>(operator_name(0));
  for (std::size_t index = 1; index < count; ++index) {
    auto target = graph.op<BenchmarkOperator>(operator_name(index));
    benchmark::DoNotOptimize(graph.add_flow(source->output, target->input));
  }
}

class BenchmarkReusableGraph final : public holoscan::Graph {
 public:
  explicit BenchmarkReusableGraph(std::size_t count) : count_(count) {}

 protected:
  void compose() override { author_linear_chain(*this, count_); }

 private:
  std::size_t count_;
};

template <std::size_t RemainingDepth, std::size_t MaximumDepth>
class BenchmarkNestedGraph final : public holoscan::Graph {
 public:
  explicit BenchmarkNestedGraph(std::size_t total_operators) : total_operators_(total_operators) {}

 protected:
  void compose() override {
    constexpr std::size_t kLevels = MaximumDepth + 1U;
    const std::size_t remainder = RemainingDepth == MaximumDepth ? total_operators_ % kLevels : 0U;
    author_linear_chain(*this, total_operators_ / kLevels + remainder);
    if constexpr (RemainingDepth > 0U) {
      benchmark::DoNotOptimize(embed<BenchmarkNestedGraph<RemainingDepth - 1U, MaximumDepth>>(
          "nested", total_operators_));
    }
  }

 private:
  std::size_t total_operators_;
};

template <std::size_t Levels>
void embed_nested_hierarchy(holoscan::Graph& parent, std::size_t total_operators) {
  static_assert(Levels > 0U);
  benchmark::DoNotOptimize(
      parent.embed<BenchmarkNestedGraph<Levels - 1U, Levels - 1U>>("child", total_operators));
}

void BM_GraphOperatorAuthoring(benchmark::State& state) {
  const std::size_t count = operator_count(state);
  for (auto _ : state) {
    holoscan::Graph graph;
    author_operators(graph, count);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_GraphLinearChainAuthoring(benchmark::State& state) {
  const std::size_t count = operator_count(state);
  for (auto _ : state) {
    holoscan::Graph graph;
    author_linear_chain(graph, count);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_GraphFanoutAuthoring(benchmark::State& state) {
  const std::size_t count = operator_count(state);
  for (auto _ : state) {
    holoscan::Graph graph;
    author_fanout(graph, count);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_GraphSnapshotAndHash(benchmark::State& state) {
  const std::size_t count = operator_count(state);
  for (auto _ : state) {
    state.PauseTiming();
    auto graph = std::make_unique<holoscan::Graph>();
    author_linear_chain(*graph, count);
    state.ResumeTiming();
    {
      const holoscan::GraphView view = graph->view();
      benchmark::DoNotOptimize(view.identity().hash);
    }
    state.PauseTiming();
    graph.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_GraphEmbeddedChildSize(benchmark::State& state) {
  const std::size_t count = operator_count(state);
  for (auto _ : state) {
    holoscan::Graph parent;
    benchmark::DoNotOptimize(parent.embed<BenchmarkReusableGraph>("child", count));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_GraphEmbeddedSiblingCount(benchmark::State& state) {
  constexpr std::size_t kOperatorsPerChild = 10U;
  const std::size_t count = operator_count(state);
  for (auto _ : state) {
    holoscan::Graph parent;
    for (std::size_t index = 0; index < count; ++index) {
      benchmark::DoNotOptimize(parent.embed<BenchmarkReusableGraph>(
          "child_" + std::to_string(index), kOperatorsPerChild));
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(count * kOperatorsPerChild));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_GraphEmbeddedNestingDepth(benchmark::State& state) {
  constexpr std::size_t kTotalOperators = 128U;
  using EmbedHierarchy = void (*)(holoscan::Graph&, std::size_t);
  EmbedHierarchy embed_hierarchy = nullptr;
  switch (state.range(0)) {
    case 1:
      embed_hierarchy = embed_nested_hierarchy<1U>;
      break;
    case 2:
      embed_hierarchy = embed_nested_hierarchy<2U>;
      break;
    case 4:
      embed_hierarchy = embed_nested_hierarchy<4U>;
      break;
    case 8:
      embed_hierarchy = embed_nested_hierarchy<8U>;
      break;
    default:
      state.SkipWithError("unsupported hierarchy level count");
      return;
  }

  for (auto _ : state) {
    holoscan::Graph parent;
    embed_hierarchy(parent, kTotalOperators);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kTotalOperators));
  state.SetComplexityN(state.range(0));
}

#define HOLOSCAN_GRAPH_SIZES(complexity) \
  ->ArgName("operators")->Arg(1)->Arg(10)->Arg(100)->Arg(1000)->Complexity(complexity)

BENCHMARK(BM_GraphOperatorAuthoring) HOLOSCAN_GRAPH_SIZES(benchmark::oN);
BENCHMARK(BM_GraphLinearChainAuthoring) HOLOSCAN_GRAPH_SIZES(benchmark::oAuto);
BENCHMARK(BM_GraphFanoutAuthoring) HOLOSCAN_GRAPH_SIZES(benchmark::oAuto);
BENCHMARK(BM_GraphSnapshotAndHash) HOLOSCAN_GRAPH_SIZES(benchmark::oN);
BENCHMARK(BM_GraphEmbeddedChildSize) HOLOSCAN_GRAPH_SIZES(benchmark::oAuto);
BENCHMARK(BM_GraphEmbeddedSiblingCount)
    ->ArgName("children")
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Complexity(benchmark::oAuto);
BENCHMARK(BM_GraphEmbeddedNestingDepth)
    ->ArgName("levels")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Complexity(benchmark::oAuto);

#undef HOLOSCAN_GRAPH_SIZES

}  // namespace
