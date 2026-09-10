/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/resource.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/time/manual_clock.hpp>

#include "partition_planning_allocation_probe.hpp"

namespace holoscan {
namespace {

using Clock = std::chrono::steady_clock;

class SourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<std::uint32_t> output;
};

class RelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

class SinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<std::uint32_t> input;
};

class JoinOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(1U);
    spec.input(second, "second").queue_depth(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
  Output<std::uint32_t> output;
};

void add_benchmark_clock(Graph& graph) {
  const ClockKey clock = graph.add_clock<ManualClock>("benchmark-clock", std::int64_t{0});
  graph.set_default_clock(clock);
}

[[nodiscard]] std::string name(std::string_view prefix, std::size_t index) {
  return std::string(prefix) + "-" + std::to_string(index);
}

[[nodiscard]] std::vector<OperatorHandle<RelayOperator>> add_linear(Graph& graph,
                                                                    std::size_t count) {
  std::vector<OperatorHandle<RelayOperator>> relays;
  if (count == 1U) {
    graph.op<SourceOperator>("source");
    return relays;
  }
  auto source = graph.op<SourceOperator>("source");
  auto* prior = &source->output;
  relays.reserve(count > 2U ? count - 2U : 0U);
  for (std::size_t index = 0U; index + 2U < count; ++index) {
    auto relay = graph.op<RelayOperator>(name("relay", index));
    graph.add_flow(*prior, relay->input);
    prior = &relay->output;
    relays.push_back(relay);
  }
  auto sink = graph.op<SinkOperator>("sink");
  graph.add_flow(*prior, sink->input);
  return relays;
}

void add_fanout(Graph& graph, std::size_t count) {
  auto source = graph.op<SourceOperator>("source");
  for (std::size_t index = 1U; index < count; ++index) {
    auto sink = graph.op<SinkOperator>(name("sink", index));
    graph.add_flow(source->output, sink->input);
  }
}

void add_binary_fanout(Graph& graph, std::size_t count) {
  std::size_t index{};
  while (index + 3U <= count) {
    auto source = graph.op<SourceOperator>(name("source", index));
    auto first = graph.op<SinkOperator>(name("sink-a", index));
    auto second = graph.op<SinkOperator>(name("sink-b", index));
    graph.add_flow(source->output, first->input);
    graph.add_flow(source->output, second->input);
    index += 3U;
  }
  while (index < count) {
    graph.op<SourceOperator>(name("remainder", index++));
  }
}

void add_fanin(Graph& graph, std::size_t count) {
  std::size_t index{};
  while (index + 3U <= count) {
    auto first = graph.op<SourceOperator>(name("source-a", index));
    auto second = graph.op<SourceOperator>(name("source-b", index));
    auto join = graph.op<JoinOperator>(name("join", index));
    graph.add_flow(first->output, join->first);
    graph.add_flow(second->output, join->second);
    index += 3U;
  }
  while (index < count) {
    graph.op<SourceOperator>(name("remainder", index++));
  }
}

void add_diamonds(Graph& graph, std::size_t count) {
  std::size_t index{};
  while (index + 4U <= count) {
    auto source = graph.op<SourceOperator>(name("source", index));
    auto left = graph.op<RelayOperator>(name("left", index));
    auto right = graph.op<RelayOperator>(name("right", index));
    auto join = graph.op<JoinOperator>(name("join", index));
    graph.add_flow(source->output, left->input);
    graph.add_flow(source->output, right->input);
    graph.add_flow(left->output, join->first);
    graph.add_flow(right->output, join->second);
    index += 4U;
  }
  while (index < count) {
    graph.op<SourceOperator>(name("remainder", index++));
  }
}

class EmbeddedLinearGraph final : public Graph {
 public:
  explicit EmbeddedLinearGraph(std::size_t count) : count_(count) {}

 protected:
  void compose() override {
    add_benchmark_clock(*this);
    static_cast<void>(add_linear(*this, count_));
  }

 private:
  std::size_t count_;
};

struct Options {
  std::string topology{"linear"};
  std::size_t operators{16U};
  std::string output;
  bool cross_generation_self_test{};
};

[[nodiscard]] std::optional<Options> parse_options(int argc, char** argv) {
  Options options;
  if (argc == 2 && std::string_view(argv[1]) == "--cross-generation-self-test") {
    options.cross_generation_self_test = true;
    return options;
  }
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return std::nullopt;
    }
    const std::string_view argument{argv[index]};
    const std::string value{argv[index + 1]};
    if (argument == "--topology") {
      options.topology = value;
    } else if (argument == "--operators") {
      try {
        const auto operators = std::stoull(value);
        if (operators > std::numeric_limits<std::uint32_t>::max()) {
          return std::nullopt;
        }
        options.operators = static_cast<std::size_t>(operators);
      } catch (...) {
        return std::nullopt;
      }
    } else if (argument == "--output") {
      options.output = value;
    } else {
      return std::nullopt;
    }
  }
  constexpr std::array<std::string_view, 8U> kTopologies{
      "linear", "fanout", "binary_fanout", "fanin", "diamond", "multi_root", "embedded", "hybrid"};
  if (options.output.empty() ||
      std::ranges::find(kTopologies, options.topology) == kTopologies.end()) {
    return std::nullopt;
  }
  return options;
}

[[nodiscard]] std::string hex_digest(PlanDigest digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::byte value : digest.value) {
    output << std::setw(2) << std::to_integer<unsigned>(value);
  }
  return output.str();
}

[[nodiscard]] const char* mode_name(PartitioningMode mode) noexcept {
  switch (mode) {
    case PartitioningMode::kAutomatic:
      return "automatic";
    case PartitioningMode::kConstrainedAutomatic:
      return "constrained_automatic";
    case PartitioningMode::kManual:
      return "manual";
  }
  return "unknown";
}

[[nodiscard]] std::size_t edge_count(const ExecutionPlan& plan) noexcept {
  return plan.payload_edges().size();
}

int run_cross_generation_self_test() {
  {
    benchmarking::PlanningAllocationProbe zero_size;
    void* const zero_size_allocation = ::operator new(0U);
    const auto live_stats = zero_size.snapshot();
    ::operator delete(zero_size_allocation);
    const auto final_stats = zero_size.finish();
    if (live_stats.allocation_count != 1U || live_stats.cumulative_requested_bytes != 0U ||
        live_stats.peak_live_requested_bytes != 0U || live_stats.live_requested_bytes != 0U ||
        final_stats.live_requested_bytes != 0U) {
      return 2;
    }
  }

  benchmarking::PlanningAllocationProbe first;
  void* const allocation = ::operator new(32U);
  void* const nothrow_allocation = ::operator new(32U, std::nothrow);
  void* const nothrow_array_allocation = ::operator new[](32U, std::nothrow);
  void* const aligned_nothrow_allocation = ::operator new(32U, std::align_val_t{64U}, std::nothrow);
  void* const aligned_nothrow_array_allocation =
      ::operator new[](32U, std::align_val_t{64U}, std::nothrow);
  if (nothrow_allocation == nullptr || nothrow_array_allocation == nullptr ||
      aligned_nothrow_allocation == nullptr || aligned_nothrow_array_allocation == nullptr) {
    ::operator delete(allocation);
    ::operator delete(nothrow_allocation, std::nothrow);
    ::operator delete[](nothrow_array_allocation, std::nothrow);
    ::operator delete(aligned_nothrow_allocation, std::align_val_t{64U}, std::nothrow);
    ::operator delete[](aligned_nothrow_array_allocation, std::align_val_t{64U}, std::nothrow);
    return 2;
  }
  static_cast<void>(first.finish());
  benchmarking::PlanningAllocationProbe second;
  ::operator delete(allocation);
  ::operator delete(nothrow_allocation, std::nothrow);
  ::operator delete[](nothrow_array_allocation, std::nothrow);
  ::operator delete(aligned_nothrow_allocation, std::align_val_t{64U}, std::nothrow);
  ::operator delete[](aligned_nothrow_array_allocation, std::align_val_t{64U}, std::nothrow);
  const auto stats = second.finish();
  return stats.cross_generation_activity ? 0 : 2;
}

int run(const Options& options) {
  if (options.topology == "hybrid" && options.operators > 0U && options.operators < 4U) {
    std::cerr << "hybrid topology requires zero or at least four operators\n";
    return 2;
  }

  // Exclude one-time compiler and library initialization from the measured generation.
  // A separate accepted zero-work compile also keeps the warm-up independent of topology.
  {
    Graph warmup_graph{"partition-planning-warmup"};
    add_benchmark_clock(warmup_graph);
    const ExecutionPlan warmup_plan = compile(warmup_graph);
    if (!warmup_plan.ok() || !warmup_plan.partitions().empty()) {
      std::cerr << "partition-planning warm-up compile failed\n";
      return 2;
    }
  }

  Graph graph{"partition-planning-benchmark"};
  std::vector<OperatorHandle<RelayOperator>> relays;
  if (options.operators == 0U) {
    add_benchmark_clock(graph);
  } else if (options.topology == "embedded") {
    graph.embed<EmbeddedLinearGraph>("embedded", options.operators);
  } else {
    add_benchmark_clock(graph);
    if (options.topology == "linear" || options.topology == "hybrid") {
      relays = add_linear(graph, options.operators);
    } else if (options.topology == "fanout") {
      add_fanout(graph, options.operators);
    } else if (options.topology == "binary_fanout") {
      add_binary_fanout(graph, options.operators);
    } else if (options.topology == "fanin") {
      add_fanin(graph, options.operators);
    } else if (options.topology == "diamond") {
      add_diamonds(graph, options.operators);
    } else {
      for (std::size_t index = 0U; index < options.operators; ++index) {
        graph.op<SourceOperator>(name("root", index));
      }
    }
  }
  if (options.topology == "hybrid" && options.operators != 0U) {
    auto anchor = graph.partition("benchmark-anchor");
    const std::size_t anchor_relays = std::min(relays.size(), options.operators / 4U);
    for (std::size_t index = 0U; index < anchor_relays; ++index) {
      anchor.add(relays[index]);
    }
    graph.auto_partition_remaining();
  }

  benchmarking::PlanningAllocationStats compile_allocation_stats;
  bool plan_ok{};
  std::string plan_error;
  std::size_t accepted_edge_count{};
  std::size_t partition_count{};
  std::uint32_t authored_partition_count{};
  std::uint32_t automatic_partition_count{};
  std::size_t authored_member_count{};
  std::size_t automatic_member_count{};
  PartitioningMode partitioning_mode{PartitioningMode::kAutomatic};
  PlanDigest plan_digest;
  std::int64_t elapsed{};
  benchmarking::PlanningAllocationProbe allocations;
  const auto started = Clock::now();
  {
    const ExecutionPlan plan = compile(graph);
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
    compile_allocation_stats = allocations.snapshot();
    plan_ok = plan.ok();
    if (!plan_ok) {
      plan_error = plan.json();
    } else {
      accepted_edge_count = edge_count(plan);
      const auto partitions = plan.partitions();
      partitioning_mode = partitions.mode();
      partition_count = partitions.size();
      authored_partition_count = partitions.authored_count();
      automatic_partition_count = partitions.automatic_count();
      plan_digest = plan.digest();
      for (const ExecutionPartitionView partition : partitions) {
        auto& member_count = partition.origin() == ExecutionPartitionOrigin::kAuthored
                                 ? authored_member_count
                                 : automatic_member_count;
        member_count += partition.operator_paths().size();
      }
    }
  }
  const auto final_allocation_stats = allocations.finish();
  if (!plan_ok || final_allocation_stats.cross_thread_activity ||
      final_allocation_stats.cross_generation_activity) {
    std::cerr << (plan_ok ? "allocation probe observed cross-thread or cross-generation activity"
                          : plan_error)
              << '\n';
    return 2;
  }
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 2;
  }
#if defined(__APPLE__)
  const auto process_max_rss_kib = usage.ru_maxrss / 1024;
#else
  const auto process_max_rss_kib = usage.ru_maxrss;
#endif
  std::ofstream output(options.output, std::ios::out | std::ios::trunc);
  if (!output) {
    return 2;
  }
  output << "{\n"
         << "  \"schema\":\"holoscan.ea2.partition-planning-benchmark/v1\",\n"
         << "  \"topology\":\"" << options.topology << "\",\n"
         << "  \"operator_count\":" << options.operators << ",\n"
         << "  \"warmup_compile\":\"accepted_zero_work\",\n"
         << "  \"edge_count\":" << accepted_edge_count << ",\n"
         << "  \"compile_ns\":" << elapsed << ",\n"
         << "  \"allocation_count\":" << compile_allocation_stats.allocation_count << ",\n"
         << "  \"cumulative_requested_bytes\":"
         << compile_allocation_stats.cumulative_requested_bytes << ",\n"
         << "  \"peak_live_requested_bytes\":" << compile_allocation_stats.peak_live_requested_bytes
         << ",\n"
         << "  \"live_requested_bytes_after_compile\":"
         << compile_allocation_stats.live_requested_bytes << ",\n"
         << "  \"live_requested_bytes_after_plan_release\":"
         << final_allocation_stats.live_requested_bytes << ",\n"
         << "  \"process_max_rss_kib\":" << process_max_rss_kib << ",\n"
         << "  \"partitioning_mode\":\"" << mode_name(partitioning_mode) << "\",\n"
         << "  \"partition_count\":" << partition_count << ",\n"
         << "  \"authored_partition_count\":" << authored_partition_count << ",\n"
         << "  \"automatic_partition_count\":" << automatic_partition_count << ",\n"
         << "  \"authored_member_count\":" << authored_member_count << ",\n"
         << "  \"automatic_member_count\":" << automatic_member_count << ",\n"
         << "  \"plan_digest\":\"" << hex_digest(plan_digest) << "\"\n"
         << "}\n";
  return output ? 0 : 2;
}

}  // namespace
}  // namespace holoscan

int main(int argc, char** argv) {
  const auto options = holoscan::parse_options(argc, argv);
  if (!options) {
    std::cerr << "usage: partition_planning_benchmark --topology <linear|fanout|binary_fanout|"
                 "fanin|diamond|multi_root|embedded|hybrid> --operators <nonnegative> "
                 "--output <path>\n";
    return 2;
  }
  if (options->cross_generation_self_test) {
    return holoscan::run_cross_generation_self_test();
  }
  return holoscan::run(*options);
}
