/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Two identical 2 ms producers each emit one bounded 30-sample burst into
// two identical DrainAll sinks that share one direct 20 ms clock cadence. The
// only difference between the paths is edge/port policy authored on the
// Graph: bounded kBlocking with retry on one, kDropOldest + depth 1 on the other.
// Every behavioral difference the program reports is therefore composition
// policy, not operator code.
constexpr std::chrono::nanoseconds kProducerPeriod = 2ms;
constexpr std::chrono::nanoseconds kSinkPeriod = 20ms;
constexpr std::uint32_t kBurstCount = 30U;    // bounded burst per producer
constexpr std::uint32_t kBlockingDepth = 4U;  // deliberately saturates, then makes progress

enum class PathId : std::uint8_t { kBlocking = 0, kFreshest = 1 };

struct PathStats {
  std::uint64_t produced{};
  std::uint64_t received{};
  std::uint64_t dropped{};        // cumulative sequence gaps observed by the sink
  std::uint64_t backpressured{};  // emissions rejected kBackpressured, retried next tick
  std::uint32_t last_value{};
};

// Operators only record; main() prints after wait() so interleaved operator
// output can never split the summary lines the CTest regex anchors to.
// graph.op<> requires owned copyable constructor captures (no raw pointers),
// so per-path state lives in this static observation object and operators
// carry only a PathId enum.
struct CompositionObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    blocking = {};
    freshest = {};
  }

  [[nodiscard]] static PathStats& stats_for(PathId path) {
    return path == PathId::kBlocking ? blocking : freshest;
  }

  static void record_produced(PathId path) {
    {
      const std::lock_guard lock(mutex);
      stats_for(path).produced += 1U;
    }
    ready.notify_all();
  }

  static void record_backpressured(PathId path) {
    const std::lock_guard lock(mutex);
    stats_for(path).backpressured += 1U;
  }

  static void record_received(PathId path, std::uint32_t value) {
    {
      const std::lock_guard lock(mutex);
      PathStats& stats = stats_for(path);
      if (stats.received > 0U && value > stats.last_value + 1U) {
        stats.dropped += value - stats.last_value - 1U;
      }
      stats.last_value = value;
      stats.received += 1U;
    }
    ready.notify_all();
  }

  // The run is over when both bounded bursts were fully emitted, the
  // lossless path delivered every sample, and the lossy path delivered at
  // least its freshest survivor.
  [[nodiscard]] static bool wait_for_burst_complete() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] {
      return blocking.produced >= kBurstCount && freshest.produced >= kBurstCount &&
             blocking.received >= kBurstCount && freshest.received >= 1U;
    });
  }

  [[nodiscard]] static std::uint64_t freshest_received() {
    const std::lock_guard lock(mutex);
    return freshest.received;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline PathStats blocking{};
  static inline PathStats freshest{};
};

// Bounded burst source: one emission attempt per 2 ms tick, values
// 1..kBurstCount with no gaps at the producer, then idle ticks. A full
// kBlocking edge returns finite kBackpressured; retaining next_ retries the
// exact value after the sink drains capacity.
class BurstSource final : public holoscan::Operator<> {
 public:
  explicit BurstSource(PathId path) noexcept : path_(path) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kProducerPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= kBurstCount) {
      return {};  // burst finished: tick idly until the session stops
    }
    auto emitted = output.emit(next_ + 1U);
    if (!emitted) {
      // Pre-saturation admission pressure (e.g. an in-flight cap) is a
      // transient *value*: keep the sample (next_ is not advanced) and retry
      // it on the next 2 ms tick, so the producer's sequence stays gap-free.
      if (emitted.error().code == holoscan::ErrorCode::kBackpressured) {
        CompositionObservation::record_backpressured(path_);
        return {};
      }
      // If an emission fails because the session is stopping, treat that as
      // cooperative shutdown rather than a pipeline failure.
      if (context.cancellation().stop_requested()) {
        return {};
      }
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    ++next_;
    CompositionObservation::record_produced(path_);
    return {};
  }

  holoscan::Output<std::uint32_t> output;

 private:
  PathId path_;
  std::uint32_t next_{};
};

// Deliberately slow sink: activates directly once per 20 ms clock tick and drains
// whatever survived in its data queue. DrainAll is not failure-capable, so no
// on_missing is declared; an empty batch is a valid activation. DrainAll is a
// deliberate whole-batch choice here; a scalar selector would instead make a
// bounded one-sample choice on each clock tick. On the blocking path, a full
// queue returns finite kBackpressured and the source retries the exact value.
class SlowSink final : public holoscan::Operator<> {
 public:
  explicit SlowSink(PathId path) noexcept : path_(path) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(kBlockingDepth);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kSinkPeriod}).input(input, holoscan::DrainAll{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto batch = input.selected_batch();
    if (!batch) {
      return holoscan::make_unexpected(std::move(batch).error());
    }
    for (const holoscan::Sample<std::uint32_t>& sample : *batch) {
      CompositionObservation::record_received(path_, sample.data);
    }
    return {};
  }

  holoscan::Input<std::uint32_t> input;

 private:
  PathId path_;
};

// ---- GraphView rendering helpers -----------------------------------------

[[nodiscard]] std::string hash_prefix(const holoscan::GraphAuthoringHash& hash) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string text;
  for (std::size_t index = 0; index < 4U && index < hash.value.size(); ++index) {
    const auto byte = std::to_integer<unsigned>(hash.value[index]);
    text.push_back(digits[byte >> 4U]);
    text.push_back(digits[byte & 0xFU]);
  }
  return text;
}

[[nodiscard]] std::string_view vertex_kind_name(holoscan::VertexKind kind) {
  switch (kind) {
    case holoscan::VertexKind::kOperator:
      return "operator";
    case holoscan::VertexKind::kGraphInterface:
      return "graph-interface";
    case holoscan::VertexKind::kImportedTopic:
      return "imported-topic";
    case holoscan::VertexKind::kExportedTopic:
      return "exported-topic";
  }
  return "unknown";
}

[[nodiscard]] std::string_view policy_name(holoscan::QueuePolicy policy) {
  switch (policy) {
    case holoscan::QueuePolicy::kBlocking:
      return "kBlocking";
    case holoscan::QueuePolicy::kDropOldest:
      return "kDropOldest";
  }
  return "unknown";
}

[[nodiscard]] std::string optional_policy(const std::optional<holoscan::QueuePolicy>& policy) {
  return policy ? std::string{policy_name(*policy)} : std::string{"-"};
}

[[nodiscard]] std::string optional_u32(const std::optional<std::uint32_t>& value) {
  return value ? std::to_string(*value) : std::string{"-"};
}

[[nodiscard]] std::string optional_text(const std::optional<std::string>& value) {
  return value ? *value : std::string{"-"};
}

[[nodiscard]] std::string_view constraint_name(holoscan::RegionConstraintKind kind) {
  switch (kind) {
    case holoscan::RegionConstraintKind::kStaticTopology:
      return "static_topology";
    case holoscan::RegionConstraintKind::kBoundedMemory:
      return "bounded_memory";
    case holoscan::RegionConstraintKind::kDeviceCompatibleOperators:
      return "device_compatible_ops";
    case holoscan::RegionConstraintKind::kLatencyBudget:
      return "latency_budget";
  }
  return "unknown";
}

void print_region_constraints(const holoscan::RegionRecord& region) {
  for (std::size_t index = 0; index < region.constraints.size(); ++index) {
    std::cout << (index == 0U ? "" : ",") << constraint_name(region.constraints[index].kind());
  }
}

// Walk one frozen authoring snapshot and print every record family the
// example authored. The snapshot is immutable and deep-owning: it stays
// valid even if the source Graph mutates afterwards, and the two-layer
// setup/graph_override port records preserve author intent unresolved.
void print_view(const holoscan::GraphView& view) {
  std::unordered_map<std::uint64_t, std::string> vertex_names;
  for (const holoscan::VertexRecord& vertex : view.vertices()) {
    vertex_names.emplace(vertex.id.value, vertex.name);
  }
  const auto vertex_name = [&vertex_names](holoscan::VertexId id) -> std::string {
    const auto found = vertex_names.find(id.value);
    return found == vertex_names.end() ? std::string{"?"} : found->second;
  };
  std::unordered_map<std::uint64_t, std::string> port_names;
  for (const holoscan::PortRecord& port : view.ports()) {
    port_names.emplace(port.id.value, vertex_name(port.owner) + "." + port.name);
  }
  const auto port_name = [&port_names](holoscan::GraphPortId id) -> std::string {
    const auto found = port_names.find(id.value);
    return found == port_names.end() ? std::string{"?"} : found->second;
  };

  const holoscan::GraphAuthoringIdentity identity = view.identity();
  std::cout << "[view] generation=" << view.generation() << " identity.version=" << identity.version
            << " identity.hash=" << hash_prefix(identity.hash) << "..\n";
  for (const holoscan::VertexRecord& vertex : view.vertices()) {
    std::cout << "[view] vertex id=" << vertex.id.value << " kind=" << vertex_kind_name(vertex.kind)
              << " name=" << vertex.name << '\n';
  }
  for (const holoscan::PortRecord& port : view.ports()) {
    std::cout << "[view] port " << port_name(port.id)
              << " dir=" << (port.direction == holoscan::PortDirection::kInput ? "in" : "out")
              << " setup.depth=" << optional_u32(port.setup.queue_depth)
              << " override.depth=" << optional_u32(port.graph_override.queue_depth)
              << " alias=" << optional_text(port.graph_override.diagnostic_alias) << '\n';
  }
  for (const holoscan::EdgeRecord& edge : view.edges()) {
    std::cout << "[view] edge " << port_name(edge.source) << " -> " << port_name(edge.target)
              << " policy=" << policy_name(edge.add_flow_options.queue_policy)
              << " depth=" << optional_u32(edge.add_flow_options.queue_depth)
              << " override.policy=" << optional_policy(edge.graph_override.queue_policy)
              << " override.inflight=" << optional_u32(edge.graph_override.max_inflight_samples)
              << '\n';
  }
  for (const holoscan::RegionRecord& region : view.regions()) {
    std::cout << "[view] region name=" << region.name << " members=[";
    for (std::size_t index = 0; index < region.members.size(); ++index) {
      std::cout << (index == 0U ? "" : ",") << vertex_name(region.members[index]);
    }
    std::cout << "] constraints=[";
    print_region_constraints(region);
    std::cout << "]\n";
  }
  for (const holoscan::ClockDeclarationRecord& clock : view.clocks()) {
    std::cout << "[view] clock name=" << clock.key.name()
              << " default=" << (clock.is_default ? "yes" : "no") << '\n';
  }
  for (const holoscan::ClockConversionAuthoringRecord& conversion : view.clock_conversions()) {
    std::cout << "[view] conversion " << conversion.from.name() << " -> " << conversion.to.name()
              << " scale=" << conversion.conversion.scale.numerator << '/'
              << conversion.conversion.scale.denominator
              << " offset_ns=" << conversion.conversion.offset_ns << '\n';
  }
}

}  // namespace

// REQ: SYSTEM-014
int main() {
  CompositionObservation::reset();

  holoscan::Graph graph{"ea1-graph-composition"};
  const auto burst_blocking = graph.op<BurstSource>("burst_blocking", PathId::kBlocking);
  const auto burst_freshest = graph.op<BurstSource>("burst_freshest", PathId::kFreshest);
  const auto sink_blocking = graph.op<SlowSink>("sink_blocking", PathId::kBlocking);
  const auto sink_freshest = graph.op<SlowSink>("sink_freshest", PathId::kFreshest);

  // Path 1 -- lossless: kBlocking (the default, written out here) reports
  // finite pressure when its bounded queue fills. BurstSource retries the
  // same value, so every accepted sample remains ordered and gap-free.
  (void)graph.add_flow(burst_blocking->output,
                       sink_blocking->input,
                       holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                                   .queue_depth = kBlockingDepth});

  // Path 2 -- freshest-wins: authored plain, then reshaped after the fact
  // through configurators. The flow handle is kept because configure(flow)
  // needs it.
  auto freshest_flow = graph.add_flow(burst_freshest->output, sink_freshest->input);

  // Port configurator (throwing form): shrink the sink's input history to a
  // single sample and give the port a diagnostics-only alias.
  graph.configure(sink_freshest->input).queue_depth(1).diagnostic_alias("sink");

  // Flow configurator through the non-throwing try_configure twin: the same
  // fluent verbs, but lookup failure comes back as expected<_, Error>.
  auto freshest_configurator = graph.try_configure(freshest_flow);
  if (!freshest_configurator) {
    std::cerr << "flow configurator lookup failed: "
              << holoscan::error_code_name(freshest_configurator.error().code) << '\n';
    return 1;
  }
  // max_inflight_samples is sized above the whole burst so the cap never
  // binds here; the lesson is the fluent verb on the flow configurator.
  freshest_configurator.value()
      .queue_policy(holoscan::QueuePolicy::kDropOldest)
      .max_inflight_samples(64U);

  // Producers and deliberately slow clocked sinks own independent cadences.
  graph.partition("blocking-source").add(burst_blocking);
  graph.partition("blocking-sink").add(sink_blocking);
  graph.partition("freshest-source").add(burst_freshest);
  graph.partition("freshest-sink").add(sink_freshest);

  // try_configure is the probe form for data-driven composition: a port that
  // belongs to a different Graph is a value-shaped rejection, not a throw.
  holoscan::Graph foreign{"foreign"};
  const auto rejected = foreign.try_configure(sink_freshest->input);
  if (rejected) {
    std::cerr << "foreign try_configure unexpectedly succeeded\n";
    return 1;
  }
  const std::string foreign_code{holoscan::error_code_name(rejected.error().code)};

  // Region: name the lossless path "critical" and attach a backend-neutral
  // bounded-memory constraint for the compiler to consume.
  graph.region("critical")
      .add(sink_blocking)
      .constrain(holoscan::RegionConstraint::bounded_memory());

  // Two clock domains: the runtime default plus a deterministic capture
  // clock, related by a directed affine conversion (identity rate, +0.5 ms).
  const holoscan::ClockKey runtime_clock = graph.add_clock<holoscan::RealtimeClock>("runtime");
  const holoscan::ClockKey capture_clock =
      graph.add_clock<holoscan::ManualClock>("capture", std::int64_t{0});
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(
      capture_clock, runtime_clock, holoscan::AffineClockConversion{.offset_ns = 500'000});

  // Freeze and walk the authoring snapshot BEFORE compiling: everything the
  // program authored above is visible as pointer-free records.
  const holoscan::GraphView view = graph.view();
  print_view(view);

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = CompositionObservation::wait_for_burst_complete();
  // Let the freshest-wins path settle: its counts are final once no new
  // sample lands across two consecutive quiet 100 ms polls (the sink keeps
  // ticking every 20 ms, so a quiet interval means the queue is drained).
  if (complete) {
    std::uint64_t quiet_polls = 0U;
    std::uint64_t previous = CompositionObservation::freshest_received();
    for (std::uint32_t poll = 0U; poll < 20U && quiet_polls < 2U; ++poll) {
      std::this_thread::sleep_for(100ms);
      const std::uint64_t current = CompositionObservation::freshest_received();
      quiet_polls = current == previous ? quiet_polls + 1U : 0U;
      previous = current;
    }
  }
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "graph composition run failed (complete=" << complete
              << " termination=" << static_cast<int>(session.termination()) << ")\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  diagnostic: token=" << diagnostic.token << '\n';
    }
    return 2;
  }

  // All operator callbacks have retired after wait(); everything below is
  // stable post-stop output.
  const std::lock_guard lock(CompositionObservation::mutex);
  const PathStats& blocking = CompositionObservation::blocking;
  const PathStats& freshest = CompositionObservation::freshest;
  // Dropped = emitted but never received. The gap counter cross-checks the
  // lossless path: kBlocking must deliver the full burst in order.
  const std::uint64_t blocking_lost = blocking.produced - blocking.received;
  const std::uint64_t freshest_lost = freshest.produced - freshest.received;
  if (blocking_lost != 0U || blocking.dropped != 0U || blocking.backpressured == 0U ||
      freshest_lost == 0U) {
    std::cerr << "unexpected drop pattern: blocking_lost=" << blocking_lost
              << " blocking_gaps=" << blocking.dropped
              << " blocking_backpressured=" << blocking.backpressured
              << " freshest_lost=" << freshest_lost << '\n';
    return 3;
  }
  std::cout << "[graph_composition] try_configure foreign-port rejected=yes code=" << foreign_code
            << " (value, not exception)\n";
  std::cout << "[graph_composition] blocking produced=" << blocking.produced
            << " received=" << blocking.received << " dropped=" << blocking_lost
            << " gaps=" << blocking.dropped << " backpressured=" << blocking.backpressured << '\n';
  std::cout << "[graph_composition] freshest produced=" << freshest.produced
            << " received=" << freshest.received << " dropped=" << freshest_lost << '\n';
  std::cout << "[graph_composition] result blocking_dropped=0 freshest_dropped=" << freshest_lost
            << '\n';
  std::cout << "graph composition complete\n";
  return 0;
}
