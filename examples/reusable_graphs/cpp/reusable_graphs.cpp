/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Typed reusable pipeline components.
//
// Part 1 (authoring): one ScalePairGraph is embedded directly and another is
// embedded through a WrappedScaleGraph that re-exports the inner interface
// across one nesting level. The parent wires both paths through exported
// interface ports, applies one recursive region policy, and explicitly binds
// both reusable clock-import chains to one concrete root clock.
//
// Part 2 (runtime replay): the cross-scope plan runs as authored. The exact
// same immutable ExecutionPlan is started twice; fresh operator instances in
// both sessions produce the same first scaled value under the same imported
// scheduling domain.
//
// A clearly marked what-not-to-do block shows why a retained GraphHandle
// must be revalidated with graph_valid() after its parent Graph retires.

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kTickPeriod = 2ms;
constexpr std::size_t kInstanceCount = 2U;       // direct-stage + wrapped-stage/inner
constexpr std::size_t kSamplesPerInstance = 6U;  // Bounded-run target.
constexpr std::size_t kArrivalPreview = 3U;      // Arrival entries echoed post-stop.
constexpr std::uint32_t kDirectFactor = 10U;
constexpr std::uint32_t kWrappedFactor = 1000U;

// Results are collected here and printed from main() after each session's
// wait(); per-sample stdout from operator threads would interleave
// non-atomically. Sinks record under their instance-path label, so the
// arrival log carries instance identity end to end.
struct InstanceObservation {
  struct Totals {
    std::size_t samples{};
    std::uint32_t first_value{};
    std::uint32_t last_value{};
  };

  static void reset() {
    const std::lock_guard lock(mutex);
    arrivals.clear();
    totals.clear();
  }

  static void record(const std::string& label, std::uint32_t value) {
    {
      const std::lock_guard lock(mutex);
      arrivals.emplace_back(label, value);
      Totals& entry = totals[label];
      if (entry.samples == 0U) {
        entry.first_value = value;
      }
      ++entry.samples;
      entry.last_value = value;
    }
    ready.notify_all();
  }

  // Bounded wait: one replay is demonstrated once every embedded instance
  // delivered its sample target.
  [[nodiscard]] static bool wait_for_all(std::size_t per_instance) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [per_instance] {
      constexpr std::array<std::string_view, kInstanceCount> labels{"direct-stage",
                                                                    "wrapped-stage/inner"};
      return std::ranges::all_of(labels, [per_instance](const std::string_view label) {
        const auto found = totals.find(std::string{label});
        return found != totals.end() && found->second.samples >= per_instance;
      });
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<std::pair<std::string, std::uint32_t>> arrivals;  // Guarded by mutex.
  static inline std::map<std::string, Totals> totals;                         // Guarded by mutex.
};

// Emits 1, 2, 3, ... every kTickPeriod and fans the same sequence to both
// reusable paths, making their independent constructor factors visible.
class TickSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = ++next_;
    return output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t next_{};
};

// The private body of ScalePairGraph: multiplies each sample by the factor
// the graph instance was constructed with. Parents never see this operator —
// only the graph's exported interface ports are reachable from outside.
class ScaleOp final : public holoscan::Operator<> {
 public:
  explicit ScaleOp(std::uint32_t factor) noexcept : factor_(factor) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
    spec.output(out, "out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    sample->data *= factor_;
    return out.emit(std::move(*sample));
  }

  holoscan::Input<std::uint32_t> in;
  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t factor_{};
};

// Terminal consumer; records every received value under the instance-path
// label it was constructed with (an owned std::string copy).
class LabeledSink final : public holoscan::Operator<> {
 public:
  explicit LabeledSink(std::string label) : label_(std::move(label)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    InstanceObservation::record(label_, sample->data);
    return {};
  }

  holoscan::Input<std::uint32_t> input;

 private:
  std::string label_;
};

// The interface-first reusable component: a Graph subtype whose *public C++
// surface is its interface* — one Input/Output port pair, the same port
// types operators declare. The body is private: compose() creates the
// internal operator and binds the interface pair to it with
// export_input/export_output (interface member first, inner port second).
// Anything not exported fails closed: a parent wiring to an internal port
// gets GRAPH_PORT_NOT_FOUND at authoring time.
// REQ: GRAPH-021
class ScalePairGraph final : public holoscan::Graph {
 public:
  explicit ScalePairGraph(std::uint32_t factor) noexcept : factor_(factor) {}

  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

  holoscan::Input<std::uint32_t> in{"in"};
  holoscan::Output<std::uint32_t> out{"out"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto scale = op<ScaleOp>("scale", factor_);
    export_input(in, scale->in);
    export_output(out, scale->out);
  }

 private:
  std::uint32_t factor_{};
  holoscan::ClockKey clock_import_;
};

// One nesting level: a reusable graph may itself embed reusable graphs.
// Visibility follows one rule — only ports exported at *each* level are
// reachable from further out — so this wrapper re-exports the inner
// instance's interface through its own public port pair. The nested instance
// gets the two-segment root-qualified path "wrapped-stage/inner".
// REQ: GRAPH-021
class WrappedScaleGraph final : public holoscan::Graph {
 public:
  explicit WrappedScaleGraph(std::uint32_t factor) noexcept : factor_(factor) {}

  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

  holoscan::Input<std::uint32_t> in{"in"};
  holoscan::Output<std::uint32_t> out{"out"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto inner = embed<ScalePairGraph>("inner", factor_);
    bind_clock(inner, inner->clock_import(), clock_import_);
    export_input(in, inner->in);  // Re-export: my public port -> child's exported port.
    export_output(out, inner->out);
  }

 private:
  std::uint32_t factor_{};
  holoscan::ClockKey clock_import_;
};

// Render a GraphInstancePath for display. Segments are the identity — names
// are literal segments (a name could itself contain '/'), so never split a
// rendered string back apart; always go through segments().
[[nodiscard]] std::string render_path(const std::optional<holoscan::GraphInstancePath>& path) {
  if (!path) {
    return "_root";  // Rendering token for root scope; never a real segment.
  }
  std::string rendered;
  for (const auto& segment : path->segments()) {
    if (!rendered.empty()) {
      rendered += '/';
    }
    rendered += segment;
  }
  return rendered;
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    std::cerr << "  diagnostic token=" << diagnostic.token
              << " severity=" << static_cast<int>(diagnostic.severity)
              << " service=" << diagnostic.service << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << " vertex=" << diagnostic.source.vertex.value
              << '\n';
  }
}

// Run one fresh session from the same compiled plan and prove that both
// reusable instances are reconstructed with their independent constructor state.
[[nodiscard]] bool run_replay(const holoscan::ExecutionPlan& plan, std::size_t replay_index) {
  InstanceObservation::reset();
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = InstanceObservation::wait_for_all(kSamplesPerInstance);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "replay " << replay_index
              << " did not stop cleanly at its sample target:" << " complete=" << complete
              << " termination=" << static_cast<int>(session.termination()) << '\n';
    print_diagnostics(session);
    return false;
  }

  std::vector<std::pair<std::string, std::uint32_t>> arrivals;
  std::map<std::string, InstanceObservation::Totals> totals;
  {
    const std::lock_guard lock(InstanceObservation::mutex);
    arrivals = InstanceObservation::arrivals;
    totals = InstanceObservation::totals;
  }

  const bool scaled_as_authored =
      totals.size() == kInstanceCount && totals.contains("direct-stage") &&
      totals.contains("wrapped-stage/inner") &&
      totals.at("direct-stage").first_value == kDirectFactor &&
      totals.at("direct-stage").last_value % kDirectFactor == 0U &&
      totals.at("wrapped-stage/inner").first_value == kWrappedFactor &&
      totals.at("wrapped-stage/inner").last_value % kWrappedFactor == 0U;
  if (!scaled_as_authored) {
    std::cerr << "replay " << replay_index << " did not reconstruct both instances\n";
    return false;
  }

  std::cout << "[reusable_graphs] replay=" << replay_index << " termination=stopped\n";
  for (const auto& [path, record] : totals) {
    std::cout << "[reusable_graphs] replay=" << replay_index << " instance path=" << path
              << " samples=" << record.samples << " first=" << record.first_value
              << " last=" << record.last_value << " values:";
    std::size_t shown = 0U;
    for (const auto& [label, value] : arrivals) {
      if (label == path && shown < kArrivalPreview) {
        std::cout << ' ' << value;
        ++shown;
      }
    }
    std::cout << '\n';
  }
  return scaled_as_authored;
}

struct InstanceCatalog {
  [[nodiscard]] bool complete() const noexcept {
    return !direct_path.empty() && !nested_path.empty() && direct_id && wrapped_id &&
           paths.size() == 3U;
  }

  std::vector<std::string> paths;
  std::string direct_path;
  std::string nested_path;
  holoscan::GraphId direct_id{};
  holoscan::GraphId wrapped_id{};
};

/// Collect instance identities from one authoring or canonical-plan Graph record set.
[[nodiscard]] InstanceCatalog collect_instance_catalog(
    std::span<const holoscan::GraphRecord> graphs) {
  InstanceCatalog catalog;
  for (const auto& record : graphs) {
    if (!record.path) {
      continue;  // Root scope has no path.
    }
    catalog.paths.push_back(render_path(record.path));
    if (catalog.paths.back() == "direct-stage") {
      catalog.direct_path = catalog.paths.back();
      catalog.direct_id = record.id;
    } else if (catalog.paths.back() == "wrapped-stage") {
      catalog.wrapped_id = record.id;
    }
    if (record.path->size() == 2U) {
      catalog.nested_path = catalog.paths.back();  // The re-exported inner instance.
    }
  }
  return catalog;
}

[[nodiscard]] bool verify_stage_region(const holoscan::GraphView& view, holoscan::GraphId direct_id,
                                       holoscan::GraphId wrapped_id) {
  bool region_as_authored = false;
  for (const holoscan::RegionRecord& region : view.regions()) {
    if (region.name != "stage") {
      continue;
    }
    const bool has_direct =
        std::ranges::find(region.graph_members, direct_id) != region.graph_members.end();
    const bool has_wrapped =
        std::ranges::find(region.graph_members, wrapped_id) != region.graph_members.end();
    const bool bounded = std::ranges::any_of(region.constraints, [](const auto& constraint) {
      return constraint.kind() == holoscan::RegionConstraintKind::kBoundedMemory;
    });
    region_as_authored = region.graph_members.size() == 2U && has_direct && has_wrapped && bounded;
  }
  if (region_as_authored) {
    return true;
  }

  std::cerr << "stage region did not retain both reusable members and bounded_memory\n";
  std::cerr << "  expected_graph_ids=" << direct_id.value << ',' << wrapped_id.value << '\n';
  for (const holoscan::RegionRecord& region : view.regions()) {
    std::cerr << "  region=" << region.name << " graph_members=";
    for (const holoscan::GraphId member : region.graph_members) {
      std::cerr << member.value << ',';
    }
    std::cerr << " constraints=" << region.constraints.size() << '\n';
  }
  return false;
}

[[nodiscard]] bool demonstrate_stale_handle() {
  std::optional<holoscan::GraphHandle<ScalePairGraph>> stale;
  bool valid_in_scope = false;
  {
    holoscan::Graph temporary{"temporary"};
    stale = temporary.embed<ScalePairGraph>("short-lived", 2U);
    valid_in_scope = stale->graph_valid();  // True: parent still owns the child.
  }  // `temporary` is destroyed here; `stale` now names a retired child.
  const bool valid_after_scope = stale->graph_valid();      // False.
  const bool path_after_scope = stale->path().has_value();  // Empty.
  std::string stale_deref_token{"<none>"};
  try {
    (void)(*stale)->in;  // Stale dereference throws; it never returns garbage.
  } catch (const holoscan::RuntimeError& error) {
    stale_deref_token = holoscan::make_error(error).message;
  }

  std::cout << "[reusable_graphs] stale-handle demo: valid_in_scope=" << valid_in_scope
            << " valid_after_scope_exit=" << valid_after_scope
            << " path_after_scope_exit=" << (path_after_scope ? "present" : "absent")
            << " deref_error=" << stale_deref_token << '\n';
  return valid_in_scope && !valid_after_scope && !path_after_scope &&
         stale_deref_token == "GRAPH_HANDLE_INVALID";
}

}  // namespace

int main() {
  InstanceObservation::reset();

  // -----------------------------------------------------------------------
  // Part 1 — authoring: one direct and one nested reusable instance wired
  // through exported interface ports.
  // -----------------------------------------------------------------------
  holoscan::Graph authoring{"reusable-graphs"};
  const auto source = authoring.op<TickSource>("source");

  // One instance is direct; the wrapper embeds and re-exports the other.
  // Each handle is weak, read-only, and typed.
  const auto direct = authoring.embed<ScalePairGraph>("direct-stage", kDirectFactor);
  const auto wrapped = authoring.embed<WrappedScaleGraph>("wrapped-stage", kWrappedFactor);

  // handle.path() is *parent-relative* (single segment here). Root-qualified
  // identity — including the nested "wrapped-stage/inner" — lives in the
  // GraphView records, the durable source of truth for instance identity.
  const holoscan::GraphView view = authoring.view();  // Durable snapshot; outlives handles.
  const InstanceCatalog catalog = collect_instance_catalog(view.graphs());
  if (!catalog.complete()) {
    std::cerr << "expected direct, wrapper, and inner instance records\n";
    return 1;
  }

  // Sinks are labeled with the instance path of the stage that fed them.
  const auto direct_sink = authoring.op<LabeledSink>("direct-sink", catalog.direct_path);
  const auto wrapped_sink = authoring.op<LabeledSink>("wrapped-sink", catalog.nested_path);

  // Parents wire to exported interface ports with the same compile-checked
  // add_flow used for ordinary operator ports.
  authoring.add_flow(source->output, direct->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  authoring.add_flow(source->output, wrapped->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  authoring.add_flow(
      direct->out, direct_sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  authoring.add_flow(
      wrapped->out, wrapped_sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});

  // Clock ownership is explicit at every reusable boundary. The child imports
  // a named dependency and uses it as its default; the direct parent binds
  // that exact import to its own declaration or import. Name matching never
  // substitutes for these instance-qualified records.
  const holoscan::ClockKey runtime_clock =
      authoring.add_clock<holoscan::RealtimeClock>("runtime-clock");
  authoring.set_default_clock(runtime_clock);
  authoring.bind_clock(direct, direct->clock_import(), runtime_clock);
  authoring.bind_clock(wrapped, wrapped->clock_import(), runtime_clock);

  // A region may contain embedded graph handles directly. Containing the
  // wrapper recursively covers its inner instance without flattening or
  // exposing that implementation detail to the parent.
  auto stage_region = authoring.region("stage");
  stage_region.add(direct);
  stage_region.add(wrapped);
  stage_region.constrain(holoscan::RegionConstraint::bounded_memory());

  // Select whole reusable instances without reaching into their private Operators. Each sink
  // shares its stage's owner; the compiler assigns only the remaining source automatically.
  authoring.partition("direct-owner").add(direct).add(direct_sink);
  authoring.partition("wrapped-owner").add(wrapped).add(wrapped_sink);
  authoring.auto_partition_remaining();

  const holoscan::ExecutionPlan plan = holoscan::compile(authoring);
  if (!plan.ok()) {
    std::cerr << "cross-scope reusable graph was not admitted:\n" << plan.json() << '\n';
    return 1;
  }

  const InstanceCatalog compiled_catalog = collect_instance_catalog(plan.graph().graphs());
  for (const std::string_view name : {"direct-owner", "wrapped-owner"}) {
    const auto partition = plan.partitions().lookup_named(name);
    const holoscan::GraphId expected_graph =
        name == "direct-owner" ? compiled_catalog.direct_id : compiled_catalog.wrapped_id;
    if (!partition || partition->graph_members().size() != 1U ||
        partition->graph_members().front() != expected_graph ||
        partition->operator_paths().size() != 2U) {
      std::cerr << "atomic instance selection was not preserved for " << name << '\n';
      return 1;
    }
    std::cout << "[reusable_graphs] partition=" << name << " selected_instances=1 operators=2\n";
  }

  const holoscan::GraphView final_view = authoring.view();
  const bool region_as_authored =
      verify_stage_region(final_view, catalog.direct_id, catalog.wrapped_id);
  if (!region_as_authored) {
    return 1;
  }

  std::cout << "[reusable_graphs] authored instance paths:";
  for (const std::string& path : catalog.paths) {
    std::cout << ' ' << path;
  }
  std::cout << '\n';
  std::cout << "[reusable_graphs] clock imports: direct and wrapped bound to runtime-clock\n";
  std::cout << "[reusable_graphs] region=stage graph_members=2 constraint=bounded_memory\n";

  // -----------------------------------------------------------------------
  // Part 2 — runtime replay: start two fresh sessions from the same immutable
  // plan. Reconstructed child defaults resolve through the frozen import
  // chain to the one root scheduling domain on both runs.
  // -----------------------------------------------------------------------
  const bool replay_one_ok = run_replay(plan, 1U);
  const bool replay_two_ok = run_replay(plan, 2U);

  // ---------------------------------------------------------------------
  // WHAT NOT TO DO: retaining a GraphHandle past its parent Graph's
  // lifetime. The parent owns the child; the handle is weak and never
  // extends that lifetime. After the parent retires, graph_valid() is
  // false, path() is empty, and dereferencing throws an error tagged
  // GRAPH_HANDLE_INVALID instead of returning garbage. Check
  // graph_valid() (or `if (handle)`) on any handle you retain; for
  // anything you persist or display, take a durable GraphView snapshot.
  // ---------------------------------------------------------------------
  const bool stale_demo_ok = demonstrate_stale_handle();

  // Semantic claims this example demonstrates:
  //  - explicit instance-qualified clock imports resolve to the root clock;
  //  - both instances produce their independently scaled output in two fresh
  //    sessions replayed from the same plan;
  //  - a retained handle went stale exactly as documented.
  const bool paths_as_expected =
      catalog.direct_path == "direct-stage" && catalog.nested_path == "wrapped-stage/inner";
  if (!paths_as_expected || !region_as_authored || !replay_one_ok || !replay_two_ok ||
      !stale_demo_ok) {
    std::cerr << "reusable_graphs FAIL: paths_as_expected=" << paths_as_expected
              << " region_as_authored=" << region_as_authored << " replay_one_ok=" << replay_one_ok
              << " replay_two_ok=" << replay_two_ok << " stale_demo_ok=" << stale_demo_ok << '\n';
    return 4;
  }

  std::cout << "reusable_graphs PASS: direct-stage and wrapped-stage/inner replayed twice "
               "region=stage termination=stopped\n";
  return 0;
}
