/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// compile_and_check: the guide's compile-and-check workflow as a program.
//
// Five compile calls cover four workflows built from the same two-operator pipeline:
//   1. admitted         — default clock set; plan.ok(), plan.json() written to a
//                         file;
//   2. broken           — a clock is declared but never made the default, so the
//                         OnClock trigger cannot resolve; the stable diagnostics
//                         table is printed and require_no_errors() throws at a
//                         cold path;
//   3. ok-with-warnings — a std::function-wrapped (type-erased) predicate is
//                         admitted under Profile::kStandard with a retained
//                         warning diagnostic;
//   4. digest gate      — two independent builds of the identical graph produce
//                         byte-equal plan.json() and equal plan.digest(); a small
//                         tuning edit changes the digest — the CI baseline check.
// A short clean run of the admitted plan closes the loop.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../common/temporary_workspace.hpp"

namespace {

using namespace std::chrono_literals;

// Observation state for the optional clean run. Static, not captured: graph.op<>
// rejects borrowed pointer captures, and compute() must not dangle.
struct TickObservation {
  static void reset() noexcept {
    const std::lock_guard lock(mutex);
    count = 0U;
  }

  static void record() {
    {
      const std::lock_guard lock(mutex);
      ++count;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_at_least(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return count >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t count{};
};

// Fixed-rate producer. OnClock omits the clock key, so compilation must resolve
// the graph default clock — exactly the dependency the broken variant violates.
class Ticker final : public holoscan::Operator<> {
 public:
  explicit Ticker(std::chrono::nanoseconds period) noexcept : period_(period) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const std::uint32_t value = ++next_;
    return output.emit(value);
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::chrono::nanoseconds period_;
  std::uint32_t next_{};
};

// Consumes every tick; only exercised by the clean run at the end.
class CountingSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

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
    TickObservation::record();
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

// Sink whose predicate is wrapped in std::function. Type erasure is admitted
// under Profile::kStandard, but the compiler keeps a warning diagnostic in the
// plan — the reachable ok-with-warnings case this example demonstrates.
class GatedSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(driver, "driver").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    std::function<bool(const holoscan::PredicateEvaluationContext&,
                       const holoscan::PredicateInputView<std::uint32_t>&)>
        erased = [](const holoscan::PredicateEvaluationContext&,
                    const holoscan::PredicateInputView<std::uint32_t>&) { return true; };
    holoscan::Contract result;
    result
        .trigger(holoscan::OnPredicate{
            .evaluate_on = holoscan::OnInputUpdate{driver},
            .predicate = holoscan::predicate("always-admit", std::move(erased))})
        .input(driver, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = driver.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }

  holoscan::Input<std::uint32_t> driver;
};

enum class DefaultClock : std::uint8_t {
  kSet,           // add_clock + set_default_clock — the admitted shape.
  kDeclaredOnly,  // add_clock only — the subtle broken shape: a clock exists,
                  // but no clock is *the default*, so OnClock cannot resolve.
};

// One authoring function for every ticker->sink variant, so the digest section
// compares plans whose only differences are the ones we intend.
void build_pipeline(holoscan::Graph& graph, std::chrono::nanoseconds tick_period,
                    std::uint32_t sink_queue_depth, DefaultClock default_clock) {
  const auto ticker = graph.op<Ticker>("ticker", tick_period);
  const auto sink = graph.op<CountingSink>("sink");
  graph.add_flow(
      ticker->output, sink->input, holoscan::ConnectionOptions{.queue_depth = sink_queue_depth});
  const auto clock = graph.add_clock<holoscan::RealtimeClock>("app-clock");
  if (default_clock == DefaultClock::kSet) {
    graph.set_default_clock(clock);
  }
}

void build_gated_pipeline(holoscan::Graph& graph) {
  const auto ticker = graph.op<Ticker>("ticker", std::chrono::nanoseconds{10ms});
  const auto gate = graph.op<GatedSink>("gate");
  graph.add_flow(ticker->output, gate->driver, holoscan::ConnectionOptions{.queue_depth = 2U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("app-clock"));
}

[[nodiscard]] constexpr std::string_view severity_name(
    holoscan::DiagnosticSeverity severity) noexcept {
  return severity == holoscan::DiagnosticSeverity::kError ? "error" : "warning";
}

void print_diagnostics(std::string_view label, const holoscan::ExecutionPlan& plan) {
  for (const holoscan::Diagnostic& item : plan.diagnostics()) {
    std::cout << '[' << label << "]   " << item.token << " | " << severity_name(item.severity);
    if (!item.json_pointer.empty()) {
      std::cout << " | at " << item.json_pointer;
    }
    if (!item.provided.empty()) {
      std::cout << " | provided " << item.provided;
    }
    std::cout << '\n';
  }
}

[[nodiscard]] bool has_diagnostic(const holoscan::ExecutionPlan& plan, std::string_view token,
                                  holoscan::DiagnosticSeverity severity) {
  for (const holoscan::Diagnostic& item : plan.diagnostics()) {
    if (item.token == token && item.severity == severity) {
      return true;
    }
  }
  return false;
}

// First 8 bytes of the 256-bit digest, as lowercase hex — enough to eyeball.
[[nodiscard]] std::string digest_prefix(holoscan::PlanDigest digest) {
  static constexpr std::string_view kHex = "0123456789abcdef";
  std::string text;
  for (std::size_t index = 0; index < 8U; ++index) {
    const auto byte = std::to_integer<unsigned>(digest.value[index]);
    text.push_back(kHex[byte >> 4U]);
    text.push_back(kHex[byte & 0xFU]);
  }
  return text;
}

}  // namespace

// REQ: SYSTEM-014
// The admission, rejection, warning, and digest demonstrations intentionally form one linear
// walkthrough; keep their checks next to the API calls they explain.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(int argc, char** argv) {
  const bool print_plan_json = argc == 2 && std::string_view(argv[1]) == "--print-plan-json";
  if (argc != 1 && !print_plan_json) {
    std::cerr << "usage: compile_and_check [--print-plan-json]\n";
    return 1;
  }
  // --- 1. Admitted graph: compile, gate on ok(), persist plan.json(). -------
  holoscan::Graph graph{"compile-and-check"};
  build_pipeline(graph, 10ms, 4U, DefaultClock::kSet);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  holoscan::log_diagnostics(plan);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // Allow cross-process byte comparisons without leaving an artifact in a shared directory.
  if (print_plan_json) {
    std::cout << plan.json();
    return std::cout ? 0 : 2;
  }
  std::cout << "[admitted] plan ok: yes (diagnostics: " << plan.diagnostics().size() << ")\n";

  const std::filesystem::path workspace = holoscan_examples::create_temporary_workspace();
  if (workspace.empty()) {
    std::cerr << "failed to create a private temporary workspace\n";
    return 2;
  }
  const holoscan_examples::TemporaryWorkspaceCleanup workspace_cleanup{workspace};
  const std::filesystem::path json_path = workspace / "compile_and_check_plan.json";
  {
    std::ofstream json_file(json_path, std::ios::binary | std::ios::trunc);
    json_file << plan.json();
    json_file.close();
    if (!json_file) {
      std::cerr << "failed to write " << json_path << '\n';
      return 2;
    }
  }
  std::cout << "[admitted] plan.json written: " << json_path.filename().string() << " ("
            << plan.json().size() << " bytes)\n";

  // --- 2. Broken variant: rejection is data, not an exception. --------------
  // A clock is declared, but set_default_clock() is never called; the ticker's
  // OnClock (which omitted its clock key) has nothing to resolve against.
  holoscan::Graph broken{"compile-and-check"};
  build_pipeline(broken, 10ms, 4U, DefaultClock::kDeclaredOnly);
  const holoscan::ExecutionPlan rejected = holoscan::compile(broken);
  holoscan::log_diagnostics(rejected);
  std::cout << "[broken] plan ok: " << (rejected.ok() ? "yes" : "no") << '\n';
  print_diagnostics("broken", rejected);
  const bool clock_error =
      has_diagnostic(rejected, "TEMPORAL_CLOCK_UNDECLARED", holoscan::DiagnosticSeverity::kError);
  std::cout << "[broken] clock diagnostic found: " << (clock_error ? "yes" : "no") << '\n';
  if (rejected.ok() || !clock_error) {
    std::cerr << rejected.json() << '\n';
    return 3;
  }

  // The throwing bridge for cold setup paths that cannot proceed on a rejected
  // plan. Everywhere else this example branches on values; here we prove the
  // guard actually throws holoscan::CompileError.
  bool threw = false;
  try {
    rejected.require_no_errors();
  } catch (const holoscan::CompileError&) {
    threw = true;
  }
  std::cout << "[broken] require_no_errors threw CompileError: " << (threw ? "yes" : "no") << '\n';
  if (!threw) {
    return 4;
  }

  // --- 3. ok-with-warnings: admission with retained evidence. ---------------
  holoscan::Graph gated{"compile-and-check-gated"};
  build_gated_pipeline(gated);
  const holoscan::ExecutionPlan warned = holoscan::compile(gated);
  holoscan::log_diagnostics(warned);
  std::cout << "[warned] plan ok: " << (warned.ok() ? "yes" : "no") << '\n';
  print_diagnostics("warned", warned);
  const bool warning_found = has_diagnostic(
      warned, "PREDICATE_CALLABLE_TYPE_ERASED", holoscan::DiagnosticSeverity::kWarning);
  std::cout << "[warned] admitted with warnings: " << (warned.ok() && warning_found ? "yes" : "no")
            << '\n';
  if (!warned.ok() || !warning_found) {
    std::cerr << warned.json() << '\n';
    return 5;
  }

  // --- 4. Digest gate: identical structure => identical identity. -----------
  // A second, independently authored build of the same pipeline. Plan identity
  // is structure-only, so digest and JSON must match byte for byte.
  holoscan::Graph second{"compile-and-check"};
  build_pipeline(second, 10ms, 4U, DefaultClock::kSet);
  const holoscan::ExecutionPlan second_plan = holoscan::compile(second);
  holoscan::log_diagnostics(second_plan);
  if (!second_plan.ok()) {
    std::cerr << second_plan.json() << '\n';
    return 6;
  }
  std::cout << "[digest] first  compile: " << digest_prefix(plan.digest()) << '\n';
  std::cout << "[digest] second compile: " << digest_prefix(second_plan.digest()) << '\n';
  const bool json_equal = plan.json() == second_plan.json();
  std::cout << "[digest] plan.json byte-equal: " << (json_equal ? "yes" : "no") << '\n';
  const bool stable = plan.digest() == second_plan.digest() && json_equal;
  std::cout << "digest stable: " << (stable ? "yes" : "no") << '\n';
  if (!stable) {
    return 6;
  }

  // A small tuning edit — faster tick, deeper sink queue — must move the digest;
  // this is the comparison a CI job makes against its checked-in baseline.
  holoscan::Graph edited{"compile-and-check"};
  build_pipeline(edited, 5ms, 8U, DefaultClock::kSet);
  const holoscan::ExecutionPlan edited_plan = holoscan::compile(edited);
  holoscan::log_diagnostics(edited_plan);
  if (!edited_plan.ok()) {
    std::cerr << edited_plan.json() << '\n';
    return 7;
  }
  std::cout << "[digest] edited compile: " << digest_prefix(edited_plan.digest()) << '\n';
  const bool changed = !(edited_plan.digest() == plan.digest());
  std::cout << "digest changed after edit: " << (changed ? "yes" : "no") << '\n';
  if (!changed) {
    return 7;
  }

  // --- 5. Optional clean run of the admitted plan. --------------------------
  TickObservation::reset();
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = TickObservation::wait_for_at_least(3U);
  session.request_stop();
  session.wait();
  const bool run_clean = observed && session.termination() == holoscan::RunTermination::kStopped &&
                         session.diagnostics().empty();
  std::cout << "[run] termination=stopped diagnostics=" << session.diagnostics().size()
            << " ticks>=3: " << (observed ? "yes" : "no") << '\n';
  if (!run_clean) {
    return 8;
  }

  std::cout << "compile_and_check example complete: admission, diagnostics, and digest verified\n";
  return 0;
}
