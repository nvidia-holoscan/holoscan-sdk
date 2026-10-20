/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Run and stop a Holoscan graph seven ways.
//
// Sub-run M: create an early Linux managed-signal domain, atomically bind it
//            to run_async(), then wait and close on the creator thread.
// Sub-run A: the cooperative idiom — run_async(), observe from main(), then
//            request_stop()/wait() and verify RunTermination::kStopped.
// Sub-run B: the failure channel — a compute() that returns an error fails
//            the run closed; termination() is kFailed and diagnostics carry
//            a stable token.
// Sub-run C: an operator requests immediate whole-run success; termination()
//            is kRequestedCompleted without an EOS or drain guarantee.
// Sub-run D: RunSession is move-only RAII. The moved-to owner performs an
//            explicit stop/join; the destructor remains its live-owner fail-safe.
// Sub-run E: the caller-managed blocking overload leaves process signals to
//            the embedding host while preserving blocking result mapping.
// Sub-run F: the default blocking overload adopts an early managed-signal
//            domain in a separate invocation, before another run starts threads.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kTickPeriod = 5ms;

struct StopObservation {
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

  [[nodiscard]] static bool wait_for(std::uint32_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return count >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::uint32_t count{};
};

class TickSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return out.emit(++next_);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t next_{};
};

class CountingSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = in.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    StopObservation::record();
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

class CompletingSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto value = in.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    StopObservation::record();
    if (*value == 5U) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

// Returns an error from compute() after a few samples: the runtime fails the
// run closed. This is the terminal error channel, distinct from request_stop.
class FailingSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = in.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    if (*value >= 3U) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure});  // deliberate: demonstrates fail-closed
    }
    StopObservation::record();
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

template <typename SinkT>
[[nodiscard]] holoscan::ExecutionPlan compile_pipeline(std::string_view name) {
  holoscan::Graph graph{name};
  const auto source = graph.op<TickSource>("source");
  const auto sink = graph.op<SinkT>("sink");
  graph.add_flow(source->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

}  // namespace

// Keep the successive managed and explicit shutdown demonstrations in execution order so the
// signal-domain lifetime and each session result remain visible in this walkthrough.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(int argc, char** argv) {
  const bool default_managed_only = argc == 2 && std::string_view(argv[1]) == "--default-managed";
  if (argc != 1 && !default_managed_only) {
    std::cerr << "usage: running_and_stopping [--default-managed]\n";
    return 1;
  }

  // Establish a managed async domain before compilation or other setup that may create threads.
  auto created_signals = holoscan::ManagedShutdownSignals::create();
#if defined(__linux__)
  if (!created_signals) {
    std::cerr << "managed signal domain setup failed: "
              << holoscan::runtime_error_message(created_signals.error()) << '\n';
    return 1;
  }
  auto managed_signals = std::move(created_signals).value();
#else
  if (created_signals || created_signals.error().code != holoscan::ErrorCode::kNotSupported) {
    std::cerr << "non-Linux managed signal domain did not report kNotSupported\n";
    return 1;
  }
#endif

  if (default_managed_only) {
    const holoscan::ExecutionPlan plan_f =
        compile_pipeline<CompletingSink>("default-managed-completion");
    if (!plan_f.ok()) {
      std::cerr << plan_f.json() << '\n';
      return 1;
    }
    if (holoscan::run(plan_f) != 0) {
      std::cerr << "sub-run F default managed requested completion returned nonzero\n";
      return 1;
    }
#if defined(__linux__)
    auto closed = managed_signals.close();
    if (!closed) {
      std::cerr << "managed signal domain close failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
      return 1;
    }
#endif
    std::cout << "F: default managed blocking run -> result=0\n";
    return 0;
  }

  // --- Sub-run M: early managed domain + async binding ----------------------
  StopObservation::reset();
  const holoscan::ExecutionPlan plan_m = compile_pipeline<CountingSink>("managed-async");
  if (!plan_m.ok()) {
    std::cerr << plan_m.json() << '\n';
    return 1;
  }
#if defined(__linux__)
  {
    holoscan::RunSession session = holoscan::run_async(plan_m, managed_signals);
    const bool observed = StopObservation::wait_for(3U);
    session.request_stop();
    session.wait();  // also waits for managed binding retirement
    if (!observed || session.termination() != holoscan::RunTermination::kStopped) {
      std::cerr << "sub-run M did not stop through the managed async binding\n";
      return 1;
    }
    auto closed = managed_signals.close();
    if (!closed) {
      std::cerr << "managed signal domain close failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
      return 1;
    }
    std::cout << "M: managed async domain -> termination=kStopped, close=ok\n";
  }
#else
  std::cout << "M: managed async domain -> kNotSupported on this platform\n";
#endif

  // --- Sub-run A: cooperative stop -> kStopped -------------------------------
  StopObservation::reset();
  const holoscan::ExecutionPlan plan_a = compile_pipeline<CountingSink>("run-and-stop");
  if (!plan_a.ok()) {
    std::cerr << plan_a.json() << '\n';
    return 1;
  }
  {
    holoscan::RunSession session = holoscan::run_async(plan_a);
    if (!session.is_running()) {
      std::cerr << "session should report running immediately after run_async\n";
      return 1;
    }
    const bool observed = StopObservation::wait_for(5U);
    // request_stop is nonblocking and idempotent; wait() joins the run.
    session.request_stop();
    session.request_stop();  // safe: idempotent
    session.wait();
    if (!observed || session.termination() != holoscan::RunTermination::kStopped ||
        !session.diagnostics().empty()) {
      std::cerr << "sub-run A did not stop cleanly\n";
      return 1;
    }
    std::cout << "A: cooperative stop -> termination=kStopped, diagnostics=0\n";
  }

  // --- Sub-run B: compute error -> kFailed with diagnostics ------------------
  StopObservation::reset();
  const holoscan::ExecutionPlan plan_b = compile_pipeline<FailingSink>("fail-closed");
  if (!plan_b.ok()) {
    std::cerr << plan_b.json() << '\n';
    return 1;
  }
  {
    holoscan::RunSession session = holoscan::run_async(plan_b);
    session.wait();  // the deliberate error terminates the run; wait() returns
    if (session.termination() != holoscan::RunTermination::kFailed ||
        session.diagnostics().empty()) {
      std::cerr << "sub-run B did not fail closed as expected\n";
      return 1;
    }
    // Runtime also attempts one bounded terminal ERROR summary after cleanup.
    // That log is for humans; stable structured diagnostics remain the API.
    std::cout << "B: compute error -> termination=kFailed, diagnostic="
              << holoscan::format_diagnostic(session.diagnostics().front(), plan_b.graph()) << '\n';
  }

  // --- Sub-run C: operator-requested completion -> kRequestedCompleted -------
  StopObservation::reset();
  const holoscan::ExecutionPlan plan_c = compile_pipeline<CompletingSink>("requested-completion");
  if (!plan_c.ok()) {
    std::cerr << plan_c.json() << '\n';
    return 1;
  }
  {
    holoscan::RunSession session = holoscan::run_async(plan_c);
    session.wait();
    if (session.termination() != holoscan::RunTermination::kRequestedCompleted ||
        !session.diagnostics().empty()) {
      std::cerr << "sub-run C did not report requested completion\n";
      return 1;
    }
    std::cout << "C: operator request -> termination=kRequestedCompleted, diagnostics=0\n";
  }

  // --- Sub-run D: move-only RAII ---------------------------------------------
  StopObservation::reset();
  {
    holoscan::RunSession outer = holoscan::run_async(plan_a);
    // RunSession is move-only: ownership transfers; the moved-from object is
    // inert and the moved-to object owns stop/join.
    holoscan::RunSession owner = std::move(outer);
    if (!StopObservation::wait_for(3U)) {
      std::cerr << "sub-run D observation timeout\n";
      return 1;
    }
    owner.request_stop();
    owner.wait();
    if (owner.termination() != holoscan::RunTermination::kStopped) {
      std::cerr << "sub-run D moved session did not stop\n";
      return 1;
    }
    std::cout << "D: moved session stopped; scoped owner retained stop/join authority\n";
  }

  // --- Sub-run E: caller-managed blocking execution -------------------------
  const int caller_managed_result =
      holoscan::run(plan_c, holoscan::RunSignalPolicy::kCallerManaged);
  if (caller_managed_result != 0) {
    std::cerr << "sub-run E caller-managed requested completion returned nonzero\n";
    return 1;
  }
  std::cout << "E: caller-managed blocking run -> result=0\n";

  // Note: run(plan) returns zero for kCompleted or kRequestedCompleted.
  // Requested completion is immediate whole-run success, not an EOS or drain
  // protocol. Recurring nonempty sources do not complete naturally today.
  std::cout << "running_and_stopping: all in-process sub-runs behaved as documented\n";
  return 0;
}
