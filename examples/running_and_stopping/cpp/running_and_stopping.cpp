/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Run and stop a Holoscan graph three ways.
//
// Sub-run A: the cooperative idiom — run_async(), observe from main(), then
//            request_stop()/wait() and verify RunTermination::kStopped.
// Sub-run B: the failure channel — a compute() that returns an error fails
//            the run closed; termination() is kFailed and diagnostics carry
//            a stable token.
// Sub-run C: RunSession is move-only RAII. The moved-to owner performs an
//            explicit stop/join; the destructor remains its live-owner fail-safe.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
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

int main() {
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
    std::cout << "B: compute error -> termination=kFailed, first token="
              << session.diagnostics().front().token << '\n';
  }

  // --- Sub-run C: move-only RAII ---------------------------------------------
  StopObservation::reset();
  {
    holoscan::RunSession outer = holoscan::run_async(plan_a);
    // RunSession is move-only: ownership transfers; the moved-from object is
    // inert and the moved-to object owns stop/join.
    holoscan::RunSession owner = std::move(outer);
    if (!StopObservation::wait_for(3U)) {
      std::cerr << "sub-run C observation timeout\n";
      return 1;
    }
    owner.request_stop();
    owner.wait();
    if (owner.termination() != holoscan::RunTermination::kStopped) {
      std::cerr << "sub-run C moved session did not stop\n";
      return 1;
    }
    std::cout << "C: moved session stopped; scoped owner retained stop/join authority\n";
  }

  // Note: run(plan) returns immediately with kCompleted for an accepted
  // zero-work graph. Recurring nonempty sources do not have a bounded-source
  // completion protocol, so interactive programs use run_async() and stop
  // cooperatively. See README for the distinction.
  std::cout << "running_and_stopping: all sub-runs behaved as documented\n";
  return 0;
}
