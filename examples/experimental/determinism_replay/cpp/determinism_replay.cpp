/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/time_point.hpp>

namespace {

using namespace std::chrono_literals;

// Six ticks, 5ms of logical time apart. Under the realtime run this is ~30ms
// of wall time; under the manual runs no wall-clock period is ever waited on.
constexpr std::chrono::nanoseconds kTickPeriod = 5ms;
constexpr std::uint32_t kTickLimit = 6U;

// The same graph runs three times in this one process: once against real
// time, then twice against a harness-driven ManualClock. The two manual logs
// are diffed field-by-field -- their equality IS the determinism claim.
constexpr std::size_t kRunCount = 3U;
constexpr std::size_t kRealtime = 0U;
constexpr std::size_t kManualA = 1U;
constexpr std::size_t kManualB = 2U;
constexpr std::array<std::string_view, kRunCount> kRunName{"realtime", "manual-a", "manual-b"};

constexpr std::size_t kOpCount = 2U;
constexpr std::size_t kSrc = 0U;
constexpr std::size_t kSink = 1U;

// Host wall-clock observation, deliberately read outside the graph's clock:
// it is the "does NOT replay" column of the log.
[[nodiscard]] std::int64_t wall_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// One row per activation. The first five fields are the replay-comparable
// evidence (Tag identity, TagTime coordinate, capture time in the graph's
// clock domain); wall_ns is host time and is excluded from the replay diff.
struct ActivationRecord {
  std::uint32_t generation{};
  std::uint64_t sequence{};
  std::int64_t logical_ns{};
  std::uint32_t microstep{};
  std::int64_t capture_ns{};
  std::int64_t wall_ns{};

  [[nodiscard]] bool replay_equal(const ActivationRecord& other) const noexcept {
    return generation == other.generation && sequence == other.sequence &&
           logical_ns == other.logical_ns && microstep == other.microstep &&
           capture_ns == other.capture_ns;
  }
};

// Operators run on runtime threads; main() blocks on this observation object
// and drives the manual clock between sink activations. Static because
// graph.op<> captures must be owned copyable values -- operators receive only
// their run index and never borrow pointers into main's stack.
struct ReplayLog {
  static void reset(std::size_t run) {
    const std::lock_guard lock(mutex);
    for (auto& log : logs.at(run)) {
      log.clear();
    }
  }

  static void record(std::size_t run, std::size_t op, ActivationRecord row) {
    {
      const std::lock_guard lock(mutex);
      auto& log = logs.at(run).at(op);
      // The realtime source keeps ticking idly after its emit budget; cap the
      // log so every run yields the same six-row shape per operator.
      if (log.size() < kTickLimit) {
        log.push_back(row);
      }
    }
    if (op == kSink) {
      ready.notify_all();
    }
  }

  [[nodiscard]] static bool wait_for_sink(std::size_t run, std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [run, expected] { return logs.at(run).at(kSink).size() >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<std::array<std::vector<ActivationRecord>, kOpCount>, kRunCount> logs;
};

// A clock-driven counter source. Every activation freezes three kinds of
// "when": ctx.tag() (identity), ctx.tag_time() (logical order -- the
// replay-stable coordinate), and ctx.activation_time() (physical time in the
// graph's clock domain). For a synthesized value the activation instant *is*
// the acquisition instant, so it is the honest capture_time stamp here; a
// real sensor driver would label its hardware timestamp with at_ns() instead.
class TickSource final : public holoscan::Operator<> {
 public:
  TickSource(std::size_t run, std::uint32_t limit) noexcept : run_(run), limit_(limit) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out");  // max_emits_per_compute defaults to 1
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& ctx) override {
    const holoscan::Tag id = ctx.tag();
    const holoscan::TagTime order = ctx.tag_time();
    const holoscan::TimePoint capture = ctx.activation_time();
    ReplayLog::record(run_,
                      kSrc,
                      ActivationRecord{.generation = id.generation,
                                       .sequence = id.sequence,
                                       .logical_ns = order.logical_ns,
                                       .microstep = order.microstep,
                                       .capture_ns = capture.timestamp_ns,
                                       .wall_ns = wall_now_ns()});
    if (next_ >= limit_) {
      return {};  // budget spent: keep ticking idly; main() stops the run
    }
    return out.emit(++next_, holoscan::EmitOptions{.capture_time = capture});
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::size_t run_;
  std::uint32_t limit_;
  std::uint32_t next_{};
};

// The sink receives the full Sample<T> -- not just the payload -- so it can
// log the capture_time that traveled with the data next to its own activation
// coordinates. Under the manual runs both columns reproduce exactly.
class ReplaySink final : public holoscan::Operator<> {
 public:
  explicit ReplaySink(std::size_t run) noexcept : run_(run) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& ctx) override {
    HOLOSCAN_TRY(auto sample, in.receive());
    const holoscan::Tag id = ctx.tag();
    const holoscan::TagTime order = ctx.tag_time();
    ReplayLog::record(run_,
                      kSink,
                      ActivationRecord{.generation = id.generation,
                                       .sequence = id.sequence,
                                       .logical_ns = order.logical_ns,
                                       .microstep = order.microstep,
                                       .capture_ns = sample.metadata.capture_time().timestamp_ns,
                                       .wall_ns = wall_now_ns()});
    return {};
  }

  holoscan::Input<std::uint32_t> in;

 private:
  std::size_t run_;
};

// The graph is byte-for-byte identical across all three runs; only the clock
// declaration differs. bind_shared_clock() is the one clock-authoring verb
// that retains the live instance you constructed (add_clock() validates a
// throwaway and later replays a factory), so main() keeps a handle it may
// advance while the graph runs.
[[nodiscard]] bool run_once(std::size_t run) {
  ReplayLog::reset(run);

  holoscan::Graph graph{"determinism-replay"};
  const auto source = graph.op<TickSource>("source", run, kTickLimit);
  const auto sink = graph.op<ReplaySink>("sink", run);
  graph.add_flow(source->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});

  std::shared_ptr<holoscan::ManualClock> manual;
  if (run == kRealtime) {
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  } else {
    manual = std::make_shared<holoscan::ManualClock>(0);
    graph.set_default_clock(graph.bind_shared_clock("replay", manual));
  }

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return false;
  }

  holoscan::RunSession session = holoscan::run_async(plan);

  bool complete = false;
  if (manual == nullptr) {
    // Realtime: OnClock free-runs against the wall; just observe.
    complete = ReplayLog::wait_for_sink(run, kTickLimit);
  } else {
    // Manual: the runtime never advances this clock on its own -- it polls
    // now() and fires the tick only once the coordinate is reached. The
    // OnClock phase is captured at run start, where the clock already stands
    // (t=0), so tick 0 fires without any advance. After that, strict
    // lockstep: advance exactly one period, then wait for exactly one more
    // sink activation before touching the clock again. Every clock read any
    // operator performs lands on the scripted 0, 5ms, 10ms ... grid.
    complete = ReplayLog::wait_for_sink(run, 1U);
    for (std::uint32_t step = 1U; complete && step < kTickLimit; ++step) {
      manual->advance_by(kTickPeriod);
      complete = ReplayLog::wait_for_sink(run, step + 1U);
    }
  }

  session.request_stop();
  session.wait();

  if (!complete) {
    std::cerr << "determinism_replay[" << kRunName.at(run) << "]: timed out waiting for "
              << kTickLimit << " sink activations\n";
    return false;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "determinism_replay[" << kRunName.at(run) << "]: unexpected termination state\n";
    return false;
  }

  // main() is the sole printer: the logs are the artifact, so the console
  // view is rendered after the run instead of racing the runtime threads.
  const auto& sink_log = ReplayLog::logs.at(run).at(kSink);
  for (std::size_t i = 0; i < 2U && i < sink_log.size(); ++i) {
    const ActivationRecord& row = sink_log[i];
    std::cout << "[" << kRunName.at(run) << "] sink tag=gen" << row.generation << "/seq"
              << row.sequence << " tag_time=" << row.logical_ns << "ns/us" << row.microstep
              << " capture=" << row.capture_ns << "ns\n";
  }
  return true;
}

[[nodiscard]] bool replay_columns_equal(std::size_t run_a, std::size_t run_b) {
  const std::lock_guard lock(ReplayLog::mutex);
  for (std::size_t op = 0; op < kOpCount; ++op) {
    const auto& a = ReplayLog::logs.at(run_a).at(op);
    const auto& b = ReplayLog::logs.at(run_b).at(op);
    if (a.size() != b.size()) {
      return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (!a[i].replay_equal(b[i])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  for (std::size_t run = 0; run < kRunCount; ++run) {
    if (!run_once(run)) {
      return 2;
    }
  }

  // The determinism claim: two manual runs of the same graph over the same
  // scripted time produce field-identical Tag / TagTime / capture logs.
  const bool manual_match = replay_columns_equal(kManualA, kManualB);
  // The contrast claim: the realtime run's coordinates sit on a live-clock
  // grid, not the scripted one, so its replay columns do not match.
  const bool realtime_differs = !replay_columns_equal(kRealtime, kManualA);
  // The non-claim: wall-clock host time never replays -- run B executed
  // strictly after run A even though their logical logs are identical.
  const bool wall_advanced = ReplayLog::logs.at(kManualB).at(kSink).front().wall_ns >
                             ReplayLog::logs.at(kManualA).at(kSink).back().wall_ns;

  if (!manual_match || !realtime_differs || !wall_advanced) {
    std::cerr << "determinism_replay: diff failed (manual_match=" << manual_match
              << " realtime_differs=" << realtime_differs << " wall_advanced=" << wall_advanced
              << ")\n";
    return 3;
  }

  std::cout << "determinism_replay: manual_replay_match=yes realtime_tagtime_differs=yes"
               " wall_clock_replayed=no ticks="
            << kTickLimit << " termination=stopped\n";
  return 0;
}
