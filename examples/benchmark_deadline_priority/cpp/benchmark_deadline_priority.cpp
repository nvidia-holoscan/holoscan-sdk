/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Demonstrates the impact of the EventBasedScheduler's dispatcher thread
// on Holoscan SDK operators using `SCHED_DEADLINE` scheduling.
//
// Per core: one high-priority operator (500 us busy-spin) and one or more
// background operators (configurable busy-spin).  The hi operator runs for a
// fixed number of iterations (CountCondition).  Background operators run until
// the hi operator on the same core finishes, monitored via a shared atomic
// flag and a BooleanCondition kill-switch.
//
// Hi op at 50% utilization, bg ops share 20% utilization. We cap it at 70%
// utilization because the well-known rate-monotonic scheduling also maxes out
// at ~70%, and it leaves headroom for scheduling overheads and background tasks.
// A reference (hi-only) run captures reference timings; drift is computed
// relative to that reference. Drift is one-sided: only positive samples
// (current run lagging the reference) are kept. Early ticks (d <= 0) are
// discarded, and drift_late_count / drift_total_samples reports how many
// of the compared ticks were late. Mean/p99/max drift are therefore
// conditional on being late, not averages over all ticks.
//
// With -p CORE, a pass pins the GXF dispatcher thread to the given core
// under SCHED_FIFO priority 99 (via GXF_EBS_DISPATCHER_* env vars) so the two
// results can be compared side by side.
//
// Options:
//   -n N   Background ops per core (default 1).
//   -w US  Background work duration in us (default 1000).
//   -p C   Pin dispatcher to core C with SCHED_FIFO 99 (extra pass).
//
// Requires root / CAP_SYS_ADMIN for SCHED_DEADLINE.
//
// Examples:
//   sudo taskset -c 0-3 ./benchmark_deadline_priority
//   sudo taskset -c 0-3 ./benchmark_deadline_priority -n 4
//   sudo taskset -c 0-5 ./benchmark_deadline_priority -n 2 -p 5

#include <getopt.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "fmt/format.h"
#include "holoscan/holoscan.hpp"

namespace {

void busy_spin_us(int64_t duration_us) {
  auto start = std::chrono::steady_clock::now();
  auto target = std::chrono::microseconds(duration_us);
  volatile int64_t sink = 0;
  while (std::chrono::steady_clock::now() - start < target) {
    for (int i = 0; i < 100; ++i) {
      sink += i;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Operator
// ---------------------------------------------------------------------------

class BusySourceOp : public holoscan::Operator {
 public:
  static constexpr int64_t kWarmup = 100;
  static constexpr int64_t kCooldown = 100;

  HOLOSCAN_OPERATOR_FORWARD_ARGS(BusySourceOp)
  BusySourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.param(work_duration_us_,
               "work_duration_us",
               "Work Duration (us)",
               "Microseconds of CPU busy-work per invocation",
               int64_t(50));
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto begin = std::chrono::steady_clock::now();
    if (count_ == kWarmup) {
      first_begin_ = begin;
    } else if (count_ > kWarmup) {
      int64_t gap_us =
          std::chrono::duration_cast<std::chrono::microseconds>(begin - last_end_).count();
      dispatch_gaps_us_.push_back(gap_us);
      int64_t interval_us =
          std::chrono::duration_cast<std::chrono::microseconds>(begin - last_begin_).count();
      invocation_intervals_us_.push_back(interval_us);
      int64_t elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(begin - first_begin_).count();
      begin_offsets_us_.push_back(elapsed_us);
    }
    last_begin_ = begin;
    busy_spin_us(work_duration_us_.get());
    last_end_ = std::chrono::steady_clock::now();
    count_++;

    if (running_flag_ && count_ >= target_count_) {
      running_flag_->store(false, std::memory_order_release);
    }

    if (peer_running_ && !peer_running_->load(std::memory_order_acquire)) {
      self_condition_->disable_tick();
    }
  }

  void stop() override {
    HOLOSCAN_LOG_INFO("Stopping operator '{}'", name());
    if (running_flag_) {
      running_flag_->store(false, std::memory_order_release);
    }
    for (auto* v : {&dispatch_gaps_us_, &invocation_intervals_us_, &begin_offsets_us_}) {
      if (static_cast<int64_t>(v->size()) > kCooldown) {
        v->erase(v->end() - kCooldown, v->end());
      } else {
        v->clear();
      }
    }
  }

  void set_running_flag(std::shared_ptr<std::atomic<bool>> flag, int64_t target_count) {
    running_flag_ = std::move(flag);
    target_count_ = target_count;
  }

  void monitor_peer(std::shared_ptr<std::atomic<bool>> peer_running,
                    std::shared_ptr<holoscan::BooleanCondition> self_condition) {
    peer_running_ = std::move(peer_running);
    self_condition_ = std::move(self_condition);
  }

  int64_t count() const { return count_; }
  const std::vector<int64_t>& dispatch_gaps_us() const { return dispatch_gaps_us_; }
  const std::vector<int64_t>& invocation_intervals_us() const { return invocation_intervals_us_; }
  const std::vector<int64_t>& begin_offsets_us() const { return begin_offsets_us_; }

 private:
  holoscan::Parameter<int64_t> work_duration_us_;
  int64_t count_ = 0;
  std::chrono::steady_clock::time_point first_begin_;
  std::chrono::steady_clock::time_point last_begin_;
  std::chrono::steady_clock::time_point last_end_;
  std::vector<int64_t> dispatch_gaps_us_;
  std::vector<int64_t> invocation_intervals_us_;
  std::vector<int64_t> begin_offsets_us_;

  std::shared_ptr<std::atomic<bool>> running_flag_;
  int64_t target_count_ = 0;
  std::shared_ptr<std::atomic<bool>> peer_running_;
  std::shared_ptr<holoscan::BooleanCondition> self_condition_;
};

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class DispatcherPriorityApp : public holoscan::Application {
 public:
  static constexpr int64_t kDlRuntimeHeadroomUs = 100;
  // Hi op gets a 50% CPU reservation; the aggregate bg budget per core is 20%
  // (split evenly across bg_per_core operators). Keep hi + bg utilization at
  // ~70% to leave headroom for dispatcher overhead and scheduling jitter,
  // matching the rate-monotonic schedulability bound.
  static constexpr double kHiUtil = 0.50;
  static constexpr double kBgTotalUtilPerCore = 0.20;

  struct DlParams {
    uint64_t hi_runtime_ns = 0;
    uint64_t hi_period_ns = 0;
    uint64_t bg_runtime_ns = 0;
    uint64_t bg_period_ns = 0;
    double bg_util_each = 0.0;
  };

  // Single source of truth for the SCHED_DEADLINE runtime/period derivations
  // used both when configuring the kernel (compose()) and when reporting the
  // values back to the user (run_deadline_benchmark()).
  static DlParams compute_dl_params(int64_t hi_work_us, int64_t bg_work_us, int bg_per_core) {
    DlParams p;
    p.hi_runtime_ns = static_cast<uint64_t>(hi_work_us + kDlRuntimeHeadroomUs) * 1000;
    p.hi_period_ns = static_cast<uint64_t>(p.hi_runtime_ns / kHiUtil);
    if (bg_per_core > 0) {
      p.bg_runtime_ns = static_cast<uint64_t>(bg_work_us + kDlRuntimeHeadroomUs) * 1000;
      p.bg_util_each = kBgTotalUtilPerCore / bg_per_core;
      p.bg_period_ns = static_cast<uint64_t>(p.bg_runtime_ns / p.bg_util_each);
    }
    return p;
  }

  enum class Mode { kNonRt, kRtDeadline };

  struct Options {
    Mode mode = Mode::kNonRt;
    int num_iterations = 5000;
    int64_t hi_work_us = 500;
    int64_t lo_work_us = 1000;
    int lo_per_core = 1;
    std::vector<uint32_t> cores;
  };

  void set_options(const Options& options) { options_ = options; }

  void compose() override {
    using namespace holoscan;

    if (options_.cores.empty()) {
      throw std::runtime_error("No cores provided to benchmark");
    }

    // All modes use per-core pools.
    std::vector<std::shared_ptr<ThreadPool>> pools;
    pools.reserve(options_.cores.size());
    for (size_t i = 0; i < options_.cores.size(); ++i) {
      pools.push_back(make_thread_pool(fmt::format("pool_core_{}", options_.cores[i]), 0));
    }

    if (options_.mode == Mode::kRtDeadline) {
      const DlParams dl =
          compute_dl_params(options_.hi_work_us, options_.lo_work_us, options_.lo_per_core);

      for (size_t c = 0; c < options_.cores.size(); ++c) {
        auto hi_running = std::make_shared<std::atomic<bool>>(true);
        std::vector<uint32_t> pin = {options_.cores[c]};
        auto& pool = pools[c];

        auto hi_op =
            make_operator<BusySourceOp>(fmt::format("hi_{}", c),
                                        make_condition<CountCondition>(options_.num_iterations),
                                        Arg("work_duration_us", options_.hi_work_us));
        hi_op->set_running_flag(hi_running, options_.num_iterations);
        add_operator(hi_op);
        pool->add_realtime(hi_op,
                           SchedulingPolicy::kDeadline,
                           true,
                           pin,
                           0,
                           dl.hi_runtime_ns,
                           dl.hi_period_ns,
                           dl.hi_period_ns);
        hi_ops_.push_back(hi_op);

        for (int l = 0; l < options_.lo_per_core; ++l) {
          auto bool_cond = make_condition<BooleanCondition>(fmt::format("stop_bg_{}_{}", c, l),
                                                            Arg("enable_tick", true));
          auto bg_op = make_operator<BusySourceOp>(fmt::format("bg_{}_{}", c, l),
                                                   bool_cond,
                                                   Arg("work_duration_us", options_.lo_work_us));
          bg_op->monitor_peer(hi_running, std::move(bool_cond));
          add_operator(bg_op);
          pool->add_realtime(bg_op,
                             SchedulingPolicy::kDeadline,
                             true,
                             pin,
                             0,
                             dl.bg_runtime_ns,
                             dl.bg_period_ns,
                             dl.bg_period_ns);
          lo_ops_.push_back(bg_op);
        }
      }
    } else {
      // kNonRt (SCHED_OTHER): per-core hi + bg, all pinned.
      for (size_t c = 0; c < options_.cores.size(); ++c) {
        auto hi_running = std::make_shared<std::atomic<bool>>(true);
        std::vector<uint32_t> pin = {options_.cores[c]};
        auto& pool = pools[c];

        auto hi_op =
            make_operator<BusySourceOp>(fmt::format("hi_{}", c),
                                        make_condition<CountCondition>(options_.num_iterations),
                                        Arg("work_duration_us", options_.hi_work_us));
        hi_op->set_running_flag(hi_running, options_.num_iterations);
        add_operator(hi_op);
        pool->add(hi_op, true, pin);
        hi_ops_.push_back(hi_op);

        for (int l = 0; l < options_.lo_per_core; ++l) {
          auto bool_cond = make_condition<BooleanCondition>(fmt::format("stop_lo_{}_{}", c, l),
                                                            Arg("enable_tick", true));
          auto lo_op = make_operator<BusySourceOp>(fmt::format("lo_{}_{}", c, l),
                                                   bool_cond,
                                                   Arg("work_duration_us", options_.lo_work_us));
          lo_op->monitor_peer(hi_running, std::move(bool_cond));
          add_operator(lo_op);
          pool->add(lo_op, true, pin);
          lo_ops_.push_back(lo_op);
        }
      }
    }

    scheduler(
        make_scheduler<EventBasedScheduler>("ebs", Arg("stop_on_deadlock_timeout", int64_t(5))));
  }

  const std::vector<std::shared_ptr<BusySourceOp>>& hi_ops() const { return hi_ops_; }
  const std::vector<std::shared_ptr<BusySourceOp>>& lo_ops() const { return lo_ops_; }

 private:
  Options options_;
  std::vector<std::shared_ptr<BusySourceOp>> hi_ops_;
  std::vector<std::shared_ptr<BusySourceOp>> lo_ops_;
};

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct GapStats {
  double min_us = 0;
  double mean_us = 0;
  double p99_us = 0;
  double max_us = 0;
  size_t count = 0;
};

GapStats compute_gap_stats(std::vector<int64_t>& samples) {
  GapStats s;
  s.count = samples.size();
  if (samples.empty())
    return s;

  std::sort(samples.begin(), samples.end());
  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  s.mean_us = sum / samples.size();
  s.min_us = samples.front();
  s.max_us = samples.back();
  s.p99_us = samples[std::min(samples.size() - 1, static_cast<size_t>(samples.size() * 0.99))];
  return s;
}

// ---------------------------------------------------------------------------
// Stats collection
// ---------------------------------------------------------------------------

struct RunStats {
  double wall_s = 0;
  int64_t hi_ticks = 0;
  int64_t bg_ticks = 0;
  double throughput_hz = 0;
  GapStats dispatch_gap;
  GapStats invocation_gap;
  double overhead_mean = 0, overhead_p99 = 0, overhead_max = 0;
  double overrun_pct = 0, missed_pct = 0;
  GapStats drift;
  size_t drift_late_count = 0, drift_total_samples = 0;
  bool has_bg = false;
  bool has_drift = false;
};

RunStats collect_stats(DispatcherPriorityApp* app,
                       const std::vector<std::vector<int64_t>>& ref_offsets, double hi_period_us,
                       double wall_s) {
  RunStats rs;
  rs.wall_s = wall_s;

  std::vector<int64_t> all_intervals, all_dispatch_gaps, all_drifts;

  const auto& hi_ops = app->hi_ops();
  for (size_t i = 0; i < hi_ops.size(); ++i) {
    const auto& op = hi_ops[i];
    rs.hi_ticks += op->count();
    const auto& iv = op->invocation_intervals_us();
    all_intervals.insert(all_intervals.end(), iv.begin(), iv.end());
    const auto& dg = op->dispatch_gaps_us();
    all_dispatch_gaps.insert(all_dispatch_gaps.end(), dg.begin(), dg.end());

    const auto& offsets = op->begin_offsets_us();
    const auto& base = (i < ref_offsets.size()) ? ref_offsets[i] : std::vector<int64_t>{};
    size_t common = std::min(offsets.size(), base.size());
    rs.drift_total_samples += common;
    // One-sided drift: keep only samples where this run lagged the reference
    // (d > 0). Early ticks (d <= 0) are intentionally dropped, so the reported
    // mean/p99/max drift are conditional on being late. The fraction of late
    // ticks is surfaced separately as drift_late_count / drift_total_samples.
    for (size_t n = 0; n < common; ++n) {
      int64_t d = offsets[n] - base[n];
      if (d > 0)
        all_drifts.push_back(d);
    }
  }
  for (const auto& op : app->lo_ops()) {
    rs.bg_ticks += op->count();
  }

  rs.has_bg = rs.bg_ticks > 0;
  rs.has_drift = !ref_offsets.empty() && rs.has_bg;
  rs.drift_late_count = all_drifts.size();

  rs.dispatch_gap = compute_gap_stats(all_dispatch_gaps);
  rs.invocation_gap = compute_gap_stats(all_intervals);
  rs.drift = compute_gap_stats(all_drifts);

  rs.overhead_mean = rs.invocation_gap.mean_us - hi_period_us;
  rs.overhead_p99 = rs.invocation_gap.p99_us - hi_period_us;
  rs.overhead_max = rs.invocation_gap.max_us - hi_period_us;

  int64_t overrun_threshold = static_cast<int64_t>(hi_period_us) + 100;
  int64_t threshold_1_5x = static_cast<int64_t>(hi_period_us * 1.5);
  size_t overrun_count = 0, missed_count = 0;
  for (auto v : all_intervals) {
    if (v > overrun_threshold)
      ++overrun_count;
    if (v > threshold_1_5x)
      ++missed_count;
  }
  rs.overrun_pct = all_intervals.empty()
                       ? 0.0
                       : static_cast<double>(overrun_count) / all_intervals.size() * 100.0;
  rs.missed_pct = all_intervals.empty()
                      ? 0.0
                      : static_cast<double>(missed_count) / all_intervals.size() * 100.0;
  rs.throughput_hz = (wall_s > 0) ? (rs.hi_ticks + rs.bg_ticks) / wall_s : 0;

  return rs;
}

void print_comparison_table(const std::vector<std::string>& col_names,
                            const std::vector<RunStats>& cols, double hi_period_us) {
  const int lw = 32;
  const int cw = 16;

  auto sep = [&]() { std::cout << std::string(lw + (cw + 1) * cols.size(), '-') << "\n"; };

  auto row_f = [&](const std::string& label, auto fn) {
    std::cout << fmt::format("{:<{}}", label, lw);
    for (size_t i = 0; i < cols.size(); ++i) {
      std::string val = fn(cols[i]);
      std::cout << fmt::format("| {:>{}}", val, cw - 2) << "  ";
    }
    std::cout << "\n";
  };

  auto fval = [](double v) { return fmt::format("{:.1f}", v); };
  auto fsgn = [](double v) { return fmt::format("{:+.1f}", v); };
  auto fpct = [](double v) { return fmt::format("{:.2f}%%", v); };
  auto fint = [](int64_t v) { return fmt::format("{}", v); };
  auto dash = []() -> std::string { return "-"; };

  std::cout << "\n" << std::string(lw + (cw + 1) * cols.size(), '=') << "\n";
  std::cout << fmt::format("RESULTS  (hi target period: {:.0f} us)\n", hi_period_us);
  std::cout << std::string(lw + (cw + 1) * cols.size(), '=') << "\n";

  std::cout << fmt::format("{:<{}}", "Metric", lw);
  for (const auto& n : col_names)
    std::cout << fmt::format("| {:>{}}", n, cw - 2) << "  ";
  std::cout << "\n";
  sep();

  row_f("Wall time (s)",
        [&](const RunStats& s) -> std::string { return fmt::format("{:.3f}", s.wall_s); });
  row_f("Hi ticks", [&](const RunStats& s) -> std::string { return fint(s.hi_ticks); });
  row_f("Bg ticks",
        [&](const RunStats& s) -> std::string { return s.has_bg ? fint(s.bg_ticks) : dash(); });
  row_f("Throughput (ops/s)",
        [&](const RunStats& s) -> std::string { return fval(s.throughput_hz); });
  sep();
  row_f("Dispatch gap mean (us)",
        [&](const RunStats& s) -> std::string { return fval(s.dispatch_gap.mean_us); });
  row_f("Dispatch gap p99 (us)",
        [&](const RunStats& s) -> std::string { return fval(s.dispatch_gap.p99_us); });
  sep();
  row_f("Invocation gap mean (us)",
        [&](const RunStats& s) -> std::string { return fval(s.invocation_gap.mean_us); });
  row_f("Invocation gap p99 (us)",
        [&](const RunStats& s) -> std::string { return fval(s.invocation_gap.p99_us); });
  sep();
  row_f("Overhead mean (us)",
        [&](const RunStats& s) -> std::string { return fsgn(s.overhead_mean); });
  row_f("Overhead p99 (us)",
        [&](const RunStats& s) -> std::string { return fsgn(s.overhead_p99); });
  row_f("Overhead max (us)",
        [&](const RunStats& s) -> std::string { return fsgn(s.overhead_max); });
  sep();
  row_f("Overrun (>period+100us) rate",
        [&](const RunStats& s) -> std::string { return fpct(s.overrun_pct); });
  row_f("Missed (>1.5x period) rate",
        [&](const RunStats& s) -> std::string { return fpct(s.missed_pct); });
  sep();
  row_f("Drift late count", [&](const RunStats& s) -> std::string {
    return s.has_drift ? fmt::format("{}", s.drift_late_count) : dash();
  });
  row_f("Drift total samples", [&](const RunStats& s) -> std::string {
    return s.has_drift ? fmt::format("{}", s.drift_total_samples) : dash();
  });
  row_f("Drift mean (us)", [&](const RunStats& s) -> std::string {
    return s.has_drift ? fsgn(s.drift.mean_us) : dash();
  });
  row_f("Drift p99 (us)", [&](const RunStats& s) -> std::string {
    return s.has_drift ? fsgn(s.drift.p99_us) : dash();
  });
  row_f("Drift max (us)", [&](const RunStats& s) -> std::string {
    return s.has_drift ? fsgn(s.drift.max_us) : dash();
  });
  sep();
}

// ---------------------------------------------------------------------------
// SCHED_DEADLINE benchmark
// ---------------------------------------------------------------------------

void run_deadline_benchmark(const std::vector<uint32_t>& cores, int bg_per_core, int64_t hi_work_us,
                            int64_t bg_work_us, int num_iterations, int dispatcher_pin_core) {
  using App = DispatcherPriorityApp;
  int available_cpus = static_cast<int>(cores.size());

  const auto dl = App::compute_dl_params(hi_work_us, bg_work_us, bg_per_core);
  double hi_period_us = static_cast<double>(dl.hi_period_ns) / 1000.0;

  int total_ops = available_cpus * (1 + bg_per_core);

  // --- Config summary ---
  std::cout << "SCHED_DEADLINE/Dispatcher Benchmark\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << fmt::format("  Cores              : {}\n", available_cpus);
  std::cout << fmt::format(
      "  Ops/core           : 1 hi + {} bg = {}\n", bg_per_core, 1 + bg_per_core);
  std::cout << fmt::format("  Total ops          : {}\n", total_ops);
  std::cout << fmt::format("  Hi iterations      : {}\n", num_iterations);
  std::cout << fmt::format("  Hi work            : {} us\n", hi_work_us);
  std::cout << fmt::format("  Hi DL runtime      : {} us\n", dl.hi_runtime_ns / 1000);
  std::cout << fmt::format(
      "  Hi DL period       : {:.0f} us  ({:.0f}%% util)\n", hi_period_us, App::kHiUtil * 100);
  if (bg_per_core > 0) {
    std::cout << fmt::format("  Bg work            : {} us\n", bg_work_us);
    std::cout << fmt::format("  Bg DL runtime      : {} us\n", dl.bg_runtime_ns / 1000);
    std::cout << fmt::format("  Bg DL period       : {} us  ({:.1f}%% util each, {:.0f}%% total)\n",
                             dl.bg_period_ns / 1000,
                             dl.bg_util_each * 100,
                             App::kBgTotalUtilPerCore * 100);
  }
  std::cout << fmt::format(
      "  Total util/core    : {:.0f}%%\n",
      (App::kHiUtil + (bg_per_core > 0 ? App::kBgTotalUtilPerCore : 0.0)) * 100);
  if (dispatcher_pin_core >= 0) {
    std::cout << fmt::format("  Dispatcher pin     : core {} (SCHED_FIFO pri 99)\n",
                             dispatcher_pin_core);
  }
  std::cout << std::string(70, '=') << "\n\n";

  std::vector<std::string> col_names;
  std::vector<RunStats> col_stats;

  // --- Reference run: hi ops only (no bg) ---
  std::cout << "Running reference (hi-only, no bg)...\n" << std::flush;
  auto ref_app = holoscan::make_application<App>();
  ref_app->set_options({App::Mode::kRtDeadline, num_iterations, hi_work_us, bg_work_us, 0, cores});
  auto wall_start = std::chrono::steady_clock::now();
  ref_app->run();
  auto wall_end = std::chrono::steady_clock::now();
  double ref_wall_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(wall_end - wall_start).count();

  std::vector<std::vector<int64_t>> ref_offsets;
  for (const auto& op : ref_app->hi_ops()) {
    ref_offsets.push_back(op->begin_offsets_us());
  }

  std::vector<std::vector<int64_t>> empty_offsets;
  col_names.push_back("Hi-only (ref)");
  col_stats.push_back(collect_stats(ref_app.get(), empty_offsets, hi_period_us, ref_wall_s));

  // --- Default run: hi + bg ops ---
  std::cout << "Running SCHED_DEADLINE with " << bg_per_core << " bg/core...\n" << std::flush;
  auto app = holoscan::make_application<App>();
  app->set_options(
      {App::Mode::kRtDeadline, num_iterations, hi_work_us, bg_work_us, bg_per_core, cores});
  wall_start = std::chrono::steady_clock::now();
  app->run();
  wall_end = std::chrono::steady_clock::now();
  double wall_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(wall_end - wall_start).count();

  col_names.push_back("Default");
  col_stats.push_back(collect_stats(app.get(), ref_offsets, hi_period_us, wall_s));

  // --- Pinned-dispatcher pass ---
  if (dispatcher_pin_core >= 0) {
    std::cout << "Running with pinned dispatcher (core " << dispatcher_pin_core << ")...\n"
              << std::flush;

    setenv("GXF_EBS_DISPATCHER_CPU_CORE", std::to_string(dispatcher_pin_core).c_str(), 1);
    setenv("GXF_EBS_DISPATCHER_SCHED_POLICY", "SCHED_FIFO", 1);
    setenv("GXF_EBS_DISPATCHER_SCHED_PRIORITY", "99", 1);

    auto pinned_app = holoscan::make_application<App>();
    pinned_app->set_options(
        {App::Mode::kRtDeadline, num_iterations, hi_work_us, bg_work_us, bg_per_core, cores});
    wall_start = std::chrono::steady_clock::now();
    pinned_app->run();
    wall_end = std::chrono::steady_clock::now();
    wall_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(wall_end - wall_start).count();

    unsetenv("GXF_EBS_DISPATCHER_CPU_CORE");
    unsetenv("GXF_EBS_DISPATCHER_SCHED_POLICY");
    unsetenv("GXF_EBS_DISPATCHER_SCHED_PRIORITY");

    col_names.push_back("Pinned Dispatcher");
    col_stats.push_back(collect_stats(pinned_app.get(), ref_offsets, hi_period_us, wall_s));
  }

  print_comparison_table(col_names, col_stats, hi_period_us);
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
            << "SCHED_DEADLINE/Dispatcher benchmark (requires root).\n\n"
            << "Options:\n"
            << "  -n N          Background ops per core (default 1)\n"
            << "  -w US         Background work duration in us (default 1000)\n"
            << "  -p CORE       Pin dispatcher thread to CORE with SCHED_FIFO 99\n"
            << "                (runs an extra pass to compare against unpinned)\n"
            << "  -h, --help    Show this help message\n\n"
            << "Examples:\n"
            << "  sudo taskset -c 0-3 " << prog << "\n"
            << "  sudo taskset -c 0-3 " << prog << " -n 4\n"
            << "  sudo taskset -c 0-5 " << prog << " -n 2 -p 5\n";
}

int main(int argc, char** argv) {
  int bg_per_core = 1;
  int64_t lo_work_us = 1000;
  int dispatcher_pin_core = -1;
  int num_iterations = 5000;

  static struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                         {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "hn:w:p:", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'n': {
        int v = std::atoi(optarg);
        if (v < 1)
          v = 1;
        bg_per_core = v;
        break;
      }
      case 'w':
        lo_work_us = std::atol(optarg);
        if (lo_work_us < 0)
          lo_work_us = 0;
        break;
      case 'p':
        dispatcher_pin_core = std::atoi(optarg);
        break;
      case 'h':
      default:
        print_usage(argv[0]);
        return (opt == 'h') ? 0 : 1;
    }
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  sched_getaffinity(0, sizeof(cpuset), &cpuset);
  int available_cpus = CPU_COUNT(&cpuset);

  // This benchmark is designed and validated for up to 64 logical CPUs. The
  // core-count cap below (63 / ops_per_core) reserves one CPU for non-DL
  // work on a 64-CPU machine; with more CPUs visible that invariant no
  // longer holds. Restrict affinity with `taskset -c ...` to fit.
  if (available_cpus > 64) {
    std::cerr << "ERROR: this benchmark supports at most 64 logical CPUs, but " << available_cpus
              << " are visible in the process affinity mask.\n"
              << "Run with e.g. `taskset -c 0-63 " << argv[0] << "` to limit it.\n";
    return 1;
  }

  std::vector<uint32_t> cores;
  for (int i = 0; i < CPU_SETSIZE && static_cast<int>(cores.size()) < available_cpus; ++i) {
    if (CPU_ISSET(i, &cpuset)) {
      cores.push_back(static_cast<uint32_t>(i));
    }
  }

  struct {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
  } attr = {};
  attr.size = sizeof(attr);
  attr.sched_policy = 6;  // SCHED_DEADLINE
  attr.sched_runtime = 1000000;
  attr.sched_deadline = 10000000;
  attr.sched_period = 10000000;
  if (syscall(SYS_sched_setattr, 0, &attr, 0) != 0) {
    std::cerr << "ERROR: SCHED_DEADLINE requires root / CAP_SYS_ADMIN in container and\n"
              << "sudo sysctl -w kernel.sched_rt_runtime_us=-1 on host\n"
              << "Then, run with: sudo " << argv[0] << " [-n BG_PER_CORE]\n";
    return 1;
  }
  attr = {};
  attr.size = sizeof(attr);
  syscall(SYS_sched_setattr, 0, &attr, 0);

  // Cap the total number of SCHED_DEADLINE worker threads at 63 so that at
  // least one logical CPU on the (up to 64-CPU) machine stays free of DL
  // workers for the EBS dispatcher and other SCHED_OTHER work. Each operator
  // gets one pinned DL thread, so workers = cores.size() * (1 + bg_per_core).
  // Integer division rounds down to keep the invariant workers <= 63 while
  // dropping partial per-core groups. Machines larger than 64 CPUs are
  // rejected upstream in main().
  constexpr int kMaxWorkerThreads = 63;
  const int ops_per_core = 1 + bg_per_core;
  if (static_cast<int>(cores.size()) > kMaxWorkerThreads / ops_per_core) {
    cores.resize(kMaxWorkerThreads / ops_per_core);
  }
  const int64_t hi_work_us = 500;
  run_deadline_benchmark(
      cores, bg_per_core, hi_work_us, lo_work_us, num_iterations, dispatcher_pin_core);
  return 0;
}
